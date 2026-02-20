"""Extract action labels from colonist.io game JSONs for BC policy training.

Replays game events and at each sampling point (turnState→2), looks ahead
to classify the first Discrete(290)-compatible action the human takes.

Output: parquet file(s) with (game_id, event_index, action_index, num_valid_actions)
that can be joined with existing feature parquet on (game_id, event_index).

Usage:
    python3 -m robottler.extract_actions --workers 6
    python3 -m robottler.extract_actions --workers 6 --limit 100  # quick test
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from catanatron.models.enums import RESOURCES, ActionType
from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY, ACTION_SPACE_SIZE

from bridge.config import COLONIST_RESOURCE_TO_CATAN, COLONIST_VALUE_TO_DEVCARD
from datasets.format_adapter import (
    adapt_tiles,
    adapt_vertices,
    adapt_edges,
    extract_harbor_pairs,
    build_index_maps,
)
from bridge.coordinate_map import build_coordinate_mapper

logging.getLogger("bridge.coordinate_map").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMES_DIR = Path("datasets/games")
DEFAULT_OUTPUT_DIR = Path("datasets/parquet")

RES_ID = COLONIST_RESOURCE_TO_CATAN  # {1:WOOD, 2:BRICK, ...}

# Pre-build reverse lookup: (ActionType, value) -> action_index
_ACTION_LOOKUP = {}
for idx, (atype, aval) in enumerate(ACTIONS_ARRAY):
    _ACTION_LOOKUP[(atype, aval)] = idx


def _action_index(action_type, value):
    """Map (ActionType, value) to Discrete(290) index, or -1 if not found."""
    return _ACTION_LOOKUP.get((action_type, value), -1)


# ---------------------------------------------------------------------------
# Action classification from colonist.io events
# ---------------------------------------------------------------------------

def _classify_event(event, mapper, vertex_idx_to_xyz, edge_idx_to_xyz,
                    prev_player_hands, prev_bank):
    """Classify a single event into a Discrete(290) action index.

    Returns (action_index, action_type_str) or (-1, reason_str) if unmappable.
    """
    sc = event.get("stateChange", {})
    cs = sc.get("currentState", {})
    ms = sc.get("mapState", {})
    dev_sc = sc.get("mechanicDevelopmentCardsState", {})

    # --- End turn ---
    new_ts = cs.get("turnState")
    if new_ts in (1, 0, 3):
        idx = _action_index(ActionType.END_TURN, None)
        return idx, "END_TURN"

    # --- Build settlement ---
    corners = ms.get("tileCornerStates", {})
    if corners:
        for cidx_str, corner in corners.items():
            bt = corner.get("buildingType")
            if bt is None:
                continue
            cidx = int(cidx_str)
            xyz = vertex_idx_to_xyz.get(cidx)
            if xyz is None:
                continue
            node_id = mapper.colonist_vertex_to_catan.get(xyz)
            if node_id is None:
                continue
            if bt == 2:
                idx = _action_index(ActionType.BUILD_CITY, node_id)
                return idx, "BUILD_CITY"
            else:
                idx = _action_index(ActionType.BUILD_SETTLEMENT, node_id)
                return idx, "BUILD_SETTLEMENT"

    # --- Build road ---
    edges = ms.get("tileEdgeStates", {})
    if edges:
        for eidx_str, edge in edges.items():
            owner = edge.get("owner")
            if owner is None or owner < 0:
                continue
            eidx = int(eidx_str)
            xyz = edge_idx_to_xyz.get(eidx)
            if xyz is None:
                continue
            catan_edge = mapper.colonist_edge_to_catan.get(xyz)
            if catan_edge is None:
                continue
            sorted_edge = tuple(sorted(catan_edge))
            idx = _action_index(ActionType.BUILD_ROAD, sorted_edge)
            return idx, "BUILD_ROAD"

    # --- Dev card: buy or play ---
    dev_players = dev_sc.get("players", {})
    if dev_players:
        for pid, pdev in dev_players.items():
            bought = pdev.get("developmentCardsBoughtThisTurn")
            if bought:
                idx = _action_index(ActionType.BUY_DEVELOPMENT_CARD, None)
                return idx, "BUY_DEV_CARD"

            used = pdev.get("developmentCardsUsed")
            if used is not None:
                # Determine which card was played by looking at what's new
                for card_id in used:
                    dev_str = COLONIST_VALUE_TO_DEVCARD.get(card_id)
                    if dev_str == "KNIGHT":
                        idx = _action_index(ActionType.PLAY_KNIGHT_CARD, None)
                        return idx, "PLAY_KNIGHT"
                    elif dev_str == "ROAD_BUILDING":
                        idx = _action_index(ActionType.PLAY_ROAD_BUILDING, None)
                        return idx, "PLAY_ROAD_BUILDING"
                # Monopoly and Year of Plenty are harder to detect from dev state alone
                # They show up with resource changes too — return generic dev play
                return -1, "PLAY_DEV_COMPLEX"

    # --- Maritime trade ---
    bank_rc = sc.get("bankState", {}).get("resourceCards", {})
    ps_sc = sc.get("playerStates", {})
    if bank_rc and ps_sc:
        # Compute bank delta
        new_bank = dict(prev_bank)
        for rid_str, cnt in bank_rc.items():
            new_bank[int(rid_str)] = cnt
        bank_gave = {}  # resources bank gave (count < 0 = bank lost)
        bank_got = {}   # resources bank received (count > 0 = bank gained)
        for rid in set(list(prev_bank.keys()) + list(new_bank.keys())):
            delta = new_bank.get(rid, 0) - prev_bank.get(rid, 0)
            if delta > 0:
                res = RES_ID.get(rid)
                if res:
                    bank_got[res] = delta
            elif delta < 0:
                res = RES_ID.get(rid)
                if res:
                    bank_gave[res] = -delta

        if len(bank_got) == 1 and len(bank_gave) == 1:
            gave_res = list(bank_got.keys())[0]    # player gave this to bank
            got_res = list(bank_gave.keys())[0]     # player got this from bank
            gave_count = list(bank_got.values())[0]

            if gave_count == 4:
                trade_val = tuple(4 * [gave_res] + [got_res])
                idx = _action_index(ActionType.MARITIME_TRADE, trade_val)
                return idx, "MARITIME_4_1"
            elif gave_count == 3:
                trade_val = tuple(3 * [gave_res] + [None, got_res])
                idx = _action_index(ActionType.MARITIME_TRADE, trade_val)
                return idx, "MARITIME_3_1"
            elif gave_count == 2:
                trade_val = tuple(2 * [gave_res] + [None, None, got_res])
                idx = _action_index(ActionType.MARITIME_TRADE, trade_val)
                return idx, "MARITIME_2_1"

        # If we get here with bank changes, it's some other bank interaction
        return -1, "BANK_CHANGE_UNKNOWN"

    # --- Chat, timer, or other non-action event ---
    if set(sc.keys()) - {"gameLogState"} <= {"gameChatState", "timerState"}:
        return -2, "SKIP"  # sentinel: skip and look at next event

    # --- Player state only (resource distribution, not an action) ---
    if set(sc.keys()) - {"gameLogState"} == {"playerStates"}:
        return -2, "SKIP"

    return -1, "UNKNOWN"


def extract_game_actions(filepath):
    """Extract action labels for all sampling points in one game.

    Returns list of dicts with: game_id, event_index, action_index, action_type.
    """
    path = Path(filepath)
    game_id = path.stem

    try:
        with open(filepath) as f:
            game_data = json.load(f)
    except Exception:
        return None

    play_order = game_data["data"].get("playOrder", [])
    if len(play_order) != 4:
        return []
    settings = game_data["data"].get("gameSettings", {})
    if settings.get("victoryPointsToWin", 10) != 10:
        return []

    eh = game_data["data"]["eventHistory"]
    end_game = eh.get("endGameState", {})
    if not any(p.get("winningPlayer") for p in end_game.get("players", {}).values()):
        return []

    # Build coordinate mapper
    init_state = eh["initialState"]
    map_state = init_state["mapState"]
    tiles = adapt_tiles(map_state["tileHexStates"])
    port_edge_states = map_state.get("portEdgeStates", {})
    verts = adapt_vertices(map_state["tileCornerStates"], port_edge_states)
    edges_list = adapt_edges(map_state["tileEdgeStates"])
    hp = extract_harbor_pairs(port_edge_states)
    mapper = build_coordinate_mapper(tiles, verts, edges_list, harbor_pairs=hp)

    vertex_idx_to_xyz, edge_idx_to_xyz = build_index_maps(
        map_state["tileCornerStates"], map_state["tileEdgeStates"],
    )

    # Initialize bank state for tracking deltas
    bank = {}
    for rid_str, cnt in init_state.get("bankState", {}).get("resourceCards", {}).items():
        bank[int(rid_str)] = cnt

    # Initialize player hands
    player_hands = {}
    for col_str, ps in init_state.get("playerStates", {}).items():
        col = int(col_str)
        cards = ps.get("resourceCards", {}).get("cards", [])
        hand = {}
        for c in cards:
            hand[c] = hand.get(c, 0) + 1
        player_hands[col] = hand

    events = eh.get("events", [])
    results = []
    turn_state = init_state.get("currentState", {}).get("turnState", 0)

    for i, ev in enumerate(events):
        sc = ev.get("stateChange", {})
        cs = sc.get("currentState", {})

        # Track state for delta computation
        for rid_str, cnt in sc.get("bankState", {}).get("resourceCards", {}).items():
            bank[int(rid_str)] = cnt
        for col_str, ps in sc.get("playerStates", {}).items():
            col = int(col_str)
            rc = ps.get("resourceCards")
            if rc and "cards" in rc:
                hand = {}
                for c in rc["cards"]:
                    hand[c] = hand.get(c, 0) + 1
                player_hands[col] = hand

        # Check for sampling point
        new_ts = cs.get("turnState")
        if new_ts is not None:
            old_ts = turn_state
            turn_state = new_ts
            if new_ts == 2 and old_ts != 2:
                # Sampling point! Look ahead for first action
                action_idx, action_type = _find_first_action(
                    events, i, mapper, vertex_idx_to_xyz, edge_idx_to_xyz,
                    player_hands, bank,
                )
                results.append({
                    "game_id": game_id,
                    "event_index": i,
                    "action_index": action_idx,
                    "action_type": action_type,
                })

    return results


def _find_first_action(events, sample_idx, mapper, vertex_idx_to_xyz,
                        edge_idx_to_xyz, player_hands, bank):
    """Scan forward from sampling point to find first Discrete(290) action.

    Skips chat messages, player trades, and non-action state changes.
    """
    max_lookahead = 20  # don't look too far ahead
    for j in range(sample_idx + 1, min(sample_idx + max_lookahead + 1, len(events))):
        # Take snapshot of bank before this event (for delta computation)
        prev_bank = dict(bank)

        # Update bank from this event
        sc = events[j].get("stateChange", {})
        for rid_str, cnt in sc.get("bankState", {}).get("resourceCards", {}).items():
            bank[int(rid_str)] = cnt

        action_idx, action_type = _classify_event(
            events[j], mapper, vertex_idx_to_xyz, edge_idx_to_xyz,
            player_hands, prev_bank,
        )

        # Update player hands from this event (for future delta computation)
        for col_str, ps in sc.get("playerStates", {}).items():
            col = int(col_str)
            rc = ps.get("resourceCards")
            if rc and "cards" in rc:
                hand = {}
                for c in rc["cards"]:
                    hand[c] = hand.get(c, 0) + 1
                player_hands[col] = hand

        if action_idx == -2:
            continue  # SKIP sentinel (chat, timer, etc.)
        if action_type == "PLAYER_TRADE" or "tradeState" in sc:
            continue  # skip player trades, look for the real action
        return action_idx, action_type

    return -1, "NO_ACTION_FOUND"


def _safe_extract(filepath):
    try:
        return extract_game_actions(filepath)
    except Exception as e:
        logger.debug(f"Failed: {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract action labels from game JSONs")
    parser.add_argument("--limit", type=int, default=0, help="Max games (0=all)")
    parser.add_argument("--workers", type=int, default=6, help="Num workers")
    parser.add_argument("--games-dir", type=str, default=str(GAMES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    games_dir = Path(args.games_dir)
    game_files = sorted(games_dir.glob("*.json"))
    if args.limit > 0:
        game_files = game_files[:args.limit]

    total_games = len(game_files)
    logger.info(f"Extracting actions from {total_games} games with {args.workers} workers")

    t0 = time.time()
    game_paths = [str(f) for f in game_files]

    all_rows = []
    total_parsed = 0
    total_errors = 0
    total_skipped = 0
    action_counts = Counter()

    pool = Pool(processes=args.workers)
    try:
        pbar = tqdm(
            pool.imap_unordered(_safe_extract, game_paths, chunksize=20),
            total=total_games,
            unit="game",
        )
        for result in pbar:
            if result is None:
                total_errors += 1
            elif len(result) == 0:
                total_skipped += 1
            else:
                total_parsed += 1
                all_rows.extend(result)
                for r in result:
                    action_counts[r["action_type"]] += 1

            pbar.set_postfix_str(
                f"{len(all_rows)}samples {total_parsed}ok {total_errors}err"
            )
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0
    logger.info(f"Extracted {len(all_rows)} action labels from {total_parsed} games "
                f"in {elapsed:.1f}s")

    # Action distribution
    print("\nAction distribution:")
    total = sum(action_counts.values())
    for k, v in action_counts.most_common():
        print(f"  {k:25s}: {v:7d} ({v/total*100:5.1f}%)")

    mappable = sum(v for k, v in action_counts.items()
                   if not k.startswith("UNKNOWN") and not k.startswith("NO_ACTION")
                   and k != "PLAY_DEV_COMPLEX" and k != "BANK_CHANGE_UNKNOWN")
    unmappable = total - mappable
    print(f"\nMappable: {mappable}/{total} ({mappable/total*100:.1f}%)")
    print(f"Unmappable (action_index=-1): {unmappable}/{total} ({unmappable/total*100:.1f}%)")

    # Save to parquet
    df = pd.DataFrame(all_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "actions.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Saved to {out_path}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    valid = df[df["action_index"] >= 0]
    print(f"Valid actions (action_index >= 0): {len(valid)} ({len(valid)/len(df)*100:.1f}%)")
    print(f"Unique actions used: {valid['action_index'].nunique()}")


if __name__ == "__main__":
    main()
