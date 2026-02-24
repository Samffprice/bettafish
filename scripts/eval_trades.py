#!/usr/bin/env python3
"""Evaluate trade decision quality against colonist.io human game data.

Replays games from datasets/games/, reconstructs game state at each trade
response, and compares old (us-only) vs new (opponent-aware) evaluators
against the human accept/reject decision.

Usage:
    python scripts/eval_trades.py [--num-games 200] [--games-dir datasets/games]
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron.models.enums import ActionType, RESOURCES

from bridge.config import COLONIST_RESOURCE_TO_CATAN
from datasets.game_parser import (
    IncrementalGame,
    ReplayState,
    _process_event,
    CATANATRON_COLORS,
    RESOURCE_ID_TO_STR,
)
from datasets.format_adapter import (
    adapt_tiles,
    adapt_vertices,
    adapt_edges,
    extract_harbor_pairs,
    build_index_maps,
)
from bridge.coordinate_map import build_coordinate_mapper

logger = logging.getLogger(__name__)

# Trade response codes from colonist.io
RESP_PENDING = 0
RESP_ACCEPT = 1
RESP_REJECT = 2
RESP_COUNTER = 3


def _load_value_fn(model_path, blend_weight):
    """Load the blend value function."""
    from robottler.search_player import make_bb_blended_value_fn
    return make_bb_blended_value_fn(model_path, blend_weight=blend_weight)


def _eval_old(offered_catan, wanted_catan, game, our_color, creator_vps,
              vps_to_win, bb_value_fn):
    """Old evaluator: our position change only (no opponent awareness)."""
    from robottler.bitboard.convert import game_to_bitboard
    from robottler.bitboard.state import PS_WOOD
    from robottler.bitboard.masks import RESOURCE_INDEX

    if not offered_catan and not wanted_catan:
        return -1.0
    if vps_to_win > 0 and creator_vps >= vps_to_win - 1:
        return -100.0

    bb = game_to_bitboard(game)
    p_idx = bb.color_to_index[our_color]

    for resource, needed in Counter(wanted_catan).items():
        ri = PS_WOOD + RESOURCE_INDEX[resource]
        if bb.player_state[p_idx][ri] < needed:
            return -100.0

    value_before = bb_value_fn(bb, our_color)

    for r in wanted_catan:
        bb.player_state[p_idx][PS_WOOD + RESOURCE_INDEX[r]] -= 1
    for r in offered_catan:
        bb.player_state[p_idx][PS_WOOD + RESOURCE_INDEX[r]] += 1

    value_after = bb_value_fn(bb, our_color)
    return value_after - value_before


def _eval_new(offered_catan, wanted_catan, game, our_color, creator_vps,
              vps_to_win, bb_value_fn, creator_color):
    """New evaluator: opponent-aware with 1-ply lookahead."""
    from bridge.bot_interface import _evaluate_trade_blend
    return _evaluate_trade_blend(
        offered_catan, wanted_catan, game, our_color,
        creator_vps, vps_to_win, bb_value_fn,
        creator_color=creator_color,
    )


def extract_trade_decisions(filepath):
    """Extract all trade response decisions from a game file.

    Returns list of dicts with:
        - trade_id, creator, responder
        - offered_resources, wanted_resources (colonist IDs)
        - response (1=accept, 2=reject)
        - event_index (for state reconstruction timing)
    """
    with open(filepath) as f:
        data = json.load(f)

    play_order = data["data"].get("playOrder", [])
    if len(play_order) != 4:
        return [], data

    events = data["data"]["eventHistory"]["events"]

    # Track active offers and their full details
    active_offers = {}  # trade_id -> {creator, offered, wanted}
    decisions = []

    for i, ev in enumerate(events):
        sc = ev.get("stateChange", {})
        ts = sc.get("tradeState", {})
        if not ts:
            continue

        # New offers (have full details)
        ao = ts.get("activeOffers", {}) or {}
        for tid, offer in ao.items():
            if offer is None:
                # Offer removed
                active_offers.pop(tid, None)
                continue
            # Full offer creation (has creator + resources)
            if offer.get("creator") is not None and offer.get("offeredResources") is not None:
                active_offers[tid] = {
                    "creator": offer["creator"],
                    "offered": offer["offeredResources"],
                    "wanted": offer["wantedResources"],
                    "create_event": i,
                }
            # Response updates
            pr = offer.get("playerResponses", {}) or {}
            for pid_str, resp in pr.items():
                if resp in (RESP_ACCEPT, RESP_REJECT):
                    pid = int(pid_str)
                    info = active_offers.get(tid)
                    if info is None:
                        continue
                    decisions.append({
                        "trade_id": tid,
                        "creator": info["creator"],
                        "responder": pid,
                        "offered": list(info["offered"]),
                        "wanted": list(info["wanted"]),
                        "response": resp,
                        "event_index": i,
                        "create_event": info["create_event"],
                    })

    return decisions, data


def reconstruct_at_event(data, target_event):
    """Reconstruct game state up to (but not including) target_event.

    Returns (game, colonist_to_catan, rstate) or (None, None, None) on failure.
    """
    play_order = data["data"]["playOrder"]
    eh = data["data"]["eventHistory"]
    init_state = eh["initialState"]
    map_state = init_state["mapState"]

    colonist_to_catan = {}
    rstate = ReplayState()
    rstate.play_order = list(play_order)
    for i, col_color in enumerate(play_order):
        colonist_to_catan[col_color] = CATANATRON_COLORS[i]
        rstate.resources[col_color] = {r: 0 for r in RESOURCES}
        rstate.dev_cards[col_color] = {d: 0 for d in
                                        ["KNIGHT", "VICTORY_POINT", "YEAR_OF_PLENTY",
                                         "MONOPOLY", "ROAD_BUILDING"]}
        rstate.dev_cards_used[col_color] = {d: 0 for d in
                                             ["KNIGHT", "VICTORY_POINT", "YEAR_OF_PLENTY",
                                              "MONOPOLY", "ROAD_BUILDING"]}
        rstate.vp[col_color] = 0
    rstate.colonist_to_catan = colonist_to_catan

    try:
        tiles = adapt_tiles(map_state["tileHexStates"])
        port_edge_states = map_state.get("portEdgeStates", {})
        verts = adapt_vertices(map_state["tileCornerStates"], port_edge_states)
        edges_list = adapt_edges(map_state["tileEdgeStates"])
        hp = extract_harbor_pairs(port_edge_states)
        mapper = build_coordinate_mapper(tiles, verts, edges_list, harbor_pairs=hp)
    except Exception as e:
        logger.debug(f"Failed to build mapper: {e}")
        return None, None, None

    rstate.vertex_idx_to_xyz, rstate.edge_idx_to_xyz = build_index_maps(
        map_state["tileCornerStates"], map_state["tileEdgeStates"],
    )

    igame = IncrementalGame(mapper, play_order, colonist_to_catan)

    # Initial state setup
    robber_s = init_state.get("mechanicRobberState", {})
    rstate.robber_tile_index = robber_s.get("locationTileIndex", -1)
    if rstate.robber_tile_index >= 0:
        igame.set_robber(rstate.robber_tile_index)

    bank = init_state.get("bankState", {}).get("resourceCards", {})
    for rid_str, count in bank.items():
        res = RESOURCE_ID_TO_STR.get(int(rid_str))
        if res:
            rstate.bank_resources[res] = count

    cs = init_state.get("currentState", {})
    rstate.turn_state = cs.get("turnState", 0)
    rstate.action_state = cs.get("actionState", 0)
    rstate.current_turn_player = cs.get("currentTurnPlayerColor", -1)
    rstate.is_setup_phase = rstate.turn_state == 0

    for col_str, ps in init_state.get("playerStates", {}).items():
        col_color = int(col_str)
        if col_color not in rstate.resources:
            continue
        cards = ps.get("resourceCards", {}).get("cards", [])
        hand = {r: 0 for r in RESOURCES}
        for card_id in cards:
            res = RESOURCE_ID_TO_STR.get(card_id)
            if res:
                hand[res] += 1
        rstate.resources[col_color] = hand

    # Process events up to target
    events = eh.get("events", [])
    for i in range(min(target_event, len(events))):
        _process_event(rstate, igame, events[i])

    # Prepare game for evaluation
    try:
        igame.prepare_for_sampling(rstate)
    except Exception as e:
        logger.debug(f"prepare_for_sampling failed at event {target_event}: {e}")
        return None, None, None

    return igame.game, colonist_to_catan, rstate


def evaluate_games(games_dir, num_games, model_path, blend_weight):
    """Run old and new evaluators on trade decisions from game files."""
    game_files = sorted(Path(games_dir).glob("*.json"))
    if not game_files:
        print(f"No game files found in {games_dir}")
        return

    random.seed(42)
    sample = random.sample(game_files, min(num_games, len(game_files)))

    print(f"Loading value function from {model_path}...")
    bb_value_fn = _load_value_fn(model_path, blend_weight)
    print("Value function loaded.\n")

    # Counters
    stats = {
        "total_responses": 0,
        "total_accepts": 0,
        "total_rejects": 0,
        "reconstruction_failures": 0,
        "eval_errors": 0,
        "games_with_trades": 0,
        "games_processed": 0,
        # Old evaluator
        "old_accept": 0,
        "old_reject": 0,
        "old_agree_accept": 0,  # both human and old say accept
        "old_agree_reject": 0,  # both say reject
        # New evaluator
        "new_accept": 0,
        "new_reject": 0,
        "new_agree_accept": 0,
        "new_agree_reject": 0,
        # Score distributions
        "old_scores_accept": [],  # old scores when human accepted
        "old_scores_reject": [],  # old scores when human rejected
        "new_scores_accept": [],
        "new_scores_reject": [],
    }

    t0 = time.time()

    for gi, game_file in enumerate(sample):
        if (gi + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Processing game {gi+1}/{len(sample)} "
                  f"({elapsed:.1f}s, {stats['total_responses']} responses)...")

        decisions, data = extract_trade_decisions(str(game_file))
        if not decisions:
            stats["games_processed"] += 1
            continue

        stats["games_with_trades"] += 1
        stats["games_processed"] += 1

        # Group decisions by create_event to share reconstruction
        by_create = defaultdict(list)
        for d in decisions:
            by_create[d["create_event"]].append(d)

        for create_evt, decs in by_create.items():
            # Reconstruct once per trade creation point
            game, c2c, rstate = reconstruct_at_event(data, create_evt)
            if game is None:
                stats["reconstruction_failures"] += len(decs)
                continue

            for dec in decs:
                human_accept = dec["response"] == RESP_ACCEPT

                responder_color = c2c.get(dec["responder"])
                creator_color = c2c.get(dec["creator"])
                if responder_color is None or creator_color is None:
                    stats["reconstruction_failures"] += 1
                    continue

                offered_catan = [COLONIST_RESOURCE_TO_CATAN[r]
                                 for r in dec["offered"]
                                 if r in COLONIST_RESOURCE_TO_CATAN]
                wanted_catan = [COLONIST_RESOURCE_TO_CATAN[r]
                                for r in dec["wanted"]
                                if r in COLONIST_RESOURCE_TO_CATAN]

                # Get creator VPs
                creator_vps = 0
                if creator_color in game.state.color_to_index:
                    c_idx = game.state.color_to_index[creator_color]
                    creator_vps = game.state.player_state.get(
                        f"P{c_idx}_ACTUAL_VICTORY_POINTS", 0)

                # Need to set correct resources for the responder
                resp_col = dec["responder"]
                resp_idx = game.state.color_to_index[responder_color]
                hand = rstate.resources.get(resp_col, {})
                for resource in RESOURCES:
                    game.state.player_state[f"P{resp_idx}_{resource}_IN_HAND"] = \
                        hand.get(resource, 0)

                try:
                    old_score = _eval_old(
                        offered_catan, wanted_catan, game, responder_color,
                        creator_vps, 10, bb_value_fn,
                    )
                    new_score = _eval_new(
                        offered_catan, wanted_catan, game, responder_color,
                        creator_vps, 10, bb_value_fn, creator_color,
                    )
                except Exception as e:
                    logger.debug(f"Eval error: {e}")
                    stats["eval_errors"] += 1
                    continue

                stats["total_responses"] += 1
                if human_accept:
                    stats["total_accepts"] += 1
                else:
                    stats["total_rejects"] += 1

                old_accept = old_score > 0
                new_accept = new_score > 0

                if old_accept:
                    stats["old_accept"] += 1
                else:
                    stats["old_reject"] += 1
                if new_accept:
                    stats["new_accept"] += 1
                else:
                    stats["new_reject"] += 1

                if human_accept and old_accept:
                    stats["old_agree_accept"] += 1
                if not human_accept and not old_accept:
                    stats["old_agree_reject"] += 1
                if human_accept and new_accept:
                    stats["new_agree_accept"] += 1
                if not human_accept and not new_accept:
                    stats["new_agree_reject"] += 1

                if human_accept:
                    stats["old_scores_accept"].append(old_score)
                    stats["new_scores_accept"].append(new_score)
                else:
                    stats["old_scores_reject"].append(old_score)
                    stats["new_scores_reject"].append(new_score)

    elapsed = time.time() - t0
    _print_results(stats, len(sample), elapsed)


def _print_results(stats, num_games, elapsed):
    """Print comparison results."""
    total = stats["total_responses"]
    if total == 0:
        print("No trade responses found.")
        return

    accepts = stats["total_accepts"]
    rejects = stats["total_rejects"]

    old_agree = stats["old_agree_accept"] + stats["old_agree_reject"]
    new_agree = stats["new_agree_accept"] + stats["new_agree_reject"]

    old_accept_match = (stats["old_agree_accept"] / accepts * 100) if accepts > 0 else 0
    old_reject_match = (stats["old_agree_reject"] / rejects * 100) if rejects > 0 else 0
    new_accept_match = (stats["new_agree_accept"] / accepts * 100) if accepts > 0 else 0
    new_reject_match = (stats["new_agree_reject"] / rejects * 100) if rejects > 0 else 0

    old_overall = old_agree / total * 100
    new_overall = new_agree / total * 100

    old_rate = stats["old_accept"] / total * 100
    new_rate = stats["new_accept"] / total * 100
    human_rate = accepts / total * 100

    print(f"\n{'='*65}")
    print(f"Trade Evaluation Comparison ({num_games} games, "
          f"{total} responses, {elapsed:.1f}s)")
    print(f"{'='*65}")
    print(f"  Games processed: {stats['games_processed']}, "
          f"with trades: {stats['games_with_trades']}")
    print(f"  Reconstruction failures: {stats['reconstruction_failures']}")
    print(f"  Eval errors: {stats['eval_errors']}")
    print()

    print(f"{'':24s} {'Old (us-only)':>15s}   {'New (opp-aware)':>15s}")
    print(f"  {'─'*56}")
    print(f"  Human Accept match:   {old_accept_match:>14.1f}%   {new_accept_match:>14.1f}%")
    print(f"  Human Reject match:   {old_reject_match:>14.1f}%   {new_reject_match:>14.1f}%")
    print(f"  Overall agreement:    {old_overall:>14.1f}%   {new_overall:>14.1f}%")
    print()
    print(f"  Accept rate (ours):   {old_rate:>14.1f}%   {new_rate:>14.1f}%")
    print(f"  Accept rate (human):  {human_rate:>14.1f}%")
    print()

    # Score distribution summary
    if stats["old_scores_accept"]:
        old_acc_mean = sum(stats["old_scores_accept"]) / len(stats["old_scores_accept"])
        new_acc_mean = sum(stats["new_scores_accept"]) / len(stats["new_scores_accept"])
        print(f"  Mean score (human accept): old={old_acc_mean:+.4f}  new={new_acc_mean:+.4f}")
    if stats["old_scores_reject"]:
        old_rej_mean = sum(stats["old_scores_reject"]) / len(stats["old_scores_reject"])
        new_rej_mean = sum(stats["new_scores_reject"]) / len(stats["new_scores_reject"])
        print(f"  Mean score (human reject): old={old_rej_mean:+.4f}  new={new_rej_mean:+.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trade decisions against human data")
    parser.add_argument("--num-games", type=int, default=200,
                        help="Number of games to sample (default: 200)")
    parser.add_argument("--games-dir", default="datasets/games",
                        help="Directory containing game JSON files")
    parser.add_argument("--model-path", default="robottler/models/value_net_v2.pt",
                        help="Path to value network checkpoint")
    parser.add_argument("--blend-weight", type=float, default=1e8,
                        help="Blend weight for heuristic+neural (default: 1e8)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    evaluate_games(args.games_dir, args.num_games, args.model_path, args.blend_weight)


if __name__ == "__main__":
    main()
