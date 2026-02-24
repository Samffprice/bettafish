"""Extract GNN features from colonist.io human game JSONs.

Replays each game through Catanatron incrementally, converts to BitboardState
at each sampling point (after dice roll), and extracts per-node [54, 18] and
global [76] features from the acting player's perspective.

Saves in chunks to manage memory (~700 MB per 2000-game chunk), then
concatenates into final .npy files compatible with ``az_selfplay train --gnn``.

Usage:
    python3 -m datasets.extract_gnn_features \\
        --games-dir datasets/games \\
        --output-dir datasets/human_gnn_44k \\
        --workers 6

Output:
    node_features.npy   [N, 54, 18]
    global_features.npy [N, 76]
    values.npy          [N]
"""

import argparse
import json
import logging
import os
import sys
import time
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.player import Color

from bridge.coordinate_map import build_coordinate_mapper
from bridge.config import COLONIST_RESOURCE_TO_CATAN
from datasets.format_adapter import (
    adapt_tiles, adapt_vertices, adapt_edges,
    extract_harbor_pairs, build_index_maps,
)
from datasets.game_parser import (
    IncrementalGame, ReplayState, _process_event,
    CATANATRON_COLORS, RESOURCE_ID_TO_STR,
)
from robottler.bitboard.convert import game_to_bitboard
from robottler.bitboard.features import _build_node_prod
from robottler.gnn_net import (
    NODE_FEAT_DIM, GLOBAL_FEAT_DIM,
    bb_fill_node_features, bb_fill_global_features,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-game extraction
# ---------------------------------------------------------------------------

def extract_gnn_from_game(filepath):
    """Extract GNN features from one colonist.io game JSON.

    Replays the game, and at each sampling point (turnState → 2, i.e. after
    dice roll) extracts features from the acting player's perspective.

    Returns:
        (node_features [K, 54, 18], global_features [K, 76], values [K])
        or None if the game is invalid / has no samples.
    """
    with open(filepath, "r") as f:
        game_data = json.load(f)

    # Validate: 4 players, 10 VPs to win
    play_order = game_data["data"].get("playOrder", [])
    if len(play_order) != 4:
        return None
    settings = game_data["data"].get("gameSettings", {})
    if settings.get("victoryPointsToWin", 10) != 10:
        return None

    eh = game_data["data"]["eventHistory"]
    end_game = eh.get("endGameState", {})
    end_players = end_game.get("players", {})

    winner_color = None
    for col_str, pdata in end_players.items():
        if pdata.get("winningPlayer"):
            winner_color = int(col_str)
    if winner_color is None:
        return None

    # ---- Setup (mirrors game_parser.parse_game) ----
    init_state = eh["initialState"]
    map_state = init_state["mapState"]

    colonist_to_catan = {}
    rstate = ReplayState()
    rstate.play_order = list(play_order)
    for i, col_color in enumerate(play_order):
        colonist_to_catan[col_color] = CATANATRON_COLORS[i]
        rstate.resources[col_color] = {r: 0 for r in RESOURCES}
        rstate.dev_cards[col_color] = {d: 0 for d in DEVELOPMENT_CARDS}
        rstate.dev_cards_used[col_color] = {d: 0 for d in DEVELOPMENT_CARDS}
        rstate.vp[col_color] = 0
    rstate.colonist_to_catan = colonist_to_catan

    # Coordinate mapper
    tiles = adapt_tiles(map_state["tileHexStates"])
    port_edge_states = map_state.get("portEdgeStates", {})
    verts = adapt_vertices(map_state["tileCornerStates"], port_edge_states)
    edges_list = adapt_edges(map_state["tileEdgeStates"])
    hp = extract_harbor_pairs(port_edge_states)
    mapper = build_coordinate_mapper(tiles, verts, edges_list, harbor_pairs=hp)

    rstate.vertex_idx_to_xyz, rstate.edge_idx_to_xyz = build_index_maps(
        map_state["tileCornerStates"], map_state["tileEdgeStates"],
    )

    igame = IncrementalGame(mapper, play_order, colonist_to_catan)

    # Initial state
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

    # ---- Precompute static data ----
    node_prod = _build_node_prod(igame.game.state.board.map)
    node_buf = np.zeros((54, NODE_FEAT_DIM), dtype=np.float32)
    global_buf = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)

    # ---- Event loop ----
    nf_list = []
    gf_list = []
    val_list = []
    events = eh.get("events", [])

    for event in events:
        is_sample = _process_event(rstate, igame, event)
        if not is_sample:
            continue

        try:
            igame.prepare_for_sampling(rstate)
        except Exception:
            continue

        acting_col = rstate.current_turn_player
        acting_catan = colonist_to_catan.get(acting_col)
        if acting_catan is None:
            continue

        try:
            bb_state = game_to_bitboard(igame.game)
        except Exception:
            continue

        node_buf[:] = 0.0
        global_buf[:] = 0.0
        bb_fill_node_features(bb_state, acting_catan, node_buf, node_prod)
        bb_fill_global_features(bb_state, acting_catan, global_buf)

        value = 1.0 if acting_col == winner_color else -1.0

        nf_list.append(node_buf.copy())
        gf_list.append(global_buf.copy())
        val_list.append(value)

    if not nf_list:
        return None

    return (
        np.stack(nf_list),                                   # [K, 54, 18]
        np.stack(gf_list),                                   # [K, 76]
        np.array(val_list, dtype=np.float32),                # [K]
    )


def _safe_extract(filepath):
    """Multiprocessing wrapper — catches all exceptions."""
    try:
        return extract_gnn_from_game(filepath)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chunk concatenation
# ---------------------------------------------------------------------------

def concatenate_chunks(output_dir):
    """Merge chunk_*.npz files into single node_features.npy etc.

    Uses numpy memmap to write the final files so memory stays bounded
    (only one chunk loaded at a time).
    """
    chunk_files = sorted(glob(os.path.join(output_dir, 'chunk_*.npz')))
    if not chunk_files:
        print("No chunk files to concatenate.")
        return

    # Count total samples across all chunks
    total = 0
    for f in chunk_files:
        with np.load(f) as d:
            total += d['values'].shape[0]
    print(f"Concatenating {total:,} samples from {len(chunk_files)} chunks...")

    # Write final files via memmap (streaming, low memory)
    nf_path = os.path.join(output_dir, 'node_features.npy')
    gf_path = os.path.join(output_dir, 'global_features.npy')
    val_path = os.path.join(output_dir, 'values.npy')

    final_nf = np.lib.format.open_memmap(
        nf_path, mode='w+', dtype=np.float32, shape=(total, 54, NODE_FEAT_DIM))
    final_gf = np.lib.format.open_memmap(
        gf_path, mode='w+', dtype=np.float32, shape=(total, GLOBAL_FEAT_DIM))
    final_v = np.lib.format.open_memmap(
        val_path, mode='w+', dtype=np.float32, shape=(total,))

    offset = 0
    for f in tqdm(chunk_files, desc="Merging"):
        with np.load(f) as d:
            n = d['values'].shape[0]
            final_nf[offset:offset + n] = d['node_features']
            final_gf[offset:offset + n] = d['global_features']
            final_v[offset:offset + n] = d['values']
            offset += n
        final_nf.flush()
        final_gf.flush()
        final_v.flush()

    del final_nf, final_gf, final_v
    print(f"  node_features.npy:   [{total}, 54, {NODE_FEAT_DIM}]")
    print(f"  global_features.npy: [{total}, {GLOBAL_FEAT_DIM}]")
    print(f"  values.npy:          [{total}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract GNN features from colonist.io game JSONs")
    parser.add_argument('--games-dir', default='datasets/games',
                        help='Directory with *.json game files')
    parser.add_argument('--output-dir', default='datasets/human_gnn_44k',
                        help='Output directory for .npy files')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel workers (0 = sequential)')
    parser.add_argument('--chunk-size', type=int, default=2000,
                        help='Games per saved chunk')
    parser.add_argument('--concat-only', action='store_true',
                        help='Only run the final chunk concatenation step')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.concat_only:
        concatenate_chunks(args.output_dir)
        return

    # Find all game files
    game_files = sorted(glob(os.path.join(args.games_dir, '*.json')))
    print(f"Found {len(game_files):,} game files in {args.games_dir}")

    # Resume: count existing chunks
    existing_chunks = sorted(glob(os.path.join(args.output_dir, 'chunk_*.npz')))
    start_chunk = len(existing_chunks)
    start_game = start_chunk * args.chunk_size

    if start_game > 0:
        print(f"  Resuming from chunk {start_chunk} (game {start_game:,})")

    n_chunks = (len(game_files) + args.chunk_size - 1) // args.chunk_size
    total_samples = 0
    total_games = 0
    total_errors = 0
    t0 = time.time()

    for chunk_idx in range(start_chunk, n_chunks):
        chunk_start = chunk_idx * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(game_files))
        chunk_game_files = game_files[chunk_start:chunk_end]

        # Process games
        if args.workers > 1:
            with Pool(args.workers) as pool:
                results = list(tqdm(
                    pool.imap(_safe_extract, chunk_game_files),
                    total=len(chunk_game_files),
                    desc=f"Chunk {chunk_idx}/{n_chunks - 1}",
                ))
        else:
            results = []
            for f in tqdm(chunk_game_files, desc=f"Chunk {chunk_idx}/{n_chunks - 1}"):
                results.append(_safe_extract(f))

        # Accumulate chunk results
        chunk_nf = []
        chunk_gf = []
        chunk_vals = []

        for result in results:
            if result is None:
                total_errors += 1
                continue
            nf, gf, vals = result
            chunk_nf.append(nf)
            chunk_gf.append(gf)
            chunk_vals.append(vals)
            total_games += 1

        if not chunk_nf:
            print(f"  Chunk {chunk_idx}: 0 samples (all errors)")
            continue

        nf_arr = np.concatenate(chunk_nf)
        gf_arr = np.concatenate(chunk_gf)
        val_arr = np.concatenate(chunk_vals)
        del chunk_nf, chunk_gf, chunk_vals

        # Save chunk
        chunk_path = os.path.join(args.output_dir, f'chunk_{chunk_idx:04d}.npz')
        np.savez(chunk_path, node_features=nf_arr,
                 global_features=gf_arr, values=val_arr)

        n_chunk = len(val_arr)
        total_samples += n_chunk
        elapsed = time.time() - t0
        gps = total_games / elapsed if elapsed > 0 else 0
        print(f"  Chunk {chunk_idx}: {n_chunk:,} samples from "
              f"{len([r for r in results if r is not None])} games  "
              f"[total: {total_samples:,} samples, {total_games:,} games, "
              f"{gps:.1f} games/sec]")
        del nf_arr, gf_arr, val_arr

    print(f"\nExtraction complete: {total_samples:,} samples from "
          f"{total_games:,} games ({total_errors} skipped)")
    print(f"Elapsed: {time.time() - t0:.0f}s")

    # Concatenate chunks into final files
    concatenate_chunks(args.output_dir)

    print("\nTo train GNN on this data:")
    print(f"  python3 -m robottler.az_selfplay train \\")
    print(f"      --data-dir {args.output_dir} \\")
    print(f"      --output datasets/gnn_human_v1.pt \\")
    print(f"      --gnn --gnn-dims 32,64,96 \\")
    print(f"      --epochs 200 --batch-size 2048 --lr 1e-3 \\")
    print(f"      --scheduler cosine --dropout 0.2 --edge-dropout 0.1")


if __name__ == '__main__':
    main()
