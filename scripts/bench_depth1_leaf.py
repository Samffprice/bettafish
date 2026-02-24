#!/usr/bin/env python3
"""Benchmark BBMCTSPlayer with depth-1 leaf evaluation vs AlphaBeta.

Usage:
    python3 scripts/bench_depth1_leaf.py <az_checkpoint> <num_games> [workers] [sims] [mode]

Modes:
    neural           - AZ net evaluates each child, take max/min (default)
    heuristic_select - Heuristic ranks children, AZ net evaluates best
    heuristic_rank   - Heuristic evaluates children, rank-normalize to [-1,+1]
    baseline         - No depth-1 (standard MCTS for comparison)

Example:
    python3 scripts/bench_depth1_leaf.py datasets/az_v2_325k_200ep.pt 50 6 400 heuristic_select
    python3 scripts/bench_depth1_leaf.py datasets/az_v2_325k_200ep.pt 50 6 400 baseline
"""

import sys
import os
import time
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def worker(args):
    """Play one game: BBMCTSPlayer (P1) vs AlphaBeta (P2)."""
    seed, az_path, num_sims, leaf_depth1 = args

    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)

    from catanatron.models.player import Color
    from catanatron.game import Game
    from catanatron.players.minimax import AlphaBetaPlayer

    from robottler.bb_mcts_player import make_bb_mcts_player

    p1 = make_bb_mcts_player(
        Color.RED, az_path,
        num_simulations=num_sims, c_puct=1.4, temperature=0.0,
        dirichlet_alpha=0.0, dirichlet_weight=0.0,
        leaf_depth1=leaf_depth1 if leaf_depth1 != "baseline" else None,
    )
    p2 = AlphaBetaPlayer(Color.BLUE)

    game = Game(players=[p1, p2], vps_to_win=10)
    p1.fi.update_map(game.state.board.map)

    game.play()

    winner = game.winning_color()
    return 1 if winner == Color.RED else 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    az_path = sys.argv[1]
    num_games = int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    num_sims = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    mode = sys.argv[5] if len(sys.argv) > 5 else "neural"

    if mode not in ("neural", "heuristic_select", "heuristic_rank", "baseline"):
        print(f"Unknown mode: {mode}")
        print("Valid modes: neural, heuristic_select, heuristic_rank, baseline")
        sys.exit(1)

    print(f"BBMCTSPlayer (leaf_depth1={mode}, sims={num_sims}) vs AlphaBeta")
    print(f"Games: {num_games}, Workers: {workers}")
    print("=" * 60)

    mp.set_start_method("fork", force=True)
    tasks = [(i, az_path, num_sims, mode) for i in range(num_games)]

    start = time.time()
    wins = 0
    completed = 0

    with mp.Pool(workers) as pool:
        for result in pool.imap_unordered(worker, tasks):
            wins += result
            completed += 1
            elapsed = time.time() - start
            rate = completed / elapsed
            pct = 100 * wins / completed
            print(f"\r  {completed}/{num_games}: {wins}W {completed-wins}L "
                  f"({pct:.0f}%) [{rate:.1f} games/s, {elapsed:.0f}s elapsed]",
                  end="", flush=True)

    elapsed = time.time() - start
    pct = 100 * wins / num_games

    # Wilson confidence interval
    import math
    z = 1.96
    n = num_games
    p_hat = wins / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    lo = max(0, center - margin) * 100
    hi = min(1, center + margin) * 100

    print(f"\n\nResult: {wins}/{num_games} = {pct:.1f}% [{lo:.0f}%, {hi:.0f}%]")
    print(f"Time: {elapsed:.0f}s ({elapsed/num_games:.1f}s/game)")


if __name__ == "__main__":
    main()
