#!/usr/bin/env python3
"""Benchmark any combination of BBSearch and MCTS players head-to-head.

Usage:
    # BBSearch vs BBSearch
    python3 scripts/bench_bbsearch_vs_bbsearch.py \
        --p1-type bbsearch --p1-model robottler/models/value_net.pt --p1-depth 2 \
        --p2-type bbsearch --p2-model robottler/models/value_net_v2.pt --p2-depth 3 \
        --games 50 --workers 6

    # MCTS vs BBSearch
    python3 scripts/bench_bbsearch_vs_bbsearch.py \
        --p1-type mcts --p1-model datasets/az_v2_325k_200ep.pt --p1-sims 400 \
        --p2-type bbsearch --p2-model robottler/models/value_net_v2.pt --p2-depth 2 \
        --games 50 --workers 6

Per-player flags:
    --p1-type / --p2-type      Player type: bbsearch or mcts (default: bbsearch)
    --p1-model / --p2-model    Checkpoint path (BC .pt for bbsearch, AZ .pt for mcts)
    --p1-depth / --p2-depth    Search depth for bbsearch (default: 2)
    --p1-eval  / --p2-eval     Eval mode for bbsearch: blend, neural, heuristic (default: blend)
    --p1-sims  / --p2-sims     Simulations for mcts (default: 400)

Shared flags:
    --blend-weight             Blend weight for bbsearch players (default: 1e8)
    --dice-sample              Dice sampling for bbsearch players (default: 5)
    --games                    Number of games (default: 50)
    --workers                  Parallel workers (default: 6)
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_value_fn(model_path, eval_mode, blend_weight):
    """Create a value function based on eval mode (for bbsearch)."""
    if eval_mode == "heuristic":
        from robottler.search_player import make_bb_heuristic_value_fn
        return make_bb_heuristic_value_fn()
    elif eval_mode == "neural":
        from robottler.search_player import make_bb_neural_value_fn
        return make_bb_neural_value_fn(model_path)
    elif eval_mode == "blend":
        from robottler.search_player import make_bb_blended_value_fn
        return make_bb_blended_value_fn(model_path, blend_weight=blend_weight)
    else:
        raise ValueError(f"Unknown eval mode: {eval_mode}")


def make_player(color, cfg):
    """Create a player from config dict."""
    if cfg["type"] == "bbsearch":
        from robottler.search_player import BitboardSearchPlayer
        vfn = make_value_fn(cfg["model"], cfg["eval"], cfg["blend_weight"])
        player = BitboardSearchPlayer(
            color, bb_value_fn=vfn,
            depth=cfg["depth"], dice_sample_size=cfg["dice_sample"],
        )
    elif cfg["type"] == "mcts":
        from robottler.bb_mcts_player import make_bb_mcts_player
        player = make_bb_mcts_player(
            color, cfg["model"],
            num_simulations=cfg["sims"], c_puct=1.4, temperature=0.0,
            dirichlet_alpha=0.0, dirichlet_weight=0.0,
        )
    else:
        raise ValueError(f"Unknown player type: {cfg['type']}")

    if cfg.get("fog"):
        from robottler.fog_of_war import FogOfWarPlayer
        player = FogOfWarPlayer(player, use_counting=cfg.get("counting", True))

    return player


def worker(args):
    """Play one game between two players."""
    game_idx, p1_cfg, p2_cfg = args

    import numpy as np
    import random
    random.seed(game_idx)
    np.random.seed(game_idx)

    from catanatron.models.player import Color
    from catanatron.game import Game

    # Alternate colors to remove first-player bias
    if game_idx % 2 == 0:
        p1_color, p2_color = Color.RED, Color.BLUE
    else:
        p1_color, p2_color = Color.BLUE, Color.RED

    p1 = make_player(p1_color, p1_cfg)
    p2 = make_player(p2_color, p2_cfg)

    if game_idx % 2 == 0:
        game = Game(players=[p1, p2], vps_to_win=10)
    else:
        game = Game(players=[p2, p1], vps_to_win=10)

    # MCTS players need map sync
    for p in [p1, p2]:
        if hasattr(p, 'fi'):
            p.fi.update_map(game.state.board.map)

    game.play()
    winner = game.winning_color()
    return 1 if winner == p1_color else 0


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p_hat = wins / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return center, max(0, center - margin), min(1, center + margin)


def label_for(cfg):
    if cfg["type"] == "mcts":
        model_name = os.path.basename(cfg["model"]).replace(".pt", "")
        label = f"MCTS {model_name} sims={cfg['sims']}"
    else:
        parts = []
        model_name = os.path.basename(cfg["model"]).replace(".pt", "") if cfg["model"] else "none"
        parts.append(f"BBSearch {model_name}")
        parts.append(f"d{cfg['depth']}")
        parts.append(cfg["eval"])
        if cfg["eval"] == "blend":
            parts.append(f"w{cfg['blend_weight']:.0e}")
        label = " ".join(parts)
    if cfg.get("fog"):
        label += " +fog" + ("" if cfg.get("counting", True) else "-nocounting")
    return label


def main():
    parser = argparse.ArgumentParser(description="BBSearch / MCTS head-to-head benchmark")
    parser.add_argument("--p1-type", type=str, default="bbsearch",
                        choices=["bbsearch", "mcts"])
    parser.add_argument("--p1-model", type=str, required=True)
    parser.add_argument("--p1-depth", type=int, default=2)
    parser.add_argument("--p1-eval", type=str, default="blend",
                        choices=["blend", "neural", "heuristic"])
    parser.add_argument("--p1-sims", type=int, default=400)
    parser.add_argument("--p2-type", type=str, default="bbsearch",
                        choices=["bbsearch", "mcts"])
    parser.add_argument("--p2-model", type=str, required=True)
    parser.add_argument("--p2-depth", type=int, default=2)
    parser.add_argument("--p2-eval", type=str, default="blend",
                        choices=["blend", "neural", "heuristic"])
    parser.add_argument("--p2-sims", type=int, default=400)
    parser.add_argument("--p1-fog", action="store_true",
                        help="Wrap P1 in fog-of-war (hide opponent hands)")
    parser.add_argument("--p2-fog", action="store_true",
                        help="Wrap P2 in fog-of-war (hide opponent hands)")
    parser.add_argument("--no-counting", action="store_true",
                        help="With fog, disable card counting (distribute total evenly)")
    parser.add_argument("--blend-weight", type=float, default=1e8)
    parser.add_argument("--dice-sample", type=int, default=5)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    use_counting = not args.no_counting
    p1_cfg = {
        "type": args.p1_type, "model": args.p1_model,
        "depth": args.p1_depth, "eval": args.p1_eval,
        "sims": args.p1_sims, "blend_weight": args.blend_weight,
        "dice_sample": args.dice_sample,
        "fog": args.p1_fog, "counting": use_counting,
    }
    p2_cfg = {
        "type": args.p2_type, "model": args.p2_model,
        "depth": args.p2_depth, "eval": args.p2_eval,
        "sims": args.p2_sims, "blend_weight": args.blend_weight,
        "dice_sample": args.dice_sample,
        "fog": args.p2_fog, "counting": use_counting,
    }

    p1_label = label_for(p1_cfg)
    p2_label = label_for(p2_cfg)

    print(f"P1: {p1_label}")
    print(f"P2: {p2_label}")
    print(f"Games: {args.games}, Workers: {args.workers}, Colors alternate each game")
    print("=" * 60)

    mp.set_start_method("fork", force=True)
    tasks = [(i, p1_cfg, p2_cfg) for i in range(args.games)]

    start = time.time()
    wins = 0
    completed = 0

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(worker, tasks):
            wins += result
            completed += 1
            elapsed = time.time() - start
            rate = completed / elapsed
            pct = 100 * wins / completed
            print(f"\r  {completed}/{args.games}: P1 {wins}W {completed-wins}L "
                  f"({pct:.0f}%) [{rate:.1f} games/s, {elapsed:.0f}s elapsed]",
                  end="", flush=True)

    elapsed = time.time() - start
    pct = 100 * wins / args.games
    _, lo, hi = wilson_ci(wins, args.games)

    print(f"\n\nP1 win rate: {wins}/{args.games} = {pct:.1f}% [{lo*100:.0f}%, {hi*100:.0f}%]")
    print(f"Time: {elapsed:.0f}s ({elapsed/args.games:.1f}s/game)")


if __name__ == "__main__":
    main()
