"""Generate training data from search-vs-search self-play games.

Records (features, outcome) pairs for value function retraining.
Uses multiprocessing for parallel game generation.

Usage:
    # Quick smoke test (10 games)
    python3 -m robottler.self_play \
        --bc-model robottler/models/value_net_v2.pt \
        --games 10 --depth 2 --workers 1 \
        --output-dir datasets/selfplay_test

    # Full run (1k games, 8 cores)
    python3 -m robottler.self_play \
        --bc-model robottler/models/value_net_v2.pt \
        --games 1000 --depth 2 --workers 8 \
        --output-dir datasets/selfplay
"""

import argparse
import multiprocessing as mp
import os
import sys
import random
import time
import uuid

if '--workers' in sys.argv:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

from catanatron.features import create_sample
from catanatron.game import Game, GameAccumulator, TURNS_LIMIT
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points


class ValueDataAccumulator(GameAccumulator):
    """Records (features, outcome) pairs for value function training."""

    def __init__(self, game_id, sample_rate=0.3):
        self.game_id = game_id
        self.sample_rate = sample_rate
        self.records = []

    def before(self, game):
        self._step_count = 0

    def step(self, game_before_action, action):
        self._step_count += 1
        if random.random() > self.sample_rate:
            return
        for color in game_before_action.state.colors:
            sample = create_sample(game_before_action, color)
            self.records.append({
                "game_id": self.game_id,
                "color": str(color),
                "step": self._step_count,
                **{f"F_{k}": v for k, v in sample.items()},
            })

    def after(self, game):
        winner = game.winning_color()
        total_steps = max(self._step_count, 1)
        for rec in self.records:
            rec["winner"] = 1.0 if str(winner) == rec["color"] else 0.0
            rec["game_progress"] = rec["step"] / total_steps


def play_one_game(value_fn, depth, prunning, game_id, vps_to_win=10):
    """Play a single self-play game and return accumulator records."""
    from robottler.search_player import NeuralSearchPlayer

    p1 = NeuralSearchPlayer(Color.RED, value_fn=value_fn, depth=depth, prunning=prunning)
    p2 = NeuralSearchPlayer(Color.BLUE, value_fn=value_fn, depth=depth, prunning=prunning)

    acc = ValueDataAccumulator(game_id)
    game = Game([p1, p2], vps_to_win=vps_to_win)
    game.play(accumulators=[acc])

    winner = game.winning_color()
    turns = game.state.num_turns
    vp_red = get_actual_victory_points(game.state, Color.RED)
    vp_blue = get_actual_victory_points(game.state, Color.BLUE)

    return acc.records, {
        "game_id": game_id,
        "winner": str(winner),
        "turns": turns,
        "vp_red": vp_red,
        "vp_blue": vp_blue,
        "draw": winner is None or turns >= TURNS_LIMIT,
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker for parallel self-play
# ---------------------------------------------------------------------------

_sp_worker_value_fn = None
_sp_worker_depth = None
_sp_worker_prunning = None
_sp_worker_vps = None


def _sp_worker_init(bc_model_path, blend_weight, eval_mode, depth, prunning, vps_to_win):
    """Initialize per-worker state: load value function."""
    global _sp_worker_value_fn, _sp_worker_depth, _sp_worker_prunning, _sp_worker_vps
    from robottler.search_player import make_blended_value_fn
    from robottler.value_model import load_value_model

    if eval_mode == "blend":
        _sp_worker_value_fn = make_blended_value_fn(bc_model_path, blend_weight=blend_weight)
    else:
        _sp_worker_value_fn = load_value_model(bc_model_path)

    _sp_worker_depth = depth
    _sp_worker_prunning = prunning
    _sp_worker_vps = vps_to_win


def _sp_worker_play(game_id):
    """Play one game in a worker process."""
    return play_one_game(_sp_worker_value_fn, _sp_worker_depth,
                         _sp_worker_prunning, game_id, _sp_worker_vps)


def generate_games(bc_model_path, num_games, depth, workers, output_dir,
                   prunning=False, vps_to_win=10, blend_weight=1e8,
                   eval_mode="blend", shard_size=500):
    """Generate self-play games and write parquet shards."""
    from robottler.search_player import make_blended_value_fn
    from robottler.value_model import load_value_model

    os.makedirs(output_dir, exist_ok=True)

    game_ids = [f"sp_{uuid.uuid4().hex[:12]}" for _ in range(num_games)]

    all_records = []
    game_stats = []
    shard_idx = 0

    def flush_shard(records):
        nonlocal shard_idx
        if not records:
            return
        df = pd.DataFrame(records)
        path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
        df.to_parquet(path, index=False)
        shard_idx += 1
        print(f"  Wrote {path} ({len(df)} rows)")

    start = time.time()
    print(f"Generating {num_games} self-play games (depth={depth}, workers={workers})...")

    if workers <= 1:
        if eval_mode == "blend":
            print(f"Loading blend value fn (weight={blend_weight:.0e}) from {bc_model_path}...")
            value_fn = make_blended_value_fn(bc_model_path, blend_weight=blend_weight)
        else:
            print(f"Loading neural value fn from {bc_model_path}...")
            value_fn = load_value_model(bc_model_path)

        for i, gid in enumerate(tqdm(game_ids, desc="Self-play")):
            records, stats = play_one_game(value_fn, depth, prunning, gid, vps_to_win)
            all_records.extend(records)
            game_stats.append(stats)

            if len(all_records) >= shard_size * 100:
                flush_shard(all_records)
                all_records = []
    else:
        print(f"Using {workers} parallel workers")
        with mp.Pool(
            processes=workers,
            initializer=_sp_worker_init,
            initargs=(bc_model_path, blend_weight, eval_mode, depth, prunning, vps_to_win),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(_sp_worker_play, game_ids),
                total=num_games, desc="Self-play",
            ))

        for records, stats in results:
            all_records.extend(records)
            game_stats.append(stats)

            if len(all_records) >= shard_size * 100:
                flush_shard(all_records)
                all_records = []

    flush_shard(all_records)

    elapsed = time.time() - start
    n_draws = sum(1 for s in game_stats if s["draw"])
    n_decided = num_games - n_draws
    avg_turns = np.mean([s["turns"] for s in game_stats]) if game_stats else 0

    print(f"\nCompleted {num_games} games in {elapsed:.0f}s ({elapsed/num_games:.1f}s/game)")
    print(f"  Decided: {n_decided}/{num_games} ({n_decided/num_games*100:.0f}%)")
    print(f"  Avg turns: {avg_turns:.0f}")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Self-play data generation")
    parser.add_argument("--bc-model", required=True,
                        help="Path to BC value net checkpoint (.pt)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of self-play games (default: 100)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Search depth for both players (default: 2)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Thread pool size (default: 1)")
    parser.add_argument("--output-dir", default="datasets/selfplay",
                        help="Output directory for parquet shards")
    parser.add_argument("--blend-weight", type=float, default=1e8,
                        help="Blend weight for heuristic+neural (default: 1e8)")
    parser.add_argument("--eval-mode", default="blend", choices=["blend", "neural"],
                        help="Value function type (default: blend)")
    parser.add_argument("--vps", type=int, default=10,
                        help="Victory points to win (default: 10)")
    parser.add_argument("--prunning", action="store_true",
                        help="Enable search pruning (default: disabled, which is stronger)")
    args = parser.parse_args()

    if args.workers > 1:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if has_cuda:
            mp.set_start_method("spawn", force=True)
        else:
            mp.set_start_method("fork", force=True)

    generate_games(
        bc_model_path=args.bc_model,
        num_games=args.games,
        depth=args.depth,
        workers=args.workers,
        output_dir=args.output_dir,
        prunning=args.prunning,
        vps_to_win=args.vps,
        blend_weight=args.blend_weight,
        eval_mode=args.eval_mode,
    )


if __name__ == "__main__":
    main()
