"""CLI entry point for parsing colonist.io game JSONs into Parquet training data.

Usage:
    python3 -m datasets.parse_games [--limit N] [--workers N] [--shard-size N] [--output-dir DIR]
"""
import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from datasets.game_parser import parse_game

# Suppress noisy per-game logging from coordinate_map
logging.getLogger("bridge.coordinate_map").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMES_DIR = Path(__file__).parent / "games"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "parquet"


def _safe_parse_game(filepath: str):
    """Wrapper that catches per-game exceptions."""
    try:
        return parse_game(filepath)
    except Exception as e:
        logger.debug(f"Failed to parse {filepath}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Parse colonist.io games to Parquet training data")
    parser.add_argument("--limit", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--workers", type=int, default=0, help="Num workers (0=cpu_count)")
    parser.add_argument("--shard-size", type=int, default=100, help="Games per Parquet shard")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--games-dir", type=str, default=str(GAMES_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games_dir = Path(args.games_dir)
    game_files = sorted(games_dir.glob("*.json"))
    if args.limit > 0:
        game_files = game_files[: args.limit]

    total_games = len(game_files)
    workers = args.workers or max(1, cpu_count() - 1)
    logger.info(
        f"Processing {total_games} games with {workers} workers, "
        f"shard size {args.shard_size}"
    )

    t0 = time.time()
    game_paths = [str(f) for f in game_files]

    total_samples = 0
    total_parsed = 0
    total_errors = 0
    total_skipped = 0
    shard_idx = 0
    shard_rows = []
    games_in_shard = 0

    def flush_shard():
        nonlocal shard_idx, shard_rows, games_in_shard
        if not shard_rows:
            return
        df = pd.DataFrame(shard_rows)
        out_path = output_dir / f"shard_{shard_idx:05d}.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        logger.info(
            f"Wrote shard {shard_idx}: {len(shard_rows)} samples "
            f"from {games_in_shard} games -> {out_path}"
        )
        shard_idx += 1
        shard_rows = []
        games_in_shard = 0

    pool = Pool(processes=workers)
    try:
        pbar = tqdm(
            pool.imap_unordered(_safe_parse_game, game_paths, chunksize=10),
            total=total_games,
            unit="game",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {postfix}",
        )
        for result in pbar:
            if result is None:
                total_errors += 1
            elif len(result) == 0:
                total_skipped += 1
            else:
                total_parsed += 1
                total_samples += len(result)
                shard_rows.extend(result)
                games_in_shard += 1

                if games_in_shard >= args.shard_size:
                    flush_shard()

            pbar.set_postfix_str(f"{total_samples:,}s {total_parsed}ok {total_skipped}skip {total_errors}err")
    finally:
        # Flush remaining data BEFORE pool cleanup (which can hang on macOS)
        flush_shard()
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0
    metadata = {
        "total_games": total_games,
        "games_parsed": total_parsed,
        "games_skipped": total_skipped,
        "games_errored": total_errors,
        "total_samples": total_samples,
        "num_shards": shard_idx,
        "elapsed_seconds": round(elapsed, 1),
        "games_per_second": round(total_parsed / elapsed, 1) if elapsed > 0 else 0,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Done: {total_parsed} games -> {total_samples} samples "
        f"in {shard_idx} shards ({elapsed:.1f}s)"
    )
    logger.info(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
