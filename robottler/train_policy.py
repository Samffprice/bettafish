"""Train a BC policy network that predicts human actions from game states.

Preloads all data into RAM (~1.5 GB) once, then trains from memory.

Usage:
    python3 -m robottler.train_policy --epochs 200 --output robottler/models/policy_net_v1.pt
    python3 -m robottler.train_policy --epochs 200 --resume  # continue from checkpoint
"""

import argparse
import csv
import glob
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CatanPolicyNet(nn.Module):
    """MLP: strategic features -> action logits over Discrete(290)."""

    def __init__(self, input_dim: int = 176, num_actions: int = ACTION_SPACE_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# In-memory data loading
# ---------------------------------------------------------------------------

POSITIONAL_PREFIXES = ("NODE", "EDGE", "TILE", "PORT")


def _subsample_indices(game_ids, game_progress, samples_per_game):
    """Phase-stratified subsampling: early 30%, mid 40%, late 30%."""
    n_early = max(1, round(samples_per_game * 0.3))
    n_mid = max(1, round(samples_per_game * 0.4))
    n_late = samples_per_game - n_early - n_mid

    order = np.argsort(game_ids, kind="stable")
    sorted_gids = game_ids[order]
    breaks = np.concatenate([[0], np.where(sorted_gids[1:] != sorted_gids[:-1])[0] + 1,
                             [len(sorted_gids)]])

    keep = []
    for i in range(len(breaks) - 1):
        grp = order[breaks[i]:breaks[i + 1]]
        gp = game_progress[grp]
        for lo, hi, n in [(0, 0.33, n_early), (0.33, 0.66, n_mid), (0.66, 1.01, n_late)]:
            bucket = grp[(gp >= lo) & (gp < hi)]
            if len(bucket) > 0:
                chosen = np.random.choice(bucket, size=min(n, len(bucket)), replace=False)
                keep.append(chosen)
    return np.concatenate(keep) if keep else np.array([], dtype=np.intp)


def _load_one_shard(path, game_set, f_cols, read_cols, actions_df):
    """Load and filter one parquet shard. Returns tuple or None."""
    df = pd.read_parquet(path, columns=read_cols)
    df = df[df["game_id"].isin(game_set)]
    if len(df) == 0:
        return None
    df = df[df["perspective_color"] == df["acting_player_color"]]
    if len(df) == 0:
        return None
    df = df.merge(actions_df, on=["game_id", "event_index"], how="inner")
    df = df[df["action_index"] >= 0]
    if len(df) == 0:
        return None
    return (
        df[f_cols].values.astype(np.float32),
        df["action_index"].values.astype(np.int64),
        df["game_id"].values,
        df["game_progress"].values.astype(np.float32),
    )


def _load_split_parallel(shard_files, game_set, f_cols, actions_df, n_workers=8):
    """Load all samples for a split using parallel I/O."""
    read_cols = list(f_cols) + [
        "game_id", "event_index", "perspective_color",
        "acting_player_color", "game_progress",
    ]
    chunks = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_load_one_shard, p, game_set, f_cols, read_cols, actions_df): p
            for p in shard_files
        }
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="  Loading shards", leave=False)
        for future in pbar:
            result = future.result()
            if result is not None:
                chunks.append(result)
        pbar.close()

    if not chunks:
        n_feat = len(f_cols)
        return (np.empty((0, n_feat), np.float32), np.empty(0, np.int64),
                np.empty(0, object), np.empty(0, np.float32))

    return (np.concatenate([c[0] for c in chunks]),
            np.concatenate([c[1] for c in chunks]),
            np.concatenate([c[2] for c in chunks]),
            np.concatenate([c[3] for c in chunks]))


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def _cache_key(files, actions_path, max_shards):
    """Deterministic cache key based on shard list and actions file."""
    h = hashlib.md5()
    h.update(str(len(files)).encode())
    h.update(os.path.basename(files[0]).encode())
    h.update(os.path.basename(files[-1]).encode())
    h.update(str(os.path.getmtime(actions_path)).encode())
    h.update(str(max_shards or "all").encode())
    return h.hexdigest()[:12]


def _save_cache(cache_dir, **arrays):
    """Save numpy arrays as individual .npy files for fast loading."""
    os.makedirs(cache_dir, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(cache_dir, f"{name}.npy"), arr)


def _load_cache(cache_dir, keys):
    """Load numpy arrays from cache. Returns dict or None if missing."""
    result = {}
    for key in keys:
        path = os.path.join(cache_dir, f"{key}.npy")
        if not os.path.exists(path):
            return None
        result[key] = np.load(path, allow_pickle=True)
    return result


def prepare(data_dir, actions_path, max_shards=None, n_workers=8):
    """Scan shards, load actions, preload all data into RAM (cached)."""
    t0 = time.time()

    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    files = [f for f in files if "actions" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No feature parquet files in {data_dir}")
    if max_shards:
        files = files[:max_shards]
    print(f"Using {len(files)} feature shard(s)")

    # Feature columns (strategic filter)
    first = pd.read_parquet(files[0], columns=None)
    all_f_cols = sorted([c for c in first.columns if c.startswith("F_")])
    f_cols = [c for c in all_f_cols
              if not any(c[2:].startswith(p) for p in POSITIONAL_PREFIXES)]
    print(f"Features: {len(f_cols)} strategic (from {len(all_f_cols)} total)")
    feature_names = [c[2:] for c in f_cols]
    del first

    # Check cache
    key = _cache_key(files, actions_path, max_shards)
    cache_dir = os.path.join(data_dir, ".policy_cache", key)
    cache_keys = [
        "train_X", "train_y", "train_gid_codes", "train_gp",
        "val_X", "val_y", "test_X", "test_y", "means", "stds",
    ]
    cached = _load_cache(cache_dir, cache_keys)
    if cached is not None:
        elapsed = time.time() - t0
        print(f"Loaded from cache {cache_dir} in {elapsed:.1f}s")
        mem_mb = (cached["train_X"].nbytes + cached["val_X"].nbytes +
                  cached["test_X"].nbytes) / 1e6
        print(f"  {len(cached['train_X'])} train / {len(cached['val_X'])} val / "
              f"{len(cached['test_X'])} test ({mem_mb:.0f} MB)")
        return {
            "f_cols": f_cols,
            "feature_names": feature_names,
            "means": cached["means"],
            "stds": cached["stds"],
            "train_X": cached["train_X"], "train_y": cached["train_y"],
            "train_gid": cached["train_gid_codes"], "train_gp": cached["train_gp"],
            "val_X": cached["val_X"], "val_y": cached["val_y"],
            "test_X": cached["test_X"], "test_y": cached["test_y"],
            "n_train": len(cached["train_X"]),
        }

    # Load actions
    print(f"Loading actions from {actions_path}...")
    actions_df = pd.read_parquet(actions_path)
    valid_actions = actions_df[actions_df["action_index"] >= 0]
    print(f"  Total action labels: {len(actions_df)}, valid: {len(valid_actions)} "
          f"({len(valid_actions)/len(actions_df)*100:.1f}%)")
    actions_df = actions_df[["game_id", "event_index", "action_index"]]

    # Collect game IDs
    print("Collecting game IDs...")
    all_game_ids = set()
    for f in files:
        gids = pd.read_parquet(f, columns=["game_id"])["game_id"].unique()
        all_game_ids.update(gids)
    action_games = set(actions_df["game_id"].unique())
    all_game_ids = all_game_ids & action_games
    print(f"  {len(all_game_ids)} games with both features and actions")

    # Deterministic split
    game_arr = np.array(sorted(all_game_ids))
    rng = np.random.RandomState(42)
    rng.shuffle(game_arr)
    n = len(game_arr)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    train_games = set(game_arr[:n_train])
    val_games = set(game_arr[n_train:n_val])
    test_games = set(game_arr[n_val:])
    print(f"  Split: {len(train_games)} train / {len(val_games)} val / {len(test_games)} test")

    # Preload all splits with parallel I/O
    print(f"Preloading train data ({n_workers} threads)...")
    train_X, train_y, train_gid, train_gp = _load_split_parallel(
        files, train_games, f_cols, actions_df, n_workers)
    print(f"  {len(train_X)} train samples")

    print(f"Preloading val data ({n_workers} threads)...")
    val_X, val_y, _, _ = _load_split_parallel(
        files, val_games, f_cols, actions_df, n_workers)
    print(f"  {len(val_X)} val samples")

    print(f"Preloading test data ({n_workers} threads)...")
    test_X, test_y, _, _ = _load_split_parallel(
        files, test_games, f_cols, actions_df, n_workers)
    print(f"  {len(test_X)} test samples")

    # Normalization stats (train set)
    print("Computing normalization stats...")
    means = train_X.mean(axis=0)
    stds = train_X.std(axis=0)
    stds[stds == 0] = 1.0

    # Normalize
    train_X = (train_X - means) / stds
    val_X = (val_X - means) / stds
    test_X = (test_X - means) / stds

    # Factorize game_ids to ints for cache (subsampling only needs group identity)
    train_gid_codes, _ = pd.factorize(train_gid)
    train_gid_codes = train_gid_codes.astype(np.int32)

    # Save cache
    print(f"Saving cache to {cache_dir}...")
    _save_cache(
        cache_dir,
        train_X=train_X, train_y=train_y,
        train_gid_codes=train_gid_codes, train_gp=train_gp,
        val_X=val_X, val_y=val_y,
        test_X=test_X, test_y=test_y,
        means=means, stds=stds,
    )

    mem_mb = (train_X.nbytes + val_X.nbytes + test_X.nbytes) / 1e6
    elapsed = time.time() - t0
    print(f"  Data loaded in {elapsed:.0f}s ({mem_mb:.0f} MB in RAM)")

    return {
        "f_cols": f_cols,
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
        "train_X": train_X, "train_y": train_y,
        "train_gid": train_gid_codes, "train_gp": train_gp,
        "val_X": val_X, "val_y": val_y,
        "test_X": test_X, "test_y": test_y,
        "n_train": len(train_X),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, criterion, device, batch_size=16384):
    """Evaluate on preloaded data, return metrics."""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    n = len(X)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(X[start:end]).to(device)
            yb = torch.from_numpy(y[start:end]).to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(xb)

            _, top1 = logits.topk(1, dim=1)
            correct_top1 += (top1.squeeze(-1) == yb).sum().item()
            _, top3 = logits.topk(3, dim=1)
            correct_top3 += (top3 == yb.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = total_loss / n if n > 0 else 0
    top1_acc = correct_top1 / n if n > 0 else 0
    top3_acc = correct_top3 / n if n > 0 else 0
    return avg_loss, top1_acc, top3_acc, n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    info = prepare(args.data_dir, args.actions_path, args.max_shards,
                   n_workers=args.workers)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    train_X = info["train_X"]
    train_y = info["train_y"]
    train_gid = info["train_gid"]
    train_gp = info["train_gp"]

    model = CatanPolicyNet(input_dim=len(info["feature_names"]))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    start_epoch = 0

    # Resume
    resume_path = args.output + ".training"
    if args.resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt["patience_counter"]
        best_state = ckpt["best_state"]
        print(f"  Continuing from epoch {start_epoch + 1}, "
              f"best_val_loss={best_val_loss:.4f}")
        del ckpt

    model.to(device)
    criterion.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Metrics log
    log_path = args.output.replace(".pt", "_log.csv")
    log_fields = ["epoch", "train_loss", "val_loss", "val_top1", "val_top3", "lr", "patience"]
    if start_epoch == 0:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        log_writer.writeheader()
    else:
        log_file = open(log_path, "a", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()

        # Phase-stratified subsampling each epoch (different samples each time)
        if args.samples_per_game is not None:
            keep = _subsample_indices(train_gid, train_gp, args.samples_per_game)
            epoch_X = train_X[keep]
            epoch_y = train_y[keep]
        else:
            epoch_X = train_X
            epoch_y = train_y

        # Shuffle
        perm = np.random.permutation(len(epoch_X))
        epoch_X = epoch_X[perm]
        epoch_y = epoch_y[perm]

        train_loss = 0.0
        n_seen = 0
        n_batches = (len(epoch_X) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1:3d}", leave=False)
        for bi in pbar:
            start = bi * args.batch_size
            end = min(start + args.batch_size, len(epoch_X))
            xb = torch.from_numpy(epoch_X[start:end]).to(device)
            yb = torch.from_numpy(epoch_y[start:end]).to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_seen += len(xb)
            pbar.set_postfix(loss=f"{train_loss / n_seen:.4f}")
        pbar.close()
        train_loss /= max(n_seen, 1)

        # Validate
        val_loss, val_top1, val_top3, val_n = evaluate(
            model, info["val_X"], info["val_y"], criterion, device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train={train_loss:.4f} val={val_loss:.4f} | "
              f"top1={val_top1:.3f} top3={val_top3:.3f} | "
              f"lr={lr:.1e} | {epoch_time:.1f}s ({n_seen} samples)")

        log_writer.writerow({
            "epoch": epoch + 1, "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}", "val_top1": f"{val_top1:.4f}",
            "val_top3": f"{val_top3:.4f}", "lr": f"{lr:.1e}",
            "patience": patience_counter,
        })
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        torch.save({
            "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "best_state": best_state,
        }, resume_path)

    log_file.close()
    print(f"Metrics log: {log_path}")

    # Test
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_top1, test_top3, test_n = evaluate(
        model, info["test_X"], info["test_y"], criterion, device)
    print(f"\nTest: loss={test_loss:.4f} top1={test_top1:.3f} top3={test_top3:.3f} "
          f"({test_n} samples)")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "feature_names": info["feature_names"],
        "feature_means": info["means"],
        "feature_stds": info["stds"],
        "num_actions": ACTION_SPACE_SIZE,
        "val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_top1": test_top1,
        "test_top3": test_top3,
    }
    torch.save(checkpoint, args.output)
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train BC policy network")
    parser.add_argument("--data-dir", default="datasets/parquet")
    parser.add_argument("--actions-path", default="datasets/parquet/actions.parquet")
    parser.add_argument("--output", default="robottler/models/policy_net_v1.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--samples-per-game", type=int, default=8,
                        help="Phase-stratified samples per game per epoch (default: 8)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Threads for parallel shard loading (default: 8)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
