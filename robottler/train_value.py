"""Train a neural network value function on parsed Catan game data.

Streams parquet shards from disk one at a time to avoid OOM on
memory-constrained machines (e.g. 8 GB M1 MacBook).

Usage:
    python3 -m robottler.train_value [--data-dir datasets/parquet] [--epochs 50]
                                      [--batch-size 512] [--lr 1e-3]
                                      [--max-shards 30] [--resume]
"""

import argparse
import csv
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from robottler.value_model import CatanValueNet


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

def _subsample_indices(game_ids, game_progress, samples_per_game):
    """Fast phase-stratified subsampling using numpy (no pandas groupby).
    Returns row indices to keep."""
    # Allocate budget across phases: early 30%, mid 40%, late 30%
    n_early = max(1, round(samples_per_game * 0.3))
    n_mid = max(1, round(samples_per_game * 0.4))
    n_late = samples_per_game - n_early - n_mid

    # Find game boundaries via sorted game_ids
    order = np.argsort(game_ids, kind="stable")
    sorted_gids = game_ids[order]
    # Indices where a new game starts
    breaks = np.concatenate([[0], np.where(sorted_gids[1:] != sorted_gids[:-1])[0] + 1,
                             [len(sorted_gids)]])

    keep = []
    for i in range(len(breaks) - 1):
        grp = order[breaks[i]:breaks[i + 1]]  # original row indices for this game
        gp = game_progress[grp]
        for lo, hi, n in [(0, 0.33, n_early), (0.33, 0.66, n_mid), (0.66, 1.01, n_late)]:
            bucket = grp[(gp >= lo) & (gp < hi)]
            if len(bucket) > 0:
                chosen = np.random.choice(bucket, size=min(n, len(bucket)), replace=False)
                keep.append(chosen)
    return np.concatenate(keep) if keep else np.array([], dtype=np.intp)


class ShardDataset(IterableDataset):
    """Yields pre-batched (X, y) tensors from parquet shards, one shard at a
    time.  Reads only needed columns from parquet for speed."""

    def __init__(self, shard_files, game_set, f_cols, means, stds,
                 batch_size=512, shuffle=True, samples_per_game=None):
        self.shard_files = list(shard_files)
        self.game_set = game_set
        self.f_cols = f_cols
        # Columns to actually read from parquet
        self._read_cols = list(f_cols) + ["game_id", "winner"]
        if samples_per_game is not None:
            self._read_cols.append("game_progress")
        self.means = means   # float32 numpy array (n_features,)
        self.stds = stds     # float32 numpy array (n_features,)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples_per_game = samples_per_game  # None = use all samples

    def __iter__(self):
        files = list(self.shard_files)
        if self.shuffle:
            random.shuffle(files)
        for path in files:
            df = pd.read_parquet(path, columns=self._read_cols)
            mask = df["game_id"].isin(self.game_set)
            df = df[mask]
            if len(df) == 0:
                continue
            if self.samples_per_game is not None:
                keep = _subsample_indices(
                    df["game_id"].values, df["game_progress"].values,
                    self.samples_per_game)
                df = df.iloc[keep]
            X = df[self.f_cols].values.astype(np.float32)
            y = df["winner"].values.astype(np.float32)
            X = (X - self.means) / self.stds
            if self.shuffle:
                perm = np.random.permutation(len(X))
                X = X[perm]
                y = y[perm]
            # Yield pre-batched tensors
            for start in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[start:start + self.batch_size].copy())
                yb = torch.from_numpy(y[start:start + self.batch_size].copy())
                yield xb, yb


# ---------------------------------------------------------------------------
# Preparation (two lightweight passes over shards)
# ---------------------------------------------------------------------------

POSITIONAL_PREFIXES = ("NODE", "EDGE", "TILE", "PORT")


def _filter_feature_cols(f_cols, feature_filter):
    """Filter F_ columns based on feature_filter mode."""
    if feature_filter is None:
        return f_cols
    if feature_filter == "strategic":
        # Drop positional features that fingerprint board layouts
        return [c for c in f_cols
                if not any(c[2:].startswith(p) for p in POSITIONAL_PREFIXES)]
    raise ValueError(f"Unknown feature_filter: {feature_filter}")


def prepare(data_dir, max_shards=None, feature_filter=None):
    """Scan shards to collect game IDs, split them, and compute train-set
    normalization stats.  Only one shard is in memory at a time.

    data_dir can be a single path (str) or a list of paths for mixed training
    (e.g. self-play + human data).
    """
    if isinstance(data_dir, str):
        data_dir = [data_dir]

    files = []
    for d in data_dir:
        found = sorted(glob.glob(os.path.join(d, "shard_*.parquet")))
        if not found:
            found = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not found:
            print(f"Warning: no parquet files in {d}")
        files.extend(found)
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    if max_shards:
        files = files[:max_shards]
    print(f"Using {len(files)} shard(s) from {len(data_dir)} director{'y' if len(data_dir) == 1 else 'ies'}")

    # Feature columns — when mixing data from different player counts (e.g.
    # 2-player self-play + 4-player human data), intersect F_ columns so all
    # shards share the same feature set.
    f_col_sets = []
    for d in data_dir:
        d_shards = sorted(glob.glob(os.path.join(d, "shard_*.parquet")))
        if not d_shards:
            d_shards = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if d_shards:
            cols = pd.read_parquet(d_shards[0]).columns
            f_col_sets.append(set(c for c in cols if c.startswith("F_")))
    if not f_col_sets:
        raise FileNotFoundError(f"Could not read any parquet files from {data_dir}")
    f_cols = sorted(set.intersection(*f_col_sets))
    if feature_filter:
        before = len(f_cols)
        f_cols = _filter_feature_cols(f_cols, feature_filter)
        print(f"Features: {len(f_cols)} (filtered from {before}, mode={feature_filter})")
    else:
        print(f"Features: {len(f_cols)}")
    if len(f_col_sets) > 1:
        print(f"  (intersected {[len(s) for s in f_col_sets]} F_ columns → {len(f_cols)} common)")

    # --- Pass 1: collect unique game IDs (reads only one column) ----------
    print("Pass 1: Collecting game IDs...")
    all_game_ids = set()
    for i, f in enumerate(files):
        gids = pd.read_parquet(f, columns=["game_id"])["game_id"].unique()
        all_game_ids.update(gids)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} shards scanned")
    print(f"  {len(all_game_ids)} unique games")

    # Split games deterministically
    game_arr = np.array(sorted(all_game_ids))
    rng = np.random.RandomState(42)
    rng.shuffle(game_arr)
    n = len(game_arr)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    train_games = set(game_arr[:n_train])
    val_games = set(game_arr[n_train:n_val])
    test_games = set(game_arr[n_val:])
    print(f"  Split: {len(train_games)} train / {len(val_games)} val / {len(test_games)} test games")

    # --- Pass 2: compute mean/std on training features --------------------
    print("Pass 2: Computing normalization stats...")
    stats_cols = list(f_cols) + ["game_id", "winner"]
    running_sum = np.zeros(len(f_cols), dtype=np.float64)
    running_sq = np.zeros(len(f_cols), dtype=np.float64)
    n_samples = 0
    n_pos = 0
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=stats_cols)
        mask = df["game_id"].isin(train_games)
        df = df[mask]
        if len(df) == 0:
            continue
        X = df[f_cols].values.astype(np.float64)
        running_sum += X.sum(axis=0)
        running_sq += (X ** 2).sum(axis=0)
        n_samples += len(X)
        n_pos += df["winner"].values.sum()
        del df, X
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} shards ({n_samples} train samples)")

    means_f64 = running_sum / n_samples
    var_f64 = running_sq / n_samples - means_f64 ** 2
    stds_f64 = np.sqrt(np.maximum(var_f64, 0))

    means = means_f64.astype(np.float32)
    stds = stds_f64.astype(np.float32)
    stds[stds == 0] = 1.0
    pos_rate = n_pos / n_samples
    print(f"  {n_samples} train samples, positive rate: {pos_rate:.3f}")

    feature_names = [c[2:] for c in f_cols]  # strip F_ prefix
    return {
        "files": files,
        "f_cols": f_cols,
        "feature_names": feature_names,
        "train_games": train_games,
        "val_games": val_games,
        "test_games": test_games,
        "means": means,
        "stds": stds,
        "pos_rate": pos_rate,
        "n_train": n_samples,
    }


# ---------------------------------------------------------------------------
# Streaming evaluation
# ---------------------------------------------------------------------------

def evaluate_streaming(model, shard_files, game_set, f_cols, means, stds,
                       criterion, device):
    """Run inference one shard at a time, accumulate scalar predictions."""
    eval_cols = list(f_cols) + ["game_id", "winner"]
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for path in shard_files:
            df = pd.read_parquet(path, columns=eval_cols)
            mask = df["game_id"].isin(game_set)
            df = df[mask]
            if len(df) == 0:
                continue
            X = df[f_cols].values.astype(np.float32)
            y = df["winner"].values.astype(np.float32)
            X = (X - means) / stds
            xb = torch.tensor(X, device=device)
            yb = torch.tensor(y, device=device)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            n += len(xb)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y)
            del df, X, xb, yb

    avg_loss = total_loss / n if n > 0 else 0
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = ((all_preds > 0.5) == all_labels).mean()
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, acc, auc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    info = prepare(args.data_dir, args.max_shards, args.feature_filter)

    if args.no_pos_weight:
        pos_weight = None
        print("pos_weight: disabled")
    else:
        pos_weight = (1 - info["pos_rate"]) / info["pos_rate"]
        print(f"pos_weight: {pos_weight:.2f}")

    # Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    # Streaming DataLoader (ShardDataset yields pre-batched tensors)
    train_ds = ShardDataset(
        info["files"], info["train_games"], info["f_cols"],
        info["means"], info["stds"], batch_size=args.batch_size, shuffle=True,
        samples_per_game=args.samples_per_game,
    )
    train_loader = DataLoader(train_ds, batch_size=None)

    # Model (create on CPU first so resume loading is clean)
    model = CatanValueNet(input_dim=len(info["feature_names"]))
    pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_val_auc = 0.0
    patience_counter = 0
    best_state = None
    start_epoch = 0

    # Resume from training checkpoint if requested
    resume_path = args.output + ".training"
    if args.resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        best_val_auc = ckpt["best_val_auc"]
        patience_counter = ckpt["patience_counter"]
        best_state = ckpt["best_state"]
        print(f"  Continuing from epoch {start_epoch + 1}, "
              f"best_val_loss={best_val_loss:.4f}, patience={patience_counter}")
        del ckpt
    elif args.resume:
        print(f"No training checkpoint at {resume_path}, starting fresh")

    # Move model + optimizer state to device
    model.to(device)
    criterion.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Metrics CSV log
    log_path = args.output.replace(".pt", "_log.csv")
    log_fields = ["epoch", "train_loss", "val_loss", "val_acc", "val_auc", "lr", "patience"]
    if start_epoch == 0:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        log_writer.writeheader()
    else:
        log_file = open(log_path, "a", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)

    for epoch in range(start_epoch, args.epochs):
        # --- Train (streaming) ---
        model.train()
        train_loss = 0.0
        n_seen = 0
        if args.samples_per_game:
            # ~samples_per_game samples per game, ~70% of games are train
            n_samples_est = len(info["train_games"]) * args.samples_per_game
        else:
            n_samples_est = info["n_train"]
        n_batches_est = n_samples_est // args.batch_size
        pbar = tqdm(train_loader, total=n_batches_est,
                    desc=f"Epoch {epoch+1:3d}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_seen += len(xb)
            pbar.set_postfix(loss=f"{train_loss / n_seen:.4f}")
        pbar.close()
        train_loss /= n_seen

        # --- Validate (streaming) ---
        val_loss, val_acc, val_auc = evaluate_streaming(
            model, info["files"], info["val_games"], info["f_cols"],
            info["means"], info["stds"], criterion, device,
        )

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.3f} val_auc={val_auc:.3f} | lr={lr:.1e}")

        log_writer.writerow({
            "epoch": epoch + 1, "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}", "val_acc": f"{val_acc:.4f}",
            "val_auc": f"{val_auc:.4f}", "lr": f"{lr:.1e}",
            "patience": patience_counter,
        })
        log_file.flush()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save training checkpoint for resume
        torch.save({
            "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_auc": best_val_auc,
            "patience_counter": patience_counter,
            "best_state": best_state,
        }, resume_path)

    log_file.close()
    print(f"Metrics log: {log_path}")

    # --- Test evaluation (streaming) ---
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_acc, test_auc = evaluate_streaming(
        model, info["files"], info["test_games"], info["f_cols"],
        info["means"], info["stds"], criterion, device,
    )
    print(f"\nTest set: loss={test_loss:.4f} acc={test_acc:.3f} auc={test_auc:.3f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "feature_names": info["feature_names"],
        "feature_means": info["means"],
        "feature_stds": info["stds"],
        "val_loss": best_val_loss,
        "val_auc": best_val_auc,
        "test_loss": test_loss,
        "test_auc": test_auc,
    }
    torch.save(checkpoint, args.output)
    print(f"Saved model to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train Catan value network")
    parser.add_argument("--data-dir", nargs="+", default=["datasets/parquet"],
                        help="One or more directories containing parquet shards")
    parser.add_argument("--output", default="robottler/models/value_net.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Limit number of shards loaded (for quick testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last training checkpoint")
    parser.add_argument("--feature-filter", default=None, choices=["strategic"],
                        help="'strategic' = drop NODE/EDGE/TILE/PORT positional features")
    parser.add_argument("--samples-per-game", type=int, default=None,
                        help="Phase-stratified subsampling: N samples per game per epoch (e.g. 8)")
    parser.add_argument("--no-pos-weight", action="store_true",
                        help="Disable class imbalance weighting (use raw BCE)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
