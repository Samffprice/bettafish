# Neural Value Function Pipeline

End-to-end guide: parse game logs → train neural network → benchmark against hand-tuned bot.

## Prerequisites

```bash
pip3 install --break-system-packages torch scikit-learn pandas pyarrow tqdm
```

## Step 1: Download Game Logs

Place colonist.io game JSON files in `datasets/games/`. Each file should be one complete game log.

```
datasets/games/
├── game_abc123.json
├── game_def456.json
└── ...
```

## Step 2: Parse Games to Parquet

```bash
python3 -m datasets.parse_games --limit 1000
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | 0 (all) | Max games to process |
| `--workers N` | cpu_count - 1 | Parallel workers |
| `--shard-size N` | 100 | Games per parquet shard |
| `--output-dir` | `datasets/parquet/` | Output directory |
| `--games-dir` | `datasets/games/` | Input game JSONs |

**Output:**
- `datasets/parquet/shard_00000.parquet`, `shard_00001.parquet`, ...
- `datasets/parquet/metadata.json` — summary stats

**What gets skipped:**
- Non-4-player games
- Non-standard VP settings
- Games that fail coordinate mapping

Check the results:

```bash
python3 -c "
import pandas as pd, glob
files = glob.glob('datasets/parquet/shard_*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files])
print(f'Samples: {len(df):,}')
print(f'Games: {df.game_id.nunique()}')
print(f'Features: {len([c for c in df.columns if c.startswith(\"F_\")])}')
print(f'Winner rate: {df.winner.mean():.3f}')
"
```

## Step 3: Train the Neural Network

```bash
python3 -m robottler.train_value
```

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `datasets/parquet` | Parquet input directory |
| `--output` | `robottler/models/value_net.pt` | Model checkpoint path |
| `--epochs` | 50 | Max training epochs |
| `--batch-size` | 512 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-3 | L2 regularization |
| `--patience` | 10 | Early stopping patience |

**Recommended settings for different data sizes:**

| Games | Suggested flags |
|---|---|
| < 100 | `--lr 1e-4 --weight-decay 1e-2` (heavy regularization) |
| 100-500 | `--lr 5e-4 --weight-decay 1e-3` |
| 500+ | defaults should work |

**What it does:**
1. Loads all parquet shards from `--data-dir`
2. Splits by game_id: 70% train / 15% val / 15% test (no data leakage)
3. Computes per-feature mean/std from training set
4. Trains MLP with BCEWithLogitsLoss + class imbalance weighting
5. Early stops on validation loss
6. Reports test set metrics (loss, accuracy, AUC-ROC)
7. Saves checkpoint with model weights + normalization stats

**Output log per epoch:**
```
Epoch   4/100 | train_loss=0.8322 val_loss=1.0155 | val_acc=0.591 val_auc=0.607 | lr=1.0e-04
```

**Target metrics by data size (rough guide):**

| Games | Expected Test AUC |
|---|---|
| 75 | 0.55-0.60 |
| 500 | 0.65-0.70 |
| 2000+ | 0.75+ |

## Step 4: Benchmark Against Hand-Tuned Bot

```bash
python3 -m robottler.benchmark --games 200
```

| Flag | Default | Description |
|---|---|---|
| `--games` | 200 | Number of games to play |
| `--model` | `robottler/models/value_net.pt` | Model checkpoint path |

Pits 1 neural bot (RED) against 3 hand-tuned `ValueFunctionPlayer` bots. Reports win rates and average VP per player.

**What to look for:**
- Neural bot win rate vs expected 25% (fair 4-player)
- Average VP compared to hand-tuned bots
- 200 games takes ~2 minutes

## Step 5: Use in Live Play

Set bot type to `"neural"` in the bridge:

```python
bot = BotInterface(bot_type="neural")
```

Or modify `bridge/server.py` to accept `--bot-type neural` on the command line.

## Quick Reference: Full Pipeline

```bash
# 1. Parse 1000 games
python3 -m datasets.parse_games --limit 1000

# 2. Train
python3 -m robottler.train_value --lr 5e-4 --weight-decay 1e-3

# 3. Benchmark
python3 -m robottler.benchmark --games 200
```

## Architecture

```
datasets/games/*.json          → Raw colonist.io game logs
        ↓ parse_games.py
datasets/parquet/shard_*.parquet → 1002 features + labels per sample
        ↓ train_value.py
robottler/models/value_net.pt  → Trained checkpoint (weights + normalization stats)
        ↓ value_model.py
load_value_model() → value_fn(game, color) → float (win probability)
        ↓ bot_interface.py
BotInterface(bot_type="neural") → 1-step lookahead using neural value function
```

## Model Architecture

```
Input (1002 features)
  → Linear(1002, 128) → BatchNorm → ReLU → Dropout(0.5)
  → Linear(128, 64)   → BatchNorm → ReLU → Dropout(0.5)
  → Linear(64, 32)    → ReLU
  → Linear(32, 1)     → Sigmoid
Output: win probability [0, 1]
```

## Troubleshooting

**Parser hangs after progress bar completes:**
This was caused by `flush_shard()` being outside the Pool context manager — pool cleanup hung before the data was written. This is now fixed (data flushes before pool cleanup). If it still hangs after flushing, Ctrl+C safely — your shards and metadata are already saved.

**Model overfits (train loss → 0, val loss diverges):**
Increase `--weight-decay` (try 1e-2) or reduce `--lr` (try 1e-4). With < 200 games, heavy regularization is essential.

**Neural bot wins 0% of games:**
This is expected with < 200 games of training data. The hand-tuned bot has hardcoded domain knowledge. The neural bot needs 500+ games to start competing.

**Retrain after adding more data:**
Just add more game JSONs to `datasets/games/`, re-run the parser, then retrain. The old parquet files will be overwritten.

python3 -m robottler.train_value --feature-filter strategic --samples-per-game 8 --no-pos-weight --lr 3e-4 --epochs 200 --output robottler/models/value_net_v2.pt --resume

