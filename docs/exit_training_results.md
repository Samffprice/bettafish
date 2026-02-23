# Expert Iteration (ExIt) Training Results

All benchmarks vs Catanatron's AlphaBeta (depth 2) using BBMCTSPlayer with c_puct=1.4, temperature=0, no Dirichlet noise.

## Reference Baselines
| Player | vs AlphaBeta | Notes |
|--------|-------------|-------|
| BitboardSearchPlayer depth 2 | 72% (n=100) | Blend weight 1e8, dice-sample 5 |
| BitboardSearchPlayer depth 3 | 84% (n=20) | ~54s/game |
| Old MCTS (no policy) 800 sims | 10-20% | Pre-ExIt |
| Old MCTS (root policy) 800 sims | ~42% | Pre-ExIt, RL policy |

---

## Experiment Log

### 1. Initial ExIt Loop (3 iterations, 5ep each)
- **Architecture**: Shared body
- **Data**: 3 iterations × 500 expert games (depth 2) = ~165K samples
- **Config**: 5 epochs/iter, lr=1e-4, cosine, differential LR (body/value=1e-5, policy=1e-3)
- **Bug**: Policy loss was 0.0 — action masking reduced one-hot targets to single value → log(1.0) = 0
- **Fix**: Removed action masking from policy loss
- **Result**: 10% vs AB (n=10), policy loss ~2.6

### 2. 50ep cosine (272K samples)
- **Architecture**: Shared body
- **Data**: 5 iterations × 500 expert games (depth 2) = 272,520 samples
- **Config**: 50 epochs, lr=1e-4, cosine, differential LR
- **Final losses**: v=~0.4, p=1.86
- **Result**: **30% vs AB (n=10)**

### 3. 200ep cosine (272K samples) — SHARED BODY BEST
- **Architecture**: Shared body
- **Data**: 272,520 samples (5 × 500 depth-2 expert games)
- **Config**: 200 epochs, lr=1e-4, cosine, differential LR (body/value=1e-5, policy=1e-3), batch=256
- **Final losses**: v=0.19, p=1.54
- **Result**: **60% vs AB (n=10), 54% vs AB (n=100)**
- **Note**: Best shared-body model. Value loss degraded from ~0.83 → 0.19 (body adapted). The n=10 "60%" was variance; true rate is ~54%.

### 4. 500ep cosine, expert only (272K samples)
- **Architecture**: Shared body
- **Data**: 272,520 samples
- **Config**: 500 epochs, lr=1e-4, cosine, differential LR
- **Final losses**: v=0.159, p=1.48
- **Result**: **30% vs AB (n=10)**
- **Note**: Overfitting. Cosine scheduler kills LR early, model memorizes.

### 5. 500ep cosine, expert + MCTS combined
- **Architecture**: Shared body
- **Data**: 272K expert + MCTS self-play data combined
- **Config**: 500 epochs, lr=1e-4, cosine, differential LR
- **Final losses**: v=~0.17, p=1.77
- **Result**: **20% vs AB (n=10)**
- **Note**: MCTS data was bad quality (games 111→157→265 turns). Mixing bad data hurt.

### 6. 200ep flat LR + label smoothing (272K samples)
- **Architecture**: Shared body
- **Data**: 272,520 samples
- **Config**: 200 epochs, lr=1e-4, **constant LR**, differential LR, label smoothing=0.1
- **Final losses**: v=0.190, p=2.27 (inflated by smoothing)
- **Result**: **50% vs AB (n=10)**
- **Note**: Label smoothing inflates cross-entropy. Worse than cosine 200ep.

### 7. 200ep cosine (1.36M samples, depth-2)
- **Architecture**: Shared body
- **Data**: 272K (exit_v1) + 1.08M (expert_data_10k) = 1,356,745 samples
- **Config**: 200 epochs, lr=1e-4, cosine, differential LR, batch=256
- **Final losses**: v=0.509, p=1.54
- **Result**: **50% vs AB (n=10)**
- **Note**: More data didn't help. Value loss 0.51 (worse than 272K model's 0.19). Body couldn't serve both heads with 5× data.

### 8. 200ep frozen value head (1.36M samples)
- **Architecture**: Shared body, value head frozen, body+policy train
- **Data**: 1,356,745 samples
- **Config**: 200 epochs, lr=1e-4, cosine, differential LR, freeze-value (head only)
- **Final losses**: v=0.417, p=1.50
- **Result**: **40% vs AB (n=10)**
- **Note**: Body still shifted (only head frozen, not body). Value degraded anyway.

### 9. Policy-head-only (1.36M samples) — FROZEN BODY + VALUE
- **Architecture**: Shared body, everything frozen except policy_head (27,106 params)
- **Data**: 1,356,745 samples
- **Config**: 200 epochs, lr=1e-4, cosine, batch=256
- **Final losses**: v=0.827 (frozen), p=3.42
- **Result**: Not benchmarked (policy loss too high)
- **Note**: 27K params couldn't learn from frozen 128-dim features. Body features were trained for value, not policy.

---

## Split Body Architecture

Changed `CatanAlphaZeroNet` to have separate `value_body` and `policy_body` (both 176→256→128). Total params: 191,843 (value path: 86,529, policy path: 105,314). Old shared-body checkpoints auto-migrate on load.

### 10. Split-body frozen value, 500ep (1.36M depth-2)
- **Architecture**: Split body, value path frozen (value_body + value_head)
- **Data**: 1,356,745 samples (depth-2 expert)
- **Config**: 500 epochs, lr=1e-4, cosine, freeze-value, batch=2048, MPS GPU
- **Final losses**: v=0.827 (frozen), p=1.486
- **Result**: **60% vs AB (n=20), 45% vs AB (n=100), 47% at 800 sims (n=100)**
- **Note**: Best policy loss ever (1.486). Value pristine. But n=20 was variance; true rate ~45%.

### 11. Split-body unfrozen, 500ep (69K depth-3)
- **Architecture**: Split body, both paths unfrozen
- **Data**: 69,499 samples (690 depth-3 expert games)
- **Config**: 500 epochs, lr=1e-4, cosine, differential LR (value=1e-5, policy=1e-3), batch=2048
- **Final losses**: v=0.179, p=1.449
- **Result**: **18% vs AB (n=100)**
- **Note**: Value body overfitted on 69K samples. Best losses but worst performance.

### 12. Split-body unfrozen, 2000ep (69K depth-3)
- **Architecture**: Split body, both paths unfrozen
- **Data**: 69,499 samples
- **Config**: 2000 epochs, lr=1e-4, cosine, differential LR, batch=2048
- **Final losses**: v=0.055, p=1.077
- **Result**: Not benchmarked (massive overfit)
- **Note**: Extreme overfitting. Value memorized training set.

### 13. Split-body frozen value, 500ep (69K depth-3)
- **Architecture**: Split body, value path frozen
- **Data**: 69,499 samples (depth-3 expert)
- **Config**: 500 epochs, lr=1e-4, cosine, freeze-value, batch=2048
- **Final losses**: v=0.839 (frozen), p=1.815
- **Result**: **42% vs AB (n=100)**
- **Note**: Higher quality targets but 20× less data than depth-2 model.

---

## Bigger Network (v2, 325K params)

Switched from 176→256→128 (114K params) to 176→512→256 (325K params) with dropout=0.1. Fresh random init (no warm-start); only feature normalization metadata loaded from checkpoint. Rationale: 2M samples ÷ 325K params = 6.4:1 ratio (within 5-50× safe zone).

### 14. v2 325K shared, 200ep (2.07M mixed d2+d3) — **OVERALL BEST**
- **Architecture**: Shared body (512, 256), dropout=0.1, fresh init (Xavier)
- **Params**: ~325K total
- **Data**: 272K (exit_v1 d2) + 1.08M (expert_data_10k d2) + 715K (expert_depth3 d3) = **2,071,559 samples**
- **Config**: 200 epochs, lr=1e-4, cosine, uniform LR (no differential — fresh network), batch=2048
- **Final losses**: v=0.444, p=1.592 (total=2.035)
- **Loss at ep100**: v=0.471, p=1.600 (total=2.072)
- **Result**: **64% vs AB (n=100, 400 sims), 64% vs AB (n=100, 800 sims)**
- **vs BBSearch depth 2**: 38% (n=50, 400 sims)
- **Note**: New best by 10pp over exp #3. Bigger network + more data + dropout solved the "more data hurts" problem from #7. 800 sims = same as 400 → value head is the bottleneck, not search depth. Policy still plateaued at 1.59 (similar to smaller models), suggesting policy capacity isn't the limiting factor either.
- **Checkpoint**: `datasets/az_v2_325k_200ep.pt`

---

## Summary Table

| # | Model | Arch | Data | Epochs | Value Loss | Policy Loss | vs AB (n) |
|---|-------|------|------|--------|-----------|-------------|-----------|
| 1 | Initial 3-iter loop | Shared | 165K d2 | 5/iter | — | ~2.6 | 10% (10) |
| 2 | 50ep cosine | Shared | 272K d2 | 50 | ~0.4 | 1.86 | 30% (10) |
| 3 | 200ep cosine | Shared | 272K d2 | 200 | 0.19 | 1.54 | 54% (100) |
| 4 | 500ep cosine | Shared | 272K d2 | 500 | 0.16 | 1.48 | 30% (10) |
| 5 | 500ep expert+MCTS | Shared | 272K+MCTS | 500 | ~0.17 | 1.77 | 20% (10) |
| 6 | 200ep flat+smooth | Shared | 272K d2 | 200 | 0.19 | 2.27 | 50% (10) |
| 7 | 200ep cosine big | Shared | 1.36M d2 | 200 | 0.51 | 1.54 | 50% (10) |
| 8 | 200ep frozen head | Shared | 1.36M d2 | 200 | 0.42 | 1.50 | 40% (10) |
| 9 | Policy-head only | Shared | 1.36M d2 | 200 | 0.83 | 3.42 | — |
| 10 | Split frozen 500ep | Split | 1.36M d2 | 500 | 0.83 | 1.49 | 45% (100) |
| 11 | Split unfrozen d3 | Split | 69K d3 | 500 | 0.18 | 1.45 | 18% (100) |
| 12 | Split unfrozen d3 2k | Split | 69K d3 | 2000 | 0.06 | 1.08 | — |
| 13 | Split frozen d3 | Split | 69K d3 | 500 | 0.84 | 1.82 | 42% (100) |
| **14** | **v2 325K fresh** | **Shared** | **2.07M d2+d3** | **200** | **0.44** | **1.59** | **64% (100)** |

## Key Lessons

1. ~~**Shared body 200ep on 272K is the best model (54%)**~~ → **v2 325K on 2.07M is best (64%)**
2. **"More data hurts" was a capacity problem, not a data problem** — exp #7 (114K params, 1.36M data) degraded to 50%, but exp #14 (325K params, 2.07M data) improved to 64%. The old network was too small to learn from large data without overfitting the value head.
3. **Dropout 0.1 is critical for larger networks** — prevents the overfitting that plagued shared-body training on large datasets
4. **Fresh init > warm-start for larger architectures** — uniform LR (no differential) works better when the network is randomly initialized, since there's nothing to protect
5. **Split bodies protect value but lose the shared representation benefit** — 45% vs 54% (old) / 64% (new)
6. **Freezing everything except policy head fails** — 27K params can't learn from value-oriented features
7. **Unfreezing value on small data is catastrophic** — 69K samples → value memorizes → 18%
8. **Policy loss doesn't predict win rate** — model #11 has best losses (v=0.18, p=1.45) but worst performance (18%)
9. **Depth-3 targets didn't help alone** — quality couldn't compensate for 20× less data (exp #13), but depth-3 mixed into large dataset helped (exp #14)
10. **500 epochs on 272K overfits** — 200ep is the sweet spot for that dataset size
11. **MCTS self-play data is harmful** when generated by a weak policy (games 265 turns)
12. **n=10 benchmarks are unreliable** — 60% (n=10) was really 54% (n=100)
13. **More MCTS sims don't help once value head is saturated** — 400 and 800 sims both give 64%. Value head is the bottleneck.
14. **Sample-to-parameter ratio matters** — 6.4:1 (2.07M/325K) works; 12:1 (1.36M/114K) was too high for the small network's capacity

## Next Steps (Planned)
- **Value distillation**: Train value head on blend function outputs (continuous [-1,+1]) instead of binary game outcomes (+1/-1). Implementation ready: `--distill-values` flag generates blend-derived targets via `make_blend_leaf_fn` (two-tier normalized: VP signal `tanh(vp_diff/4)*0.6` + sub-VP signal `tanh(sub_vp_adv/3e8)*0.4`, std≈0.395). Train/val split (90/10) with early stopping (patience=20).
- **Then self-play**: Once value head matches blend quality, switch to MCTS self-play to surpass the blend ceiling.
