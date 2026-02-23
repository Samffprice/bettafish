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

### 15. Fine-tune exp #14 on distill data only (220K samples)
- **Architecture**: Shared body (512, 256), dropout=0.1, loaded from exp #14
- **Params**: ~325K total
- **Data**: 220K samples (1,920 depth-2 expert games with blend-derived continuous value targets)
- **Value targets**: Continuous [-0.76, +0.75] from `make_blend_leaf_fn` (two-tier normalized blend), std=0.371
- **Config**: 200 epochs (early stopped at ep 144, restored ep 124), lr=1e-4, cosine, batch=2048, 90/10 train/val split
- **Final losses**: v=0.001, p=1.39 | val: v=0.001, p=1.49
- **Result**: **48% vs AB (n=100, 400 sims)**
- **Note**: Performance DEGRADED from 64% → 48% despite lower losses. Consistent with the recurring pattern: lower loss ≠ better performance. Likely catastrophic forgetting from fine-tuning on small dataset (220K samples, 0.67:1 ratio with 325K params). Value head learned blend values perfectly (v=0.001) but lost the sharp win/loss distinctions MCTS needs.
- **Checkpoint**: `datasets/az_v2_finetune_distill.pt`

---

## Blend Decomposition Experiments

Investigated what makes the blend function work. The blend is `heuristic + 1e8 * neural`. Tested the heuristic and neural contributions separately.

### Heuristic vs Neural Magnitude
- Heuristic values: ~6e14 (dominated by VP weights ~1e14)
- Neural * 1e8: ~1e5 to 2.5e7
- **The heuristic completely dominates** (~99.999999% of the blend value)
- The neural component is a tiebreaker, NOT the primary signal

### 16. BBSearch pure heuristic (no neural) vs AB
- **Player**: BitboardSearchPlayer depth 2, dice-sample 5, **heuristic only** (no neural)
- **Result**: **53.7% vs AB (n=50)**
- **Note**: Pure heuristic BBSearch vs Catanatron's AlphaBeta (also heuristic). BBSearch has bitboard speed + dice sampling advantages, accounting for the >50% rate.

### 17. BBSearch with BC blend vs AB (control)
- **Player**: BitboardSearchPlayer depth 2, dice-sample 5, blend weight 1e8, **BC neural net** (value_net_v2.pt, trained on human games)
- **Result**: **72% vs AB (n=50)**
- **Note**: Confirms the reference baseline. The BC neural tiebreaker adds ~18pp over pure heuristic.

### 18. BBSearch with AZ v2 blend vs AB
- **Player**: BitboardSearchPlayer depth 2, dice-sample 5, blend weight 1e8, **AZ v2 neural net** (az_v2_325k_200ep.pt, trained on 2.07M expert samples)
- **Result**: **64% vs AB (n=50)**
- **Note**: The AZ v2 model is a WORSE tiebreaker than the BC net (64% vs 72%) despite being trained on higher quality data (expert games vs human games) and being a better standalone MCTS evaluator (64% vs 40% as MCTS value head). The BC net's sigmoid [0, 0.18] output may provide better calibrated tie-breaking signals than the AZ tanh [-1, +1] rescaled to [0, 1].

### 19. MCTS hybrid: AZ policy + blend_leaf value
- **Player**: BBMCTSPlayer 400 sims, c_puct=1.4, AZ v2 policy priors + `make_blend_leaf_fn` for leaf evaluation (two-tier normalized blend, std≈0.395)
- **Result**: **54% vs AB (n=50), 62% vs AB (n=50)** — average ~58%
- **Time**: ~3.3s/game (vs ~1.8s for pure AZ — blend evaluation adds ~80% overhead)
- **Note**: Replacing the AZ value head with the blend function HURT performance (~58% vs 64% pure AZ). The blend_leaf values are compressed into [-0.76, +0.75] with std=0.395. MCTS backpropagates averaged values, so when most leaves return similar values (~0.05±0.4), the tree can't distinguish good branches from bad. The AZ value head, despite higher training loss, outputs sharper signals near ±1 that give MCTS clearer guidance. **MCTS needs decisive evaluations, not accurate ones.**

### 20. MCTS hybrid: AZ policy + pure heuristic value
- **Player**: BBMCTSPlayer 400 sims, c_puct=1.4, AZ v2 policy priors + pure heuristic at leaves, normalized via `tanh(raw / 1e14)`
- **Result**: **8% vs AB (n=50)**
- **Time**: ~1.35s/game (faster than blend — no neural inference at leaves)
- **Note**: Broken normalization. Heuristic returns ~6e14, so `tanh(6e14 / 1e14) = tanh(600) ≈ 1.0` for ALL positions. MCTS sees every leaf as identical → effectively random play. Demonstrates that the raw heuristic is fundamentally incompatible with MCTS without careful normalization. The minimax search player doesn't need normalization because it only compares relative values within the same search tree.

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
| 15 | v2 finetune distill | Shared | 220K distill | 144 (ES) | 0.001 | 1.39 | 48% (100) |

### Search Player Blend Decomposition
| # | Eval Mode | Neural Source | vs AB (n) | Delta vs heuristic |
|---|-----------|-------------|-----------|-------------------|
| 16 | Pure heuristic | None | 53.7% (50) | baseline |
| 17 | BC blend 1e8 | value_net_v2 (human games) | 72.0% (50) | +18.3pp |
| 18 | AZ blend 1e8 | az_v2_325k (expert games) | 64.0% (50) | +10.3pp |

### MCTS Hybrid (AZ policy + external value)
| # | Leaf Value Source | vs AB (n) | Notes |
|---|-------------------|-----------|-------|
| 14 | AZ value head (pure) | 64% (100) | baseline MCTS |
| 19 | blend_leaf (normalized blend) | ~58% (2×50) | worse — compressed range hurts MCTS |
| 20 | pure heuristic (tanh/1e14) | 8% (50) | broken normalization — all values ≈ 1.0 |

## Key Lessons

### Training
1. **v2 325K on 2.07M is best MCTS model (64%)** — bigger network + more data + dropout
2. **"More data hurts" was a capacity problem** — exp #7 (114K params, 1.36M data) degraded to 50%, but exp #14 (325K params, 2.07M data) improved to 64%
3. **Dropout 0.1 is critical for larger networks** — prevents overfitting on large datasets
4. **Fresh init > warm-start for larger architectures** — uniform LR works when nothing to protect
5. **Split bodies protect value but lose shared representation** — 45% vs 54% (old) / 64% (new)
6. **Freezing everything except policy head fails** — 27K params can't learn from value-oriented features
7. **Unfreezing value on small data is catastrophic** — 69K samples → 18%
8. **Lower loss NEVER predicts better performance** — consistent across ALL experiments: #4, #11, #15, #19. Best losses = worst results.
9. **500 epochs on 272K overfits** — 200ep is the sweet spot for that dataset size
10. **n=10 benchmarks are unreliable** — 60% (n=10) was really 54% (n=100)
11. **Sample-to-parameter ratio matters** — 6.4:1 (2.07M/325K) works

### Value Function
12. **MCTS needs decisive values, not accurate ones** — the AZ value head (v_loss=0.44, outputs near ±1) beats both the blend_leaf (v_loss=0.001, outputs ±0.4) and distilled value head at MCTS. Sharp win/loss signals > precise position evaluations.
13. **More MCTS sims don't help once value head is saturated** — 400 and 800 sims both give 64%
14. **Value distillation hurts MCTS** — exp #15 (fine-tune on blend targets): 64% → 48%. Exp #19 (blend_leaf at inference): 64% → 58%. Both compressed value ranges degraded MCTS tree search.

### Blend / Search Player
15. **The heuristic completely dominates the blend** (~6e14 vs ~1e7 for neural*1e8). Neural is a tiebreaker only.
16. **The neural tiebreaker is worth ~18pp in minimax** — pure heuristic 53.7% vs BC blend 72%
17. **A "better" neural net makes a WORSE minimax tiebreaker** — AZ v2 (+10pp) vs BC net (+18pp). Being good at MCTS ≠ being good at tie-breaking in minimax.
18. **Minimax + heuristic > MCTS + neural at same compute** — BBSearch depth 2 (72%) beats MCTS 400 sims (64%). The hand-crafted heuristic encodes Catan knowledge that the neural net hasn't learned.

### Data
19. **MCTS self-play data is harmful** when generated by a weak policy (games 265 turns)
20. **Depth-3 targets help when mixed into large dataset** but not alone (20× less data)

## The Core Problem

Every attempt to give MCTS a "better" value function has made it worse:
- Distilling blend values into the value head: 64% → 48% (exp #15)
- Using blend directly at leaves: 64% → ~58% (exp #19)
- More MCTS sims: 64% → 64% (400 vs 800)

Meanwhile, the search player gets 72% with the same heuristic that MCTS can't effectively use. The fundamental issue may be that **MCTS with 400 sims in a high-branching stochastic game like Catan doesn't build deep enough trees to benefit from precise evaluations**. It needs coarse but decisive signals to allocate its limited simulation budget effectively.

However: MCTS (64%) vs BBSearch blend direct matchup = 38%, meaning MCTS is closer to the blend than the AB benchmark suggests (blend beats AB harder than it beats MCTS). And the 64% model plays ~86-turn games in self-play — normal Catan length, not the 265-turn garbage from earlier attempts.

---

## Self-Play Pipeline (Phase 6B)

### Motivation
Expert iteration hit a wall at 64% vs AB. The model can't surpass the expert (72% BBSearch) through imitation alone. Self-play is the path to improvement without a ceiling — the model learns from its own games, discovering strategies the expert doesn't use. The 64% model now plays reasonable ~86-turn games (vs 265 turns when self-play was first attempted at ~10-20% strength).

### Key Advantage of Self-Play Data
MCTS self-play produces **soft policy targets** (visit count distributions over all 290 actions) instead of one-hot expert targets. This is richer signal — "action A: 60%, B: 25%, C: 15%" vs "the expert picked A." The value targets remain binary game outcomes (+1/-1), which is what works best for MCTS (lesson #12: decisive > accurate).

### Pipeline Per Iteration

#### 1. Generate self-play games
```bash
python3 -m robottler.az_selfplay generate \
    --checkpoint <latest_checkpoint> \
    --games <N> --sims 400 \
    --output-dir datasets/az_selfplay_v2_rX \
    --workers 7
```
- Timing: ~3.2s per game completion (7 workers on M1), so:
  - 500 games ≈ 25 min
  - 2,000 games ≈ 1.7 hours
  - 5,000 games ≈ 4.4 hours
- Progress bar shows avg_turns (target: ~70-90 for healthy games)
- Saves incrementally — safe to Ctrl+C and resume

#### 2. Train on cumulative data
```bash
python3 -m robottler.az_selfplay train \
    --checkpoint <latest_checkpoint> \
    --data-dir datasets/az_selfplay_v2_r1 [... all prior self-play dirs ...] \
               datasets/exit_v1/iter1 ... datasets/expert_depth3 \
    --output datasets/az_v2_selfplay_rX.pt \
    --epochs 200 --batch-size 2048 --lr 1e-4 \
    --scheduler cosine
```
- Loads weights from previous checkpoint (fine-tunes, not fresh)
- No `--body-dims` flag (keeps existing 512,256 architecture)
- Early stopping (patience 20, 90/10 train/val split) prevents overfitting
- Training time: ~10-15 min on MPS

#### 3. Benchmark
```bash
python3 /tmp/bench_az_vs_ab.py datasets/az_v2_selfplay_rX.pt 100 6 400
```
- 100 games vs AlphaBeta, ~3 min
- Target: beat previous round's win rate

### Games Per Round

| Round size | Samples | % new (round 1) | Gen time | Total/iter |
|-----------|---------|-----------------|----------|------------|
| 500 | ~40K | 1.9% | 25 min | ~40 min |
| 2,000 | ~160K | 7% | 1.7 hr | ~2 hr |
| 5,000 | ~400K | 16% | 4.4 hr | ~5 hr |

**Recommendation**: Start with 500 as proof-of-concept. If it shows improvement, scale to 2,000-5,000 per round for meaningful signal.

### Data Strategy: Cumulative → Sliding Window

**Phase 1 — Cumulative** (rounds 1 through ~40-50):
- Keep ALL expert data (2.07M samples) + ALL self-play data
- Expert data provides stable base while self-play data accumulates
- Continue until self-play data alone reaches ~2M samples (~25,000 games)

**Phase 2 — Sliding Window** (once self-play data ≥ 2M samples):
- Drop expert data entirely
- Keep the most recent ~25,000 self-play games (~2M samples)
- Rationale: 325K params needs ~2M samples (6:1 ratio). Old self-play data from weaker model versions teaches bad habits.
- Slide forward: each new round drops the oldest games

**Transition point**: ~25,000 cumulative self-play games. At 2,000 games/round, that's ~12-13 rounds. At 5,000/round, ~5 rounds.

### Success Criteria
- **Round 1**: Any improvement over 64% vs AB confirms self-play is viable
- **Short-term**: Match the BBSearch expert at 72% vs AB
- **Medium-term**: Exceed 72% (surpass the expert — the whole point of self-play)
- **Long-term**: Reach 80%+ vs AB (approaching BBSearch depth 3's 84%)
- **Health check**: avg_turns should stay in 70-100 range. If games get longer (>120 turns), the model is degrading.

### Current Status
- Round 1 generation in progress: 500 games, ~86 avg turns (healthy)
- Checkpoint: `datasets/az_v2_325k_200ep.pt` (64% vs AB, exp #14)

## Open Questions
- Why does the BC net (trained on human games, weaker standalone) outperform the AZ net as a search tiebreaker? Is it the sigmoid vs tanh output range? The training data distribution? The blend weight (1e8 tuned for BC's [0, 0.18] range; AZ outputs [0, 1], so effective weight is ~5x higher)?
- Would more MCTS sims (1600, 3200) eventually close the gap with search, or is the algorithm fundamentally worse for Catan?
- Could a hybrid minimax+MCTS approach work? E.g., use minimax for the first 2 plies then MCTS for deeper exploration?
- Is the path forward improving the search player (better neural tiebreaker, deeper search, better pruning) rather than MCTS?
- At what point during self-play training should we re-evaluate the blend weight for using the AZ net as a tiebreaker in the search player?
- Should we increase MCTS sims (800, 1600) for self-play generation even if it doesn't help at inference? Deeper search during generation = higher quality training data.
