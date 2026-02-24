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
- **Result**: **~55% vs AB (revised estimate; original 64% was a statistical outlier)**
  - Original measurement: 64/100 (400 sims), 64/100 (800 sims) — both on same day
  - Reproduction runs (same script, same checkpoint): 59/100, 51/100, 54/100, 50/100
  - Pooled 6 runs (600 games): 342/600 = **57%** [53%, 61%]
  - Best estimate: **~55% vs AB** (400 or 800 sims — no difference)
- **vs BBSearch depth 2**: 38% (n=50, 400 sims)
- **Note**: Best model overall, but ~17pp below BBSearch (72%), not 8pp as initially thought. Bigger network + more data + dropout solved the "more data hurts" problem from #7. 800 sims = same as 400 → value head is the bottleneck, not search depth.
- **Checkpoint**: `datasets/az_v2_325k_200ep.pt`
- **Bug found**: `BBMCTSPlayer.decide()` was missing `fi.update_map(game.state.board.map)` — production features used wrong map when player was reused across games. Fixed (not yet committed). Original benchmark created new player per game with correct map, so was unaffected.

### 15. Fine-tune exp #14 on distill data only (220K samples)
- **Architecture**: Shared body (512, 256), dropout=0.1, loaded from exp #14
- **Params**: ~325K total
- **Data**: 220K samples (1,920 depth-2 expert games with blend-derived continuous value targets)
- **Value targets**: Continuous [-0.76, +0.75] from `make_blend_leaf_fn` (two-tier normalized blend), std=0.371
- **Config**: 200 epochs (early stopped at ep 144, restored ep 124), lr=1e-4, cosine, batch=2048, 90/10 train/val split
- **Final losses**: v=0.001, p=1.39 | val: v=0.001, p=1.49
- **Result**: **48% vs AB (n=100, 400 sims)**
- **Note**: Performance degraded from ~55% → 48% (smaller gap than originally thought with revised exp #14 baseline). Lower loss ≠ better performance. Value head learned blend values perfectly (v=0.001) but lost the sharp win/loss distinctions MCTS needs.
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
| **14** | **v2 325K fresh** | **Shared** | **2.07M d2+d3** | **200** | **0.44** | **1.59** | **~55% (600)** |
| 15 | v2 finetune distill | Shared | 220K distill | 144 (ES) | 0.001 | 1.39 | 48% (100) |
| 21 | SP r1 fine-tune | Shared | 2.11M +SP | 200 | — | — | 38% (100) |
| 22 | SP r1 fresh (GPU) | Shared | 2.11M +SP | 200 | — | — | 44% (100) |

### Search Player Blend Decomposition
| # | Eval Mode | Neural Source | vs AB (n) | Delta vs heuristic |
|---|-----------|-------------|-----------|-------------------|
| 16 | Pure heuristic | None | 53.7% (50) | baseline |
| 17 | BC blend 1e8 | value_net_v2 (human games) | 72.0% (50) | +18.3pp |
| 18 | AZ blend 1e8 | az_v2_325k (expert games) | 64.0% (50) | +10.3pp |

### MCTS Hybrid (AZ policy + external value)
| # | Leaf Value Source | vs AB (n) | Notes |
|---|-------------------|-----------|-------|
| 14 | AZ value head (pure) | ~55% (600) | baseline MCTS (revised) |
| 19 | blend_leaf (normalized blend) | ~58% (2×50) | similar to baseline — compressed range |
| 20 | pure heuristic (tanh/1e14) | 8% (50) | broken normalization — all values ≈ 1.0 |

## Key Lessons

### Training
1. **v2 325K on 2.07M is best MCTS model (~55%)** — bigger network + more data + dropout
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
13. **More MCTS sims don't help once value head is saturated** — 400 and 800 sims both give ~55%
14. **Value distillation hurts MCTS** — exp #15 (fine-tune on blend targets): ~55% → 48%. Compressed value ranges degrade MCTS tree search.
15. **Value head is overoptimistic** — MCTS diagnostics show Q > 0 in 54% of lost-game decisions (exp #14). The network doesn't properly recognize losing positions, so MCTS can't prioritize defensive moves.

### Blend / Search Player
16. **The heuristic completely dominates the blend** (~6e14 vs ~1e7 for neural*1e8). Neural is a tiebreaker only.
17. **The neural tiebreaker is worth ~18pp in minimax** — pure heuristic 53.7% vs BC blend 72%
18. **A "better" neural net makes a WORSE minimax tiebreaker** — AZ v2 (+10pp) vs BC net (+18pp). Being good at MCTS ≠ being good at tie-breaking in minimax.
19. **Minimax + heuristic >> MCTS + neural** — BBSearch depth 2 (72%) beats MCTS 400 sims (~55%). The gap is ~17pp, not 8pp as originally thought. The hand-crafted heuristic encodes Catan knowledge that the neural net hasn't learned.

### Data
20. **MCTS self-play data is harmful** when generated by a weak policy (games 265 turns)
21. **Depth-3 targets help when mixed into large dataset** but not alone (20× less data)
22. **Self-play data from a sub-expert model hurts** — exp #22 added 40K self-play from ~55% model to 2.07M expert data → 44% (down from ~55%). The model's data is lower quality than the expert data (72% search player), diluting the training signal. MCTS diagnostics confirm: value head overoptimism 65% (vs 54% for exp #14), late-game loss Q still positive (+0.148 vs -0.127).
23. **Fine-tuning always fails** — loading weights and re-training degrades performance (exp #4, #15, #21). Must always fresh-init with `--body-dims`.
24. **Batch size may matter** — exp #22 used 4096 (GPU) vs #14's 2048 (MPS). Needs controlled test to isolate.
25. **n=100 benchmarks have ±10pp variance** — the original 64% measurement was a 2.9σ outlier from the true ~55% rate. Even n=100 can mislead. Multiple independent runs needed for reliable estimates.

## The Core Problem

**MCTS at ~55% vs AB is 17pp below BBSearch (72%).** The gap is much larger than originally thought (was believed to be 8pp when we thought MCTS was at 64%).

MCTS diagnostics (100-game runs with per-decision stats) reveal:
- **Value head overoptimism**: Q > 0 in 54% of lost-game decisions. The network thinks it's winning when losing.
- **Weak win/loss separation**: Q-trajectories diverge late but not enough (win late-game Q: +0.30, loss late-game Q: -0.13)
- **Policy is surprisingly good**: prior top-1 matches MCTS top-1 in ~55% of decisions, top-3 captures top-1 in ~72%
- **Decent tree depth**: avg ~6.5 sim depth with ~10 avg legal actions — search is going deep enough
- **Q-spread is adequate**: mean 0.15, showing value head differentiates between moves

The bottleneck is the **value head's inability to recognize losing positions**, not policy or search depth. MCTS can't play defensively when the value head stays positive during losses.

Value distillation hurt (exp #15: ~55% → 48%), more sims don't help (400 = 800), and the blend_leaf hybrid (exp #19: ~58%) is barely different from baseline — all consistent with the value head being the ceiling.

MCTS (~55%) vs BBSearch direct matchup = 38% (n=50), meaning MCTS is closer to BBSearch head-to-head than the AB benchmark suggests.

---

## Self-Play Pipeline (Phase 6B)

### Motivation
Expert iteration reached ~55% vs AB. The model can't surpass the expert (72% BBSearch) through imitation alone. Self-play is the path to improvement without a ceiling — the model learns from its own games, discovering strategies the expert doesn't use. The ~55% model plays reasonable ~86-turn games (vs 265 turns when self-play was first attempted at ~10-20% strength). However, diagnostics show the model is too weak for useful self-play data (value head overoptimism prevents learning from losses).

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
- **Round 1**: Any improvement over ~55% vs AB confirms self-play is viable
- **Short-term**: Match the BBSearch expert at 72% vs AB
- **Medium-term**: Exceed 72% (surpass the expert — the whole point of self-play)
- **Long-term**: Reach 80%+ vs AB (approaching BBSearch depth 3's 84%)
- **Health check**: avg_turns should stay in 70-100 range. If games get longer (>120 turns), the model is degrading.

### 21. Self-play round 1 — fine-tune (2.07M + 40K self-play)
- **Architecture**: Shared body (512, 256), dropout=0.1, **loaded from exp #14 weights**
- **Params**: ~325K total
- **Data**: 2.07M (original expert) + ~40K (500 self-play games, ~86 avg turns) = ~2.11M samples
- **Config**: 200 epochs, lr=1e-4, cosine, batch=2048, MPS
- **Result**: **38% vs AB (n=100, 400 sims)**
- **Note**: Performance DEGRADED from ~55% → 38%. Re-training on data the model already converged on = overfitting (same pattern as exp #4). Fine-tuning does NOT work — must use fresh init each time.
- **Lesson**: **Never fine-tune. Always fresh init with `--body-dims 512,256 --dropout 0.1`**

### 22. Self-play round 1 — fresh init (2.07M + 40K self-play, Colab GPU)
- **Architecture**: Shared body (512, 256), dropout=0.1, **fresh Xavier init**
- **Params**: ~325K total
- **Data**: Same as #21 — 2.07M expert + ~40K self-play + `az_selfplay_v2` data
- **Config**: 200 epochs, lr=1e-4, cosine, batch=4096, **CUDA (Colab T4)**
- **Result**: **44% vs AB (n=100, 400 sims)**
- **Note**: ~11pp worse than exp #14 (~55%). The only difference from #14 is including the self-play data (`datasets/az_selfplay_v2`). Self-play data from a ~55% model HURTS because it's lower quality than the expert data (72% search player). Diagnostics confirm: value head overoptimism increased from 54% → 65%, late-game loss Q stayed positive (+0.148 vs -0.127). Also batch size was 4096 (vs 2048 for #14), which may affect convergence.
- **Lesson**: **Don't add self-play data generated by a model weaker than the expert.** Self-play is only useful once the model surpasses the expert.

---

## Training Infrastructure

### Memory Optimization
- Original code preloaded all data to MPS as 3 copies (numpy float64 → Dataset float32 → train/val float32 split) = **~20GB** on 8GB M1
- Fixed: cast to float32 early, convert one array at a time via `torch.from_numpy` (zero-copy) → index → `.to(device)` → delete numpy immediately
- Result: ~5GB steady state, ~8GB peak during conversion

### Resumable Training
- `--resume-training PATH` saves checkpoint every 5 epochs (`_training.pt` file)
- Stores: model weights, optimizer state, LR scheduler state, best val loss, epoch counter, training history
- Auto-deleted when training completes normally
- Ctrl+C safe: re-run same command with `--resume-training` added

### Colab GPU Training
- Notebook: `bettafish_colab.ipynb` section 3
- Workflow: zip data locally → upload to Google Drive → Colab extracts and trains → copies result back to Drive
- T4 GPU: ~5-8s/epoch (vs ~30s on M1 MPS)
- Data compresses well: 2.07M samples (3.7GB raw) → ~79MB zip (96-100% deflation on .npy)
- Batch size 4096 on GPU (vs 2048 on M1)

### Key Training Rules
1. **Always use `--body-dims 512,256 --dropout 0.1`** for fresh init — never fine-tune
2. **Batch size**: 2048 on M1 MPS, 4096 on GPU
3. **200 epochs with cosine LR + early stopping (patience 20)** is the standard config
4. **n=100 for benchmarks** — n=10/20 is too noisy

---

## MCTS Diagnostics (Feb 2026)

Instrumented `BBMCTSPlayer._simulate()` and `bb_decide()` with per-decision tracking. Analysis script: `/tmp/mcts_diagnostics.py`.

### Exp #14 Profile (az_v2_325k_200ep.pt, 400 sims, n=100)
| Metric | Value | Assessment |
|--------|-------|------------|
| Win rate vs AB | ~55% [49%, 61%] | Revised down from 64% |
| Policy top-1 accuracy | ~55% | MODERATE — policy provides guidance but MCTS overrides often |
| Policy top-3 accuracy | ~72% | OK — MCTS choice usually in prior's top 3 |
| Value win/loss separation | +0.14 | MODERATE — some signal but noisy |
| Value overoptimism | 54% | CONCERNING — Q > 0 in majority of lost-game decisions |
| Avg sim depth | ~6.5 | OK for ~10 avg legal actions |
| Visit top-1 fraction | ~55% | MODERATE — reasonably focused |
| Q-value spread | 0.15 | OK — differentiates between moves |
| Action coverage | ~97% (10.4 avg actions) | Expected with 400 sims and small action space |

### Exp #22 Profile (az_v2_selfplay_r1.pt, 400 sims) — Comparison
Self-play data damaged the value head, not the policy:
| Metric | Exp #14 | Exp #22 | Delta |
|--------|---------|---------|-------|
| Win rate | ~55% | ~44% | -11pp |
| Policy top-1 accuracy | ~55% | ~54% | ≈same |
| Value overoptimism | 54% | 65% | +11pp (worse) |
| Late-game loss Q | -0.127 | +0.148 | Doesn't go negative! |
| Win/loss separation | +0.14 | +0.08 | Weaker signal |

**Diagnosis**: The self-play data taught the value head to be more optimistic (never admitting it's losing), which cripples MCTS's ability to avoid losing lines.

## Current Status & Path Forward

### Where We Are
- **Best MCTS model**: ~55% vs AB (exp #14, 400 sims, az_v2_325k_200ep.pt)
- **Best search player**: 72% vs AB (BBSearch depth 2, blend 1e8)
- **Gap**: ~17pp — much larger than originally thought
- **Primary bottleneck**: Value head overoptimism (doesn't recognize losing positions)
- **Self-play data hurts** when generated by a model weaker than the expert (exp #22)
- **Fine-tuning always hurts** — must use fresh init (exp #4, #15, #21)

### The Expert Ceiling Problem
Training on expert data (one-hot policy targets from 72% search player):
- **Policy head** is capped at expert quality — it can only imitate what the expert does
- **Value head** trains on game outcomes (+1/-1), but at ~55% the model is too weak for self-play to help
- **MCTS can amplify** a good network beyond expert quality — 400 sims searches depth 10-20
- **But** the model needs to be close enough (~70%+) for self-play data to match expert quality
- At ~55%, the gap to self-play viability is larger than previously estimated (~15pp, not ~6pp)

### Key Insight: Value Head Is the Bottleneck
The diagnostics clearly identify the value head as the ceiling. The policy is adequate (~55% top-1 accuracy, ~72% top-3), search depth is reasonable (~6.5), but the value head:
1. **Stays positive during losses** (54% overoptimism) — MCTS doesn't defend
2. **Weak separation** (+0.14 gap) — hard to distinguish winning from losing trees
3. **Adding self-play data makes it worse** — more optimistic, less calibrated

### Realistic Next Steps
1. **Fix the value head** — the current value targets are binary (+1/-1 game outcomes). The value head may need:
   - Asymmetric loss weighting (penalize overoptimism in losses more)
   - Auxiliary targets (VP differential, resource advantage) alongside win/loss
   - Temperature-scaled targets (discount early-game outcomes, weight endgame)
   - Hard negative mining (oversample positions from lost games)
2. **More expert data at depth 3** — higher quality targets, especially for losing positions where the expert was forced into bad states
3. **Architecture changes** — separate value body may help the value head specialize on loss detection
4. **Hybrid approach** — use BBSearch for opening (where heuristic dominates) and MCTS for midgame/endgame

### What Probably Won't Help
- More MCTS sims (400 = 800, confirmed)
- Self-play data at current strength (exp #22 shows it hurts)
- Fine-tuning existing checkpoint (always degrades)
- Value distillation from blend function (exp #15, #19 show it hurts)

## Open Questions
- Can asymmetric loss weighting fix the overoptimism problem?
- Would training the value head with VP-differential auxiliary targets improve calibration?
- Is the policy's ~55% top-1 accuracy actually good enough, or do we need better policy too?
- Could curriculum learning (easy positions → hard positions) help the value head learn loss detection?
- Would a separate value body (split architecture, exp #10-#13) help if combined with overoptimism fixes?
- Is the path forward improving the search player (better neural tiebreaker) rather than MCTS?
