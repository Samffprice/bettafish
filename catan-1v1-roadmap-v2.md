# 1v1 Catan AI: Complete Development Roadmap (v2)

> **Goal:** Build the strongest possible 1v1 no-trading Catan bot, capable of beating high-ELO players on colonist.io.
>
> **Core insight:** 1v1 without trading collapses the problem to a tractable decision space. No combinatorial trade proposals, no kingmaking, no multi-agent chaos. The bot only needs to master: settlement/road/city placement, resource management, bank/port trades, dev card play, and robber placement.

---

## Architecture Overview

```
Training Pipeline (offline)
───────────────────────────────────────────────────────────
Stage 1: Behavioral Cloning ✅ COMPLETE
  44k colonist.io games (4p) → parse → (state, winner) pairs
  Train value network: board state → win probability
  Result: 0.757 AUC, 48% vs AlphaBeta (on par with hand-tuned)

Stage 2: Self-Play RL ← CURRENT
  Initialize from BC weights → MaskablePPO in Catanatron 1v1
  Policy-value network learns from millions of self-play games

Stage 3: Forward Search (MCTS)
  At each decision: simulate ahead using value network
  Pick action with best expected outcome across rollouts
───────────────────────────────────────────────────────────

Deployment (online)
───────────────────────────────────────────────────────────
colonist.io (browser + Cloudflare)
    ↕ WebSocket msgpack frames
Tampermonkey Userscript (intercepts WS)
    ↕ JSON over localhost
Bridge Server (Python)
    ↕ game state / chosen action
Trained Agent (PyTorch)
───────────────────────────────────────────────────────────
```

---

## Existing Assets

| Asset | What it provides |
|-------|-----------------|
| **Catanatron** ([github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron)) | Fast Python game engine, Gymnasium interface, built-in benchmark bots (Random, WeightedRandom, AlphaBeta), 1102 feature extraction, 1v1 support |
| **robottler** ([github.com/meesg/robottler](https://github.com/meesg/robottler)) | Tampermonkey userscript + Python bridge for colonist.io WebSocket interception. Bypasses Cloudflare by running in real browser |
| **Catan-data/dataset** ([github.com/Catan-data/dataset](https://github.com/Catan-data/dataset)) | 43,947 anonymized colonist.io games in JSON (~6.9GB). Full event histories with board state, moves, and outcomes |
| **settlers-rl** ([settlers-rl.github.io](https://settlers-rl.github.io)) | Henry Charlesworth's RL Catan project. Multi-headed action architecture, attention-based observation module, detailed training insights. Key reference for architecture decisions |
| **Dataset parser** | Built and validated. Replays colonist.io JSON through Catanatron, extracts (observation, label) pairs to parquet. 40,917 games parsed into 410 shards (~7.5M train samples). ~6.25 games/sec |
| **Neural value function** | Trained and benchmarked. 176 strategic features, 0.757 test AUC, 48% win rate vs AlphaBeta in 1v1. Checkpoint at `robottler/models/value_net_v2.pt` |
| **Benchmark harness** | Gauntlet script for 1v1 matchups with CI reporting. Tested and validated |

---

## Stage 0: Baselines ✅ COMPLETE

**Time:** 1 day
**Purpose:** Establish performance floor and ceiling for all future comparisons.

### Reporting standard

For every matchup and every future model version:

- Win rate with 95% CI (Wilson score interval)
- Average VP for both sides
- Average game length (turns)
- Sample size (minimum 1000 games for any claim)

### The Gauntlet

Standard benchmark run for every model checkpoint:

```python
def run_gauntlet(agent, num_games=1000):
    """Benchmark agent against all reference bots in 1v1."""
    opponents = {
        "Random": RandomPlayer,
        "WeightedRandom": WeightedRandomPlayer,
        "AlphaBeta": AlphaBetaPlayer,
    }
    for name, opponent_class in opponents.items():
        results = play_1v1(agent, opponent_class, num_games)
        print(f"vs {name}: {results.win_rate:.1%} "
              f"[{results.ci_low:.1%}, {results.ci_high:.1%}] "
              f"| Avg VP: {results.avg_vp:.1f} vs {results.opp_avg_vp:.1f}")
```

### Results tracking table

| Model | vs Random | vs WeightedRandom | vs AlphaBeta | Notes |
|-------|-----------|-------------------|--------------|-------|
| Random | 50.0% | ~35-40% | ~5-15% | Floor |
| WeightedRandom | ~60-65% | 50.0% | ~25-35% | — |
| AlphaBeta | ~85-95% | ~65-75% | 50.0% | Ceiling (hand-tuned) |
| **BC-value-v2** | **98.2%** | **98.2%** | **48.1%** | **Stage 1 ✅** |
| RL-v1 | ? | ? | ? | Stage 2 |
| RL+MCTS-v1 | ? | ? | ? | Stage 3 |

---

## Stage 1: Neural Value Function via Behavioral Cloning ✅ COMPLETE

**Time:** ~1 week (actual)
**Goal:** Replace AlphaBeta's hand-tuned 12-weight value function with a learned one. Same 1-step lookahead framework, better evaluation.

### Step 1: Parse full dataset

Parsed all 44k colonist.io games with enriched feature set (1102 features total, 176 used for training).

```bash
python3 -m robottler.parse_games \
    --input-dir datasets/games/ \
    --output-dir datasets/parquet/ \
    --workers 6
```

**Result:** 40,917 games → 7.5M training samples across 410 parquet shards. ~6.25 games/sec. 67 games skipped (non-4-player).

**Why 4-player data works for 1v1:** Settlement placement quality, resource valuation, build priorities, robber placement, bank trade decisions — these fundamentals transfer across game sizes. The BC model learns "what good Catan looks like" broadly. Stage 2 (self-play in actual 1v1) corrects for format-specific differences.

### Step 2: Train value network

**Architecture:**

```
Input (176 strategic features)
    → BatchNorm
    → Linear(176, 128) → ReLU → Dropout(0.3)
    → Linear(128, 64)  → ReLU → Dropout(0.3)
    → Linear(64, 32)   → ReLU → Dropout(0.3)
    → Linear(32, 1)    → Sigmoid
Output: win probability [0, 1]
```

**Training details (final working configuration):**

- **Target:** `winner` (binary 0/1)
- **Loss:** BCEWithLogitsLoss, **no pos_weight** (pos_weight=3.0 inflated val_loss misleadingly)
- **Features:** 176 strategic features filtered from 1102 (see "Lessons Learned" below)
- **Sampling:** 8 samples per game per epoch, phase-stratified, reshuffled each epoch (~230k samples/epoch)
- **Split:** By game_id — 70% train, 15% val, 15% test (prevents leakage)
- **Optimizer:** AdamW, lr=3e-4, weight_decay=1e-4
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Early stopping:** Patience 10 on val loss
- **Save:** Model weights + feature means/stds + feature names ordering

**Final metrics:**

| Metric | Value |
|--------|-------|
| Test AUC | 0.757 |
| Val AUC | 0.757 |
| Val Accuracy | 0.779 |
| Train/Val Loss Gap | 0.033 |
| Epochs to convergence | ~47 |

### Step 3: Integration and benchmark

The value function plugs directly into Catanatron's `ValueFunctionPlayer` via the `load_value_model()` wrapper. Same 1-step lookahead, swapped evaluation function.

```python
# In bot_interface.py
if self.bot_type == "neural":
    value_fn = load_value_model("robottler/models/value_net_v2.pt")
    player = ValueFunctionPlayer(color)
    player._neural_value_fn = value_fn
```

**Benchmark results (100 games each, 1v1):**

| Matchup | Win Rate | 95% CI | Avg VP | Opp VP | Turns |
|---------|----------|--------|--------|--------|-------|
| Neural vs Random | 98.2% | [96.3%, 100.0%] | 10.0 | 2.3 | 70 |
| Neural vs WeightedRandom | 98.2% | [96.3%, 100.0%] | 10.0 | 2.5 | 69 |
| Neural vs AlphaBeta | 48.1% | [38.5%, 57.7%] | 7.9 | 7.9 | 74 |

**Interpretation:** The neural bot crushes everything below AlphaBeta and plays AlphaBeta to a dead coin flip. The NN trained on human games matches a hand-tuned heuristic — from 0 wins a week prior.

### Lessons Learned (Stage 1)

Three critical discoveries during training that deviated from the original plan:

**1. Feature selection matters more than data volume.**

The original 1102 features included ~800 positional features (NODE_X_P0_SETTLEMENT, EDGE_Y_P1_ROAD, TILE_Z_IS_WHEAT). These uniquely fingerprint board layouts, causing the model to memorize "games that look like this end with player X winning" instead of learning strategy. Scaling from 933 to 44k games with all 1102 features did not fix overfitting — AUC stayed at ~0.63 with massive train/val divergence.

The fix: filter to 176 "strategic" features that describe *what's happening* without revealing *which specific game this is*:

| Feature Category | Count | Example |
|-----------------|-------|---------|
| Production rates | ~40 | `EFFECTIVE_P0_WHEAT_PRODUCTION` |
| Reachability | ~60 | `P0_1_ROAD_REACHABLE_ORE` |
| Player state | ~30 | VP, resources in hand, dev cards, army size, road length |
| Game state | ~20 | Turn number, bank contents, dev cards remaining |
| Port access | ~10 | `P0_HAS_WHEAT_PORT` |
| Aggregate counts | ~16 | Settlements placed, roads built |

**Test:** Could two completely different games produce identical feature values? If yes, the feature describes strategy. If no, it fingerprints the board.

**2. Correlated sampling caps generalization.**

Each game produces ~260 samples (every turn × 4 players), all sharing the same board and outcome. Training on all of them treats 260 correlated data points as independent, causing game-level memorization. Evidence: epoch 1 consistently produced the best val_auc because it was the model's first (least memorized) pass through the data. Every subsequent epoch made val_auc worse.

The fix: sample 8 positions per game per epoch, stratified across early/mid/late game phases, reshuffled each epoch. This gives ~230k decorrelated samples per epoch. Over 200 epochs the model sees diverse positions from each game but never the full correlated trajectory in one batch.

**3. pos_weight inflates val_loss misleadingly.**

With pos_weight=3.0 (compensating for 25% positive rate in 4-player), confident wrong predictions on winners get amplified 3x. This made val_loss appear to diverge catastrophically when the model was actually just becoming more confident (not necessarily more wrong). Removing pos_weight and tracking AUC as the primary metric resolved the diagnostic confusion.

**Results progression:**

| Experiment | Test AUC | Train/Val Gap | Key Change |
|-----------|----------|---------------|------------|
| 1102 features, 933 games | 0.627 | ~0.37 | Baseline |
| 1102 features, 44k games | 0.620 | ~1.28 | More data didn't help |
| 176 features, 44k games | 0.699 | ~0.25 | Feature selection fixed fingerprinting |
| 176 features, 8/game sampling, no pos_weight | 0.757 | 0.033 | Decorrelated sampling + calibration |

### Why not keep the hand-tuned 12 weights?

The 12 weights are a ceiling for hand-engineering, not a ceiling for the problem. They encode broad heuristics (production is good, VP is good, longest road has value). A neural net trained on 44k games can learn:

- **Nonlinear interactions:** "Ore matters more when you have wheat" (city-building synergy)
- **Conditional strategies:** "Dev cards are more valuable when behind on board position"
- **Positional nuance:** Subtle differences in node quality that aren't captured by raw production numbers
- **Phase-dependent weights:** Early-game vs late-game valuation shifts

These are things a linear function structurally cannot represent.

---

## Stage 2: Self-Play Reinforcement Learning ← CURRENT

**Time:** 3-6 weeks
**Goal:** Move from "plays like a decent human" to "plays better than any human." The bot learns from its own experience rather than imitating human play.

### The conceptual shift

Stage 1 trained a value function: "given this board state, who's winning?" That's useful but passive — it copies human judgment including human mistakes.

Stage 2 trains a policy: "given this board state, what should I do?" The bot plays millions of games against itself, discovers what works through trial and error, and develops strategies that humans may never have considered.

### Architecture: Policy-Value Network

Switch from a value-only network to a dual-headed policy-value network:

```
Input (176 strategic features)
    → Shared trunk: 128 → 64 (with BatchNorm, ReLU, Dropout)
    ├── Policy head → 64 → action_space_size → softmax
    │   (probability distribution over legal actions)
    └── Value head → 32 → 1 → sigmoid
        (win probability estimate)
```

**Action space:** In 1v1 without trading, Catanatron's action space is ~290 discrete actions (place settlement at node X, build road on edge Y, buy dev card, play knight on tile Z, etc.). MaskablePPO handles the variable legality — illegal actions are masked out before the softmax.

**Note:** The BC model is value-only. Before starting RL, the policy head must be added to the architecture. The value head initializes from the BC checkpoint; the policy head initializes randomly or from a separately trained BC policy.

### Initialization from BC

Don't start from random weights. This is the key advantage over settlers-rl, whose author identified lack of human data as his biggest handicap.

**Value head:** Initialize from the Stage 1 trained value network (0.757 AUC). On step 1 of RL, the bot already has a strong sense of which positions are good.

**Policy head:** Two options:

1. **Train a separate BC policy model** on the 44k dataset that predicts human actions given game states (a few days of additional work). This means the bot starts RL already playing roughly like a human.
2. **Initialize randomly** and rely on the good value head to guide early exploration.

Option 1 is strongly preferred. It dramatically reduces the number of RL steps needed to reach competent play.

### Training framework

**Library:** [sb3-contrib](https://github.com/Stable-Baselines3/stable-baselines3) MaskablePPO

**Environment:** Catanatron's Gymnasium interface configured for 1v1

```python
import gymnasium
env = gymnasium.make("catanatron/Catanatron-v0", config={
    "num_players": 2,
    # Additional config as needed
})
```

### Self-play curriculum

Training against a fixed opponent leads to overfitting to that opponent's weaknesses. A curriculum of progressively harder opponents prevents this:

| Phase | Steps | Opponents | Purpose |
|-------|-------|-----------|---------|
| A | 0 – 500k | WeightedRandom | Easy wins, validates reward signal works |
| B | 500k – 2M | AlphaBeta | Learns to beat hand-tuned heuristic play |
| C | 2M – 5M | Frozen self checkpoints (50%) + AlphaBeta (25%) + latest self (25%) | Builds robustness against diverse strategies |
| D | 5M+ | Latest self only | Pure self-play, discovers novel strategies |

### Reward function

```python
reward = +1.0 if win else -1.0
```

Start with pure win/loss. In 1v1, the signal is clean — no ambiguity about partial credit. Only add reward shaping if convergence stalls after 2M+ steps. Potential shaping terms (use sparingly):

- Small bonus for gaining VP (+0.01 per VP gained)
- Small penalty for opponent gaining VP (-0.005 per opponent VP)

**Warning:** Reward shaping can create degenerate strategies (e.g., the bot learns to farm VP bonuses instead of winning). Pure win/loss is safer.

### Hyperparameters (starting point)

```python
config = {
    "n_envs": 8,              # Parallel environments (M1 has 8 cores)
    "n_steps": 256,            # Steps per env before update
    "batch_size": 2048,        # n_envs × n_steps
    "n_epochs": 10,            # PPO epochs per batch
    "gamma": 0.999,            # High discount — Catan games are long
    "gae_lambda": 0.95,        # GAE parameter
    "clip_range": 0.2,         # PPO clip
    "ent_coef": 0.01,          # Entropy bonus (prevents premature convergence)
    "vf_coef": 0.5,            # Value loss coefficient
    "learning_rate": 3e-4,     # Standard PPO rate
    "max_grad_norm": 0.5,      # Gradient clipping
}
```

**Value function normalization is critical for stability.** Catan game outcomes are binary (win/loss) but intermediate value estimates need to be well-calibrated for PPO to work.

### Monitoring

Track every 50k steps:

- Win rate vs each reference bot (Random, WeightedRandom, AlphaBeta)
- Win rate vs frozen checkpoints from 500k steps ago
- Average game length
- Policy entropy (should decrease slowly, not collapse)
- Value function accuracy on a held-out set of positions

Save checkpoint every 500k steps. Run the Gauntlet on every saved checkpoint.

### Compute estimates

On M1 Mac:

- Catanatron 1v1 speed: ~6-10 games/sec (faster than 4-player)
- Decisions per game: ~60-80 in 1v1
- Throughput: ~400-600 decisions/sec
- 5M steps: ~2.5-3.5 hours of wall time (simulation only)
- Including training updates: ~6-12 hours per full run

Total Stage 2 time is dominated by iteration — running experiments, checking results, tuning hyperparameters. Budget 3-6 weeks of intermittent work.

### Realistic expectations

| Checkpoint | vs AlphaBeta | Notes |
|------------|--------------|-------|
| BC initialization (step 0) | ~48% | Established in Stage 1 |
| Phase A complete (500k) | 50-60% | Should beat WeightedRandom consistently |
| Phase B complete (2M) | 55-65% | Starting to exploit AlphaBeta's weaknesses |
| Phase C complete (5M) | 65-75% | Robust against diverse strategies |
| Phase D mature (10M+) | 70-80% | Approaching skill ceiling |

---

## Stage 3: Forward Search (MCTS)

**Time:** 2-3 weeks
**Goal:** Push from 70-80% to 85%+ vs AlphaBeta. This is the AlphaZero-style enhancement that settlers-rl showed nearly doubled performance.

### Why search matters

The policy network outputs "what I think I should do" based on pattern recognition. Forward search asks "what actually happens if I do this" by simulating ahead. It catches tactical mistakes the policy would make and finds moves the policy wouldn't consider.

### How it works

At each decision point:

1. **Enumerate legal actions** (Catanatron provides these)
2. **For each candidate action**, simulate forward:
   - Apply the action in a copy of the game state
   - Roll dice randomly for the next few turns
   - Use the value network to evaluate the resulting position
   - Repeat N times (different dice rolls) to get expected value
3. **Pick the action with the highest average expected value**

### settlers-rl's result

From [settlers-rl.github.io](https://settlers-rl.github.io):

> Forward search (simplified MCTS) boosted win rate from 25% to 47% in 4-player games. The author noted this was one of the highest-impact additions.

In 1v1, the impact should be at least as large — the game tree is simpler and the search can go deeper with the same computation budget.

### Implementation options

**Option A: Simple N-step lookahead (recommended first)**

```python
def search_action(game_state, value_fn, legal_actions, n_rollouts=50, depth=3):
    """Evaluate each action by simulating forward."""
    best_action = None
    best_value = -float('inf')

    for action in legal_actions:
        total_value = 0
        for _ in range(n_rollouts):
            sim = copy_game_state(game_state)
            sim.apply(action)
            # Simulate 'depth' more steps with random opponent play
            for _ in range(depth):
                if sim.is_terminal():
                    break
                opp_action = random.choice(sim.legal_actions())
                sim.apply(opp_action)
                # Random dice roll happens inside Catanatron
            total_value += value_fn(sim)
        avg_value = total_value / n_rollouts
        if avg_value > best_value:
            best_value = avg_value
            best_action = action

    return best_action
```

**Option B: MCTS with UCB (higher ceiling, more complex)**

Full Monte Carlo Tree Search with the value network as the evaluation function and the policy network as the prior for node selection. This is what AlphaZero uses and provides the strongest play, but is significantly more complex to implement correctly.

**Recommendation:** Start with Option A. If it meaningfully improves win rate (which settlers-rl suggests it will), then consider MCTS for further gains.

### Speed considerations

Forward search is expensive. For each decision:

- 15 legal actions × 50 rollouts × 3 depth = 2,250 game simulations
- At ~10μs per simulation step in Catanatron: ~23ms per decision
- Acceptable for online play (colonist.io has generous turn timers)

If too slow, reduce rollouts for non-critical decisions (e.g., when there are only 2-3 legal actions, search is cheap anyway).

### Realistic expectations

| Model | vs AlphaBeta |
|-------|--------------|
| RL policy only (no search) | 70-80% |
| RL + simple lookahead (50 rollouts, depth 3) | 80-88% |
| RL + MCTS (if implemented) | 85-92% |

**The ceiling is dice variance.** In 1v1 Catan, even a perfect player loses 15-25% of games because the dice don't cooperate. A bot that achieves 85% vs AlphaBeta is likely playing near-optimally.

---

## Key Lessons from settlers-rl

Henry Charlesworth's project ([settlers-rl.github.io](https://settlers-rl.github.io), [github.com/henrycharlesworth/settlers_of_catan_RL](https://github.com/henrycharlesworth/settlers_of_catan_RL)) provides several directly applicable insights:

### What he built

- Trained a Catan RL agent from scratch (no human data) for ~1 month on RTX 3090
- ~450M decisions total
- Multi-headed action architecture with attention-based observation
- Clear learning progress but did not reach human-level play

### His key takeaways (and how they apply to us)

1. **"No human data was my biggest handicap."**
   We have 44k games. This is our single largest advantage. BC pretraining means our RL doesn't waste millions of steps learning that wheat is useful.

2. **"Trading was a mistake to include early."**
   We're scoping to 1v1 no-trading from the start. This massively simplifies the action space and eliminates the combinatorial explosion that consumed his agent's capacity.

3. **"Forward search nearly doubled win rate."**
   Went from 25% to 47% in 4-player games. This validates Stage 3 as high-impact and worth the implementation effort.

4. **"Multi-headed actions + attention on tiles worked well."**
   His architecture is worth studying but may be overkill for 1v1 no-trading. A simpler MLP policy over Catanatron's flat action space is a reasonable starting point. If the flat approach plateaus, his multi-headed design is a documented fallback.

5. **"Would remove trading if starting over."**
   Further confirmation that our 1v1 no-trading scope is the right call.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| BC value function doesn't generalize from 4-player to 1v1 | Medium | Stage 2 self-play in actual 1v1 corrects format-specific patterns. 4-player data just provides warm start. |
| PPO doesn't converge in self-play | Medium | Start with curriculum (vs fixed bots), not pure self-play. Monitor policy entropy. Fall back to DQN or MCTS-only if PPO fails. |
| Mode collapse (bot learns one narrow strategy) | Medium | Maintain opponent diversity in curriculum. Track action entropy. |
| Catanatron too slow for RL throughput | Low-Medium | Profile early. 1v1 is faster than 4-player. If still slow, vectorize envs or optimize hot paths with Cython. |
| Overfitting to Catanatron's rule implementation | Low | Only matters when deploying to colonist.io. Catanatron's rules are very accurate. Stage 3 fine-tuning on live games addresses edge cases. |
| Forward search too slow for real-time play | Low | colonist.io has generous turn timers. Reduce rollout count for simple decisions. |

---

## Compute Requirements

All stages run on M1 MacBook Pro:

| Stage | Hardware | Time | Notes |
|-------|----------|------|-------|
| 0: Baselines | CPU only | ~30 min | 3 matchups × 1000 games |
| 1: Parse 44k games | CPU, 6 workers | ~2 hours | Parallelized across cores |
| 1: Train value net | MPS (M1 GPU) | ~1-2 hours | 176 features, ~47 epochs |
| 1: Benchmark | CPU only | ~10 min | 1000 1v1 games |
| 2: Self-play RL (5M steps) | CPU + MPS | ~6-12 hours per run | Multiple runs over weeks |
| 3: Forward search | CPU only | Inference time | ~23ms per decision |

**Bottleneck:** Catanatron simulation speed during Stage 2. Profile early. If throughput is insufficient, options include: vectorized environments (SubprocVecEnv), Cython-optimized game logic, or running on a cloud GPU instance for the training loop while simulating locally.

---

## Deployment to colonist.io

Deployment uses the robottler bridge and is independent of training. The trained model runs locally; only WebSocket messages go to colonist.io.

### Architecture

```
colonist.io (browser tab with Tampermonkey)
    ↕ WebSocket interception (msgpack frames)
Tampermonkey Userscript
    ↕ JSON over localhost:5555
Bridge Server (Python)
    - Maintains Catanatron mirror of colonist.io game state
    - Translates colonist.io events → Catanatron state updates
    - Translates agent actions → colonist.io WebSocket messages
    ↕ game state / chosen action
Agent (PyTorch)
    - Policy network (+ optional forward search)
    - Runs inference on CPU/MPS
```

### Coordinate mapping

colonist.io and Catanatron use different coordinate systems. The bridge must translate:

- Corner coords (x, y, z) → Catanatron node IDs
- Edge IDs → Catanatron edge IDs
- Hex coords (x, y) → Catanatron tile IDs

This mapping was validated during dataset parsing. The same translation layer serves both training and deployment.

### Cloudflare bypass

colonist.io uses Cloudflare Turnstile. Automated browser approaches (Playwright, Patchright) fail with Error 600010 (bot detection via Chrome DevTools Protocol). The robottler approach — a Tampermonkey userscript running in a real browser session — bypasses this naturally since Cloudflare sees a genuine browser with a real user session.

---

## Milestone Checklist

### Stage 0: Baselines ✅
- [x] Run all 1v1 matchups (1000 games each)
- [x] Record results in tracking table
- [x] Confirm AlphaBeta is the strongest built-in bot

### Stage 1: Neural Value Function ✅
- [x] Parse 44k games with enriched features (6 workers)
- [x] Discover feature selection > data volume (176 strategic features)
- [x] Implement decorrelated sampling (8 per game per epoch)
- [x] Train value network — test AUC: 0.757
- [x] Benchmark neural ValueFunctionPlayer vs AlphaBeta — 48.1% win rate
- [x] Checkpoint saved: `robottler/models/value_net_v2.pt`

### Stage 2: Self-Play RL
- [ ] Build policy-value network (add policy head to existing value net)
- [ ] (Optional) Train BC policy head on 44k dataset
- [ ] Set up MaskablePPO with Catanatron 1v1 env
- [ ] Phase A: Train vs WeightedRandom (500k steps)
- [ ] Phase B: Train vs AlphaBeta (2M steps)
- [ ] Phase C: Train vs checkpoint pool (5M steps)
- [ ] Phase D: Pure self-play (5M+ steps)
- [ ] Run Gauntlet at each checkpoint
- [ ] Target: 70-80% vs AlphaBeta

### Stage 3: Forward Search
- [ ] Implement simple N-step lookahead
- [ ] Benchmark search-enhanced bot vs AlphaBeta
- [ ] Tune rollout count and depth for speed/quality tradeoff
- [ ] (Optional) Implement full MCTS
- [ ] Target: 85%+ vs AlphaBeta

### Deployment
- [ ] Fork robottler, update bridge for 1v1
- [ ] Validate coordinate mapping on live game
- [ ] Test bot in real colonist.io 1v1 match
- [ ] Iterate on bridge reliability (reconnection, error handling)

---

## References

- **Catanatron:** [github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron) — Game engine, Gymnasium interface, benchmark bots
- **robottler:** [github.com/meesg/robottler](https://github.com/meesg/robottler) — colonist.io WebSocket interception bridge
- **Catan-data/dataset:** [github.com/Catan-data/dataset](https://github.com/Catan-data/dataset) — 43,947 anonymized colonist.io games
- **settlers-rl:** [settlers-rl.github.io](https://settlers-rl.github.io) — Henry Charlesworth's RL Catan agent, architecture docs, training insights
- **settlers-rl code:** [github.com/henrycharlesworth/settlers_of_catan_RL](https://github.com/henrycharlesworth/settlers_of_catan_RL)
- **SB3 MaskablePPO:** [sb3-contrib docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- **Catanatron Gymnasium:** `catanatron/Catanatron-v0` environment
