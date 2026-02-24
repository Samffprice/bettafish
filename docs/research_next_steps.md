# Research: Breaking the ~55% MCTS Ceiling

After exhaustively ruling out training-objective fixes (calibration exp #23, ranking loss exp #24, distillation exp #15, self-play exp #22), we researched what techniques exist for making AI superhuman in stochastic games like Catan.

## The Core Problem

A 325K-param MLP doing 1-ply evaluation cannot match depth-2 alpha-beta with hand-crafted heuristic. We've proven this is architectural, not a training problem. Both agents converged on the same insight: **stop trying to fix the neural value function — instead inject the strong heuristic directly into MCTS search.**

---

## Tier 1: High Impact, Directly Applicable

### 1. MCTS-Minimax Hybrid (Minimax at MCTS Leaves)
**Source**: Baier & Winands (2015), "MCTS-Minimax Hybrids"

Replace neural value evaluation at MCTS leaf nodes with depth-1 or depth-2 minimax search using the blend function. MCTS provides broad exploration + stochastic handling, minimax provides accurate local evaluation.

**Implementation**: Modify `BBMCTSPlayer._evaluate_and_expand()` to call `bb_search()` depth-1 instead of neural forward pass.

**Expected impact**: High. Directly addresses the ranking problem — minimax gives perfect local ranking. Main cost is speed (~100µs neural → ~10-50ms depth-1 minimax), so fewer sims per second, but each sim is much more accurate.

**Trade-off**: 800 sims × 100µs = 80ms/move with neural. Depth-1 minimax ~10ms × 100 sims = 1s/move. Still playable.

### 2. Implicit Minimax Backups (IMMCTS)
**Source**: Lanctot et al., "Monte Carlo Tree Search with Heuristic Evaluations"

Store TWO value estimates at each MCTS node: (a) neural/MCTS average, (b) minimax-backed heuristic value. Back up heuristic values via minimax operators (max at our nodes, min at opponent nodes). Blend both in UCB formula.

**Expected impact**: 10-20% win rate improvement in tactical games. Retains MCTS averaging for stochastic elements while getting minimax accuracy for tactical decisions.

### 3. Gumbel AlphaZero / Sequential Halving
**Source**: Danihelka et al. (2022), "Policy improvement by planning with Gumbel"

Replace PUCT action selection with Gumbel top-k sequential halving. Optimizes for **simple regret** (finding best action) instead of cumulative regret (exploring all actions). Provably better with fixed simulation budget.

**Relevance**: We found no improvement from 400→800 sims — the allocation strategy may be the bottleneck, not total sims. Gumbel AZ guarantees improvement even with just 2 simulations.

**Implementation**: Moderate. Replace UCB selection at root with sequential halving.

### 4. Heterogeneous GNN / Cross-Dimensional Network
**Source**: Gendre & Kaneko (2020), "Playing Catan with Cross-dimensional Neural Network"

Our MLP flattens the entire board into a 176-dim vector, losing all spatial structure. A GNN would process tiles, vertices, and edges as separate node types with message passing along the actual board topology.

Gendre & Kaneko's cross-dimensional network was the **first RL agent to beat JSettlers** in Catan. It processes 0D (global stats), 1D (edges/vertices), and 2D (tiles) separately with cross-dimensional attention.

**Expected impact**: Potentially transformative for value function quality. The MLP's ranking failure (tau=0.44) may largely stem from inability to reason about spatial relationships.

**Implementation**: High effort. Requires new architecture, new feature encoding, retraining from scratch.

---

## Tier 2: Medium Impact, Worth Trying

### 5. ChanceProbCut (Speed Up Depth-3 Search)
**Source**: Schadd et al. (2009)

Forward-prune chance nodes statistically unlikely to affect the decision. Uses shallow search to predict deep search values at chance nodes, prunes if correlation is high enough.

**Relevance**: Our depth-3 BBSearch already hits 84% vs AB but takes ~54s/game. ChanceProbCut + Star2 could bring this down to ~15-20s, making depth-3 practical for live play.

**Implementation**: Moderate. Add statistical pruning inside the expectimax chance-node loop.

### 6. TD(lambda) Self-Play Value Learning
**Source**: Tesauro (1995), TD-Gammon

In stochastic games, TD learning from self-play produces better value functions than supervised training on fixed datasets. TD-Gammon proved this: stochasticity actually *helps* TD learning by providing natural exploration.

**Key insight**: We've only done supervised training (MSE on expert labels). TD learning with temporal difference targets may produce value functions that generalize better, avoiding the "lower loss ≠ better play" disconnect.

**Implementation**: Moderate. Play games with current network, compute TD(lambda) returns, update weights online.

### 7. Expert Iteration with Policy-Only Training
**Source**: Anthony et al. (2017), "Thinking Fast and Slow with Deep Learning and Tree Search"

Abandon value training entirely. Train ONLY a policy network to mimic the minimax expert's action choices. Use MCTS with just policy priors and outcome averaging (no value head).

**Rationale**: Removes the bottleneck of training a weak value function. The policy doesn't need to rank states — it just needs to put high probability on the expert's chosen action.

**Trade-off**: Loses value-guided search. MCTS would rely on rollout averaging, which is noisy for Catan's long games (~80 turns).

### 8. Minimax-Initialized Node Priors (MCTS-IP-M)
**Source**: Baier & Winands (2015)

Pre-compute depth-1 minimax values for each child action to initialize MCTS visit counts and values. Instead of starting from zero, each child starts with an accurate estimate.

**Relevance**: The strongest hybrid variant in Baier & Winands experiments. Since the core problem is ranking, giving MCTS accurate initial rankings from minimax may be enough.

---

## Tier 3: Lower Priority / Higher Effort

### 9. Stochastic MuZero
Learn a dynamics model instead of using the simulator. Lower priority since we have a perfect simulator, but the afterstate representation concept (evaluate *after* our action, *before* dice) could help.

### 10. AlphaZeroES (Evolution Strategies)
Direct score maximization via ES instead of loss minimization. Addresses "lower loss ≠ better play" but requires many games per gradient update (expensive).

### 11. Transformers / Attention over Board
Attention-based architecture for board evaluation. Lower priority than GNN since board has explicit graph structure.

---

## Recommended Implementation Order

1. **MCTS-Minimax Hybrid** (Tier 1.1) — Fastest to implement, highest confidence. Just swap the leaf evaluator in `_evaluate_and_expand()`. Benchmark immediately.

2. **Gumbel AlphaZero** (Tier 1.3) — Independent of value function quality. Can be combined with any evaluator. Moderate implementation effort.

3. **ChanceProbCut for depth-3** (Tier 2.5) — If MCTS-minimax works well, making depth-3 search faster is the natural next step for the alpha-beta player.

4. **GNN Architecture** (Tier 1.4) — Highest effort but potentially highest payoff. Do this after proving the MCTS-minimax hybrid concept works.

---

## Key Papers

- Baier & Winands (2015), "MCTS-Minimax Hybrids" — hybrid search variants
- Gendre & Kaneko (2020), "Playing Catan with Cross-dimensional Neural Network" — first RL agent to beat JSettlers
- Danihelka et al. (2022), "Policy improvement by planning with Gumbel" — Gumbel AlphaZero
- Schadd et al. (2009), "ChanceProbCut" — chance node pruning
- Tesauro (1995), "Temporal Difference Learning and TD-Gammon" — TD self-play in stochastic games
- Szita, Chaslot, Spronck (2009), "Monte-Carlo Tree Search in Settlers of Catan" — MCTS struggles with long horizons
- Driss & Cazenave (2022), "Deep Catan" — ExIt applied to Catan with CNN
- Lanctot et al., "Monte Carlo Tree Search with Heuristic Evaluations" — implicit minimax backups
- Anthony et al. (2017), "Thinking Fast and Slow with Deep Learning and Tree Search" — Expert Iteration
