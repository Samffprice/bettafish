# CATAN AI — Project Scope Document (v2)

**Author:** Sam / Abide Robotics
**Date:** February 2026
**Status:** Draft
**Goal:** Build an AI agent that beats high-ELO players on colonist.io

---

## 1. Vision

Build a Catan AI agent capable of consistently beating high-ELO players on colonist.io. Rather than building from scratch, we combine two existing open-source projects — **robottler** (colonist.io browser bridge) and **Catanatron** (Catan game engine) — with a custom RL training layer on top.

---

## 2. Existing Projects

### robottler ([github.com/meesg/robottler](https://github.com/meesg/robottler))

A working colonist.io bot built with a Tampermonkey userscript + Python backend. Already handles:

- WebSocket interception via userscript (`@run-at document-start` patches WS before colonist.io loads)
- Game state parsing from msgpack WS frames
- Legal action enumeration and execution (settlements, roads, cities, bank trades, robber)
- Naive bot logic (weighted random with basic heuristics)
- Turn lifecycle management

Known limitations (from their TODO):

- WS interception is flaky (~50/50 on whether it runs before colonist.io's JS)
- Bot logic is tightly coupled to colonist.io message format
- No player-to-player trading
- No dev card play beyond basic knight
- No long-term planning

### Catanatron ([github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron))

A full Catan game engine in Python, purpose-built for AI research. Provides:

- Complete rules implementation (board gen, resource production, building, dev cards, longest road, largest army, robber, ports, VP tracking)
- Fast game simulation for self-play
- Built-in benchmark bots (RandomPlayer, WeightedRandomPlayer, heuristic ValueFunction bot)
- State representation suitable for ML
- Game replay and analysis tools

The author documented five approaches to Catan AI (random, weighted random, MCTS, supervised value function, AlphaZero-style RL) and got decent results with heuristic search but not superhuman play.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────┐
│                  colonist.io                     │
│              (real browser + CF)                  │
└────────────────┬───────────────┬────────────────┘
                 │ WS frames     │ WS frames
                 ▼               ▲
┌─────────────────────────────────────────────────┐
│          Tampermonkey Userscript                  │
│      (intercepts WS, forwards to local server)   │
└────────────────┬───────────────┬────────────────┘
                 │ JSON/msgpack  │ actions
                 ▼               ▲
┌─────────────────────────────────────────────────┐
│            Bridge Server (Python)                 │
│  - Parses colonist.io state → internal format    │
│  - Translates agent actions → WS messages        │
│  - Handles game lifecycle + error recovery       │
└────────────────┬───────────────┬────────────────┘
                 │ game state    │ chosen action
                 ▼               ▲
┌─────────────────────────────────────────────────┐
│              Agent (Python)                       │
│  - Trained on Catanatron via self-play RL        │
│  - Policy + value network (PyTorch)              │
│  - Action masking for legal moves                │
└─────────────────────────────────────────────────┘

        ┌──── Training Loop (offline) ────┐
        │                                  │
        │   Catanatron Engine              │
        │   (millions of self-play games)  │
        │          ↕                       │
        │   RL Training (PPO / MCTS+NN)   │
        │          ↓                       │
        │   Trained Model Checkpoint       │
        │                                  │
        └──────────────────────────────────┘
```

The key insight: **training happens entirely in Catanatron** (fast, headless, no browser needed). The **bridge only exists for deployment** — plugging the trained agent into live colonist.io games.

---

## 4. Success Criteria

- **V1:** Fork robottler, stabilize the WS bridge, and play a complete colonist.io bot game with Catanatron's heuristic bot as the brain (no RL yet)
- **V2:** Train an RL agent via self-play in Catanatron that beats Catanatron's best heuristic bot >60% of the time
- **V3:** Deploy the RL agent through the bridge and achieve >50% win rate against colonist.io's built-in bots
- **V4:** Achieve >1500 ELO on colonist.io ranked play
- **Stretch:** Top-100 ELO on colonist.io leaderboard

---

## 5. Phased Roadmap

### Phase 1: Bridge + Heuristic Bot (V1)

**Goal:** Get robottler's bridge working reliably with Catanatron's heuristic bot as the decision maker.

**Estimated effort:** 1–2 weeks

#### 1a. Fork and stabilize robottler

- Fix the WS interception reliability issue (their `@run-at document-start` sometimes loses the race with colonist.io's JS — may need to explore alternative injection methods or add a connection retry)
- Document the full colonist.io msgpack protocol from captured game data
- Map every message type to a game event (board setup, dice, resource distribution, builds, trades, robber, dev cards, game end)

#### 1b. State translation layer

- Build a translator between colonist.io's WS message format and Catanatron's internal game state representation
- This is the core integration work — robottler speaks "colonist.io protocol" and Catanatron speaks "Catanatron game state," and we need a clean adapter between them
- Handle the gaps: colonist.io has hidden information (opponent hands), Catanatron may assume perfect info for certain operations

#### 1c. Action translation layer

- Map Catanatron's action output format to the WS messages colonist.io expects
- Validate that every action type round-trips correctly: settlement placement, road building, city upgrade, dev card purchase/play, bank trade, robber movement, discard

#### 1d. Integration test

- Run Catanatron's existing `WeightedRandomPlayer` or `ValueFunctionPlayer` as the brain through the full pipeline: colonist.io → userscript → bridge → Catanatron bot → bridge → userscript → colonist.io
- Play 10+ complete games vs colonist.io bots without crashes
- Log full game transcripts for debugging

> **Deliverable:** A working pipeline where Catanatron's heuristic bot plays complete games on colonist.io via the browser bridge. Full game logs captured.

---

### Phase 2: RL Training in Catanatron (V2)

**Goal:** Train an RL agent through self-play that outperforms Catanatron's existing heuristic bots.

**Estimated effort:** 4–6 weeks

#### 2a. Training environment setup

- Wrap Catanatron's game engine as a Gymnasium (OpenAI Gym) environment
- Define observation space: compact tensor encoding of board state (~1,300 features — see State Representation below)
- Define action space with masking: policy outputs over all possible actions, illegal actions masked to zero
- Implement reward function: +1 for win, -1 for loss. Optional intermediate shaping for VP milestones.

#### 2b. Self-play training

- Start with PPO (via stable-baselines3) for fast iteration
- 4-player self-play: all seats controlled by copies of the agent, with periodic opponent pool updates to prevent overfitting to one playstyle
- Training target: >10,000 games/hour. Profile Catanatron's speed — if too slow, optimize hot paths or consider Cython bindings
- Track metrics: win rate vs heuristic bots, average VP, game length, resource efficiency

#### 2c. Evaluation

- Benchmark against Catanatron's existing bots (RandomPlayer, WeightedRandomPlayer, ValueFunctionPlayer)
- Run 1,000+ game evaluation sets for statistical significance
- If PPO plateaus, explore AlphaZero-style MCTS + neural network approach

#### 2d. Trading policy

- Structured trade offers: agent proposes (offer_type, offer_amount, request_type, request_amount)
- Opponents accept/reject based on their own policy during self-play
- Bank/port trades included from the start
- Player-to-player trades added once basic play is strong

> **Deliverable:** Trained model checkpoint that beats Catanatron's best heuristic bot >60% of the time in 4-player games.

---

### Phase 3: Live Deployment (V3)

**Goal:** Deploy the trained agent on colonist.io and validate against real bots.

**Estimated effort:** 2–3 weeks

- Swap Catanatron's heuristic bot for the trained RL model in the bridge pipeline
- Handle inference latency — model should respond in <500ms to not trigger timeout
- Add human-like timing: variable delays between actions (1–5s), occasional pauses
- Run 50+ games against colonist.io bots, measure win rate
- Debug edge cases where colonist.io's rules implementation differs from Catanatron's

> **Deliverable:** RL agent winning >50% of games against colonist.io bots through the live bridge.

---

### Phase 4: Ranked Play (V4)

**Goal:** Compete against real players and climb ELO.

**Estimated effort:** 4–6 weeks (ongoing)

- OAuth login (Google/Discord) with saved browser session for ranked access
- Opponent modeling: adapt trade/robber strategy based on observed opponent behavior
- Online learning: feed live game transcripts back into training pipeline
- Anti-detection: natural timing variance, avoid suspiciously optimal patterns
- Monitoring: track win rate, ELO progression, game logs, common failure modes

---

## 6. State Representation

(Shared between Catanatron training and live play)

| Feature Group | Encoding | Dimensions |
|--------------|----------|------------|
| Board topology | 19 hexes: resource type (one-hot 6), number token (one-hot 11), robber (1) | 342 |
| Intersections | 54 vertices: building type, owner (one-hot 4), port type | 540 |
| Edges (roads) | 72 edges: has road (1), owner (one-hot 4) | 360 |
| Own resources | 5 resource counts (normalized) | 5 |
| Opponent info | Per opponent: total cards, VP, dev cards played, army size, road length | 24 |
| Game phase | Turn number, phase, dice roll, dev cards in hand | ~20 |

Total: ~1,300 features. Small enough for a 3–5 layer network (256–512 units).

---

## 7. Action Space

| Action Type | Parameters | Max Options |
|------------|-----------|-------------|
| Place settlement | Vertex ID (0–53) | ~54 |
| Place road | Edge ID (0–71) | ~72 |
| Upgrade to city | Vertex ID of own settlement | ~5 |
| Buy dev card | None | 1 |
| Play dev card | Card type + params | ~10 |
| Propose trade | Offer + request resource/amount | ~100 |
| Accept/reject trade | Binary | 2 |
| Move robber | Hex ID + steal target | ~57 |
| Discard | Resource combination | Variable |
| Bank/port trade | Ratio + resource types | ~25 |
| End turn | None | 1 |

Illegal actions masked at each step. Policy network outputs over full space, masked before sampling.

---

## 8. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Game engine | Catanatron (fork) | Proven, fast, built for AI research |
| Browser bridge | robottler (fork) | Working colonist.io WS intercept |
| WS interception | Tampermonkey userscript | Bypasses Cloudflare (runs in real browser) |
| Bridge server | Python (FastAPI or raw WebSocket) | Connects userscript to agent |
| ML framework | PyTorch | Flexibility for custom architectures |
| RL library | stable-baselines3 (PPO) or custom MCTS | SB3 for fast iteration |
| Training env | Gymnasium wrapper around Catanatron | Standard RL interface |
| Monitoring | Weights & Biases | Training curves, game metrics |
| Data storage | SQLite (game logs), filesystem (checkpoints) | Simple, no infra overhead |

---

## 9. Why Not Playwright?

We tried. Colonist.io uses Cloudflare Turnstile with aggressive bot detection:

- Standard Playwright: blocked immediately (Error 600010)
- Patchright (undetected Playwright fork): also blocked (same 600010)
- Cloudflare detects Chrome DevTools Protocol usage, which all Playwright-based tools rely on
- Camoufox + residential proxies might work but adds cost and complexity

The userscript approach avoids all of this — your real browser session handles Cloudflare naturally, and the userscript operates within the already-authenticated page context.

---

## 10. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Colonist.io protocol changes break bridge | High | Abstract the translation layer. Protocol is msgpack with readable keys — changes should be easy to detect and fix. |
| Account ban for bot play | High | Use alt accounts for testing. Consider research partnership with colonist.io team. |
| Catanatron's rules differ from colonist.io | Medium | Validate with side-by-side game logs. Fix Catanatron or add colonist.io-specific patches. |
| WS interception race condition (robottler's known issue) | Medium | Explore MutationObserver or service worker approaches. Add connection retry logic. |
| RL training doesn't converge | Medium | Start with imitation learning from Catanatron's heuristic bot. Use reward shaping. Compare PPO vs MCTS. |
| State translation loses information | Medium | Build comprehensive test suite comparing colonist.io states with Catanatron equivalents. |
| Game engine too slow for self-play | Low | Catanatron is already optimized. Profile and add Cython if needed. |

---

## 11. Open Questions

1. How complete is robottler's protocol documentation? Need to capture fresh games and compare against their parser to find gaps.
2. How closely does Catanatron's rules implementation match colonist.io? Edge cases around longest road, robber on 7, dev card timing.
3. Does Catanatron support 3-player games? Colonist.io bot games are often 4-player but need to confirm.
4. What's Catanatron's simulation speed? Need >10k games/hour for viable self-play training.
5. Should we contribute improvements upstream to both repos, or maintain private forks?
6. Compute budget for training — single GPU (RTX 3080/4090) likely sufficient for PPO.

---

## 12. First Steps (This Week)

1. **Capture a game:** Install the Tampermonkey WS capture script, play a bot game on colonist.io, save the frame dump.
2. **Decode the protocol:** Write a Python script to decode the msgpack frames and catalog every message type.
3. **Fork both repos:** Clone robottler and Catanatron, get both running locally.
4. **Map the gap:** Compare robottler's game state format with Catanatron's internal representation. Document what the translation layer needs to handle.
5. **Proof of concept:** Get Catanatron's `WeightedRandomPlayer` making one valid move through the bridge → colonist.io pipeline.


Best_game replay : https://colonist.io/replay?gameId=208954305&playerColor=5