"""1v1 Gauntlet Benchmark — test any agent against Random, WeightedRandom, AlphaBeta.

Usage:
    # Neural model vs all opponents
    python3 -m robottler.benchmark --model robottler/models/value_net_v2.pt --games 1000

    # RL model vs all opponents
    python3 -m robottler.benchmark --rl-model robottler/models/rl_checkpoints/latest.zip \
        --bc-model robottler/models/value_net_v2.pt --games 1000

    # Search player (neural value + expectimax) vs all opponents
    python3 -m robottler.benchmark --search --bc-model robottler/models/value_net_v2.pt \
        --games 100 --search-depth 2

    # Policy-guided search (neural value + RL pruning + deeper search)
    python3 -m robottler.benchmark --search --bc-model robottler/models/value_net_v2.pt \
        --rl-model robottler/models/rl_checkpoints/rl_value_500000_steps.zip \
        --games 100 --search-depth 3 --top-k 10

    # Baseline matchups only (no model needed)
    python3 -m robottler.benchmark --baselines --games 1000

    # Quick sanity check
    python3 -m robottler.benchmark --baselines --games 20
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import time

from tqdm import tqdm

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points
from catanatron.players.value import base_fn, DEFAULT_WEIGHTS


def wilson_ci(wins, total, z=1.96):
    """Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = wins / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return center, center - spread, center + spread


def make_player(kind, color, rl_model_path=None, bc_model_path=None,
                search_depth=2, top_k=10, value_fn=None, prunning=False,
                rl_model=None, dice_sample_size=None,
                mcts_simulations=800, mcts_c_puct=1.4, policy_fn=None):
    """Create a player of the given kind with the given color."""
    if kind == "random":
        return RandomPlayer(color)
    elif kind == "weighted":
        return WeightedRandomPlayer(color)
    elif kind == "alphabeta":
        return AlphaBetaPlayer(color)
    elif kind == "value":
        return ValueFunctionPlayer(color)
    elif kind == "rl":
        from robottler.rl_player import RLPlayer
        if not rl_model_path or not bc_model_path:
            raise ValueError("rl player requires --rl-model and --bc-model")
        return RLPlayer(color, rl_model_path, bc_model_path)
    elif kind == "search":
        from robottler.search_player import NeuralSearchPlayer
        if not value_fn and not bc_model_path:
            raise ValueError("search player requires --bc-model or pre-loaded value_fn")
        return NeuralSearchPlayer(
            color, bc_path=bc_model_path, depth=search_depth,
            prunning=prunning, value_fn=value_fn,
            dice_sample_size=dice_sample_size,
        )
    elif kind == "policy_search":
        from robottler.search_player import PolicyGuidedSearchPlayer
        if not rl_model_path and not rl_model:
            raise ValueError("policy_search player requires --rl-model or pre-loaded rl_model")
        if not value_fn and not bc_model_path:
            raise ValueError("policy_search player requires --bc-model or pre-loaded value_fn")
        return PolicyGuidedSearchPlayer(
            color, bc_model_path, rl_model_path,
            depth=search_depth, top_k=top_k, prunning=prunning,
            value_fn=value_fn, rl_model=rl_model,
        )
    elif kind == "bb_search":
        from robottler.search_player import BitboardSearchPlayer
        if not value_fn and not bc_model_path:
            raise ValueError("bb_search player requires --bc-model or pre-loaded bb_value_fn")
        return BitboardSearchPlayer(
            color, bc_path=bc_model_path, depth=search_depth,
            prunning=prunning, bb_value_fn=value_fn,
            dice_sample_size=dice_sample_size,
        )
    elif kind == "mcts":
        from robottler.mcts_player import MCTSPlayer
        if not value_fn and not bc_model_path:
            raise ValueError("mcts player requires --bc-model or pre-loaded value_fn")
        return MCTSPlayer(
            color, value_fn=value_fn, num_simulations=mcts_simulations,
            c_puct=mcts_c_puct, policy_fn=policy_fn,
        )
    else:
        raise ValueError(f"Unknown player kind: {kind}")


# ---------------------------------------------------------------------------
# Multiprocessing worker for parallel game execution
# ---------------------------------------------------------------------------

# Per-process globals (set by _worker_init)
_worker_p1_kind = None
_worker_p2_kind = None
_worker_kwargs = None


def _worker_init(p1_kind, p2_kind, kwargs):
    """Initialize per-worker state. Called once per process.

    We re-create value functions per-process since PyTorch models
    can't be pickled across process boundaries.
    """
    global _worker_p1_kind, _worker_p2_kind, _worker_kwargs
    _worker_p1_kind = p1_kind
    _worker_p2_kind = p2_kind

    # Re-build value_fn from paths in each worker (can't pickle torch models)
    kw = dict(kwargs)
    kw.pop("p1_value_fn", None)  # remove unpicklable value_fn
    kw.pop("rl_model", None)
    kw.pop("policy_fn", None)

    # Rebuild value_fn from _rebuild_spec if provided
    rebuild = kw.pop("_rebuild_spec", None)
    if rebuild:
        fn_type = rebuild["type"]
        if fn_type == "blend":
            from robottler.search_player import make_blended_value_fn
            kw["p1_value_fn"] = make_blended_value_fn(
                rebuild["bc_model_path"], blend_weight=rebuild["blend_weight"])
        elif fn_type == "neural":
            from robottler.value_model import load_value_model
            kw["p1_value_fn"] = load_value_model(rebuild["bc_model_path"])
        elif fn_type == "scaled_neural":
            from robottler.search_player import make_scaled_neural_value_fn
            kw["p1_value_fn"] = make_scaled_neural_value_fn(
                rebuild["bc_model_path"], scale=rebuild["blend_weight"])
        elif fn_type == "heuristic":
            kw["p1_value_fn"] = base_fn(DEFAULT_WEIGHTS)
        elif fn_type == "bb_neural":
            from robottler.search_player import make_bb_neural_value_fn
            kw["p1_value_fn"] = make_bb_neural_value_fn(rebuild["bc_model_path"])
        elif fn_type == "bb_blend":
            from robottler.search_player import make_bb_blended_value_fn
            kw["p1_value_fn"] = make_bb_blended_value_fn(
                rebuild["bc_model_path"], blend_weight=rebuild["blend_weight"])
        elif fn_type == "rl_value":
            from robottler.search_player import make_rl_value_fn
            kw["p1_value_fn"] = make_rl_value_fn(
                rebuild["rl_model_path"], rebuild["bc_model_path"])
        elif fn_type == "rl_blend":
            from robottler.search_player import make_rl_blended_value_fn
            kw["p1_value_fn"] = make_rl_blended_value_fn(
                rebuild["rl_model_path"], rebuild["bc_model_path"],
                blend_weight=rebuild["blend_weight"])
        elif fn_type == "mcts_neural":
            from robottler.mcts_player import make_mcts_value_fn
            kw["p1_value_fn"] = make_mcts_value_fn(rebuild["bc_model_path"])
        elif fn_type == "mcts_blend":
            from robottler.mcts_player import make_mcts_blend_value_fn
            kw["p1_value_fn"] = make_mcts_blend_value_fn(
                rebuild["bc_model_path"], blend_weight=rebuild["blend_weight"])
        elif fn_type == "mcts_heuristic":
            from robottler.mcts_player import make_mcts_heuristic_value_fn
            kw["p1_value_fn"] = make_mcts_heuristic_value_fn()

    _worker_kwargs = kw


def _worker_play_one_game(_game_idx):
    """Play a single game in a worker process. Returns (winner_color, vp_red, vp_blue, turns)."""
    kw = _worker_kwargs
    p1 = make_player(_worker_p1_kind, Color.RED,
                     rl_model_path=kw.get("rl_model_path"),
                     bc_model_path=kw.get("bc_model_path"),
                     search_depth=kw.get("search_depth", 2),
                     top_k=kw.get("top_k", 10),
                     value_fn=kw.get("p1_value_fn"),
                     prunning=kw.get("prunning", False),
                     rl_model=kw.get("rl_model"),
                     dice_sample_size=kw.get("dice_sample_size"),
                     mcts_simulations=kw.get("mcts_simulations", 800),
                     mcts_c_puct=kw.get("mcts_c_puct", 1.4),
                     policy_fn=kw.get("policy_fn"))
    p2 = make_player(_worker_p2_kind, Color.BLUE)

    if kw.get("p1_value_fn") is not None and hasattr(p1, "_neural_value_fn") and _worker_p1_kind == "value":
        p1._neural_value_fn = kw["p1_value_fn"]

    game = Game([p1, p2], vps_to_win=kw.get("vps_to_win", 10))
    game.play()

    winner = game.winning_color()
    vp_red = get_actual_victory_points(game.state, Color.RED)
    vp_blue = get_actual_victory_points(game.state, Color.BLUE)
    turns = game.state.num_turns
    return (str(winner), vp_red, vp_blue, turns)


def run_matchup(p1_kind, p2_kind, num_games, p1_value_fn=None, vps_to_win=10,
                rl_model_path=None, bc_model_path=None,
                search_depth=2, top_k=10, prunning=False, rl_model=None,
                dice_sample_size=None, mcts_simulations=800, mcts_c_puct=1.4,
                policy_fn=None, workers=1, rebuild_spec=None):
    """Run num_games 1v1 games and return stats.

    workers: number of parallel processes (1 = sequential, no multiprocessing overhead).
    rebuild_spec: dict describing how to reconstruct value_fn in workers (for multiprocessing).

    Returns dict with: p1_wins, p2_wins, draws, p1_vps, p2_vps, turns, total
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0
    p1_vps = []
    p2_vps = []
    turn_counts = []

    label = f"{p1_kind} vs {p2_kind}"

    if workers <= 1:
        # Sequential path — no multiprocessing overhead
        for _ in tqdm(range(num_games), desc=f"  {label}", leave=False):
            p1 = make_player(p1_kind, Color.RED, rl_model_path, bc_model_path,
                             search_depth=search_depth, top_k=top_k,
                             value_fn=p1_value_fn, prunning=prunning,
                             rl_model=rl_model, dice_sample_size=dice_sample_size,
                             mcts_simulations=mcts_simulations, mcts_c_puct=mcts_c_puct,
                             policy_fn=policy_fn)
            p2 = make_player(p2_kind, Color.BLUE)

            if p1_value_fn is not None and hasattr(p1, "_neural_value_fn") and p1_kind == "value":
                p1._neural_value_fn = p1_value_fn

            game = Game([p1, p2], vps_to_win=vps_to_win)
            game.play()

            winner = game.winning_color()
            vp_red = get_actual_victory_points(game.state, Color.RED)
            vp_blue = get_actual_victory_points(game.state, Color.BLUE)
            turns = game.state.num_turns

            p1_vps.append(vp_red)
            p2_vps.append(vp_blue)
            turn_counts.append(turns)

            if winner is None or turns >= TURNS_LIMIT:
                draws += 1
            elif winner == Color.RED:
                p1_wins += 1
            else:
                p2_wins += 1
    else:
        # Parallel path
        init_kwargs = {
            "bc_model_path": bc_model_path,
            "rl_model_path": rl_model_path,
            "search_depth": search_depth,
            "top_k": top_k,
            "prunning": prunning,
            "dice_sample_size": dice_sample_size,
            "mcts_simulations": mcts_simulations,
            "mcts_c_puct": mcts_c_puct,
            "vps_to_win": vps_to_win,
        }
        if rebuild_spec:
            init_kwargs["_rebuild_spec"] = rebuild_spec

        with mp.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(p1_kind, p2_kind, init_kwargs),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker_play_one_game, range(num_games)),
                total=num_games, desc=f"  {label}", leave=False,
            ))

        for winner_str, vp_red, vp_blue, turns in results:
            p1_vps.append(vp_red)
            p2_vps.append(vp_blue)
            turn_counts.append(turns)
            if winner_str == "None" or turns >= TURNS_LIMIT:
                draws += 1
            elif winner_str == str(Color.RED):
                p1_wins += 1
            else:
                p2_wins += 1

    return {
        "p1_kind": p1_kind,
        "p2_kind": p2_kind,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "total": num_games,
        "vps_to_win": vps_to_win,
        "p1_avg_vp": sum(p1_vps) / len(p1_vps) if p1_vps else 0,
        "p2_avg_vp": sum(p2_vps) / len(p2_vps) if p2_vps else 0,
        "avg_turns": sum(turn_counts) / len(turn_counts) if turn_counts else 0,
        "draw_rate": draws / num_games if num_games else 0,
    }


def print_result(result):
    """Print a single matchup result line."""
    wr, lo, hi = wilson_ci(result["p1_wins"], result["total"] - result["draws"])
    p1 = result["p1_kind"]
    p2 = result["p2_kind"]
    draws_str = f" | Draws: {result['draws']}" if result["draws"] > 0 else ""
    print(
        f"  {p1:15s} vs {p2:15s}: "
        f"{wr*100:5.1f}% [{lo*100:.1f}%, {hi*100:.1f}%] | "
        f"VP: {result['p1_avg_vp']:.1f} vs {result['p2_avg_vp']:.1f} | "
        f"{result['avg_turns']:.0f} turns{draws_str}"
    )


def save_csv(results, path):
    """Append gauntlet results to CSV."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "p1", "p2", "games", "vps_to_win",
                "p1_wins", "p2_wins", "draws",
                "win_rate", "ci_lo", "ci_hi",
                "p1_avg_vp", "p2_avg_vp", "avg_turns",
            ])
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            decided = r["total"] - r["draws"]
            wr, lo, hi = wilson_ci(r["p1_wins"], decided)
            writer.writerow([
                ts, r["p1_kind"], r["p2_kind"], r["total"],
                r["vps_to_win"],
                r["p1_wins"], r["p2_wins"], r["draws"],
                f"{wr:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                f"{r['p1_avg_vp']:.2f}", f"{r['p2_avg_vp']:.2f}",
                f"{r['avg_turns']:.1f}",
            ])


def run_gauntlet(model_path, num_games, vps_to_win=10):
    """Run neural model against all opponents."""
    from robottler.value_model import load_value_model

    print(f"Loading model from {model_path}...")
    value_fn = load_value_model(model_path)

    matchups = [
        ("value", "random"),
        ("value", "weighted"),
        ("value", "alphabeta"),
    ]

    results = []
    print(f"\nGauntlet Results ({num_games} games each, 1v1, {vps_to_win} VP)")
    print("=" * 75)

    for p1_kind, p2_kind in matchups:
        result = run_matchup(p1_kind, p2_kind, num_games, p1_value_fn=value_fn,
                             vps_to_win=vps_to_win)
        results.append(result)
        print_result(result)

    return results


def run_rl_gauntlet(rl_model_path, bc_model_path, num_games, vps_to_win=10):
    """Run RL model against all opponents."""
    print(f"Loading RL model from {rl_model_path}...")

    matchups = [
        ("rl", "random"),
        ("rl", "weighted"),
        ("rl", "alphabeta"),
    ]

    results = []
    print(f"\nRL Gauntlet Results ({num_games} games each, 1v1, {vps_to_win} VP)")
    print("=" * 75)

    for p1_kind, p2_kind in matchups:
        result = run_matchup(p1_kind, p2_kind, num_games, vps_to_win=vps_to_win,
                             rl_model_path=rl_model_path, bc_model_path=bc_model_path)
        results.append(result)
        print_result(result)

    return results


def run_search_gauntlet(bc_model_path, num_games, vps_to_win=10,
                        rl_model_path=None, search_depth=2, top_k=10,
                        eval_mode="heuristic", blend_weight=1e10,
                        use_policy=False, dice_sample_size=None, workers=1):
    """Run search player against all opponents.

    eval_mode controls the leaf evaluation function:
      - "heuristic": hand-tuned heuristic (AlphaBeta baseline with our search infra)
      - "neural": pure BC neural value function (unscaled)
      - "scaled_neural": neural * blend_weight (tests if neural just needs scaling)
      - "blend": heuristic + blend_weight * neural
      - "rl_value": RL model's PPO value head
      - "rl_blend": heuristic + blend_weight * RL value head

    use_policy: if True, use PolicyGuidedSearchPlayer (RL policy pruning + search).

    Pre-loads value functions and RL model once and shares across all games.
    """
    from robottler.search_player import (
        make_scaled_neural_value_fn,
        make_blended_value_fn, make_rl_value_fn, make_rl_blended_value_fn,
    )
    from robottler.value_model import load_value_model

    label_parts = [f"depth={search_depth}", f"eval={eval_mode}"]

    # Build rebuild_spec for multiprocessing (workers reconstruct value_fn from paths)
    rebuild_spec = {"type": eval_mode, "bc_model_path": bc_model_path,
                    "blend_weight": blend_weight, "rl_model_path": rl_model_path}

    if eval_mode == "heuristic":
        heuristic_fn = base_fn(DEFAULT_WEIGHTS)
        value_fn = heuristic_fn
    elif eval_mode == "neural":
        print(f"Loading BC model from {bc_model_path}...")
        value_fn = load_value_model(bc_model_path)
    elif eval_mode == "scaled_neural":
        print(f"Loading scaled neural (scale={blend_weight:.0e}) from {bc_model_path}...")
        value_fn = make_scaled_neural_value_fn(bc_model_path, scale=blend_weight)
        label_parts.append(f"scale={blend_weight:.0e}")
    elif eval_mode == "blend":
        print(f"Loading blend (weight={blend_weight:.0e}) from {bc_model_path}...")
        value_fn = make_blended_value_fn(bc_model_path, blend_weight=blend_weight)
        label_parts.append(f"weight={blend_weight:.0e}")
    elif eval_mode == "rl_value":
        if not rl_model_path:
            raise ValueError("rl_value eval_mode requires --rl-model")
        print(f"Loading RL value head from {rl_model_path}...")
        value_fn = make_rl_value_fn(rl_model_path, bc_model_path)
    elif eval_mode == "rl_blend":
        if not rl_model_path:
            raise ValueError("rl_blend eval_mode requires --rl-model")
        print(f"Loading RL blend (weight={blend_weight:.0e}) from {rl_model_path}...")
        value_fn = make_rl_blended_value_fn(rl_model_path, bc_model_path,
                                             blend_weight=blend_weight)
        label_parts.append(f"weight={blend_weight:.0e}")
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    # Pre-load RL model once for policy-guided search
    rl_model = None
    if use_policy:
        if not rl_model_path:
            raise ValueError("Policy-guided search requires --rl-model")
        from sb3_contrib import MaskablePPO
        print(f"Loading RL policy from {rl_model_path}...")
        rl_model = MaskablePPO.load(rl_model_path)
        p1_kind = "policy_search"
        label_parts.append(f"top_k={top_k}")
        label = f"PolicySearch ({', '.join(label_parts)})"
    else:
        p1_kind = "search"
        label = f"Search ({', '.join(label_parts)})"

    opponents = ["random", "weighted", "value", "alphabeta"]

    results = []
    if workers > 1:
        label_parts.append(f"workers={workers}")
    print(f"\n{label} Gauntlet ({num_games} games each, 1v1, {vps_to_win} VP)")
    print("=" * 80)

    if dice_sample_size is not None:
        label_parts.append(f"dice={dice_sample_size}")

    for p2 in opponents:
        result = run_matchup(
            p1_kind, p2, num_games, p1_value_fn=value_fn, vps_to_win=vps_to_win,
            rl_model_path=rl_model_path, bc_model_path=bc_model_path,
            search_depth=search_depth, top_k=top_k, rl_model=rl_model,
            dice_sample_size=dice_sample_size,
            workers=workers, rebuild_spec=rebuild_spec,
        )
        results.append(result)
        print_result(result)

    return results


def run_mcts_gauntlet(bc_model_path, num_games, vps_to_win=10,
                      rl_model_path=None, mcts_simulations=800,
                      mcts_c_puct=1.4, eval_mode="neural",
                      blend_weight=1e8, use_policy=False, workers=1):
    """Run MCTS player against all opponents.

    eval_mode controls the value function:
      - "neural": pure BC neural value function (already [0,1])
      - "blend": sigmoid-normalized heuristic + blend_weight * neural
      - "heuristic": sigmoid-normalized heuristic only

    use_policy: if True, use RL policy for MCTS priors (requires --rl-model).
    """
    from robottler.mcts_player import (
        MCTSPlayer, make_mcts_value_fn, make_mcts_blend_value_fn,
        make_mcts_heuristic_value_fn, make_policy_fn,
    )

    label_parts = [f"sims={mcts_simulations}", f"c_puct={mcts_c_puct}", f"eval={eval_mode}"]

    # Build rebuild_spec for multiprocessing
    rebuild_type = f"mcts_{eval_mode}"
    rebuild_spec = {"type": rebuild_type, "bc_model_path": bc_model_path,
                    "blend_weight": blend_weight}

    if eval_mode == "neural":
        print(f"Loading BC model from {bc_model_path}...")
        value_fn = make_mcts_value_fn(bc_model_path)
    elif eval_mode == "blend":
        print(f"Loading blend (weight={blend_weight:.0e}) from {bc_model_path}...")
        value_fn = make_mcts_blend_value_fn(bc_model_path, blend_weight=blend_weight)
        label_parts.append(f"weight={blend_weight:.0e}")
    elif eval_mode == "heuristic":
        value_fn = make_mcts_heuristic_value_fn()
    else:
        raise ValueError(f"Unknown MCTS eval_mode: {eval_mode}")

    policy_fn = None
    if use_policy:
        if not rl_model_path:
            raise ValueError("MCTS policy priors require --rl-model")
        print(f"Loading RL policy from {rl_model_path}...")
        policy_fn = make_policy_fn(rl_model_path, bc_model_path)
        label_parts.append("policy=RL")

    label = f"MCTS ({', '.join(label_parts)})"
    opponents = ["random", "weighted", "value", "alphabeta"]

    results = []
    if workers > 1:
        label_parts.append(f"workers={workers}")
    print(f"\n{label} Gauntlet ({num_games} games each, 1v1, {vps_to_win} VP)")
    print("=" * 80)

    for p2 in opponents:
        result = run_matchup(
            "mcts", p2, num_games, p1_value_fn=value_fn, vps_to_win=vps_to_win,
            rl_model_path=rl_model_path, bc_model_path=bc_model_path,
            mcts_simulations=mcts_simulations, mcts_c_puct=mcts_c_puct,
            policy_fn=policy_fn,
            workers=workers, rebuild_spec=rebuild_spec,
        )
        results.append(result)
        print_result(result)

    return results


def run_baselines(num_games, vps_to_win=10, workers=1):
    """Run reference matchups between built-in players."""
    matchups = [
        ("random", "random"),
        ("weighted", "random"),
        ("alphabeta", "random"),
        ("alphabeta", "weighted"),
    ]

    results = []
    print(f"\nBaseline Results ({num_games} games each, 1v1, {vps_to_win} VP)")
    print("=" * 75)

    for p1_kind, p2_kind in matchups:
        result = run_matchup(p1_kind, p2_kind, num_games, vps_to_win=vps_to_win,
                             workers=workers)
        results.append(result)
        print_result(result)

    return results


def run_bb_search_gauntlet(bc_model_path, num_games, vps_to_win=10,
                           search_depth=2, eval_mode="neural",
                           blend_weight=1e10, dice_sample_size=None, workers=1):
    """Run bitboard search player against all opponents.

    eval_mode controls the leaf evaluation function:
      - "neural": pure BC neural value function (default for bitboard)
      - "blend": heuristic + blend_weight * neural
    """
    from robottler.search_player import make_bb_neural_value_fn, make_bb_blended_value_fn

    label_parts = [f"depth={search_depth}", f"eval={eval_mode}"]

    # Build rebuild_spec for multiprocessing
    rebuild_type = "bb_neural" if eval_mode == "neural" else "bb_blend"
    rebuild_spec = {"type": rebuild_type, "bc_model_path": bc_model_path,
                    "blend_weight": blend_weight}

    if eval_mode == "neural":
        print(f"Loading BB neural value fn from {bc_model_path}...")
        value_fn = make_bb_neural_value_fn(bc_model_path)
    elif eval_mode == "blend":
        print(f"Loading BB blend (weight={blend_weight:.0e}) from {bc_model_path}...")
        value_fn = make_bb_blended_value_fn(bc_model_path, blend_weight=blend_weight)
        label_parts.append(f"weight={blend_weight:.0e}")
    else:
        raise ValueError(f"Unsupported eval_mode for bb_search: {eval_mode}")

    label = f"BB-Search ({', '.join(label_parts)})"
    opponents = ["random", "weighted", "value", "alphabeta"]

    if dice_sample_size is not None:
        label_parts.append(f"dice={dice_sample_size}")

    results = []
    if workers > 1:
        label_parts.append(f"workers={workers}")
    print(f"\n{label} Gauntlet ({num_games} games each, 1v1, {vps_to_win} VP)")
    print("=" * 80)

    for p2 in opponents:
        result = run_matchup(
            "bb_search", p2, num_games, p1_value_fn=value_fn, vps_to_win=vps_to_win,
            bc_model_path=bc_model_path,
            search_depth=search_depth,
            dice_sample_size=dice_sample_size,
            workers=workers, rebuild_spec=rebuild_spec,
        )
        results.append(result)
        print_result(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="1v1 Gauntlet Benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to neural model checkpoint (.pt)")
    parser.add_argument("--rl-model", type=str, default=None,
                        help="Path to RL model checkpoint (.zip)")
    parser.add_argument("--bc-model", type=str, default="robottler/models/value_net_v2.pt",
                        help="Path to BC checkpoint for RL normalization (default: value_net_v2.pt)")
    parser.add_argument("--search", action="store_true",
                        help="Run search player gauntlet (requires --bc-model)")
    parser.add_argument("--bb-search", action="store_true",
                        help="Run bitboard search player gauntlet (requires --bc-model)")
    parser.add_argument("--policy-search", action="store_true",
                        help="Use RL policy pruning for search (requires --rl-model)")
    parser.add_argument("--search-depth", type=int, default=2,
                        help="Search depth for search player (default: 2)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K actions for policy-guided search (default: 10)")
    parser.add_argument("--eval-mode", type=str, default="heuristic",
                        choices=["heuristic", "neural", "scaled_neural", "blend", "rl_value", "rl_blend"],
                        help="Leaf evaluation function for search (default: heuristic)")
    parser.add_argument("--blend-weight", type=float, default=1e10,
                        help="Weight for neural/RL signal in blend/scaled modes (default: 1e10)")
    parser.add_argument("--dice-sample", type=int, default=None,
                        help="Sample top-N dice outcomes instead of all 11 (e.g. 5)")
    parser.add_argument("--mcts", action="store_true",
                        help="Run MCTS player gauntlet (requires --bc-model)")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per decision (default: 800)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                        help="MCTS exploration constant (default: 1.4)")
    parser.add_argument("--mcts-policy", action="store_true",
                        help="Use RL policy priors for MCTS (requires --rl-model)")
    parser.add_argument("--baselines", action="store_true",
                        help="Run baseline matchups (no model needed)")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of games per matchup (default: 1000)")
    parser.add_argument("--vps", type=int, default=10,
                        help="Victory points to win (default: 10)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes per matchup (default: 1)")
    args = parser.parse_args()

    if not args.model and not args.rl_model and not args.search and not args.bb_search and not args.policy_search and not args.mcts and not args.baselines:
        parser.error("Specify --model, --rl-model, --search, --bb-search, --policy-search, --mcts, or --baselines (or combine)")

    if args.workers > 1:
        import torch
        if torch.cuda.is_available():
            mp.set_start_method("spawn", force=True)
        else:
            mp.set_start_method("fork", force=True)
        print(f"Using {args.workers} parallel workers per matchup")

    all_results = []
    start = time.time()

    if args.model:
        all_results.extend(run_gauntlet(args.model, args.games, vps_to_win=args.vps))

    if args.search or args.policy_search:
        all_results.extend(run_search_gauntlet(
            args.bc_model, args.games, vps_to_win=args.vps,
            rl_model_path=args.rl_model,
            search_depth=args.search_depth, top_k=args.top_k,
            eval_mode=args.eval_mode, blend_weight=args.blend_weight,
            use_policy=args.policy_search,
            dice_sample_size=args.dice_sample,
            workers=args.workers))

    if args.bb_search:
        all_results.extend(run_bb_search_gauntlet(
            args.bc_model, args.games, vps_to_win=args.vps,
            search_depth=args.search_depth,
            eval_mode=args.eval_mode, blend_weight=args.blend_weight,
            dice_sample_size=args.dice_sample,
            workers=args.workers))
    elif args.rl_model:
        all_results.extend(run_rl_gauntlet(
            args.rl_model, args.bc_model, args.games, vps_to_win=args.vps))

    if args.mcts:
        # MCTS supports neural/blend/heuristic eval modes
        mcts_eval = args.eval_mode if args.eval_mode in ("neural", "blend", "heuristic") else "neural"
        all_results.extend(run_mcts_gauntlet(
            args.bc_model, args.games, vps_to_win=args.vps,
            rl_model_path=args.rl_model,
            mcts_simulations=args.simulations, mcts_c_puct=args.c_puct,
            eval_mode=mcts_eval, blend_weight=args.blend_weight,
            use_policy=args.mcts_policy,
            workers=args.workers))

    if args.baselines:
        all_results.extend(run_baselines(args.games, vps_to_win=args.vps,
                                         workers=args.workers))

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.0f}s")

    if all_results:
        csv_path = os.path.join(os.path.dirname(__file__), "models", "gauntlet_results.csv")
        save_csv(all_results, csv_path)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
