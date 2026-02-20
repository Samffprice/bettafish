"""Search players: AlphaBeta search with pluggable value functions.

NeuralSearchPlayer     — pure neural value fn (BC net) as leaf evaluator
BlendedSearchPlayer    — hand-tuned heuristic + neural blend
RLValueSearchPlayer    — PPO value head as leaf evaluator
PolicyGuidedSearchPlayer — any of above + RL policy pruning for deeper search
"""

import time

import numpy as np
import torch

from catanatron.models.enums import ActionType
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.tree_search_utils import execute_spectrum
from catanatron.players.value import base_fn, get_value_fn, DEFAULT_WEIGHTS
from robottler.value_model import load_value_model
from robottler.zobrist import ZobristTracker, TranspositionTable, EXACT, LOWERBOUND, UPPERBOUND

# Action ordering priority for alpha-beta pruning (lower = try first).
# Best moves first → tighter bounds → more cutoffs.
_ACTION_ORDER = {
    ActionType.BUILD_CITY: 0,
    ActionType.BUILD_SETTLEMENT: 1,
    ActionType.BUY_DEVELOPMENT_CARD: 2,
    ActionType.PLAY_KNIGHT_CARD: 3,
    ActionType.PLAY_MONOPOLY: 4,
    ActionType.PLAY_YEAR_OF_PLENTY: 5,
    ActionType.PLAY_ROAD_BUILDING: 6,
    ActionType.BUILD_ROAD: 7,
    ActionType.MARITIME_TRADE: 8,
    ActionType.END_TURN: 9,
    ActionType.ROLL: 10,
    ActionType.DISCARD: 10,
    ActionType.MOVE_ROBBER: 10,
}

def _action_sort_key(action):
    return _ACTION_ORDER.get(action.action_type, 10)


POSITIONAL_PREFIXES = ("NODE", "EDGE", "TILE", "PORT")


# ---------------------------------------------------------------------------
# Helper: build RL feature extractor (shared by RLValue and PolicyGuided)
# ---------------------------------------------------------------------------

def _load_feature_stats(bc_path):
    """Load strategic feature names and normalization stats from BC checkpoint."""
    from catanatron.features import get_feature_ordering

    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    bc_feature_names = ckpt["feature_names"]
    bc_means = ckpt["feature_means"]
    bc_stds = ckpt["feature_stds"]

    all_2p_features = get_feature_ordering(num_players=2)
    strategic_names = [
        f for f in all_2p_features
        if not any(f.startswith(p) for p in POSITIONAL_PREFIXES)
    ]

    bc_col_indices = np.array(
        [bc_feature_names.index(f) for f in strategic_names], dtype=np.intp
    )
    means = bc_means[bc_col_indices].astype(np.float32)
    stds = bc_stds[bc_col_indices].astype(np.float32)
    return strategic_names, means, stds


def _extract_obs(game, color, strategic_names, means, stds):
    """Extract normalized 98-dim strategic observation from game state."""
    from catanatron.features import create_sample_vector
    vec = create_sample_vector(game, color, strategic_names)
    x = np.array(vec, dtype=np.float32)
    return (x - means) / stds


# ---------------------------------------------------------------------------
# Value function factories (used by benchmark to pre-load once)
# ---------------------------------------------------------------------------

def make_scaled_neural_value_fn(bc_path, scale=1e10):
    """Create a scaled neural value function: scale * neural(game, color).

    Tests whether the neural net was fine all along — just needed its tiny
    [0.001, 0.18] output range amplified to work with search comparisons.
    """
    neural_fn = load_value_model(bc_path)

    def value_fn(game, p0_color):
        return scale * neural_fn(game, p0_color)

    return value_fn


def make_blended_value_fn(bc_path, blend_weight=1e10):
    """Create a blended value function: heuristic + blend_weight * neural.

    The heuristic provides strong local signal (VP-dominated, ~3e14 per VP).
    The neural net provides strategic tiebreaking (~1e10 * 0.1 = 1e9).
    """
    neural_fn = load_value_model(bc_path)
    heuristic_fn = base_fn(DEFAULT_WEIGHTS)

    def value_fn(game, p0_color):
        h = heuristic_fn(game, p0_color)
        n = neural_fn(game, p0_color)
        return h + blend_weight * n

    return value_fn


def make_rl_value_fn(rl_model_path, bc_path):
    """Create a value function from the RL model's value head.

    The PPO value head was trained via advantage estimation during self-play,
    predicting expected return from each state. Returns values in ~[-1, 1].
    """
    from sb3_contrib import MaskablePPO
    rl_model = MaskablePPO.load(rl_model_path)
    strategic_names, means, stds = _load_feature_stats(bc_path)

    def value_fn(game, p0_color):
        obs = _extract_obs(game, p0_color, strategic_names, means, stds)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            val = rl_model.policy.predict_values(obs_t)
        return val.item()

    return value_fn


def make_rl_blended_value_fn(rl_model_path, bc_path, blend_weight=1e10):
    """Create a blended value function: heuristic + blend_weight * RL_value_head.

    The RL value head returns values in [-1, 1], better calibrated than the
    BC net for local state differences.
    """
    from sb3_contrib import MaskablePPO
    rl_model = MaskablePPO.load(rl_model_path)
    strategic_names, means, stds = _load_feature_stats(bc_path)
    heuristic_fn = base_fn(DEFAULT_WEIGHTS)

    def value_fn(game, p0_color):
        h = heuristic_fn(game, p0_color)
        obs = _extract_obs(game, p0_color, strategic_names, means, stds)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rl_val = rl_model.policy.predict_values(obs_t)
        return h + blend_weight * rl_val.item()

    return value_fn


# ---------------------------------------------------------------------------
# Player classes
# ---------------------------------------------------------------------------

class NeuralSearchPlayer(AlphaBetaPlayer):
    """AlphaBeta search with a custom value function as leaf evaluator.

    Accepts any value_fn(game, p0_color) -> float. Reuses all existing
    search logic (alpha-beta pruning, expand_spectrum, probabilistic outcomes).

    Automatically deepens search in two critical phases:
    - Endgame (any player VP >= vps_to_win - 2): depth + 2 (min 4) for 3-4p,
      depth + 1 (min 3) for 1v1 (tighter turn timer)
    - Opening (initial build phase): depth + 2 (min 4)
    """

    def __init__(self, color, bc_path=None, depth=2, prunning=True, value_fn=None,
                 dice_sample_size=None, use_tt=False):
        super().__init__(color, depth=depth, prunning=prunning)
        if value_fn is not None:
            self._neural_value_fn = value_fn
        elif bc_path is not None:
            self._neural_value_fn = load_value_model(bc_path)
        else:
            raise ValueError("Either bc_path or value_fn must be provided")
        self.use_value_function = True
        self.dice_sample_size = dice_sample_size
        self.use_tt = use_tt
        self._tt = None  # created fresh per decide() call
        self._tt_stats = None  # stats from last search

    def value_function(self, game, p0_color):
        return self._neural_value_fn(game, p0_color)

    def decide(self, game, playable_actions):
        from catanatron.state_functions import get_actual_victory_points

        original_depth = self.depth
        try:
            my_vp = get_actual_victory_points(game.state, self.color)
            opp_colors = [c for c in game.state.colors if c != self.color]
            opp_vp = max(get_actual_victory_points(game.state, c) for c in opp_colors)
            endgame_threshold = game.vps_to_win - 2
            endgame = my_vp >= endgame_threshold or opp_vp >= endgame_threshold

            is_1v1 = len(game.state.colors) == 2

            if endgame:
                if is_1v1:
                    # 1v1 has ~40s turn timer shared across multiple decisions;
                    # cap at depth 3 to stay responsive
                    self.depth = max(original_depth + 1, 3)
                else:
                    self.depth = max(original_depth + 2, 4)
            elif game.state.is_initial_build_phase:
                self.depth = max(original_depth + 2, 4)

            # Create fresh TT per decision
            if self.use_tt:
                self._tt = TranspositionTable()
            else:
                self._tt = None

            result = super().decide(game, playable_actions)

            if self._tt is not None:
                self._tt_stats = {
                    'hits': self._tt.hits,
                    'misses': self._tt.misses,
                    'stores': self._tt.stores,
                    'size': len(self._tt),
                }
                self._tt = None  # free memory

            return result
        finally:
            self.depth = original_depth

    def _tt_compute_hash(self, game):
        """Compute Zobrist hash for TT lookup. Lazy: creates tracker on the fly."""
        tracker = ZobristTracker()
        tracker.init(game)
        return tracker.zobrist_hash

    def alphabeta(self, game, depth, alpha, beta, deadline):
        """Optimized AlphaBeta with lazy expansion, Star2 chance-node pruning,
        and optional transposition table.

        Lazy expansion: calls execute_spectrum per-action instead of
        expand_spectrum for all actions upfront. Actions pruned by alpha-beta
        never have their outcomes computed (~60-70% fewer game copies).

        Star2 pruning: within a single action's chance outcomes, sorts by
        probability (descending) and prunes remaining outcomes when bounds
        show they can't affect the result.

        TT: when use_tt=True, probes/stores positions keyed by Zobrist hash.
        TT hits at sufficient depth return cached scores. Even on TT miss,
        the stored best action is tried first for better move ordering.
        """
        if depth == 0 or game.winning_color() is not None or time.time() >= deadline:
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
            )
            return None, value_fn(game, self.color)

        # --- TT Probe ---
        tt = self._tt
        zhash = None
        tt_best_action = None
        if tt is not None:
            zhash = self._tt_compute_hash(game)
            hit, tt_score, tt_best_action = tt.probe(zhash, depth, alpha, beta)
            if hit:
                return tt_best_action, tt_score

        maximizing = game.state.current_color() == self.color
        actions = self.get_actions(game)
        actions.sort(key=_action_sort_key)

        # --- Move ordering: try TT best action first ---
        if tt_best_action is not None and tt_best_action in actions:
            actions = [tt_best_action] + [a for a in actions if a != tt_best_action]

        if maximizing:
            best_action = None
            best_value = float("-inf")
            for action in actions:
                outcomes = execute_spectrum(game, action, self.dice_sample_size)
                # Sort by probability descending for Star2 effectiveness
                outcomes.sort(key=lambda x: x[1], reverse=True)

                expected_value = 0.0
                consumed_p = 0.0
                obs_max = float("-inf")
                obs_min = float("inf")
                pruned = False

                for outcome, proba in outcomes:
                    result = self.alphabeta(outcome, depth - 1, alpha, beta, deadline)
                    v = result[1]
                    expected_value += proba * v
                    consumed_p += proba
                    obs_max = max(obs_max, v)
                    obs_min = min(obs_min, v)

                    remaining_p = 1.0 - consumed_p
                    if remaining_p < 1e-9:
                        break

                    # Star2: prune if remaining outcomes can't beat alpha
                    if consumed_p >= 0.3:
                        upper = expected_value + remaining_p * obs_max
                        if upper <= alpha:
                            expected_value = upper
                            pruned = True
                            break

                if expected_value > best_value:
                    best_action = action
                    best_value = expected_value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break

            # --- TT Store ---
            if tt is not None and zhash is not None:
                if best_value <= alpha:
                    flag = UPPERBOUND
                elif best_value >= beta:
                    flag = LOWERBOUND
                else:
                    flag = EXACT
                tt.store(zhash, depth, best_value, flag, best_action)

            return best_action, best_value
        else:
            best_action = None
            best_value = float("inf")
            for action in actions:
                outcomes = execute_spectrum(game, action, self.dice_sample_size)
                outcomes.sort(key=lambda x: x[1], reverse=True)

                expected_value = 0.0
                consumed_p = 0.0
                obs_max = float("-inf")
                obs_min = float("inf")
                pruned = False

                for outcome, proba in outcomes:
                    result = self.alphabeta(outcome, depth - 1, alpha, beta, deadline)
                    v = result[1]
                    expected_value += proba * v
                    consumed_p += proba
                    obs_max = max(obs_max, v)
                    obs_min = min(obs_min, v)

                    remaining_p = 1.0 - consumed_p
                    if remaining_p < 1e-9:
                        break

                    # Star2: prune if remaining outcomes can't go below beta
                    if consumed_p >= 0.3:
                        lower = expected_value + remaining_p * obs_min
                        if lower >= beta:
                            expected_value = lower
                            pruned = True
                            break

                if expected_value < best_value:
                    best_action = action
                    best_value = expected_value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

            # --- TT Store ---
            if tt is not None and zhash is not None:
                if best_value >= beta:
                    flag = LOWERBOUND
                elif best_value <= alpha:
                    flag = UPPERBOUND
                else:
                    flag = EXACT
                tt.store(zhash, depth, best_value, flag, best_action)

            return best_action, best_value

    def __repr__(self):
        return (
            f"NeuralSearchPlayer({self.color}, depth={self.depth}, prunning={self.prunning})"
        )


class PolicyGuidedSearchPlayer(NeuralSearchPlayer):
    """Search player + RL policy for action pruning.

    Uses the RL policy to rank actions and only searches the top-K most
    promising. This prunes the search tree dramatically, allowing deeper
    search within the same time budget.
    """

    def __init__(self, color, bc_path, rl_model_path, depth=3, top_k=10,
                 prunning=True, value_fn=None, rl_model=None):
        super().__init__(color, bc_path=bc_path, depth=depth, prunning=prunning,
                         value_fn=value_fn)

        if rl_model is not None:
            self._rl_model = rl_model
        else:
            from sb3_contrib import MaskablePPO
            self._rl_model = MaskablePPO.load(rl_model_path)

        self._strategic_names, self._means, self._stds = _load_feature_stats(bc_path)
        self.top_k = top_k

    def _extract_obs(self, game):
        return _extract_obs(game, self.color, self._strategic_names,
                            self._means, self._stds)

    def _compute_mask(self, actions):
        """Build boolean action mask from a list of Catanatron actions."""
        from catanatron.gym.envs.catanatron_env import to_action_space, ACTION_SPACE_SIZE
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for action in actions:
            try:
                idx = to_action_space(action)
                mask[idx] = True
            except (ValueError, IndexError):
                pass
        return mask

    def _get_action_probs(self, obs, mask):
        """Get action probabilities from RL policy with masking."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            dist = self._rl_model.policy.get_distribution(obs_t, action_masks=mask_t)
            probs = dist.distribution.probs.squeeze(0).numpy()
        return probs

    def get_actions(self, game):
        """Override to prune actions by RL policy probability (our turn only)."""
        base_actions = super().get_actions(game)

        # Only prune our own actions; leave opponent actions untouched
        if game.state.current_color() != self.color:
            return base_actions
        if len(base_actions) <= self.top_k:
            return base_actions

        obs = self._extract_obs(game)
        mask = self._compute_mask(base_actions)
        probs = self._get_action_probs(obs, mask)

        from catanatron.gym.envs.catanatron_env import to_action_space
        scored = []
        for action in base_actions:
            try:
                idx = to_action_space(action)
                scored.append((probs[idx], action))
            except (ValueError, IndexError):
                scored.append((0.0, action))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [a for _, a in scored[:self.top_k]]

    def __repr__(self):
        return (
            f"PolicyGuidedSearchPlayer({self.color}, depth={self.depth}, "
            f"top_k={self.top_k}, prunning={self.prunning})"
        )


# ---------------------------------------------------------------------------
# Bitboard Search Player
# ---------------------------------------------------------------------------

_BB_DETERMINISTIC_ACTIONS = None

def _bb_execute_spectrum(bb_state, action, max_dice_outcomes=None):
    """Bitboard version of execute_spectrum.

    Returns [(bb_copy, proba), ...] for stochastic outcomes.
    """
    from collections import Counter
    import math
    from catanatron.models.enums import (
        DEVELOPMENT_CARDS, RESOURCES, ActionType, Action, ActionRecord,
    )
    from catanatron.models.map import number_probability
    from robottler.bitboard.actions import bb_apply_action
    from robottler.bitboard.convert import DEV_ID_TO_NAME
    from robottler.bitboard.state import (
        PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND,
        PS_RESOURCE_START, PS_RESOURCE_END,
    )

    global _BB_DETERMINISTIC_ACTIONS
    if _BB_DETERMINISTIC_ACTIONS is None:
        _BB_DETERMINISTIC_ACTIONS = {
            ActionType.END_TURN, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD,
            ActionType.BUILD_CITY, ActionType.PLAY_KNIGHT_CARD,
            ActionType.PLAY_YEAR_OF_PLENTY, ActionType.PLAY_ROAD_BUILDING,
            ActionType.MARITIME_TRADE, ActionType.DISCARD, ActionType.PLAY_MONOPOLY,
        }

    if action.action_type in _BB_DETERMINISTIC_ACTIONS:
        copy = bb_state.copy()
        bb_apply_action(copy, action)
        return [(copy, 1)]

    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        remaining = 25 - bb_state.dev_deck_idx
        if remaining <= 0:
            copy = bb_state.copy()
            bb_apply_action(copy, action)
            return [(copy, 1)]

        _DEV_HAND_PS = {
            "KNIGHT": PS_KNIGHT_HAND, "YEAR_OF_PLENTY": PS_YOP_HAND,
            "MONOPOLY": PS_MONO_HAND, "ROAD_BUILDING": PS_RB_HAND,
            "VICTORY_POINT": PS_VP_HAND,
        }

        card_counts = Counter()
        for idx in range(bb_state.dev_deck_idx, 25):
            card_counts[DEV_ID_TO_NAME[int(bb_state.dev_deck[idx])]] += 1
        pidx = bb_state.color_to_index[action.color]
        for opp in range(bb_state.num_players):
            if opp == pidx:
                continue
            for card in DEVELOPMENT_CARDS:
                card_counts[card] += int(bb_state.player_state[opp, _DEV_HAND_PS[card]])

        total = sum(card_counts.values())
        if total == 0:
            copy = bb_state.copy()
            bb_apply_action(copy, action)
            return [(copy, 1)]

        results = []
        for card in card_counts:
            option_action = Action(action.color, action.action_type, card)
            copy = bb_state.copy()
            try:
                bb_apply_action(copy, option_action)
            except Exception:
                pass
            results.append((copy, card_counts[card] / total))
        return results

    elif action.action_type == ActionType.ROLL:
        results = []
        for roll in range(2, 13):
            outcome = (roll // 2, math.ceil(roll / 2))
            option_action = Action(action.color, action.action_type, outcome)
            copy = bb_state.copy()
            bb_apply_action(copy, option_action)
            results.append((copy, number_probability(roll)))

        if max_dice_outcomes is not None and len(results) > max_dice_outcomes:
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:max_dice_outcomes]
            total_p = sum(p for _, p in results)
            results = [(s, p / total_p) for s, p in results]
        return results

    elif action.action_type == ActionType.MOVE_ROBBER:
        (coordinate, robbed_color) = action.value
        if robbed_color is None:
            copy = bb_state.copy()
            bb_apply_action(copy, action)
            return [(copy, 1)]

        robbed_pidx = bb_state.color_to_index[robbed_color]
        total_res = int(np.sum(
            bb_state.player_state[robbed_pidx, PS_RESOURCE_START:PS_RESOURCE_END]
        ))
        if total_res == 0:
            copy = bb_state.copy()
            bb_apply_action(copy, action)
            return [(copy, 1)]

        results = []
        for card in RESOURCES:
            option_action = Action(action.color, action.action_type, (coordinate, robbed_color))
            option_record = ActionRecord(action=option_action, result=card)
            copy = bb_state.copy()
            try:
                bb_apply_action(copy, option_action, option_record)
            except Exception:
                pass
            results.append((copy, 1 / 5.0))
        return results

    else:
        raise RuntimeError(f"Unknown ActionType {action.action_type}")


def make_bb_neural_value_fn(bc_path):
    """Create a neural value function that operates on BitboardState.

    Uses direct vector extraction (FeatureIndexer) and zero-copy tensor creation
    for optimized inference. Uses ONNX Runtime when available (~3x faster than PyTorch).
    """
    from robottler.bitboard.features import FeatureIndexer, bb_fill_feature_vector

    checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    feature_names = checkpoint["feature_names"]
    n_features = len(feature_names)
    feature_index_map = {name: idx for idx, name in enumerate(feature_names)}
    means_np = checkpoint["feature_means"].astype(np.float32)
    stds_np = checkpoint["feature_stds"].astype(np.float32)

    from robottler.value_model import CatanValueNet
    model = CatanValueNet(input_dim=n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Pre-allocate reusable buffer
    input_buf = np.zeros(n_features, dtype=np.float32)
    fi = None  # lazy-init FeatureIndexer (needs catan_map from first call)

    def value_fn(bb_state, p0_color):
        nonlocal fi
        if fi is None:
            fi = FeatureIndexer(feature_index_map, bb_state.catan_map)
        else:
            fi.update_map(bb_state.catan_map)
        input_buf[:] = 0.0
        bb_fill_feature_vector(bb_state, p0_color, input_buf, fi)
        np.subtract(input_buf, means_np, out=input_buf)
        np.divide(input_buf, stds_np, out=input_buf)
        x = torch.from_numpy(input_buf).unsqueeze(0)
        with torch.inference_mode():
            return torch.sigmoid(model(x)).item()

    return value_fn


def _bb_heuristic(bb_state, p0_color, node_prod=None):
    """Bitboard port of base_fn heuristic from catanatron/players/value.py.

    Args:
        node_prod: Optional precomputed [54, 5] node production array.
            If None, computes from catan_map.node_production (slower).
    """
    from robottler.bitboard.features import (
        compute_production_np, compute_reachability_np,
    )
    from robottler.bitboard.state import (
        PS_VP, PS_SETTLE_AVAIL, PS_CITY_AVAIL, PS_ROADS_AVAIL,
        PS_WOOD, PS_BRICK, PS_SHEEP, PS_WHEAT, PS_ORE,
        PS_LONGEST_ROAD, PS_PLAYED_KNIGHT,
        PS_KNIGHT_HAND, PS_YOP_HAND, PS_MONO_HAND, PS_RB_HAND, PS_VP_HAND,
        PS_RESOURCE_START, PS_RESOURCE_END,
    )
    from robottler.bitboard.masks import (
        NUM_TILES, TILE_NODES, NODE_BIT, RESOURCE_INDEX, popcount64, bitscan,
    )

    # Lazy-build node_prod if not provided
    if node_prod is None:
        node_prod = np.zeros((54, 5), dtype=np.float64)
        for nid, counter in bb_state.catan_map.node_production.items():
            for resource, prob in counter.items():
                node_prod[nid, RESOURCE_INDEX[resource]] = prob

    p0_idx = bb_state.color_to_index[p0_color]
    ps = bb_state.player_state[p0_idx]
    p1_idx = (p0_idx + 1) % bb_state.num_players

    params = DEFAULT_WEIGHTS

    # Production (numpy — no dict)
    prod_0 = compute_production_np(bb_state, p0_idx, node_prod, consider_robber=True)
    prod_1 = compute_production_np(bb_state, p1_idx, node_prod, consider_robber=True)

    proba_point = 2.778 / 100
    TRANSLATE_VARIETY = 3

    prod_sum = float(np.sum(prod_0))
    prod_variety = int(np.count_nonzero(prod_0)) * TRANSLATE_VARIETY * proba_point
    production = prod_sum + prod_variety
    enemy_production = float(np.sum(prod_1))

    # Reachability (numpy — no dict, only levels 0 and 1)
    reach = compute_reachability_np(bb_state, p0_idx, node_prod, max_level=1)
    reach_0 = float(np.sum(reach[0]))
    reach_1 = float(np.sum(reach[1]))

    # Hand synergy
    wheat = int(ps[PS_WHEAT])
    ore = int(ps[PS_ORE])
    sheep = int(ps[PS_SHEEP])
    brick = int(ps[PS_BRICK])
    wood = int(ps[PS_WOOD])
    dist_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
    dist_settle = (max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)) / 4.0
    hand_synergy = (2 - dist_city - dist_settle) / 2

    num_in_hand = int(np.sum(ps[PS_RESOURCE_START:PS_RESOURCE_END]))
    discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0

    # Owned tiles count
    owned_nodes = list(bitscan(bb_state.settlement_bb[p0_idx] | bb_state.city_bb[p0_idx]))
    owned_tiles = set()
    for node in owned_nodes:
        for tid in range(NUM_TILES):
            if TILE_NODES[tid] & NODE_BIT[node]:
                owned_tiles.add(tid)
    num_tiles = len(owned_tiles)

    # Buildable nodes count
    buildable = bb_state.board_buildable & bb_state.reachable_bb[p0_idx]
    num_buildable_nodes = popcount64(buildable)
    longest_road_factor = params["longest_road"] if num_buildable_nodes == 0 else 0.1

    longest_road_length = int(ps[PS_LONGEST_ROAD])
    num_dev_cards = int(ps[PS_KNIGHT_HAND] + ps[PS_YOP_HAND] + ps[PS_MONO_HAND] +
                        ps[PS_RB_HAND] + ps[PS_VP_HAND])
    knights_played = int(ps[PS_PLAYED_KNIGHT])

    return float(
        int(ps[PS_VP]) * params["public_vps"]
        + production * params["production"]
        + enemy_production * params["enemy_production"]
        + reach_0 * params["reachable_production_0"]
        + reach_1 * params["reachable_production_1"]
        + hand_synergy * params["hand_synergy"]
        + num_buildable_nodes * params["buildable_nodes"]
        + num_tiles * params["num_tiles"]
        + num_in_hand * params["hand_resources"]
        + discard_penalty
        + longest_road_length * longest_road_factor
        + num_dev_cards * params["hand_devs"]
        + knights_played * params["army_size"]
    )


def make_bb_blended_value_fn(bc_path, blend_weight=1e10):
    """Create a blended value function for BitboardState: heuristic + weight * neural.

    Shares FeatureIndexer and node_prod between neural and heuristic evaluation.
    Uses ONNX Runtime when available (~3x faster than PyTorch).
    """
    from robottler.bitboard.features import FeatureIndexer, bb_fill_feature_vector

    checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    feature_names = checkpoint["feature_names"]
    n_features = len(feature_names)
    feature_index_map = {name: idx for idx, name in enumerate(feature_names)}
    means_np = checkpoint["feature_means"].astype(np.float32)
    stds_np = checkpoint["feature_stds"].astype(np.float32)

    from robottler.value_model import CatanValueNet
    model = CatanValueNet(input_dim=n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_buf = np.zeros(n_features, dtype=np.float32)
    fi = None  # lazy-init

    def value_fn(bb_state, p0_color):
        nonlocal fi
        if fi is None:
            fi = FeatureIndexer(feature_index_map, bb_state.catan_map)
        else:
            fi.update_map(bb_state.catan_map)

        # Neural eval
        input_buf[:] = 0.0
        bb_fill_feature_vector(bb_state, p0_color, input_buf, fi)
        np.subtract(input_buf, means_np, out=input_buf)
        np.divide(input_buf, stds_np, out=input_buf)
        x = torch.from_numpy(input_buf).unsqueeze(0)
        with torch.inference_mode():
            n = torch.sigmoid(model(x)).item()

        # Heuristic eval (shares node_prod from FeatureIndexer)
        h = _bb_heuristic(bb_state, p0_color, fi.node_prod)

        return h + blend_weight * n

    return value_fn


class BitboardSearchPlayer(NeuralSearchPlayer):
    """AlphaBeta search using bitboard state for fast copy/apply/movegen.

    Converts Game → BitboardState at the start of decide(), then runs
    the entire search tree using bitboard operations.

    Args:
        bb_value_fn: A function (bb_state, color) -> float that evaluates
            BitboardState leaves. Use make_bb_blended_value_fn() or
            make_bb_neural_value_fn() to create one.
    """

    def __init__(self, color, bc_path=None, depth=2, prunning=True,
                 bb_value_fn=None, dice_sample_size=None, use_tt=False):
        # NeuralSearchPlayer needs value_fn for Game objects (used in decide()
        # if we ever fall back). Provide a dummy if only bb_value_fn is given.
        if bb_value_fn is not None and bc_path is None:
            # Need a dummy game value_fn for the parent class
            super().__init__(color, bc_path=None, depth=depth, prunning=prunning,
                             value_fn=lambda game, c: 0.0,
                             dice_sample_size=dice_sample_size, use_tt=use_tt)
        else:
            super().__init__(color, bc_path=bc_path, depth=depth, prunning=prunning,
                             dice_sample_size=dice_sample_size, use_tt=use_tt)

        if bb_value_fn is not None:
            self._bb_eval_fn = bb_value_fn
        elif bc_path is not None:
            self._bb_eval_fn = make_bb_neural_value_fn(bc_path)
        else:
            raise ValueError("Either bc_path or bb_value_fn must be provided")

    def decide(self, game, playable_actions):
        """Convert to bitboard, then run search."""
        from catanatron.state_functions import get_actual_victory_points
        from robottler.bitboard.convert import game_to_bitboard

        original_depth = self.depth
        try:
            my_vp = get_actual_victory_points(game.state, self.color)
            opp_colors = [c for c in game.state.colors if c != self.color]
            opp_vp = max(get_actual_victory_points(game.state, c) for c in opp_colors)
            endgame_threshold = game.vps_to_win - 2
            endgame = my_vp >= endgame_threshold or opp_vp >= endgame_threshold
            is_1v1 = len(game.state.colors) == 2

            if endgame:
                if is_1v1:
                    self.depth = max(original_depth + 1, 3)
                else:
                    self.depth = max(original_depth + 2, 4)
            elif game.state.is_initial_build_phase:
                self.depth = max(original_depth + 2, 4)

            if self.use_tt:
                self._tt = TranspositionTable()
            else:
                self._tt = None

            bb_state = game_to_bitboard(game)
            deadline = time.time() + 240
            result = self.bb_alphabeta(bb_state, self.depth, float("-inf"), float("inf"), deadline)

            if self._tt is not None:
                self._tt_stats = {
                    'hits': self._tt.hits,
                    'misses': self._tt.misses,
                    'stores': self._tt.stores,
                    'size': len(self._tt),
                }
                self._tt = None

            if result[0] is None:
                return playable_actions[0]
            return result[0]
        finally:
            self.depth = original_depth

    def bb_alphabeta(self, bb_state, depth, alpha, beta, deadline):
        """AlphaBeta search on BitboardState."""
        from robottler.bitboard.movegen import bb_generate_actions

        if depth == 0 or bb_state.winning_player() >= 0 or time.time() >= deadline:
            return None, self._bb_eval_fn(bb_state, self.color)

        tt = self._tt
        zhash = None
        tt_best_action = None
        if tt is not None:
            zhash = bb_state.zobrist_hash
            hit, tt_score, tt_best_action = tt.probe(zhash, depth, alpha, beta)
            if hit:
                return tt_best_action, tt_score

        maximizing = bb_state.colors[bb_state.current_player_idx] == self.color
        actions = bb_generate_actions(bb_state)
        actions.sort(key=_action_sort_key)

        if tt_best_action is not None and tt_best_action in actions:
            actions = [tt_best_action] + [a for a in actions if a != tt_best_action]

        if maximizing:
            best_action = None
            best_value = float("-inf")
            for action in actions:
                outcomes = _bb_execute_spectrum(bb_state, action, self.dice_sample_size)
                outcomes.sort(key=lambda x: x[1], reverse=True)

                expected_value = 0.0
                consumed_p = 0.0
                obs_max = float("-inf")

                for outcome, proba in outcomes:
                    result = self.bb_alphabeta(outcome, depth - 1, alpha, beta, deadline)
                    v = result[1]
                    expected_value += proba * v
                    consumed_p += proba
                    obs_max = max(obs_max, v)

                    remaining_p = 1.0 - consumed_p
                    if remaining_p < 1e-9:
                        break
                    if consumed_p >= 0.3:
                        upper = expected_value + remaining_p * obs_max
                        if upper <= alpha:
                            expected_value = upper
                            break

                if expected_value > best_value:
                    best_action = action
                    best_value = expected_value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break

            if tt is not None and zhash is not None:
                if best_value <= alpha:
                    flag = UPPERBOUND
                elif best_value >= beta:
                    flag = LOWERBOUND
                else:
                    flag = EXACT
                tt.store(zhash, depth, best_value, flag, best_action)

            return best_action, best_value
        else:
            best_action = None
            best_value = float("inf")
            for action in actions:
                outcomes = _bb_execute_spectrum(bb_state, action, self.dice_sample_size)
                outcomes.sort(key=lambda x: x[1], reverse=True)

                expected_value = 0.0
                consumed_p = 0.0
                obs_min = float("inf")

                for outcome, proba in outcomes:
                    result = self.bb_alphabeta(outcome, depth - 1, alpha, beta, deadline)
                    v = result[1]
                    expected_value += proba * v
                    consumed_p += proba
                    obs_min = min(obs_min, v)

                    remaining_p = 1.0 - consumed_p
                    if remaining_p < 1e-9:
                        break
                    if consumed_p >= 0.3:
                        lower = expected_value + remaining_p * obs_min
                        if lower >= beta:
                            expected_value = lower
                            break

                if expected_value < best_value:
                    best_action = action
                    best_value = expected_value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

            if tt is not None and zhash is not None:
                if best_value >= beta:
                    flag = LOWERBOUND
                elif best_value <= alpha:
                    flag = UPPERBOUND
                else:
                    flag = EXACT
                tt.store(zhash, depth, best_value, flag, best_action)

            return best_action, best_value

    def __repr__(self):
        return (
            f"BitboardSearchPlayer({self.color}, depth={self.depth}, "
            f"prunning={self.prunning})"
        )
