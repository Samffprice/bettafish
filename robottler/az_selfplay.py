"""AlphaZero self-play data generation and training.

Generates games using MCTS + current network, records:
- Feature vectors (for both players, at each decision point)
- MCTS visit count distributions (policy targets)
- Game outcomes (value targets: +1 win, -1 loss, 0 draw)

Training uses combined loss:
    L = value_weight * MSE(value_pred, outcome) + CrossEntropy(policy_pred, visit_dist)

Usage:
    # Generate self-play games (iteration 0)
    python3 -m robottler.az_selfplay generate \
        --checkpoint robottler/models/az_iter0.pt \
        --games 500 --sims 400 --output-dir datasets/az_selfplay/iter0

    # Train on self-play data
    python3 -m robottler.az_selfplay train \
        --data-dir datasets/az_selfplay/iter0 \
        --checkpoint robottler/models/az_iter0.pt \
        --output robottler/models/az_iter1.pt \
        --epochs 20

    # Evaluate new vs old
    python3 -m robottler.az_selfplay evaluate \
        --new-checkpoint robottler/models/az_iter1.pt \
        --old-checkpoint robottler/models/az_iter0.pt \
        --games 200 --sims 400

    # Full loop (generate + train + evaluate)
    python3 -m robottler.az_selfplay loop \
        --start-checkpoint robottler/models/az_iter0.pt \
        --iterations 10 --games-per-iter 500 --sims 400 \
        --output-dir datasets/az_selfplay

    # Expert Iteration: generate data with search player
    python3 -m robottler.az_selfplay expert \
        --bc-model robottler/models/value_net_v2.pt \
        --games 1000 --search-depth 2 --dice-sample 5 \
        --output-dir datasets/expert_data --workers 6

    # ExIt loop: expert generation + differential LR training
    python3 -m robottler.az_selfplay loop \
        --start-checkpoint robottler/models/az_iter0.pt \
        --expert --bc-model robottler/models/value_net_v2.pt \
        --search-depth 2 --dice-sample 5 --differential-lr \
        --iterations 10 --games-per-iter 500 \
        --output-dir datasets/exit_v1 --workers 6
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

# When --workers is passed, disable internal threading BEFORE importing
# torch/numpy. PyTorch's OMP threads deadlock with fork()-based multiprocessing.
# Without --workers, leave threading unrestricted for normal single-process use.
if '--workers' in sys.argv:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, to_action_space

from robottler.alphazero_net import (
    CatanAlphaZeroNet, load_checkpoint, save_checkpoint,
    bb_actions_to_mask,
)
from robottler.bb_mcts_player import BBMCTSPlayer, BBMCTSNode, make_bb_mcts_player
from robottler.bitboard.convert import game_to_bitboard
from robottler.bitboard.features import FeatureIndexer, bb_fill_feature_vector
from robottler.bitboard.movegen import bb_generate_actions
from robottler.gnn_net import (
    CatanGNNet, NODE_FEAT_DIM, GLOBAL_FEAT_DIM,
    bb_fill_node_features, bb_fill_global_features,
    save_checkpoint_gnn, load_checkpoint_gnn,
)


# ---------------------------------------------------------------------------
# Blend bootstrap: full blend (heuristic + neural) as MCTS leaf evaluator
# ---------------------------------------------------------------------------

def make_blend_leaf_fn(bc_path, blend_weight=1e8):
    """Create a blend leaf evaluator for MCTS bootstrap.

    Uses the full alpha-beta blend (heuristic + neural) with two-tier
    normalization to map values into [-1, +1] for MCTS backpropagation.

    blend_weight=1e8 matches the proven alpha-beta config (69-83% vs AB).
    At 1e8, neural (~1e7) is a tiebreaker below production (~1e8).
    At 1e10, neural drowns out everything except VPs — don't use it.

    The raw blend spans ~14 orders of magnitude (VP weight 3e14 vs
    production 1e8). A single tanh scale crushes sub-VP differences to
    noise. Instead we decompose the advantage into:
      1. VP signal: tanh(vp_diff / 4) * 0.6  — coarse win proximity
      2. Sub-VP signal: tanh(sub_vp_adv / 3e8) * 0.4  — production,
         neural, reachability differences

    Sub-VP differences (~0.01–0.27) stay above MCTS exploration noise
    (~0.02 for c_puct=1.4, uniform priors), so the blend's strategic
    information actually guides search.

    Args:
        bc_path: Path to value_net_v2.pt (BC-trained checkpoint)
        blend_weight: Weight for neural component in blend (default 1e8)
    """
    import math
    from robottler.value_model import CatanValueNet
    from robottler.search_player import _bb_heuristic
    from robottler.bitboard.state import PS_VP

    VP_HEURISTIC_WEIGHT = 3e14  # from DEFAULT_WEIGHTS["public_vps"]

    # Load BC neural net (shared across both perspective evaluations)
    checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    feature_names = checkpoint["feature_names"]
    n_features = len(feature_names)
    feature_index_map = {name: idx for idx, name in enumerate(feature_names)}
    means_np = checkpoint["feature_means"].astype(np.float32)
    stds_np = checkpoint["feature_stds"].astype(np.float32)

    model = CatanValueNet(input_dim=n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Two buffers: one per perspective to avoid re-zeroing between evals
    buf_a = np.zeros(n_features, dtype=np.float32)
    buf_b = np.zeros(n_features, dtype=np.float32)
    fi = None  # lazy-init

    def _blend_one(bb_state, color, buf):
        """Evaluate blend for a single perspective. Returns raw blend value."""
        buf[:] = 0.0
        bb_fill_feature_vector(bb_state, color, buf, fi)
        np.subtract(buf, means_np, out=buf)
        np.divide(buf, stds_np, out=buf)
        x = torch.from_numpy(buf).unsqueeze(0)
        with torch.inference_mode():
            n = torch.sigmoid(model(x)).item()
        h = _bb_heuristic(bb_state, color, fi.node_prod)
        return h + blend_weight * n

    def leaf_fn(bb_state, color):
        nonlocal fi
        if fi is None:
            fi = FeatureIndexer(feature_index_map, bb_state.catan_map)
        else:
            fi.update_map(bb_state.catan_map)

        # Evaluate from both perspectives (shared model + FeatureIndexer)
        my_idx = bb_state.color_to_index[color]
        opp_idx = (my_idx + 1) % bb_state.num_players
        opp_color = bb_state.colors[opp_idx]

        my_blend = _blend_one(bb_state, color, buf_a)
        opp_blend = _blend_one(bb_state, opp_color, buf_b)

        # VP counts
        my_vps = float(bb_state.player_state[my_idx][PS_VP])
        opp_vps = float(bb_state.player_state[opp_idx][PS_VP])
        vp_diff = my_vps - opp_vps

        # Decompose: remove VP component to isolate sub-VP blend
        adv = my_blend - opp_blend
        vp_component = vp_diff * VP_HEURISTIC_WEIGHT
        sub_vp_adv = adv - vp_component

        # Two-tier normalization
        # VP: 1 VP → 0.15, 2 VP → 0.28, 5 VP → 0.51
        vp_signal = math.tanh(vp_diff / 4.0) * 0.6

        # Sub-VP: production ~2e8, neural ~1e7 (at blend_weight=1e8)
        # 3e8 scale keeps production diffs at ~0.23, neural at ~0.01-0.07
        sub_signal = math.tanh(sub_vp_adv / 3e8) * 0.4

        return max(-1.0, min(1.0, vp_signal + sub_signal))

    return leaf_fn


# ---------------------------------------------------------------------------
# Self-play data generation
# ---------------------------------------------------------------------------

class SelfPlayRecord:
    """A single training example from self-play."""
    __slots__ = ('features', 'policy_target', 'value_target')

    def __init__(self, features, policy_target, value_target):
        self.features = features            # np.float32 (n_features,)
        self.policy_target = policy_target  # np.float32 (ACTION_SPACE_SIZE,) visit distribution
        self.value_target = value_target    # float in {-1, 0, +1}


def generate_one_game(az_net, feature_indexer, feature_means, feature_stds,
                      num_simulations=400, c_puct=1.4, temperature_threshold=15,
                      vps_to_win=10):
    """Play one self-play game, returning list of SelfPlayRecords.

    Args:
        az_net: CatanAlphaZeroNet (shared between both players)
        feature_indexer: FeatureIndexer instance
        feature_means, feature_stds: normalization arrays
        num_simulations: MCTS sims per decision
        c_puct: exploration constant
        temperature_threshold: use temp=1 for first N decisions, then temp=0
        vps_to_win: victory points to win

    Returns:
        list of SelfPlayRecord
    """
    n_features = len(feature_means)
    means_t = torch.tensor(feature_means, dtype=torch.float32)
    stds_t = torch.tensor(feature_stds, dtype=torch.float32)

    # Create two MCTS players sharing the same network
    p1 = BBMCTSPlayer(
        color=Color.RED, az_net=az_net,
        feature_indexer=feature_indexer,
        feature_means=feature_means, feature_stds=feature_stds,
        num_simulations=num_simulations, c_puct=c_puct,
        temperature=1.0,  # will be adjusted per-decision
        dirichlet_alpha=0.3, dirichlet_weight=0.25,
    )
    p2 = BBMCTSPlayer(
        color=Color.BLUE, az_net=az_net,
        feature_indexer=feature_indexer,
        feature_means=feature_means, feature_stds=feature_stds,
        num_simulations=num_simulations, c_puct=c_puct,
        temperature=1.0,
        dirichlet_alpha=0.3, dirichlet_weight=0.25,
    )

    game = Game(players=[p1, p2], vps_to_win=vps_to_win)
    feature_indexer.update_map(game.state.board.map)

    records = []
    decision_count = 0
    feat_buf = np.zeros(n_features, dtype=np.float32)

    while game.winning_color() is None and game.state.num_turns < 1000:
        actions = game.playable_actions
        current = game.state.current_player()

        if len(actions) == 1:
            game.execute(actions[0])
            continue

        if not isinstance(current, BBMCTSPlayer):
            game.execute(actions[0])
            continue

        # Adjust temperature: exploration early, greedy later
        decision_count += 1
        if decision_count <= temperature_threshold:
            current.temperature = 1.0
        else:
            current.temperature = 0.0

        # Convert to bitboard
        bb_state = game_to_bitboard(game)

        # Extract features BEFORE the decision
        feat_buf[:] = 0.0
        bb_fill_feature_vector(bb_state, current.color, feat_buf, feature_indexer)

        # Run MCTS
        root = BBMCTSNode(bb_state.copy())
        bb_actions = bb_generate_actions(root.bb_state)
        if not root.is_expanded and not root.is_terminal and not root.is_chance:
            current._evaluate_and_expand(root, bb_actions)

        if current.dirichlet_alpha > 0 and isinstance(root.children, list):
            current._add_dirichlet_noise(root)

        for _ in range(num_simulations):
            current._simulate(root)

        # Extract visit distribution as policy target
        policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        total_visits = 0
        if isinstance(root.children, list):
            for action, child in root.children:
                try:
                    idx = to_action_space(action)
                    policy_target[idx] = child.visit_count
                    total_visits += child.visit_count
                except (ValueError, IndexError):
                    pass

        if total_visits > 0:
            policy_target /= total_visits

        # Select action
        if current.temperature <= 0:
            best_action, _ = root.most_visited_child()
        else:
            best_action = current._sample_action(root)

        # Record
        records.append(SelfPlayRecord(
            features=feat_buf.copy(),
            policy_target=policy_target,
            value_target=0.0,  # filled after game ends
        ))

        # Execute the selected action through the Game object
        for a in actions:
            if a.action_type == best_action.action_type and a.value == best_action.value:
                game.execute(a)
                break
        else:
            game.execute(actions[0])

    # Fill in value targets based on game outcome
    winner = game.winning_color()
    for rec in records:
        if winner is None:
            rec.value_target = 0.0
        else:
            # All records are from the current player's perspective at that point
            # Since we extract features from current.color's perspective,
            # and the current player alternates, we need to track who was playing
            rec.value_target = 0.0  # placeholder

    # Re-do: track which color each record belongs to
    # Actually, we need to fix this. Let me restructure to track the color.
    return records, winner, game.state.num_turns


def generate_one_game_v2(az_net, feature_indexer, feature_means, feature_stds,
                         num_simulations=400, c_puct=1.4, temperature_threshold=15,
                         vps_to_win=10, leaf_value_fn=None):
    """Play one self-play game with proper color tracking.

    Args:
        leaf_value_fn: Optional external leaf evaluator fn(bb_state, color) -> [-1,+1].
            When set, MCTS uses this for value estimation with uniform priors
            instead of the AZ network (for blend-bootstrap self-play).

    Returns:
        records: list of (features, policy_target, color) tuples
        winner: Color or None
        num_turns: int
    """
    n_features = len(feature_means)

    p1 = BBMCTSPlayer(
        color=Color.RED, az_net=az_net,
        feature_indexer=feature_indexer,
        feature_means=feature_means, feature_stds=feature_stds,
        num_simulations=num_simulations, c_puct=c_puct,
        temperature=1.0,
        dirichlet_alpha=0.3, dirichlet_weight=0.25,
        leaf_value_fn=leaf_value_fn, leaf_use_policy=False,
    )
    p2 = BBMCTSPlayer(
        color=Color.BLUE, az_net=az_net,
        feature_indexer=feature_indexer,
        feature_means=feature_means, feature_stds=feature_stds,
        num_simulations=num_simulations, c_puct=c_puct,
        temperature=1.0,
        dirichlet_alpha=0.3, dirichlet_weight=0.25,
        leaf_value_fn=leaf_value_fn, leaf_use_policy=False,
    )

    game = Game(players=[p1, p2], vps_to_win=vps_to_win)
    feature_indexer.update_map(game.state.board.map)

    raw_records = []  # (features, policy_target, color)
    decision_count = 0
    feat_buf = np.zeros(n_features, dtype=np.float32)

    while game.winning_color() is None and game.state.num_turns < 1000:
        actions = game.playable_actions
        current = game.state.current_player()

        if len(actions) == 1:
            game.execute(actions[0])
            continue

        if not isinstance(current, BBMCTSPlayer):
            game.execute(actions[0])
            continue

        decision_count += 1
        if decision_count <= temperature_threshold:
            current.temperature = 1.0
        else:
            current.temperature = 0.0

        bb_state = game_to_bitboard(game)
        feat_buf[:] = 0.0
        bb_fill_feature_vector(bb_state, current.color, feat_buf, feature_indexer)

        # Run MCTS
        root = BBMCTSNode(bb_state.copy())
        bb_actions = bb_generate_actions(root.bb_state)
        if not root.is_expanded and not root.is_terminal and not root.is_chance:
            current._evaluate_and_expand(root, bb_actions)
        if current.dirichlet_alpha > 0 and isinstance(root.children, list):
            current._add_dirichlet_noise(root)

        for _ in range(num_simulations):
            current._simulate(root)

        # Visit distribution
        policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        total_visits = 0
        if isinstance(root.children, list):
            for action, child in root.children:
                try:
                    idx = to_action_space(action)
                    policy_target[idx] = child.visit_count
                    total_visits += child.visit_count
                except (ValueError, IndexError):
                    pass
        if total_visits > 0:
            policy_target /= total_visits

        # Select action
        if current.temperature <= 0:
            best_action, _ = root.most_visited_child()
        else:
            best_action = current._sample_action(root)

        raw_records.append((feat_buf.copy(), policy_target, current.color))

        # Execute
        for a in actions:
            if a.action_type == best_action.action_type and a.value == best_action.value:
                game.execute(a)
                break
        else:
            game.execute(actions[0])

    winner = game.winning_color()

    # Convert to SelfPlayRecords with correct value targets
    records = []
    for features, policy_target, color in raw_records:
        if winner is None:
            value_target = 0.0
        elif winner == color:
            value_target = 1.0
        else:
            value_target = -1.0
        records.append(SelfPlayRecord(features, policy_target, value_target))

    return records, winner, game.state.num_turns


# ---------------------------------------------------------------------------
# Expert Iteration: self-play with BitboardSearchPlayer
# ---------------------------------------------------------------------------

def generate_one_game_expert(bb_value_fn, feature_indexer, feature_means, feature_stds,
                             search_depth=2, dice_sample_size=5, vps_to_win=10,
                             blend_leaf_fn=None, with_ranking=False,
                             with_graph_features=False):
    """Play one game with BitboardSearchPlayer (both sides) for ExIt training.

    Records:
    - Features (176-dim, same as MCTS path)
    - One-hot policy target: the search player's chosen action
    - Value target: continuous blend evaluation (if blend_leaf_fn provided),
      or binary +1/-1 game outcome (fallback)
    - Optionally: per-node features [54, 18] and global features [76] for GNN

    Args:
        bb_value_fn: Bitboard value function (from make_bb_blended_value_fn)
        feature_indexer: FeatureIndexer instance
        feature_means, feature_stds: normalization arrays
        search_depth: search depth for BitboardSearchPlayer
        dice_sample_size: number of dice outcomes to sample
        vps_to_win: victory points to win
        blend_leaf_fn: Optional fn(bb_state, color) -> float in [-1, +1].
            When provided, records continuous blend evaluation as value target
            instead of binary game outcome. Use make_blend_leaf_fn() to create.
        with_ranking: If True, evaluate all legal child states with bb_value_fn
            at each decision point and return per-action scores for ranking loss.
        with_graph_features: If True, also extract per-node [54, 18] and
            global [76] features at each decision point for GNN training.

    Returns:
        records: list of SelfPlayRecord
        winner: Color or None
        num_turns: int
        ranking_records: list of (child_features, child_scores) tuples
            Only returned when with_ranking=True, otherwise omitted.
        graph_features: list of (node_features[54,18], global_features[76]) tuples
            Only returned when with_graph_features=True, otherwise omitted.
    """
    from robottler.search_player import BitboardSearchPlayer

    n_features = len(feature_means)

    p1 = BitboardSearchPlayer(
        color=Color.RED, bb_value_fn=bb_value_fn,
        depth=search_depth, prunning=False,
        dice_sample_size=dice_sample_size,
    )
    p2 = BitboardSearchPlayer(
        color=Color.BLUE, bb_value_fn=bb_value_fn,
        depth=search_depth, prunning=False,
        dice_sample_size=dice_sample_size,
    )

    game = Game(players=[p1, p2], vps_to_win=vps_to_win)
    feature_indexer.update_map(game.state.board.map)

    from robottler.bitboard.actions import bb_apply_action

    raw_records = []  # (features, policy_target, color, value_target_or_None)
    ranking_records = []  # (child_features_array, child_scores_array)
    graph_records = []  # (node_features[54,18], global_features[76])
    feat_buf = np.zeros(n_features, dtype=np.float32)
    if with_graph_features:
        node_buf = np.zeros((54, NODE_FEAT_DIM), dtype=np.float32)
        global_buf = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
        node_prod = feature_indexer.node_prod

    while game.winning_color() is None and game.state.num_turns < 1000:
        actions = game.playable_actions
        current = game.state.current_player()

        if len(actions) == 1:
            game.execute(actions[0])
            continue

        if not isinstance(current, BitboardSearchPlayer):
            game.execute(actions[0])
            continue

        # Extract features BEFORE the decision
        bb_state = game_to_bitboard(game)
        feat_buf[:] = 0.0
        bb_fill_feature_vector(bb_state, current.color, feat_buf, feature_indexer)

        # Extract graph features if requested
        if with_graph_features:
            node_buf[:] = 0.0
            global_buf[:] = 0.0
            bb_fill_node_features(bb_state, current.color, node_buf, node_prod)
            bb_fill_global_features(bb_state, current.color, global_buf)

        # Compute blend value target if available
        blend_value = None
        if blend_leaf_fn is not None:
            blend_value = blend_leaf_fn(bb_state, current.color)

        # Ranking data: evaluate all legal child states
        if with_ranking and len(actions) >= 2:
            child_feats = []
            child_scores = []
            child_buf = np.zeros(n_features, dtype=np.float32)
            for action in actions:
                try:
                    to_action_space(action)
                except (ValueError, IndexError):
                    continue  # skip unmappable actions
                child_bb = bb_state.copy()
                bb_apply_action(child_bb, action)
                child_score = bb_value_fn(child_bb, current.color)
                child_buf[:] = 0.0
                bb_fill_feature_vector(child_bb, current.color, child_buf, feature_indexer)
                child_feats.append(child_buf.copy())
                child_scores.append(child_score)
            if len(child_feats) >= 2:
                ranking_records.append((
                    np.array(child_feats, dtype=np.float32),
                    np.array(child_scores, dtype=np.float32),
                ))

        # Get the search player's chosen action
        chosen = current.decide(game, actions)

        # One-hot policy target
        policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        try:
            idx = to_action_space(chosen)
            policy_target[idx] = 1.0
        except (ValueError, IndexError):
            # Action not in standard space — skip this record
            game.execute(chosen)
            continue

        raw_records.append((feat_buf.copy(), policy_target, current.color, blend_value))
        if with_graph_features:
            graph_records.append((node_buf.copy(), global_buf.copy()))
        game.execute(chosen)

    winner = game.winning_color()

    # Convert to SelfPlayRecords with value targets
    records = []
    for features, policy_target, color, blend_value in raw_records:
        if blend_value is not None:
            # Use continuous blend evaluation
            value_target = blend_value
        elif winner is None:
            value_target = 0.0
        elif winner == color:
            value_target = 1.0
        else:
            value_target = -1.0
        records.append(SelfPlayRecord(features, policy_target, value_target))

    result = [records, winner, game.state.num_turns]
    if with_ranking:
        result.append(ranking_records)
    if with_graph_features:
        result.append(graph_records)
    return tuple(result)


# ---------------------------------------------------------------------------
# Multiprocessing workers for expert iteration
# ---------------------------------------------------------------------------

_expert_worker_value_fn = None
_expert_worker_fi = None
_expert_worker_means = None
_expert_worker_stds = None
_expert_worker_kwargs = None
_expert_worker_blend_leaf_fn = None


def _expert_worker_init(bc_path, blend_weight, search_depth, dice_sample_size,
                         vps_to_win, distill_values, with_ranking=False,
                         with_graph_features=False):
    """Initialize per-worker state for expert iteration: load blend value fn."""
    global _expert_worker_value_fn, _expert_worker_fi
    global _expert_worker_means, _expert_worker_stds, _expert_worker_kwargs
    global _expert_worker_blend_leaf_fn

    from robottler.search_player import make_bb_blended_value_fn

    _expert_worker_value_fn = make_bb_blended_value_fn(bc_path, blend_weight=blend_weight)

    # Create blend leaf fn for value distillation if requested
    _expert_worker_blend_leaf_fn = None
    if distill_values:
        _expert_worker_blend_leaf_fn = make_blend_leaf_fn(bc_path, blend_weight=1e8)

    # Load feature info from the AZ checkpoint (or BC checkpoint) for feature extraction
    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    feat_names = ckpt['feature_names']
    _expert_worker_means = ckpt['feature_means']
    _expert_worker_stds = ckpt['feature_stds']
    feature_index_map = {name: i for i, name in enumerate(feat_names)}

    from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
    catan_map = build_map(BASE_MAP_TEMPLATE)
    _expert_worker_fi = FeatureIndexer(feature_index_map, catan_map)

    _expert_worker_kwargs = {
        'search_depth': search_depth,
        'dice_sample_size': dice_sample_size,
        'vps_to_win': vps_to_win,
        'with_ranking': with_ranking,
        'with_graph_features': with_graph_features,
    }


def _expert_worker_play_one(_game_idx):
    """Play one expert game in a worker."""
    with_ranking = _expert_worker_kwargs.get('with_ranking', False)
    with_graph_features = _expert_worker_kwargs.get('with_graph_features', False)
    result = generate_one_game_expert(
        _expert_worker_value_fn, _expert_worker_fi,
        _expert_worker_means, _expert_worker_stds,
        blend_leaf_fn=_expert_worker_blend_leaf_fn,
        **_expert_worker_kwargs,
    )
    # Unpack variable-length result tuple
    records, winner, turns = result[0], result[1], result[2]
    idx = 3
    if with_ranking:
        ranking_records = result[idx]; idx += 1
    else:
        ranking_records = []
    if with_graph_features:
        graph_records = result[idx]; idx += 1
    else:
        graph_records = []
    features = [r.features for r in records]
    policies = [r.policy_target for r in records]
    values = [r.value_target for r in records]
    # Serialize ranking records as lists for pickling
    ranking_serial = [(cf.tolist(), cs.tolist()) for cf, cs in ranking_records]
    # Serialize graph features as lists for pickling
    graph_serial = [(nf.tolist(), gf.tolist()) for nf, gf in graph_records]
    return (features, policies, values, str(winner), turns, ranking_serial, graph_serial)


def generate_expert_games(bc_path, num_games, search_depth=2, dice_sample_size=5,
                          blend_weight=1e10, vps_to_win=10, output_dir=None,
                          workers=1, distill_values=False, with_ranking=False,
                          with_graph_features=False):
    """Generate expert iteration data using BitboardSearchPlayer.

    Same output format as generate_games() (features.npy, policies.npy, values.npy)
    so training code works unchanged. With --with-graph-features, also saves
    node_features.npy [N, 54, 18] and global_features.npy [N, 76] for GNN training.

    Args:
        bc_path: Path to BC value net (e.g., value_net_v2.pt) for blend value fn
        num_games: Number of games to generate
        search_depth: Search depth for BitboardSearchPlayer
        dice_sample_size: Number of dice outcomes to sample
        blend_weight: Weight for neural component in blend (default 1e10)
        vps_to_win: Victory points to win
        output_dir: Directory to save data (with resume support)
        workers: Number of parallel worker processes
        with_ranking: If True, also generate ranking data (per-action expert scores)
        with_graph_features: If True, also extract per-node [54, 18] and
            global [76] features at each decision point for GNN training.

    Returns:
        all_records: list of SelfPlayRecord
        stats: dict with game statistics
    """
    from robottler.search_player import make_bb_blended_value_fn

    # Check for existing progress
    start_game = 0
    all_records = []
    total_turns = 0
    winners = {Color.RED: 0, Color.BLUE: 0, None: 0}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        progress_path = os.path.join(output_dir, '_progress.json')
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                prog = json.load(f)
            start_game = prog.get('games_completed', 0)
            total_turns = prog.get('total_turns', 0)
            for k, v in prog.get('winners', {}).items():
                if k in ('None', 'null'):
                    winners[None] = v
                else:
                    name = k.replace('Color.', '') if k.startswith('Color.') else k
                    winners[Color[name]] = v
            if start_game >= num_games:
                print(f"  Already completed {start_game}/{num_games} games, skipping")
                if os.path.exists(os.path.join(output_dir, 'features.npy')):
                    feats, pols, vals = load_records(output_dir)
                    for i in range(len(vals)):
                        all_records.append(SelfPlayRecord(feats[i], pols[i], vals[i]))
                stats = {
                    'num_games': num_games,
                    'total_records': len(all_records),
                    'avg_turns': total_turns / max(num_games, 1),
                    'avg_records_per_game': len(all_records) / max(num_games, 1),
                    'winners': {str(k): v for k, v in winners.items()},
                }
                return all_records, stats
            elif start_game > 0:
                print(f"  Resuming from game {start_game}/{num_games}")
                if os.path.exists(os.path.join(output_dir, 'features.npy')):
                    feats, pols, vals = load_records(output_dir)
                    for i in range(len(vals)):
                        all_records.append(SelfPlayRecord(feats[i], pols[i], vals[i]))

    remaining = num_games - start_game

    all_ranking_records = []
    all_node_features = []  # list of [54, 18] arrays
    all_global_features = []  # list of [76] arrays

    # Resume graph features from previous progress
    if with_graph_features and start_game > 0 and output_dir is not None:
        nf_path = os.path.join(output_dir, 'node_features.npy')
        gf_path = os.path.join(output_dir, 'global_features.npy')
        if os.path.exists(nf_path) and os.path.exists(gf_path):
            prev_nf, prev_gf = load_graph_records(output_dir)
            all_node_features = list(prev_nf)
            all_global_features = list(prev_gf)
            print(f"  Resumed {len(all_node_features)} graph feature records")
            del prev_nf, prev_gf

    if workers <= 1:
        # Sequential path
        bb_value_fn = make_bb_blended_value_fn(bc_path, blend_weight=blend_weight)
        seq_blend_leaf_fn = make_blend_leaf_fn(bc_path, blend_weight=1e8) if distill_values else None

        ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
        feat_names = ckpt['feature_names']
        feat_means = ckpt['feature_means']
        feat_stds = ckpt['feature_stds']
        feature_index_map = {name: i for i, name in enumerate(feat_names)}

        from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
        catan_map = build_map(BASE_MAP_TEMPLATE)
        fi = FeatureIndexer(feature_index_map, catan_map)

        for i in tqdm(range(start_game, num_games), desc="Expert games",
                      initial=start_game, total=num_games):
            result = generate_one_game_expert(
                bb_value_fn, fi, feat_means, feat_stds,
                search_depth=search_depth,
                dice_sample_size=dice_sample_size,
                vps_to_win=vps_to_win,
                blend_leaf_fn=seq_blend_leaf_fn,
                with_ranking=with_ranking,
                with_graph_features=with_graph_features,
            )
            records, winner, turns = result[0], result[1], result[2]
            idx = 3
            if with_ranking:
                all_ranking_records.extend(result[idx]); idx += 1
            if with_graph_features:
                graph_recs = result[idx]; idx += 1
                for nf, gf in graph_recs:
                    all_node_features.append(nf)
                    all_global_features.append(gf)

            all_records.extend(records)
            total_turns += turns
            winners[winner] = winners.get(winner, 0) + 1

            if output_dir is not None:
                save_records(all_records, output_dir, quiet=True)
                if with_graph_features and all_node_features:
                    _save_graph_features(all_node_features, all_global_features, output_dir)
                with open(os.path.join(output_dir, '_progress.json'), 'w') as f:
                    json.dump({
                        'games_completed': i + 1,
                        'total_games': num_games,
                        'total_turns': total_turns,
                        'total_records': len(all_records),
                        'winners': {str(k): v for k, v in winners.items()},
                        'method': 'expert',
                    }, f)
    else:
        # Parallel path — save incrementally after each game completes
        print(f"Using {workers} parallel workers for expert game generation")
        if distill_values:
            print(f"  Value distillation enabled (blend leaf fn)")
        if with_ranking:
            print(f"  Ranking data generation enabled")
        if with_graph_features:
            print(f"  Graph feature extraction enabled (for GNN training)")
        games_done = start_game
        with mp.Pool(
            processes=workers,
            initializer=_expert_worker_init,
            initargs=(bc_path, blend_weight, search_depth, dice_sample_size,
                      vps_to_win, distill_values, with_ranking, with_graph_features),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_expert_worker_play_one, range(remaining)),
                total=remaining, desc="Expert games", initial=start_game,
            ):
                features_list, policies_list, values_list, winner_str, turns, ranking_serial, graph_serial = result
                for feat, pol, val in zip(features_list, policies_list, values_list):
                    all_records.append(SelfPlayRecord(feat, pol, val))
                # Deserialize ranking records
                for cf_list, cs_list in ranking_serial:
                    all_ranking_records.append((
                        np.array(cf_list, dtype=np.float32),
                        np.array(cs_list, dtype=np.float32),
                    ))
                # Deserialize graph features
                for nf_list, gf_list in graph_serial:
                    all_node_features.append(np.array(nf_list, dtype=np.float32))
                    all_global_features.append(np.array(gf_list, dtype=np.float32))
                total_turns += turns
                if winner_str in ('None', 'null'):
                    winners[None] = winners.get(None, 0) + 1
                else:
                    name = winner_str.replace('Color.', '') if winner_str.startswith('Color.') else winner_str
                    winners[Color[name]] = winners.get(Color[name], 0) + 1

                games_done += 1
                # Save every 10 games
                if output_dir is not None and games_done % 10 == 0:
                    save_records(all_records, output_dir, quiet=True)
                    if with_graph_features and all_node_features:
                        _save_graph_features(all_node_features, all_global_features, output_dir)
                    with open(os.path.join(output_dir, '_progress.json'), 'w') as f:
                        json.dump({
                            'games_completed': games_done,
                            'total_games': num_games,
                            'total_turns': total_turns,
                            'total_records': len(all_records),
                            'winners': {str(k): v for k, v in winners.items()},
                            'method': 'expert',
                        }, f)

        # Final save
        if output_dir is not None:
            save_records(all_records, output_dir, quiet=True)
            if with_graph_features and all_node_features:
                _save_graph_features(all_node_features, all_global_features, output_dir)
            with open(os.path.join(output_dir, '_progress.json'), 'w') as f:
                json.dump({
                    'games_completed': games_done,
                    'total_games': num_games,
                    'total_turns': total_turns,
                    'total_records': len(all_records),
                    'winners': {str(k): v for k, v in winners.items()},
                    'method': 'expert',
                }, f)

    # Save ranking data if generated
    if with_ranking and all_ranking_records and output_dir is not None:
        save_ranking_records(all_ranking_records, output_dir)

    stats = {
        'num_games': num_games,
        'total_records': len(all_records),
        'avg_turns': total_turns / max(num_games, 1),
        'avg_records_per_game': len(all_records) / max(num_games, 1),
        'winners': {str(k): v for k, v in winners.items()},
    }
    return all_records, stats


# ---------------------------------------------------------------------------
# Multiprocessing worker for parallel self-play
# ---------------------------------------------------------------------------

_az_worker_net = None
_az_worker_fi = None
_az_worker_means = None
_az_worker_stds = None
_az_worker_kwargs = None
_az_worker_leaf_fn = None


def _az_worker_init(checkpoint_path, num_simulations, c_puct,
                    temperature_threshold, vps_to_win, blend_bootstrap_path=None):
    """Initialize per-worker state: load model, build FeatureIndexer."""
    global _az_worker_net, _az_worker_fi, _az_worker_means, _az_worker_stds
    global _az_worker_kwargs, _az_worker_leaf_fn

    net, ckpt = load_checkpoint(checkpoint_path)
    feat_names = ckpt['feature_names']
    _az_worker_means = ckpt['feature_means']
    _az_worker_stds = ckpt['feature_stds']
    feature_index_map = {name: i for i, name in enumerate(feat_names)}

    from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
    catan_map = build_map(BASE_MAP_TEMPLATE)
    _az_worker_fi = FeatureIndexer(feature_index_map, catan_map)
    _az_worker_net = net

    # Create blend bootstrap leaf fn if requested (each worker gets its own)
    _az_worker_leaf_fn = None
    if blend_bootstrap_path is not None:
        _az_worker_leaf_fn = make_blend_leaf_fn(blend_bootstrap_path)

    _az_worker_kwargs = {
        'num_simulations': num_simulations,
        'c_puct': c_puct,
        'temperature_threshold': temperature_threshold,
        'vps_to_win': vps_to_win,
        'leaf_value_fn': _az_worker_leaf_fn,
    }


def _az_worker_play_one(_game_idx):
    """Play one self-play game in a worker. Returns (features_list, policies_list, values_list, winner_str, turns)."""
    records, winner, turns = generate_one_game_v2(
        _az_worker_net, _az_worker_fi, _az_worker_means, _az_worker_stds,
        **_az_worker_kwargs,
    )
    # Pack records into numpy arrays for pickling efficiency
    features = [r.features for r in records]
    policies = [r.policy_target for r in records]
    values = [r.value_target for r in records]
    return (features, policies, values, str(winner), turns)


def generate_games(checkpoint_path, num_games, num_simulations=400,
                   c_puct=1.4, temperature_threshold=15, vps_to_win=10,
                   output_dir=None, workers=1, blend_bootstrap_path=None):
    """Generate multiple self-play games with incremental saving.

    If output_dir is provided, saves after every game so progress survives
    interruption. On resume, skips already-completed games.

    Args:
        workers: number of parallel processes (1 = sequential).
        blend_bootstrap_path: Path to BC value net (e.g., value_net_v2.pt).
            When set, uses the BC neural net as MCTS leaf evaluator with
            uniform priors instead of the AZ network.

    Returns:
        all_records: list of SelfPlayRecord
        stats: dict with game statistics
    """
    # Check for existing progress
    start_game = 0
    all_records = []
    total_turns = 0
    winners = {Color.RED: 0, Color.BLUE: 0, None: 0}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        progress_path = os.path.join(output_dir, '_progress.json')
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                prog = json.load(f)
            start_game = prog.get('games_completed', 0)
            total_turns = prog.get('total_turns', 0)
            for k, v in prog.get('winners', {}).items():
                if k in ('None', 'null'):
                    winners[None] = v
                else:
                    name = k.replace('Color.', '') if k.startswith('Color.') else k
                    winners[Color[name]] = v
            if start_game >= num_games:
                print(f"  Already completed {start_game}/{num_games} games, skipping")
                if os.path.exists(os.path.join(output_dir, 'features.npy')):
                    feats, pols, vals = load_records(output_dir)
                    for i in range(len(vals)):
                        all_records.append(SelfPlayRecord(feats[i], pols[i], vals[i]))
                stats = {
                    'num_games': num_games,
                    'total_records': len(all_records),
                    'avg_turns': total_turns / max(num_games, 1),
                    'avg_records_per_game': len(all_records) / max(num_games, 1),
                    'winners': {str(k): v for k, v in winners.items()},
                }
                return all_records, stats
            elif start_game > 0:
                print(f"  Resuming from game {start_game}/{num_games}")
                if os.path.exists(os.path.join(output_dir, 'features.npy')):
                    feats, pols, vals = load_records(output_dir)
                    for i in range(len(vals)):
                        all_records.append(SelfPlayRecord(feats[i], pols[i], vals[i]))

    remaining = num_games - start_game

    # Create blend bootstrap leaf fn if requested
    leaf_value_fn = None
    if blend_bootstrap_path is not None:
        leaf_value_fn = make_blend_leaf_fn(blend_bootstrap_path)
        print(f"  Using blend bootstrap from {blend_bootstrap_path}")

    if workers <= 1:
        # Sequential path — load model once
        net, ckpt = load_checkpoint(checkpoint_path)
        feat_names = ckpt['feature_names']
        feat_means = ckpt['feature_means']
        feat_stds = ckpt['feature_stds']
        feature_index_map = {name: i for i, name in enumerate(feat_names)}

        from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
        catan_map = build_map(BASE_MAP_TEMPLATE)
        fi = FeatureIndexer(feature_index_map, catan_map)

        _sp_games_done = 0
        pbar = tqdm(range(start_game, num_games), desc="Self-play",
                    initial=start_game, total=num_games)
        for i in pbar:
            records, winner, turns = generate_one_game_v2(
                net, fi, feat_means, feat_stds,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temperature_threshold=temperature_threshold,
                vps_to_win=vps_to_win,
                leaf_value_fn=leaf_value_fn,
            )
            all_records.extend(records)
            total_turns += turns
            _sp_games_done += 1
            pbar.set_postfix(avg_turns=f"{total_turns/_sp_games_done:.0f}", refresh=True)
            winners[winner] = winners.get(winner, 0) + 1

            if output_dir is not None:
                save_records(all_records, output_dir, quiet=True)
                with open(os.path.join(output_dir, '_progress.json'), 'w') as f:
                    json.dump({
                        'games_completed': i + 1,
                        'total_games': num_games,
                        'total_turns': total_turns,
                        'total_records': len(all_records),
                        'winners': {str(k): v for k, v in winners.items()},
                        'checkpoint': checkpoint_path,
                    }, f)
    else:
        # Parallel path — each worker loads its own model
        print(f"Using {workers} parallel workers for self-play generation")
        with mp.Pool(
            processes=workers,
            initializer=_az_worker_init,
            initargs=(checkpoint_path, num_simulations, c_puct,
                      temperature_threshold, vps_to_win, blend_bootstrap_path),
        ) as pool:
            game_results = []
            _sp_total_turns = 0
            _sp_games_done = 0
            pbar = tqdm(
                pool.imap_unordered(_az_worker_play_one, range(remaining)),
                total=remaining, desc="Self-play", initial=start_game,
            )
            for result in pbar:
                game_results.append(result)
                _sp_games_done += 1
                _sp_total_turns += result[4]  # turns
                avg_t = _sp_total_turns / _sp_games_done
                pbar.set_postfix(avg_turns=f"{avg_t:.0f}", refresh=True)

                # Incremental save — process this result immediately
                features_list, policies_list, values_list, winner_str, turns = result
                for feat, pol, val in zip(features_list, policies_list, values_list):
                    all_records.append(SelfPlayRecord(feat, pol, val))
                total_turns += turns
                if winner_str in ('None', 'null'):
                    winners[None] = winners.get(None, 0) + 1
                else:
                    name = winner_str.replace('Color.', '') if winner_str.startswith('Color.') else winner_str
                    winners[Color[name]] = winners.get(Color[name], 0) + 1

                if output_dir is not None:
                    save_records(all_records, output_dir, quiet=True)
                    with open(os.path.join(output_dir, '_progress.json'), 'w') as f:
                        json.dump({
                            'games_completed': start_game + _sp_games_done,
                            'total_games': num_games,
                            'total_turns': total_turns,
                            'total_records': len(all_records),
                            'winners': {str(k): v for k, v in winners.items()},
                            'checkpoint': checkpoint_path,
                        }, f)

    stats = {
        'num_games': num_games,
        'total_records': len(all_records),
        'avg_turns': total_turns / max(num_games, 1),
        'avg_records_per_game': len(all_records) / max(num_games, 1),
        'winners': {str(k): v for k, v in winners.items()},
    }
    return all_records, stats


def _save_graph_features(node_features_list, global_features_list, output_dir):
    """Save graph features (node + global) for GNN training.

    Accepts either a list of individual arrays or a pre-stacked numpy array.
    """
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(node_features_list, np.ndarray):
        nf = node_features_list
    else:
        nf = np.array(node_features_list, dtype=np.float32)
    if isinstance(global_features_list, np.ndarray):
        gf = global_features_list
    else:
        gf = np.array(global_features_list, dtype=np.float32)
    np.save(os.path.join(output_dir, 'node_features.npy'), nf)
    np.save(os.path.join(output_dir, 'global_features.npy'), gf)


def save_records(records, output_dir, quiet=False):
    """Save self-play records to disk."""
    os.makedirs(output_dir, exist_ok=True)

    features = np.array([r.features for r in records], dtype=np.float32)
    policies = np.array([r.policy_target for r in records], dtype=np.float32)
    values = np.array([r.value_target for r in records], dtype=np.float32)

    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'policies.npy'), policies)
    np.save(os.path.join(output_dir, 'values.npy'), values)

    if not quiet:
        print(f"Saved {len(records)} records to {output_dir}")
        print(f"  features: {features.shape}")
        print(f"  policies: {policies.shape}")
        print(f"  values: {values.shape}, mean={values.mean():.3f}")


def load_records(data_dir):
    """Load self-play records from disk."""
    features = np.load(os.path.join(data_dir, 'features.npy'))
    policies = np.load(os.path.join(data_dir, 'policies.npy'))
    values = np.load(os.path.join(data_dir, 'values.npy'))
    return features, policies, values


def load_graph_records(data_dir):
    """Load graph features (node + global) for GNN training.

    Returns:
        node_features: [N, 54, 18] float32
        global_features: [N, 76] float32
    """
    nf = np.load(os.path.join(data_dir, 'node_features.npy'))
    gf = np.load(os.path.join(data_dir, 'global_features.npy'))
    return nf, gf


def save_ranking_records(ranking_records, output_dir):
    """Save ranking data in CSR-style format.

    Args:
        ranking_records: list of (child_features[n_children, 176], child_scores[n_children])
        output_dir: directory to save to
    """
    os.makedirs(output_dir, exist_ok=True)
    all_feats = []
    all_scores = []
    offsets = [0]
    for child_feats, child_scores in ranking_records:
        all_feats.append(child_feats)
        all_scores.append(child_scores)
        offsets.append(offsets[-1] + len(child_scores))
    cat_feats = np.concatenate(all_feats)
    cat_scores = np.concatenate(all_scores)
    offsets_arr = np.array(offsets, dtype=np.int64)
    np.save(os.path.join(output_dir, 'ranking_child_features.npy'), cat_feats)
    np.save(os.path.join(output_dir, 'ranking_child_scores.npy'), cat_scores)
    np.save(os.path.join(output_dir, 'ranking_offsets.npy'), offsets_arr)
    print(f"Saved ranking data to {output_dir}: "
          f"{len(ranking_records)} positions, {len(cat_scores)} children total")


def load_ranking_records(data_dir):
    """Load ranking data from CSR-style format.

    Returns:
        child_feats: [total_children, 176] float32
        child_scores: [total_children] float32
        offsets: [num_positions + 1] int64
    """
    child_feats = np.load(os.path.join(data_dir, 'ranking_child_features.npy'))
    child_scores = np.load(os.path.join(data_dir, 'ranking_child_scores.npy'))
    offsets = np.load(os.path.join(data_dir, 'ranking_offsets.npy'))
    return child_feats, child_scores, offsets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class SelfPlayDataset(Dataset):
    """PyTorch dataset for self-play data.

    Supports mixed self-play + human data. Human samples have all-zero
    policy targets (policy loss is masked out for them).
    """

    def __init__(self, features, policies, values, means, stds):
        # Normalize features
        self.features = torch.tensor((features - means) / stds, dtype=torch.float32)
        self.policies = torch.tensor(policies, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.features[idx], self.policies[idx], self.values[idx]


def _load_human_data(parquet_dir, feature_names, max_samples=None):
    """Load human game data from parquet shards for AZ training.

    Extracts the 176 strategic features matching the AZ network, with
    value targets from game outcomes. Policy targets are all-zero
    (policy loss is skipped for human samples).

    Args:
        parquet_dir: Path to parquet shard directory (e.g., datasets/parquet)
        feature_names: List of 176 feature names from AZ checkpoint
        max_samples: Cap on number of human samples to load (None = all)

    Returns:
        features: (N, 176) float32
        policies: (N, 290) float32 — all zeros (no visit distribution)
        values: (N,) float32 — +1 win, -1 loss
    """
    import glob
    import pandas as pd

    f_cols = [f"F_{name}" for name in feature_names]
    read_cols = f_cols + ["winner"]

    shard_files = sorted(glob.glob(os.path.join(parquet_dir, "shard_*.parquet")))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.parquet files in {parquet_dir}")

    all_features = []
    all_values = []
    total = 0

    for path in shard_files:
        df = pd.read_parquet(path, columns=read_cols)
        feats = df[f_cols].values.astype(np.float32)
        # Human data: winner=1.0 means this perspective won
        # Map to AZ convention: +1 win, -1 loss
        vals = df["winner"].values.astype(np.float32) * 2.0 - 1.0

        all_features.append(feats)
        all_values.append(vals)
        total += len(vals)

        if max_samples and total >= max_samples:
            break

    features = np.concatenate(all_features)
    values = np.concatenate(all_values)

    if max_samples and len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        values = values[idx]

    # All-zero policies — policy loss will be masked out for these samples
    policies = np.zeros((len(features), ACTION_SPACE_SIZE), dtype=np.float32)

    print(f"  Loaded {len(features)} human samples from {parquet_dir}")
    return features, policies, values


def _train_gnn(checkpoint_path, data_dirs, output_path, epochs=200,
               batch_size=2048, lr=1e-3, value_weight=1.0,
               label_smoothing=0.0, scheduler='cosine',
               dropout=0.2, gnn_dims=None, edge_dropout=0.1,
               resume_training=None, max_samples=0):
    """Train GNN (CatanGNNet) on graph features.

    Loads node_features.npy [N, 54, 18] and global_features.npy [N, 76]
    from data_dirs. Creates a fresh GNN and trains with MSE value + CE policy loss.

    Data stays on CPU; only the current batch is moved to GPU. This allows
    training on datasets much larger than GPU memory (e.g., 2.5M samples on T4).
    """
    import gc

    # Pick best available device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Parse GNN dimensions
    if gnn_dims is None:
        gnn_dim, global_dim, body_dim = 32, 64, 96
    else:
        gnn_dim, global_dim, body_dim = gnn_dims

    # Load graph features from all data dirs
    all_node_features = []
    all_global_features = []
    all_policies = []
    all_values = []
    has_any_policy = False
    for d in data_dirs:
        nf_path = os.path.join(d, 'node_features.npy')
        gf_path = os.path.join(d, 'global_features.npy')
        if not os.path.exists(nf_path) or not os.path.exists(gf_path):
            raise FileNotFoundError(
                f"GNN training requires node_features.npy and global_features.npy in {d}. "
                f"Generate with: python -m robottler.az_selfplay expert --with-graph-features "
                f"or python -m datasets.extract_gnn_features")
        nf, gf = load_graph_records(d)
        all_node_features.append(nf)
        all_global_features.append(gf)

        # Load values (required)
        val_path = os.path.join(d, 'values.npy')
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"values.npy not found in {d}")
        v = np.load(val_path)
        all_values.append(v)

        # Load policies if available (human game data has no policy labels)
        pol_path = os.path.join(d, 'policies.npy')
        if os.path.exists(pol_path):
            all_policies.append(np.load(pol_path))
            has_any_policy = True
        else:
            all_policies.append(None)  # placeholder, don't allocate zeros
            print(f"  Note: no policies.npy in {d} — value-only training for this dir")

    node_features = np.concatenate(all_node_features).astype(np.float32)
    del all_node_features; gc.collect()
    global_features = np.concatenate(all_global_features).astype(np.float32)
    del all_global_features; gc.collect()
    values = np.concatenate(all_values).astype(np.float32)
    del all_values; gc.collect()

    # Only concatenate policies if at least one dir has them
    if has_any_policy:
        # Fill None entries with zeros for dirs that lacked policies
        for i, p in enumerate(all_policies):
            if p is None:
                n = len(values) if len(all_policies) == 1 else 0
                # Determine the length from node_features shape
                all_policies[i] = np.zeros(
                    (node_features.shape[0] if len(all_policies) == 1
                     else all_policies[i - 1].shape[0],  # fallback
                     ACTION_SPACE_SIZE), dtype=np.float32)
        policies = np.concatenate([p for p in all_policies if p is not None]).astype(np.float32)
    else:
        policies = None
    del all_policies; gc.collect()

    n_total = len(values)

    # Subsample if --max-samples specified (for limited-RAM environments)
    if max_samples > 0 and n_total > max_samples:
        print(f"  Subsampling {max_samples:,} from {n_total:,} samples")
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, max_samples, replace=False)
        idx.sort()
        node_features = node_features[idx]
        global_features = global_features[idx]
        values = values[idx]
        if policies is not None:
            policies = policies[idx]
        n_total = max_samples
        gc.collect()

    print(f"GNN training data: {n_total:,} samples from {len(data_dirs)} dir(s)")
    print(f"  Node features: {node_features.shape} "
          f"({node_features.nbytes / 1e9:.1f} GB)")
    print(f"  Global features: {global_features.shape}")
    print(f"  Policies: {'yes' if policies is not None else 'none (value-only)'}")
    print(f"  Value distribution: mean={values.mean():.3f}, std={values.std():.3f}")

    # Compute normalization stats (per-feature mean/std)
    # Node features: compute over [N*54, 18] (flatten node dimension)
    nf_flat = node_features.reshape(-1, NODE_FEAT_DIM)
    node_feat_means = nf_flat.mean(axis=0).astype(np.float32)
    node_feat_stds = nf_flat.std(axis=0).astype(np.float32)
    node_feat_stds[node_feat_stds < 1e-6] = 1.0  # avoid div by zero
    del nf_flat

    global_feat_means = global_features.mean(axis=0).astype(np.float32)
    global_feat_stds = global_features.std(axis=0).astype(np.float32)
    global_feat_stds[global_feat_stds < 1e-6] = 1.0

    # Normalize in-place
    node_features = (node_features - node_feat_means) / node_feat_stds
    global_features = (global_features - global_feat_means) / global_feat_stds

    # Create fresh GNN
    net = CatanGNNet(
        node_feat_dim=NODE_FEAT_DIM,
        global_feat_dim=GLOBAL_FEAT_DIM,
        num_actions=ACTION_SPACE_SIZE,
        gnn_dim=gnn_dim,
        global_dim=global_dim,
        body_dim=body_dim,
        gnn_layers=2,
        dropout=dropout,
        edge_dropout=edge_dropout,
    )
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  GNN: gnn_dim={gnn_dim}, global_dim={global_dim}, body_dim={body_dim}, "
          f"dropout={dropout}, edge_dropout={edge_dropout}, {total_params:,} params")
    print(f"  Data/param ratio: {n_total/total_params:.1f}:1")

    net.to(device)
    net.train()
    print(f"  Training on device: {device}")

    # Train/val split — keep data on CPU, move batches to GPU
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    perm = np.random.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    del perm

    # CPU tensors (no GPU copy of full dataset)
    train_nf = torch.from_numpy(node_features[train_idx])
    val_nf = torch.from_numpy(node_features[val_idx])
    del node_features; gc.collect()

    train_gf = torch.from_numpy(global_features[train_idx])
    val_gf = torch.from_numpy(global_features[val_idx])
    del global_features; gc.collect()

    train_val = torch.from_numpy(values[train_idx])
    val_val = torch.from_numpy(values[val_idx])
    del values; gc.collect()

    if policies is not None:
        train_pol = torch.from_numpy(policies[train_idx])
        val_pol = torch.from_numpy(policies[val_idx])
        del policies; gc.collect()
    else:
        train_pol = None
        val_pol = None

    del train_idx, val_idx; gc.collect()

    print(f"  {n_train:,} train + {n_val:,} val samples "
          f"(data on CPU, batches to {device})")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    if scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        lr_scheduler = None

    # Resume support
    start_epoch = 0
    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    history = []
    train_ckpt_path = output_path.replace('.pt', '_training.pt')

    if resume_training is not None and os.path.exists(resume_training):
        print(f"  Resuming training from {resume_training}")
        resume_ckpt = torch.load(resume_training, map_location='cpu', weights_only=False)
        net.load_state_dict(resume_ckpt['model_state_dict'])
        net.to(device)
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in resume_ckpt and lr_scheduler is not None:
            lr_scheduler.load_state_dict(resume_ckpt['lr_scheduler_state_dict'])
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_val_loss = resume_ckpt.get('best_val_loss', float('inf'))
        if 'best_state_dict' in resume_ckpt:
            best_state_dict = resume_ckpt['best_state_dict']
        patience_counter = resume_ckpt.get('patience_counter', 0)
        history = resume_ckpt.get('training_history', [])
        print(f"  Resuming from epoch {start_epoch}/{epochs} "
              f"(best_val_loss={best_val_loss:.4f}, patience={patience_counter}/20)")
        del resume_ckpt

    patience = 20

    def _gnn_loss(nf_b, gf_b, pol_b, val_b):
        """Compute GNN loss on a batch (all tensors already on device)."""
        v_pred, logits = net(nf_b, gf_b)
        v_pred = v_pred.squeeze(-1)
        v_loss = F.mse_loss(v_pred, val_b)
        if pol_b is not None:
            has_policy = pol_b.sum(dim=-1) > 0
            if has_policy.any():
                sp_logits = logits[has_policy]
                sp_policy = pol_b[has_policy]
                if label_smoothing > 0:
                    n_classes = sp_policy.shape[-1]
                    sp_policy = (1 - label_smoothing) * sp_policy + label_smoothing / n_classes
                log_probs = F.log_softmax(sp_logits, dim=-1).clamp(min=-100.0)
                p_loss = -(sp_policy * log_probs).sum(dim=-1).mean()
            else:
                p_loss = torch.tensor(0.0, device=nf_b.device)
        else:
            p_loss = torch.tensor(0.0, device=nf_b.device)
        return value_weight * v_loss + p_loss, v_loss, p_loss

    for epoch in range(start_epoch, epochs):
        net.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_p_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train)

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            # Move only this batch to device
            nf_b = train_nf[idx].to(device, non_blocking=True)
            gf_b = train_gf[idx].to(device, non_blocking=True)
            val_b = train_val[idx].to(device, non_blocking=True)
            pol_b = train_pol[idx].to(device, non_blocking=True) if train_pol is not None else None

            loss, v_loss, p_loss = _gnn_loss(nf_b, gf_b, pol_b, val_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            n_batches += 1

        if lr_scheduler is not None:
            lr_scheduler.step()
        avg_loss = total_loss / n_batches
        avg_v = total_v_loss / n_batches
        avg_p = total_p_loss / n_batches

        # Validation (in batches to avoid OOM)
        net.eval()
        with torch.no_grad():
            val_losses = []
            val_v_losses = []
            val_p_losses = []
            for vs in range(0, n_val, batch_size):
                ve = min(vs + batch_size, n_val)
                vnf = val_nf[vs:ve].to(device, non_blocking=True)
                vgf = val_gf[vs:ve].to(device, non_blocking=True)
                vvl = val_val[vs:ve].to(device, non_blocking=True)
                vpl = val_pol[vs:ve].to(device, non_blocking=True) if val_pol is not None else None
                vl, vv, vp = _gnn_loss(vnf, vgf, vpl, vvl)
                w = ve - vs
                val_losses.append(vl.item() * w)
                val_v_losses.append(vv.item() * w)
                val_p_losses.append(vp.item() * w)
            val_loss_val = sum(val_losses) / n_val
            val_v_val = sum(val_v_losses) / n_val
            val_p_val = sum(val_p_losses) / n_val

        hist_entry = {
            'epoch': epoch, 'loss': avg_loss, 'v_loss': avg_v, 'p_loss': avg_p,
            'val_loss': val_loss_val, 'val_v_loss': val_v_val, 'val_p_loss': val_p_val,
        }
        history.append(hist_entry)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
                  f"(v={avg_v:.4f}, p={avg_p:.4f}) | "
                  f"val={val_loss_val:.4f} (v={val_v_val:.4f}, p={val_p_val:.4f})")

        # Early stopping
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(val loss {val_loss_val:.4f} vs best {best_val_loss:.4f})")
                break

        # Save training checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            _save_ckpt = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_state_dict': best_state_dict,
                'patience_counter': patience_counter,
                'training_history': history,
            }
            if lr_scheduler is not None:
                _save_ckpt['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            torch.save(_save_ckpt, train_ckpt_path)
            del _save_ckpt

    # Clean up training checkpoint
    if os.path.exists(train_ckpt_path):
        os.remove(train_ckpt_path)

    # Restore best model
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
        best_epoch = min(history, key=lambda h: h.get('val_loss', float('inf')))
        print(f"  Restored best model from epoch {best_epoch['epoch']+1} "
              f"(val_loss={best_val_loss:.4f})")

    net.to('cpu').eval()
    save_checkpoint_gnn(
        net, optimizer, 1,
        node_feat_means, node_feat_stds,
        global_feat_means, global_feat_stds,
        output_path,
        extra={'training_history': history, 'num_samples': n_total,
               'best_val_loss': best_val_loss},
    )
    print(f"Saved GNN checkpoint to {output_path}")
    return {'history': history, 'iteration': 1, 'num_samples': n_total}


def train_on_data(checkpoint_path, data_dirs, output_path, epochs=5,
                  batch_size=256, lr=1e-4, value_weight=1.0,
                  human_data_dir=None, human_mix_ratio=0.5,
                  freeze_value=False, differential_lr=False,
                  label_smoothing=0.0, scheduler='cosine',
                  body_dims=None, dropout=0.0,
                  resume_training=None,
                  loss_asymmetry=0.0, negative_oversample=0.0,
                  ranking_data_dirs=None, ranking_weight=1.0,
                  ranking_margin=0.3,
                  gnn=False, gnn_dims=None, edge_dropout=0.1,
                  max_samples=0):
    """Train AlphaZero network on self-play data, optionally mixed with human data.

    Args:
        checkpoint_path: Path to current checkpoint (for architecture + normalization)
        data_dirs: List of data directories to load
        output_path: Where to save the trained checkpoint
        epochs: Number of training epochs (default 5 — nudge, don't memorize)
        batch_size: Batch size
        lr: Learning rate (default 1e-4 — conservative to preserve warm-start)
        value_weight: Weight of value loss vs policy loss
        human_data_dir: Path to parquet directory with human BC data.
            When set, human samples are mixed in at human_mix_ratio.
        human_mix_ratio: Fraction of training data that is human (default 0.5).
            Human samples contribute only value loss (no policy targets).
        freeze_value: If True, freeze value head (train policy only).
            Use for early iterations where the warm-start value head is
            better than what self-play can teach.
        differential_lr: If True, use per-group learning rates:
            body/value at lr*0.1, policy at lr*10. Protects warm-started
            weights while letting the cold-started policy head learn fast.
        label_smoothing: Smoothing factor for one-hot policy targets (0.0-1.0).
            Smoothed target = (1-ε)*target + ε/num_classes. Prevents
            overconfident logits and reduces overfitting on ExIt data.
        scheduler: LR scheduler type. 'cosine' (default) or 'constant'.
        body_dims: Tuple of body layer dimensions (e.g., (512, 256)).
            When specified, creates a fresh network with this architecture
            instead of loading weights from the checkpoint. The checkpoint
            is only used for feature normalization metadata.
        dropout: Dropout rate in body layers (default 0.0). Only used
            when body_dims is specified.
        resume_training: Path to a training checkpoint to resume from.
            Restores model weights, optimizer state, LR scheduler state,
            best val loss, and epoch counter. Resumes from last completed epoch.
        loss_asymmetry: Extra weight on value loss for losing positions (value < 0).
            A value of 2.0 means losses from losing positions are weighted 3x
            (1 + 2.0) vs 1x for winning positions. Default 0.0 (symmetric).
        negative_oversample: Target fraction of losing positions per batch.
            0.0 = uniform sampling (default). 0.6 = 60% of each batch is
            from losing positions. Uses weighted random sampling.
        ranking_data_dirs: List of directories containing ranking data
            (ranking_child_features.npy, ranking_child_scores.npy, ranking_offsets.npy).
            When provided, adds pairwise ranking loss to the training objective.
        ranking_weight: Weight of ranking loss relative to other losses (default 1.0).
        ranking_margin: Margin for MarginRankingLoss (default 0.3). The network
            must predict at least this gap between better and worse moves.
            Sweep values: 0.1 (conservative), 0.3 (moderate), 0.5 (aggressive).
        gnn: If True, train a GNN (CatanGNNet) instead of MLP (CatanAlphaZeroNet).
            Requires node_features.npy and global_features.npy in data_dirs.
        gnn_dims: Tuple of (gnn_dim, global_dim, body_dim) for GNN architecture.
            Default (32, 64, 96) = ~45K params.
        edge_dropout: Edge dropout rate for GNN DropEdge regularization (default 0.1).

    Returns:
        training stats dict
    """
    if gnn:
        return _train_gnn(
            checkpoint_path, data_dirs, output_path, epochs=epochs,
            batch_size=batch_size, lr=lr, value_weight=value_weight,
            label_smoothing=label_smoothing, scheduler=scheduler,
            dropout=dropout, gnn_dims=gnn_dims, edge_dropout=edge_dropout,
            resume_training=resume_training, max_samples=max_samples,
        )
    # Pick best available device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load metadata from checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if body_dims is not None:
        # Create fresh network with new architecture
        net = CatanAlphaZeroNet(
            input_dim=ckpt.get('input_dim', 176),
            num_actions=ckpt.get('num_actions', ACTION_SPACE_SIZE),
            body_dims=tuple(body_dims),
            dropout=dropout,
            shared_body=True,
        )
        total_params = sum(p.numel() for p in net.parameters())
        print(f"  Fresh network: body_dims={tuple(body_dims)}, dropout={dropout}, "
              f"shared_body=True, {total_params:,} params")
    else:
        net, _ = load_checkpoint(checkpoint_path)

    net.to(device)
    net.train()
    print(f"  Training on device: {device}")

    feat_names = ckpt['feature_names']
    feat_means = ckpt['feature_means']
    feat_stds = ckpt['feature_stds']
    iteration = 1 if body_dims is not None else ckpt.get('iteration', 0) + 1

    # Load self-play data
    all_features = []
    all_policies = []
    all_values = []
    for d in data_dirs:
        f, p, v = load_records(d)
        all_features.append(f)
        all_policies.append(p)
        all_values.append(v)

    sp_features = np.concatenate(all_features).astype(np.float32)
    del all_features
    sp_policies = np.concatenate(all_policies).astype(np.float32)
    del all_policies
    sp_values = np.concatenate(all_values).astype(np.float32)
    del all_values
    n_sp = len(sp_values)

    print(f"Self-play data: {n_sp} samples from {len(data_dirs)} dir(s)")
    print(f"  Value distribution: mean={sp_values.mean():.3f}, std={sp_values.std():.3f}")

    # Optionally mix in human data
    if human_data_dir is not None and human_mix_ratio > 0:
        n_human = int(n_sp * human_mix_ratio / (1 - human_mix_ratio))
        h_feat, h_pol, h_val = _load_human_data(
            human_data_dir, list(feat_names), max_samples=n_human)
        sp_features = np.concatenate([sp_features, h_feat.astype(np.float32)])
        del h_feat
        sp_policies = np.concatenate([sp_policies, h_pol.astype(np.float32)])
        del h_pol
        sp_values = np.concatenate([sp_values, h_val.astype(np.float32)])
        del h_val
        print(f"Mixed training: {n_sp} self-play + {len(sp_values)-n_sp} human "
              f"= {len(sp_values)} total")

    # Normalize features in-place
    f_means = np.asarray(feat_means, dtype=np.float32)
    f_stds = np.asarray(feat_stds, dtype=np.float32)
    sp_features -= f_means
    sp_features /= f_stds

    # Train/val split — convert one array at a time to minimize peak memory
    import gc
    n_total = len(sp_values)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    perm = torch.randperm(n_total)
    train_perm = perm[:n_train]
    val_perm = perm[n_train:]
    del perm

    # Features: numpy → torch (zero-copy) → index → to device → free numpy
    _t = torch.from_numpy(sp_features)
    train_feat = _t[train_perm].to(device)
    val_feat = _t[val_perm].to(device)
    del _t, sp_features; gc.collect()

    _t = torch.from_numpy(sp_policies)
    train_pol = _t[train_perm].to(device)
    val_pol = _t[val_perm].to(device)
    del _t, sp_policies; gc.collect()

    _t = torch.from_numpy(sp_values)
    train_val = _t[train_perm].to(device)
    val_val = _t[val_perm].to(device)
    del _t, sp_values, train_perm, val_perm; gc.collect()

    n_samples = n_train
    print(f"  {n_train} train + {n_val} val samples on {device}")

    # Load ranking data if provided
    rank_feats_t = None
    rank_scores_t = None
    rank_offsets_t = None
    n_rank_positions = 0
    if ranking_data_dirs:
        all_rank_feats = []
        all_rank_scores = []
        all_rank_offsets = [np.array([0], dtype=np.int64)]
        offset_base = 0
        for d in ranking_data_dirs:
            rf, rs, ro = load_ranking_records(d)
            all_rank_feats.append(rf)
            all_rank_scores.append(rs)
            # Shift offsets by accumulated base (skip first 0)
            all_rank_offsets.append(ro[1:] + offset_base)
            offset_base += len(rs)
        cat_rf = np.concatenate(all_rank_feats).astype(np.float32)
        cat_rs = np.concatenate(all_rank_scores).astype(np.float32)
        cat_ro = np.concatenate(all_rank_offsets)
        n_rank_positions = len(cat_ro) - 1
        # Normalize ranking features with same means/stds
        cat_rf -= f_means
        cat_rf /= f_stds
        rank_feats_t = torch.from_numpy(cat_rf).to(device)
        rank_scores_t = torch.from_numpy(cat_rs).to(device)
        rank_offsets_t = cat_ro  # keep on CPU for indexing
        del cat_rf, cat_rs, all_rank_feats, all_rank_scores, all_rank_offsets
        print(f"  Ranking data: {n_rank_positions} positions, "
              f"{len(rank_scores_t)} children, "
              f"margin={ranking_margin}, weight={ranking_weight}")

    # Freeze value path — what gets frozen depends on shared vs split body
    if freeze_value:
        if net.shared_body:
            # Shared body: freeze only value_head (body is shared, can't freeze it)
            for param in net.value_head.parameters():
                param.requires_grad = False
            trainable = [p for p in net.parameters() if p.requires_grad]
            print(f"  Value head FROZEN (shared body stays trainable) "
                  f"({sum(p.numel() for p in trainable)} trainable params)")
        else:
            # Split body: freeze value_body + value_head
            for param in net.value_body.parameters():
                param.requires_grad = False
            for param in net.value_head.parameters():
                param.requires_grad = False
            trainable = [p for p in net.parameters() if p.requires_grad]
            print(f"  Value path FROZEN — training policy_body + policy_head only "
                  f"({sum(p.numel() for p in trainable)} params)")

    if differential_lr and not freeze_value:
        # Protect warm-started value path, let policy path learn fast
        if net.shared_body:
            param_groups = [
                {'params': list(net.body.parameters()), 'lr': lr * 0.1},
                {'params': list(net.value_head.parameters()), 'lr': lr * 0.1},
                {'params': list(net.policy_head.parameters()), 'lr': lr * 10},
            ]
        else:
            param_groups = [
                {'params': list(net.value_body.parameters()), 'lr': lr * 0.1},
                {'params': list(net.value_head.parameters()), 'lr': lr * 0.1},
                {'params': list(net.policy_body.parameters()), 'lr': lr * 10},
                {'params': list(net.policy_head.parameters()), 'lr': lr * 10},
            ]
        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
        print(f"  Differential LR: value={lr*0.1:.1e}, policy={lr*10:.1e}")
    elif freeze_value:
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    if scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        lr_scheduler = None  # constant LR

    if label_smoothing > 0:
        print(f"  Label smoothing: {label_smoothing}")
    if loss_asymmetry > 0:
        print(f"  Asymmetric value loss: losing positions weighted {1+loss_asymmetry:.1f}x")
    if negative_oversample > 0:
        print(f"  Negative oversampling: target {negative_oversample*100:.0f}% losing positions per batch")

    # Resume from training checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    history = []

    if resume_training is not None and os.path.exists(resume_training):
        print(f"  Resuming training from {resume_training}")
        resume_ckpt = torch.load(resume_training, map_location='cpu', weights_only=False)
        net.load_state_dict(resume_ckpt['model_state_dict'])
        net.to(device)
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in resume_ckpt and lr_scheduler is not None:
            lr_scheduler.load_state_dict(resume_ckpt['lr_scheduler_state_dict'])
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_val_loss = resume_ckpt.get('best_val_loss', float('inf'))
        if 'best_state_dict' in resume_ckpt:
            best_state_dict = resume_ckpt['best_state_dict']
        patience_counter = resume_ckpt.get('patience_counter', 0)
        history = resume_ckpt.get('training_history', [])
        print(f"  Resuming from epoch {start_epoch}/{epochs} "
              f"(best_val_loss={best_val_loss:.4f}, patience={patience_counter}/20)")
        del resume_ckpt

    # Training checkpoint path (for resume support)
    train_ckpt_path = output_path.replace('.pt', '_training.pt')

    def _compute_loss(feat_b, pol_b, val_b):
        """Compute combined loss on a batch."""
        v_pred, logits = net(feat_b)
        v_pred = v_pred.squeeze(-1)
        if loss_asymmetry > 0:
            # Per-sample MSE with higher weight on losing positions
            per_sample_mse = (v_pred - val_b) ** 2
            weights = torch.where(val_b < 0,
                                  1.0 + loss_asymmetry,
                                  torch.ones_like(val_b))
            v_loss = (per_sample_mse * weights).mean()
        else:
            v_loss = F.mse_loss(v_pred, val_b)
        has_policy = pol_b.sum(dim=-1) > 0
        if has_policy.any():
            sp_logits = logits[has_policy]
            sp_policy = pol_b[has_policy]
            if label_smoothing > 0:
                n_classes = sp_policy.shape[-1]
                sp_policy = (1 - label_smoothing) * sp_policy + label_smoothing / n_classes
            log_probs = F.log_softmax(sp_logits, dim=-1).clamp(min=-100.0)
            p_loss = -(sp_policy * log_probs).sum(dim=-1).mean()
        else:
            p_loss = torch.tensor(0.0, device=feat_b.device)
        return value_weight * v_loss + p_loss, v_loss, p_loss

    def _compute_ranking_loss(position_indices):
        """Compute pairwise ranking loss for a batch of positions.

        Batches all children from all positions into a single forward pass,
        then splits by position for pairwise loss computation.
        """
        # Gather all children into one contiguous batch
        slices = []
        for pos_idx in position_indices:
            s = rank_offsets_t[pos_idx]
            e = rank_offsets_t[pos_idx + 1]
            if e - s >= 2:
                slices.append((s, e))
        if not slices:
            return torch.tensor(0.0, device=device)

        # Single forward pass for all children
        all_indices = torch.cat([torch.arange(s, e, device=device) for s, e in slices])
        all_feats = rank_feats_t[all_indices]
        all_v, _ = net(all_feats)
        all_v = all_v.squeeze(-1)
        all_scores = rank_scores_t[all_indices]

        # Split back by position and compute pairwise losses
        total_rloss = 0.0
        n_pairs = 0
        offset = 0
        for s, e in slices:
            n_children = e - s
            v_pred = all_v[offset:offset + n_children]
            scores = all_scores[offset:offset + n_children]
            offset += n_children
            # All pairs: score_diff[i,j] > 0 means i is better than j
            score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)
            pred_diff = v_pred.unsqueeze(0) - v_pred.unsqueeze(1)
            mask = score_diff > 0
            if mask.sum() == 0:
                continue
            pair_loss = F.relu(ranking_margin - pred_diff[mask])
            total_rloss = total_rloss + pair_loss.sum()
            n_pairs = n_pairs + mask.sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=device)
        return total_rloss / n_pairs

    patience = 20

    # Precompute sampling weights for negative oversampling
    if negative_oversample > 0:
        is_neg = (train_val < 0).float()
        n_neg = is_neg.sum().item()
        n_pos = n_samples - n_neg
        if n_neg > 0 and n_pos > 0:
            # Set weights so that negative fraction in expectation = negative_oversample
            # w_neg / (w_neg * n_neg + w_pos * n_pos) = target_frac
            # Let w_pos = 1, solve for w_neg:
            target_frac = negative_oversample
            w_neg = target_frac * n_pos / ((1 - target_frac) * n_neg)
            sample_weights = torch.where(is_neg > 0, w_neg, 1.0)
            print(f"  Negative oversampling: {n_neg}/{n_samples} neg samples "
                  f"({n_neg/n_samples*100:.1f}%), target={target_frac*100:.0f}%, "
                  f"w_neg={w_neg:.2f}")
        else:
            sample_weights = None
            print(f"  Warning: negative_oversample={negative_oversample} but "
                  f"n_neg={n_neg}, n_pos={n_pos} — using uniform sampling")
    else:
        sample_weights = None

    # Ranking batch size: sample positions per training batch.
    # Each position has ~10 children avg, so 64 positions = ~640 children per forward pass.
    rank_positions_per_batch = min(64, n_rank_positions) if n_rank_positions > 0 else 0

    for epoch in range(start_epoch, epochs):
        net.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_p_loss = 0.0
        total_r_loss = 0.0
        n_batches = 0

        # Shuffle training indices each epoch (weighted or uniform)
        if sample_weights is not None:
            # multinomial inherits device from sample_weights (same as train_val)
            perm = torch.multinomial(sample_weights, n_samples, replacement=True)
        else:
            perm = torch.randperm(n_samples, device=device)

        # Shuffle ranking position indices
        if n_rank_positions > 0:
            rank_perm = np.random.permutation(n_rank_positions)
            rank_idx_ptr = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            loss, v_loss, p_loss = _compute_loss(
                train_feat[idx], train_pol[idx], train_val[idx])

            # Add ranking loss if available
            r_loss_val = 0.0
            if n_rank_positions > 0 and rank_positions_per_batch > 0:
                # Get next batch of ranking positions (wrap around)
                if rank_idx_ptr + rank_positions_per_batch > n_rank_positions:
                    rank_perm = np.random.permutation(n_rank_positions)
                    rank_idx_ptr = 0
                pos_batch = rank_perm[rank_idx_ptr:rank_idx_ptr + rank_positions_per_batch]
                rank_idx_ptr += rank_positions_per_batch
                r_loss = _compute_ranking_loss(pos_batch)
                loss = loss + ranking_weight * r_loss
                r_loss_val = r_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            total_r_loss += r_loss_val
            n_batches += 1

        if lr_scheduler is not None:
            lr_scheduler.step()
        avg_loss = total_loss / n_batches
        avg_v = total_v_loss / n_batches
        avg_p = total_p_loss / n_batches
        avg_r = total_r_loss / n_batches

        # Validation loss
        net.eval()
        with torch.no_grad():
            val_loss, val_v, val_p = _compute_loss(val_feat, val_pol, val_val)
            val_loss_val = val_loss.item()
            val_v_val = val_v.item()
            val_p_val = val_p.item()

        hist_entry = {
            'epoch': epoch, 'loss': avg_loss, 'v_loss': avg_v, 'p_loss': avg_p,
            'val_loss': val_loss_val, 'val_v_loss': val_v_val, 'val_p_loss': val_p_val,
        }
        if n_rank_positions > 0:
            hist_entry['r_loss'] = avg_r
        history.append(hist_entry)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            r_str = f", r={avg_r:.4f}" if n_rank_positions > 0 else ""
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
                  f"(v={avg_v:.4f}, p={avg_p:.4f}{r_str}) | "
                  f"val={val_loss_val:.4f} (v={val_v_val:.4f}, p={val_p_val:.4f})")

        # Early stopping check
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(val loss {val_loss_val:.4f} vs best {best_val_loss:.4f})")
                break

        # Save training checkpoint every 5 epochs (for resume on Ctrl+C)
        if (epoch + 1) % 5 == 0:
            _save_train_ckpt = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_state_dict': best_state_dict,
                'patience_counter': patience_counter,
                'training_history': history,
            }
            if lr_scheduler is not None:
                _save_train_ckpt['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            torch.save(_save_train_ckpt, train_ckpt_path)
            del _save_train_ckpt

    # Clean up training checkpoint
    if os.path.exists(train_ckpt_path):
        os.remove(train_ckpt_path)

    # Restore best model
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
        best_epoch = min(history, key=lambda h: h.get('val_loss', float('inf')))
        print(f"  Restored best model from epoch {best_epoch['epoch']+1} "
              f"(val_loss={best_val_loss:.4f})")

    # Unfreeze value path before saving (so it's trainable in next iteration)
    if freeze_value:
        if not net.shared_body:
            for param in net.value_body.parameters():
                param.requires_grad = True
        for param in net.value_head.parameters():
            param.requires_grad = True

    net.to('cpu').eval()
    n_total_samples = n_train + n_val
    save_checkpoint(
        net, optimizer, iteration,
        feat_names, feat_means, feat_stds,
        output_path,
        extra={'training_history': history, 'num_samples': n_total_samples,
               'freeze_value': freeze_value, 'differential_lr': differential_lr,
               'human_samples': n_total_samples - n_sp if human_data_dir else 0,
               'best_val_loss': best_val_loss,
               'loss_asymmetry': loss_asymmetry,
               'negative_oversample': negative_oversample,
               'ranking_weight': ranking_weight if ranking_data_dirs else 0,
               'ranking_margin': ranking_margin if ranking_data_dirs else 0,
               'ranking_positions': n_rank_positions},
    )
    print(f"Saved trained checkpoint to {output_path} (iteration {iteration})")

    return {'history': history, 'iteration': iteration, 'num_samples': n_total_samples}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_eval_worker_new_net = None
_eval_worker_old_net = None
_eval_worker_fi_map = None
_eval_worker_means = None
_eval_worker_stds = None
_eval_worker_sims = None
_eval_worker_c_puct = None


def _eval_worker_init(new_path, old_path, num_simulations, c_puct):
    """Initialize per-worker state for evaluation."""
    global _eval_worker_new_net, _eval_worker_old_net
    global _eval_worker_fi_map, _eval_worker_means, _eval_worker_stds
    global _eval_worker_sims, _eval_worker_c_puct

    new_net, new_ckpt = load_checkpoint(new_path)
    old_net, _ = load_checkpoint(old_path)

    feat_names = new_ckpt['feature_names']
    _eval_worker_means = new_ckpt['feature_means']
    _eval_worker_stds = new_ckpt['feature_stds']
    _eval_worker_fi_map = {name: i for i, name in enumerate(feat_names)}
    _eval_worker_new_net = new_net
    _eval_worker_old_net = old_net
    _eval_worker_sims = num_simulations
    _eval_worker_c_puct = c_puct


def _eval_worker_play_one(game_idx):
    """Play one evaluation game. Returns 1 if new wins, 0 otherwise."""
    from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
    catan_map = build_map(BASE_MAP_TEMPLATE)
    fi = FeatureIndexer(_eval_worker_fi_map, catan_map)

    if game_idx % 2 == 0:
        p_new = BBMCTSPlayer(
            Color.RED, _eval_worker_new_net, fi,
            _eval_worker_means, _eval_worker_stds,
            num_simulations=_eval_worker_sims, c_puct=_eval_worker_c_puct,
            temperature=0.0, dirichlet_alpha=0.0,
        )
        p_old = BBMCTSPlayer(
            Color.BLUE, _eval_worker_old_net, fi,
            _eval_worker_means, _eval_worker_stds,
            num_simulations=_eval_worker_sims, c_puct=_eval_worker_c_puct,
            temperature=0.0, dirichlet_alpha=0.0,
        )
        new_color = Color.RED
    else:
        p_new = BBMCTSPlayer(
            Color.BLUE, _eval_worker_new_net, fi,
            _eval_worker_means, _eval_worker_stds,
            num_simulations=_eval_worker_sims, c_puct=_eval_worker_c_puct,
            temperature=0.0, dirichlet_alpha=0.0,
        )
        p_old = BBMCTSPlayer(
            Color.RED, _eval_worker_old_net, fi,
            _eval_worker_means, _eval_worker_stds,
            num_simulations=_eval_worker_sims, c_puct=_eval_worker_c_puct,
            temperature=0.0, dirichlet_alpha=0.0,
        )
        new_color = Color.BLUE

    game = Game(players=[p_new, p_old] if new_color == Color.RED
                else [p_old, p_new], vps_to_win=10)
    fi.update_map(game.state.board.map)

    turns = 0
    while game.winning_color() is None and turns < 1000:
        actions = game.playable_actions
        current = game.state.current_player()
        action = current.decide(game, actions)
        game.execute(action)
        turns += 1

    return 1 if game.winning_color() == new_color else 0


def evaluate_checkpoints(new_path, old_path, num_games=200,
                         num_simulations=400, c_puct=1.4, workers=1):
    """Evaluate new checkpoint vs old checkpoint.

    Both play as MCTS with their respective networks. New plays as both
    colors (half the games each). Returns win rate of new.
    """
    if workers > 1:
        print(f"  Using {workers} parallel workers for evaluation")
        with mp.Pool(
            processes=workers,
            initializer=_eval_worker_init,
            initargs=(new_path, old_path, num_simulations, c_puct),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(_eval_worker_play_one, range(num_games)),
                total=num_games, desc="Evaluating",
            ))
        new_wins = sum(results)
    else:
        new_net, new_ckpt = load_checkpoint(new_path)
        old_net, _ = load_checkpoint(old_path)

        feat_names = new_ckpt['feature_names']
        feat_means = new_ckpt['feature_means']
        feat_stds = new_ckpt['feature_stds']
        feature_index_map = {name: i for i, name in enumerate(feat_names)}

        from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
        catan_map = build_map(BASE_MAP_TEMPLATE)

        new_wins = 0
        for i in tqdm(range(num_games), desc="Evaluating"):
            fi_copy = FeatureIndexer(feature_index_map, catan_map)

            if i % 2 == 0:
                p_new = BBMCTSPlayer(
                    Color.RED, new_net, fi_copy, feat_means, feat_stds,
                    num_simulations=num_simulations, c_puct=c_puct,
                    temperature=0.0, dirichlet_alpha=0.0,
                )
                p_old = BBMCTSPlayer(
                    Color.BLUE, old_net, fi_copy, feat_means, feat_stds,
                    num_simulations=num_simulations, c_puct=c_puct,
                    temperature=0.0, dirichlet_alpha=0.0,
                )
                new_color = Color.RED
            else:
                p_new = BBMCTSPlayer(
                    Color.BLUE, new_net, fi_copy, feat_means, feat_stds,
                    num_simulations=num_simulations, c_puct=c_puct,
                    temperature=0.0, dirichlet_alpha=0.0,
                )
                p_old = BBMCTSPlayer(
                    Color.RED, old_net, fi_copy, feat_means, feat_stds,
                    num_simulations=num_simulations, c_puct=c_puct,
                    temperature=0.0, dirichlet_alpha=0.0,
                )
                new_color = Color.BLUE

            game = Game(players=[p_new, p_old] if new_color == Color.RED
                        else [p_old, p_new], vps_to_win=10)
            fi_copy.update_map(game.state.board.map)

            turns = 0
            while game.winning_color() is None and turns < 1000:
                actions = game.playable_actions
                current = game.state.current_player()
                action = current.decide(game, actions)
                game.execute(action)
                turns += 1

            if game.winning_color() == new_color:
                new_wins += 1

    win_rate = new_wins / num_games
    ci = 1.96 * (win_rate * (1 - win_rate) / num_games) ** 0.5
    print(f"\nNew vs Old: {win_rate*100:.1f}% [{(win_rate-ci)*100:.1f}%, {(win_rate+ci)*100:.1f}%]")
    print(f"  ({new_wins}/{num_games} wins)")

    return win_rate, ci


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def _load_loop_state(output_dir):
    """Load persistent loop state, or return defaults."""
    state_path = os.path.join(output_dir, '_loop_state.json')
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return None


def _save_loop_state(output_dir, state):
    """Atomically save loop state (write to tmp then rename)."""
    state_path = os.path.join(output_dir, '_loop_state.json')
    tmp_path = state_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def run_loop(start_checkpoint, iterations, games_per_iter, num_simulations,
             output_dir, c_puct=1.4, epochs=5, batch_size=256, lr=1e-4,
             eval_games=100, accept_threshold=0.55, window_size=5, workers=1,
             blend_bootstrap_path=None, human_data_dir=None,
             human_mix_ratio=0.5, freeze_value=False, accept_all=False,
             expert=False, bc_model=None, search_depth=2, dice_sample_size=5,
             differential_lr=False, label_smoothing=0.0, scheduler='cosine'):
    """Run the full AlphaZero training loop. Fully resumable.

    State is persisted to output_dir/_loop_state.json after each step.
    If interrupted, re-running with the same arguments resumes from
    the last completed step.

    Each iteration has 3 phases tracked by 'phase':
        'generate' → 'train' → 'evaluate' → next iteration

    For each iteration:
    1. Generate self-play games with current best network
    2. Train new network on recent self-play data
    3. Evaluate new vs current best
    4. Accept if win rate > threshold

    Args:
        blend_bootstrap_path: Path to BC value net for iteration 1 bootstrap.
            When set, iteration 1 self-play uses the BC neural net as MCTS
            leaf evaluator. Iterations 2+ use the trained AZ network.
        human_data_dir: Path to parquet dir with human BC data for mixing.
        human_mix_ratio: Fraction of training data that is human (default 0.5).
        freeze_value: If True, freeze value head during training.
        expert: If True, use Expert Iteration (BitboardSearchPlayer) for
            data generation instead of MCTS self-play. Auto-sets accept_all
            since rejection doesn't make sense (expert quality is fixed).
        bc_model: Path to BC value net for expert search player blend.
            Required when expert=True.
        search_depth: Search depth for expert player (default 2).
        dice_sample_size: Dice outcomes for expert player (default 5).
        differential_lr: If True, use differential learning rates.
    """
    if expert:
        accept_all = True  # Expert data quality is fixed; rejection is meaningless

    os.makedirs(output_dir, exist_ok=True)

    # Load or initialize loop state
    state = _load_loop_state(output_dir)
    if state is not None:
        current_best = state['current_best']
        start_iter = state['next_iteration']
        phase = state.get('phase', 'generate')
        print(f"Resuming from iteration {start_iter}, phase '{phase}'")
        print(f"  Current best: {current_best}")
    else:
        current_best = start_checkpoint
        start_iter = 1
        phase = 'generate'
        _save_loop_state(output_dir, {
            'current_best': current_best,
            'next_iteration': start_iter,
            'phase': phase,
            'start_checkpoint': start_checkpoint,
        })

    for it in range(start_iter, iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {it}/{iterations}")
        print(f"{'='*60}")

        iter_data_dir = os.path.join(output_dir, f"iter{it}")
        new_ckpt_path = os.path.join(output_dir, f"az_iter{it}.pt")

        # --- Phase 1: Generate ---
        if phase == 'generate':
            if expert:
                print(f"\n[1/3] Generating {games_per_iter} expert games "
                      f"(search depth={search_depth}, dice_sample={dice_sample_size})...")
                records, stats = generate_expert_games(
                    bc_model, games_per_iter,
                    search_depth=search_depth, dice_sample_size=dice_sample_size,
                    output_dir=iter_data_dir, workers=workers,
                )
            else:
                # Blend bootstrap only for iteration 1
                iter_bootstrap = blend_bootstrap_path if it == 1 else None
                if iter_bootstrap:
                    print(f"\n[1/3] Generating {games_per_iter} self-play games "
                          f"(blend bootstrap from {iter_bootstrap})...")
                else:
                    print(f"\n[1/3] Generating {games_per_iter} self-play games...")
                records, stats = generate_games(
                    current_best, games_per_iter,
                    num_simulations=num_simulations, c_puct=c_puct,
                    output_dir=iter_data_dir, workers=workers,
                    blend_bootstrap_path=iter_bootstrap,
                )
            # Final save with summary
            save_records(records, iter_data_dir)
            print(f"  Stats: {stats}")

            phase = 'train'
            _save_loop_state(output_dir, {
                'current_best': current_best,
                'next_iteration': it,
                'phase': phase,
                'start_checkpoint': start_checkpoint,
            })

        # --- Phase 2: Train ---
        if phase == 'train':
            data_dirs = []
            for past_it in range(max(1, it - window_size + 1), it + 1):
                d = os.path.join(output_dir, f"iter{past_it}")
                if os.path.exists(os.path.join(d, 'features.npy')):
                    data_dirs.append(d)

            print(f"\n[2/3] Training on {len(data_dirs)} data dir(s)...")
            train_stats = train_on_data(
                current_best, data_dirs, new_ckpt_path,
                epochs=epochs, batch_size=batch_size, lr=lr,
                human_data_dir=human_data_dir,
                human_mix_ratio=human_mix_ratio,
                freeze_value=freeze_value,
                differential_lr=differential_lr,
                label_smoothing=label_smoothing,
                scheduler=scheduler,
            )

            phase = 'evaluate'
            _save_loop_state(output_dir, {
                'current_best': current_best,
                'next_iteration': it,
                'phase': phase,
                'new_ckpt_path': new_ckpt_path,
                'start_checkpoint': start_checkpoint,
            })

        # --- Phase 3: Evaluate ---
        if phase == 'evaluate':
            if accept_all:
                print(f"\n[3/3] Auto-accepting (--accept-all)")
                win_rate, ci = 1.0, 0.0
                current_best = new_ckpt_path
                print(f"  ACCEPTED (always-accept mode)")
            else:
                print(f"\n[3/3] Evaluating new vs current best ({eval_games} games)...")
                win_rate, ci = evaluate_checkpoints(
                    new_ckpt_path, current_best,
                    num_games=eval_games, num_simulations=num_simulations,
                    c_puct=c_puct, workers=workers,
                )

                if win_rate >= accept_threshold:
                    print(f"  ACCEPTED (win rate {win_rate*100:.1f}% >= {accept_threshold*100:.0f}%)")
                    current_best = new_ckpt_path
                else:
                    print(f"  REJECTED (win rate {win_rate*100:.1f}% < {accept_threshold*100:.0f}%)")

            # Save iteration metadata
            meta_path = os.path.join(output_dir, f"iter{it}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump({
                    'iteration': it,
                    'games': games_per_iter,
                    'win_rate_vs_previous': win_rate,
                    'accepted': win_rate >= accept_threshold,
                    'current_best': current_best,
                    'new_ckpt': new_ckpt_path,
                }, f, indent=2)

            # Advance to next iteration
            phase = 'generate'
            _save_loop_state(output_dir, {
                'current_best': current_best,
                'next_iteration': it + 1,
                'phase': phase,
                'start_checkpoint': start_checkpoint,
            })

    print(f"\n{'='*60}")
    print(f"Training loop complete. Best checkpoint: {current_best}")
    return current_best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AlphaZero self-play and training")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Generate
    gen_parser = subparsers.add_parser('generate', help='Generate self-play games')
    gen_parser.add_argument('--checkpoint', required=True, help='Network checkpoint')
    gen_parser.add_argument('--games', type=int, default=500)
    gen_parser.add_argument('--sims', type=int, default=400)
    gen_parser.add_argument('--c-puct', type=float, default=1.4)
    gen_parser.add_argument('--output-dir', required=True)
    gen_parser.add_argument('--workers', type=int, default=1,
                            help='Parallel worker processes (default: 1)')
    gen_parser.add_argument('--blend-bootstrap', default=None, metavar='PATH',
                            help='Path to BC value net (e.g., value_net_v2.pt) for '
                                 'leaf evaluation bootstrap. Uses BC neural net instead '
                                 'of AZ network for MCTS value estimation.')

    # Train
    train_parser = subparsers.add_parser('train', help='Train on self-play data')
    train_parser.add_argument('--checkpoint', default=None,
                              help='Network checkpoint (required for MLP, optional for --gnn)')
    train_parser.add_argument('--data-dir', nargs='+', required=True)
    train_parser.add_argument('--output', required=True)
    train_parser.add_argument('--epochs', type=int, default=5)
    train_parser.add_argument('--batch-size', type=int, default=256)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--value-weight', type=float, default=1.0)
    train_parser.add_argument('--human-data', default=None, metavar='PATH',
                              help='Parquet dir with human BC data to mix in')
    train_parser.add_argument('--human-mix-ratio', type=float, default=0.5,
                              help='Fraction of training data from human games (default 0.5)')
    train_parser.add_argument('--freeze-value', action='store_true',
                              help='Freeze value head (train body + policy only)')
    train_parser.add_argument('--differential-lr', action='store_true',
                              help='Use differential LR: body/value at lr*0.1, '
                                   'policy at lr*10 (protects warm-start)')
    train_parser.add_argument('--label-smoothing', type=float, default=0.0,
                              help='Label smoothing for policy targets (default 0.0)')
    train_parser.add_argument('--scheduler', choices=['cosine', 'constant'],
                              default='cosine', help='LR scheduler (default: cosine)')
    train_parser.add_argument('--body-dims', default=None,
                              help='Body layer dimensions as comma-separated ints '
                                   '(e.g., "512,256"). Creates a fresh network instead '
                                   'of loading weights from checkpoint.')
    train_parser.add_argument('--dropout', type=float, default=0.0,
                              help='Dropout rate in body layers (default 0.0). '
                                   'Only used with --body-dims.')
    train_parser.add_argument('--resume-training', default=None, metavar='PATH',
                              help='Resume from a training checkpoint (_training.pt). '
                                   'Restores model, optimizer, scheduler, and epoch.')
    train_parser.add_argument('--loss-asymmetry', type=float, default=0.0,
                              help='Extra weight on value loss for losing positions. '
                                   '2.0 = losing positions weighted 3x (default: 0.0)')
    train_parser.add_argument('--negative-oversample', type=float, default=0.0,
                              help='Target fraction of losing positions per batch. '
                                   '0.6 = 60%% losing samples per batch (default: 0.0)')
    train_parser.add_argument('--ranking-data', nargs='+', default=None, metavar='DIR',
                              help='Directories containing ranking data '
                                   '(ranking_child_features.npy, ranking_child_scores.npy, '
                                   'ranking_offsets.npy). Adds pairwise ranking loss.')
    train_parser.add_argument('--ranking-weight', type=float, default=1.0,
                              help='Weight of ranking loss (default: 1.0)')
    train_parser.add_argument('--ranking-margin', type=float, default=0.3,
                              help='Margin for MarginRankingLoss (default: 0.3). '
                                   'Sweep: 0.1 (conservative), 0.3 (moderate), 0.5 (aggressive)')
    train_parser.add_argument('--gnn', action='store_true',
                              help='Train a GNN (CatanGNNet) instead of MLP. '
                                   'Requires node_features.npy and global_features.npy in data dirs.')
    train_parser.add_argument('--gnn-dims', default=None,
                              help='GNN dimensions as "gnn_dim,global_dim,body_dim" '
                                   '(e.g., "32,64,96"). Default: 32,64,96 (~45K params).')
    train_parser.add_argument('--edge-dropout', type=float, default=0.1,
                              help='Edge dropout (DropEdge) rate for GNN training (default 0.1).')
    train_parser.add_argument('--max-samples', type=int, default=0,
                              help='Max training samples (0 = use all). '
                                   'Use 500000-1000000 for Colab T4 to fit in RAM.')

    # Expert iteration data generation
    expert_parser = subparsers.add_parser('expert',
                                          help='Generate expert iteration data')
    expert_parser.add_argument('--bc-model', required=True,
                               help='BC value net (e.g., value_net_v2.pt) for blend')
    expert_parser.add_argument('--games', type=int, default=1000)
    expert_parser.add_argument('--search-depth', type=int, default=2)
    expert_parser.add_argument('--dice-sample', type=int, default=5)
    expert_parser.add_argument('--output-dir', required=True)
    expert_parser.add_argument('--workers', type=int, default=1,
                               help='Parallel worker processes (default: 1)')
    expert_parser.add_argument('--distill-values', action='store_true',
                               help='Record blend function evaluation as value target '
                                    'instead of binary game outcome. Produces continuous '
                                    'value targets in [-1, +1] for knowledge distillation.')
    expert_parser.add_argument('--with-ranking', action='store_true',
                               help='Generate ranking data: evaluate all legal child states '
                                    'at each decision point with the blend function. '
                                    'Saves ranking_child_features.npy, ranking_child_scores.npy, '
                                    'ranking_offsets.npy for pairwise ranking loss training.')
    expert_parser.add_argument('--with-graph-features', action='store_true',
                               help='Extract per-node [54, 18] and global [76] features '
                                    'at each decision point for GNN training. '
                                    'Saves node_features.npy and global_features.npy.')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate new vs old')
    eval_parser.add_argument('--new-checkpoint', required=True)
    eval_parser.add_argument('--old-checkpoint', required=True)
    eval_parser.add_argument('--games', type=int, default=200)
    eval_parser.add_argument('--sims', type=int, default=400)
    eval_parser.add_argument('--workers', type=int, default=1,
                             help='Parallel worker processes (default: 1)')

    # Full loop
    loop_parser = subparsers.add_parser('loop', help='Full training loop')
    loop_parser.add_argument('--start-checkpoint', required=True)
    loop_parser.add_argument('--iterations', type=int, default=10)
    loop_parser.add_argument('--games-per-iter', type=int, default=500)
    loop_parser.add_argument('--sims', type=int, default=400)
    loop_parser.add_argument('--output-dir', required=True)
    loop_parser.add_argument('--epochs', type=int, default=5)
    loop_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate (default 1e-4)')
    loop_parser.add_argument('--eval-games', type=int, default=100)
    loop_parser.add_argument('--workers', type=int, default=1,
                             help='Parallel worker processes (default: 1)')
    loop_parser.add_argument('--blend-bootstrap', default=None, metavar='PATH',
                             help='Path to BC value net for iteration 1 bootstrap. '
                                  'Uses BC neural net as MCTS leaf evaluator for the '
                                  'first iteration only; subsequent iterations use the '
                                  'trained AZ network.')
    loop_parser.add_argument('--human-data', default=None, metavar='PATH',
                             help='Parquet dir with human BC data to mix in during training')
    loop_parser.add_argument('--human-mix-ratio', type=float, default=0.5,
                             help='Fraction of training data from human games (default 0.5)')
    loop_parser.add_argument('--freeze-value', action='store_true',
                             help='Freeze value head (train body + policy only)')
    loop_parser.add_argument('--accept-all', action='store_true',
                             help='Always accept new model (skip evaluation). '
                                  'Follows AlphaZero approach of always using latest model.')
    loop_parser.add_argument('--expert', action='store_true',
                             help='Use Expert Iteration: generate data with '
                                  'BitboardSearchPlayer instead of MCTS self-play')
    loop_parser.add_argument('--bc-model', default=None, metavar='PATH',
                             help='BC value net for expert search player blend '
                                  '(required with --expert)')
    loop_parser.add_argument('--search-depth', type=int, default=2,
                             help='Search depth for expert player (default 2)')
    loop_parser.add_argument('--dice-sample', type=int, default=5,
                             help='Dice outcomes for expert player (default 5)')
    loop_parser.add_argument('--differential-lr', action='store_true',
                             help='Use differential LR: body/value at lr*0.1, '
                                  'policy at lr*10 (protects warm-start)')
    loop_parser.add_argument('--label-smoothing', type=float, default=0.0,
                             help='Label smoothing for policy targets (default 0.0)')
    loop_parser.add_argument('--scheduler', choices=['cosine', 'constant'],
                             default='cosine', help='LR scheduler (default: cosine)')

    args = parser.parse_args()

    if hasattr(args, 'workers') and args.workers > 1:
        # spawn is required for CUDA but very slow to start (~2-3 min for 15 workers).
        # fork is instant and safe on CPU-only Linux.
        if torch.cuda.is_available():
            mp.set_start_method("spawn", force=True)
        else:
            mp.set_start_method("fork", force=True)

    if args.command == 'generate':
        records, stats = generate_games(
            args.checkpoint, args.games,
            num_simulations=args.sims, c_puct=args.c_puct,
            output_dir=args.output_dir,
            workers=args.workers,
            blend_bootstrap_path=args.blend_bootstrap,
        )
        save_records(records, args.output_dir)
        print(f"Stats: {stats}")

    elif args.command == 'expert':
        records, stats = generate_expert_games(
            args.bc_model, args.games,
            search_depth=args.search_depth,
            dice_sample_size=args.dice_sample,
            output_dir=args.output_dir,
            workers=args.workers,
            distill_values=args.distill_values,
            with_ranking=args.with_ranking,
            with_graph_features=args.with_graph_features,
        )
        save_records(records, args.output_dir)
        print(f"Stats: {stats}")

    elif args.command == 'train':
        if not args.gnn and args.checkpoint is None:
            parser.error("--checkpoint is required for MLP training (use --gnn for GNN)")
        body_dims = None
        if args.body_dims:
            body_dims = tuple(int(x) for x in args.body_dims.split(','))
        gnn_dims = None
        if hasattr(args, 'gnn_dims') and args.gnn_dims:
            gnn_dims = tuple(int(x) for x in args.gnn_dims.split(','))
        train_on_data(
            args.checkpoint, args.data_dir, args.output,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, value_weight=args.value_weight,
            human_data_dir=args.human_data,
            human_mix_ratio=args.human_mix_ratio,
            freeze_value=args.freeze_value,
            differential_lr=args.differential_lr,
            label_smoothing=args.label_smoothing,
            scheduler=args.scheduler,
            body_dims=body_dims, dropout=args.dropout,
            resume_training=args.resume_training,
            loss_asymmetry=args.loss_asymmetry,
            negative_oversample=args.negative_oversample,
            ranking_data_dirs=args.ranking_data,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
            gnn=args.gnn, gnn_dims=gnn_dims,
            edge_dropout=args.edge_dropout,
            max_samples=args.max_samples,
        )

    elif args.command == 'evaluate':
        evaluate_checkpoints(
            args.new_checkpoint, args.old_checkpoint,
            num_games=args.games, num_simulations=args.sims,
            workers=args.workers,
        )

    elif args.command == 'loop':
        if args.expert and not args.bc_model:
            parser.error("--expert requires --bc-model")
        run_loop(
            args.start_checkpoint, args.iterations,
            args.games_per_iter, args.sims,
            args.output_dir,
            epochs=args.epochs, lr=args.lr,
            eval_games=args.eval_games,
            workers=args.workers,
            blend_bootstrap_path=args.blend_bootstrap,
            human_data_dir=args.human_data,
            human_mix_ratio=args.human_mix_ratio,
            freeze_value=args.freeze_value,
            accept_all=args.accept_all,
            expert=args.expert,
            bc_model=args.bc_model,
            search_depth=args.search_depth,
            dice_sample_size=args.dice_sample,
            differential_lr=args.differential_lr,
            label_smoothing=args.label_smoothing,
            scheduler=args.scheduler,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
