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
"""

import argparse
import json
import multiprocessing as mp
import os
import time

# Must be set BEFORE importing torch/numpy to prevent internal thread creation,
# which deadlocks with fork()-based multiprocessing.
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
                         vps_to_win=10):
    """Play one self-play game with proper color tracking.

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
# Multiprocessing worker for parallel self-play
# ---------------------------------------------------------------------------

_az_worker_net = None
_az_worker_fi = None
_az_worker_means = None
_az_worker_stds = None
_az_worker_kwargs = None


def _az_worker_init(checkpoint_path, num_simulations, c_puct,
                    temperature_threshold, vps_to_win):
    """Initialize per-worker state: load model, build FeatureIndexer."""
    global _az_worker_net, _az_worker_fi, _az_worker_means, _az_worker_stds, _az_worker_kwargs

    net, ckpt = load_checkpoint(checkpoint_path)
    feat_names = ckpt['feature_names']
    _az_worker_means = ckpt['feature_means']
    _az_worker_stds = ckpt['feature_stds']
    feature_index_map = {name: i for i, name in enumerate(feat_names)}

    from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
    catan_map = build_map(BASE_MAP_TEMPLATE)
    _az_worker_fi = FeatureIndexer(feature_index_map, catan_map)
    _az_worker_net = net
    _az_worker_kwargs = {
        'num_simulations': num_simulations,
        'c_puct': c_puct,
        'temperature_threshold': temperature_threshold,
        'vps_to_win': vps_to_win,
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
                   output_dir=None, workers=1):
    """Generate multiple self-play games with incremental saving.

    If output_dir is provided, saves after every game so progress survives
    interruption. On resume, skips already-completed games.

    workers: number of parallel processes (1 = sequential).

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

        for i in tqdm(range(start_game, num_games), desc="Self-play",
                      initial=start_game, total=num_games):
            records, winner, turns = generate_one_game_v2(
                net, fi, feat_means, feat_stds,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temperature_threshold=temperature_threshold,
                vps_to_win=vps_to_win,
            )
            all_records.extend(records)
            total_turns += turns
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
                      temperature_threshold, vps_to_win),
        ) as pool:
            game_results = list(tqdm(
                pool.imap_unordered(_az_worker_play_one, range(remaining)),
                total=remaining, desc="Self-play", initial=start_game,
            ))

        for features_list, policies_list, values_list, winner_str, turns in game_results:
            for feat, pol, val in zip(features_list, policies_list, values_list):
                all_records.append(SelfPlayRecord(feat, pol, val))
            total_turns += turns
            if winner_str in ('None', 'null'):
                winners[None] = winners.get(None, 0) + 1
            else:
                name = winner_str.replace('Color.', '') if winner_str.startswith('Color.') else winner_str
                winners[Color[name]] = winners.get(Color[name], 0) + 1

        # Save all at once after parallel generation
        if output_dir is not None:
            save_records(all_records, output_dir, quiet=True)
            with open(os.path.join(output_dir, '_progress.json'), 'w') as f:
                json.dump({
                    'games_completed': num_games,
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class SelfPlayDataset(Dataset):
    """PyTorch dataset for self-play data."""

    def __init__(self, features, policies, values, means, stds):
        # Normalize features
        self.features = torch.tensor((features - means) / stds, dtype=torch.float32)
        self.policies = torch.tensor(policies, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.features[idx], self.policies[idx], self.values[idx]


def train_on_data(checkpoint_path, data_dirs, output_path, epochs=20,
                  batch_size=256, lr=1e-3, value_weight=1.0):
    """Train AlphaZero network on self-play data.

    Args:
        checkpoint_path: Path to current checkpoint (for architecture + normalization)
        data_dirs: List of data directories to load
        output_path: Where to save the trained checkpoint
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        value_weight: Weight of value loss vs policy loss

    Returns:
        training stats dict
    """
    net, ckpt = load_checkpoint(checkpoint_path)
    net.train()

    feat_means = ckpt['feature_means']
    feat_stds = ckpt['feature_stds']
    iteration = ckpt.get('iteration', 0) + 1

    # Load all data
    all_features = []
    all_policies = []
    all_values = []
    for d in data_dirs:
        f, p, v = load_records(d)
        all_features.append(f)
        all_policies.append(p)
        all_values.append(v)

    features = np.concatenate(all_features)
    policies = np.concatenate(all_policies)
    values = np.concatenate(all_values)

    print(f"Training on {len(values)} samples from {len(data_dirs)} data dir(s)")
    print(f"  Value distribution: mean={values.mean():.3f}, std={values.std():.3f}")

    dataset = SelfPlayDataset(features, policies, values, feat_means, feat_stds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_v_loss = 0.0
        total_p_loss = 0.0
        n_batches = 0

        for feat_batch, policy_batch, value_batch in loader:
            # Build action mask from policy targets (nonzero = legal)
            action_mask = policy_batch > 0

            # Forward
            value_pred, logits = net(feat_batch, action_mask)
            value_pred = value_pred.squeeze(-1)

            # Value loss: MSE
            v_loss = F.mse_loss(value_pred, value_batch)

            # Policy loss: cross-entropy with visit distribution
            # Only compute over legal actions (nonzero targets) to avoid 0 * -inf = nan
            log_probs = F.log_softmax(logits, dim=-1)
            # Clamp log_probs to avoid -inf * 0 producing nan
            log_probs = log_probs.clamp(min=-100.0)
            p_loss = -(policy_batch * log_probs).sum(dim=-1).mean()

            loss = value_weight * v_loss + p_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_v = total_v_loss / n_batches
        avg_p = total_p_loss / n_batches
        history.append({'epoch': epoch, 'loss': avg_loss, 'v_loss': avg_v, 'p_loss': avg_p})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
                  f"(v={avg_v:.4f}, p={avg_p:.4f})")

    net.eval()
    save_checkpoint(
        net, optimizer, iteration,
        ckpt['feature_names'], feat_means, feat_stds,
        output_path,
        extra={'training_history': history, 'num_samples': len(values)},
    )
    print(f"Saved trained checkpoint to {output_path} (iteration {iteration})")

    return {'history': history, 'iteration': iteration, 'num_samples': len(values)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoints(new_path, old_path, num_games=200,
                         num_simulations=400, c_puct=1.4):
    """Evaluate new checkpoint vs old checkpoint.

    Both play as MCTS with their respective networks. New plays as both
    colors (half the games each). Returns win rate of new.
    """
    new_net, new_ckpt = load_checkpoint(new_path)
    old_net, old_ckpt = load_checkpoint(old_path)

    # Use new checkpoint's normalization (should be same)
    feat_names = new_ckpt['feature_names']
    feat_means = new_ckpt['feature_means']
    feat_stds = new_ckpt['feature_stds']
    feature_index_map = {name: i for i, name in enumerate(feat_names)}

    from catanatron.models.map import build_map, BASE_MAP_TEMPLATE
    catan_map = build_map(BASE_MAP_TEMPLATE)
    fi = FeatureIndexer(feature_index_map, catan_map)

    new_wins = 0
    for i in tqdm(range(num_games), desc="Evaluating"):
        fi_copy = FeatureIndexer(feature_index_map, catan_map)

        if i % 2 == 0:
            # New plays RED
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
            # New plays BLUE
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
             output_dir, c_puct=1.4, epochs=20, batch_size=256, lr=1e-3,
             eval_games=100, accept_threshold=0.55, window_size=5, workers=1):
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
    """
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
            print(f"\n[1/3] Generating {games_per_iter} self-play games...")
            records, stats = generate_games(
                current_best, games_per_iter,
                num_simulations=num_simulations, c_puct=c_puct,
                output_dir=iter_data_dir, workers=workers,
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
            print(f"\n[3/3] Evaluating new vs current best ({eval_games} games)...")
            win_rate, ci = evaluate_checkpoints(
                new_ckpt_path, current_best,
                num_games=eval_games, num_simulations=num_simulations,
                c_puct=c_puct,
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

    # Train
    train_parser = subparsers.add_parser('train', help='Train on self-play data')
    train_parser.add_argument('--checkpoint', required=True)
    train_parser.add_argument('--data-dir', nargs='+', required=True)
    train_parser.add_argument('--output', required=True)
    train_parser.add_argument('--epochs', type=int, default=20)
    train_parser.add_argument('--batch-size', type=int, default=256)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--value-weight', type=float, default=1.0)

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate new vs old')
    eval_parser.add_argument('--new-checkpoint', required=True)
    eval_parser.add_argument('--old-checkpoint', required=True)
    eval_parser.add_argument('--games', type=int, default=200)
    eval_parser.add_argument('--sims', type=int, default=400)

    # Full loop
    loop_parser = subparsers.add_parser('loop', help='Full training loop')
    loop_parser.add_argument('--start-checkpoint', required=True)
    loop_parser.add_argument('--iterations', type=int, default=10)
    loop_parser.add_argument('--games-per-iter', type=int, default=500)
    loop_parser.add_argument('--sims', type=int, default=400)
    loop_parser.add_argument('--output-dir', required=True)
    loop_parser.add_argument('--epochs', type=int, default=20)
    loop_parser.add_argument('--eval-games', type=int, default=100)
    loop_parser.add_argument('--workers', type=int, default=1,
                             help='Parallel worker processes (default: 1)')

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
        )
        save_records(records, args.output_dir)
        print(f"Stats: {stats}")

    elif args.command == 'train':
        train_on_data(
            args.checkpoint, args.data_dir, args.output,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, value_weight=args.value_weight,
        )

    elif args.command == 'evaluate':
        evaluate_checkpoints(
            args.new_checkpoint, args.old_checkpoint,
            num_games=args.games, num_simulations=args.sims,
        )

    elif args.command == 'loop':
        run_loop(
            args.start_checkpoint, args.iterations,
            args.games_per_iter, args.sims,
            args.output_dir,
            epochs=args.epochs, eval_games=args.eval_games,
            workers=args.workers,
        )

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
