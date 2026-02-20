"""Gym wrapper that filters CatanatronEnv to strategic features + normalizes.

Wraps the raw 664-dim observation to a normalized 98-dim strategic feature
vector, and exposes action_masks() for MaskablePPO compatibility.
"""

import random as _random

import gymnasium as gym
import numpy as np
import torch

from catanatron.features import get_feature_ordering
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer


POSITIONAL_PREFIXES = ("NODE", "EDGE", "TILE", "PORT")


def _get_strategic_names(num_players=2):
    """Return the strategic (non-positional) feature names for N-player games."""
    all_feats = get_feature_ordering(num_players=num_players)
    return [f for f in all_feats if not any(f.startswith(p) for p in POSITIONAL_PREFIXES)]


def _make_enemy(opponent_kind, bc_path=None, shared_value_fn=None):
    """Create an opponent player (always RED) from a string kind.

    Supports: random, weighted, alphabeta, value, rl:<model_path>,
              search, search:<depth>
    """
    if opponent_kind == "random":
        return RandomPlayer(Color.RED)
    elif opponent_kind == "weighted":
        return WeightedRandomPlayer(Color.RED)
    elif opponent_kind == "alphabeta":
        return AlphaBetaPlayer(Color.RED)
    elif opponent_kind == "value":
        from robottler.value_model import load_value_model
        if bc_path is None:
            raise ValueError("'value' opponent requires bc_path")
        player = ValueFunctionPlayer(Color.RED)
        player._neural_value_fn = load_value_model(bc_path)
        return player
    elif opponent_kind.startswith("rl:"):
        from robottler.rl_player import RLPlayer
        model_path = opponent_kind[3:]
        if bc_path is None:
            raise ValueError("'rl' opponent requires bc_path")
        return RLPlayer(Color.RED, model_path, bc_path, deterministic=False)
    elif opponent_kind == "search" or opponent_kind.startswith("search:"):
        from robottler.search_player import NeuralSearchPlayer, make_blended_value_fn
        if bc_path is None:
            raise ValueError("'search' opponent requires bc_path")
        depth = 2
        if ":" in opponent_kind:
            depth = int(opponent_kind.split(":")[1])
        if shared_value_fn is None:
            shared_value_fn = make_blended_value_fn(bc_path, blend_weight=1e8)
        return NeuralSearchPlayer(
            Color.RED, depth=depth, prunning=False, value_fn=shared_value_fn,
        )
    elif opponent_kind == "mcts" or opponent_kind.startswith("mcts:"):
        from robottler.mcts_player import MCTSPlayer, make_mcts_value_fn
        if bc_path is None:
            raise ValueError("'mcts' opponent requires bc_path")
        sims = 200  # Lower sims for training speed
        if ":" in opponent_kind:
            sims = int(opponent_kind.split(":")[1])
        mcts_value_fn = make_mcts_value_fn(bc_path)
        return MCTSPlayer(
            Color.RED, value_fn=mcts_value_fn, num_simulations=sims,
        )
    else:
        raise ValueError(f"Unknown opponent kind: {opponent_kind}")


class StrategicCatanEnv(gym.Wrapper):
    """Filters CatanatronEnv obs to normalized strategic features.

    - Raw obs: 664-dim (2-player) → 98 strategic features
    - Normalizes using BC v2 means/stds for those 98 features
    - Overrides observation_space to Box(shape=(N_strategic,))
    - Provides action_masks() for MaskablePPO
    """

    def __init__(self, env, bc_checkpoint_path):
        super().__init__(env)

        # Load BC checkpoint for normalization stats and feature names
        ckpt = torch.load(bc_checkpoint_path, map_location="cpu", weights_only=False)
        bc_feature_names = ckpt["feature_names"]  # 176 names
        bc_means = ckpt["feature_means"]  # shape (176,)
        bc_stds = ckpt["feature_stds"]  # shape (176,)

        # Get the 2-player feature ordering (664 features) and strategic subset (98)
        all_2p_features = get_feature_ordering(num_players=2)
        strategic_names = _get_strategic_names(num_players=2)

        # feature_indices: which of the 664 gym features correspond to our 98 strategic set
        self.feature_indices = np.array(
            [all_2p_features.index(f) for f in strategic_names], dtype=np.intp
        )

        # bc_col_indices: which of BC's 176 features correspond to those 98
        bc_col_indices = np.array(
            [bc_feature_names.index(f) for f in strategic_names], dtype=np.intp
        )

        # Extract means/stds for our 98 features from the BC checkpoint
        self.means = bc_means[bc_col_indices].astype(np.float32)
        self.stds = bc_stds[bc_col_indices].astype(np.float32)

        self.n_strategic = len(strategic_names)

        # Override observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_strategic,), dtype=np.float32,
        )

    def observation(self, obs):
        """Filter 664-dim raw obs → 98 strategic features → normalize."""
        filtered = obs[self.feature_indices].astype(np.float32)
        return (filtered - self.means) / self.stds

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def action_masks(self):
        """Return boolean mask over Discrete(290) action space."""
        valid = self.env.get_valid_actions()  # list of ints (indices)
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[valid] = True
        return mask


def make_env(opponent="random", vps_to_win=10, bc_path="robottler/models/value_net_v2.pt"):
    """Factory that returns a StrategicCatanEnv (not vectorized).

    Each call creates a fresh env with its own opponent instance,
    safe for SubprocVecEnv parallelism.
    """
    def _init():
        enemy = _make_enemy(opponent, bc_path=bc_path)
        base_env = CatanatronEnv(config={
            "enemies": [enemy],
            "vps_to_win": vps_to_win,
            "representation": "vector",
            "reward_function": _win_loss_reward,
        })
        return StrategicCatanEnv(base_env, bc_path)
    return _init()


def _preload_search_value_fn(opponent, bc_path):
    """Pre-load the blend value function once if opponent is search-based."""
    if opponent == "search" or opponent.startswith("search:"):
        from robottler.search_player import make_blended_value_fn
        return make_blended_value_fn(bc_path, blend_weight=1e8)
    return None


def make_vec_env(opponent="random", vps_to_win=10, bc_path="robottler/models/value_net_v2.pt",
                 n_envs=8, use_subproc=False):
    """Create a vectorized env with n_envs parallel environments.

    Uses SubprocVecEnv by default for isolation (each env gets its own
    opponent in a separate process, no shared state). Falls back to
    DummyVecEnv if use_subproc=False — safe here because each _init()
    creates a fresh enemy instance.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    # Pre-load shared value fn for search opponents (avoids N torch.load calls)
    shared_vfn = _preload_search_value_fn(opponent, bc_path)

    def _make_init(seed_offset):
        def _init():
            enemy = _make_enemy(opponent, bc_path=bc_path,
                                shared_value_fn=shared_vfn)
            base_env = CatanatronEnv(config={
                "enemies": [enemy],
                "vps_to_win": vps_to_win,
                "representation": "vector",
                "reward_function": _win_loss_reward,
            })
            env = StrategicCatanEnv(base_env, bc_path)
            return env
        return _init

    env_fns = [_make_init(i) for i in range(n_envs)]
    if use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


class MixedOpponentCatanEnv(StrategicCatanEnv):
    """StrategicCatanEnv that randomly swaps opponents each episode.

    Pre-creates a pool of opponent Player instances and swaps the enemy
    on the underlying CatanatronEnv before each reset().
    """

    def __init__(self, env, bc_checkpoint_path, enemy_pool):
        super().__init__(env, bc_checkpoint_path)
        self._enemy_pool = enemy_pool

    def reset(self, **kwargs):
        enemy = _random.choice(self._enemy_pool)
        base = self.env  # CatanatronEnv
        base.enemies = [enemy]
        base.players = [base.p0] + base.enemies
        return super().reset(**kwargs)


def make_mixed_vec_env(opponent_specs, vps_to_win=10,
                       bc_path="robottler/models/value_net_v2.pt",
                       n_envs=8):
    """Create a vectorized env with mixed opponents.

    Args:
        opponent_specs: list of opponent kinds, e.g.
            ["weighted", "value", "rl:path/to/checkpoint.zip", "search"]
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Pre-load shared value fn if any opponent is search-based
    shared_vfns = {}
    for spec in opponent_specs:
        if (spec == "search" or spec.startswith("search:")) and spec not in shared_vfns:
            shared_vfns[spec] = _preload_search_value_fn(spec, bc_path)

    def _make_init(seed_offset):
        def _init():
            # Each env gets its own enemy pool (separate Player instances)
            pool = [_make_enemy(spec, bc_path=bc_path,
                                shared_value_fn=shared_vfns.get(spec))
                    for spec in opponent_specs]
            base_env = CatanatronEnv(config={
                "enemies": [pool[0]],
                "vps_to_win": vps_to_win,
                "representation": "vector",
                "reward_function": _win_loss_reward,
            })
            return MixedOpponentCatanEnv(base_env, bc_path, pool)
        return _init

    env_fns = [_make_init(i) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


def _win_loss_reward(game, p0_color):
    """Pure win/loss reward: +1 win, -1 loss, 0 ongoing."""
    winning_color = game.winning_color()
    if winning_color == p0_color:
        return 1
    elif winning_color is None:
        return 0
    else:
        return -1
