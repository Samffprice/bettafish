"""Catanatron Player wrapping a trained MaskablePPO policy.

Used for:
- Self-play opponents (curriculum Phase C/D)
- Benchmark evaluation via robottler.benchmark --rl-model
"""

import numpy as np
import torch

from sb3_contrib import MaskablePPO

from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.models.player import Player
from catanatron.gym.envs.catanatron_env import to_action_space, from_action_space, ACTION_SPACE_SIZE


POSITIONAL_PREFIXES = ("NODE", "EDGE", "TILE", "PORT")


class RLPlayer(Player):
    """Catanatron Player that uses a trained MaskablePPO model to decide actions.

    Extracts strategic features from the game state, normalizes with BC stats,
    and queries the RL policy with action masking.
    """

    def __init__(self, color, model_path, bc_path, is_bot=True, deterministic=True):
        super().__init__(color, is_bot)
        self.deterministic = deterministic

        # Load RL model
        self.model = MaskablePPO.load(model_path)

        # Load BC checkpoint for normalization stats
        ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
        bc_feature_names = ckpt["feature_names"]  # 176 names
        bc_means = ckpt["feature_means"]
        bc_stds = ckpt["feature_stds"]

        # Feature mappings
        self._all_2p_features = get_feature_ordering(num_players=2)
        self._strategic_names = [
            f for f in self._all_2p_features
            if not any(f.startswith(p) for p in POSITIONAL_PREFIXES)
        ]

        # Which of BC's 176 columns correspond to our 98 strategic features
        bc_col_indices = np.array(
            [bc_feature_names.index(f) for f in self._strategic_names], dtype=np.intp
        )
        self.means = bc_means[bc_col_indices].astype(np.float32)
        self.stds = bc_stds[bc_col_indices].astype(np.float32)

    def _extract_obs(self, game):
        """Extract normalized 98-dim strategic observation from game state."""
        vec = create_sample_vector(game, self.color, self._strategic_names)
        x = np.array(vec, dtype=np.float32)
        return (x - self.means) / self.stds

    def _compute_mask(self, playable_actions):
        """Build boolean action mask from playable Catanatron actions."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        for action in playable_actions:
            try:
                idx = to_action_space(action)
                mask[idx] = True
            except (ValueError, IndexError):
                pass  # skip unmappable actions (shouldn't happen)
        return mask

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        obs = self._extract_obs(game)
        mask = self._compute_mask(playable_actions)

        action_int, _ = self.model.predict(
            obs, deterministic=self.deterministic, action_masks=mask
        )
        return from_action_space(int(action_int), playable_actions)
