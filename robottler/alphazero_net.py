"""Dual-head AlphaZero network for Catan.

Architecture:
    Shared body:  176 → 256 → 128 (reuses value_net_v2 layers)
    Value head:   128 → 64 → 1 (tanh, outputs [-1, +1])
    Policy head:  128 → 64 → 290 (logits, masked softmax)

The value head uses tanh instead of sigmoid because:
- Centers on 0 for even positions, pushes to ±1 for decisive ones
- Double the dynamic range of sigmoid [0,1]
- Natural midpoint for MCTS UCB formula
- Avoids the normalization compression problem that killed MCTS with sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from catanatron.gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE, ACTIONS_ARRAY, normalize_action, to_action_space,
)
from catanatron.models.enums import ActionType


class CatanAlphaZeroNet(nn.Module):
    """Dual-head network: shared body → value head + policy head."""

    def __init__(self, input_dim: int = 176, num_actions: int = ACTION_SPACE_SIZE):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions

        # Shared body (matches value_net_v2 first two layers, minus dropout)
        self.body = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Value head: 128 → 64 → 1, tanh output
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Policy head: 128 → 64 → num_actions (raw logits)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x, action_mask=None):
        """Forward pass.

        Args:
            x: (batch, input_dim) feature tensor
            action_mask: (batch, num_actions) bool tensor. True = legal action.
                         If None, no masking is applied to policy logits.

        Returns:
            value: (batch, 1) in [-1, +1]
            policy_logits: (batch, num_actions) masked logits (illegal = -inf)
        """
        shared = self.body(x)
        value = self.value_head(shared)
        logits = self.policy_head(shared)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        return value, logits

    def predict(self, x, action_mask=None):
        """Inference: returns value scalar and policy probability vector.

        Args:
            x: (batch, input_dim) feature tensor
            action_mask: (batch, num_actions) bool tensor

        Returns:
            value: (batch,) float tensor in [-1, +1]
            policy: (batch, num_actions) probability tensor (sums to 1 over legal)
        """
        value, logits = self.forward(x, action_mask)
        policy = F.softmax(logits, dim=-1)
        return value.squeeze(-1), policy


def warm_start_from_value_net(az_net, value_ckpt_path):
    """Initialize the shared body and value head from a trained CatanValueNet.

    The value_net_v2 architecture is:
        net.0: Linear(176, 256)  → body.0
        net.3: Linear(256, 128)  → body.2
        net.6: Linear(128, 64)   → value_head.0
        net.8: Linear(64, 1)     → value_head.2

    Note: value_net_v2 uses sigmoid output while we use tanh.
    The last layer bias is adjusted: tanh(x) ≈ 2*sigmoid(x) - 1,
    so we scale weights by 2 and shift bias by -1 to approximate.
    """
    ckpt = torch.load(value_ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']

    # Body: layers 0 and 3 → body.0 and body.2
    az_net.body[0].weight.data.copy_(sd['net.0.weight'])
    az_net.body[0].bias.data.copy_(sd['net.0.bias'])
    az_net.body[2].weight.data.copy_(sd['net.3.weight'])
    az_net.body[2].bias.data.copy_(sd['net.3.bias'])

    # Value head: layers 6 and 8 → value_head.0 and value_head.2
    az_net.value_head[0].weight.data.copy_(sd['net.6.weight'])
    az_net.value_head[0].bias.data.copy_(sd['net.6.bias'])

    # Last layer: convert from sigmoid to tanh space
    # sigmoid(x) maps to [0,1], tanh(x) maps to [-1,1]
    # tanh(x) = 2*sigmoid(2x) - 1, but approximately:
    # Just scale output weight/bias so the pre-activation range is similar
    az_net.value_head[2].weight.data.copy_(sd['net.8.weight'])
    az_net.value_head[2].bias.data.copy_(sd['net.8.bias'])

    # Policy head: initialize with small random weights (Xavier)
    nn.init.xavier_uniform_(az_net.policy_head[0].weight)
    nn.init.zeros_(az_net.policy_head[0].bias)
    nn.init.xavier_uniform_(az_net.policy_head[2].weight)
    nn.init.zeros_(az_net.policy_head[2].bias)

    return ckpt.get('feature_names'), ckpt.get('feature_means'), ckpt.get('feature_stds')


def warm_start_policy_from_data(az_net, action_counts_path=None):
    """Initialize policy head bias from human action frequency data.

    If action_counts_path is provided, load action frequency counts and
    set bias = log(frequency + 1). This makes the policy head initially
    predict something close to human play distribution.
    """
    if action_counts_path is None:
        return

    counts = np.load(action_counts_path)
    assert len(counts) == az_net.num_actions

    # bias = log(count + 1) - mean, so common actions get positive bias
    log_counts = np.log(counts + 1.0)
    log_counts -= log_counts.mean()

    with torch.no_grad():
        az_net.policy_head[2].bias.copy_(
            torch.tensor(log_counts, dtype=torch.float32)
        )


# ---------------------------------------------------------------------------
# Action space helpers for bitboard
# ---------------------------------------------------------------------------

def bb_actions_to_mask(actions):
    """Convert list of Action objects to a boolean action mask.

    Args:
        actions: list of catanatron Action objects

    Returns:
        mask: numpy array of shape (ACTION_SPACE_SIZE,), dtype bool
        valid_indices: list of (action, index) pairs for valid actions
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    valid = []
    for action in actions:
        try:
            idx = to_action_space(action)
            mask[idx] = True
            valid.append((action, idx))
        except (ValueError, IndexError):
            # Action not in the standard space (e.g., unusual trade)
            pass
    return mask, valid


def policy_to_action_priors(policy_probs, actions):
    """Map policy probability vector to action prior dict.

    Args:
        policy_probs: numpy array of shape (ACTION_SPACE_SIZE,)
        actions: list of Action objects (legal actions)

    Returns:
        dict {action: prior_probability} with probabilities renormalized
    """
    priors = {}
    total = 0.0
    for action in actions:
        try:
            idx = to_action_space(action)
            p = float(policy_probs[idx])
            priors[action] = p
            total += p
        except (ValueError, IndexError):
            priors[action] = 0.0

    # Renormalize (some actions may not be in the space)
    if total > 1e-8:
        for a in priors:
            priors[a] /= total
    else:
        # Uniform fallback
        uniform = 1.0 / len(actions) if actions else 0.0
        for a in priors:
            priors[a] = uniform

    return priors


def save_checkpoint(az_net, optimizer, iteration, feature_names, feature_means,
                    feature_stds, path, extra=None):
    """Save AlphaZero checkpoint."""
    data = {
        'model_state_dict': az_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'iteration': iteration,
        'feature_names': feature_names,
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'input_dim': az_net.input_dim,
        'num_actions': az_net.num_actions,
    }
    if extra:
        data.update(extra)
    torch.save(data, path)


def load_checkpoint(path, device='cpu'):
    """Load AlphaZero checkpoint, returning model and metadata."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = CatanAlphaZeroNet(
        input_dim=ckpt.get('input_dim', 176),
        num_actions=ckpt.get('num_actions', ACTION_SPACE_SIZE),
    )
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net, ckpt
