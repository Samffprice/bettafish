"""Dual-head AlphaZero network for Catan.

Architecture (configurable):
    Body:         input_dim → body_dims[0] → body_dims[1] → ... (shared or split)
    Value head:   body_out → body_out//2 → 1 (tanh, outputs [-1, +1])
    Policy head:  body_out → body_out//2 → 290 (logits, masked softmax)

Default (v1): body_dims=(256, 128), dropout=0, shared_body=True  ~96K params
Bigger (v2):  body_dims=(512, 256), dropout=0.1, shared_body=True ~325K params

When shared_body=False, separate value_body and policy_body prevent policy
training from corrupting the warm-started value function (at the cost of
losing shared representation benefits).
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
    """Dual-head network: shared or split body -> value/policy heads."""

    def __init__(self, input_dim: int = 176, num_actions: int = ACTION_SPACE_SIZE,
                 body_dims: tuple = (256, 128), dropout: float = 0.0,
                 shared_body: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.body_dims = tuple(body_dims)
        self.dropout = dropout
        self.shared_body = shared_body

        def _make_body():
            layers = []
            prev = input_dim
            for dim in body_dims:
                layers.append(nn.Linear(prev, dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = dim
            return nn.Sequential(*layers)

        if shared_body:
            self.body = _make_body()
        else:
            self.value_body = _make_body()
            self.policy_body = _make_body()

        body_out = body_dims[-1]
        head_hidden = max(64, body_out // 2)

        # Value head: body_out -> head_hidden -> 1, tanh output
        self.value_head = nn.Sequential(
            nn.Linear(body_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
            nn.Tanh(),
        )

        # Policy head: body_out -> head_hidden -> num_actions (raw logits)
        self.policy_head = nn.Sequential(
            nn.Linear(body_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_actions),
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
        if self.shared_body:
            h = self.body(x)
            value = self.value_head(h)
            logits = self.policy_head(h)
        else:
            value = self.value_head(self.value_body(x))
            logits = self.policy_head(self.policy_body(x))

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

    def body_parameters(self):
        """Return body parameters (shared or value_body+policy_body)."""
        if self.shared_body:
            return list(self.body.parameters())
        else:
            return list(self.value_body.parameters()) + list(self.policy_body.parameters())

    def value_path_parameters(self):
        """Return all value-path parameters (body/value_body + value_head)."""
        if self.shared_body:
            return list(self.body.parameters()) + list(self.value_head.parameters())
        else:
            return list(self.value_body.parameters()) + list(self.value_head.parameters())

    def policy_path_parameters(self):
        """Return all policy-path parameters (body/policy_body + policy_head)."""
        if self.shared_body:
            return list(self.body.parameters()) + list(self.policy_head.parameters())
        else:
            return list(self.policy_body.parameters()) + list(self.policy_head.parameters())


def warm_start_from_value_net(az_net, value_ckpt_path):
    """Initialize body and value head from a trained CatanValueNet.

    Only works with body_dims=(256, 128) and dropout=0.0 (matching value_net_v2).

    The value_net_v2 architecture is:
        net.0: Linear(176, 256)  -> body[0] / value_body[0] / policy_body[0]
        net.3: Linear(256, 128)  -> body[2] / value_body[2] / policy_body[2]
        net.6: Linear(128, 64)   -> value_head[0]
        net.8: Linear(64, 1)     -> value_head[2]

    Note: value_net_v2 uses sigmoid output while we use tanh.
    The last layer is adjusted: W_new = W_old/2, b_new = b_old/2.
    """
    assert az_net.body_dims == (256, 128) and az_net.dropout == 0.0, \
        f"warm_start requires body_dims=(256, 128) dropout=0, got {az_net.body_dims} dropout={az_net.dropout}"

    ckpt = torch.load(value_ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']

    if az_net.shared_body:
        # Shared body
        az_net.body[0].weight.data.copy_(sd['net.0.weight'])
        az_net.body[0].bias.data.copy_(sd['net.0.bias'])
        az_net.body[2].weight.data.copy_(sd['net.3.weight'])
        az_net.body[2].bias.data.copy_(sd['net.3.bias'])
    else:
        # Value body
        az_net.value_body[0].weight.data.copy_(sd['net.0.weight'])
        az_net.value_body[0].bias.data.copy_(sd['net.0.bias'])
        az_net.value_body[2].weight.data.copy_(sd['net.3.weight'])
        az_net.value_body[2].bias.data.copy_(sd['net.3.bias'])

        # Policy body: same starting weights
        az_net.policy_body[0].weight.data.copy_(sd['net.0.weight'])
        az_net.policy_body[0].bias.data.copy_(sd['net.0.bias'])
        az_net.policy_body[2].weight.data.copy_(sd['net.3.weight'])
        az_net.policy_body[2].bias.data.copy_(sd['net.3.bias'])

    # Value head
    az_net.value_head[0].weight.data.copy_(sd['net.6.weight'])
    az_net.value_head[0].bias.data.copy_(sd['net.6.bias'])

    # Last layer: convert from sigmoid to tanh space
    az_net.value_head[2].weight.data.copy_(sd['net.8.weight'] * 0.5)
    az_net.value_head[2].bias.data.copy_(sd['net.8.bias'] * 0.5)

    # Policy head: Xavier init
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
        'body_dims': list(az_net.body_dims),
        'dropout': az_net.dropout,
        'shared_body': az_net.shared_body,
    }
    if extra:
        data.update(extra)
    torch.save(data, path)


def load_checkpoint(path, device='cpu'):
    """Load AlphaZero checkpoint, returning model and metadata.

    Handles all checkpoint formats:
    - New format: has body_dims, dropout, shared_body stored explicitly
    - Old shared body: has 'body.*' keys in state dict
    - Old split body: has 'value_body.*' + 'policy_body.*' keys
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']

    body_dims = tuple(ckpt.get('body_dims', (256, 128)))
    dropout_val = ckpt.get('dropout', 0.0)

    # Determine shared vs split body
    if 'shared_body' in ckpt:
        shared = ckpt['shared_body']
    else:
        # Old checkpoint: detect from state dict keys
        has_body = any(k.startswith('body.') for k in sd)
        has_split = any(k.startswith('value_body.') for k in sd)
        shared = has_body and not has_split

    net = CatanAlphaZeroNet(
        input_dim=ckpt.get('input_dim', 176),
        num_actions=ckpt.get('num_actions', ACTION_SPACE_SIZE),
        body_dims=body_dims,
        dropout=dropout_val,
        shared_body=shared,
    )

    net.load_state_dict(sd)
    net.eval()
    return net, ckpt
