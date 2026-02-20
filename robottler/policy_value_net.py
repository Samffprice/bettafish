"""Custom sb3 feature extractor + BC weight loading for MaskablePPO.

Architecture (maps to BC v2 layers, no dropout):
    Feature Extractor (shared trunk):
        Input(98) → Linear(98,256) → ReLU → Linear(256,128) → ReLU → out(128)
    sb3 Policy head (random init):
        Linear(128,64) → ReLU → Linear(64, 290)
    sb3 Value head (from BC):
        Linear(128,64) → ReLU → Linear(64,1)
"""

import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from catanatron.features import get_feature_ordering


POSITIONAL_PREFIXES = ("NODE", "EDGE", "TILE", "PORT")


class CatanFeatureExtractor(BaseFeaturesExtractor):
    """Shared trunk: two linear layers initialized from BC v2 weights.

    Input: 98-dim normalized strategic features
    Output: 128-dim representation fed to sb3 policy/value heads
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        in_dim = observation_space.shape[0]
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.trunk(observations)


def _get_bc_column_indices(bc_feature_names):
    """Find which columns of BC's 176 features correspond to the 98 2p strategic features."""
    all_2p = get_feature_ordering(num_players=2)
    strategic_2p = [f for f in all_2p if not any(f.startswith(p) for p in POSITIONAL_PREFIXES)]
    col_indices = [bc_feature_names.index(f) for f in strategic_2p]
    return np.array(col_indices, dtype=np.intp)


def load_bc_weights(model, bc_path):
    """Transfer BC v2 weights into an sb3 MaskablePPO model.

    Weight mapping:
        BC net.0 (Linear 176→256) → feature_extractor.trunk.0 (Linear 98→256, column subset)
        BC net.3 (Linear 256→128) → feature_extractor.trunk.2 (Linear 256→128, direct copy)
        BC net.6 (Linear 128→64)  → value_net MLP layer 0 (via mlp_extractor.value_net.0)
        BC net.8 (Linear 64→1)    → value_net output (via value_net, the final Linear(64,1))
        Policy head: stays random-initialized
    """
    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    bc_state = ckpt["model_state_dict"]
    bc_feature_names = ckpt["feature_names"]
    col_indices = _get_bc_column_indices(bc_feature_names)

    # Access sb3 model internals
    sd = model.policy.state_dict()

    # BC net.0: [256, 176] → column subset → [256, 98]
    bc_w0 = bc_state["net.0.weight"][:, col_indices]
    bc_b0 = bc_state["net.0.bias"]

    # BC net.3: [128, 256] → direct copy
    bc_w3 = bc_state["net.3.weight"]
    bc_b3 = bc_state["net.3.bias"]

    # --- Feature extractor (sb3 may create up to 3 copies) ---
    for prefix in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        w0_key = f"{prefix}.trunk.0.weight"
        if w0_key in sd:
            sd[w0_key] = bc_w0
            sd[f"{prefix}.trunk.0.bias"] = bc_b0
            sd[f"{prefix}.trunk.2.weight"] = bc_w3
            sd[f"{prefix}.trunk.2.bias"] = bc_b3

    # --- Value head ---
    # BC net.6 (Linear 128→64) → sb3 mlp_extractor.value_net.0
    sd["mlp_extractor.value_net.0.weight"] = bc_state["net.6.weight"]  # [64, 128]
    sd["mlp_extractor.value_net.0.bias"] = bc_state["net.6.bias"]  # [64]

    # BC net.8 (Linear 64→1) → sb3 value_net (final projection)
    sd["value_net.weight"] = bc_state["net.8.weight"]  # [1, 64]
    sd["value_net.bias"] = bc_state["net.8.bias"]  # [1]

    # Policy head (mlp_extractor.policy_net.*, action_net.*) stays random
    model.policy.load_state_dict(sd)
    return model


def load_bc_policy_weights(model, value_path, policy_path):
    """Transfer BOTH BC value and BC policy weights into MaskablePPO.

    Weight mapping:
        Shared trunk (98→256→128): from BC policy net (drives the representation)
        Value head (128→64→1): from BC value net (net.6, net.8)
        Policy head (128→64→290): from BC policy net (net.4, net.6)

    BC policy architecture (CatanPolicyNet):
        net.0: Linear(176,256)   → trunk layer 0
        net.3: Linear(256,128)   → trunk layer 1
        net.6: Linear(128,64)    → policy MLP layer 0
        net.9: Linear(64,290)    → action output
    """
    val_ckpt = torch.load(value_path, map_location="cpu", weights_only=False)
    pol_ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
    val_state = val_ckpt["model_state_dict"]
    pol_state = pol_ckpt["model_state_dict"]
    val_names = val_ckpt["feature_names"]
    col_indices = _get_bc_column_indices(val_names)

    sd = model.policy.state_dict()

    # --- Shared trunk from POLICY net ---
    pol_w0 = pol_state["net.0.weight"][:, col_indices]  # [256, 176] → [256, 98]
    pol_b0 = pol_state["net.0.bias"]
    pol_w3 = pol_state["net.3.weight"]  # [128, 256]
    pol_b3 = pol_state["net.3.bias"]

    for prefix in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        w0_key = f"{prefix}.trunk.0.weight"
        if w0_key in sd:
            sd[w0_key] = pol_w0
            sd[f"{prefix}.trunk.0.bias"] = pol_b0
            sd[f"{prefix}.trunk.2.weight"] = pol_w3
            sd[f"{prefix}.trunk.2.bias"] = pol_b3

    # --- Value head from VALUE net ---
    sd["mlp_extractor.value_net.0.weight"] = val_state["net.6.weight"]
    sd["mlp_extractor.value_net.0.bias"] = val_state["net.6.bias"]
    sd["value_net.weight"] = val_state["net.8.weight"]
    sd["value_net.bias"] = val_state["net.8.bias"]

    # --- Policy head from POLICY net ---
    # BC policy: net.6 = Linear(128,64), net.9 = Linear(64,290)
    # sb3: mlp_extractor.policy_net.0 = Linear(128,64), action_net = Linear(64,290)
    sd["mlp_extractor.policy_net.0.weight"] = pol_state["net.6.weight"]
    sd["mlp_extractor.policy_net.0.bias"] = pol_state["net.6.bias"]
    sd["action_net.weight"] = pol_state["net.9.weight"]
    sd["action_net.bias"] = pol_state["net.9.bias"]

    model.policy.load_state_dict(sd)
    return model


def create_model(bc_path, env=None, **ppo_kwargs):
    """Create a MaskablePPO model with CatanFeatureExtractor and load BC weights.

    If env is None, creates a temporary env just for model construction.
    Returns the model with BC weights loaded.
    """
    from sb3_contrib import MaskablePPO
    from robottler.catan_env import make_env

    if env is None:
        env = make_env("random", vps_to_win=10, bc_path=bc_path)

    policy_kwargs = {
        "features_extractor_class": CatanFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": dict(pi=[64], vf=[64]),
    }

    defaults = dict(
        n_steps=256,
        batch_size=2048,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        verbose=1,
    )
    defaults.update(ppo_kwargs)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        **defaults,
    )

    load_bc_weights(model, bc_path)
    return model
