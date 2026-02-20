"""Neural network value function for Catan win probability prediction."""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable

from catanatron.features import create_sample, get_feature_ordering


class CatanValueNet(nn.Module):
    """MLP: 1002 features -> win probability."""

    def __init__(self, input_dim: int = 1002):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_value_model(model_path: str) -> Callable:
    """Load trained model + normalization stats, return a value function.

    Returns:
        value_fn(game, p0_color) -> float  (same signature as base_fn())
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    feature_names = checkpoint["feature_names"]
    feature_means = checkpoint["feature_means"]  # numpy array
    feature_stds = checkpoint["feature_stds"]  # numpy array

    model = CatanValueNet(input_dim=len(feature_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    means_t = torch.tensor(feature_means, dtype=torch.float32)
    stds_t = torch.tensor(feature_stds, dtype=torch.float32)

    def value_fn(game, p0_color) -> float:
        sample = create_sample(game, p0_color)
        # Build fixed-length vector; use 0 for features missing in this game
        # (e.g., P2/P3 features in a 2-player game)
        vec = [float(sample.get(f, 0.0)) for f in feature_names]
        x = torch.tensor(vec, dtype=torch.float32)
        x = (x - means_t) / stds_t
        with torch.no_grad():
            logit = model(x.unsqueeze(0))
            prob = torch.sigmoid(logit).item()
        return prob

    return value_fn
