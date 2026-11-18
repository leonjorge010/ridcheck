"""Tiny MLP the ESP32-S3 can run after int8 export."""

from __future__ import annotations

import torch
from torch import nn

from ridcheck.features import N_FEATURES, WINDOW


class SpoofNet(nn.Module):
    """Window of receiver features in, one logit out.

    Input shape is (batch, 8, n_features), oldest fix first. The flattened
    window goes through two ReLU layers. The integer weights stay a few
    kilobytes, which is what the ESP32-S3 multiplies.
    """

    def __init__(self, hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        width = WINDOW * N_FEATURES
        h1, h2 = hidden
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(width, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(features.shape[0], -1)
        return self.net(flat).squeeze(-1)
