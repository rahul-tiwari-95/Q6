"""
CNN Q-Network with a Dueling head.

Architecture
------------
Input:  (B, 6, 25, 25) float tensor (channel-encoded state from utils.state_encoder)

    Conv2d(6 -> 32,  k=3, p=1) -> ReLU
    Conv2d(32 -> 64, k=3, p=1) -> ReLU
    Flatten
    Linear(64*25*25 -> 256) -> ReLU
    Split into two heads:
      Value     V(s):  Linear(256 -> 128) -> ReLU -> Linear(128 -> 1)
      Advantage A(s,a): Linear(256 -> 128) -> ReLU -> Linear(128 -> n_actions)
    Combine: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

The dueling decomposition separates "how good is this state" from
"how much better is this action than average" — useful when many states
have similar value regardless of action.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDuelingQNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        grid_size: int = 25,
        n_actions: int = 4,
        conv_channels: Tuple[int, int] = (32, 64),
        fc_hidden: int = 256,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.n_actions = n_actions

        c1, c2 = conv_channels
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)

        flat_size = c2 * grid_size * grid_size
        self.fc_shared = nn.Linear(flat_size, fc_hidden)

        self.value_hidden = nn.Linear(fc_hidden, head_hidden)
        self.value_out = nn.Linear(head_hidden, 1)

        self.adv_hidden = nn.Linear(fc_hidden, head_hidden)
        self.adv_out = nn.Linear(head_hidden, n_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        h = F.relu(self.fc_shared(x))

        v = F.relu(self.value_hidden(h))
        v = self.value_out(v)  # (B, 1)

        a = F.relu(self.adv_hidden(h))
        a = self.adv_out(a)    # (B, n_actions)

        # Dueling combination with mean-baseline (more stable than max-baseline).
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str | None = None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))
