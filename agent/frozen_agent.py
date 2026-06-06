"""
Frozen agent wrapper — loads a saved DQNv2 state_dict and exposes a greedy
`act(state)` interface. Used as a non-learning opponent in self-play.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from config import ACTION_SIZE, GRID_SIZE
from model.cnn_q_network import CNNDuelingQNetwork
from utils.state_encoder import NUM_CHANNELS, encode_state


class FrozenAgent:
    """Read-only DQNv2 wrapper. No learning, no exploration."""

    def __init__(self, network: CNNDuelingQNetwork, device: str, grid_size: int = GRID_SIZE,
                 action_size: int = ACTION_SIZE):
        self.network = network
        self.device = device
        self.grid_size = grid_size
        self.action_size = action_size
        self.network.eval()

    @classmethod
    def load(cls, path: str, device: str | None = None,
             grid_size: int = GRID_SIZE, action_size: int = ACTION_SIZE) -> "FrozenAgent":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        net = CNNDuelingQNetwork(
            in_channels=NUM_CHANNELS, grid_size=grid_size, n_actions=action_size
        ).to(device)
        ckpt = torch.load(path, map_location=device)
        net.load_state_dict(ckpt["qnetwork_local"])
        return cls(net, device, grid_size, action_size)

    def act(self, state: np.ndarray, training: bool = False) -> int:
        # `training` accepted for API parity with DQNv2Agent; ignored here.
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            qs = self.network(x)
        return int(qs.argmax(dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            qs = self.network(x)
        return qs.cpu().numpy().flatten()
