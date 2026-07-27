"""
Frozen PPO agent wrapper — loads a saved CNNActorCritic state_dict and exposes
a greedy `act(state)` interface, mirroring `agent.frozen_agent.FrozenAgent`'s
API so the FSP opponent-pool code in train_v8.py is structurally identical to
train_phase2.py. Used as a non-learning opponent in self-play.

Frozen opponents act greedily (argmax logits), not by sampling — this matches
the DQN side's convention (FrozenAgent is also greedy) and keeps "what does
this historical snapshot do" deterministic for a given state.
"""

from __future__ import annotations

import numpy as np
import torch

from config import ACTION_SIZE, GRID_SIZE
from model.actor_critic_network import CNNActorCritic
from utils.state_encoder import NUM_CHANNELS, encode_state


class FrozenPPOAgent:
    """Read-only PPO wrapper. No learning; always acts greedily (argmax logits)."""

    def __init__(
        self,
        network: CNNActorCritic,
        device: str,
        grid_size: int = GRID_SIZE,
        action_size: int = ACTION_SIZE,
    ) -> None:
        self.network = network
        self.device = device
        self.grid_size = grid_size
        self.action_size = action_size
        self.network.eval()

    @classmethod
    def load(
        cls,
        path: str,
        device: str | None = None,
        grid_size: int = GRID_SIZE,
        action_size: int = ACTION_SIZE,
    ) -> "FrozenPPOAgent":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        net = CNNActorCritic(
            in_channels=NUM_CHANNELS, grid_size=grid_size, n_actions=action_size
        ).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(ckpt)
        return cls(net, device, grid_size, action_size)

    def act(self, state: np.ndarray, training: bool = False) -> int:
        # `training` accepted for API parity with PPOAgent/FrozenAgent; ignored here.
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            logits, _value = self.network(x)
        return int(logits.argmax(dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Policy logits, not Q-values — see PPOAgent.q_values() docstring."""
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)).unsqueeze(0).to(self.device)
            logits, _value = self.network(x)
        return logits.cpu().numpy().flatten()
