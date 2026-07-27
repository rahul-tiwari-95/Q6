"""
CNN Actor-Critic network for Independent PPO self-play (v8).

Architecture
------------
Input:  (B, 6, 25, 25) float tensor (same channel encoder as the DQN agents)

    Conv2d(6 -> 32,  k=3, p=1) -> ReLU
    Conv2d(32 -> 64, k=3, p=1) -> ReLU
    Flatten
    Linear(64*25*25 -> 256) -> ReLU
    Split into two heads:
      Policy pi(a|s): Linear(256 -> n_actions)   (logits; Categorical externally)
      Value   V(s):   Linear(256 -> 1)

Deliberately matches `model.cnn_q_network.CNNDuelingQNetwork`'s conv trunk
(same conv_channels, same fc_hidden) so v8's DQN-vs-PPO comparison isolates
the learning algorithm, not network capacity. See Q6.md section 3.3.

Orthogonal init with a small-std policy head and near-zero-std value head
(CleanRL convention) keeps the initial policy close to uniform and the
initial value estimate small, which measurably stabilizes early PPO
training compared to the Kaiming init used for the DQN networks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(layer: nn.Module, std: float = 2 ** 0.5, bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        grid_size: int = 25,
        n_actions: int = 4,
        conv_channels: Tuple[int, int] = (32, 64),
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.n_actions = n_actions

        c1, c2 = conv_channels
        self.trunk = nn.Sequential(
            _layer_init(nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(c1, c2, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(c2 * grid_size * grid_size, fc_hidden)),
            nn.ReLU(),
        )
        self.actor = _layer_init(nn.Linear(fc_hidden, n_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(fc_hidden, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits (B, n_actions), value (B,))."""
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        h = self.trunk(x.float())
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (action, log_prob, entropy, value).

        If `action` is given, evaluates its log_prob/entropy under the current
        policy instead of sampling a new one — used during the PPO update pass
        to re-evaluate actions taken during rollout collection under the
        (now-updated) policy.
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
