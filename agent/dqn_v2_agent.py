"""
DQN v2 Agent — CNN encoder + Dueling head + Double DQN target.

Differences from `agent.dq_agent.DQNAgent`:

1. Network: `model.cnn_q_network.CNNDuelingQNetwork` (operates on (6,25,25)
   channel-encoded states instead of flat 625-vectors).
2. Update rule: Double DQN —
       a* = argmax_a Q_local(s', a)
       y  = r + gamma * Q_target(s', a*) * (1 - done)
   instead of vanilla `y = r + gamma * max_a Q_target(s', a)`.
3. States are stored RAW (flat uint8) in the replay buffer to keep memory
   small; channel-encoding happens at sample time on the GPU/CPU.

The old `DQNAgent` is preserved for A/B comparison.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ACTION_SIZE, BATCH_SIZE, BUFFER_SIZE, EPSILON_DECAY, EPSILON_MIN,
    EPSILON_START, GAMMA, GRID_SIZE, LEARNING_RATE, TAU, UPDATE_EVERY,
)
from model.cnn_q_network import CNNDuelingQNetwork
from utils.state_encoder import NUM_CHANNELS, encode_batch, encode_state


# ----------------------------- replay buffer -----------------------------

class FlatReplayBuffer:
    """Stores raw uint8 flat states; encodes on sample. ~6x memory saving."""

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        # Cast to uint8 to save memory (values are 0..6).
        self.buffer.append((
            np.asarray(state, dtype=np.uint8),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.uint8),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.int32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.int32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ----------------------------- agent -----------------------------

class DQNv2Agent:
    def __init__(
        self,
        action_size: int = ACTION_SIZE,
        grid_size: int = GRID_SIZE,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        tau: float = TAU,
        batch_size: int = BATCH_SIZE,
        buffer_size: int = BUFFER_SIZE,
        update_every: int = UPDATE_EVERY,
        device: str | None = None,
    ) -> None:
        self.action_size = action_size
        self.grid_size = grid_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.qnetwork_local = CNNDuelingQNetwork(
            in_channels=NUM_CHANNELS, grid_size=grid_size, n_actions=action_size
        ).to(self.device)
        self.qnetwork_target = CNNDuelingQNetwork(
            in_channels=NUM_CHANNELS, grid_size=grid_size, n_actions=action_size
        ).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = FlatReplayBuffer(buffer_size)
        self.t_step = 0

        # exploration
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        # diagnostics
        self.learning_step = 0
        self.last_loss: float | None = None
        self.last_mean_q: float | None = None

    # ---------------- act / step ----------------

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)) \
                     .unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            qs = self.qnetwork_local(x)
            self.qnetwork_local.train()
        return int(qs.argmax(dim=1).item())

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions at `state` (numpy, shape (action_size,))."""
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)) \
                     .unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            qs = self.qnetwork_local(x)
            self.qnetwork_local.train()
        return qs.cpu().numpy().flatten()

    def step(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self.learn()

    # ---------------- learn (Double DQN) ----------------

    def learn(self) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Encode states and next_states to (B, 6, 25, 25) tensors on device
        s = torch.from_numpy(encode_batch(states, self.grid_size)).to(self.device)
        s_next = torch.from_numpy(encode_batch(next_states, self.grid_size)).to(self.device)
        a = torch.from_numpy(actions).to(self.device).unsqueeze(1)
        r = torch.from_numpy(rewards).to(self.device).unsqueeze(1)
        d = torch.from_numpy(dones).to(self.device).unsqueeze(1)

        # ---- Double DQN target ----
        # 1) pick best NEXT actions using LOCAL net
        with torch.no_grad():
            next_actions = self.qnetwork_local(s_next).argmax(dim=1, keepdim=True)
            # 2) evaluate those actions using TARGET net
            next_q = self.qnetwork_target(s_next).gather(1, next_actions)
            y = r + self.gamma * next_q * (1.0 - d)

        # Current Q estimates for taken actions
        q = self.qnetwork_local(s).gather(1, a)
        loss = F.smooth_l1_loss(q, y)  # Huber — more robust than MSE for DQN

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        self.optimizer.step()

        # ---- soft update target network ----
        with torch.no_grad():
            for tp, lp in zip(self.qnetwork_target.parameters(),
                              self.qnetwork_local.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * lp)

        self.learning_step += 1
        self.last_loss = float(loss.item())
        self.last_mean_q = float(q.mean().item())
        return {"loss": self.last_loss, "mean_q": self.last_mean_q}

    # ---------------- epsilon ----------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, value: float) -> None:
        self.epsilon = float(value)

    # ---------------- persistence ----------------

    def save(self, path: str) -> None:
        torch.save({
            "qnetwork_local": self.qnetwork_local.state_dict(),
            "qnetwork_target": self.qnetwork_target.state_dict(),
            "epsilon": self.epsilon,
            "learning_step": self.learning_step,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.qnetwork_local.load_state_dict(ckpt["qnetwork_local"])
        self.qnetwork_target.load_state_dict(ckpt["qnetwork_target"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self.learning_step = int(ckpt.get("learning_step", 0))
