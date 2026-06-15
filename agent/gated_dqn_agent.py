"""
Gated DQN Agent — Double DQN with GatedOptionNetwork and context replay buffer.

Differences from DQNv2Agent
----------------------------
1. Network: GatedOptionNetwork (dual evade/collect heads + learned gate MLP).
2. Replay buffer: ContextReplayBuffer stores per-transition context vectors
   (B, 3) = [hunter_dist_norm, pellets_remaining_norm, lives_norm].
3. Context is passed through forward during both action selection and learning.
4. `add_experience()` is public so the CHER module can inject counterfactual
   transitions directly into the buffer.
5. `q_values_options()` exposes evade/collect/gate breakdowns for the
   dashboard Brain tab.
6. `load_warm_start()` maps a DQNv2Agent checkpoint onto the gated network
   (shared trunk copied, both advantage heads initialized from old adv head).

Context vector (3,)
-------------------
Component 0: hunter_dist_norm      = hunter_manhattan_dist / (2 * grid_size)
Component 1: pellets_remaining_norm = pellets_remaining / TARGET_PELLETS
Component 2: lives_norm             = lives / KRISHNA_LIVES

Caller is responsible for building the context from the env info dict using
the helper `info_to_context(info, grid_size)` defined at module level.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ACTION_SIZE, BATCH_SIZE, BUFFER_SIZE, EPSILON_DECAY, EPSILON_MIN,
    EPSILON_START, GAMMA, GRID_SIZE, LEARNING_RATE, TAU, UPDATE_EVERY,
    TARGET_PELLETS, INITIAL_LIVES,
)
from model.gated_option_network import GatedOptionNetwork
from utils.state_encoder import NUM_CHANNELS, encode_batch, encode_state


# ---------------------------------------------------------------------------
# Helper — build gate context from env info dict
# ---------------------------------------------------------------------------

def info_to_context(
    info: Dict,
    grid_size: int = GRID_SIZE,
    target_pellets: int = TARGET_PELLETS,
    max_lives: int = INITIAL_LIVES,
) -> np.ndarray:
    """
    Convert an env step info dict into a (3,) float32 gate context vector.

    Components are normalized to [0, 1]:
      0: hunter distance    / (2 * grid_size)   — max Manhattan = 2*(G-1)
      1: pellets remaining  / target_pellets
      2: lives remaining    / max_lives
    """
    hunter_dist = float(info.get("hunter_manhattan_dist", 0))
    pellets     = float(info.get("pellets_remaining", 0))
    lives       = float(info.get("lives", max_lives))
    return np.array(
        [
            hunter_dist / (2.0 * grid_size),
            pellets / float(target_pellets),
            lives / float(max_lives),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Context-aware Replay Buffer
# ---------------------------------------------------------------------------

class ContextReplayBuffer:
    """
    Replay buffer that stores context vectors alongside state transitions.

    Memory layout per transition:
        state        : np.uint8  (grid_size^2,)
        context      : np.float32 (3,)
        action       : int
        reward       : float
        next_state   : np.uint8  (grid_size^2,)
        next_context : np.float32 (3,)
        done         : bool
    """

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        context: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_context: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((
            np.asarray(state,        dtype=np.uint8),
            np.asarray(context,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state,   dtype=np.uint8),
            np.asarray(next_context, dtype=np.float32),
            bool(done),
        ))

    def add_batch(self, experiences: List[Tuple]) -> None:
        """Bulk-add a list of (state, context, action, reward, next_state, next_context, done)."""
        for exp in experiences:
            self.add(*exp)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, contexts, actions, rewards, next_states, next_contexts, dones = zip(*batch)
        return (
            np.stack(states).astype(np.int32),
            np.stack(contexts).astype(np.float32),
            np.array(actions,      dtype=np.int64),
            np.array(rewards,      dtype=np.float32),
            np.stack(next_states).astype(np.int32),
            np.stack(next_contexts).astype(np.float32),
            np.array(dones,        dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Gated DQN Agent
# ---------------------------------------------------------------------------

class GatedDQNAgent:
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
        gate_reg_weight: float = 0.01,
    ) -> None:
        self.action_size = action_size
        self.grid_size = grid_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.gate_reg_weight = float(gate_reg_weight)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.qnetwork_local = GatedOptionNetwork(
            in_channels=NUM_CHANNELS, grid_size=grid_size, n_actions=action_size
        ).to(self.device)
        self.qnetwork_target = GatedOptionNetwork(
            in_channels=NUM_CHANNELS, grid_size=grid_size, n_actions=action_size
        ).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

        self.optimizer = torch.optim.Adam(
            self.qnetwork_local.parameters(), lr=learning_rate
        )
        self.memory = ContextReplayBuffer(buffer_size)
        self.t_step = 0

        # Exploration
        self.epsilon: float = EPSILON_START
        self.epsilon_min: float = EPSILON_MIN
        self.epsilon_decay: float = EPSILON_DECAY

        # Diagnostics
        self.learning_step: int = 0
        self.last_loss: Optional[float] = None
        self.last_mean_q: Optional[float] = None
        self.last_mean_gate: Optional[float] = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        state: np.ndarray,
        context: np.ndarray,
        training: bool = True,
    ) -> int:
        """ε-greedy action selection with gate context."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)) \
                     .unsqueeze(0).to(self.device)
            c = torch.from_numpy(context).unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            qs = self.qnetwork_local(x, c)
            self.qnetwork_local.train()
        return int(qs.argmax(dim=1).item())

    def q_values(self, state: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Return blended Q-values for all actions, shape (action_size,)."""
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)) \
                     .unsqueeze(0).to(self.device)
            c = torch.from_numpy(context).unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            qs = self.qnetwork_local(x, c)
            self.qnetwork_local.train()
        return qs.cpu().numpy().flatten()

    def q_values_options(
        self, state: np.ndarray, context: np.ndarray
    ) -> Dict[str, np.ndarray | float]:
        """
        Return per-option Q-values and gate scalar — used by the dashboard.

        Returns dict with keys: q_evade, q_collect, q_blend, gate.
        """
        with torch.no_grad():
            x = torch.from_numpy(encode_state(state, self.grid_size)) \
                     .unsqueeze(0).to(self.device)
            c = torch.from_numpy(context).unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            q_e, q_c, q_b, gate = self.qnetwork_local.forward_options(x, c)
            self.qnetwork_local.train()
        return {
            "q_evade":   q_e.cpu().numpy().flatten(),
            "q_collect": q_c.cpu().numpy().flatten(),
            "q_blend":   q_b.cpu().numpy().flatten(),
            "gate":      float(gate.item()),
        }

    # ------------------------------------------------------------------
    # Experience storage (public so CHER can inject counterfactuals)
    # ------------------------------------------------------------------

    def add_experience(
        self,
        state: np.ndarray,
        context: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_context: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the replay buffer (no learning triggered)."""
        self.memory.add(state, context, action, reward, next_state, next_context, done)

    def step(
        self,
        state: np.ndarray,
        context: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_context: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience and trigger a learning step every `update_every` calls."""
        self.memory.add(state, context, action, reward, next_state, next_context, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self.learn()

    # ------------------------------------------------------------------
    # Learning (Double DQN with context)
    # ------------------------------------------------------------------

    def learn(self) -> Dict[str, float]:
        (states, contexts, actions, rewards,
         next_states, next_contexts, dones) = self.memory.sample(self.batch_size)

        # Encode to (B, 6, 25, 25) tensors
        s      = torch.from_numpy(encode_batch(states,      self.grid_size)).to(self.device)
        s_next = torch.from_numpy(encode_batch(next_states, self.grid_size)).to(self.device)
        c      = torch.from_numpy(contexts).to(self.device)
        c_next = torch.from_numpy(next_contexts).to(self.device)
        a      = torch.from_numpy(actions).to(self.device).unsqueeze(1)
        r      = torch.from_numpy(rewards).to(self.device).unsqueeze(1)
        d      = torch.from_numpy(dones).to(self.device).unsqueeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.qnetwork_local(s_next, c_next).argmax(dim=1, keepdim=True)
            next_q = self.qnetwork_target(s_next, c_next).gather(1, next_actions)
            y = r + self.gamma * next_q * (1.0 - d)

        # Current Q for taken actions — use forward_options to also get gate for regularization
        _, _, q_blend, gate_batch = self.qnetwork_local.forward_options(s, c)
        q = q_blend.gather(1, a)

        # Gate entropy regularization: maximise H(gate) = -g*log(g) - (1-g)*log(1-g)
        # Penalises gate collapsing to either extreme (especially 1.0 = pure evasion)
        gate_entropy = -(gate_batch * torch.log(gate_batch + 1e-8)
                         + (1.0 - gate_batch) * torch.log(1.0 - gate_batch + 1e-8))
        loss = F.smooth_l1_loss(q, y) - self.gate_reg_weight * gate_entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update
        with torch.no_grad():
            for tp, lp in zip(self.qnetwork_target.parameters(),
                              self.qnetwork_local.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * lp)

        self.learning_step += 1
        self.last_loss      = float(loss.item())
        self.last_mean_q    = float(q.mean().item())
        self.last_mean_gate = float(gate_batch.mean().detach().item())
        return {
            "loss":      self.last_loss,
            "mean_q":    self.last_mean_q,
            "mean_gate": self.last_mean_gate,
        }

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self, value: float) -> None:
        self.epsilon = float(value)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "qnetwork_local":  self.qnetwork_local.state_dict(),
            "qnetwork_target": self.qnetwork_target.state_dict(),
            "epsilon":         self.epsilon,
            "learning_step":   self.learning_step,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.qnetwork_local.load_state_dict(ckpt["qnetwork_local"])
        self.qnetwork_target.load_state_dict(ckpt["qnetwork_target"])
        self.epsilon       = float(ckpt.get("epsilon", self.epsilon))
        self.learning_step = int(ckpt.get("learning_step", 0))

    def load_warm_start(self, path: str) -> None:
        """
        Warm-start from a DQNv2Agent checkpoint (CNNDuelingQNetwork).
        Shared trunk + value head are copied; advantage head seeds both
        evade and collect heads; gate starts from random init.
        """
        self.qnetwork_local.load_from_cnn_dueling(path, map_location=self.device)
        self.qnetwork_target.load_from_cnn_dueling(path, map_location=self.device)
        self.qnetwork_target.eval()
