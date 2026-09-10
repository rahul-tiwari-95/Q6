"""Compact Double DQN baseline with agent-owned exploration and replay RNG."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DQN:
    def __init__(self, observation_size: int, seed: int = 0, gamma: float = 0.97,
                 capacity: int = 12000, batch_size: int = 64):
        if observation_size <= 0:
            raise ValueError("observation_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if capacity < max(256, batch_size):
            raise ValueError("capacity must be >= max(256, batch_size) to reach the learning warmup")
        self.observation_size, self.gamma = observation_size, gamma
        self.capacity, self.batch_size = capacity, batch_size
        self.rng = np.random.default_rng(seed)
        # Initialization should not advance another agent's global Torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online = nn.Sequential(nn.Linear(observation_size, 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=0.001)
        self.states = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.ends = np.empty(capacity, dtype=np.float32)
        self.count = self.cursor = self.steps = self.updates = 0
        self.last_loss = None

    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        # Evaluation with epsilon=0 does not consume any RNG.
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(4))
        with torch.no_grad():
            return int(self.online(torch.from_numpy(observation).unsqueeze(0)).argmax(1).item())

    def observe(self, state, action, reward, next_state, ended):
        index = self.cursor
        self.states[index], self.next_states[index] = state, next_state
        self.actions[index], self.rewards[index], self.ends[index] = action, reward, ended
        self.cursor = (index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        self.steps += 1
        if self.count >= max(256, self.batch_size) and self.steps % 4 == 0:
            self.learn()

    def learn(self):
        indices = self.rng.choice(self.count, self.batch_size, replace=False)
        states = torch.from_numpy(self.states[indices])
        next_states = torch.from_numpy(self.next_states[indices])
        actions = torch.from_numpy(self.actions[indices]).unsqueeze(1)
        rewards = torch.from_numpy(self.rewards[indices])
        ended = torch.from_numpy(self.ends[indices])
        with torch.no_grad():
            next_action = self.online(next_states).argmax(1, keepdim=True)
            target = rewards + self.gamma * (1 - ended) * self.target(next_states).gather(1, next_action).squeeze(1)
        predicted = self.online(states).gather(1, actions).squeeze(1)
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        with torch.no_grad():
            for target_param, param in zip(self.target.parameters(), self.online.parameters()):
                target_param.lerp_(param, 0.01)
        self.last_loss = float(loss.item())
        self.updates += 1

    def state_dict(self) -> dict:
        # All continuation state, not just network weights. At capacity every
        # slot is live; before capacity only the prefix has been initialized.
        return {"observation_size": self.observation_size, "gamma": self.gamma,
                "capacity": self.capacity, "batch_size": self.batch_size,
                "online": copy.deepcopy(self.online.state_dict()),
                "target": copy.deepcopy(self.target.state_dict()),
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "states": self.states[:self.count].copy(),
                "next_states": self.next_states[:self.count].copy(),
                "actions": self.actions[:self.count].copy(),
                "rewards": self.rewards[:self.count].copy(),
                "ends": self.ends[:self.count].copy(), "count": self.count,
                "cursor": self.cursor, "steps": self.steps, "updates": self.updates,
                "last_loss": self.last_loss}

    @classmethod
    def from_state(cls, state: dict) -> "DQN":
        agent = cls(state["observation_size"], gamma=state["gamma"],
                    capacity=state["capacity"], batch_size=state["batch_size"])
        agent.online.load_state_dict(state["online"])
        agent.target.load_state_dict(state["target"])
        agent.optimizer.load_state_dict(state["optimizer"])
        agent.rng.bit_generator.state = copy.deepcopy(state["rng"])
        for key in ("count", "cursor", "steps", "updates", "last_loss"):
            setattr(agent, key, state[key])
        for key in ("states", "next_states", "actions", "rewards", "ends"):
            getattr(agent, key)[:agent.count] = state[key]
        return agent

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for tensor in self.online.state_dict().values():
            digest.update(tensor.detach().numpy().tobytes())
        return digest.hexdigest()
