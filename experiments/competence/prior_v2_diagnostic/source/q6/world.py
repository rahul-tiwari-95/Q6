"""Deterministic collection kernel with independent state layers.

Actions are labels 0..3. ``action_mapping`` maps each label to a physical
direction (up, down, left, right). The mapping is part of the observation.
Episodes are finite-horizon tasks; both success and timeout end their return.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class WorldConfig:
    size: int = 5
    wall_count: int = 4
    pellet_count: int = 1
    horizon: int = 32
    action_mapping: tuple = (0, 1, 2, 3)
    step_cost: float = -0.01
    pellet_reward: float = 1.0
    shaping_weight: float = 0.2
    gamma: float = 0.97

    def __post_init__(self):
        if self.size < 3 or self.horizon < 1:
            raise ValueError("size must be >= 3 and horizon >= 1")
        if self.wall_count < 0 or self.pellet_count < 1:
            raise ValueError("wall_count must be >= 0 and pellet_count >= 1")
        if self.wall_count + self.pellet_count + 1 > self.size**2:
            raise ValueError("not enough free cells for agent and pellets")
        if sorted(self.action_mapping) != [0, 1, 2, 3]:
            raise ValueError("action_mapping must be a permutation of 0..3")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")


def reachable(walls: np.ndarray, start: tuple) -> set:
    """Flood fill using four-neighbor motion."""
    seen, queue = {start}, deque([start])
    size = len(walls)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DELTAS:
            nxt = r + dr, c + dc
            if (0 <= nxt[0] < size and 0 <= nxt[1] < size
                    and not walls[nxt] and nxt not in seen):
                seen.add(nxt)
                queue.append(nxt)
    return seen


class CollectionWorld:
    def __init__(self, config: WorldConfig | None = None, seed: int = 0):
        self.config = config or WorldConfig()
        self.rng = np.random.default_rng(seed)
        self.walls = np.zeros((self.config.size, self.config.size), dtype=bool)
        self.pellets = self.walls.copy()
        self.position = (0, 0)
        self.elapsed = 0
        self.terminated = self.truncated = False
        self._ready = False

    @property
    def observation_size(self) -> int:
        return 3 * self.config.size**2 + 17

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.config.size
        # Retry complete proposals, accepting only connected free space. A
        # bounded fallback prevents high-density configurations from hanging.
        for _ in range(1000):
            self.walls = np.zeros((n, n), dtype=bool)
            chosen = self.rng.choice(n * n, self.config.wall_count, replace=False)
            self.walls.flat[chosen] = True
            free = np.argwhere(~self.walls)
            start = tuple(int(v) for v in free[0])
            if len(reachable(self.walls, start)) == len(free):
                break
        else:
            raise ValueError("unable to generate a connected map; reduce wall_count")
        picks = self.rng.choice(len(free), self.config.pellet_count + 1, replace=False)
        self.position = tuple(int(v) for v in free[picks[0]])
        self.pellets = np.zeros_like(self.walls)
        for pick in picks[1:]:
            self.pellets[tuple(free[pick])] = True
        self.elapsed = 0
        self.terminated = self.truncated = False
        self._ready = True
        return self.observe(), self.info()

    def observe(self) -> np.ndarray:
        """All reward/transition-relevant state, including mapping and clock."""
        if not self._ready:
            raise RuntimeError("reset before observing")
        occupancy = np.zeros_like(self.walls, dtype=np.float32)
        occupancy[self.position] = 1.0
        layers = np.stack((self.walls, self.pellets, occupancy)).astype(np.float32)
        mapping = np.eye(4, dtype=np.float32)[list(self.config.action_mapping)]
        remaining = max(0.0, 1.0 - self.elapsed / self.config.horizon)
        return np.concatenate((layers.ravel(), [remaining], mapping.ravel())).astype(np.float32)

    def _potential(self) -> float:
        if not self.pellets.any():
            return 0.0
        seen, queue = {self.position}, deque([(self.position, 0)])
        n = self.config.size
        while queue:
            (r, c), distance = queue.popleft()
            if self.pellets[r, c]:
                return -distance / float(n)
            for dr, dc in DELTAS:
                nxt = r + dr, c + dc
                if (0 <= nxt[0] < n and 0 <= nxt[1] < n
                        and not self.walls[nxt] and nxt not in seen):
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        raise RuntimeError("unreachable pellet violates world invariant")

    def step(self, action: int):
        if not self._ready:
            raise RuntimeError("reset before stepping")
        if self.terminated or self.truncated:
            raise RuntimeError("episode ended; reset before stepping")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError("action must be an integer in 0..3")
        before = self._potential()
        dr, dc = DELTAS[self.config.action_mapping[int(action)]]
        candidate = self.position[0] + dr, self.position[1] + dc
        n = self.config.size
        if 0 <= candidate[0] < n and 0 <= candidate[1] < n and not self.walls[candidate]:
            self.position = candidate
        collected = bool(self.pellets[self.position])
        if collected:
            self.pellets[self.position] = False
        self.elapsed += 1
        self.terminated = not bool(self.pellets.any())
        self.truncated = not self.terminated and self.elapsed >= self.config.horizon
        base = self.config.step_cost + self.config.pellet_reward * collected
        after = 0.0 if self.terminated or self.truncated else self._potential()
        reward = base + self.config.shaping_weight * (self.config.gamma * after - before)
        info = self.info()
        info.update(base_reward=float(base), collected=collected)
        return self.observe(), float(reward), self.terminated, self.truncated, info

    def info(self) -> dict:
        return {"position": list(self.position), "elapsed": self.elapsed,
                "pellets_remaining": int(self.pellets.sum()),
                "success": self.terminated, "terminated": self.terminated,
                "truncated": self.truncated}

    def state_dict(self) -> dict:
        """JSON-compatible state including owned RNG for exact continuation."""
        return {"config": asdict(self.config), "walls": self.walls.astype(int).tolist(),
                "pellets": self.pellets.astype(int).tolist(), "position": list(self.position),
                "elapsed": self.elapsed, "terminated": self.terminated,
                "truncated": self.truncated, "ready": self._ready,
                "rng": copy.deepcopy(self.rng.bit_generator.state)}

    @classmethod
    def from_state(cls, state: dict) -> "CollectionWorld":
        config = dict(state["config"])
        config["action_mapping"] = tuple(config["action_mapping"])
        world = cls(WorldConfig(**config))
        world.walls = np.asarray(state["walls"], dtype=bool).copy()
        world.pellets = np.asarray(state["pellets"], dtype=bool).copy()
        world.position = tuple(state["position"])
        world.elapsed = int(state["elapsed"])
        world.terminated, world.truncated = bool(state["terminated"]), bool(state["truncated"])
        world._ready = bool(state["ready"])
        world.rng.bit_generator.state = copy.deepcopy(state["rng"])
        return world

    def clone(self) -> "CollectionWorld":
        return self.from_state(self.state_dict())
