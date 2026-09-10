"""Exact visible-state reference for the single-pellet finite-horizon task.

This is an engineered diagnostic, never a training signal in the pilot. It
uses an independent transition table and dynamic programming over position
and remaining time. All map, goal, agent, clock and action-rule inputs come
from the observation. Only public reward/horizon parameters come from config.
"""
from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np

from .world import WorldConfig


class VisibleOptimalQ:
    """Optimal shaped discounted action values under the observed rule.

    q_values(observation) returns four float64 values in action-label order.
    Terminal observations return zeros; act() rejects them. This reference
    explicitly supports one pellet, rather than silently approximating the
    combinatorial multi-pellet task. Maps are cached independently of action
    labels because a permutation does not change optimal physical behavior.
    """
    _directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, config: WorldConfig, max_cached_maps: int = 128):
        if config.pellet_count != 1:
            raise ValueError('optimal reference supports exactly one pellet')
        if max_cached_maps < 1:
            raise ValueError('max_cached_maps must be positive')
        self.config = config
        self.max_cached_maps = max_cached_maps
        self._cache = OrderedDict()

    def _decode(self, observation):
        n = self.config.size
        observation = np.asarray(observation)
        if observation.shape != (3 * n * n + 17,) or not np.isfinite(observation).all():
            raise ValueError('observation must contain finite visible layers, clock and mapping')
        layers = observation[:3 * n * n].reshape(3, n, n)
        if not np.isin(layers, (0, 1)).all():
            raise ValueError('visible layers must be binary')
        walls, pellets, occupancy = layers.astype(bool)
        if occupancy.sum() != 1 or pellets.sum() > 1:
            raise ValueError('expected one agent and at most one remaining pellet')
        if np.any(walls & (pellets | occupancy)):
            raise ValueError('agent/pellet cannot overlap a wall')
        position = int(np.flatnonzero(occupancy)[0])
        if pellets.flat[position]:
            raise ValueError('an uncollected pellet cannot overlap the agent')
        clock = float(observation[3 * n * n])
        remaining = int(round(clock * self.config.horizon))
        if not 0 <= remaining <= self.config.horizon or not np.isclose(
                clock, remaining / self.config.horizon, atol=1e-6, rtol=0):
            raise ValueError('clock does not encode a valid remaining step count')
        mapping = observation[-16:].reshape(4, 4)
        if (not np.isin(mapping, (0, 1)).all() or
                not np.all(mapping.sum(axis=0) == 1) or not np.all(mapping.sum(axis=1) == 1)):
            raise ValueError('observed action mapping must be a permutation matrix')
        labels = mapping.argmax(axis=1)
        goal = int(np.flatnonzero(pellets)[0]) if pellets.any() else None
        return walls, goal, position, remaining, labels

    def _table(self, walls, goal):
        key = (walls.tobytes(), goal)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        n, horizon = self.config.size, self.config.horizon
        # Independent shortest-path potential from the visible goal.
        distance = np.full(n * n, np.inf)
        distance[goal] = 0
        queue = deque([goal])
        while queue:
            cell = queue.popleft()
            row, column = divmod(cell, n)
            for dr, dc in self._directions:
                rr, cc = row + dr, column + dc
                if 0 <= rr < n and 0 <= cc < n and not walls[rr, cc]:
                    neighbor = rr * n + cc
                    if not np.isfinite(distance[neighbor]):
                        distance[neighbor] = distance[cell] + 1
                        queue.append(neighbor)
        free = np.flatnonzero(~walls.ravel())
        if not np.isfinite(distance[free]).all():
            raise ValueError('visible free space must be connected to the pellet')
        active = free[free != goal]
        successor = np.empty((len(active), 4), dtype=int)
        for index, cell in enumerate(active):
            row, column = divmod(int(cell), n)
            for action, (dr, dc) in enumerate(self._directions):
                rr, cc = row + dr, column + dc
                successor[index, action] = (rr * n + cc if
                    0 <= rr < n and 0 <= cc < n and not walls[rr, cc] else cell)
        collected = successor == goal
        immediate = self.config.step_cost + self.config.pellet_reward * collected
        values = np.zeros((horizon + 1, n * n))
        shaped_q = np.zeros((horizon + 1, n * n, 4))
        offset = self.config.shaping_weight * distance[active, None] / n
        for remaining in range(1, horizon + 1):
            base_q = immediate + self.config.gamma * values[remaining - 1, successor] * (~collected)
            values[remaining, active] = base_q.max(axis=1)
            # Terminal potential is zero at success AND timeout. Consequently
            # Q_shaped*(s,a) = Q_base*(s,a) - weight * Phi(s), for every action.
            shaped_q[remaining, active] = base_q + offset
        result = shaped_q, distance
        self._cache[key] = result
        if len(self._cache) > self.max_cached_maps:
            self._cache.popitem(last=False)
        return result

    def q_values(self, observation) -> np.ndarray:
        walls, goal, position, remaining, labels = self._decode(observation)
        if goal is None or remaining == 0:
            return np.zeros(4, dtype=np.float64)
        table, _ = self._table(walls, goal)
        return table[remaining, position, labels].copy()

    def act(self, observation) -> int:
        _, goal, _, remaining, _ = self._decode(observation)
        if goal is None or remaining == 0:
            raise ValueError('cannot act after success or timeout')
        return int(self.q_values(observation).argmax())

    def can_finish(self, observation) -> bool:
        """Whether collection is reachable in remaining ticks, irrespective of Q ties."""
        walls, goal, position, remaining, _ = self._decode(observation)
        if goal is None:
            return True
        if remaining == 0:
            return False
        _, distance = self._table(walls, goal)
        return bool(distance[position] <= remaining)

    def action_can_finish(self, observation) -> np.ndarray:
        """Whether each action leaves a possible collection by the deadline.

        This is a reachability diagnostic, independent of reward ranking.
        Terminal observations have no available decisions and return false.
        """
        walls, goal, position, remaining, labels = self._decode(observation)
        if goal is None or remaining == 0:
            return np.zeros(4, dtype=bool)
        _, distance = self._table(walls, goal)
        n = self.config.size
        row, column = divmod(position, n)
        physical = np.zeros(4, dtype=bool)
        for action, (dr, dc) in enumerate(self._directions):
            rr, cc = row + dr, column + dc
            successor = (rr * n + cc if
                         0 <= rr < n and 0 <= cc < n and not walls[rr, cc] else position)
            physical[action] = distance[successor] <= remaining - 1
        return physical[labels]
