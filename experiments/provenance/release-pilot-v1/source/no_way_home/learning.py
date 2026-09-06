"""A citizen that LEARNS its mitigation policy via tabular Q-learning,
instead of following a fixed script. This is Phase 4 of the roadmap
("learning adapter") arriving after Phase 2's substrate-without-learning was
actually built and tested -- not skipped to get here faster.

Tests action learning over supplied message-count features. Deduplication
is provided by feature_unique_rate; it is not learned from raw messages.
Two learner configs are
compared in run_learning_citizen.py: one handicapped to raw_message_rate
only (structurally can never do better than the naive heuristic's
information), one given both raw_message_rate and unique_origin_rate (has
the information to behave like the aware heuristic, but nothing hand-codes
that it should weight unique_rate over raw_rate -- it has to learn that).

Tested standalone against the existing fixed heuristics from policies.py,
not yet wired into the full ballot/election institution from
institutions.py -- that composition is a natural next step, kept separate
here so this result isn't confounded with the election layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from no_way_home.world import WorldState

FeatureFn = Callable[[WorldState], float]


def _bin_index(value: float, edges: list) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges) - 1


def feature_raw_rate(state: WorldState) -> float:
    return state.messages.raw_message_rate(current_tick=state.tick + 1, window=30)


def feature_unique_rate(state: WorldState) -> float:
    return state.messages.unique_origin_rate(current_tick=state.tick + 1, window=30)


RATE_BINS = [0.0, 0.1, 0.3, 0.6, 1.0, 2.0, float("inf")]


@dataclass
class TabularQMandateLearner:
    """Epsilon-greedy tabular Q-learning. Action: mitigate this tick, or
    not (binary). Reward: -shortfall_this_tick -- the same primary metric
    every fixed policy in this project is judged on, so comparisons are
    apples to apples. Callable as policy_fn(state, rng) -> bool, the same
    interface every other policy uses -- a drop-in replacement, and a
    stateful object: pass the SAME instance across many run() calls during
    training so its Q-table persists and improves episode over episode,
    then call .freeze() and evaluate on held-out seeds."""
    feature_fns: list
    bin_edges: list
    gamma: float = 0.9
    epsilon: float = 0.2
    alpha_floor: float = 0.05
    q_table: dict = field(default_factory=dict)  # {(state_bin, action): value}
    visit_counts: dict = field(default_factory=dict)  # {(state_bin, action): count}
    training: bool = True
    _last_bin: tuple = field(default=None, init=False)
    _last_action: bool = field(default=None, init=False)

    def _state_bin(self, state: WorldState) -> tuple:
        return tuple(_bin_index(f(state), edges) for f, edges in zip(self.feature_fns, self.bin_edges))

    def _q(self, state_bin: tuple, action: bool) -> float:
        return self.q_table.get((state_bin, action), 0.0)

    def __call__(self, state: WorldState, rng: np.random.Generator) -> bool:
        # Q-update from the PREVIOUS tick's now-known outcome, before
        # picking this tick's action -- standard online Q-learning.
        #
        # Learning rate: floored decay, alpha = max(alpha_floor, 1/visits).
        # It becomes constant after 1/alpha_floor visits. This is a practical
        # tracking heuristic, not an asymptotic convergence guarantee.
        self._learn_transition(state, terminal=False)

        state_bin = self._state_bin(state)
        if self.training and rng.random() < self.epsilon:
            action = bool(rng.random() < 0.5)
        else:
            action = self._q(state_bin, True) > self._q(state_bin, False)  # ties -> False, conservative

        self._last_bin = state_bin
        self._last_action = action
        return action

    def _learn_transition(self, state: WorldState, terminal: bool) -> None:
        if self.training and self._last_bin is not None and state.events:
            reward = -state.events[-1]["shortfall_this_tick"]
            current_bin = self._state_bin(state)
            best_next_q = 0.0 if terminal else max(self._q(current_bin, a) for a in (True, False))
            td_target = reward + self.gamma * best_next_q
            key = (self._last_bin, self._last_action)
            old_q = self.q_table.get(key, 0.0)
            self.visit_counts[key] = self.visit_counts.get(key, 0) + 1
            alpha = max(self.alpha_floor, 1.0 / self.visit_counts[key])
            self.q_table[key] = old_q + alpha * (td_target - old_q)

    def start_episode(self) -> None:
        """Reset transition bookkeeping while retaining learned values."""
        self._last_bin = None
        self._last_action = None

    def end_episode(self, state: WorldState) -> None:
        """Consume the last reward once, without bootstrapping past termination."""
        self._learn_transition(state, terminal=True)
        self.start_episode()

    def freeze(self) -> None:
        """Stop learning/exploring -- pure exploitation, for evaluation."""
        self.training = False
        self.epsilon = 0.0

    def q_table_report(self) -> str:
        """Human-readable dump: for each state bin, which action does the
        learner prefer, and by how much."""
        lines = []
        bins = sorted({b for (b, a) in self.q_table.keys()})
        for b in bins:
            q_true = self._q(b, True)
            q_false = self._q(b, False)
            pref = "MITIGATE" if q_true > q_false else "don't"
            lines.append(f"  bin={b}  Q(mitigate)={q_true:+7.2f}  Q(don't)={q_false:+7.2f}  -> prefers {pref}")
        return "\n".join(lines)
