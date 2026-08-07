"""Permanent nonlearning control policies (cognition ladder C1-C4, plus the
two trivial fixed scripts), per PREREGISTRATION.md §2 (I-9) and §6.

Every policy has the same signature: policy_fn(state, rng) -> bool (mitigate
this tick or not). None of them learn; none of them update parameters. The
oracle is the one exception that sees hidden state directly -- it exists
purely as an upper-bound diagnostic, never as a claim about what's achievable
without cheating.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from no_way_home.world import WorldState

PolicyFn = Callable[[WorldState, np.random.Generator], bool]

# How many recent ticks the need-first heuristic looks at to estimate
# current sickness rate. A real lagging indicator, not hidden-state access.
NEED_FIRST_WINDOW = 20
NEED_FIRST_SICKNESS_THRESHOLD = 0.15  # observed sick-fraction that triggers mitigation


def zero_intelligence(state: WorldState, rng: np.random.Generator) -> bool:
    """C1: legal random action, no constraint at all."""
    return bool(rng.random() < 0.5)


def zero_intelligence_constrained(state: WorldState, rng: np.random.Generator) -> bool:
    """C2 (ZI-C): random, but never proposes mitigation the world can't
    afford -- the minimal-rationality / no-loss constraint from Gode &
    Sunder (1993), the paper I-9 explicitly names."""
    if state.wealth < state.cfg.mitigation_wealth_cost:
        return False
    return bool(rng.random() < 0.5)


def need_first_heuristic(state: WorldState, rng: np.random.Generator) -> bool:
    """C3: fixed scripted heuristic. Mitigate if the OBSERVED sickness rate
    over the last NEED_FIRST_WINDOW ticks exceeds a threshold. This uses only
    information any agent could actually observe (recent sickness), not the
    hidden blight regime -- so it necessarily lags the true regime shift by
    however many ticks it takes the sickness rate to rise and cross the
    threshold."""
    recent = state.events[-NEED_FIRST_WINDOW:]
    if not recent:
        return False
    observed_rate = sum(e["n_sick"] for e in recent) / (len(recent) * state.cfg.n_agents)
    return observed_rate > NEED_FIRST_SICKNESS_THRESHOLD


MESSAGE_WINDOW = 30
# Deliberately the SAME threshold for both -- same decision rule ("act if
# seeing > 0.3 pieces of evidence per tick"), differing only in what counts
# as one piece of evidence: a raw message, or a deduplicated origin.
LINEAGE_NAIVE_THRESHOLD = 0.30
LINEAGE_AWARE_THRESHOLD = 0.30


def lineage_naive_heuristic(state: WorldState, rng: np.random.Generator) -> bool:
    """Mitigate if raw message volume (REPORT + every FORWARD, uncorrected)
    over the last MESSAGE_WINDOW ticks crosses a threshold. This is the
    policy I-7 warns about directly: it can't tell one loudly-forwarded
    report from many independent ones."""
    rate = state.messages.raw_message_rate(current_tick=state.tick + 1, window=MESSAGE_WINDOW)
    return rate > LINEAGE_NAIVE_THRESHOLD


def lineage_aware_heuristic(state: WorldState, rng: np.random.Generator) -> bool:
    """Same trigger logic as lineage_naive_heuristic, but counting unique
    origin_ids instead of raw messages -- dedupes forwards of the same
    report before deciding whether there's real evidence of trouble."""
    rate = state.messages.unique_origin_rate(current_tick=state.tick + 1, window=MESSAGE_WINDOW)
    return rate > LINEAGE_AWARE_THRESHOLD


def always_mitigate(state: WorldState, rng: np.random.Generator) -> bool:
    return True


def never_mitigate(state: WorldState, rng: np.random.Generator) -> bool:
    return False


def greedy_state_oracle(state: WorldState, rng: np.random.Generator) -> bool:
    """Diagnostic only, per PREREGISTRATION.md §4: sees the hidden blight
    state directly and mitigates exactly when it's active. NOT a strict
    regret upper bound -- it has perfect STATE information but a naive,
    non-budget-aware spending policy (try every tick blight is high), so it
    can legitimately be beaten by a policy with worse information but better
    spending discipline. That turned out to matter in the smoke test: see
    results/smoke_test_v1.md. A true upper bound would need to jointly solve
    state inference AND optimal wealth-timing, which this doesn't attempt --
    that gap is itself the argument for why a learned policy might add
    something beyond either half alone."""
    return bool(state.blight_high)


POLICIES: dict[str, PolicyFn] = {
    "C1_zero_intelligence": zero_intelligence,
    "C2_zero_intelligence_constrained": zero_intelligence_constrained,
    "C3_need_first_heuristic": need_first_heuristic,
    "C3b_lineage_naive_heuristic": lineage_naive_heuristic,
    "C3c_lineage_aware_heuristic": lineage_aware_heuristic,
    "always_mitigate": always_mitigate,
    "never_mitigate": never_mitigate,
    "greedy_state_oracle": greedy_state_oracle,
}
