"""The epistemic-delegation institution, per PREREGISTRATION.md §3 and §7.

This is the actual mechanism the project is named for: citizens hold two
separate equal, nontransferable ballots each term -- a MANDATE ballot
(authorize mitigation spending this term, or not) and an EXECUTOR ballot
(elect one of four frozen, qualified candidates to make the tick-by-tick
technical call within that authorization). Executors don't learn. Citizens'
voting rule is itself a fixed script for now, not learned -- this is Phase 2
("substrate without learning") per the roadmap, still one step before any
learning agent.

Four candidates spanning a real skill range, all reading only observable
messages (never the hidden blight state -- I-7):

- responsive       -- mitigates when the unique-origin report rate is
                       elevated (the "smart, lineage-aware" instrument)
- responsive_naive -- mitigates when the RAW message rate is elevated
                       (the "gets fooled by forward storms" instrument)
- constant_spender  -- mitigates almost every tick once authorized
                       (wasteful: drains the term's budget fast)
- cautious          -- mitigates only on a very strong signal
                       (under-reacts: saves budget, risks missing real need)

Public qualification: each candidate's score is its mean need-shortfall on a
held-out CALIBRATION seed set, run standalone with mandate always authorized
(isolating instrument skill from the mandate decision) -- never the same
seeds the actual election run uses, and never anything about the true hidden
state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from no_way_home.messages import MessageLog
from no_way_home.metrics import need_shortfall_per_10k
from no_way_home.world import WorldConfig, WorldState

InstrumentFn = Callable[[WorldState, np.random.Generator], bool]


def instrument_responsive(state: WorldState, rng: np.random.Generator) -> bool:
    rate = state.messages.unique_origin_rate(current_tick=state.tick + 1, window=30)
    return rate > 0.30


def instrument_responsive_naive(state: WorldState, rng: np.random.Generator) -> bool:
    rate = state.messages.raw_message_rate(current_tick=state.tick + 1, window=30)
    return rate > 0.30


def instrument_constant_spender(state: WorldState, rng: np.random.Generator) -> bool:
    return True


def instrument_cautious(state: WorldState, rng: np.random.Generator) -> bool:
    rate = state.messages.unique_origin_rate(current_tick=state.tick + 1, window=30)
    return rate > 0.60  # double the responsive threshold -- waits for stronger evidence


CANDIDATES: dict[str, InstrumentFn] = {
    "A_responsive": instrument_responsive,
    "B_responsive_naive": instrument_responsive_naive,
    "C_constant_spender": instrument_constant_spender,
    "D_cautious": instrument_cautious,
}


def _mandate_authorized(state: WorldState, window: int, threshold: float) -> bool:
    """Citizen mandate vote, simplified to a scripted rule (not learned):
    authorize mitigation spending this term iff the unique-origin report
    rate looks elevated. Citizens use their own lineage-aware reading here
    regardless of which instrument they're about to elect -- the mandate and
    the executor choice are genuinely separate powers (I-6).

    Deliberately a much lower threshold than any instrument's own trigger
    (0.30 for responsive/naive, 0.60 for cautious): the mandate's job is a
    coarse "is anything going on at all" gate (pre-shift noise ceiling is
    ~0.03, post-shift floor is ~1.2, so 0.10 sits comfortably between them),
    not "should we act right now" -- that's the instrument's job, within an
    authorized term. Setting them to the same threshold was a real bug found
    by running this: mandate-authorized then trivially implied
    responsive-triggered on the same signal, and raw rate >= unique rate
    means naive and constant collapsed into the same thing too. All three
    scored identically until this was separated out."""
    rate = state.messages.unique_origin_rate(current_tick=state.tick + 1, window=window)
    return rate > threshold


def calibrate_public_qualification(cfg: WorldConfig, calibration_seeds: list, mandate_window: int,
                                    mandate_threshold: float) -> dict:
    """Run each candidate standalone (mandate gate still applied, so the
    score reflects real achievable performance, not a hypothetical
    always-authorized world) on held-out calibration seeds. Returns
    {candidate_name: mean_need_shortfall}, lower is better -- this is the
    PUBLIC score voters see; it is real evidence, not the true hidden-state
    performance, and it is dated (computed once, before the election run,
    exactly as real qualification evidence would be)."""
    from no_way_home.world import run  # local import: avoid a cycle at module load

    scores = {}
    for name, instrument in CANDIDATES.items():
        def gated_policy(state, rng, _instrument=instrument):
            if not _mandate_authorized(state, mandate_window, mandate_threshold):
                return False
            return _instrument(state, rng)

        vals = [need_shortfall_per_10k(run(cfg, gated_policy, seed=s)) for s in calibration_seeds]
        scores[name] = sum(vals) / len(vals)
    return scores


@dataclass
class ElectionRecord:
    tick: int
    mandate_authorized: bool
    winner: str
    vote_tally: dict


@dataclass
class EpistemicDelegationInstitution:
    """Stateful policy_fn (matches the same signature every other policy in
    policies.py uses -- __call__(state, rng) -> bool) implementing one full
    term cycle: at the start of each term, hold the executor election (and,
    unless reevaluate_mandate_every_tick, the mandate vote too); every tick
    within the term, defer to the elected executor's instrument policy,
    gated by the mandate.

    reevaluate_mandate_every_tick exists for exactly one purpose: isolating
    the cost of a TERM-LENGTH mandate commitment from the cost of noisy
    voting. Found by running this -- with the mandate committed once per
    200-tick term (the real design, matching PREREGISTRATION.md §7.2's
    reselection-at-fixed-terms structure), the institution scored ~4-5%
    worse than a hard-coded best-executor baseline even when the election
    picked the objectively-best candidate 100% of the time. Re-evaluating
    the mandate every tick instead closes the gap to exactly zero (checked
    directly: 0/2000 tick mismatches across 5 seeds) -- so the entire cost
    is the mandate's commitment length, not election/voting noise at all.
    That's a real, mechanism-isolated finding, not a design flaw to silently
    patch away: a term-length mandate is *supposed* to be less responsive
    than continuous re-authorization -- that's the whole point of having a
    term -- and this is what that tradeoff actually costs, measured."""
    qualification_scores: dict
    election_interval: int = 200
    mandate_window: int = 30
    mandate_threshold: float = 0.10
    n_voters: int = 24
    correct_vote_prob: float = 0.7  # fraction of voters who vote for the best-scored candidate
    reevaluate_mandate_every_tick: bool = False
    history: list = field(default_factory=list)
    _current_executor: str | None = field(default=None, init=False)
    _current_mandate: bool = field(default=False, init=False)

    def __call__(self, state: WorldState, rng: np.random.Generator) -> bool:
        if self._current_executor is None or state.tick % self.election_interval == 0:
            self._current_mandate = _mandate_authorized(state, self.mandate_window, self.mandate_threshold)
            self._current_executor, tally = self._hold_election(rng)
            self.history.append(ElectionRecord(
                tick=state.tick, mandate_authorized=self._current_mandate,
                winner=self._current_executor, vote_tally=tally,
            ))
        if self.reevaluate_mandate_every_tick:
            self._current_mandate = _mandate_authorized(state, self.mandate_window, self.mandate_threshold)
        if not self._current_mandate:
            return False
        return CANDIDATES[self._current_executor](state, rng)

    def _hold_election(self, rng: np.random.Generator) -> tuple:
        best = min(self.qualification_scores, key=self.qualification_scores.get)
        names = list(self.qualification_scores.keys())
        tally = {n: 0 for n in names}
        for _ in range(self.n_voters):
            choice = best if rng.random() < self.correct_vote_prob else names[rng.integers(len(names))]
            tally[choice] += 1
        winner = max(tally, key=tally.get)
        return winner, tally
