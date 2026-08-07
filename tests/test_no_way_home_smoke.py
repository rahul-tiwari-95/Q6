"""Correctness tests for the No Way Home A1 nontriviality smoke test.

Not a replay/hash-audit suite like the parked seven-contract system in
.memory/no-way-home-phase0-v2/ -- just the basics that would actually catch
a real bug: determinism, conservation, and the regime shift firing when it
should.
"""

import numpy as np
import pytest

from no_way_home.institutions import (
    CANDIDATES,
    EpistemicDelegationInstitution,
    calibrate_public_qualification,
)
from no_way_home.messages import MessageLog
from no_way_home.metrics import mitigation_rate, need_shortfall_per_10k
from no_way_home.policies import (
    LINEAGE_AWARE_THRESHOLD,
    LINEAGE_NAIVE_THRESHOLD,
    MESSAGE_WINDOW,
    POLICIES,
    lineage_aware_heuristic,
    lineage_naive_heuristic,
)
from no_way_home.world import WorldConfig, WorldState, run


def test_same_seed_same_policy_is_deterministic():
    cfg = WorldConfig(n_ticks=200)
    s1 = run(cfg, POLICIES["C1_zero_intelligence"], seed=42)
    s2 = run(cfg, POLICIES["C1_zero_intelligence"], seed=42)
    assert s1.events == s2.events


def test_different_seeds_produce_different_trajectories():
    cfg = WorldConfig(n_ticks=200)
    s1 = run(cfg, POLICIES["C1_zero_intelligence"], seed=1)
    s2 = run(cfg, POLICIES["C1_zero_intelligence"], seed=2)
    assert s1.events != s2.events


def test_regime_shift_fires_at_configured_tick():
    cfg = WorldConfig(n_ticks=50, shift_tick=20)
    state = run(cfg, POLICIES["never_mitigate"], seed=0)
    pre = [e for e in state.events if e["tick"] < 20]
    post = [e for e in state.events if e["tick"] >= 20]
    assert all(not e["blight_high"] for e in pre)
    assert all(e["blight_high"] for e in post)


def test_never_mitigate_is_never_worse_than_always_mitigate_under_stable_low_blight():
    """Under a world that never shifts (shift_tick beyond n_ticks), paying
    the mitigation cost is pure waste -- always_mitigate must not beat
    never_mitigate there. This is a real invariant, not a tuned expectation."""
    cfg = WorldConfig(n_ticks=500, shift_tick=10_000)  # shift never fires
    never = run(cfg, POLICIES["never_mitigate"], seed=0)
    always = run(cfg, POLICIES["always_mitigate"], seed=0)
    assert need_shortfall_per_10k(never) <= need_shortfall_per_10k(always)


def test_never_mitigate_is_worse_than_always_mitigate_under_stable_high_blight():
    """Symmetric check: if blight is high from tick 1, never-mitigating
    must not beat always-mitigating."""
    cfg = WorldConfig(n_ticks=500, shift_tick=1)  # shift fires immediately
    never = run(cfg, POLICIES["never_mitigate"], seed=0)
    always = run(cfg, POLICIES["always_mitigate"], seed=0)
    assert need_shortfall_per_10k(always) <= need_shortfall_per_10k(never)


def test_stocks_never_go_negative():
    cfg = WorldConfig(n_ticks=500)
    for policy_name, policy_fn in POLICIES.items():
        state = run(cfg, policy_fn, seed=0)
        for e in state.events:
            assert e["food_stock"] >= 0, policy_name
            assert e["medicine_stock"] >= 0, policy_name


def test_need_shortfall_metric_matches_manual_sum():
    cfg = WorldConfig(n_ticks=300)
    state = run(cfg, POLICIES["never_mitigate"], seed=0)
    manual = sum(e["shortfall_this_tick"] for e in state.events) / len(state.events) * 10_000
    assert need_shortfall_per_10k(state) == pytest.approx(manual)


def test_mitigation_rate_zero_for_never_mitigate():
    cfg = WorldConfig(n_ticks=200)
    state = run(cfg, POLICIES["never_mitigate"], seed=0)
    assert mitigation_rate(state) == 0.0


def test_raw_vs_unique_message_rate_distinguishes_copies_from_independent_reports():
    """The exact I-7 claim, checked directly: ten independent reports and
    one report forwarded ten times must look different under unique-origin
    counting and must NOT be reliably distinguishable under raw counting."""
    independent = MessageLog()
    for loc in range(10):
        independent.report(tick=5, locality=loc)

    copies = MessageLog()
    copies.report(tick=5, locality=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        copies.maybe_forward(tick=5, rng=rng, hub_locality=None, hub_forward_boost=1.0)

    assert independent.unique_origin_rate(6, 10) == pytest.approx(1.0)
    assert copies.unique_origin_rate(6, 10) == pytest.approx(0.1)
    # raw rate is roughly the same order of magnitude for both -- that's the
    # whole point, a naive counter can't tell them apart
    assert copies.raw_message_rate(6, 10) > copies.unique_origin_rate(6, 10) * 5


def test_lineage_aware_ignores_a_forward_storm_that_fools_lineage_naive():
    """Constructed, deterministic version of what the full-world runs found
    empirically (results/provenance_test_v1.md): a single report, forwarded
    many times, crosses the naive raw-count threshold but not the
    unique-origin threshold at the same numeric cutoff."""
    cfg = WorldConfig()
    state = WorldState.initial(cfg)
    state.tick = 29  # so the [tick+1-window, tick+1) window covers tick 5 onward
    state.messages.report(tick=5, locality=0)
    rng = np.random.default_rng(0)
    for _ in range(15):
        state.messages.maybe_forward(tick=6, rng=rng, hub_locality=None, hub_forward_boost=1.0)

    raw_rate = state.messages.raw_message_rate(state.tick + 1, MESSAGE_WINDOW)
    unique_rate = state.messages.unique_origin_rate(state.tick + 1, MESSAGE_WINDOW)
    assert raw_rate > LINEAGE_NAIVE_THRESHOLD, "test setup should cross the naive threshold"
    assert unique_rate <= LINEAGE_AWARE_THRESHOLD, "test setup should NOT cross the aware threshold"

    assert lineage_naive_heuristic(state, rng) is True
    assert lineage_aware_heuristic(state, rng) is False


def test_world_physics_identical_across_policies_that_consume_different_amounts_of_randomness():
    """The real bug found while building the election test: a policy that
    draws extra randomness internally (e.g. holding a 24-voter election)
    must NOT perturb the physical world's sickness/forwarding trajectory
    under the same seed. Compare a policy that consumes zero internal
    randomness (never_mitigate) against one that consumes a lot (many calls
    to rng.random() via zero_intelligence, called every tick) -- the
    sequence of blight_high / n_sick per locality must be byte-identical."""
    cfg = WorldConfig(n_ticks=300)
    quiet = run(cfg, POLICIES["never_mitigate"], seed=7)
    noisy = run(cfg, POLICIES["C1_zero_intelligence"], seed=7)
    quiet_physics = [(e["blight_high"], e["sick_per_locality"]) for e in quiet.events]
    noisy_physics = [(e["blight_high"], e["sick_per_locality"]) for e in noisy.events]
    assert quiet_physics == noisy_physics


def test_zero_intelligence_constrained_never_exceeds_wealth():
    """ZI-C must never leave wealth negative -- the whole point of the
    constraint (Gode & Sunder's no-loss rule, I-9)."""
    cfg = WorldConfig(n_ticks=500)
    state = run(cfg, POLICIES["C2_zero_intelligence_constrained"], seed=0)
    for e in state.events:
        assert e["wealth"] >= 0 - 1e-9


def test_election_reliably_selects_the_true_best_candidate():
    """With correct_vote_prob=0.7 and 24 voters, plurality voting should
    converge to the objectively-best candidate essentially every term --
    checked directly against held-out calibration scores, not assumed."""
    cfg = WorldConfig()
    scores = calibrate_public_qualification(cfg, list(range(1000, 1005)), mandate_window=30, mandate_threshold=0.10)
    true_best = min(scores, key=scores.get)
    for seed in range(3):
        institution = EpistemicDelegationInstitution(qualification_scores=scores)
        run(cfg, institution, seed=seed)
        winners = [r.winner for r in institution.history]
        assert winners.count(true_best) / len(winners) >= 0.9


def test_reactive_mandate_ablation_matches_hardcoded_baseline_exactly():
    """The mechanism-isolation finding, locked in: with the SAME election
    but the mandate re-evaluated every tick instead of committed for the
    200-tick term, the institution's mitigate decision must match a
    hard-coded best-executor baseline on every single tick -- proving the
    term-committed institution's cost (see results/election_test_v1.md) is
    entirely about commitment length, not about voting or qualification."""
    cfg = WorldConfig(n_ticks=1000)
    scores = calibrate_public_qualification(cfg, list(range(1000, 1005)), mandate_window=30, mandate_threshold=0.10)
    best = min(scores, key=scores.get)

    def hardcoded_best(state, rng):
        from no_way_home.institutions import _mandate_authorized
        if not _mandate_authorized(state, 30, 0.10):
            return False
        return CANDIDATES[best](state, rng)

    reactive = EpistemicDelegationInstitution(qualification_scores=scores, reevaluate_mandate_every_tick=True)
    hardcoded_state = run(cfg, hardcoded_best, seed=0)
    reactive_state = run(cfg, reactive, seed=0)
    mismatches = sum(1 for a, b in zip(hardcoded_state.events, reactive_state.events) if a["mitigate"] != b["mitigate"])
    assert mismatches == 0
