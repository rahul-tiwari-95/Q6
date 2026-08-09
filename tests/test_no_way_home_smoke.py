"""Correctness tests for the No Way Home A1 nontriviality smoke test.

Not a replay/hash-audit suite like the parked seven-contract system in
.memory/no-way-home-phase0-v2/ -- just the basics that would actually catch
a real bug: determinism, conservation, and the regime shift firing when it
should.
"""

import numpy as np
import pytest

from no_way_home.channels import CHANNELS, ChannelTag
from no_way_home.institutions import (
    CANDIDATES,
    EpistemicDelegationInstitution,
    calibrate_public_qualification,
)
from no_way_home.learning import RATE_BINS, TabularQMandateLearner, feature_raw_rate, feature_unique_rate
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


def test_lineage_role_matches_is_forward_exactly():
    """Increment 1 of ENVIRONMENT_REDESIGN.md: lineage_role is a derived
    property, not new state, so it must never disagree with is_forward --
    every REPORT (is_forward=False) Initiates a fresh lineage, every FORWARD
    (is_forward=True) is Happens-only. Reuses the same independent-reports-
    vs-forwarded-copies construction as
    test_raw_vs_unique_message_rate_distinguishes_copies_from_independent_reports
    rather than a new fixture, since that's already the canonical way this
    suite builds a log with both message kinds present."""
    independent = MessageLog()
    for loc in range(10):
        independent.report(tick=5, locality=loc)
    assert len(independent.messages) == 10
    assert all(m.lineage_role == "Initiates" for m in independent.messages)

    copies = MessageLog()
    copies.report(tick=5, locality=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        copies.maybe_forward(tick=5, rng=rng, hub_locality=None, hub_forward_boost=1.0)
    reports = [m for m in copies.messages if not m.is_forward]
    forwards = [m for m in copies.messages if m.is_forward]
    assert len(reports) == 1 and len(forwards) == 10, "test setup should produce 1 report + 10 forwards"
    assert all(m.lineage_role == "Initiates" for m in reports)
    assert all(m.lineage_role == "Happens-only" for m in forwards)


def test_channel_manipulability_tags_match_environment_redesign():
    """Regression guard on channels.py's registry itself: an accidental
    edit (e.g. a typo changing REPORT's manipulability from "signal" to
    something else) would silently invalidate the audit these tags exist
    to support, without touching any simulation behavior -- nothing else
    in the suite would catch it."""
    assert CHANNELS["blight_exposure"] == ChannelTag(emitter_class="autogenic", manipulability="index")
    assert CHANNELS["resource_ledger"] == ChannelTag(emitter_class="autogenic", manipulability="cue")
    assert CHANNELS["report"] == ChannelTag(emitter_class="allogenic", manipulability="signal")
    assert CHANNELS["forward"] == ChannelTag(emitter_class="allogenic", manipulability="index-of-a-claim")


def test_blight_exposure_index_channel_is_invariant_to_policy():
    """The manipulability invariant itself (ENVIRONMENT_REDESIGN.md §2):
    blight_exposure is tagged `index` with zero agent write access -- no
    allogenic (agent-driven) emitter, i.e. no policy's mitigation choice or
    message-generation code, may influence WorldState.blight_high. Checked
    structurally, not by re-reading world.py's source: every registered
    policy's blight_high trajectory for the same seed must be identical,
    since the only thing that's allowed to set it is step()'s own
    regime-shift logic. If any policy's own actions ever leaked into the
    regime signal -- the exact failure this tag exists to rule out -- this
    test would catch it directly."""
    assert CHANNELS["blight_exposure"].manipulability == "index"
    assert CHANNELS["blight_exposure"].emitter_class == "autogenic"

    cfg = WorldConfig(n_ticks=1000, shift_tick=400)
    reference = [e["blight_high"] for e in run(cfg, POLICIES["never_mitigate"], seed=7).events]
    assert any(reference), "test setup should actually cross into the high-blight regime"

    for policy_name, policy_fn in POLICIES.items():
        trajectory = [e["blight_high"] for e in run(cfg, policy_fn, seed=7).events]
        assert trajectory == reference, policy_name


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


def _train_both_features_learner(cfg, n_episodes=50, start_seed=2000):
    learner = TabularQMandateLearner(feature_fns=[feature_raw_rate, feature_unique_rate],
                                      bin_edges=[RATE_BINS, RATE_BINS])
    for seed in range(start_seed, start_seed + n_episodes):
        run(cfg, learner, seed=seed)
    return learner


def test_floored_decay_learner_reliably_avoids_forward_storm_bins():
    """The actual headline finding, locked in: with the real (floored-decay)
    learner, the forward-storm-signature bins (high raw_rate, low
    unique_rate) consistently prefer NOT mitigating, and this holds across
    two different training-data amounts -- i.e. it's converged, not luck."""
    cfg = WorldConfig()
    forward_storm_bins = [(3, 1), (4, 1)]

    for n_episodes in (50, 200):
        learner = _train_both_features_learner(cfg, n_episodes=n_episodes)
        visited = {b for b, a in learner.q_table.keys()}
        for b in forward_storm_bins:
            assert b in visited, f"expected bin {b} to be visited with {n_episodes} training episodes"
            assert learner._q(b, True) <= learner._q(b, False), (
                f"bin {b} should prefer NOT mitigating after {n_episodes} episodes, "
                f"got Q(mitigate)={learner._q(b, True)} Q(dont)={learner._q(b, False)}"
            )


def test_floored_decay_learner_outcome_is_stable_across_training_amounts():
    """Companion to the bin-level check: aggregate outcome shouldn't swing
    wildly between 50 and 200 training episodes either, unlike the
    constant-alpha version did during development (60864 -> 63491, and a
    bin's preference flipped) -- see results/learning_citizen_v1.md."""
    cfg = WorldConfig()
    means = []
    for n_episodes in (50, 200):
        learner = _train_both_features_learner(cfg, n_episodes=n_episodes)
        learner.freeze()
        vals = [need_shortfall_per_10k(run(cfg, learner, seed=s)) for s in range(10)]
        means.append(sum(vals) / len(vals))
    relative_diff = abs(means[0] - means[1]) / means[0]
    assert relative_diff < 0.05, f"outcome should be stable across training amounts, got {means}"
