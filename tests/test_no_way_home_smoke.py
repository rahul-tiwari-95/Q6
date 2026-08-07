"""Correctness tests for the No Way Home A1 nontriviality smoke test.

Not a replay/hash-audit suite like the parked seven-contract system in
.memory/no-way-home-phase0-v2/ -- just the basics that would actually catch
a real bug: determinism, conservation, and the regime shift firing when it
should.
"""

import pytest

from no_way_home.metrics import mitigation_rate, need_shortfall_per_10k
from no_way_home.policies import POLICIES
from no_way_home.world import WorldConfig, run


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


def test_zero_intelligence_constrained_never_exceeds_wealth():
    """ZI-C must never leave wealth negative -- the whole point of the
    constraint (Gode & Sunder's no-loss rule, I-9)."""
    cfg = WorldConfig(n_ticks=500)
    state = run(cfg, POLICIES["C2_zero_intelligence_constrained"], seed=0)
    for e in state.events:
        assert e["wealth"] >= 0 - 1e-9
