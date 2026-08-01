"""
Tests for utils.hierarchical_pool.HierarchicalOpponentPool.

Focus: Ablation 1 — fitness-rectified hard-tier sampling.

Validates:
  - Basic two-tier add/sample behaviour still works (unchanged from v7).
  - record_outcome() updates a per-snapshot EMA score, persisted across reload.
  - hard_tier_weights() always returns a valid (non-negative, normalizable)
    distribution and never zeroes out a snapshot (uniform floor).
  - Rectified sampling shifts probability mass away from low-score snapshots
    and toward high-score ones, relative to the legacy recency-only sampler.
  - rectified=False reproduces the exact legacy p_latest behaviour.
  - Score bookkeeping survives FIFO eviction (stale scores pruned, no
    unbounded growth / KeyErrors).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent.dqn_v2_agent import DQNv2Agent
from utils.hierarchical_pool import HierarchicalOpponentPool


@pytest.fixture
def agent():
    return DQNv2Agent(device="cpu")


def _fill_hard_tier(pool: HierarchicalOpponentPool, agent, n: int, easy_max: int = 5):
    """Push `easy_max` easy snapshots then `n` hard snapshots; return hard paths."""
    for _ in range(easy_max):
        pool.add_snapshot(agent)
    paths = [pool.add_snapshot(agent) for _ in range(n)]
    return paths


# ---------------------------------------------------------------------------
# Basic tiering (unchanged behaviour)
# ---------------------------------------------------------------------------

class TestBasicTiering:
    def test_empty_pool_sample_returns_none(self, tmp_path):
        pool = HierarchicalOpponentPool(tmp_path / "pool")
        assert pool.sample(np.random.default_rng(0)) is None

    def test_easy_tier_fills_first(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=3, hard_max=5)
        for _ in range(3):
            pool.add_snapshot(agent)
        assert len(pool.easy_pool) == 3
        assert len(pool.hard_pool) == 0
        pool.add_snapshot(agent)
        assert len(pool.hard_pool) == 1

    def test_len_is_sum_of_tiers(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=2, hard_max=5)
        for _ in range(4):
            pool.add_snapshot(agent)
        assert len(pool) == 4


# ---------------------------------------------------------------------------
# record_outcome / EMA bookkeeping
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    def test_first_outcome_sets_score_directly(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=1, hard_max=5)
        [path] = _fill_hard_tier(pool, agent, n=1, easy_max=1)
        pool.record_outcome(path, 1.0)
        assert pool._scores[path] == pytest.approx(1.0)

    def test_subsequent_outcomes_use_ema(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=5, ema_alpha=0.1
        )
        [path] = _fill_hard_tier(pool, agent, n=1, easy_max=1)
        pool.record_outcome(path, 1.0)
        pool.record_outcome(path, 0.0)
        # EMA: 0.9*1.0 + 0.1*0.0 = 0.9
        assert pool._scores[path] == pytest.approx(0.9)

    def test_score_clipped_to_unit_interval(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=1, hard_max=5)
        [path] = _fill_hard_tier(pool, agent, n=1, easy_max=1)
        pool.record_outcome(path, 5.0)
        assert pool._scores[path] == pytest.approx(1.0)
        pool.record_outcome(path, -5.0)
        assert pool._scores[path] >= 0.0

    def test_scores_persist_across_reload(self, tmp_path, agent):
        pool_dir = tmp_path / "pool"
        pool = HierarchicalOpponentPool(pool_dir, easy_max=1, hard_max=5)
        [path] = _fill_hard_tier(pool, agent, n=1, easy_max=1)
        pool.record_outcome(path, 0.75)

        pool2 = HierarchicalOpponentPool(pool_dir, easy_max=1, hard_max=5)
        assert pool2._scores[path] == pytest.approx(0.75)

    def test_stale_scores_pruned_on_eviction(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=1, hard_max=2)
        paths = _fill_hard_tier(pool, agent, n=2, easy_max=1)
        for p in paths:
            pool.record_outcome(p, 0.5)
        assert set(pool._scores.keys()) == set(paths)

        # Evict paths[0] via FIFO by adding one more hard snapshot.
        new_path = pool.add_snapshot(agent)
        assert paths[0] not in pool.hard_pool.paths()
        assert paths[0] not in pool._scores
        assert new_path in pool.hard_pool.paths()


# ---------------------------------------------------------------------------
# hard_tier_weights() — validity of the rectified distribution
# ---------------------------------------------------------------------------

class TestHardTierWeights:
    def test_empty_hard_tier_returns_empty(self, tmp_path):
        pool = HierarchicalOpponentPool(tmp_path / "pool")
        assert pool.hard_tier_weights().size == 0

    def test_weights_nonnegative_and_normalizable(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=1, hard_max=5)
        paths = _fill_hard_tier(pool, agent, n=5, easy_max=1)
        for i, p in enumerate(paths):
            pool.record_outcome(p, score=i / 4.0)  # 0.0, 0.25, 0.5, 0.75, 1.0

        weights = pool.hard_tier_weights()
        assert weights.shape == (5,)
        assert (weights >= 0.0).all()
        total = weights.sum()
        assert total > 0.0
        probs = weights / total
        assert probs.sum() == pytest.approx(1.0)

    def test_no_snapshot_fully_excluded(self, tmp_path, agent):
        """Even the worst-performing snapshot keeps nonzero weight (uniform floor)."""
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=3, rectify_floor=0.1
        )
        paths = _fill_hard_tier(pool, agent, n=3, easy_max=1)
        # One snapshot Krishna is dominated by (score=0), others it beats (score=1).
        pool.record_outcome(paths[0], 0.0)
        pool.record_outcome(paths[1], 1.0)
        pool.record_outcome(paths[2], 1.0)

        weights = pool.hard_tier_weights()
        assert (weights > 0.0).all(), "every snapshot must retain nonzero weight"

    def test_low_score_snapshot_gets_less_weight_than_high_score(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=1, hard_max=2)
        paths = _fill_hard_tier(pool, agent, n=2, easy_max=1)
        pool.record_outcome(paths[0], 0.0)   # Krishna is dominated
        pool.record_outcome(paths[1], 1.0)   # Krishna is winning

        weights = pool.hard_tier_weights()
        assert weights[1] > weights[0]


# ---------------------------------------------------------------------------
# Rectified sampling — shifts mass toward beatable opponents
# ---------------------------------------------------------------------------

class TestRectifiedSampling:
    def test_rectified_sampling_favours_high_score_snapshot(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=2, rectified=True, rectify_floor=0.05,
        )
        paths = _fill_hard_tier(pool, agent, n=2, easy_max=1)
        pool.record_outcome(paths[0], 0.0)   # dominating Krishna -> low weight
        pool.record_outcome(paths[1], 1.0)   # Krishna winning -> high weight

        rng = np.random.default_rng(0)
        counts = {p: 0 for p in paths}
        for _ in range(2000):
            sampled = pool.sample(rng, p_easy=0.0)
            counts[sampled] += 1

        assert counts[paths[1]] > counts[paths[0]] * 3, (
            "rectified sampling should strongly favour the snapshot Krishna is "
            f"beating; got counts={counts}"
        )

    def test_floor_keeps_dominated_snapshot_reachable(self, tmp_path, agent):
        """The dominated snapshot must still be sampled sometimes, never 0%."""
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=2, rectified=True, rectify_floor=0.1,
        )
        paths = _fill_hard_tier(pool, agent, n=2, easy_max=1)
        pool.record_outcome(paths[0], 0.0)
        pool.record_outcome(paths[1], 1.0)

        rng = np.random.default_rng(1)
        counts = {p: 0 for p in paths}
        for _ in range(3000):
            counts[pool.sample(rng, p_easy=0.0)] += 1

        assert counts[paths[0]] > 0, (
            "uniform floor must keep the dominated snapshot reachable, not fully excluded"
        )

    def test_uniform_scores_give_near_uniform_sampling(self, tmp_path, agent):
        """When all snapshots score equally, rectified weights reduce to (near) uniform."""
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=3, rectified=True,
        )
        paths = _fill_hard_tier(pool, agent, n=3, easy_max=1)
        for p in paths:
            pool.record_outcome(p, 0.5)

        weights = pool.hard_tier_weights()
        assert np.allclose(weights, weights[0]), "equal scores should yield equal weights"


# ---------------------------------------------------------------------------
# rectified=False — legacy recency-only fallback preserved
# ---------------------------------------------------------------------------

class TestLegacyFallback:
    def test_rectified_false_delegates_to_hard_pool_sample(self, tmp_path, agent, monkeypatch):
        """With rectified=False, hard-tier sampling must delegate directly to
        OpponentPool.sample (pure p_latest recency) and never consult scores,
        regardless of how skewed the recorded outcomes are."""
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=2, rectified=False,
        )
        paths = _fill_hard_tier(pool, agent, n=2, easy_max=1)
        # Skew scores heavily — should have zero effect on legacy sampling.
        pool.record_outcome(paths[0], 0.0)
        pool.record_outcome(paths[1], 1.0)

        orig_sample = pool.hard_pool.sample
        calls: list[float] = []

        def spy(rng, p_latest=0.7):
            calls.append(p_latest)
            return orig_sample(rng, p_latest=p_latest)

        monkeypatch.setattr(pool.hard_pool, "sample", spy)

        rng = np.random.default_rng(3)
        for _ in range(10):
            pool.sample(rng, p_easy=0.0, p_latest=0.42)

        assert calls == [0.42] * 10, (
            "rectified=False must delegate every hard-tier draw to "
            "hard_pool.sample(p_latest=...) unchanged"
        )

    def test_default_is_rectified_true(self, tmp_path):
        pool = HierarchicalOpponentPool(tmp_path / "pool")
        assert pool.rectified is True


# ---------------------------------------------------------------------------
# mean_hard_tier_score() — logging convenience
# ---------------------------------------------------------------------------

class TestMeanHardTierScore:
    def test_zero_when_hard_tier_empty(self, tmp_path):
        pool = HierarchicalOpponentPool(tmp_path / "pool")
        assert pool.mean_hard_tier_score() == 0.0

    def test_matches_manual_mean(self, tmp_path, agent):
        pool = HierarchicalOpponentPool(tmp_path / "pool", easy_max=1, hard_max=3)
        paths = _fill_hard_tier(pool, agent, n=3, easy_max=1)
        pool.record_outcome(paths[0], 0.2)
        pool.record_outcome(paths[1], 0.8)
        # paths[2] never scored -> defaults to pool.default_score (0.5)
        expected = np.mean([0.2, 0.8, pool.default_score])
        assert pool.mean_hard_tier_score() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# rectify_floor — constructor default, and set_rectify_floor() runtime hook
#
# Added after ablation 1's first real run (versions/v7_ablation1_rectified.md)
# found a mid-training vulnerability window caused by full-strength
# rectification (fixed floor=0.1) kicking in at the same episode
# easy_warmup_eps ends. `rectify_floor` can now be raised at construction
# time, and/or ramped in gradually at runtime via set_rectify_floor()
# (train_phase3.py's --rectify-warmup-eps, mirroring the anchor_weight decay
# pattern — see TestSetAnchorWeight in tests/test_gated_dqn_agent.py).
# ---------------------------------------------------------------------------

class TestRectifyFloorDefault:
    def test_default_rectify_floor_is_one_tenth(self, tmp_path):
        """Backward compat: the completed ablation-1 run used rectify_floor=0.1
        without ever setting it explicitly, so the constructor default must
        stay 0.1 for that run to remain reproducible from its recorded args."""
        pool = HierarchicalOpponentPool(tmp_path / "pool")
        assert pool.rectify_floor == pytest.approx(0.1)

    def test_rectify_floor_settable_at_construction(self, tmp_path):
        pool = HierarchicalOpponentPool(tmp_path / "pool", rectify_floor=0.3)
        assert pool.rectify_floor == pytest.approx(0.3)


class TestSetRectifyFloor:
    def test_setter_updates_floor(self, tmp_path):
        pool = HierarchicalOpponentPool(tmp_path / "pool", rectify_floor=0.1)
        pool.set_rectify_floor(0.75)
        assert pool.rectify_floor == pytest.approx(0.75)
        pool.set_rectify_floor(0.1)
        assert pool.rectify_floor == pytest.approx(0.1)

    def test_setter_takes_effect_on_next_weights_call(self, tmp_path, agent):
        """set_rectify_floor() must not be cached — hard_tier_weights() should
        reflect the new floor immediately on the very next call, since the
        training loop calls it once per episode before sampling."""
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=2, rectify_floor=0.1
        )
        paths = _fill_hard_tier(pool, agent, n=2, easy_max=1)
        pool.record_outcome(paths[0], 0.0)
        pool.record_outcome(paths[1], 1.0)

        low_floor_weights = pool.hard_tier_weights()
        pool.set_rectify_floor(1.0)
        high_floor_weights = pool.hard_tier_weights()

        # Raising the floor must not decrease any weight, and must shrink the
        # gap between the highest- and lowest-weighted snapshot (closer to
        # uniform), matching the "raise the floor" mitigation's intent.
        assert (high_floor_weights >= low_floor_weights).all()
        low_ratio = low_floor_weights.max() / low_floor_weights.min()
        high_ratio = high_floor_weights.max() / high_floor_weights.min()
        assert high_ratio < low_ratio


class TestUniformWarmupFloor:
    def test_constant_is_one(self):
        assert HierarchicalOpponentPool.UNIFORM_WARMUP_FLOOR == pytest.approx(1.0)

    def test_uniform_warmup_floor_yields_near_uniform_weights(self, tmp_path, agent):
        """UNIFORM_WARMUP_FLOOR is meant as a near-uniform starting point for
        the rectify_warmup_eps ramp. Since scores/baseline are bounded to
        [0, 1], a floor of 1.0 should keep the max/min weight ratio small
        (<= 2x) even under maximally skewed scores — much flatter than the
        default floor=0.1, which can produce an 11x skew in the same case."""
        pool = HierarchicalOpponentPool(
            tmp_path / "pool", easy_max=1, hard_max=2,
            rectify_floor=HierarchicalOpponentPool.UNIFORM_WARMUP_FLOOR,
        )
        paths = _fill_hard_tier(pool, agent, n=2, easy_max=1)
        pool.record_outcome(paths[0], 0.0)   # maximally dominated
        pool.record_outcome(paths[1], 1.0)   # maximally winning

        weights = pool.hard_tier_weights()
        ratio = weights.max() / weights.min()
        assert ratio <= 2.0 + 1e-9, f"expected near-uniform weights, got ratio={ratio}"
