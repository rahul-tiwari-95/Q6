"""
Smoke/integration tests for train_phase3.py's rectify-floor warmup schedule.

Ablation 1's first real run (versions/v7_ablation1_rectified.md) found a
mid-training vulnerability window caused by full-strength rectification
(fixed rectify_floor=0.1) kicking in at the exact same episode the
easy_warmup_eps curriculum ends. `--rectify-warmup-eps` decouples the two by
ramping the *effective* floor from a near-uniform starting value
(HierarchicalOpponentPool.UNIFORM_WARMUP_FLOOR) down to the configured
rectify_floor over that many episodes, mirroring how anchor_weight decays
per-episode for ablation 2.

These tests spy on HierarchicalOpponentPool.set_rectify_floor() — the hook
train_phase3.py's per-episode loop calls — to verify:
  1. rectify_warmup_eps=0 (default) never touches the floor at runtime, so
     the completed ablation-1 run's exact behaviour is unchanged.
  2. rectify_warmup_eps>0 ramps the floor linearly and matches the documented
     formula exactly, clamping at the configured rectify_floor once the
     warmup window elapses.

Follows the run_training()-smoke-test pattern already established in
tests/test_train_phase2.py (5 episodes, device="cpu", REPO_ROOT redirected
to tmp_path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import train_phase3 as tp3
from utils.hierarchical_pool import HierarchicalOpponentPool


def _run_smoke(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(tp3, "REPO_ROOT", tmp_path)
    (tmp_path / "training_runs").mkdir()

    calls: list[float] = []
    orig_setter = HierarchicalOpponentPool.set_rectify_floor

    def spy(self, floor):
        calls.append(float(floor))
        return orig_setter(self, floor)

    monkeypatch.setattr(HierarchicalOpponentPool, "set_rectify_floor", spy)

    kwargs = dict(
        episodes=5, device="cpu", seed=1, name="smoke_rectify_warmup",
        snapshot_every=2, p_latest=0.5, replay_every=5, warm_start=None,
        cher_safe_dist=6, cher_pellet_reach=4, cher_bonus=30.0,
        epsilon_start=0.15,
    )
    kwargs.update(overrides)
    tp3.run_training(**kwargs)
    return calls


class TestRectifyWarmupDefaultOff:
    def test_zero_warmup_never_calls_setter(self, tmp_path, monkeypatch):
        """rectify_warmup_eps=0 (the default) must leave the floor exactly as
        constructed for the whole run — no set_rectify_floor() calls at all —
        so the completed ablation-1 run stays reproducible from its recorded
        args (rectify_warmup_eps didn't exist when it ran, so its absence
        must be indistinguishable from 'off')."""
        calls = _run_smoke(
            tmp_path, monkeypatch,
            rectify_floor=0.1, rectify_warmup_eps=0,
        )
        assert calls == []


class TestRectifyWarmupRamp:
    def test_ramp_matches_linear_formula_and_clamps(self, tmp_path, monkeypatch):
        target_floor = 0.2
        warmup_eps = 4
        calls = _run_smoke(
            tmp_path, monkeypatch,
            rectify_floor=target_floor, rectify_warmup_eps=warmup_eps,
        )

        assert len(calls) == 5  # one call per episode
        start = HierarchicalOpponentPool.UNIFORM_WARMUP_FLOOR
        expected = [
            start + (target_floor - start) * min(1.0, ep / warmup_eps)
            for ep in range(1, 6)
        ]
        for got, want in zip(calls, expected):
            assert got == pytest.approx(want)

        # Strictly tightening while ramping, then clamped flat at the target.
        assert calls[0] > calls[1] > calls[2] > calls[3]
        assert calls[3] == pytest.approx(target_floor)
        assert calls[4] == pytest.approx(target_floor)

    def test_first_episode_starts_near_uniform_floor(self, tmp_path, monkeypatch):
        """With a long warmup window, episode 1's effective floor should sit
        close to UNIFORM_WARMUP_FLOOR, not the tightened target — this is the
        whole point of decoupling from easy_warmup_eps ending abruptly."""
        calls = _run_smoke(
            tmp_path, monkeypatch,
            rectify_floor=0.1, rectify_warmup_eps=5000,
        )
        assert calls[0] == pytest.approx(
            HierarchicalOpponentPool.UNIFORM_WARMUP_FLOOR, rel=1e-2
        )
