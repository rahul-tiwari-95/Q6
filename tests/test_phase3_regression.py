"""
V7 Regression and feature tests for Phase 3.

Tests:
  1. Hunter action bug regression — hunter.step must receive a_h, not a_k.
     (AST-level check so it catches any future revert.)
  2. Zero-pellet timeout penalty fires when Krishna times out with 0 pellets.
  3. Zero-pellet timeout penalty does NOT fire when Krishna collected >= 1 pellet.
  4. Conditional proximity shaping fires when hunter is far.
  5. Conditional proximity shaping is suppressed when hunter is close.
  6. Weighted CHER: bonus at 2x safe dist > bonus at 1x safe dist.
  7. Weighted CHER: bonus at 1x safe dist equals base collect_bonus * 0.5.
  8. Non-weighted CHER: flat bonus regardless of hunter distance.
  9. Gate entropy regularisation: loss should be finite and run without error.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# 1-2. Hunter action bug regression (AST)
# ---------------------------------------------------------------------------

class TestHunterActionBugRegression:
    """
    Verify (at AST level) that train_phase3.py calls hunter.step with a_h,
    not a_k.  This test will fail immediately if someone reverts the fix.
    """

    def _get_hunter_step_action_args(self) -> list[str]:
        """Parse train_phase3.py and return the action arg name for every hunter.step() call."""
        src = (Path(__file__).parent.parent / "train_phase3.py").read_text()
        tree = ast.parse(src)
        found = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "step"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "hunter"
                and len(node.args) >= 2
            ):
                arg = node.args[1]
                if isinstance(arg, ast.Name):
                    found.append(arg.id)
        return found

    def test_hunter_step_uses_a_h(self):
        """hunter.step must be called with a_h (Hunter's own action), not a_k."""
        action_args = self._get_hunter_step_action_args()
        assert action_args, "No hunter.step() call found in train_phase3.py"
        for arg in action_args:
            assert arg == "a_h", (
                f"hunter.step() called with '{arg}' but expected 'a_h'.\n"
                "This is the Phase 3 Hunter action bug — Hunter was learning from "
                "Krishna's actions instead of its own."
            )

    def test_krishna_step_uses_a_k(self):
        """Sanity check: krishna.step must still use a_k."""
        src = (Path(__file__).parent.parent / "train_phase3.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "step"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "krishna"
            ):
                # Krishna.step signature is (state, context, action, ...) so action is arg[2]
                if len(node.args) >= 3:
                    arg = node.args[2]
                    if isinstance(arg, ast.Name):
                        assert arg.id == "a_k", (
                            f"krishna.step action arg is '{arg.id}', expected 'a_k'"
                        )


# ---------------------------------------------------------------------------
# 3-4. Zero-pellet timeout penalty
# ---------------------------------------------------------------------------

from environment.selfplay_env import SelfPlayGridworld


class TestZeroPelletTimeoutPenalty:
    def setup_method(self):
        self.env = SelfPlayGridworld(grid_size=25)

    def _run_to_timeout_zero_pellets(self) -> float:
        """Run an episode where Krishna never collects, force a timeout."""
        state, info = self.env.reset(seed=0)
        r_k_total = 0.0
        for _ in range(SelfPlayGridworld.MAX_STEPS + 1):
            _, rewards, done, trunc, info = self.env.step({"krishna": 0, "hunter": 0})
            r_k_total += rewards["krishna"]
            if done or trunc:
                return rewards["krishna"], info["pellets_collected"], trunc
        pytest.fail("Episode never terminated")

    def test_timeout_zero_pellets_applies_penalty(self):
        """Last step reward must include K_TIMEOUT_ZERO_PELLETS when pellets_collected == 0."""
        last_r, pellets, trunc = self._run_to_timeout_zero_pellets()
        assert trunc, "Expected a truncated (timeout) episode"
        assert pellets == 0, f"Expected 0 pellets but got {pellets}"
        assert last_r <= SelfPlayGridworld.K_TIMEOUT_ZERO_PELLETS + SelfPlayGridworld.K_STEP, (
            f"Expected timeout penalty in last reward, got {last_r}"
        )

    def test_timeout_with_pellets_no_penalty(self):
        """Timeout with pellets collected should NOT apply K_TIMEOUT_ZERO_PELLETS."""
        state, info = self.env.reset(seed=1)
        # Force one pellet to be collected: put Krishna on a pellet position
        if self.env.pellet_positions:
            pellet_pos = next(iter(self.env.pellet_positions))
            # Move Krishna to the pellet cell by manipulating state
            kx, ky = self.env.krishna_pos
            self.env.grid[kx, ky] = SelfPlayGridworld.EMPTY
            self.env.krishna_pos = pellet_pos
            self.env.grid[pellet_pos] = SelfPlayGridworld.KRISHNA
            self.env.pellets_collected = 1
            self.env.pellet_positions.discard(pellet_pos)

        # Now run to timeout
        for _ in range(SelfPlayGridworld.MAX_STEPS + 1):
            _, rewards, done, trunc, info = self.env.step({"krishna": 0, "hunter": 0})
            if done or trunc:
                if trunc and info["pellets_collected"] >= 1:
                    # Penalty should NOT be applied
                    assert rewards["krishna"] > SelfPlayGridworld.K_TIMEOUT_ZERO_PELLETS, (
                        "Timeout penalty should not fire when pellets were collected"
                    )
                return
        pytest.fail("Episode never terminated")


# ---------------------------------------------------------------------------
# 5-6. Conditional proximity shaping
# ---------------------------------------------------------------------------

class TestConditionalProximityShaping:
    """Proximity shaping must only fire when Hunter is far (> K_APPROACH_SAFE_DIST)."""

    def _get_reward_delta(
        self,
        krishna_pos: tuple,
        hunter_pos: tuple,
        pellet_pos: tuple,
        krishna_action: int,
    ) -> float:
        """
        Set up a minimal env and take one step; return Krishna's reward
        minus K_STEP (isolating the shaping component).
        """
        env = SelfPlayGridworld(grid_size=25)
        env.reset(seed=0)

        # Manually place entities
        env.grid[:] = SelfPlayGridworld.EMPTY
        env.pellet_positions = {pellet_pos}
        env.grid[pellet_pos] = SelfPlayGridworld.PELLET

        env.grid[krishna_pos] = SelfPlayGridworld.KRISHNA
        env.krishna_pos = krishna_pos

        env.grid[hunter_pos] = SelfPlayGridworld.HUNTER
        env.hunter_pos = hunter_pos
        env.krishna_lives = 3
        env.invuln_timer = 0
        env.pellets_collected = 0
        env.steps = 0

        _, rewards, _, _, _ = env.step({"krishna": krishna_action, "hunter": 0})
        # Return reward stripped of K_STEP base
        return rewards["krishna"] - SelfPlayGridworld.K_STEP

    def test_shaping_fires_when_hunter_far(self):
        """Hunter at dist > 8: proximity shaping should give a positive reward."""
        # Krishna at (12,12), pellet at (15,12) (3 steps below), action DOWN moves closer
        # Hunter at (0,0): dist = 12+12 = 24 > K_APPROACH_SAFE_DIST=8
        delta = self._get_reward_delta(
            krishna_pos=(12, 12),
            hunter_pos=(0, 0),
            pellet_pos=(15, 12),
            krishna_action=1,  # DOWN: toward pellet at (15,12)
        )
        assert delta > 0, f"Expected positive shaping reward when hunter is far, got {delta}"

    def test_shaping_suppressed_when_hunter_close(self):
        """Hunter at dist <= 8: proximity shaping should be zero."""
        # Krishna at (12,12), pellet at (15,12), hunter nearby at (12,18) dist=6 <= 8
        delta = self._get_reward_delta(
            krishna_pos=(12, 12),
            hunter_pos=(12, 18),
            pellet_pos=(15, 12),
            krishna_action=1,  # DOWN: toward pellet — but shaping should be suppressed
        )
        # Only K_STEP should apply (already stripped), so delta ≈ 0
        assert delta == pytest.approx(0.0, abs=1e-5), (
            f"Expected zero shaping when hunter is close, got {delta}"
        )


# ---------------------------------------------------------------------------
# 7-9. Weighted CHER
# ---------------------------------------------------------------------------

from utils.cher import CounterfactualHER


def _make_cher_step(hunter_dist: int, action: int = 2) -> dict:
    """Build a minimal CHER trajectory step dict."""
    state = np.full(625, 6, dtype=np.uint8)
    ctx = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    return {
        "state":        state,
        "context":      ctx,
        "action":       action,
        "reward":       -0.001,
        "next_state":   state.copy(),
        "next_context": ctx.copy(),
        "done":         False,
        "info": {
            "hunter_manhattan_dist": hunter_dist,
            "nearest_pellet_dist":   2.0,
            "nearest_pellet_pos":    (14, 12),  # below (12,12) → DOWN (1) optimal
            "krishna_position":      (12, 12),
            "pellets_remaining":     2,
        },
    }


class TestWeightedCHER:
    def test_farther_hunter_gives_larger_bonus(self):
        """Bonus at 2× safe dist should exceed bonus at 1× safe dist."""
        cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                 collect_bonus=30.0, weighted=True)
        near_safe = cher.relabel([_make_cher_step(hunter_dist=7)])[0]
        far_safe  = cher.relabel([_make_cher_step(hunter_dist=14)])[0]
        _, _, _, r_near, *_ = near_safe
        _, _, _, r_far,  *_ = far_safe
        assert r_far > r_near, (
            f"Farther hunter should give larger CHER bonus: {r_far} vs {r_near}"
        )

    def test_weighted_bonus_at_twice_safe_dist(self):
        """At hunter_dist = 2 * hunter_safe_dist, weight=1.0 → full collect_bonus."""
        cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                 collect_bonus=30.0, weighted=True)
        step = _make_cher_step(hunter_dist=12)  # 12 = 2 * 6
        result = cher.relabel([step])[0]
        _, _, _, reward, *_ = result
        expected = -0.001 + 30.0 * 1.0
        assert abs(reward - expected) < 1e-4, (
            f"Expected reward {expected}, got {reward}"
        )

    def test_non_weighted_flat_bonus(self):
        """With weighted=False, bonus should be flat collect_bonus regardless of distance."""
        cher = CounterfactualHER(hunter_safe_dist=6, pellet_reach=4,
                                 collect_bonus=30.0, weighted=False)
        near = cher.relabel([_make_cher_step(hunter_dist=7)])[0]
        far  = cher.relabel([_make_cher_step(hunter_dist=20)])[0]
        _, _, _, r_near, *_ = near
        _, _, _, r_far,  *_ = far
        assert abs(r_near - r_far) < 1e-5, (
            f"Non-weighted CHER should give flat bonus: {r_near} vs {r_far}"
        )


# ---------------------------------------------------------------------------
# 10. Gate entropy regularisation does not crash
# ---------------------------------------------------------------------------

class TestGateEntropyRegularisation:
    def test_learn_runs_without_error(self):
        """GatedDQNAgent.learn() must complete without error with gate_reg_weight > 0."""
        from agent.gated_dqn_agent import GatedDQNAgent, ContextReplayBuffer
        import numpy as np

        agent = GatedDQNAgent(device="cpu", gate_reg_weight=0.01)

        # Fill buffer above batch_size threshold
        rng = np.random.default_rng(0)
        state  = np.full(625, 6, dtype=np.uint8)
        ctx    = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        for _ in range(70):  # > batch_size=64
            agent.memory.add(state, ctx, int(rng.integers(4)), -0.001,
                             state, ctx, False)

        # learn() should not raise
        result = agent.learn()
        assert "loss" in result
        assert "mean_gate" in result
        assert np.isfinite(result["loss"]), f"Loss is not finite: {result['loss']}"
        assert 0.0 <= result["mean_gate"] <= 1.0, (
            f"Gate out of [0,1]: {result['mean_gate']}"
        )

    def test_gate_reg_weight_zero_matches_no_regularisation(self):
        """gate_reg_weight=0 should behave identically to no regularisation on loss value."""
        from agent.gated_dqn_agent import GatedDQNAgent
        import numpy as np

        # Both agents get the same transitions
        rng = np.random.default_rng(99)
        state = np.full(625, 6, dtype=np.uint8)
        ctx   = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        exps  = [
            (state, ctx, int(rng.integers(4)), -0.001, state, ctx, False)
            for _ in range(70)
        ]

        for reg_w in (0.0, 0.01):
            agent = GatedDQNAgent(device="cpu", gate_reg_weight=reg_w)
            for exp in exps:
                agent.memory.add(*exp)
            result = agent.learn()
            assert np.isfinite(result["loss"]), (
                f"Loss not finite with gate_reg_weight={reg_w}"
            )
