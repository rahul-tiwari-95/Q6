"""Tests for SelfPlayGridworld."""

from __future__ import annotations

import numpy as np
import pytest

from environment.selfplay_env import SelfPlayGridworld


@pytest.fixture
def env():
    e = SelfPlayGridworld()
    e.reset(seed=0)
    return e


class TestResetAndShape:
    def test_state_shape_and_dtype(self, env):
        state, info = env.reset(seed=0)
        assert state.shape == (env.grid_size * env.grid_size,)
        assert state.dtype == np.int32

    def test_reset_is_deterministic_with_seed(self):
        e1 = SelfPlayGridworld()
        s1, _ = e1.reset(seed=123)
        e2 = SelfPlayGridworld()
        s2, _ = e2.reset(seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_reset_places_both_agents(self, env):
        assert env.krishna_pos is not None
        assert env.hunter_pos is not None
        assert env.krishna_pos != env.hunter_pos

    def test_reset_places_target_pellets(self, env):
        assert len(env.pellet_positions) == env.TARGET_PELLETS

    def test_no_greedy_or_patrol_cells(self, env):
        # Only WALL/PELLET/KRISHNA/HUNTER/EMPTY should appear
        unique = set(int(v) for v in np.unique(env.grid))
        assert unique.issubset({env.WALL, env.PELLET, env.KRISHNA,
                                env.HUNTER, env.EMPTY})


class TestStepAPI:
    def test_step_returns_5_tuple(self, env):
        out = env.step({"krishna": 0, "hunter": 0})
        assert len(out) == 5
        state, rewards, terminated, truncated, info = out
        assert state.shape == (env.grid_size * env.grid_size,)
        assert isinstance(rewards, dict)
        assert set(rewards.keys()) == {"krishna", "hunter"}
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_missing_agent_key_raises(self, env):
        with pytest.raises(ValueError):
            env.step({"krishna": 0})

    def test_invalid_action_raises(self, env):
        with pytest.raises(ValueError):
            env.step({"krishna": 99, "hunter": 0})

    def test_step_before_reset_raises(self):
        e = SelfPlayGridworld()
        with pytest.raises(RuntimeError):
            e.step({"krishna": 0, "hunter": 0})


class TestRewardSignals:
    def test_step_baseline_reward_small_negative(self):
        """Find a seed where Krishna's first action doesn't hit wall/pellet/hunter."""
        for seed in range(40):
            e = SelfPlayGridworld()
            e.reset(seed=seed)
            for a in range(4):
                # Sandbox a copy by snapshotting & restoring grid? Simpler: just try.
                snap_pos = e.krishna_pos
                _, rewards, _, _, _ = e.step({"krishna": a, "hunter": a})
                if abs(rewards["krishna"] - e.K_STEP) < 1e-9:
                    return  # clean step found
                # reset for next attempt
                e.reset(seed=seed)
        pytest.skip("Couldn't find a clean baseline step across many seeds.")

    def test_krishna_caught_negative_reward(self):
        """Hunter steps onto Krishna's cell -> krishna gets K_CAUGHT, hunter gets H_CATCH."""
        e = SelfPlayGridworld()
        e.reset(seed=7)
        # Forcibly place hunter adjacent to krishna with no wall between
        kx, ky = e.krishna_pos
        # Pick an adjacent empty cell for hunter
        for dx, dy, action in [(-1, 0, 1), (1, 0, 0), (0, -1, 3), (0, 1, 2)]:
            hx, hy = kx + dx, ky + dy
            if (0 <= hx < e.grid_size and 0 <= hy < e.grid_size
                    and e.grid[hx, hy] in (e.EMPTY, e.PELLET)):
                # Move hunter to (hx, hy); 'action' is the direction hunter should
                # step to reach (kx, ky) from (hx, hy)
                e.grid[e.hunter_pos] = e.EMPTY
                e.hunter_pos = (hx, hy)
                e.grid[hx, hy] = e.HUNTER
                e.invuln_timer = 0
                _, rewards, _, _, info = e.step({"krishna": -0 if False else 0, "hunter": action})
                # Krishna may have stepped (action 0=Up could move); only assert hunter outcome
                # Actually we use a no-op-ish krishna action (0=Up) — if it moves, hunter still catches old/new cell?
                # Hunter moves toward kx,ky, but krishna may have moved. To make this deterministic,
                # we re-do the test with krishna trying to move into a wall (action that keeps it put).
                break
        else:
            pytest.skip("No adjacent empty cell for hunter; try another seed.")

        # Re-do cleanly: krishna tries to move into a wall (stays put), hunter steps onto krishna
        e.reset(seed=7)
        kx, ky = e.krishna_pos
        # Find a wall-adjacent direction so Krishna's move is a no-op (no penalty)
        # Use action that points to grid boundary
        # Build a controlled scenario instead
        e.grid[:] = e.EMPTY
        e.grid[0, 0] = e.WALL  # wall present somewhere
        e.krishna_pos = (10, 10)
        e.hunter_pos = (10, 11)
        e.grid[10, 10] = e.KRISHNA
        e.grid[10, 11] = e.HUNTER
        e.pellet_positions = set()
        e.invuln_timer = 0
        e.krishna_lives = 3
        # Krishna tries action 0 (Up) -> moves to (9,10). Hunter action 2 (Left) -> (10,10).
        # After krishna moves, krishna is at (9,10); hunter goes to (10,10) — empty now, no catch.
        # Instead: krishna chooses 1 (Down) -> (11,10). Hunter chooses 0 (Up) ... still misses.
        # Easiest: make krishna stay (try to walk into wall). Put wall at (9,10).
        e.grid[9, 10] = e.WALL
        # Krishna action 0 (Up): tries (9,10), wall -> stays (10,10), wall penalty applies.
        # Hunter action 2 (Left): (10,11)->(10,10). Catch!
        _, rewards, terminated, _, info = e.step({"krishna": 0, "hunter": 2})
        assert info["caught_this_step"] is True
        # Hunter gets H_CATCH plus the per-step cost
        assert rewards["hunter"] == pytest.approx(e.H_CATCH + e.H_STEP)
        # Krishna gets caught penalty + wall penalty + step cost
        assert rewards["krishna"] == pytest.approx(e.K_CAUGHT + e.K_WALL + e.K_STEP)
        assert info["times_caught"] == 1

    def test_krishna_pellet_reward(self):
        e = SelfPlayGridworld()
        e.reset(seed=0)
        # Put a pellet directly down from krishna
        e.grid[:] = e.EMPTY
        e.krishna_pos = (5, 5)
        e.hunter_pos = (20, 20)
        e.grid[5, 5] = e.KRISHNA
        e.grid[20, 20] = e.HUNTER
        e.grid[6, 5] = e.PELLET
        e.pellet_positions = {(6, 5)}
        e.pellets_collected = 0
        e.invuln_timer = 0
        # Krishna action 1 (Down) collects pellet; Hunter action 0 (Up).
        # On pellet collection the proximity shaping is NOT added (else branch skipped).
        # Expected: K_STEP + K_PELLET
        _, rewards, terminated, _, info = e.step({"krishna": 1, "hunter": 0})
        assert rewards["krishna"] == pytest.approx(e.K_STEP + e.K_PELLET)
        assert info["pellets_collected"] == 1


class TestTermination:
    def test_krishna_wins_on_all_pellets(self):
        e = SelfPlayGridworld()
        e.reset(seed=0)
        e.grid[:] = e.EMPTY
        e.krishna_pos = (5, 5)
        e.hunter_pos = (20, 20)
        e.grid[5, 5] = e.KRISHNA
        e.grid[20, 20] = e.HUNTER
        e.grid[6, 5] = e.PELLET
        e.pellet_positions = {(6, 5)}
        e.pellets_collected = e.TARGET_PELLETS - 1
        e.invuln_timer = 0
        _, rewards, terminated, _, info = e.step({"krishna": 1, "hunter": 0})
        assert terminated is True
        assert info["winner"] == "krishna"
        assert rewards["krishna"] >= e.K_WIN

    def test_hunter_wins_on_krishna_zero_lives(self):
        e = SelfPlayGridworld()
        e.reset(seed=0)
        e.grid[:] = e.EMPTY
        e.krishna_pos = (10, 10)
        e.hunter_pos = (10, 11)
        e.grid[10, 10] = e.KRISHNA
        e.grid[10, 11] = e.HUNTER
        e.pellet_positions = set()
        e.invuln_timer = 0
        e.krishna_lives = 1
        e.grid[9, 10] = e.WALL  # ensure krishna's Up action stays put
        _, rewards, terminated, _, info = e.step({"krishna": 0, "hunter": 2})
        assert terminated is True
        assert info["winner"] == "hunter"
        assert rewards["hunter"] >= e.H_WIN

    def test_timeout_truncates(self):
        e = SelfPlayGridworld()
        e.reset(seed=0)
        e.steps = e.MAX_STEPS - 1
        _, _, terminated, truncated, info = e.step({"krishna": 0, "hunter": 0})
        assert truncated is True
        assert terminated is False
        assert info["winner"] == "timeout"
