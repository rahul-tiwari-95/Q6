"""
Tests for agent/ppo_agent.py — RolloutBuffer, GAE, and the clipped-surrogate
PPO update used by v8's Independent PPO self-play port.

The GAE tests hand-compute expected advantages/returns for a tiny 3-step
rollout so a silent sign/indexing bug (the same class of bug that produced
the confounded v7 result — see Q6.md section 4) would be caught immediately
rather than discovered after a multi-thousand-episode run.
"""

import numpy as np
import pytest
import torch

from agent.ppo_agent import PPOAgent, RolloutBuffer
from config import ACTION_SIZE, GRID_SIZE


def _random_flat_state(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 7, size=GRID_SIZE * GRID_SIZE, dtype=np.int32)


def _fill_buffer(agent: PPOAgent, rewards, values, dones, actions=None, logprobs=None):
    """Directly populate the buffer with known scalars, bypassing store()'s
    encoding path — used only to hand-verify _compute_gae in isolation."""
    n = len(rewards)
    actions = actions or [0] * n
    logprobs = logprobs or [0.0] * n
    for i in range(n):
        agent.buffer.add(
            state=_random_flat_state(i),
            action=actions[i],
            logprob=logprobs[i],
            reward=rewards[i],
            done=dones[i],
            value=values[i],
        )


# ----------------------------- rollout buffer -----------------------------

class TestRolloutBuffer:
    def test_grows_and_reports_full(self):
        buf = RolloutBuffer(rollout_len=3)
        assert not buf.full()
        for i in range(3):
            buf.add(_random_flat_state(i), 0, 0.0, 1.0, False, 0.5)
        assert buf.full()
        assert len(buf) == 3

    def test_reset_clears(self):
        buf = RolloutBuffer(rollout_len=3)
        for i in range(3):
            buf.add(_random_flat_state(i), 0, 0.0, 1.0, False, 0.5)
        buf.reset()
        assert len(buf) == 0
        assert not buf.full()


# ----------------------------- GAE (hand-computed) -----------------------------

class TestGAE:
    """gamma=0.5, gae_lambda=0.5 chosen so the recursion is exact-by-hand
    with simple fractions — not meant to resemble real training hyperparams."""

    def _agent(self) -> PPOAgent:
        return PPOAgent(device="cpu", gamma=0.5, gae_lambda=0.5, rollout_len=3)

    def test_no_episode_boundary(self):
        agent = self._agent()
        _fill_buffer(agent, rewards=[1.0, 2.0, 3.0], values=[0.5, 0.5, 0.5],
                     dones=[False, False, False])

        advantages, returns = agent._compute_gae(next_value=1.0, next_done=False)

        expected_adv = np.array([1.375, 2.5, 3.0], dtype=np.float32)
        expected_ret = np.array([1.875, 3.0, 3.5], dtype=np.float32)
        np.testing.assert_allclose(advantages, expected_adv, atol=1e-5)
        np.testing.assert_allclose(returns, expected_ret, atol=1e-5)

    def test_episode_boundary_zeroes_bootstrap(self):
        # Same rewards/values as above, but dones[1]=True: the state at t=1
        # is a fresh reset, so t=0's advantage must NOT bootstrap across it.
        agent = self._agent()
        _fill_buffer(agent, rewards=[1.0, 2.0, 3.0], values=[0.5, 0.5, 0.5],
                     dones=[False, True, False])

        advantages, returns = agent._compute_gae(next_value=1.0, next_done=False)

        # advantages[1] and [2] are unaffected — the boundary sits before them.
        expected_adv = np.array([0.5, 2.5, 3.0], dtype=np.float32)
        expected_ret = np.array([1.0, 3.0, 3.5], dtype=np.float32)
        np.testing.assert_allclose(advantages, expected_adv, atol=1e-5)
        np.testing.assert_allclose(returns, expected_ret, atol=1e-5)

    def test_next_done_true_zeroes_final_bootstrap(self):
        agent = self._agent()
        _fill_buffer(agent, rewards=[1.0, 2.0, 3.0], values=[0.5, 0.5, 0.5],
                     dones=[False, False, False])

        advantages, _returns = agent._compute_gae(next_value=99.0, next_done=True)
        # next_value must be irrelevant when next_done=True.
        advantages_ignored_value, _ = agent._compute_gae(next_value=-99.0, next_done=True)
        np.testing.assert_allclose(advantages, advantages_ignored_value, atol=1e-5)

    def test_varying_values_with_boundary(self):
        # Regression guard flagged in the v8 PPO review: the other cases here
        # all use a CONSTANT values array, so a `values[t]` vs `values[t+1]`
        # indexing swap in _compute_gae's t<n-1 branch would silently pass
        # them (values[t] == values[t+1] everywhere). This case uses a
        # varying values array combined with a mid-rollout done boundary,
        # so that specific bug class produces a different, failing result.
        agent = self._agent()
        _fill_buffer(agent, rewards=[1.0, 2.0, 3.0], values=[0.2, 0.5, 0.9],
                     dones=[False, True, False])

        advantages, returns = agent._compute_gae(next_value=1.0, next_done=False)

        expected_adv = np.array([0.8, 2.6, 2.6], dtype=np.float32)
        expected_ret = np.array([1.0, 3.1, 3.5], dtype=np.float32)
        np.testing.assert_allclose(advantages, expected_adv, atol=1e-5)
        np.testing.assert_allclose(returns, expected_ret, atol=1e-5)


# ----------------------------- act / update integration -----------------------------

class TestActAndUpdate:
    def test_act_returns_valid_types_and_ranges(self):
        agent = PPOAgent(device="cpu", rollout_len=8)
        action, logprob, value = agent.act(_random_flat_state(0))
        assert 0 <= action < ACTION_SIZE
        assert np.isfinite(logprob)
        assert np.isfinite(value)

    def test_act_greedy_deterministic(self):
        agent = PPOAgent(device="cpu")
        state = _random_flat_state(0)
        a1 = agent.act_greedy(state)
        a2 = agent.act_greedy(state)
        assert a1 == a2

    def test_full_rollout_update_produces_finite_loss_and_resets_buffer(self):
        rollout_len = 16
        agent = PPOAgent(device="cpu", rollout_len=rollout_len, num_minibatches=2, update_epochs=2)

        state = _random_flat_state(0)
        prev_done = True
        for i in range(rollout_len):
            action, logprob, value = agent.act(state)
            reward = float(np.random.default_rng(i).normal())
            done = (i == 7)  # inject one synthetic episode boundary
            agent.store(state, action, logprob, reward, prev_done, value)
            prev_done = done
            state = _random_flat_state(i + 1)

        assert agent.ready_to_update()
        next_value = agent.get_value(state)
        metrics = agent.update(next_value=next_value, next_done=prev_done)

        for v in metrics.values():
            assert np.isfinite(v), f"non-finite metric: {metrics}"
        assert len(agent.buffer) == 0  # update() must reset the buffer
        assert agent.update_step == 1

    def test_update_on_empty_buffer_raises(self):
        agent = PPOAgent(device="cpu", rollout_len=4)
        with pytest.raises(RuntimeError):
            agent.update(next_value=0.0, next_done=False)


# ----------------------------- persistence -----------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        agent = PPOAgent(device="cpu")
        path = tmp_path / "agent.pth"
        agent.save(str(path))

        agent2 = PPOAgent(device="cpu")
        agent2.load(str(path))

        state = _random_flat_state(0)
        assert agent.act_greedy(state) == agent2.act_greedy(state)
