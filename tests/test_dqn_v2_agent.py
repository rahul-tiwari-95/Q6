"""
Tests for agent/dqn_v2_agent.py — Double DQN + CNN + Dueling.
"""

import numpy as np
import pytest
import torch

from config import GRID_SIZE, ACTION_SIZE, EMPTY, KRISHNA, HUNTER
from agent.dqn_v2_agent import DQNv2Agent, FlatReplayBuffer


def _random_flat_state(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 7, size=GRID_SIZE * GRID_SIZE, dtype=np.int32)


# ----------------------------- replay buffer -----------------------------

class TestReplayBuffer:
    def test_grows_then_caps(self):
        buf = FlatReplayBuffer(capacity=5)
        for i in range(8):
            buf.add(_random_flat_state(i), i % 4, float(i), _random_flat_state(i + 100), False)
        assert len(buf) == 5

    def test_sample_shapes_and_dtypes(self):
        buf = FlatReplayBuffer(capacity=50)
        for i in range(40):
            buf.add(_random_flat_state(i), i % 4, float(i), _random_flat_state(i + 100), i % 7 == 0)
        s, a, r, sn, d = buf.sample(16)
        assert s.shape == (16, GRID_SIZE * GRID_SIZE)
        assert sn.shape == (16, GRID_SIZE * GRID_SIZE)
        assert a.shape == (16,) and a.dtype == np.int64
        assert r.shape == (16,) and r.dtype == np.float32
        assert d.shape == (16,) and d.dtype == np.float32
        # State values are still in 0..6
        assert s.min() >= 0 and s.max() <= 6


# ----------------------------- agent basics -----------------------------

class TestAgentBasics:
    def test_act_returns_valid_action_random_branch(self):
        agent = DQNv2Agent(device="cpu")
        agent.epsilon = 1.0  # force random
        for _ in range(20):
            a = agent.act(_random_flat_state(0), training=True)
            assert 0 <= a < ACTION_SIZE

    def test_act_returns_valid_action_greedy_branch(self):
        agent = DQNv2Agent(device="cpu")
        agent.epsilon = 0.0  # force greedy
        a = agent.act(_random_flat_state(0), training=True)
        assert 0 <= a < ACTION_SIZE

    def test_act_deterministic_when_epsilon_zero(self):
        torch.manual_seed(0)
        agent = DQNv2Agent(device="cpu")
        agent.epsilon = 0.0
        s = _random_flat_state(7)
        a1 = agent.act(s, training=True)
        a2 = agent.act(s, training=True)
        a3 = agent.act(s, training=False)
        assert a1 == a2 == a3

    def test_q_values_shape(self):
        agent = DQNv2Agent(device="cpu")
        qs = agent.q_values(_random_flat_state(0))
        assert qs.shape == (ACTION_SIZE,)
        assert np.all(np.isfinite(qs))


# ----------------------------- target init / update -----------------------------

class TestTargetNetwork:
    def test_target_initially_equals_local(self):
        agent = DQNv2Agent(device="cpu")
        for tp, lp in zip(agent.qnetwork_target.parameters(),
                          agent.qnetwork_local.parameters()):
            assert torch.allclose(tp, lp)

    def test_soft_update_moves_target_toward_local(self):
        torch.manual_seed(0)
        agent = DQNv2Agent(device="cpu", batch_size=8, update_every=1)
        # fill buffer
        for i in range(50):
            agent.memory.add(_random_flat_state(i), i % 4, 1.0 if i % 5 == 0 else -0.01,
                              _random_flat_state(i + 100), i % 9 == 0)
        # Snapshot target before
        before = [p.clone() for p in agent.qnetwork_target.parameters()]
        agent.learn()
        after = list(agent.qnetwork_target.parameters())
        # At least one target param should have changed slightly
        any_changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert any_changed, "target network did not update at all"


# ----------------------------- double DQN learn step -----------------------------

class TestLearnStep:
    def test_learn_returns_finite_metrics(self):
        torch.manual_seed(0)
        agent = DQNv2Agent(device="cpu", batch_size=8, update_every=1)
        for i in range(40):
            agent.memory.add(_random_flat_state(i), i % 4, float(i % 7),
                              _random_flat_state(i + 1), i % 9 == 0)
        metrics = agent.learn()
        assert np.isfinite(metrics["loss"])
        assert np.isfinite(metrics["mean_q"])

    def test_loss_decreases_on_repeated_learning(self):
        """Train on a fixed mini-buffer many times — loss should drop a lot."""
        torch.manual_seed(0)
        agent = DQNv2Agent(device="cpu", batch_size=16, update_every=1,
                           learning_rate=1e-3)
        for i in range(64):
            agent.memory.add(_random_flat_state(i), i % 4, float(i % 5),
                              _random_flat_state(i + 1), i % 11 == 0)
        first = agent.learn()["loss"]
        for _ in range(200):
            agent.learn()
        last = agent.learn()["loss"]
        assert last < first * 0.5, f"loss did not decrease: {first=:.4f} {last=:.4f}"

    def test_step_triggers_learning_only_when_buffer_full_enough(self):
        agent = DQNv2Agent(device="cpu", batch_size=64, update_every=4)
        # Below threshold: should never learn
        for i in range(10):
            agent.step(_random_flat_state(i), i % 4, 0.0, _random_flat_state(i + 1), False)
        assert agent.learning_step == 0


# ----------------------------- env integration -----------------------------

class TestEnvIntegration:
    def test_full_episode_no_crash(self):
        from environment.hunter_gridworld import HunterGridworld
        env = HunterGridworld(difficulty_level=4, curriculum_phase=1)
        agent = DQNv2Agent(device="cpu", batch_size=8, update_every=4)
        agent.epsilon = 0.5
        state, _ = env.reset(seed=0)
        for _ in range(50):
            a = agent.act(state, training=True)
            next_state, r, done, trunc, _ = env.step(a)
            agent.step(state, a, r, next_state, done)
            state = next_state
            if done or trunc:
                break

    def test_epsilon_decay(self):
        agent = DQNv2Agent(device="cpu")
        start = agent.epsilon
        for _ in range(10):
            agent.decay_epsilon()
        assert agent.epsilon < start
        # Cannot go below min
        for _ in range(100_000):
            agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_min - 1e-9


# ----------------------------- save / load -----------------------------

class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        torch.manual_seed(0)
        agent1 = DQNv2Agent(device="cpu")
        agent1.epsilon = 0.42
        state = _random_flat_state(3)
        qs1 = agent1.q_values(state)
        path = str(tmp_path / "agent.pth")
        agent1.save(path)

        agent2 = DQNv2Agent(device="cpu")
        agent2.load(path)
        qs2 = agent2.q_values(state)
        np.testing.assert_allclose(qs1, qs2, atol=1e-6)
        assert agent2.epsilon == pytest.approx(0.42)
