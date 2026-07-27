"""
Tests for agent/frozen_ppo_agent.py.
"""

import numpy as np

from agent.frozen_ppo_agent import FrozenPPOAgent
from agent.ppo_agent import PPOAgent
from config import ACTION_SIZE, GRID_SIZE


def _random_flat_state(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 7, size=GRID_SIZE * GRID_SIZE, dtype=np.int32)


class TestFrozenPPOAgent:
    def test_load_matches_source_network_greedy_action(self, tmp_path):
        agent = PPOAgent(device="cpu")
        path = tmp_path / "krishna.pth"
        agent.save(str(path))

        frozen = FrozenPPOAgent.load(str(path), device="cpu")
        state = _random_flat_state(0)

        assert frozen.act(state) == agent.act_greedy(state)

    def test_act_is_deterministic(self, tmp_path):
        agent = PPOAgent(device="cpu")
        path = tmp_path / "krishna.pth"
        agent.save(str(path))
        frozen = FrozenPPOAgent.load(str(path), device="cpu")

        state = _random_flat_state(1)
        actions = {frozen.act(state) for _ in range(10)}
        assert len(actions) == 1  # greedy => always the same action for the same state

    def test_action_in_range(self, tmp_path):
        agent = PPOAgent(device="cpu")
        path = tmp_path / "hunter.pth"
        agent.save(str(path))
        frozen = FrozenPPOAgent.load(str(path), device="cpu")

        for i in range(10):
            a = frozen.act(_random_flat_state(i))
            assert 0 <= a < ACTION_SIZE

    def test_q_values_shape(self, tmp_path):
        agent = PPOAgent(device="cpu")
        path = tmp_path / "krishna.pth"
        agent.save(str(path))
        frozen = FrozenPPOAgent.load(str(path), device="cpu")

        qs = frozen.q_values(_random_flat_state(0))
        assert qs.shape == (ACTION_SIZE,)
