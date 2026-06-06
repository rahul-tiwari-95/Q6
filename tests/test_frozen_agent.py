"""Tests for FrozenAgent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.dqn_v2_agent import DQNv2Agent
from agent.frozen_agent import FrozenAgent
from environment.selfplay_env import SelfPlayGridworld


@pytest.fixture
def saved_path(tmp_path):
    agent = DQNv2Agent(device="cpu")
    p = tmp_path / "agent.pth"
    agent.save(str(p))
    return str(p)


def test_frozen_load_and_act(saved_path):
    f = FrozenAgent.load(saved_path, device="cpu")
    env = SelfPlayGridworld()
    state, _ = env.reset(seed=0)
    for _ in range(5):
        a = f.act(state)
        assert 0 <= a < 4
        _, _, done, trunc, _ = env.step({"krishna": 0, "hunter": a})
        if done or trunc:
            break


def test_frozen_q_values_shape(saved_path):
    f = FrozenAgent.load(saved_path, device="cpu")
    env = SelfPlayGridworld()
    state, _ = env.reset(seed=0)
    qs = f.q_values(state)
    assert qs.shape == (4,)
    assert qs.dtype == np.float32


def test_frozen_act_is_deterministic(saved_path):
    f = FrozenAgent.load(saved_path, device="cpu")
    env = SelfPlayGridworld()
    state, _ = env.reset(seed=0)
    actions = [f.act(state) for _ in range(10)]
    assert all(a == actions[0] for a in actions)
