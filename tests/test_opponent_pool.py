"""Tests for OpponentPool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agent.dqn_v2_agent import DQNv2Agent
from agent.opponent_pool import OpponentPool


@pytest.fixture
def agent():
    return DQNv2Agent(device="cpu")


class TestAddAndSample:
    def test_empty_pool_sample_returns_none(self, tmp_path):
        pool = OpponentPool(tmp_path / "pool")
        assert pool.sample(np.random.default_rng(0)) is None
        assert len(pool) == 0

    def test_add_snapshot_writes_file(self, tmp_path, agent):
        pool = OpponentPool(tmp_path / "pool")
        path = pool.add_snapshot(agent, metadata={"episode": 1})
        assert Path(path).exists()
        assert len(pool) == 1

    def test_sample_returns_existing_path(self, tmp_path, agent):
        pool = OpponentPool(tmp_path / "pool")
        pool.add_snapshot(agent, metadata={"episode": 1})
        rng = np.random.default_rng(0)
        for _ in range(5):
            p = pool.sample(rng)
            assert p is not None
            assert Path(p).exists()

    def test_latest_returns_most_recent(self, tmp_path, agent):
        pool = OpponentPool(tmp_path / "pool")
        first = pool.add_snapshot(agent, metadata={"episode": 1})
        second = pool.add_snapshot(agent, metadata={"episode": 2})
        assert pool.latest() == second
        assert pool.latest() != first


class TestEviction:
    def test_fifo_eviction_at_max_size(self, tmp_path, agent):
        pool = OpponentPool(tmp_path / "pool", max_size=3)
        paths = [pool.add_snapshot(agent, metadata={"i": i}) for i in range(5)]
        # Only last 3 kept
        assert len(pool) == 3
        assert not Path(paths[0]).exists()
        assert not Path(paths[1]).exists()
        for p in paths[2:]:
            assert Path(p).exists()


class TestPersistence:
    def test_reload_picks_up_existing_snapshots(self, tmp_path, agent):
        pool = OpponentPool(tmp_path / "pool", max_size=10)
        pool.add_snapshot(agent, metadata={"episode": 1})
        pool.add_snapshot(agent, metadata={"episode": 2})

        pool2 = OpponentPool(tmp_path / "pool", max_size=10)
        assert len(pool2) == 2
        # New snapshots get new ids (no overwrite)
        new_path = pool2.add_snapshot(agent, metadata={"episode": 3})
        assert Path(new_path).exists()
        assert len(pool2) == 3
