"""Scientific controls for the bounded A-only comparison; no main-study run."""

import copy
import csv
import gzip
import hashlib
import io
import json

import numpy as np
import pytest
import torch

import q6.competence as competence
from q6.competence import (CONDITIONS, FIXED_TASKS, TRAIN_FIELDS, TaskSampler, Trainer,
                           canonical_task, evaluate_episode, evaluation_draws, run_study, summarize)
from q6.learning import DQN
from q6.optimal import VisibleOptimalQ
from q6.world import CollectionWorld, WorldConfig


def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_same(left, right)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_task_support_balance_and_stream_rng_ownership():
    fixed = TaskSampler("fixed_16", 2)
    for _ in range(5):
        assert set(fixed.next_seed() for _ in range(16)) == set(FIXED_TASKS)
    stream, reference = TaskSampler("stream", 2), TaskSampler("stream", 2)
    assert [stream.next_seed() for _ in range(16)] == list(FIXED_TASKS)
    assert [reference.next_seed() for _ in range(16)] == list(FIXED_TASKS)
    np.random.seed(87)
    np.random.random(100)
    draws = [stream.next_seed() for _ in range(1000)]
    assert draws == [reference.next_seed() for _ in range(1000)]
    assert all(0 <= seed < 900000 and seed not in FIXED_TASKS for seed in draws)
    assert {TaskSampler("fixed_1", seed).next_seed() for seed in range(5)} == {200000}


def test_canonical_task_ignores_rng_identity_but_tracks_physical_state():
    env = CollectionWorld()
    env.reset(seed=200000)
    clone = env.clone()
    clone.rng = np.random.default_rng(99)
    assert canonical_task(env) == canonical_task(clone)
    clone.position = tuple(np.argwhere(env.pellets)[0])
    assert canonical_task(env) != canonical_task(clone)


def test_evaluation_checkpoint_preserves_exact_future_training():
    torch.set_num_threads(1)
    agents = [DQN(92, seed=7, capacity=512) for _ in range(2)]
    buffers = [io.StringIO(), io.StringIO()]
    trainers = [Trainer(agent, "fixed_16", 7, 400,
                        csv.DictWriter(buffer, fieldnames=TRAIN_FIELDS))
                for agent, buffer in zip(agents, buffers)]
    trainers[0].advance_to(400, float("inf"))
    trainers[1].advance_to(123, float("inf"))
    before = copy.deepcopy(agents[1].state_dict())
    oracle = VisibleOptimalQ(WorldConfig())
    for mode in ("greedy", "epsilon_0_1"):
        row, trajectory = evaluate_episode(agents[1], WorldConfig(), oracle, condition="fixed_16",
            seed=7, checkpoint=123, mode=mode, panel="heldout", map_seed=920000)
        repeat, other = evaluate_episode(agents[1], WorldConfig(), oracle, condition="stream",
            seed=7, checkpoint=400, mode=mode, panel="heldout", map_seed=920000)
        assert trajectory["steps"] == other["steps"]
        assert row["success"] == repeat["success"]
    assert_same(before, agents[1].state_dict())
    trainers[1].advance_to(400, float("inf"))
    assert agents[1].updates > 0
    assert_same(agents[0].state_dict(), agents[1].state_dict())
    assert_same(trainers[0].env.state_dict(), trainers[1].env.state_dict())
    assert_same(trainers[0].current, trainers[1].current)
    assert_same(trainers[0].exposure, trainers[1].exposure)
    assert_same(trainers[0].sampler.rng.bit_generator.state, trainers[1].sampler.rng.bit_generator.state)
    assert buffers[0].getvalue() == buffers[1].getvalue()
    a = evaluation_draws(7, 920000, 0, 32)
    b = evaluation_draws(7, 920000, 1, 32)
    assert not np.array_equal(a[0], b[0])


def test_unwinnable_ties_do_not_inflate_optimality_and_noops_are_not_revisits():
    config = WorldConfig(horizon=1)
    agent = DQN(92)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
    # This preselected task's shortest path takes two moves; one tick is insufficient.
    row, _ = evaluate_episode(agent, config, VisibleOptimalQ(config), condition="fixed_1", seed=0,
        checkpoint=0, mode="greedy", panel="probe", map_seed=200000)
    assert row["winnable_steps"] == row["optimal_winnable_actions"] == 0
    assert summarize([row])["optimal_action_rate"] is None
    assert row["moved_revisits"] == 0
    assert row["noop_steps"] + row["moving_steps"] == row["steps"]


def test_small_real_run_preserves_artifacts_counts_and_ineligible_gates(tmp_path):
    protocol = tmp_path / "declared.md"
    protocol.write_text("Test-only predeclared protocol; not main-study evidence.\n")
    output = tmp_path / "smoke"
    result = run_study(output, protocol, seeds=[0], phase_steps=16, eval_episodes=1, max_seconds=60,
                       checkpoints=[0, 3, 16], smoke=True)
    assert result["run"]["status"] == "complete"
    assert result["run"]["train_steps"] == 48
    assert result["gates"]["eligible"] is False
    assert not any(result["gates"]["heldout"].values())
    assert not any(result["gates"]["repeated_support"].values())
    with gzip.open(output / "training.csv.gz", "rt") as handle:
        training = list(csv.DictReader(handle))
    assert sum(int(r["steps"]) for r in training) == 48
    for condition in CONDITIONS:
        rows = [r for r in training if r["condition"] == condition["id"]]
        assert sum(int(r["steps"]) for r in rows) == 16
        assert min(int(r["start_step"]) for r in rows) == 0
        assert max(int(r["end_step"]) for r in rows) == 16
    assert len(result["aggregate"]) == 3 * 3 * 2 * 2
    assert len(result["references"]) == 3 * 2 * 2
    assert all(t["map_seed"] == (200000 if t["panel"] == "probe" else 920000)
               for t in result["trajectories"])
    for source, digest in result["protocol"]["source_sha256"].items():
        assert hashlib.sha256((output / "source" / source).read_bytes()).hexdigest() == digest
    assert (output / "protocol.md").read_bytes() == protocol.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["algorithm"] == "sha256"
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
    assert "--checkpoints 0,3,16" in (output / "command.txt").read_text()
    assert {r["transitions"] for r in result["task_exposure_summary"]} == {16}


def test_partial_checkpoint_cannot_enter_summary_or_pass_gate(tmp_path, monkeypatch):
    protocol = tmp_path / "declared.md"
    protocol.write_text("Test budget interruption.\n")
    original = competence.evaluate_episode
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise competence.BudgetReached("test admission cap")
        return original(*args, **kwargs)

    monkeypatch.setattr(competence, "evaluate_episode", interrupt)
    result = run_study(tmp_path / "partial", protocol, seeds=[0], phase_steps=4, eval_episodes=1)
    assert result["run"]["status"] == "incomplete_admission_cap"
    assert result["run"]["train_steps"] == 0
    assert result["aggregate"] == []
    assert result["gates"]["eligible"] is False
    with (tmp_path / "partial/evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["checkpoint_complete"] == "0"
    assert result["run"]["progress"] == [{"condition": "fixed_1", "seed": 0,
        "checkpoint": 0, "evaluation_complete": False}]
