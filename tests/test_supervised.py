"""Bounded checks for exact targets, leakage, evaluation isolation and artifacts."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest
import torch

import q6.supervised as supervised
from q6.learning import DQN
from q6.optimal import VisibleOptimalQ
from q6.supervised import (BatchSampler, enumerate_panel, full_state_metrics, layout_key,
                           rollout, run_study, select_layouts, supervised_update)
from q6.world import CollectionWorld, WorldConfig


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            same(left, right)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_enumerated_targets_match_clone_bellman_and_declared_row_order():
    config = WorldConfig(size=3, wall_count=1, horizon=4)
    oracle = VisibleOptimalQ(config)
    selected = select_layouts(config, [700007], 1, 970000)
    data = enumerate_panel(config, selected["train"], oracle)
    env = CollectionWorld(config)
    env.reset(seed=700007)
    positions = np.argwhere(~(env.walls | env.pellets))
    assert len(data["observations"]) == 7 * 4
    np.testing.assert_array_equal(data["positions"], np.repeat(positions, 4, axis=0))
    np.testing.assert_array_equal(data["remaining"], np.tile(np.arange(1, 5), 7))
    for index, observation in enumerate(data["observations"]):
        env.position = tuple(int(v) for v in data["positions"][index])
        env.elapsed = config.horizon - int(data["remaining"][index])
        for action in range(4):
            clone = env.clone()
            nxt, reward, terminated, truncated, _ = clone.step(action)
            expected = reward + (0.0 if terminated or truncated else config.gamma * oracle.q_values(nxt).max())
            assert data["targets"][index, action] == pytest.approx(expected, abs=1e-12)
        assert data["winnable"][index] == oracle.can_finish(observation)
    assert not data["winnable"].all()


def test_layout_split_ignores_spawn_and_keeps_duplicate_training_weight():
    env = CollectionWorld()
    env.reset(seed=700009)
    original = layout_key(env)
    env.position = tuple(int(v) for v in np.argwhere(~(env.walls | env.pellets))[0])
    env.elapsed = 7
    assert layout_key(env) == original
    selected = select_layouts(env.config, [700009, 700009], 2, 700009)
    assert selected["train_duplicate_layouts"] == {original: 2}
    assert selected["collision_skips"][0]["map_seed"] == 700009
    assert selected["collision_skips"][0]["reason"] == "training_layout"
    fresh = [r["layout_hash"] for r in selected["heldout"]]
    assert len(set(fresh)) == 2 and original not in fresh


def test_fixed_target_updates_and_evaluation_keep_target_replay_and_rng_untouched():
    torch.set_num_threads(1)
    config = WorldConfig()
    oracle = VisibleOptimalQ(config)
    data = enumerate_panel(config, select_layouts(config, [700010], 1, 970000)["train"], oracle)
    observations = torch.from_numpy(data["observations"])
    targets = torch.from_numpy(data["targets"].astype(np.float32))
    saved_targets = targets.clone()
    agent = DQN(92, seed=2)
    sampler = BatchSampler(len(observations), 2)
    target_before = copy.deepcopy(agent.target.state_dict())
    learner_rng = copy.deepcopy(agent.rng.bit_generator.state)
    drawn = []
    for _ in range(3):
        indices = sampler.next_batch()
        assert len(set(indices)) == 64
        drawn.append(indices.copy())
        supervised_update(agent, observations, targets, indices)
    assert agent.steps == agent.count == agent.updates == 0
    same(agent.target.state_dict(), target_before)
    same(agent.rng.bit_generator.state, learner_rng)
    before_eval = agent.state_dict()
    sampler_rng = copy.deepcopy(sampler.rng.bit_generator.state)
    full_state_metrics(agent, data, seed=2, checkpoint=3, panel="train")
    for mode in ("greedy", "epsilon_0_1"):
        rollout(agent, config, oracle, seed=2, checkpoint=3, mode=mode, panel="heldout", map_seed=970000)
    same(agent.state_dict(), before_eval)
    same(sampler.rng.bit_generator.state, sampler_rng)
    assert torch.equal(targets, saved_targets)
    flat = np.concatenate(drawn)
    np.testing.assert_array_equal(sampler.counts, np.bincount(flat, minlength=len(observations)))
    assert sampler.digest.hexdigest() == hashlib.sha256(flat.astype("<i8").tobytes()).hexdigest()


def test_all_action_loss_and_winnable_metric_denominators():
    agent = DQN(92)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
    observations = torch.zeros((64, 92))
    targets = torch.tensor([[0., 1., 2., 3.]]).repeat(64, 1)
    # SmoothL1 values [0,.5,1.5,2.5] average across all four labels.
    assert supervised_update(agent, observations, targets, np.arange(64)) == pytest.approx(1.125)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
    exact = np.array([[1., 1.0000005, 0., 0.], [1., 1.01, 0., 0.], [.1, .1, .1, .1]])
    data = {"observations": np.zeros((3, 92), np.float32), "targets": exact,
            "winnable": np.array([True, True, False]), "map_seeds": np.ones(3, np.int64),
            "remaining": np.array([2, 2, 1])}
    rows, _, errors = full_state_metrics(agent, data, seed=0, checkpoint=0, panel="train")
    row = next(r for r in rows if r["time_bucket"] == "all")
    assert row["states"] == 3 and row["action_values"] == 12
    assert row["winnable_states"] == 2 and row["impossible_states"] == 1
    assert row["optimal_action_rate"] == .5  # 1e-6 tolerance; impossible tie excluded.
    assert row["mean_abs_q_error"] == pytest.approx(np.abs(exact).mean())
    assert row["rmse_q_error"] == pytest.approx(np.sqrt(np.square(exact).mean()))
    assert row["q95_abs_q_error"] == pytest.approx(np.quantile(errors, .95))
    assert row["mean_signed_q_bias"] == pytest.approx(-exact.mean())
    assert row["winnable_mean_action_regret"] == pytest.approx((.0000005 + .01) / 2)
    assert row["impossible_action_regret"] == 0


def fake_prior(tmp_path):
    path = tmp_path / "prior"
    (path / "models").mkdir(parents=True)
    agent = DQN(92, seed=0)
    torch.save({"online": agent.online.state_dict(), "observation_size": 92, "parameter_hash": agent.parameter_hash()},
               path / "models/stream_seed0_step120000.pt")
    (path / "task_exposure.json").write_text(json.dumps([{"condition": "stream", "seed": 0, "tasks": [{"map_seed": 800001}]}]))
    (path / "protocol.json").write_text(json.dumps({"source_sha256": {"test_stub": "not_main_study_evidence"}}))
    return path


def test_real_tiny_run_preserves_dense_data_sources_sampling_and_ineligible_gates(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Test-only fixed-target smoke.\n")
    output = tmp_path / "smoke"
    result = run_study(output, protocol, seeds=[0], updates=4, train_maps=1, fresh_maps=1,
        train_seed_start=700010, fresh_seed_start=970000, prior_dir=fake_prior(tmp_path), checkpoints=[0, 2, 4], smoke=True)
    assert result["run"]["status"] == "complete"
    assert result["run"]["train_updates"] == 4 and result["run"]["training_examples"] == 256
    assert result["gates"]["eligible"] is result["gates"]["training_fit"] is result["gates"]["fresh"] is False
    dataset = np.load(output / "dataset.npz")
    metadata = json.loads((output / "dataset_metadata.json").read_text())
    for name in dataset.files:
        assert hashlib.sha256(dataset[name].tobytes()).hexdigest() == metadata["arrays"][name]["sha256"]
    assert metadata["split_check"]["layout_intersections"] == metadata["split_check"]["observation_intersections"] == 0
    predictions = np.load(output / "predictions_seed0.npz")
    assert predictions["train"].shape == predictions["heldout"].shape == (640, 4)
    counts = np.load(output / "sample_counts.npz")["seed0"]
    assert counts.sum() == 256
    sampler = BatchSampler(640, 0)
    for _ in range(4):
        sampler.next_batch()
    np.testing.assert_array_equal(counts, sampler.counts)
    assert result["sampling"]["0"]["batch_index_sha256"] == sampler.digest.hexdigest()
    with (output / "evaluations.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 18
    assert {r["mode"] for r in result["references"] if r["policy"] == "prior_stream"} == {"greedy", "epsilon_0_1"}
    assert all(t["map_seed"] == (700010 if t["panel"] == "train" else 970000) for t in result["trajectories"])
    manifest = json.loads((output / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
    for path, digest in result["protocol"]["source_sha256"].items():
        assert hashlib.sha256((output / "source" / path).read_bytes()).hexdigest() == digest


def test_incomplete_checkpoint_has_no_gate_or_partial_summary(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Test-only interruption.\n")
    original = supervised.rollout
    calls = 0

    def stop(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise supervised.BudgetReached("test cap")
        return original(*args, **kwargs)

    monkeypatch.setattr(supervised, "rollout", stop)
    output = tmp_path / "partial"
    result = run_study(output, protocol, seeds=[0], updates=1, train_maps=1, fresh_maps=1,
        train_seed_start=700020, fresh_seed_start=970000, prior_dir=fake_prior(tmp_path))
    assert result["run"]["status"] == "incomplete_admission_cap"
    assert result["run"]["train_updates"] == 0
    assert result["aggregate"] == result["state_aggregate"] == []
    assert result["gates"]["eligible"] is False
    with (output / "evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["checkpoint_complete"] == "0"


def test_cap_during_prior_preparation_preserves_resolved_dataset_metadata(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Test-only preparation cap.\n")

    def stop(*args, **kwargs):
        raise supervised.BudgetReached("test preparation cap")

    monkeypatch.setattr(supervised, "prior_overlap", stop)
    output = tmp_path / "preparation_cap"
    result = run_study(output, protocol, seeds=[0], updates=1, train_maps=1, fresh_maps=1,
        train_seed_start=700020, fresh_seed_start=970000)
    assert result["run"]["status"] == "incomplete_admission_cap"
    assert result["protocol"] == json.loads((output / "protocol.json").read_text())
    assert result["protocol"]["dataset"]["train_states"] == 640
    assert json.loads((output / "dataset_metadata.json").read_text())["status"] == "complete"
    assert result["gates"]["eligible"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
