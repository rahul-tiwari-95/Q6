"""Controlled-target mathematics, pairing and interrupted-study evidence checks."""
import copy
import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import q6.fixed_targets as fixed
from q6.learning import DQN
from q6.optimal import VisibleOptimalQ
from q6.supervised import BatchSampler, enumerate_panel, layout_key, supervised_update
from q6.world import CollectionWorld, WorldConfig


def small_bank(config=None):
    config = config or WorldConfig(size=3, wall_count=1, horizon=4)
    selection = fixed.select_fixed_layouts(config, [710007], 1, 980007, [])
    data = enumerate_panel(config, selection["train"], VisibleOptimalQ(config))
    return config, data, fixed.build_transitions(config, data)


@pytest.mark.parametrize("world", [WorldConfig(size=3, wall_count=1, horizon=4), WorldConfig(), WorldConfig(size=3, wall_count=1, horizon=1)])
def test_compact_successors_match_every_actual_transition_and_bellman(world):
    config, data, transitions = small_bank(world)
    env = CollectionWorld(config)
    env.reset(seed=710007)
    for i in range(len(data["observations"])):
        env.position = tuple(map(int, data["positions"][i]))
        env.elapsed = config.horizon - int(data["remaining"][i])
        for action in range(4):
            following, reward, success, timeout, _ = env.clone().step(action)
            assert transitions["rewards"][i, action] == reward
            assert transitions["ends"][i, action] == (success or timeout)
            index = transitions["successor_indices"][i, action]
            if success or timeout:
                assert index == -1
            else:
                np.testing.assert_array_equal(following, data["observations"][index])
    checked = fixed.validate_transitions(config, data, transitions)
    assert checked["action_targets_checked"] == len(data["observations"]) * 4
    assert checked["maximum_bellman_residual"] < 1e-12
    transitions["rewards"][0, 0] += .01
    with pytest.raises(fixed.ConsistencyError, match="Bellman"):
        fixed.validate_transitions(config, data, transitions)


def test_double_dqn_online_selection_target_evaluation_terminal_mask_and_soft_timing():
    torch.set_num_threads(1)
    agent = DQN(92)
    with torch.no_grad():
        for p in list(agent.online.parameters()) + list(agent.target.parameters()):
            p.zero_()
        agent.online[-1].bias.copy_(torch.tensor([0., 4., 2., 1.]))
        agent.target[-1].bias.copy_(torch.tensor([7., 1., 9., 3.]))
    observations = torch.zeros((64, 92))
    transitions = {"successor_indices": torch.zeros((64, 4), dtype=torch.int32),
                   "rewards": torch.full((64, 4), .2), "ends": torch.zeros((64, 4), dtype=torch.bool)}
    transitions["ends"][:, 2:] = True
    transitions["successor_indices"][:, 2:] = -1
    indices = np.arange(64)
    labels = fixed.double_dqn_targets(agent, observations, transitions, indices)
    expected = torch.tensor([[1.17, 1.17, .2, .2]]).repeat(64, 1)
    torch.testing.assert_close(labels, expected)
    assert not labels.requires_grad
    before_target = copy.deepcopy(agent.target.state_dict())
    before_rng = copy.deepcopy(agent.rng.bit_generator.state)
    manual_loss = torch.nn.functional.smooth_l1_loss(agent.online(observations), expected).item()
    actual_loss = fixed.fixed_update(agent, observations, torch.zeros((64, 4)), transitions, indices, "double_dqn")
    assert actual_loss == pytest.approx(manual_loss)
    for key, online in agent.online.state_dict().items():
        torch.testing.assert_close(agent.target.state_dict()[key], before_target[key].lerp(online, .01), rtol=0, atol=0)
    assert all(p.grad is None for p in agent.target.parameters())
    assert agent.count == agent.steps == agent.updates == 0
    assert agent.rng.bit_generator.state == before_rng


def test_exact_arm_reproduces_existing_computation_with_paired_sampling():
    torch.set_num_threads(1)
    config = WorldConfig()
    data = enumerate_panel(config, fixed.select_fixed_layouts(config, [710010], 1, 980010, [])["train"], VisibleOptimalQ(config))
    observations = torch.from_numpy(data["observations"])
    targets = torch.from_numpy(data["targets"].astype(np.float32))
    left, right = DQN(92, seed=2), DQN(92, seed=2)
    samplers = [BatchSampler(len(observations), 2), BatchSampler(len(observations), 2)]
    target_before = fixed.module_hash(left.target)
    for _ in range(4):
        a, b = (s.next_batch() for s in samplers)
        np.testing.assert_array_equal(a, b)
        fixed.fixed_update(left, observations, targets, {}, a, "exact_q")
        supervised_update(right, observations, targets, b)
    assert left.parameter_hash() == right.parameter_hash()
    assert fixed.module_hash(left.target) == target_before
    assert samplers[0].digest.hexdigest() == samplers[1].digest.hexdigest()
    np.testing.assert_array_equal(samplers[0].counts, samplers[1].counts)


def test_previous_fresh_layout_is_excluded_even_when_spawn_changes():
    config = WorldConfig()
    env = CollectionWorld(config)
    env.reset(seed=980000)
    key = layout_key(env)
    env.position = tuple(map(int, np.argwhere(~(env.walls | env.pellets))[0]))
    env.elapsed = 5
    assert layout_key(env) == key
    selected = fixed.select_fixed_layouts(config, [710000], 2, 980000, [{"layout_hash": key}])
    assert selected["collision_skips"][0]["reason"] == "previous_supervised_fresh_layout"
    assert all(r["layout_hash"] != key for r in selected["heldout"])


def test_efficiency_counts_failures_and_process_memory_units(monkeypatch):
    rows = [{"panel": "heldout", "map_seed": 1, "success": True, "steps": 4},
            {"panel": "heldout", "map_seed": 2, "success": True, "steps": 7},
            {"panel": "heldout", "map_seed": 3, "success": False, "steps": 32}]
    assert fixed.efficient_rate(rows, {("heldout", 1): 2, ("heldout", 2): 3, ("heldout", 3): 3}) == 1 / 3
    monkeypatch.setattr(fixed.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=17))
    monkeypatch.setattr(fixed.platform, "system", lambda: "Darwin")
    assert fixed.peak_rss_bytes() == 17
    monkeypatch.setattr(fixed.platform, "system", lambda: "Linux")
    assert fixed.peak_rss_bytes() == 17 * 1024
    with pytest.raises(fixed.MemoryReached):
        fixed.guard(float("inf"), 1)


def smoke(tmp_path, **kwargs):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Declared separate smoke fixture\n")
    return fixed.run_study(tmp_path / "study", protocol, seeds=[0], updates=3, train_maps=1, fresh_maps=1,
        train_seed_start=710013, fresh_seed_start=980013, smoke=True, **kwargs)


def test_smoke_artifacts_preserve_complete_pairing_and_never_pass_gates(tmp_path):
    result = smoke(tmp_path)
    assert result["run"]["status"] == "complete"
    assert result["run"]["train_updates"] == 6
    assert result["run"]["training_examples"] == 384
    assert not result["gates"]["eligible"]
    assert not any(c[k] for c in result["gates"]["per_condition"] for k in ("fresh", "training_fit", "efficient"))
    assert all(result["provenance"]["paired_consistency"][0][k] for k in
        ("complete", "initial_weights_identical", "batch_index_sha256_identical", "per_row_counts_identical"))
    assert result["run"]["resource_checks"] > 6
    directory = tmp_path / "study"
    assert len(list((directory / "models").glob("*.pt"))) == 4
    assert len(result["trajectories"]) == 20
    assert len(result["paired_differences"]["per_layout"]) == 2
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    raw = list(csv.DictReader((directory / "evaluations.csv").open()))
    assert len(raw) == 24 and all(r["checkpoint_complete"] == "1" for r in raw)
    with np.load(directory / "sample_counts.npz") as counts:
        np.testing.assert_array_equal(counts["exact_q_seed0"], counts["double_dqn_seed0"])


def test_consistency_and_partial_checkpoint_failures_finalize_without_gate_evidence(tmp_path, monkeypatch):
    def inconsistent(*args, **kwargs):
        raise fixed.ConsistencyError("deliberately corrupted transition fixture")
    monkeypatch.setattr(fixed, "validate_transitions", inconsistent)
    first = tmp_path / "consistency"
    first.mkdir()
    result = smoke(first)
    assert result["run"]["status"] == "inconsistent_not_gate_evidence"
    assert result["run"]["train_updates"] == 0
    assert not result["gates"]["eligible"]
    assert "corrupted" in result["provenance"]["stop_reason"]
    monkeypatch.undo()
    original = fixed.rollout
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise fixed.BudgetReached()
        return original(*args, **kwargs)
    monkeypatch.setattr(fixed, "rollout", interrupted)
    second = tmp_path / "partial"
    second.mkdir()
    result = smoke(second)
    assert result["run"]["status"] == "incomplete_admission_cap"
    assert not result["aggregate"] and not result["state_aggregate"]
    assert result["run"]["progress"][0]["evaluation_complete"] is False
    raw = list(csv.DictReader((second / "study" / "evaluations.csv").open()))
    assert len(raw) == 1 and raw[0]["checkpoint_complete"] == "0"
    assert (second / "study" / "manifest.json").is_file()


def test_postaggregation_memory_guard_disqualifies_finished_updates(tmp_path, monkeypatch):
    memory = {"bytes": 1}
    original = fixed.efficient_rate
    def aggregate_then_exceed(*args, **kwargs):
        value = original(*args, **kwargs)
        memory["bytes"] = 5 * 1024**3
        return value
    monkeypatch.setattr(fixed, "peak_rss_bytes", lambda: memory["bytes"])
    monkeypatch.setattr(fixed, "efficient_rate", aggregate_then_exceed)
    result = smoke(tmp_path)
    assert result["run"]["train_updates"] == 6
    assert result["run"]["status"] == "incomplete_memory_cap"
    assert not result["gates"]["eligible"]
    assert not any(c[k] for c in result["gates"]["per_condition"] for k in ("fresh", "training_fit", "efficient"))
    assert json.loads((tmp_path / "study" / "provenance.json").read_text())["stop_reason"]
