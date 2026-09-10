"""Collection provenance, unique-support sampling and controlled-study integrity."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest
import torch

import q6.coverage as coverage
from q6.fixed_targets import build_transitions, double_dqn_targets, fixed_update
from q6.learning import DQN
from q6.optimal import VisibleOptimalQ
from q6.supervised import BatchSampler, enumerate_panel, layout_key
from q6.world import CollectionWorld, WorldConfig


def bank(config=None, seeds=(720000, 720001)):
    config = config or WorldConfig(size=3, wall_count=1, horizon=8)
    selection = coverage.select_coverage_layouts(config, seeds, 1, 990000, [], [])
    data = enumerate_panel(config, selection["train"], VisibleOptimalQ(config))
    return config, data, build_transitions(config, data)


def test_collector_reconstructs_actions_and_actual_steps_without_global_rng_or_learner(tmp_path):
    config, data, transitions = bank()
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = copy.deepcopy(np.random.get_state())
    result, arrays = coverage.collect_support(config, data, transitions, tmp_path, episodes_per_map=3)
    assert result["status"] == "complete"
    assert result["collection_episodes"] == result["complete_collection_episodes"] == 6
    assert result["collection_steps"] <= 6 * config.horizon
    torch.testing.assert_close(torch.random.get_rng_state(), torch_before, rtol=0, atol=0)
    for before, after in zip(numpy_before, np.random.get_state()):
        np.testing.assert_equal(before, after)
    raw = list(csv.DictReader((tmp_path / "collection_steps.csv").open()))
    counts = np.zeros(len(data["observations"]), np.uint32)
    env = CollectionWorld(config)
    last = None
    for row in raw:
        episode = int(row["map_seed"]), int(row["repetition"])
        if episode != last:
            env.reset(seed=episode[0])
            rng = np.random.default_rng(np.random.SeedSequence([*episode, 77301]))
            last = episode
        index, following = int(row["current_row"]), int(row["next_row"])
        np.testing.assert_array_equal(env.observe(), data["observations"][index])
        action = int(rng.integers(4))
        assert action == int(row["action"])
        observation, reward, terminal, timeout, _ = env.step(action)
        assert float(row["reward"]) == reward
        assert int(row["terminated"]) == terminal and int(row["truncated"]) == timeout
        if terminal or timeout:
            assert following == -1
        else:
            np.testing.assert_array_equal(observation, data["observations"][following])
        counts[index] += 1
    np.testing.assert_array_equal(arrays["visited_counts"], counts)
    np.testing.assert_array_equal(arrays["support_indices"], np.flatnonzero(counts))
    assert counts.sum() == len(raw) == result["collection_steps"]
    assert np.count_nonzero(counts) < counts.sum()  # Duplicate visits survive only in diagnostic counts.


def test_support_sampler_preserves_local_draws_global_identity_and_no_unvisited_exposure():
    support = np.array([0, 2, 4, 7, 9], np.int32)
    sampler = coverage.SupportSampler(support, 12, seed=1, batch_size=3)
    local = BatchSampler(5, 1, batch_size=3)
    expected_counts = np.zeros(12, np.uint32)
    global_draws = []
    for _ in range(6):
        positions = local.next_batch()
        expected = support[positions]
        actual = sampler.next_batch()
        np.testing.assert_array_equal(actual, expected)
        expected_counts[expected] += 1
        global_draws.extend(actual)
    np.testing.assert_array_equal(sampler.counts, expected_counts)
    np.testing.assert_array_equal(sampler.local.counts, local.counts)
    assert sampler.local.digest.hexdigest() == local.digest.hexdigest()
    assert sampler.digest.hexdigest() == hashlib.sha256(np.asarray(global_draws, dtype="<i8").tobytes()).hexdigest()
    assert not sampler.counts[np.setdiff1d(np.arange(12), support)].any()
    with pytest.raises(ValueError):
        coverage.SupportSampler([1, 1, 2], 12, 0, batch_size=2)
    with pytest.raises(ValueError):
        coverage.SupportSampler(support, 12, 0, batch_size=64)


def test_exhaustive_sampling_reproduces_previous_double_dqn_updates_bitwise():
    torch.set_num_threads(1)
    _, data, arrays = bank(WorldConfig(), (720007,))
    observations = torch.from_numpy(data["observations"])
    exact = torch.from_numpy(data["targets"].astype(np.float32))
    transitions = {k: torch.from_numpy(v.astype(np.float32) if k == "rewards" else v) for k, v in arrays.items()}
    left, right = DQN(92, seed=2), DQN(92, seed=2)
    old = BatchSampler(len(observations), 2)
    new = coverage.SupportSampler(np.arange(len(observations)), len(observations), 2)
    for _ in range(4):
        a, b = old.next_batch(), new.next_batch()
        np.testing.assert_array_equal(a, b)
        fixed_update(left, observations, exact, transitions, a, "double_dqn")
        coverage.fixed_update(right, observations, exact, transitions, b, "double_dqn")
    assert left.parameter_hash() == right.parameter_hash()
    assert coverage.module_hash(left.target) == coverage.module_hash(right.target)
    assert old.digest.hexdigest() == new.digest.hexdigest() == new.local.digest.hexdigest()
    np.testing.assert_array_equal(old.counts, new.counts)


def test_counterfactual_successor_queries_retain_outside_support_without_sampling_them():
    agent = DQN(92)
    with torch.no_grad():
        for p in list(agent.online.parameters()) + list(agent.target.parameters()):
            p.zero_()
        agent.online[-1].bias[1] = 1
        agent.target[-1].bias[1] = 2
    observations = torch.zeros((66, 92))
    support = np.arange(64, dtype=np.int32)
    sampler = coverage.SupportSampler(support, 66, 0)
    transitions = {"successor_indices": torch.full((66, 4), 65, dtype=torch.int32),
        "ends": torch.zeros((66, 4), dtype=torch.bool), "rewards": torch.zeros((66, 4))}
    indices = sampler.next_batch()
    targets = double_dqn_targets(agent, observations, transitions, indices)
    torch.testing.assert_close(targets, torch.full((64, 4), 1.94))
    assert sampler.counts[65] == 0 and 65 not in sampler.support
    assert not targets.requires_grad


def test_coverage_denominators_goal_near_clock_independence_and_successor_counts(tmp_path):
    config, data, transitions = bank()
    _, arrays = coverage.collect_support(config, data, transitions, tmp_path, episodes_per_map=2)
    metrics = coverage.coverage_metrics(data, transitions, arrays["visited_counts"])
    visited = arrays["visited_counts"] > 0
    assert metrics["unique_current_states"] == visited.sum()
    assert sum(r["visited_states"] for r in metrics["by_map"]) == visited.sum()
    assert sum(r["states"] for r in metrics["by_time_bucket"] if r["time_bucket"] != "all") == len(visited)
    env = CollectionWorld(config)
    near = []
    for seed, pos in zip(data["map_seeds"], data["positions"]):
        env.reset(seed=int(seed))
        env.position = tuple(map(int, pos))
        near.append(-env._potential() * config.size <= 2 + 1e-12)
    near = np.array(near)
    assert metrics["overall"]["goal_near_states"] == near.sum()
    assert metrics["overall"]["visited_goal_near_states"] == (near & visited).sum()
    assert metrics["overall"]["winnable_states"] == data["winnable"].sum()
    successors = transitions["successor_indices"][visited][~transitions["ends"][visited]]
    outside = successors[~visited[successors]]
    assert metrics["successor_queries"]["nonterminal_transitions"] == len(successors)
    assert metrics["successor_queries"]["outside_support_nonterminal_transitions"] == len(outside)
    assert metrics["successor_queries"]["unique_outside_support_destinations"] == len(np.unique(outside))


def test_fresh_selection_rejects_both_previous_panels():
    config = WorldConfig()
    env = CollectionWorld(config)
    panels = []
    for seed in (990000, 990001):
        env.reset(seed=seed)
        panels.append({"layout_hash": layout_key(env)})
    selected = coverage.select_coverage_layouts(config, [720000], 2, 990000, [panels[0]], [panels[1]])
    assert [r["reason"] for r in selected["collision_skips"][:2]] == ["previous_fixed_targets_fresh_layout", "previous_supervised_fresh_layout"]
    assert not {r["layout_hash"] for r in selected["heldout"]} & {r["layout_hash"] for r in panels}


def smoke(tmp_path, **kwargs):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Separate bounded coverage test protocol\n")
    return coverage.run_study(tmp_path / "study", protocol, seeds=[0], updates=3, train_maps=2, fresh_maps=1,
        train_seed_start=720000, fresh_seed_start=990000, smoke=True, **kwargs)


def test_smoke_preserves_collection_and_separate_sampling_with_no_gate_evidence(tmp_path):
    result = smoke(tmp_path)
    assert result["run"]["status"] == "complete"
    assert result["run"]["train_updates"] == 6
    assert result["run"]["training_examples"] == 384
    assert result["coverage"]["collection_episodes"] == 32
    assert not result["gates"]["eligible"]
    assert not any(c[k] for c in result["gates"]["per_condition"] for k in ("fresh", "training_fit", "efficient"))
    pair = result["provenance"]["paired_consistency"][0]
    assert pair["initial_weights_identical"] and pair["same_update_count"]
    assert pair["same_batch_indices_required"] is False
    a, b = (result["sampling"][condition]["0"] for condition in coverage.CONDITIONS)
    assert a["batch_index_sha256"] != b["batch_index_sha256"]
    directory = tmp_path / "study"
    with np.load(directory / "sample_counts.npz") as counts, np.load(directory / "collection.npz") as collected:
        assert not counts["collected_unique_seed0"][collected["visited_counts"] == 0].any()
        assert counts["exhaustive_seed0"].sum() == counts["collected_unique_seed0"].sum() == 192
    for row in result["paired_differences"]["per_layout"]:
        assert row["efficient_success_delta"] in (-1, 0, 1)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


def test_collection_interruption_is_saved_before_any_learning(tmp_path, monkeypatch):
    original = coverage.collect_support
    def interrupt(*args, **kwargs):
        checks = 0
        def enforce():
            nonlocal checks
            checks += 1
            if checks == 5:
                raise coverage.MemoryReached("fixture memory cap")
        kwargs["enforce"] = enforce
        return original(*args, **kwargs)
    monkeypatch.setattr(coverage, "collect_support", interrupt)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "incomplete_memory_cap"
    assert result["run"]["train_updates"] == 0
    assert result["coverage"]["collection_steps"] == 3
    assert result["coverage"]["complete_collection_episodes"] == 0
    assert not result["gates"]["eligible"]
    assert not list((tmp_path / "study" / "models").glob("*.pt"))
    assert (tmp_path / "study" / "collection_steps.csv").is_file()
    assert (tmp_path / "study" / "manifest.json").is_file()


def test_small_support_fails_explicitly_without_replacement_or_training(tmp_path):
    result = smoke(tmp_path, collection_episodes_per_map=1)
    assert result["run"]["status"] == "inconsistent_not_gate_evidence"
    assert result["run"]["train_updates"] == 0
    assert result["coverage"]["unique_current_states"] < 64
    assert "configuration failure" in result["provenance"]["stop_reason"]
    assert not result["gates"]["eligible"]
