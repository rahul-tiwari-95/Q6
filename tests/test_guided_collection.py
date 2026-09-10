"""Collection must record actual complete episodes without changing the learner."""
import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import q6.guided_collection as study
from q6.fixed_targets import ConsistencyError, module_hash
from q6.competence import BudgetReached
from q6.recorded_actions import build_recorded_table
from q6.world import CollectionWorld, WorldConfig


def collection_fixture():
    config = WorldConfig(horizon=4)
    env = CollectionWorld(config)
    observations, maps, positions, clocks = [], [], [], []
    for seed in (300000, 300032):
        env.reset(seed=seed)
        for position in np.argwhere(~(env.walls | env.pellets)):
            for remaining in range(1, config.horizon + 1):
                env.position = tuple(map(int, position))
                env.elapsed = config.horizon - remaining
                observations.append(env.observe())
                maps.append(seed)
                positions.append(env.position)
                clocks.append(remaining)
    data = {"observations": np.asarray(observations, np.float32),
            "map_seeds": np.asarray(maps, np.int64),
            "positions": np.asarray(positions, np.int16),
            "remaining": np.asarray(clocks, np.int16)}
    online = torch.nn.Linear(92, 4)
    with torch.no_grad():
        online.weight.zero_()
        online.bias.zero_()
    online.requires_grad_(False).eval()
    collector = SimpleNamespace(online=online, target=copy.deepcopy(online),
                               parameter_hash=lambda: module_hash(online))
    return config, data, collector


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_collection_random_slots_match_owned_rng_and_guided_ties_choose_zero(tmp_path):
    config, data, collector = collection_fixture()
    torch_before = torch.get_rng_state().clone()
    before = collector.parameter_hash(), module_hash(collector.target)
    summary, arrays = study.collect_mixture(config, data, collector, tmp_path,
                                          bank_id=2)
    records = read_rows(tmp_path / "collection_steps.csv")
    episodes = read_rows(tmp_path / "collection_episodes.csv")
    assert len(episodes) == 32
    assert all(int(row["complete"]) == 1 for row in episodes)
    assert sum(int(row["steps"]) for row in episodes) == len(records)
    assert int(arrays["visited_counts"].sum()) == len(records)
    for seed in np.unique(data["map_seeds"]):
        guided_paths = []
        for repetition in range(16):
            rows = [r for r in records if int(r["map_seed"]) == seed
                    and int(r["repetition"]) == repetition]
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), repetition, 77301, 2]))
            expected = [int(rng.integers(4)) if repetition < 8 else 0 for _ in rows]
            assert [int(r["action"]) for r in rows] == expected
            assert int(rows[-1]["terminated"]) + int(rows[-1]["truncated"]) == 1
            assert all(not (int(r["terminated"]) or int(r["truncated"])) for r in rows[:-1])
            if repetition >= 8:
                guided_paths.append([(r["current_row"], r["action"], r["next_row"]) for r in rows])
        assert all(route == guided_paths[0] for route in guided_paths)
    assert before == (collector.parameter_hash(), module_hash(collector.target))
    assert torch.equal(torch_before, torch.get_rng_state())
    assert all(p.grad is None for p in collector.online.parameters())
    table, table_summary = build_recorded_table(records, data, arrays["support_indices"])
    assert table_summary["all_nonterminal_successors_in_support_verified"]
    assert int(table["occurrences"].sum()) == len(records)
    assert table_summary["duplicate_records"] > 0


def test_collector_never_reads_exact_targets_or_counterfactual_transitions(tmp_path):
    config, data, collector = collection_fixture()
    class ForbiddenArray:
        def __getitem__(self, key):
            pytest.fail("collector consulted oracle labels")
        def __array__(self, *args, **kwargs):
            pytest.fail("collector converted oracle labels")
    data["targets"] = ForbiddenArray()
    data["winnable"] = ForbiddenArray()
    study.collect_mixture(config, data, collector, tmp_path, bank_id=1)
    assert len(read_rows(tmp_path / "collection_episodes.csv")) == 32


def test_guided_routes_are_shared_across_banks_but_random_streams_differ(tmp_path):
    config, data, collector = collection_fixture()
    paths = []
    for bank in (1, 3):
        folder = tmp_path / str(bank)
        study.collect_mixture(config, data, collector, folder, bank_id=bank)
        rows = read_rows(folder / "collection_steps.csv")
        paths.append({component: [(r["map_seed"], r["repetition"], r["current_row"], r["action"])
                     for r in rows if (int(r["repetition"]) >= 8) == (component == "guided")]
                      for component in ("random", "guided")})
    assert paths[0]["guided"] == paths[1]["guided"]
    assert paths[0]["random"] != paths[1]["random"]


def test_nonfinite_collector_prediction_is_rejected(tmp_path):
    config, data, collector = collection_fixture()
    with torch.no_grad():
        collector.online.bias[0] = float("nan")
    summary, _ = study.collect_mixture(config, data, collector, tmp_path, bank_id=1)
    assert summary["status"] == "inconsistent_not_evidence"
    assert "nonfinite" in summary["stop_reason"]
    episodes = read_rows(tmp_path / "collection_episodes.csv")
    assert len(episodes) == 9 and int(episodes[-1]["complete"]) == 0
    assert int(episodes[-1]["terminated"]) == int(episodes[-1]["truncated"]) == 0


def test_guard_interruption_does_not_invent_terminal_edges(tmp_path):
    config, data, collector = collection_fixture()
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise BudgetReached("synthetic interruption")
    summary, _ = study.collect_mixture(config, data, collector, tmp_path, bank_id=1, enforce=stop)
    assert summary["status"] == "incomplete_admission_cap"
    assert summary["complete_collection_episodes"] == 0
    rows = read_rows(tmp_path / "collection_steps.csv")
    env = CollectionWorld(config)
    env.reset(seed=300000)
    for row in rows:
        _, reward, terminated, truncated, _ = env.step(int(row["action"]))
        assert float(row["reward"]) == reward
        assert int(row["terminated"]) == int(terminated)
        assert int(row["truncated"]) == int(truncated)


def test_random_slot_identity_detects_changed_reward_and_missing_step(tmp_path):
    config, data, collector = collection_fixture()
    study.collect_mixture(config, data, collector, tmp_path / "original", bank_id=1)
    original = tmp_path / "original/collection_steps.csv"
    assert study.verify_random_slots(original, original)["identical"]
    rows = read_rows(original)
    altered = tmp_path / "altered.csv"
    for mutation in ("reward", "missing"):
        changed = copy.deepcopy(rows)
        if mutation == "reward":
            changed[0]["reward"] = str(float(changed[0]["reward"]) + 0.01)
        else:
            changed.pop(0)
        with altered.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(changed)
        assert not study.verify_random_slots(original, altered)["identical"]


def test_random_component_never_calls_collector_and_occurrence_is_not_support_weight(tmp_path):
    config, data, collector = collection_fixture()
    def forbidden(*args, **kwargs):
        pytest.fail("random episode queried the collector")
    collector.online.forward = forbidden
    summary, arrays = study.collect_mixture(config, data, collector, tmp_path,
                                          bank_id=1, random_episodes=16)
    assert summary["status"] == "complete"
    records = read_rows(tmp_path / "collection_steps.csv")
    table, metrics = build_recorded_table(records, data, arrays["support_indices"])
    repeated, repeated_metrics = build_recorded_table(records * 3, data, arrays["support_indices"])
    for key in ("observed", "rewards", "ends", "terminated", "truncated", "successor_indices"):
        np.testing.assert_array_equal(table[key], repeated[key])
    np.testing.assert_array_equal(table["occurrences"] * 3, repeated["occurrences"])
    assert metrics["recorded_edges"] == repeated_metrics["recorded_edges"]


def test_mutating_collector_disqualifies_collection(tmp_path):
    config, data, collector = collection_fixture()
    forward = collector.online.forward
    def mutate(observations):
        with torch.no_grad():
            collector.online.bias.add_(0.01)
        return forward(observations)
    collector.online.forward = mutate
    summary, _ = study.collect_mixture(config, data, collector, tmp_path, bank_id=1)
    assert summary["status"] == "inconsistent_not_evidence"
    assert not summary["collector_parameters_rng_unchanged"]
    assert "frozen collector" in summary["stop_reason"]


def test_preparation_budget_failure_preserves_an_ineligible_run(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic admission failure; no research evidence.\n")
    result = study.run_study(tmp_path / "run", protocol, bank_ids=[1], seeds=[0],
                             updates=2, panel_count=1, maps_per_panel=1,
                             smoke=True, max_seconds=0.000001)
    assert result["run"]["status"] == "incomplete_admission_cap"
    assert result["run"]["train_updates"] == result["run"]["collection_steps"] == 0
    assert not result["robustness"]["eligible"]
    saved = json.loads((tmp_path / "run/results.json").read_text())
    assert saved["run"] == result["run"]
    assert json.loads((tmp_path / "run/manifest.json").read_text())["status"] == "incomplete_admission_cap"
