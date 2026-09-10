"""Familiar masks diagnose fixed policies; they must not invent experience."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import q6.familiar_starts as study
from q6.fixed_targets import ConsistencyError
from q6.world import CollectionWorld, WorldConfig


def reachability_fixture():
    # Deliberately shuffled clocks. Row 0 has a high-return failure and a
    # lower-return success: return optimality must not define reachability.
    remaining = np.array([3, 1, 2, 1, 2], np.int32)
    maps = np.full(5, 300000, np.int64)
    table = {"observed": np.zeros((5, 4), bool),
             "rewards": np.full((5, 4), np.nan),
             "ends": np.ones((5, 4), bool),
             "terminated": np.zeros((5, 4), bool),
             "truncated": np.zeros((5, 4), bool),
             "successor_indices": np.full((5, 4), -1, np.int32),
             "occurrences": np.zeros((5, 4), np.uint32)}
    # row, action, reward, successor, success, timeout
    edges = [(0, 0, 10., 2, False, False),
             (0, 2, -10., 4, False, False),
             (1, 0, 10., -1, False, True),
             (2, 0, 10., 1, False, False),
             (3, 1, -10., -1, True, False),
             (4, 1, -10., 3, False, False),
             (4, 3, -10., -1, True, False)]
    for row, action, reward, nxt, success, timeout in edges:
        table["observed"][row, action] = True
        table["rewards"][row, action] = reward
        table["ends"][row, action] = success or timeout
        table["terminated"][row, action] = success
        table["truncated"][row, action] = timeout
        table["successor_indices"][row, action] = nxt
        table["occurrences"][row, action] = 1
    return table, remaining, maps


def test_success_reachability_is_independent_of_return_and_handles_last_move_success():
    table, remaining, maps = reachability_fixture()
    before = copy.deepcopy(table)
    reachable, distance, _ = study.logged_success_paths(table, remaining, maps)
    np.testing.assert_array_equal(reachable, [True, False, False, True, True])
    np.testing.assert_array_equal(distance, [2, -1, -1, 1, 1])
    for key in table:
        np.testing.assert_array_equal(table[key], before[key])


def test_missing_actions_and_occurrence_counts_cannot_create_successful_paths():
    table, remaining, maps = reachability_fixture()
    expected = study.logged_success_paths(table, remaining, maps)[:2]
    missing = ~table["observed"]
    table["terminated"][missing] = True
    table["truncated"][missing] = True
    table["rewards"][missing] = 1e20
    table["successor_indices"][missing] = 9999
    table["occurrences"] *= 1000
    actual = study.logged_success_paths(table, remaining, maps)[:2]
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fault", ["outside", "empty_mask", "cross_map", "clock", "flags", "terminal_index"])
def test_invalid_logged_graph_is_rejected_without_fallback(fault):
    table, remaining, maps = reachability_fixture()
    if fault == "outside":
        table["successor_indices"][0, 0] = 999
    elif fault == "empty_mask":
        table["observed"][2] = False
    elif fault == "cross_map":
        maps[2] += 1
    elif fault == "clock":
        remaining[2] = 3
    elif fault == "flags":
        table["truncated"][3, 1] = True
    else:
        table["successor_indices"][3, 1] = 1
    with pytest.raises(ConsistencyError):
        study.logged_success_paths(table, remaining, maps)


def test_reachability_obeys_resource_guard_before_completion():
    table, remaining, maps = reachability_fixture()
    def stop():
        raise RuntimeError("test budget exhausted")
    with pytest.raises(RuntimeError, match="test budget exhausted"):
        study.logged_success_paths(table, remaining, maps, enforce=stop)


def test_action_selection_uses_lowest_allowed_tie_and_rejects_empty_mask():
    q = np.array([9., -2., -2., 8.])
    mask = np.array([False, True, True, False])
    assert study.select_action(q, mask, "unrestricted") == 0
    assert study.select_action(q, mask, "logged") == 1
    assert study.select_action(q, np.zeros(4, bool), "unrestricted") == 0
    with pytest.raises(ConsistencyError, match="no fallback"):
        study.select_action(q, np.zeros(4, bool), "logged")
    with pytest.raises(ConsistencyError):
        study.select_action([np.nan, 0., 0., 0.], mask, "unrestricted")


def rollout_fixture():
    config = WorldConfig(horizon=3)
    env = CollectionWorld(config)
    env.reset(seed=300000)
    observations, positions, clocks = [], [], []
    for position in np.argwhere(~(env.walls | env.pellets)):
        for remaining in (1, 2, 3):
            env.position = tuple(map(int, position))
            env.elapsed = 3 - remaining
            observations.append(env.observe())
            positions.append(position)
            clocks.append(remaining)
    data = {"observations": np.array(observations), "positions": np.array(positions),
            "remaining": np.array(clocks), "map_seeds": np.full(len(clocks), 300000)}
    lookup = study.make_row_lookup(data)
    n = len(clocks)
    table = {"observed": np.ones((n, 4), bool), "rewards": np.zeros((n, 4)),
             "ends": np.zeros((n, 4), bool), "terminated": np.zeros((n, 4), bool),
             "truncated": np.zeros((n, 4), bool), "successor_indices": np.full((n, 4), -1)}
    for row in range(n):
        for action in range(4):
            env.reset(seed=300000)
            env.position = tuple(map(int, data["positions"][row]))
            env.elapsed = 3 - int(data["remaining"][row])
            _, reward, terminated, truncated, _ = env.step(action)
            table["rewards"][row, action] = reward
            table["terminated"][row, action] = terminated
            table["truncated"][row, action] = truncated
            table["ends"][row, action] = terminated or truncated
            if not (terminated or truncated):
                table["successor_indices"][row, action] = lookup[(300000, *env.position, 3-env.elapsed)]
    network = torch.nn.Linear(92, 4)
    with torch.no_grad():
        network.weight.zero_()
        network.bias.copy_(torch.tensor([3., 2., 1., 0.]))
    model = SimpleNamespace(online=network)
    kwargs = dict(bank_id=1, condition="constrained_bootstrap", seed=0,
                  map_seed=300000, action_set="unrestricted")
    env.reset(seed=300000)
    first = lookup[(300000, *env.position, 3)]
    return model, config, data, lookup, table, np.zeros((n, 4)), kwargs, first


def test_unrecorded_action_can_land_in_support_and_terminal_is_not_an_exit():
    model, config, data, lookup, table, graph, kwargs, first = rollout_fixture()
    table["observed"][first, 0] = False
    before = {k: v.clone() for k, v in model.online.state_dict().items()}
    episode, steps, _ = study.evaluate_start(model, config, data, lookup, table, graph, **kwargs)
    assert episode["off_mask_actions"] == 1 and episode["first_off_mask_step"] == 1
    assert episode["support_exits"] == episode["unsupported_steps"] == 0
    assert episode["first_support_exit_step"] is None
    assert steps[0]["successor_supported"] and not steps[0]["support_exit"]
    assert steps[-1]["successor_row"] == -1 and not steps[-1]["support_exit"]
    for key, value in model.online.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.online.parameters())


def test_support_exit_reentry_and_occupancy_have_distinct_counts():
    model, config, data, lookup, table, graph, kwargs, first = rollout_fixture()
    successor = table["successor_indices"][first, 0]
    assert successor >= 0
    table["observed"][first, 0] = False
    table["observed"][successor] = False
    episode, steps, _ = study.evaluate_start(model, config, data, lookup, table, graph, **kwargs)
    assert episode["first_off_mask_step"] == episode["first_support_exit_step"] == 1
    assert episode["support_exits"] == episode["support_reentries"] == 1
    assert episode["unsupported_steps"] == 1 and episode["supported_steps"] == 2
    assert episode["off_mask_actions"] == 1
    assert steps[1]["current_supported"] is False and not steps[1]["off_mask_action"]


def test_masked_rollout_rejects_logged_edge_into_missing_support():
    model, config, data, lookup, table, graph, kwargs, first = rollout_fixture()
    table["observed"][table["successor_indices"][first, 0]] = False
    kwargs["action_set"] = "logged"
    with pytest.raises(ConsistencyError, match="closure"):
        study.evaluate_start(model, config, data, lookup, table, graph, **kwargs)


def test_paired_mask_interaction_and_pooled_step_denominators():
    model, config, data, lookup, table, graph, kwargs, _ = rollout_fixture()
    original, _, _ = study.evaluate_start(model, config, data, lookup, table, graph, **kwargs)
    rows = []
    # Two starts: DDQN gains 0.5 efficient success with a mask; exact gains
    # 1.0. The interaction is +0.5, not either model's raw masked rate.
    values = [[0, 1, 0, 1], [1, 1, 0, 1]]
    for offset, outcomes in enumerate(values):
        for index, (condition, action_set) in enumerate(
                (c, a) for c in study.CONDITIONS for a in study.ACTION_SETS):
            row = {**original, "condition": condition, "action_set": action_set,
                   "map_seed": 300000 + offset, "efficient_success": outcomes[index],
                   "success": outcomes[index], "steps": 2 if offset == 0 else 8,
                   "noop_steps": 1 if offset == 0 else 0,
                   "supported_steps": 1 if offset == 0 else 8,
                   "unsupported_steps": 1 if offset == 0 else 0,
                   "off_mask_actions": 1 if offset == 0 else 0}
            rows.append(row)
    _, aggregates, _, paired, pooled = study.summarize_crossed(rows, [], [1], [0], ["block_0"])
    interaction = next(x for x in paired["aggregate"] if x["block"] == "all" and x["comparison"] == "interaction")
    assert interaction["efficient_success_delta"] == .5 and interaction["starts"] == 2
    assert len(paired["per_start"]) == 10
    assert next(x for x in pooled["paired"] if x["block"] == "all" and x["comparison"] == "interaction")["efficient_success_delta"] == .5
    for row in aggregates:
        assert row["noop_rate"] == .1  # 1/10 steps, not mean(1/2,0/8)
        assert row["off_mask_rate"] == pytest.approx(1/9)  # supported decisions


def test_interrupted_episode_streams_completed_steps_without_an_invented_outcome():
    model, config, data, lookup, table, graph, kwargs, _ = rollout_fixture()
    saved, admissions = [], []
    def guard():
        admissions.append(1)
        if len(admissions) == 3:
            raise study.BudgetReached("fixture admission stop")
    with pytest.raises(study.BudgetReached):
        study.evaluate_start(model, config, data, lookup, table, graph,
                             enforce=guard, step_sink=saved.append, **kwargs)
    assert [x["step"] for x in saved] == [1, 2]
    assert not any(x["terminated"] or x["truncated"] for x in saved)


def test_duplicate_state_identity_is_rejected():
    _, _, data, _, _, _, _, _ = rollout_fixture()
    for key, value in data.items():
        data[key] = np.concatenate([value, value[:1]])
    with pytest.raises(ConsistencyError, match="duplicate"):
        study.make_row_lookup(data)


def compact_run(path, **kwargs):
    protocol = path / "declaration.md"
    protocol.write_text("Separate integration fixture; never research evidence.\n")
    return study.run_study(path / "study", protocol, banks=[1], seeds=[0],
                           map_seeds=[300000], smoke=True, **kwargs)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path = tmp_path_factory.mktemp("familiar-compact")
    import q6.learning
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("a frozen diagnostic cannot construct a learner or optimizer")
        patch.setattr(q6.learning.DQN, "__init__", forbidden)
        patch.setattr(torch.optim.Adam, "__init__", forbidden)
        result = compact_run(path)
    return path, result


def test_compact_run_preserves_final_models_raw_steps_and_reference_counts(completed):
    import gzip
    path, result = completed
    assert result["run"]["status"] == "complete"
    assert result["run"]["learner_episodes"] == 4
    assert result["run"]["reference_episodes"] == result["run"]["frozen_models"] == 2
    assert result["run"]["train_updates"] == result["run"]["collection_steps"] == 0
    assert result["run"]["new_labels_fitted"] == 0
    assert len(result["trajectories"]) == 6
    assert not result["robustness"]["eligible"]
    assert result["provenance"]["expected_cells"]["complete"]
    assert all(m["unchanged"] and m["checkpoint"] == 30000 for m in result["provenance"]["models"])
    assert result["prediction_slices"]["new_inference_rows"] == 0
    with gzip.open(path / "study/steps.jsonl.gz", "rt") as stream:
        steps = [json.loads(line) for line in stream]
    assert len(steps) == result["run"]["step_records"]
    masked = [s for s in steps if s["action_set"] == "logged"]
    assert all(s["current_supported"] and s["selected_action_recorded"] and not s["support_exit"] for s in masked)
    manifest = json.loads((path / "study/manifest.json").read_text())
    for file, digest in manifest["files"].items():
        assert hashlib.sha256((path / "study" / file).read_bytes()).hexdigest() == digest


def test_expected_cells_reject_duplicates_missing_and_duplicated_references(completed):
    path, result = completed
    rows = json.loads((path / "study/episodes.json").read_text())
    refs = json.loads((path / "study/references.json").read_text())
    check = lambda episodes, references: study.verify_cells(episodes, references, [1], [0], [300000])["complete"]
    assert check(rows, refs)
    assert not check(rows + rows[:1], refs)
    assert not check(rows[:-1], refs)
    assert not check(rows, refs + refs[:1])
    assert not check(rows, refs[:-1])


def test_resource_stop_preserves_partial_evidence_and_disqualifies_it(tmp_path, monkeypatch):
    original, calls = study.evaluate_start, []
    def stop_after_one(*args, **kwargs):
        if kwargs.get("policy", "learner") == "learner":
            calls.append(1)
            if len(calls) == 2:
                raise study.BudgetReached("fixture stops before second learner episode")
        return original(*args, **kwargs)
    monkeypatch.setattr(study, "evaluate_start", stop_after_one)
    result = compact_run(tmp_path)
    assert result["run"]["status"] == "incomplete_admission_cap"
    assert result["run"]["learner_episodes"] == 1
    assert not result["robustness"]["eligible"]
    assert not result["provenance"]["expected_cells"]["complete"]
    assert (tmp_path / "study/manifest.json").is_file()


def test_error_offset_decomposition_does_not_confuse_value_fit_with_ranking():
    mask = np.array([[True, True, False, False], [True, False, False, False]])
    targets = np.array([[1., 3., np.nan, np.nan], [5., np.nan, np.nan, np.nan]])
    predicted = np.array([[4., 2., 100., 100.], [7., 100., 100., 100.]])
    result = study.prediction_slice_metrics(predicted, targets, mask)
    # Row errors[3,-1] have mean1 and centered[2,-2]; one-action error2
    # contributes only an offset. Equal state MSE=(5+4)/2=4.5.
    assert result["state_mean_squared_error"] == 4.5
    assert result["state_mean_squared_offset"] == 2.5
    assert result["centered_state_mean_squared_error"] == 4.
    assert result["restricted_action_agreement"] == .5
    assert result["multiple_action_states"] == 1
