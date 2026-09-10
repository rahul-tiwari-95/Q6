"""Frozen inference, prospective panels, paired arithmetic and admission integrity."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest
import torch

import q6.learning
import q6.panel_evaluation as study
from q6.competence import ROOT
from q6.world import CollectionWorld, WorldConfig


def test_frozen_policy_allocates_no_optimizer_replay_or_training_state(monkeypatch):
    path = ROOT / "experiments/equal_support/pilot_v1/models/collected_unique_seed0_update30000.pt"
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    def forbidden(*args, **kwargs):
        pytest.fail("frozen evaluation cannot instantiate a learner or optimizer")
    monkeypatch.setattr(q6.learning.DQN, "__init__", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = copy.deepcopy(np.random.get_state())
    model = study.FrozenPolicy(snapshot)
    assert set(vars(model)) == {"observation_size", "online", "target"}
    assert not model.online.training and not model.target.training
    assert not any(p.requires_grad for network in (model.online, model.target) for p in network.parameters())
    result = model.online(torch.zeros((2, 92)))
    assert result.shape == (2, 4) and not result.requires_grad
    assert model.parameter_hash() == snapshot["parameter_hash"]
    assert study.module_hash(model.target) == snapshot["target_parameter_hash"]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_before, rtol=0, atol=0)
    for left, right in zip(numpy_before, np.random.get_state()):
        np.testing.assert_equal(left, right)
    snapshot["parameter_hash"] = "incorrect"
    with pytest.raises(study.ConsistencyError, match="hashes"):
        study.FrozenPolicy(snapshot)


def test_selection_excludes_prior_layouts_and_every_previously_selected_panel():
    env = CollectionWorld()
    names = ["training_layout", "previous_supervised", "previous_fixed", "previous_coverage", "previous_equal"]
    excluded = {}
    for offset, name in enumerate(names):
        env.reset(seed=1010000 + offset)
        excluded[name] = {study.layout_key(env)}
    selected = study.select_panels(env.config, excluded, 2, 2, 1010000, 1, lambda: None)
    assert selected["panels"][0]["map_seeds"] == [1010005, 1010006]
    assert selected["panels"][1]["map_seeds"] == [1010007, 1010008]
    assert [r["reason"] for r in selected["collision_skips"][:5]] == names
    assert any(r["reason"] == "earlier_selected_layout" for r in selected["collision_skips"])
    assert selected["pairwise_panel_intersections"][0]["intersection"] == 0
    assert not any(selected["exclusion_intersections"].values())


def smoke(tmp_path, **kwargs):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Separate frozen-policy smoke; no training or collection\n")
    return study.run_study(tmp_path / "study", protocol, seeds=[0], panel_count=2, maps_per_panel=2,
        panel_seed_start=1010000, smoke=True, **kwargs)


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path = tmp_path_factory.mktemp("panel-complete")
    return path, smoke(path)


def test_smoke_has_expected_cells_greedy_pairs_and_unchanged_per_panel_hashes(completed):
    path, result = completed
    assert result["run"]["status"] == "complete"
    assert result["run"]["learner_episodes"] == 36 and result["run"]["reference_episodes"] == 12
    assert result["run"]["new_training_updates"] == result["run"]["new_collection_steps"] == result["run"]["new_support_draws"] == 0
    assert len(result["trajectories"]) == 16
    assert result["provenance"]["expected_cells"]["complete"]
    assert result["provenance"]["all_loaded_models_unchanged"]
    assert len(result["paired_differences"]["per_layout"]) == 12
    assert len(result["paired_differences"]["per_seed"]) == len(result["paired_differences"]["aggregate"]) == 9
    assert all(r["mode"] == "greedy" for r in result["paired_differences"]["per_layout"])
    assert not result["descriptive_thresholds"]["eligible"] and not result["robustness"]["eligible"]
    assert "gates" not in result
    for record in result["provenance"]["models"]:
        assert record["source_sha256"] == record["source_sha256_after"] == record["saved_sha256_after"]
        assert len(record["panel_checks"]) == 2 and record["unchanged"]
        for check in record["panel_checks"]:
            assert check["online_before"] == check["online_hash"]
            assert check["target_before"] == check["target_hash"]
            assert check["source_sha256_before"] == check["source_sha256"] == check["saved_sha256"]
    directory = path / "study"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


def test_paired_efficiency_and_blocked_rate_use_raw_denominators(completed):
    path, result = completed
    with (path / "study/evaluations.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    with (path / "study/references.csv").open() as handle:
        references = list(csv.DictReader(handle))
    planner = {(r["panel"], int(r["map_seed"])): int(r["steps"]) for r in references if r["policy"] == "shortest_path"}
    lookup = {(r["condition"], r["panel"], int(r["seed"]), int(r["map_seed"])): r for r in raw if r["mode"] == "greedy"}
    comparisons = {r["id"]: r for r in study.COMPARISONS}
    for pair in result["paired_differences"]["per_layout"]:
        comparison = comparisons[pair["comparison"]]
        key = pair["panel"], pair["seed"], pair["map_seed"]
        left, right = lookup[(comparison["left"], *key)], lookup[(comparison["right"], *key)]
        distance = planner[(pair["panel"], pair["map_seed"])]
        efficient = lambda r: int(r["success"]) == 1 and int(r["steps"]) <= 2 * distance
        assert pair["efficient_success_delta"] == int(efficient(right)) - int(efficient(left))
        assert pair["noop_rate_delta"] == pytest.approx(int(right["noop_steps"]) / int(right["steps"]) - int(left["noop_steps"]) / int(left["steps"]))
    for aggregate in result["aggregate"]:
        selected = [r for r in raw if r["condition"] == aggregate["condition"] and r["mode"] == aggregate["mode"] and
                    (aggregate["panel"] == "all" or r["panel"] == aggregate["panel"])]
        assert aggregate["noop_rate"] == pytest.approx(sum(int(r["noop_steps"]) for r in selected) / sum(int(r["steps"]) for r in selected))
        assert aggregate["efficient_success_episodes"] / len(selected) == aggregate["efficient_success_rate"]


def test_expected_cells_reject_duplicate_and_missing_rows(completed):
    path, result = completed
    def typed(name):
        with (path / "study" / name).open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in ("seed", "map_seed", "repetition", "checkpoint_complete"):
                row[key] = int(row[key])
        return rows
    rows, refs = typed("evaluations.csv"), typed("references.csv")
    assert study.verify_expected_cells(rows, refs, result["panels"], [0])["complete"]
    assert not study.verify_expected_cells(rows + [rows[0]], refs, result["panels"], [0])["complete"]
    assert not study.verify_expected_cells(rows[:-1], refs, result["panels"], [0])["complete"]


def test_partial_panel_preserves_rows_but_is_ineligible(tmp_path, monkeypatch):
    original = study.rollout
    learner_calls = 0
    def interrupted(*args, **kwargs):
        nonlocal learner_calls
        if kwargs.get("policy", "learner") == "learner":
            learner_calls += 1
            if learner_calls == 2:
                raise study.BudgetReached("fixture episode admission cap")
        return original(*args, **kwargs)
    monkeypatch.setattr(study, "rollout", interrupted)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "incomplete_admission_cap"
    assert result["run"]["learner_episodes"] == 1
    assert result["run"]["completed_learner_episodes"] == 0
    assert not result["aggregate"]
    assert not result["descriptive_thresholds"]["eligible"]
    assert not result["provenance"]["expected_cells"]["complete"]
    assert (tmp_path / "study/manifest.json").is_file()


def test_resource_cap_after_descriptive_aggregation_disqualifies_finished_evaluation(tmp_path, monkeypatch):
    original = study.descriptive_summaries
    def summarized_then_capped(*args, **kwargs):
        result = original(*args, **kwargs)
        def stopped(*args, **kwargs):
            raise study.MemoryReached("fixture after aggregation cap")
        monkeypatch.setattr(study, "guard", stopped)
        return result
    monkeypatch.setattr(study, "descriptive_summaries", summarized_then_capped)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "incomplete_memory_cap"
    assert result["run"]["learner_episodes"] == 36
    assert not result["descriptive_thresholds"]["eligible"] and not result["robustness"]["eligible"]
    assert not any(r["success_reference_met"] or r["efficiency_reference_met"] for r in result["descriptive_thresholds"]["per_seed"])
