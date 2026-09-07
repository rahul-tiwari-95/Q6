"""Independent bank RNGs, frozen ordering, unchanged updates and evidence integrity."""
import csv
import hashlib
import json

import numpy as np
import pytest

import q6.bank_replication as study
from q6.coverage import collect_support
from q6.fixed_targets import build_transitions
from q6.optimal import VisibleOptimalQ
from q6.supervised import enumerate_panel
from q6.world import CollectionWorld, WorldConfig


def small_data():
    config = WorldConfig(size=3, wall_count=1, horizon=8)
    data = enumerate_panel(config, [{"map_seed": 740000}], VisibleOptimalQ(config))
    return config, data, build_transitions(config, data)


def test_collector_suffix_defines_new_reconstructible_streams_and_empty_default(tmp_path):
    config, data, transitions = small_data()
    streams = []
    for suffix in ((), (1,), (2,), (3,)):
        directory = tmp_path / ("legacy" if not suffix else f"bank{suffix[0]}")
        result, arrays = collect_support(config, data, transitions, directory, episodes_per_map=3, rng_suffix=suffix)
        assert result["collection_episodes"] == 3
        rows = list(csv.DictReader((directory / "collection_steps.csv").open()))
        for repetition in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([740000, repetition, 77301, *suffix]))
            chosen = [int(r["action"]) for r in rows if int(r["repetition"]) == repetition]
            assert chosen == [int(rng.integers(4)) for _ in chosen]
        np.testing.assert_array_equal(arrays["support_indices"], np.flatnonzero(arrays["visited_counts"]))
        streams.append([int(r["action"]) for r in rows])
    assert len({tuple(s) for s in streams}) == 4
    implicit = tmp_path / "implicit"
    collect_support(config, data, transitions, implicit, episodes_per_map=3)
    assert (implicit / "collection_steps.csv").read_bytes() == (tmp_path / "legacy/collection_steps.csv").read_bytes()


def test_uniform_draw_uses_owned_bank_id_stream_and_realized_cardinality():
    supports = []
    for bank_id, size in ((1, 477), (2, 396), (3, 495)):
        result = study.choose_uniform_support(1280, size, bank_id)
        expected = np.sort(np.random.default_rng(np.random.SeedSequence([88301, bank_id])).choice(1280, size=size, replace=False)).astype(np.int32)
        np.testing.assert_array_equal(result, expected)
        assert len(result) == len(np.unique(result)) == size
        supports.append(result)
    with pytest.raises(study.ConsistencyError):
        study.choose_uniform_support(1280, 63, 1)


def smoke(tmp_path, **kwargs):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Bounded independent-bank smoke fixture\n")
    return study.run_study(tmp_path / "study", protocol, seeds=[0], updates=3, train_maps=2,
        train_seed_start=740000, panel_count=2, maps_per_panel=2, panel_seed_start=1100000, smoke=True, **kwargs)


def test_all_banks_frozen_before_updates_and_all_fits_before_any_evaluation(tmp_path, monkeypatch):
    original_prepare, original_update, original_rollout = study.prepare_banks, study.fixed_update, study.rollout
    state = {"prepared": False, "updates": 0, "rollouts": 0}
    def prepared(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        assert len(result[0]) == len(result[1]) == 3
        assert all(not array.flags.writeable for pair in result[1].values() for array in pair.values())
        state["prepared"] = True
        return result
    def update(*args, **kwargs):
        assert state["prepared"] and state["rollouts"] == 0
        state["updates"] += 1
        return original_update(*args, **kwargs)
    def rollout(*args, **kwargs):
        assert state["updates"] == 18  # Three banks × two arms × three fixture updates.
        state["rollouts"] += 1
        return original_rollout(*args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("this runner must never use online collection/replay learning")
    monkeypatch.setattr(study, "prepare_banks", prepared)
    monkeypatch.setattr(study, "fixed_update", update)
    monkeypatch.setattr(study, "rollout", rollout)
    monkeypatch.setattr(study.DQN, "observe", forbidden)
    monkeypatch.setattr(study.DQN, "learn", forbidden)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "complete"
    assert result["run"]["train_updates"] == 18 and result["run"]["training_examples"] == 1152
    assert result["run"]["collection_episodes"] == 96
    assert result["run"]["learner_episodes"] == 72 and result["run"]["reference_episodes"] == 12
    assert len(result["trajectories"]) == 28 and len(result["support_intersections"]) == 12
    assert len(result["paired_differences"]["per_layout"]) == 12
    assert not result["descriptive_thresholds"]["eligible"]
    assert result["provenance"]["snapshot_integrity"]["complete"]
    assert result["provenance"]["snapshot_integrity"]["expected"] == 12
    assert all(r["unchanged"] and r["read_only"] for r in result["provenance"]["support_integrity"])
    assert all(r["online_matches_prior"] and r["target_matches_prior"] for r in result["provenance"]["initialization_consistency"])
    for bank in result["sampling"].values():
        a, b = (bank[condition]["0"] for condition in study.CONDITIONS)
        assert a["local_batch_index_sha256"] == b["local_batch_index_sha256"]
        assert a["global_batch_index_sha256"] != b["global_batch_index_sha256"]
        assert a["outside_support_direct_samples"] == b["outside_support_direct_samples"] == 0
    for row in result["pooled"]["aggregate"]:
        assert row["noop_rate"] == row["noop_steps"] / row["evaluation_steps"]
        group = [r for r in result["aggregate"] if (r["condition"], r["panel"], r["mode"]) == (row["condition"], row["panel"], row["mode"])]
        assert row["mean_bank_noop_rate"] == pytest.approx(np.mean([r["noop_rate"] for r in group]))
    directory = tmp_path / "study"
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((directory / path).read_bytes()).hexdigest() == digest


def test_collector_cap_preserves_completed_bank_supports_and_no_training(tmp_path, monkeypatch):
    original = study.collect_support
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            def cap():
                raise study.MemoryReached("second-bank collector fixture cap")
            kwargs["enforce"] = cap
        return original(*args, **kwargs)
    monkeypatch.setattr(study, "collect_support", interrupted)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "incomplete_memory_cap"
    assert result["run"]["train_updates"] == result["run"]["learner_episodes"] == 0
    assert result["banks"][0]["status"] == "complete"
    with np.load(tmp_path / "study/supports.npz") as arrays:
        assert set(arrays.files) == {"collected_unique_bank1", "uniform_subset_bank1"}
    assert not result["descriptive_thresholds"]["eligible"]
    assert (tmp_path / "study/banks/bank2/collection_steps.csv").is_file()


def test_nonfinite_loss_is_ineligible_and_preserves_initial_snapshot(tmp_path, monkeypatch):
    def bad_update(*args, **kwargs):
        return float("nan")
    monkeypatch.setattr(study, "fixed_update", bad_update)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "inconsistent_not_evidence"
    assert "nonfinite" in result["run"]["stop_reason"]
    assert not result["descriptive_thresholds"]["eligible"]
    assert (tmp_path / "study/models/bank1_collected_unique_seed0_update0.pt").is_file()
    assert result["run"]["learner_episodes"] == 0


def test_guard_after_aggregation_disqualifies_finished_measurements(tmp_path, monkeypatch):
    original = study.descriptive_summaries
    def summaries_then_cap(*args, **kwargs):
        result = original(*args, **kwargs)
        def cap(*args, **kwargs):
            raise study.MemoryReached("post-aggregation fixture cap")
        monkeypatch.setattr(study, "guard", cap)
        return result
    monkeypatch.setattr(study, "descriptive_summaries", summaries_then_cap)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "incomplete_memory_cap"
    assert result["run"]["learner_episodes"] == 72
    assert not result["descriptive_thresholds"]["eligible"] and not result["robustness"]["eligible"]
    assert not any(r["success_reference_met"] or r["efficiency_reference_met"] for r in result["descriptive_thresholds"]["per_seed"])
