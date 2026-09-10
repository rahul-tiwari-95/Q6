"""Equal-cardinality support controls without collection or algorithm changes."""
import copy
import hashlib
import json

import numpy as np
import pytest

from q6 import coverage, equal_support
from q6.fixed_targets import ConsistencyError, build_transitions
from q6.optimal import VisibleOptimalQ
from q6.supervised import enumerate_panel, layout_key
from q6.world import CollectionWorld, WorldConfig


def test_uniform_subset_is_one_owned_declared_draw_with_exact_size():
    before = copy.deepcopy(np.random.get_state())
    actual = equal_support.uniform_support(1280, 427)
    expected = np.sort(np.random.default_rng(np.random.SeedSequence([88301])).choice(1280, size=427, replace=False)).astype(np.int32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32 and len(np.unique(actual)) == 427
    for left, right in zip(before, np.random.get_state()):
        np.testing.assert_equal(left, right)
    with pytest.raises(ConsistencyError):
        equal_support.uniform_support(100, 63)


def small_bank():
    config = WorldConfig()
    selected = coverage.select_coverage_layouts(config, [730000], 1, 991000, [], [])
    data = enumerate_panel(config, selected["train"], VisibleOptimalQ(config))
    return data, build_transitions(config, data)


def test_archive_support_is_copied_unchanged_and_counts_never_become_weights(tmp_path):
    data, transitions = small_bank()
    prior = tmp_path / "prior"
    prior.mkdir()
    indices = np.arange(0, len(data["observations"]), 3, dtype=np.int32)
    counts = np.zeros(len(data["observations"]), np.uint32)
    counts[indices] = np.arange(1, len(indices) + 1)
    np.savez_compressed(prior / "collection.npz", support_indices=indices, visited_counts=counts)
    comparison = equal_support.EqualSupportComparison()
    comparison.configure_protocol({"budget": {}, "optimizer": {}})
    output = tmp_path / "output"
    output.mkdir()
    metadata, _, supports = comparison.prepare_supports(data, transitions, output, prior,
        smoke=False, required=False, enforce=lambda: None)
    np.testing.assert_array_equal(supports["collected_unique"], indices)
    assert metadata["collected_support_identical_to_archive"]
    assert len(supports["uniform_subset"]) == len(indices)
    assert metadata["new_collection_steps"] == metadata["new_collection_episodes"] == 0
    for arm in metadata["per_condition"]:
        assert "visits" not in arm["overall"]
        assert all("visits" not in r for r in arm["by_map"] + arm["by_time_bucket"])
        edges = arm["successor_queries"]
        assert edges["terminal_transitions"] + edges["nonterminal_transitions"] == len(indices) * 4
        assert "supported" in edges["denominator"]
    counts[indices[0]] = 0
    np.savez_compressed(prior / "collection.npz", support_indices=indices, visited_counts=counts)
    with pytest.raises(ConsistencyError, match="identity"):
        comparison.prepare_supports(data, transitions, output, prior, smoke=False, required=False, enforce=lambda: None)


def test_fresh_panel_excludes_all_three_prior_panels():
    env = CollectionWorld()
    previous = []
    for seed in (991000, 991001, 991002):
        env.reset(seed=seed)
        previous.append({"layout_hash": layout_key(env)})
    comparison = equal_support.EqualSupportComparison()
    comparison.fixed_metadata = {"heldout": [previous[1]]}
    selected = comparison.select_layouts(env.config, [730000], 1, 991000, [previous[0]], [previous[2]], float("inf"))
    assert [r["reason"] for r in selected["collision_skips"][:3]] == [
        "previous_coverage_fresh_layout", "previous_fixed_targets_fresh_layout", "previous_supervised_fresh_layout"]
    assert selected["heldout"][0]["map_seed"] == 991003


def test_final_diagnostics_use_each_arms_own_mask_and_winnable_denominator(tmp_path):
    targets = np.eye(4, dtype=np.float64)
    data = {"train": {"targets": targets, "winnable": np.array([True, True, False, True])}}
    supports = {"collected_unique": np.array([0, 1]), "uniform_subset": np.array([2, 3])}
    left = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]], np.float32)
    right = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], np.float32)
    for seed in (0, 1):
        np.savez_compressed(tmp_path / f"predictions_collected_unique_seed{seed}.npz", train=left)
        np.savez_compressed(tmp_path / f"predictions_uniform_subset_seed{seed}.npz", train=right)
    result = equal_support.final_support_diagnostics(tmp_path, data, supports, [0, 1])
    pooled = {(r["condition"], r["subset"]): r for r in result["pooled"]}
    assert pooled[("collected_unique", "support")]["optimal_action_rate"] == .5
    assert pooled[("collected_unique", "outside_support")]["optimal_action_rate"] == 0
    assert pooled[("uniform_subset", "support")]["optimal_action_rate"] == 1
    assert pooled[("uniform_subset", "support")]["winnable_states"] == 2
    assert pooled[("uniform_subset", "support")]["unique_winnable_states"] == 1
    assert pooled[("uniform_subset", "support")]["states"] == 4
    assert pooled[("uniform_subset", "support")]["mean_abs_q_error"] == .25
    assert "not a gate" in result["classification"]


def smoke(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Separate synthetic smoke fixture; no collection\n")
    return equal_support.run_study(tmp_path / "study", protocol, seeds=[0], updates=3, train_maps=2, fresh_maps=1,
        train_seed_start=730000, fresh_seed_start=991000, smoke=True)


def test_shared_runner_equal_smoke_pairs_local_counts_without_collection(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("equal-support study must never collect")
    monkeypatch.setattr(coverage, "collect_support", forbidden)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "complete"
    assert result["run"]["train_updates"] == 6 and result["run"]["collection_steps"] == 0
    assert result["coverage"]["support_size"] == 427
    assert result["coverage"]["synthetic_smoke_support"]
    pair = result["provenance"]["paired_consistency"][0]
    assert pair["initial_weights_identical"] and pair["local_batch_index_sha256_identical"] and pair["local_per_row_counts_identical"]
    assert not pair["same_batch_indices_required"] and pair["same_local_batch_indices_required"]
    a, b = (result["sampling"][condition]["0"] for condition in equal_support.CONDITIONS)
    assert a["local_batch_index_sha256"] == b["local_batch_index_sha256"]
    assert a["global_batch_index_sha256"] != b["global_batch_index_sha256"]
    directory = tmp_path / "study"
    with np.load(directory / "supports.npz") as supports, np.load(directory / "sample_counts.npz") as counts:
        for condition in equal_support.CONDITIONS:
            outside = np.ones(1280, bool)
            outside[supports[condition]] = False
            assert not counts[f"{condition}_seed0"][outside].any()
    assert len(result["support_diagnostics"]["per_seed"]) == 4
    assert not result["gates"]["eligible"]
    assert not any(c[k] for c in result["gates"]["per_condition"] for k in ("fresh", "training_fit", "efficient"))
    assert not (directory / "collection_steps.csv").exists()
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((directory / path).read_bytes()).hexdigest() == digest


def test_post_diagnostic_resource_cap_keeps_measurements_but_disqualifies_run(tmp_path, monkeypatch):
    original = equal_support.final_support_diagnostics
    def diagnostic_then_cap(*args, **kwargs):
        result = original(*args, **kwargs)
        def stopped(*args, **kwargs):
            raise coverage.MemoryReached("post-diagnostic fixture cap")
        monkeypatch.setattr(coverage, "guard", stopped)
        return result
    monkeypatch.setattr(equal_support, "final_support_diagnostics", diagnostic_then_cap)
    result = smoke(tmp_path)
    assert result["run"]["status"] == "incomplete_memory_cap"
    assert result["run"]["train_updates"] == 6
    assert result["support_diagnostics"]["pooled"]
    assert not result["gates"]["eligible"]
