"""Independently audit archived equal-size support evidence, without training.

Reuses the transition and raw-metric checks from earlier portable audits. The
uniform bank and batch schedules are reconstructed from their declared RNGs;
the historical collector is validated from its saved trace, never recollected.
--skip-forward-inference omits only bit-exact model forward reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
import audit_coverage_study as coverage_audit
import audit_fixed_targets_study as common
from audit_fixed_targets_study import close, read_json

CONDITIONS = ("collected_unique", "uniform_subset")
PANELS, BUCKETS = common.PANELS, common.BUCKETS


def audit_inputs(study, prior, fixed, earlier, data, transitions, metadata, protocol, result, World, config, *, smoke):
    provenance = read_json(study / "provenance.json")
    assert provenance == result["provenance"] and metadata["status"] == "complete"
    for name, spec in metadata["arrays"].items():
        assert list(data[name].shape) == spec["shape"] and str(data[name].dtype) == spec["dtype"]
        assert common.array_sha(data[name]) == spec["sha256"], name
    for name, digest in provenance["files"].items():
        assert common.sha(prior / name) == digest, ("historical input hash", name)
    for name, digest in provenance["archived_collection_files"].items():
        assert common.sha(prior / name) == digest, ("historical collection hash", name)
    for name in ("dataset_metadata.json", "sampling.json", "protocol.json", "collection.npz", "coverage.json"):
        assert common.sha(study / f"prior_{name}") == common.sha(prior / name)
    for archive, capture, field in ((earlier, "earlier_supervised_dataset_metadata.json", "earlier_supervised_metadata_sha256"),
                                    (fixed, "fixed_targets_dataset_metadata.json", "fixed_targets_metadata_sha256")):
        assert common.sha(archive / "dataset_metadata.json") == provenance[field] == common.sha(study / capture)
    assert provenance["new_collection_episodes"] == provenance["new_collection_steps"] == 0
    archive_metadata = [read_json(directory / "dataset_metadata.json") for directory in (prior, fixed, earlier)]
    previous_sets = [{r["layout_hash"] for r in item["heldout"]} for item in archive_metadata]
    train = {r["layout_hash"] for r in metadata["train"]}
    fresh = {r["layout_hash"] for r in metadata["heldout"]}
    assert not fresh & (train | set.union(*previous_sets)) and len(fresh) == len(metadata["heldout"])
    assert metadata["train_duplicate_layouts"] == {k: v for k, v in Counter(r["layout_hash"] for r in metadata["train"]).items() if v > 1}
    observed = {p: {hashlib.sha256(row.tobytes()).digest() for row in data[f"{p}_observations"]} for p in PANELS}
    assert not observed["train"] & observed["heldout"] and set(metadata["split_check"].values()) == {0}
    env = World(config)
    for panel in PANELS:
        for row in metadata[panel]:
            env.reset(seed=row["map_seed"])
            assert common.layout_hash(env) == row["layout_hash"] and list(env.position) == row["original_start"]
    candidate = 991000 if smoke else 960000
    accepted, rejected, used = [], [], set()
    while len(accepted) < len(metadata["heldout"]):
        env.reset(seed=candidate)
        key = common.layout_hash(env)
        reason = ("training_layout" if key in train else "previous_coverage_fresh_layout" if key in previous_sets[0] else
                  "previous_fixed_targets_fresh_layout" if key in previous_sets[1] else
                  "previous_supervised_fresh_layout" if key in previous_sets[2] else "earlier_fresh_layout" if key in used else None)
        row = {"map_seed": candidate, "layout_hash": key, "original_start": list(env.position)}
        if reason:
            rejected.append({**row, "reason": reason})
        else:
            accepted.append(row)
            used.add(key)
        candidate += 1
    assert accepted == metadata["heldout"] and rejected == metadata["collision_skips"]
    if smoke:
        assert [r["map_seed"] for r in metadata["train"]] == [730000, 730001]
    else:
        assert [r["map_seed"] for r in metadata["train"]] == list(range(300000, 300256))
        assert len(metadata["heldout"]) == 64
        with np.load(prior / "dataset.npz", allow_pickle=False) as old:
            for name, array in data.items():
                if name.startswith("train_"):
                    assert np.array_equal(array, old[name]), ("archived dataset identity", name)
        with np.load(prior / "transitions.npz", allow_pickle=False) as old:
            assert set(transitions) == set(old.files)
            for key, array in transitions.items():
                assert np.array_equal(array, old[key]), ("archived transition identity", key)
        assert provenance["training_arrays_identical"] and provenance["transition_arrays_identical"] and provenance["replication_required"]
        previous_protocol = read_json(prior / "protocol.json")
        for name in ("q6/world.py", "q6/learning.py", "q6/fixed_targets.py", "q6/supervised.py", "q6/competence.py"):
            assert protocol["source_sha256"][name] == previous_protocol["source_sha256"][name], ("unchanged algorithm/evaluation source", name)
    return provenance


def audit_supports(study, prior, data, metadata, transitions, protocol, result, *, smoke):
    description = read_json(study / "coverage.json")
    assert description == result["coverage"] and description["status"] == "complete"
    supports = dict(np.load(study / "supports.npz", allow_pickle=False))
    assert set(supports) == set(CONDITIONS)
    n = len(data["train_observations"])
    archived = dict(np.load(prior / "collection.npz", allow_pickle=False))
    for condition, rows in supports.items():
        spec = description["arrays"][condition]
        assert rows.dtype == np.int32 and rows.ndim == 1 and len(rows) >= 64
        assert np.all(rows >= 0) and np.all(rows < n) and np.all(np.diff(rows) > 0)
        assert list(rows.shape) == spec["shape"] and str(rows.dtype) == spec["dtype"]
        assert common.array_sha(rows) == spec["sha256"]
    collected = supports["collected_unique"]
    assert len(collected) == len(supports["uniform_subset"]) == description["support_size"]
    assert description["new_collection_episodes"] == description["new_collection_steps"] == 0
    assert description["synthetic_smoke_support"] == smoke
    if smoke:
        assert np.array_equal(collected, np.arange(0, n, 3, dtype=np.int32))
        historical_report = {"historical_collector_reconstructed": False}
    else:
        assert n == 163840 and len(collected) == 59626 and description["collected_support_identical_to_archive"]
        assert np.array_equal(collected, archived["support_indices"])
        assert np.array_equal(collected, np.flatnonzero(archived["visited_counts"]))
        # Validate the original trace against the independently checked table.
        # This reconstructs historical random actions, not new collection.
        _, historical_report = coverage_audit.audit_collection(prior, data, metadata, transitions,
            read_json(prior / "protocol.json"), read_json(prior / "results.json"))
        historical_report["historical_collector_reconstructed"] = True
    rng = np.random.default_rng(np.random.SeedSequence([88301]))
    expected = np.sort(rng.choice(n, size=len(collected), replace=False)).astype(np.int32)
    assert np.array_equal(expected, supports["uniform_subset"]), "single declared uniform subset draw"
    intersection = len(np.intersect1d(*supports.values()))
    union = len(np.union1d(*supports.values()))
    overlap = description["intersection"]
    assert overlap["states"] == intersection and overlap["union_states"] == union
    close(overlap["jaccard"], intersection / union)
    close(overlap["overlap_fraction"], intersection / len(collected))
    close(overlap["fraction_of_collected"], intersection / len(collected))
    close(overlap["fraction_of_uniform"], intersection / len(supports["uniform_subset"]))
    assert protocol["support_selection"]["actual_support_size"] == len(collected)
    near = np.zeros(n, bool)
    for seed in np.unique(data["train_map_seeds"]):
        rows = np.flatnonzero(data["train_map_seeds"] == seed)
        for position in np.unique(data["train_positions"][rows], axis=0):
            local = rows[np.all(data["train_positions"][rows] == position, axis=1)]
            if data["train_remaining"][local[data["train_winnable"][local]]].min() <= 2:
                near[local] = True
    assert {r["condition"] for r in description["per_condition"]} == set(CONDITIONS)
    for record in description["per_condition"]:
        support = supports[record["condition"]]
        mask = np.zeros(n, bool)
        mask[support] = True
        assert record["total_training_states"] == n and record["unique_current_states"] == len(support)
        def fraction(row, selected):
            win, nearby = selected & data["train_winnable"], selected & near
            expected = {"states": int(selected.sum()), "visited_states": int((selected & mask).sum()),
                "winnable_states": int(win.sum()), "visited_winnable_states": int((win & mask).sum()),
                "goal_near_states": int(nearby.sum()), "visited_goal_near_states": int((nearby & mask).sum())}
            assert "visits" not in row, "membership must not fabricate collector visits"
            for key, value in expected.items():
                assert row[key] == value, (record["condition"], key)
            for group, metric in ((selected, "coverage_rate"), (win, "winnable_coverage_rate"), (nearby, "goal_near_coverage_rate")):
                if group.any():
                    close(row[metric], mask[group].mean(), metric)
                else:
                    assert row[metric] is None
        fraction(record["overall"], np.ones(n, bool))
        assert {r["map_seed"] for r in record["by_map"]} == {r["map_seed"] for r in metadata["train"]}
        assert {r["time_bucket"] for r in record["by_time_bucket"]} == set(BUCKETS)
        for row in record["by_map"]:
            fraction(row, data["train_map_seeds"] == row["map_seed"])
        for row in record["by_time_bucket"]:
            low, high = (1, 32) if row["time_bucket"] == "all" else map(int, row["time_bucket"].split("-"))
            fraction(row, (data["train_remaining"] >= low) & (data["train_remaining"] <= high))
        for top, key in (("current_state_fraction", "coverage_rate"), ("winnable_current_state_fraction", "winnable_coverage_rate"),
                         ("goal_near_current_state_fraction", "goal_near_coverage_rate")):
            close(record[top], record["overall"][key])
        live = transitions["successor_indices"][support][~transitions["ends"][support]]
        outside = live[~mask[live]]
        access = record["successor_queries"]
        expected = {"all_action_transitions": len(support) * 4, "nonterminal_transitions": len(live),
            "outside_support_nonterminal_transitions": len(outside), "unique_nonterminal_destinations": len(np.unique(live)),
            "unique_outside_support_destinations": len(np.unique(outside))}
        for key, value in expected.items():
            assert access[key] == value, (record["condition"], key)
        assert access["terminal_transitions"] == len(support) * 4 - len(live)
        close(access["outside_support_fraction"], len(outside) / len(live))
    return supports, {**historical_report, "support_size_each": len(collected), "support_intersection": intersection,
        "support_union": union, "new_collection_episodes": 0, "new_collection_steps": 0}


def audit_sampling(study, prior, data, transitions, supports, protocol, result, provenance, *, smoke):
    sampling = read_json(study / "sampling.json")
    assert sampling == result["sampling"] and set(sampling) == set(CONDITIONS)
    global_arrays = dict(np.load(study / "sample_counts.npz", allow_pickle=False))
    local_arrays = dict(np.load(study / "local_sample_counts.npz", allow_pickle=False))
    old_sampling = read_json(prior / "sampling.json")
    old_counts = dict(np.load(prior / "sample_counts.npz", allow_pickle=False))
    n, final = len(data["train_observations"]), max(protocol["checkpoints"])
    expected_keys = {f"{c}_seed{s}" for c in CONDITIONS for s in protocol["seeds"]}
    assert set(global_arrays) == set(local_arrays) == expected_keys
    for seed in protocol["seeds"]:
        for condition in CONDITIONS:
            support = supports[condition]
            rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
            local_counts, global_counts = np.zeros(len(support), np.uint32), np.zeros(n, np.uint32)
            local_digest, global_digest = hashlib.sha256(), hashlib.sha256()
            for _ in range(final):
                local = rng.choice(len(support), 64, replace=False)
                selected = support[local].astype(np.int64)
                local_counts[local] += 1
                global_counts[selected] += 1
                local_digest.update(local.astype("<i8").tobytes())
                global_digest.update(selected.astype("<i8").tobytes())
            row = sampling[condition][str(seed)]
            assert row["updates"] == final and row["examples_seen"] == final * 64 and row["rng_seed_tuple"] == [seed, 66301]
            assert row["support_states"] == len(support) and row["unique_states_sampled"] == np.count_nonzero(global_counts)
            assert row["local_batch_index_sha256"] == local_digest.hexdigest()
            assert row["global_batch_index_sha256"] == row["batch_index_sha256"] == global_digest.hexdigest()
            assert np.array_equal(local_counts, local_arrays[f"{condition}_seed{seed}"])
            assert np.array_equal(global_counts, global_arrays[f"{condition}_seed{seed}"])
            mask = np.zeros(n, bool)
            mask[support] = True
            assert not global_counts[~mask].any(), "direct sample outside own support"
            live = ~transitions["ends"]
            outside = live & ~mask[np.maximum(transitions["successor_indices"], 0)]
            queries = int(np.dot(global_counts.astype(np.uint64), live.sum(1).astype(np.uint64)))
            external = int(np.dot(global_counts.astype(np.uint64), outside.sum(1).astype(np.uint64)))
            assert row["successor_queries"]["nonterminal_queries"] == queries
            assert row["successor_queries"]["outside_support_queries"] == external
            close(row["successor_queries"]["outside_support_fraction"], external / queries)
            if condition == "collected_unique" and not smoke:
                assert global_digest.hexdigest() == old_sampling[condition][str(seed)]["global_batch_index_sha256"]
                assert np.array_equal(global_counts, old_counts[f"{condition}_seed{seed}"])
        left, right = (sampling[c][str(seed)] for c in CONDITIONS)
        assert left["local_batch_index_sha256"] == right["local_batch_index_sha256"]
        assert left["global_batch_index_sha256"] != right["global_batch_index_sha256"]
        assert np.array_equal(*(local_arrays[f"{c}_seed{seed}"] for c in CONDITIONS))
    assert {r["seed"] for r in provenance["paired_consistency"]} == set(protocol["seeds"])
    for row in provenance["paired_consistency"]:
        for key in ("complete", "initial_weights_identical", "same_update_count", "same_local_batch_indices_required",
                    "local_batch_index_sha256_identical", "local_per_row_counts_identical"):
            assert row[key]
        assert row["same_batch_indices_required"] is False
    if not smoke:
        assert {r["seed"] for r in provenance["collected_replication"]} == set(protocol["seeds"])
        for row in provenance["collected_replication"]:
            for key in ("applicable", "final_weights_identical", "target_weights_identical", "initial_weights_identical", "batch_index_sha256_identical", "global_counts_identical"):
                assert row[key]
            path = prior / "models" / f"collected_unique_seed{row['seed']}_update30000.pt"
            assert common.sha(path) == row["archive_model_sha256"]
            before = torch.load(path, map_location="cpu", weights_only=True)
            after = torch.load(study / "models" / path.name, map_location="cpu", weights_only=True)
            for key in ("online", "target"):
                assert set(before[key]) == set(after[key])
                assert all(torch.equal(before[key][name], after[key][name]) for name in before[key]), ("historical tensor identity", key)
            assert before["parameter_hash"] == after["parameter_hash"] and before["target_parameter_hash"] == after["target_parameter_hash"]


def audit_support_diagnostics(study, data, supports, predictions, protocol, result):
    diagnostic = read_json(study / "support_diagnostics.json")
    assert diagnostic == result["support_diagnostics"]
    assert "not a gate" in diagnostic["classification"]
    target, win = data["train_targets"], data["train_winnable"]
    expected_keys = {(c, s, group) for c in CONDITIONS for s in protocol["seeds"] for group in ("support", "outside_support")}
    assert {(r["condition"], r["seed"], r["subset"]) for r in diagnostic["per_seed"]} == expected_keys
    assert len(diagnostic["per_seed"]) == len(expected_keys)
    errors = {}
    for row in diagnostic["per_seed"]:
        condition, seed, group = row["condition"], row["seed"], row["subset"]
        mask = np.zeros(len(target), bool)
        mask[supports[condition]] = True
        if group == "outside_support":
            mask = ~mask
        prediction = predictions[condition, seed]["train"]
        error = prediction.astype(np.float64) - target
        regret = target.max(1) - target[np.arange(len(target)), prediction.argmax(1)]
        winnable = mask & win
        expected = {"states": int(mask.sum()), "action_values": 4 * int(mask.sum()), "winnable_states": int(winnable.sum()),
            "impossible_states": int((mask & ~win).sum()), "optimal_winnable_actions": int((winnable & (regret <= 1e-6)).sum()),
            "abs_error_sum": np.abs(error[mask]).sum(), "squared_error_sum": np.square(error[mask]).sum()}
        for key, value in expected.items():
            close(row[key], value, key)
        errors[condition, seed, group] = np.abs(error[mask]).mean(1)
        check_diagnostic_ratios(row, errors[condition, seed, group])
    assert {(r["condition"], r["subset"]) for r in diagnostic["pooled"]} == {(c, p) for c in CONDITIONS for p in ("support", "outside_support")}
    assert len(diagnostic["pooled"]) == 4
    for row in diagnostic["pooled"]:
        selected = [r for r in diagnostic["per_seed"] if (r["condition"], r["subset"]) == (row["condition"], row["subset"])]
        assert row["seeds"] == len(protocol["seeds"]) == len(selected)
        assert row["unique_states"] == selected[0]["states"] and row["unique_winnable_states"] == selected[0]["winnable_states"]
        for key in ("states", "action_values", "winnable_states", "impossible_states", "optimal_winnable_actions", "abs_error_sum", "squared_error_sum"):
            close(row[key], sum(r[key] for r in selected), key)
        check_diagnostic_ratios(row, np.concatenate([errors[row["condition"], s, row["subset"]] for s in protocol["seeds"]]))
    return len(diagnostic["per_seed"]), len(diagnostic["pooled"])


def check_diagnostic_ratios(row, per_state_error):
    close(row["mean_abs_q_error"], row["abs_error_sum"] / row["action_values"])
    close(row["rmse_q_error"], np.sqrt(row["squared_error_sum"] / row["action_values"]))
    close(row["q95_abs_q_error"], np.quantile(per_state_error, .95))
    if row["winnable_states"]:
        close(row["optimal_action_rate"], row["optimal_winnable_actions"] / row["winnable_states"])
    else:
        assert row["optimal_action_rate"] is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/equal_support/pilot_v1"))
    parser.add_argument("--prior-study", type=Path, default=Path("experiments/coverage/pilot_v1"))
    parser.add_argument("--fixed-study", type=Path, default=Path("experiments/fixed_targets/pilot_v1"))
    parser.add_argument("--supervised-study", type=Path, default=Path("experiments/supervised/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Assertions must be enabled; do not run python -O.")
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p.resolve() if p.is_absolute() else (root / p).resolve()
    study, prior, fixed, earlier = map(resolve, (args.study, args.prior_study, args.fixed_study, args.supervised_study))
    sys.path.insert(0, str(study / "source"))
    from q6.learning import DQN
    from q6.world import CollectionWorld, WorldConfig
    torch.set_num_threads(1)
    started = time.monotonic()
    result, protocol, metadata = [read_json(study / name) for name in ("results.json", "protocol.json", "dataset_metadata.json")]
    smoke = bool(protocol["smoke"] or protocol["deviations"])
    assert not smoke or args.allow_smoke, "Smoke/deviating artifacts require explicit --allow-smoke."
    assert result["run"]["status"] == "complete" and protocol["id"] == "equal-support-v1"
    assert [c["id"] for c in protocol["conditions"]] == list(CONDITIONS)
    assert protocol["budget"]["maximum_collection_episodes"] == protocol["budget"]["maximum_collection_steps"] == 0
    assert protocol["collection"]["new_episodes"] == protocol["collection"]["new_steps"] == 0
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2] and protocol["checkpoints"] == [0, 1000, 3000, 10000, 30000]
        assert protocol["runtime"]["python"].startswith("3.12.") and protocol["runtime"]["torch"].split("+")[0] == "2.8.0"
        assert protocol["runtime"]["numpy"] == "2.0.2" and protocol["git"]["dirty"] is False
        assert protocol["support_selection"]["declared_support_size"] == 59626
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig())
    data = dict(np.load(study / "dataset.npz", allow_pickle=False))
    transitions = dict(np.load(study / "transitions.npz", allow_pickle=False))
    report = {"study": str(args.study), "smoke_validation_only": smoke,
        "manifest_files": common.audit_manifest(study, protocol, result)}
    provenance = audit_inputs(study, prior, fixed, earlier, data, transitions, metadata, protocol, result, CollectionWorld, config, smoke=smoke)
    report.update(common.audit_transitions(data, metadata, transitions, config, CollectionWorld))
    supports, support_report = audit_supports(study, prior, data, metadata, transitions, protocol, result, smoke=smoke)
    report.update(support_report)
    audit_sampling(study, prior, data, transitions, supports, protocol, result, provenance, smoke=smoke)
    predictions, models = coverage_audit.audit_model_predictions(study, data, protocol, CONDITIONS, result, DQN, skip_forward=args.skip_forward_inference)
    report.update(coverage_audit.audit_raw_tables(study, metadata, data, protocol, result, predictions, smoke=smoke,
        conditions=CONDITIONS, left_only_interpretation="collected_only_fresh_gate_met", right_only_interpretation="uniform_only_fresh_gate_met"))
    per_seed, pooled = audit_support_diagnostics(study, data, supports, predictions, protocol, result)
    maximum = 2 * len(protocol["seeds"]) * max(protocol["checkpoints"])
    assert result["run"]["train_updates"] == maximum and result["run"]["training_examples"] == maximum * 64
    assert result["run"]["model_parameter_count"] == 20420 and result["run"]["resource_checks"] > maximum
    assert result["run"]["peak_rss_bytes"] <= protocol["budget"]["peak_process_rss_bytes"]
    assert result["run"]["wall_seconds"] <= protocol["budget"]["admission_seconds"]
    assert result["run"]["collection_steps"] == 0
    progress = result["run"]["progress"]
    assert len(progress) == 2 * len(protocol["seeds"]) * len(protocol["checkpoints"]) and all(r["evaluation_complete"] for r in progress)
    assert {(r["condition"], r["seed"], r["checkpoint"]) for r in progress} == {
        (c, s, cp) for c in CONDITIONS for s in protocol["seeds"] for cp in protocol["checkpoints"]}
    assert read_json(study / "trajectories.json") == result["trajectories"]
    assert len(result["trajectories"]) == len(progress) * 4 + 4
    report.update(model_snapshots=models, forward_inference_checked=not args.skip_forward_inference,
        dense_prediction_arrays_reproduced=0 if args.skip_forward_inference else 4 * len(protocol["seeds"]),
        local_and_global_sampler_streams_reconstructed=4 * len(protocol["seeds"]), matched_local_sampling=True,
        collected_historical_online_target_and_global_exposure_identity_checked=not smoke,
        no_outside_support_current_state_samples=True, own_support_diagnostic_seed_rows=per_seed,
        own_support_diagnostic_pooled_rows=pooled, audit_seconds=time.monotonic() - started)
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
