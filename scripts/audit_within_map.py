"""Independent audit of quota-matched within-map support replacements.

Checks declared support draws, exact replay/map pairing, immutable archived
controls, exposure, raw outcomes and preselected recordings. It does not train,
collect or repeat the full learned-policy evaluation. Portable mode omits only
regeneration of neural outputs for saved recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
import audit_fixed_targets_study as common
import audit_bank_replication as bank_audit
import audit_map_replay as shared
import audit_panel_evaluation as replay_audit
from audit_fixed_targets_study import close, csv_rows, read_json

CONDITIONS = ("collected_unique", "within_map_uniform")


def reconstruct_support(original, map_seeds, bank):
    """Independent one-shot draws; no labels, observations or outcomes consulted."""
    assert original.dtype == np.int32 and np.all(np.diff(original) > 0)
    pieces, quotas = [], []
    for map_seed in np.unique(map_seeds):
        available = np.flatnonzero(map_seeds == map_seed)
        selected = original[map_seeds[original] == map_seed]
        assert len(available) == 640 and 0 < len(selected) <= len(available)
        rng = np.random.default_rng(np.random.SeedSequence([bank, int(map_seed), 105301]))
        replacement = np.sort(available[rng.choice(len(available), size=len(selected), replace=False)]).astype(np.int32)
        pieces.append(replacement)
        quotas.append({"map_seed": int(map_seed), "states": len(selected), "intersection": len(np.intersect1d(selected, replacement))})
    result = np.concatenate(pieces).astype(np.int32)
    assert len(result) == len(original) and np.all(np.diff(result) > 0)
    assert np.array_equal(map_seeds[result], map_seeds[original]), "each sorted local index retains its map"
    return result, quotas


def reconstruct_streams(original, replacement, map_seeds, seed, treatment_updates):
    """Reconstruct historical 30k draws and the treatment's declared prefix."""
    assert 0 < treatment_updates <= 30000 and len(original) == len(replacement)
    assert np.array_equal(map_seeds[original], map_seeds[replacement])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
    baseline = {"global_counts": np.zeros(len(map_seeds), np.uint32), "local_counts": np.zeros(len(original), np.uint32),
                "map_counts": np.zeros(256, np.uint32), "digests": {k: hashlib.sha256() for k in ("local", "global", "map")}}
    treatment = {"global_counts": np.zeros(len(map_seeds), np.uint32), "local_counts": np.zeros(len(original), np.uint32),
                 "map_counts": np.zeros(256, np.uint32), "digests": {k: hashlib.sha256() for k in ("local", "global", "map")}}
    baseline_prefix = {k: hashlib.sha256() for k in ("local", "global", "map")}
    for update in range(30000):
        local = rng.choice(len(original), size=64, replace=False)
        rows = original[local].astype(np.int64)
        maps = map_seeds[rows].astype(np.int64)
        assert len(np.unique(rows)) == 64
        baseline["local_counts"][local] += 1
        baseline["global_counts"][rows] += 1
        baseline["map_counts"] += np.bincount(maps - 300000, minlength=256).astype(np.uint32)
        for field, values in (("local", local), ("global", rows), ("map", maps)):
            baseline["digests"][field].update(values.astype("<i8").tobytes())
        if update < treatment_updates:
            current = replacement[local].astype(np.int64)
            assert np.array_equal(map_seeds[current], maps)
            treatment["local_counts"][local] += 1
            treatment["global_counts"][current] += 1
            treatment["map_counts"] += np.bincount(maps - 300000, minlength=256).astype(np.uint32)
            for field, values in (("local", local), ("global", current), ("map", maps)):
                treatment["digests"][field].update(values.astype("<i8").tobytes())
            for field, values in (("local", local), ("global", rows), ("map", maps)):
                baseline_prefix[field].update(values.astype("<i8").tobytes())
    for group, updates in ((baseline, 30000), (treatment, treatment_updates)):
        assert int(group["global_counts"].sum()) == int(group["local_counts"].sum()) == int(group["map_counts"].sum()) == updates * 64
        group["digests"] = {k: v.hexdigest() for k, v in group["digests"].items()}
    baseline_prefix = {k: v.hexdigest() for k, v in baseline_prefix.items()}
    assert treatment["digests"]["local"] == baseline_prefix["local"] and treatment["digests"]["map"] == baseline_prefix["map"]
    if treatment_updates == 30000:
        assert np.array_equal(treatment["local_counts"], baseline["local_counts"])
        assert np.array_equal(treatment["map_counts"], baseline["map_counts"])
    return baseline, treatment, baseline_prefix


def validate_inputs(study, archives, protocol):
    metadata = read_json(study / "dataset_metadata.json")
    arrays = dict(np.load(study / "dataset.npz", allow_pickle=False))
    transitions = dict(np.load(study / "transitions.npz", allow_pickle=False))
    assert metadata["status"] == "complete" and set(arrays) == set(metadata["arrays"])
    for name, value in arrays.items():
        assert name.startswith("train_")
        bank_audit.check_array(value, metadata["arrays"][name])
    assert set(transitions) == set(metadata["transition_arrays"])
    for name, value in transitions.items():
        bank_audit.check_array(value, metadata["transition_arrays"][name])
    for archive in (archives["coverage"], archives["bank_replication"]):
        old_metadata = read_json(shared.checked_archive_file(archive, "dataset_metadata.json"))
        assert metadata["train"] == old_metadata["train"]
        with np.load(shared.checked_archive_file(archive, "dataset.npz"), allow_pickle=False) as old:
            assert all(np.array_equal(value, old[name]) for name, value in arrays.items())
        with np.load(shared.checked_archive_file(archive, "transitions.npz"), allow_pickle=False) as old:
            assert all(np.array_equal(value, old[name]) for name, value in transitions.items())
    assert [r["map_seed"] for r in metadata["train"]] == list(range(300000, 300256))
    data = {k.removeprefix("train_"): v for k, v in arrays.items()}
    assert len(data["observations"]) == 163840
    saved = dict(np.load(study / "supports.npz", allow_pickle=False))
    old = dict(np.load(shared.checked_archive_file(archives["bank_replication"], "supports.npz"), allow_pickle=False))
    assert set(saved) == {f"{c}_bank{b}" for c in CONDITIONS for b in protocol["bank_ids"]}
    supports, quota_rows = {}, {}
    for bank, size in zip((1, 2, 3), (59839, 59984, 59838)):
        original = old[f"collected_unique_bank{bank}"]
        assert len(original) == size and original.dtype == np.int32
        replacement, quotas = reconstruct_support(original, data["map_seeds"], bank)
        for condition, expected in ((CONDITIONS[0], original), (CONDITIONS[1], replacement)):
            actual = saved[f"{condition}_bank{bank}"]
            assert actual.dtype == np.int32 and np.array_equal(actual, expected)
            supports[bank, condition] = actual
        quota_rows[bank] = quotas
    for name in ("q6/world.py", "q6/learning.py", "q6/fixed_targets.py", "q6/competence.py", "q6/supervised.py", "q6/optimal.py"):
        for archive in (archives["coverage"], archives["bank_replication"]):
            old_protocol = read_json(shared.checked_archive_file(archive, "protocol.json"))
            assert protocol["source_sha256"][name] == old_protocol["source_sha256"][name]
            assert common.sha(shared.checked_archive_file(archive, "source/" + name)) == old_protocol["source_sha256"][name]
    return data, arrays, metadata, transitions, supports, quota_rows


def reconstruct_panels(study, archives, protocol, result, metadata, World, config):
    excluded = {}
    for name, directory in archives.items():
        if name in ("panel_evaluation", "bank_replication", "map_replay"):
            selection = read_json(shared.checked_archive_file(directory, "panels.json"))
            layouts = [r for p in selection["panels"] for r in p["layouts"]]
        else:
            saved = read_json(shared.checked_archive_file(directory, "dataset_metadata.json"))
            layouts = saved["heldout"]
        excluded[f"previous_{name}_fresh_layout"] = {r["layout_hash"] for r in layouts}
        if name == "coverage":
            excluded["training_layout"] = {r["layout_hash"] for r in metadata["train"]}
    selection = read_json(study / "panels.json")
    assert selection["status"] == "complete" and selection["selection_uses_outcomes"] is False
    assert selection["panels"] == result["panels"] == protocol["panels"]
    spec = protocol["panel_selection"]
    env, used, panels, rejected, task_hashes, planner = World(config), set(), [], [], {}, {}
    for index in range(spec["count"]):
        panel = {"id": f"panel_{index}", "scan_start": spec["start"] + index * spec["stride"], "layouts": [], "map_seeds": []}
        candidate = panel["scan_start"]
        while len(panel["layouts"]) < spec["maps_per_panel"]:
            env.reset(seed=candidate)
            key = common.layout_hash(env)
            reason = next((name for name, values in excluded.items() if key in values), None)
            if reason is None and key in used:
                reason = "earlier_selected_layout"
            record = {"map_seed": candidate, "layout_hash": key, "original_start": list(env.position)}
            if reason:
                rejected.append({"panel": panel["id"], **record, "reason": reason})
            else:
                panel["layouts"].append(record)
                panel["map_seeds"].append(candidate)
                used.add(key)
                payload = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(),
                           "position": list(env.position), "config": asdict(config)}
                task_hashes[candidate] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                goal = tuple(map(int, np.argwhere(env.pellets)[0]))
                distance, queue = {goal: 0}, deque([goal])
                while queue:
                    p = queue.popleft()
                    for dr, dc in replay_audit.DELTAS:
                        q = p[0] + dr, p[1] + dc
                        if 0 <= q[0] < config.size and 0 <= q[1] < config.size and not env.walls[q] and q not in distance:
                            distance[q] = distance[p] + 1
                            queue.append(q)
                planner[candidate] = distance[env.position]
            candidate += 1
        panels.append(panel)
    assert panels == selection["panels"] and rejected == selection["collision_skips"]
    assert len(used) == selection["unique_selected_layouts"] == spec["count"] * spec["maps_per_panel"]
    assert selection["excluded_layout_counts"] == {k: len(v) for k, v in excluded.items()}
    return selection, task_hashes, planner


def validate_models(study, archive, result):
    protocol = result["protocol"]
    banks, seeds, schedule = protocol["bank_ids"], protocol["seeds"], protocol["checkpoints"]
    final = max(schedule)
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: Path(p).resolve() if Path(p).is_absolute() else (root / p).resolve()
    conditions = [r["id"] for r in protocol["conditions"]]
    snapshot_rows = read_json(study / "snapshots.json")
    expected_snapshots = {(b, "within_map_uniform", s, cp) for b in banks for s in seeds for cp in schedule}
    expected_snapshots |= {(b, "collected_unique", s, 30000) for b in banks for s in seeds}
    assert {(r["bank_id"], r["condition"], r["seed"], r["checkpoint"]) for r in snapshot_rows} == expected_snapshots
    assert len(snapshot_rows) == len(expected_snapshots)
    files = {f"models/bank{b}_{c}_seed{s}_update{cp}.pt" for b, c, s, cp in expected_snapshots}
    assert {p.relative_to(study).as_posix() for p in (study / "models").glob("*.pt")} == files
    initial, states, agents = {}, {}, {}
    for bank in banks:
        for seed in seeds:
            path = shared.checked_archive_file(archive, f"models/bank{bank}_collected_unique_seed{seed}_update0.pt")
            state = torch.load(path, map_location="cpu", weights_only=True)
            assert state["optimizer_updates"] == 0 and state["observation_size"] == 92
            assert replay_audit.tensor_hash(state["online"]) == state["parameter_hash"]
            assert replay_audit.tensor_hash(state["target"]) == state["target_parameter_hash"] == state["parameter_hash"]
            initial[bank, seed] = state
    for row in snapshot_rows:
        bank, condition, seed, checkpoint = row["bank_id"], row["condition"], row["seed"], row["checkpoint"]
        name = f"models/bank{bank}_{condition}_seed{seed}_update{checkpoint}.pt"
        path = study / name
        assert row["saved"] == name and common.sha(path) == row["sha256"]
        assert row["historical"] == (condition == "collected_unique")
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["optimizer_updates"] == checkpoint and state["observation_size"] == 92
        assert replay_audit.tensor_hash(state["online"]) == state["parameter_hash"] == row["online_hash"]
        assert replay_audit.tensor_hash(state["target"]) == state["target_parameter_hash"] == row["target_hash"]
        if condition == "collected_unique":
            assert common.sha(path) == common.sha(shared.checked_archive_file(archive, name))
        elif checkpoint == 0:
            assert state["parameter_hash"] == initial[bank, seed]["parameter_hash"]
            assert state["target_parameter_hash"] == initial[bank, seed]["target_parameter_hash"]
        if condition == "collected_unique" or checkpoint == final:
            states[bank, condition, seed] = state
            agents[bank, condition, seed] = bank_audit.bare_network(state["online"])
    records = read_json(study / "models.json")
    assert records == result["provenance"]["models"]
    expected_models = {(b, c, s) for b in banks for c in conditions for s in seeds}
    assert {(r["bank_id"], r["condition"], r["seed"]) for r in records} == expected_models and len(records) == len(expected_models)
    for row in records:
        bank, condition, seed = row["bank_id"], row["condition"], row["seed"]
        state = states[bank, condition, seed]
        assert row["checkpoint"] == state["optimizer_updates"] == (30000 if condition == "collected_unique" else final)
        assert row["initial_online_hash"] == initial[bank, seed]["parameter_hash"]
        assert row["initial_target_hash"] == initial[bank, seed]["target_parameter_hash"]
        source = archive / row["saved"] if condition == "collected_unique" else study / row["saved"]
        assert resolve(row["source"]) == source.resolve()
        file_hash = common.sha(study / row["saved"])
        assert row["source_before"] == row["source_after"] == row["file_before"] == row["file_after"] == file_hash == common.sha(source)
        assert row["online_before"] == row["online_after"] == state["parameter_hash"]
        assert row["target_before"] == row["target_after"] == state["target_parameter_hash"]
        assert row["unchanged_during_evaluation"]
        assert {r["panel"] for r in row["panel_checks"]} == {p["id"] for p in result["panels"]}
        assert len(row["panel_checks"]) == len(result["panels"])
        for check in row["panel_checks"]:
            assert check["unchanged"]
            for field in ("source", "file", "online", "target"):
                assert check[field + "_before"] == check[field + "_after"] == row[field + "_before"]
    return states, agents, initial, snapshot_rows


def validate_losses(study, result):
    protocol = result["protocol"]
    final = max(protocol["checkpoints"])
    checkpoints = sorted(set(range(100, final + 1, 100)) | {cp for cp in protocol["checkpoints"] if cp})
    rows = csv_rows(study / "losses.csv")
    expected = {(b, s, cp) for b in protocol["bank_ids"] for s in protocol["seeds"] for cp in checkpoints}
    key = lambda r: (int(r["bank_id"]), int(r["seed"]), int(r["checkpoint"]))
    assert {key(r) for r in rows} == expected and len(rows) == len(expected)
    for bank in protocol["bank_ids"]:
        for seed in protocol["seeds"]:
            group = [r for r in rows if int(r["bank_id"]) == bank and int(r["seed"]) == seed]
            assert [int(r["checkpoint"]) for r in group] == checkpoints
            previous = 0
            for row in group:
                assert row["condition"] == "within_map_uniform"
                checkpoint = int(row["checkpoint"])
                assert int(row["updates_in_window"]) == checkpoint - previous
                assert int(row["training_examples"]) == checkpoint * 64
                for field in ("mean_loss", "last_loss"):
                    assert np.isfinite(float(row[field])) and float(row[field]) >= 0
                previous = checkpoint
    assert {(r["bank_id"], r["checkpoint"]) for r in result["loss_aggregate"]} == {(b, cp) for b in protocol["bank_ids"] for cp in checkpoints}
    assert len(result["loss_aggregate"]) == len(protocol["bank_ids"]) * len(checkpoints)
    for row in result["loss_aggregate"]:
        group = [r for r in rows if (int(r["bank_id"]), int(r["checkpoint"])) == (row["bank_id"], row["checkpoint"])]
        assert row["condition"] == "within_map_uniform" and row["seeds"] == len(group) == len(protocol["seeds"])
        assert all(row["updates_in_window"] == int(r["updates_in_window"]) for r in group)
        close(row["mean_loss"], np.mean([float(r["mean_loss"]) for r in group]))
    assert result["provenance"]["loss_integrity"] == {"expected_windows": len(expected), "actual_windows": len(rows), "complete": True, "finite": True}
    return len(rows)


def validate_exposure(study, result, data, transitions, supports, counts):
    exposure = read_json(study / "exposure.json")
    assert exposure == result["exposure"]
    key = lambda r: (r["bank_id"], r["condition"], r["seed"])
    expected = set(counts)
    assert {key(r) for r in exposure["summaries"]} == expected and len(exposure["summaries"]) == len(expected)
    assert {key(r) + (r["map_seed"],) for r in exposure["per_map"]} == {k + (m,) for k in expected for m in range(300000, 300256)}
    assert len(exposure["per_map"]) == len(expected) * 256
    buckets = {"all": "all", "1_8": "1-8", "9_16": "9-16", "17_24": "17-24", "25_32": "25-32"}
    assert {key(r) + (r["bucket"],) for r in exposure["per_clock"]} == {k + (b,) for k in expected for b in buckets}
    assert len(exposure["per_clock"]) == len(expected) * len(buckets)
    for summary in exposure["summaries"]:
        group_key = key(summary)
        bank, condition, seed = group_key
        actual = shared.independent_exposure(data, transitions, supports[bank, condition], counts[group_key])
        total = actual["presentations"]
        assert summary["source"] == ("archived_baseline" if condition == "collected_unique" else "new_treatment")
        assert summary["updates"] == result["sampling"][str(bank)][condition][str(seed)]["updates"]
        assert summary["presentations"] == total and summary["unique_states_sampled"] == actual["unique_sampled_states"]
        close(summary["map_fraction_min"], actual["map_min"] / total)
        close(summary["map_fraction_max"], actual["map_max"] / total)
        close(summary["map_fraction_cv"], actual["map_cv"])
        state_counts = counts[group_key][supports[bank, condition]]
        distribution = summary["state_count_distribution"]
        assert distribution["denominator"] == "all supported current states, including zero presentations"
        assert distribution["states"] == len(state_counts) and distribution["zero_states"] == int((state_counts == 0).sum())
        for field, value in (("minimum", state_counts.min()), ("maximum", state_counts.max()), ("mean", state_counts.mean()),
                             ("median", np.median(state_counts)), ("q95", np.quantile(state_counts, .95))):
            close(distribution[field], value)
        assert summary["outside_support_direct_samples"] == 0
        for field in ("nonterminal_queries", "outside_support_queries", "outside_support_fraction"):
            close(summary["successor_queries"][field], actual[field])
        def check_category(row, calculated):
            assert row["presentations"] == calculated["presentations"]
            assert row["supported_states"] == calculated["support_states"]
            assert row["unique_states_sampled"] == calculated["unique_sampled_states"]
            close(row["presentation_fraction"], calculated["presentations"] / total)
        assert set(summary["category_exposure"]) == {"winnable", "impossible", "goal_near", "goal_far"}
        for category, row in summary["category_exposure"].items():
            check_category(row, actual[category])
        for row in (r for r in exposure["per_clock"] if key(r) == group_key):
            check_category(row, actual["by_time_bucket"][buckets[row["bucket"]]])
        for row in (r for r in exposure["per_map"] if key(r) == group_key):
            mask = data["map_seeds"] == row["map_seed"]
            check_category(row, {"presentations": int(counts[group_key][mask].sum()), "support_states": int(np.count_nonzero(mask[supports[bank, condition]])),
                                 "unique_sampled_states": int(np.count_nonzero(counts[group_key][mask]))})
        for row in [r for r in exposure["per_clock"] + exposure["per_map"] if key(r) == group_key]:
            assert row["source"] == summary["source"] and row["updates"] == summary["updates"]
    return len(exposure["per_map"]), len(exposure["per_clock"])


def validate_sampling(study, archive, result, data, supports):
    sampling = read_json(study / "sampling.json")
    assert sampling == result["sampling"]
    old_sampling = read_json(shared.checked_archive_file(archive, "sampling.json"))
    saved = {kind: dict(np.load(study / filename, allow_pickle=False)) for kind, filename in
             (("global", "sample_counts.npz"), ("local", "local_sample_counts.npz"), ("map", "map_counts.npz"))}
    old = {kind: dict(np.load(shared.checked_archive_file(archive, filename), allow_pickle=False)) for kind, filename in
           (("global", "sample_counts.npz"), ("local", "local_sample_counts.npz"))}
    protocol = result["protocol"]
    banks, seeds, updates = protocol["bank_ids"], protocol["seeds"], max(protocol["checkpoints"])
    expected = {f"bank{b}_{c}_seed{s}" for b in banks for c in CONDITIONS for s in seeds}
    assert all(set(arrays) == expected for arrays in saved.values())
    assert set(sampling) == set(map(str, banks))
    provenance = result["provenance"]
    reconstruction = provenance["baseline_sampling_reconstruction"]
    paired = provenance["paired_map_sampling_consistency"]
    key = lambda r: (r["bank_id"], r["seed"])
    fits = {(b, s) for b in banks for s in seeds}
    assert {key(r) for r in reconstruction} == {key(r) for r in paired} == fits
    assert len(reconstruction) == len(paired) == len(fits)
    counts = {}
    fields = {"local_batch_index_sha256": "local", "global_batch_index_sha256": "global", "map_index_sha256": "map"}
    for bank in banks:
        assert set(sampling[str(bank)]) == set(CONDITIONS)
        for condition in CONDITIONS:
            assert set(sampling[str(bank)][condition]) == set(map(str, seeds))
        for seed in seeds:
            baseline, treatment, prefix = reconstruct_streams(supports[bank, CONDITIONS[0]], supports[bank, CONDITIONS[1]], data["map_seeds"], seed, updates)
            for condition, actual in ((CONDITIONS[0], baseline), (CONDITIONS[1], treatment)):
                record = sampling[str(bank)][condition][str(seed)]
                name = f"bank{bank}_{condition}_seed{seed}"
                support = supports[bank, condition]
                for kind in ("global", "local", "map"):
                    value = saved[kind][name]
                    assert value.dtype == np.uint32 and np.array_equal(value, actual[kind + "_counts"]), (bank, condition, seed, kind)
                for field, kind in fields.items():
                    assert record[field] == actual["digests"][kind], (bank, condition, seed, field)
                assert record["rng_seed_tuple"] == [seed, 66301]
                assert record["map_ids"] == list(range(300000, 300256))
                assert record["support_states"] == len(support) and record["outside_support_direct_samples"] == 0
                assert record["unique_states_sampled"] == int(np.count_nonzero(actual["global_counts"]))
                assert record["examples_seen"] == int(actual["global_counts"].sum())
                assert np.array_equal(actual["global_counts"][support], actual["local_counts"])
                assert int(actual["global_counts"].sum()) == int(actual["global_counts"][support].sum())
                if condition == CONDITIONS[0]:
                    original = old_sampling[str(bank)][condition][str(seed)]
                    assert all(record[k] == v for k, v in original.items()), "archived baseline metadata changed"
                    assert record["updates"] == 30000 and record["new_updates"] == 0 and record["historical"] is True
                    assert np.array_equal(saved["global"][name], old["global"][name])
                    assert np.array_equal(saved["local"][name], old["local"][name])
                    compared = record["compared_prefix"]
                    assert compared["updates"] == updates
                    for field, kind in fields.items():
                        assert compared[field] == prefix[kind]
                    for kind in ("local", "map"):
                        bank_audit.check_array(treatment[kind + "_counts"], compared["count_arrays"][kind])
                else:
                    assert record["updates"] == record["new_updates"] == updates and record["historical"] is False
                counts[bank, condition, seed] = actual["global_counts"]
            reconstructed = next(r for r in reconstruction if key(r) == (bank, seed))
            assert reconstructed["reconstructed_updates"] == 30000 and reconstructed["verified"]
            for field in ("local_digest_identical", "global_digest_identical", "local_counts_identical", "global_counts_identical"):
                assert reconstructed[field]
            assert reconstructed["map_index_sha256"] == baseline["digests"]["map"]
            assert set(reconstructed["count_arrays"]) == {"global", "local", "map"}
            for kind in ("global", "local", "map"):
                bank_audit.check_array(baseline[kind + "_counts"], reconstructed["count_arrays"][kind])
            agreement = next(r for r in paired if key(r) == (bank, seed))
            assert agreement["compared_updates"] == updates and agreement["baseline_updates"] == 30000
            assert all(agreement[k] for k in ("map_digest_identical", "map_counts_identical", "local_digest_identical", "local_counts_identical", "complete"))
    return counts


def validate_provenance(study, archives, result, data, transitions, supports, quota_rows, initial, snapshots, counts, rows, refs):
    protocol, provenance = result["protocol"], result["provenance"]
    assert read_json(study / "provenance.json") == provenance
    assert set(provenance["archives"]) == set(archives)
    source_records = {}
    for name, directory in archives.items():
        record = provenance["archives"][name]
        manifest = read_json(directory / "manifest.json")
        assert common.sha(study / f"{name}_manifest.json") == common.sha(directory / "manifest.json") == record["files"]["manifest.json"]
        for filename, digest in record["files"].items():
            assert common.sha(directory / filename) == digest
            if filename != "manifest.json":
                assert manifest["files"][filename] == digest
            source_records[name, filename] = digest
        metadata_name = "panels.json" if name in ("panel_evaluation", "bank_replication", "map_replay") else "dataset_metadata.json"
        assert common.sha(study / f"{name}_{metadata_name}") == common.sha(directory / metadata_name)
        if name in ("coverage", "bank_replication"):
            for filename in ("protocol.json", "protocol.md"):
                assert common.sha(study / f"{name}_{filename}") == common.sha(directory / filename)
    assert {(r["archive"], r["file"]) for r in provenance["source_integrity"]} == set(source_records)
    assert len(provenance["source_integrity"]) == len(source_records)
    for row in provenance["source_integrity"]:
        assert row["before"] == row["after"] == source_records[row["archive"], row["file"]] and row["unchanged"]
    assert provenance["training_arrays_identical"] and provenance["transition_arrays_identical"]
    assert provenance["all_supports_frozen_before_training"] and provenance["all_treatment_fits_finished_before_evaluation"]
    assert provenance["snapshot_integrity"] == {"expected": len(snapshots), "actual": len(snapshots), "complete": True, "unchanged": True}
    banks = read_json(study / "banks.json")
    assert banks == result["banks"] and [b["bank_id"] for b in banks] == protocol["bank_ids"]
    old_banks = read_json(shared.checked_archive_file(archives["bank_replication"], "banks.json"))
    # Every position has 32 clock rows. Winnability first becomes true at the
    # shortest physical distance, independently validated by audit_transitions.
    distance = np.min(np.where(data["winnable"], data["remaining"], 33).reshape(-1, 32), axis=1)
    near = np.repeat(distance <= 2, 32)
    for bank in banks:
        bank_id = bank["bank_id"]
        original, replacement = supports[bank_id, CONDITIONS[0]], supports[bank_id, CONDITIONS[1]]
        old = next(r for r in old_banks if r["bank_id"] == bank_id)
        assert bank["collection"] == {**old["collection"], "source": "inherited_archive_not_new_collection"}
        assert bank["status"] == "complete" and bank["support_size"] == len(original) == len(replacement)
        overlap, union = len(np.intersect1d(original, replacement)), len(np.union1d(original, replacement))
        assert bank["intersection"]["states"] == overlap and bank["intersection"]["union_states"] == union
        close(bank["intersection"]["fraction_of_each"], overlap / len(original))
        close(bank["intersection"]["jaccard"], overlap / union)
        assert bank["uniform_rng"] == f"SeedSequence([{bank_id},map_seed,105301])"
        assert bank["quotas"]["exact"] and len(bank["quotas"]["by_map"]) == 256
        for quota, expected in zip(bank["quotas"]["by_map"], quota_rows[bank_id]):
            m = expected["map_seed"]
            assert quota["map_seed"] == m and quota["collected_states"] == quota["treatment_states"] == expected["states"]
            assert quota["identical"] and quota["candidate_states"] == 640 and quota["rng_seed_tuple"] == [bank_id, m, 105301]
            candidates = np.flatnonzero(data["map_seeds"] == m)
            chosen = replacement[data["map_seeds"][replacement] == m]
            assert quota["candidate_row_sha256"] == hashlib.sha256(candidates.astype("<i8").tobytes()).hexdigest()
            assert quota["selected_row_sha256"] == hashlib.sha256(chosen.astype("<i4").tobytes()).hexdigest()
        assert {r["condition"] for r in bank["coverage"]["per_condition"]} == set(CONDITIONS)
        assert len(bank["coverage"]["per_condition"]) == len(CONDITIONS)
        for condition in CONDITIONS:
            bank_audit.check_array(supports[bank_id, condition], bank["arrays"][condition])
            current = next(r for r in bank["coverage"]["per_condition"] if r["condition"] == condition)
            bank_audit.check_composition(current, data, transitions, supports[bank_id, condition], near)
            if condition == CONDITIONS[0]:
                old_coverage = next(r for r in old["coverage"]["per_condition"] if r["condition"] == condition)
                assert current == old_coverage
    assert {r["bank_id"] for r in provenance["support_integrity"]} == set(protocol["bank_ids"])
    assert len(provenance["support_integrity"]) == len(protocol["bank_ids"])
    for row in provenance["support_integrity"]:
        assert row["unchanged"] and row["read_only"] and row["before"] == row["after"]
        assert set(row["before"]) == set(CONDITIONS)
        for condition, spec in row["before"].items():
            bank_audit.check_array(supports[row["bank_id"], condition], spec)
    expected_init = {(b, s) for b in protocol["bank_ids"] for s in protocol["seeds"]}
    assert set(provenance["prior_initial_models"]) == {f"{b}:{s}" for b, s in expected_init}
    for bank, seed in expected_init:
        row = provenance["prior_initial_models"][f"{bank}:{seed}"]
        assert row["online_hash"] == initial[bank, seed]["parameter_hash"] and row["target_hash"] == initial[bank, seed]["target_parameter_hash"]
        assert row["sha256"] == common.sha(archives["bank_replication"] / f"models/bank{bank}_collected_unique_seed{seed}_update0.pt")
    assert {(r["bank_id"], r["seed"]) for r in provenance["initialization_consistency"]} == expected_init
    assert len(provenance["initialization_consistency"]) == len(expected_init)
    for row in provenance["initialization_consistency"]:
        assert row["models"] == 2 and row["online_identical"] and row["target_identical"]
    assert {(r["bank_id"], r["condition"], r["seed"]) for r in provenance["count_integrity"]} == set(counts)
    assert len(provenance["count_integrity"]) == len(counts)
    for row in provenance["count_integrity"]:
        assert all(row[k] for k in ("sum_matches", "local_matches", "map_sum_matches", "on_support"))
    cells = provenance["expected_cells"]
    assert {r["bank_id"] for r in cells} == set(protocol["bank_ids"]) and len(cells) == len(protocol["bank_ids"])
    for row in cells:
        assert row["complete"]
        assert row["expected_learner_cells"] == row["complete_learner_cells"] == row["unique_complete_learner_cells"] == len(rows) // len(protocol["bank_ids"])
        assert row["expected_reference_cells"] == row["complete_reference_cells"] == row["unique_complete_reference_cells"] == len(refs)
    return len(source_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/within_map/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true")
    archive_names = ("supervised", "fixed_targets", "coverage", "equal_support", "panel_evaluation", "bank_replication", "map_replay")
    for name in archive_names:
        parser.add_argument("--" + name.replace("_", "-") + "-study", type=Path)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Assertions must be enabled; do not use python -O.")
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p.resolve() if p.is_absolute() else (root / p).resolve()
    study = resolve(args.study)
    archives = {name: resolve(getattr(args, name + "_study") or Path(f"experiments/{name}/pilot_v1")) for name in archive_names}
    sys.path.insert(0, str(study / "source"))
    from q6.world import CollectionWorld, WorldConfig
    torch.set_num_threads(1)
    started = time.monotonic()
    result, protocol = read_json(study / "results.json"), read_json(study / "protocol.json")
    smoke = bool(protocol["smoke"] or protocol["deviations"])
    assert not smoke or args.allow_smoke, "Smoke/deviating artifacts require explicit --allow-smoke."
    assert protocol["id"] == "within-map-v1" and result["run"]["status"] == "complete"
    assert result["artifacts"]["report"] == "docs/experiments/within_map_results_v1.md"
    assert protocol["bank_ids"] == [1, 2, 3]
    assert [r["id"] for r in protocol["conditions"]] == ["collected_unique", "within_map_uniform"]
    assert protocol["primary_comparison"] == "within_map_minus_collected"
    final = max(protocol["checkpoints"])
    assert protocol["evaluation_checkpoints"] == {"collected_unique": 30000, "within_map_uniform": final}
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2] and protocol["git"]["dirty"] is False
        assert protocol["runtime"]["python"].startswith("3.12.")
        assert protocol["runtime"]["torch"].split("+")[0] == "2.8.0" and protocol["runtime"]["numpy"] == "2.0.2"
        assert protocol["panel_selection"] == {"count": 8, "maps_per_panel": 64, "start": 1060000, "stride": 1000}
        assert protocol["checkpoints"] == [0, 1000, 3000, 10000, 30000]
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig()) and protocol["rule_visibility"] == "observed"
    assert all(protocol["collection"][k] == 0 for k in ("new_steps", "new_episodes"))
    assert protocol["evaluation"]["final_only"] and protocol["evaluation"]["after_all_treatment_fits"]
    assert protocol["evaluation"]["greedy_repetitions"] == 1 and protocol["evaluation"]["epsilon_0_1_repetitions"] == 2
    assert protocol["evaluation"]["optimal_q_atol"] == 1e-6 and protocol["evaluation"]["optimal_q_rtol"] == 0
    assert protocol["sampling"]["rng"] == "SeedSequence([seed,66301])"
    assert protocol["sampling"]["implementation"] == "unchanged coverage.SupportSampler"
    assert protocol["sampling"]["paired_map_schedule_within_bank"] and protocol["sampling"]["paired_local_schedule_within_bank"]
    assert protocol["sampling"]["full_bank_detached_successors"]
    assert protocol["support_selection"]["rng"] == "SeedSequence([bank_id,map_seed,105301])"
    assert protocol["support_selection"]["same_map_quotas"] and protocol["support_selection"]["shared_across_learner_seeds"]
    assert protocol["support_selection"]["clock_or_target_selection"] is False
    assert protocol["collection"]["new_support_draws"] == protocol["collection"]["per_map_subset_draws"] == 768
    assert protocol["collection"]["replacement_supports"] == 3
    for key, value in {"learning_rate": .001, "gradient_norm_cap": 5., "gamma": .97, "target_tau": .01, "batch_size": 64}.items():
        close(protocol["optimizer"][key], value)
    assert protocol["dataset"]["fresh_state_enumeration"] is False
    report = {"study": str(args.study), "smoke_validation_only": smoke,
              "manifest_files": common.audit_manifest(study, protocol, result)}
    data, arrays, metadata, transitions, supports, quota_rows = validate_inputs(study, archives, protocol)
    transition_data = dict(arrays)
    for key, value in data.items():
        transition_data["heldout_" + key] = np.empty((0,) + value.shape[1:], dtype=value.dtype)
    transition_report = common.audit_transitions(transition_data, {**metadata, "heldout": []}, transitions, config, CollectionWorld)
    selection, task_hashes, planner = reconstruct_panels(study, archives, protocol, result, metadata, CollectionWorld, config)
    states, agents, initial, snapshots = validate_models(study, archives["bank_replication"], result)
    counts = validate_sampling(study, archives["bank_replication"], result, data, supports)
    exposure_maps, exposure_clocks = validate_exposure(study, result, data, transitions, supports, counts)
    loss_rows = validate_losses(study, result)
    rows, refs, paired = shared.validate_episode_tables(study, result, states, task_hashes, planner)
    shared.validate_thresholds(result, eligible=not smoke)
    source_inputs = validate_provenance(study, archives, result, data, transitions, supports, quota_rows, initial, snapshots, counts, rows, refs)
    replay_count, replay_steps = shared.validate_replays(study, result, rows, refs, agents, CollectionWorld, config,
                                                skip_forward=args.skip_forward_inference)
    run, budget = result["run"], protocol["budget"]
    fits = {(b, s) for b in protocol["bank_ids"] for s in protocol["seeds"]}
    assert run["fits"] == run["frozen_baselines"] == len(fits) and len(states) == 2 * len(fits)
    assert run["banks"] == len(protocol["bank_ids"]) and run["model_parameter_count"] == 20420
    assert run["train_updates"] == len(fits) * final == budget["maximum_updates"] and budget["updates_per_fit"] == final
    assert run["training_examples"] == run["train_updates"] * 64 and run["historical_baseline_updates"] == len(fits) * 30000
    assert all(run[k] == 0 for k in ("baseline_new_updates", "collection_episodes", "collection_steps"))
    assert run["support_draws"] == run["per_map_subset_draws"] == 768 and run["replacement_supports"] == 3
    assert run["baseline_reconstructed_updates"] == len(fits) * 30000
    assert all(budget[k] == 0 for k in ("new_baseline_updates", "maximum_collection_steps", "maximum_collection_episodes"))
    assert run["learner_episodes"] == len(rows) and run["reference_episodes"] == run["unique_reference_episodes"] == len(refs)
    assert run["panels"] == len(selection["panels"])
    assert 0 < run["wall_seconds"] <= budget["admission_seconds"] <= 1200
    assert 0 < run["peak_rss_bytes"] <= budget["peak_process_rss_bytes"] == 4 * 1024 ** 3
    assert run["resource_checks"] > run["train_updates"] + run["baseline_reconstructed_updates"] + len(rows) + len(refs)
    assert run["resource_limits"] == {"seconds": budget["admission_seconds"], "peak_process_rss_bytes": budget["peak_process_rss_bytes"], "torch_threads": 1}
    assert protocol["runtime"]["torch_threads"] == 1 and budget["all_phases_included"]
    assert run["interpretation"] == ("smoke_or_deviation_descriptive_only" if smoke else "within_map_descriptive_only") and run["stop_reason"] is None
    completed = [r for r in run["progress"] if r.get("training_complete")]
    assert {(r["bank_id"], r["seed"]) for r in completed} == fits and len(completed) == len(fits)
    assert all(r["updates"] == final for r in completed)
    progress = [r for r in run["progress"] if r.get("snapshot_saved")]
    assert {(r["bank_id"], r["seed"], r["checkpoint"]) for r in progress} == {(b, s, cp) for b, s in fits for cp in protocol["checkpoints"]}
    assert len(progress) == len(fits) * len(protocol["checkpoints"]) and all(r["condition"] == "within_map_uniform" for r in progress)
    evaluated = [r for r in run["progress"] if r.get("evaluation_complete")]
    expected_evaluated = {(b, c, s, p["id"]) for b, s in fits for c in ("collected_unique", "within_map_uniform") for p in result["panels"]}
    assert {(r["bank_id"], r["condition"], r["seed"], r["panel"]) for r in evaluated} == expected_evaluated
    assert len(evaluated) == len(expected_evaluated)
    assert len(run["progress"]) == len(progress) + len(completed) + len(evaluated)
    report.update(training_states=len(data["observations"]), reused_support_sizes={str(b): len(supports[b, CONDITIONS[0]]) for b in protocol["bank_ids"]}, per_map_quota_draws_checked=768,
        selected_layouts=len(task_hashes), rejected_candidates=len(selection["collision_skips"]), transition_validation=transition_report,
        archive_input_hashes_checked=source_inputs, inherited_counts_and_controls_identical=True,
        treatment_sampling_streams_checked=3 * len(fits), paired_local_and_map_streams_checked=len(fits), historical_sampler_updates_reconstructed=len(fits) * 30000,
        model_snapshots_checked=len(snapshots), frozen_controls_checked=len(fits), final_models_checked=len(states),
        initial_models_match_archive=True, frozen_evaluation_identity_unchanged=True,
        exposure_map_rows_checked=exposure_maps, exposure_clock_rows_checked=exposure_clocks,
        loss_rows_checked=loss_rows, learner_episode_rows=len(rows), reference_episode_rows=len(refs),
        paired_layout_rows=len(paired["per_layout"]), paired_seed_rows=len(paired["per_seed"]), paired_aggregate_rows=len(paired["aggregate"]),
        recorded_replays_checked=replay_count, recorded_steps_checked=replay_steps,
        replay_forward_inference_checked=not args.skip_forward_inference, forward_inference_checked=not args.skip_forward_inference,
        full_policy_evaluation_repeated=False, new_training_updates=0, new_collection_steps=0,
        bank_effects=result["robustness"]["bank_effects"], audit_seconds=time.monotonic() - started)
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
