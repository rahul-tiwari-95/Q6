"""Independent audit of exact targets derived only from recorded transitions.

Reconstructs logged edges and sampler exposure, verifies frozen controls and
raw outcomes, and checks preselected recordings without fitting a model or
repeating the full learned-policy evaluation. Portable mode skips only neural
forward regeneration for saved recordings and final support predictions.
"""
from __future__ import annotations

import argparse
import csv
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
import audit_within_map as within_audit
import audit_panel_evaluation as replay_audit
from audit_fixed_targets_study import close, csv_rows, read_json

CONDITIONS = ("constrained_bootstrap", "logged_graph")


def reconstruct_recorded_edges(path, data, transitions, support):
    """Read the archived evidence, preserving exact deterministic edge identity."""
    n = len(data["observations"])
    arrays = {"observed": np.zeros((n, 4), bool), "rewards": np.full((n, 4), np.nan, np.float64),
              "ends": np.ones((n, 4), bool), "terminated": np.zeros((n, 4), bool),
              "truncated": np.zeros((n, 4), bool), "successor_indices": np.full((n, 4), -1, np.int32),
              "occurrences": np.zeros((n, 4), np.uint32)}
    allowed = np.zeros(n, bool)
    allowed[support] = True
    flags = {}
    with path.open(newline="") as handle:
        for item in csv.DictReader(handle):
            row, action, following = (int(item[k]) for k in ("current_row", "action", "next_row"))
            terminated, truncated = int(item["terminated"]), int(item["truncated"])
            assert 0 <= row < n and allowed[row] and 0 <= action < 4
            assert terminated in (0, 1) and truncated in (0, 1) and not (terminated and truncated)
            reward, ended = float(item["reward"]), bool(terminated or truncated)
            assert np.isfinite(reward) and int(item["map_seed"]) == int(data["map_seeds"][row])
            assert int(item["remaining"]) == int(data["remaining"][row])
            assert reward == float(transitions["rewards"][row, action])
            assert ended == bool(transitions["ends"][row, action])
            assert following == int(transitions["successor_indices"][row, action])
            if ended:
                assert following == -1
                if truncated:
                    assert data["remaining"][row] == 1
            else:
                assert 0 <= following < n
                assert data["map_seeds"][following] == data["map_seeds"][row]
                assert data["remaining"][following] == data["remaining"][row] - 1
            if arrays["observed"][row, action]:
                assert arrays["rewards"][row, action] == reward
                assert arrays["ends"][row, action] == ended and arrays["successor_indices"][row, action] == following
                assert flags[row, action] == (terminated, truncated)
            else:
                arrays["observed"][row, action] = True
                arrays["rewards"][row, action] = reward
                arrays["ends"][row, action] = ended
                arrays["terminated"][row, action] = bool(terminated)
                arrays["truncated"][row, action] = bool(truncated)
                arrays["successor_indices"][row, action] = following
                flags[row, action] = terminated, truncated
            arrays["occurrences"][row, action] += 1
    assert np.array_equal(np.flatnonzero(arrays["observed"].any(1)), support)
    assert np.array_equal(arrays["occurrences"] > 0, arrays["observed"])
    assert np.isnan(arrays["rewards"][~arrays["observed"]]).all()
    assert arrays["ends"][~arrays["observed"]].all()
    assert not arrays["terminated"][~arrays["observed"]].any() and not arrays["truncated"][~arrays["observed"]].any()
    assert (arrays["successor_indices"][~arrays["observed"]] == -1).all()
    return arrays


def recorded_exposure(recorded, support, counts):
    """Membership, collector frequency, and replay presentations are distinct."""
    mask = recorded["observed"]
    action_counts = mask.sum(1)
    live = mask & ~recorded["ends"]
    weights = counts.astype(np.uint64)
    supported = np.zeros(len(mask), bool)
    supported[support] = True
    outside = live & ~supported[np.maximum(recorded["successor_indices"], 0)]
    assert (action_counts[support] >= 1).all() and (action_counts[support] <= 4).all()
    return {"supported_states": len(support), "observed_edges": int(mask.sum()),
        "logged_occurrences": int(recorded["occurrences"].sum()),
        "states_by_action_count": {str(k): int((action_counts[support] == k).sum()) for k in range(1, 5)},
        "edges_by_action": mask.sum(0).astype(int).tolist(),
        "terminal_edges": int((mask & recorded["ends"]).sum()), "nonterminal_edges": int(live.sum()),
        "outside_support_edges": int(outside.sum()),
        "state_presentations": int(weights.sum()),
        "supervised_action_presentations": int(np.dot(weights, action_counts.astype(np.uint64))),
        "nonterminal_action_presentations": int(np.dot(weights, live.sum(1).astype(np.uint64))),
        "outside_support_successor_presentations": int(np.dot(weights, outside.sum(1).astype(np.uint64))),
        "action_presentations": (weights[:, None] * mask).sum(0).astype(np.uint64).tolist()}


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
    supports = {}
    for bank, size in zip((1, 2, 3), (59839, 59984, 59838)):
        original = old[f"collected_unique_bank{bank}"]
        assert len(original) == size and original.dtype == np.int32 and np.all(np.diff(original) > 0)
        for condition in CONDITIONS:
            actual = saved[f"{condition}_bank{bank}"]
            assert actual.dtype == np.int32 and np.array_equal(actual, original)
            supports[bank, condition] = actual
    for name in ("q6/world.py", "q6/learning.py", "q6/fixed_targets.py", "q6/competence.py", "q6/supervised.py", "q6/optimal.py"):
        for archive in (archives["coverage"], archives["bank_replication"]):
            old_protocol = read_json(shared.checked_archive_file(archive, "protocol.json"))
            assert protocol["source_sha256"][name] == old_protocol["source_sha256"][name]
            assert common.sha(shared.checked_archive_file(archive, "source/" + name)) == old_protocol["source_sha256"][name]
    return data, arrays, metadata, transitions, supports


def validate_membership(summary, table, support):
    actual = recorded_exposure(table, support, np.zeros(len(table["observed"]), np.uint32))
    assert summary["supported_states"] == actual["supported_states"]
    assert summary["recorded_edges"] == actual["observed_edges"]
    assert summary["all_action_edges"] == 4 * len(support)
    close(summary["observed_action_fraction"], actual["observed_edges"] / (4 * len(support)))
    assert summary["collector_steps"] == actual["logged_occurrences"]
    assert summary["duplicate_records"] == actual["logged_occurrences"] - actual["observed_edges"]
    assert summary["states_by_observed_actions"] == [{"actions": k, "states": actual["states_by_action_count"][str(k)]} for k in range(1, 5)]
    assert summary["nonterminal_recorded_edges"] == actual["nonterminal_edges"]
    assert summary["terminal_recorded_edges"] == actual["terminal_edges"]
    assert summary["nonterminal_successors_outside_support"] == actual["outside_support_edges"] == 0
    assert summary["nonterminal_successors_in_support"] == actual["nonterminal_edges"]
    assert summary["all_nonterminal_successors_in_support_verified"] is True
    live = table["observed"] & ~table["ends"]
    assert summary["distinct_nonterminal_successor_states"] == len(np.unique(table["successor_indices"][live]))
    assert set(summary["arrays"]) == set(table)
    for name, array in table.items():
        bank_audit.check_array(array, summary["arrays"][name])
    return actual


def reconstruct_panels(study, archives, protocol, result, metadata, World, config):
    excluded = {}
    for name, directory in archives.items():
        if name in ("panel_evaluation", "bank_replication", "map_replay", "within_map", "recorded_actions", "constrained_bootstrap"):
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
    expected_snapshots = {(b, "logged_graph", s, cp) for b in banks for s in seeds for cp in schedule}
    expected_snapshots |= {(b, "constrained_bootstrap", s, 30000) for b in banks for s in seeds}
    assert {(r["bank_id"], r["condition"], r["seed"], r["checkpoint"]) for r in snapshot_rows} == expected_snapshots
    assert len(snapshot_rows) == len(expected_snapshots)
    files = {f"models/bank{b}_{c}_seed{s}_update{cp}.pt" for b, c, s, cp in expected_snapshots}
    assert {p.relative_to(study).as_posix() for p in (study / "models").glob("*.pt")} == files
    initial, states, agents = {}, {}, {}
    for bank in banks:
        for seed in seeds:
            path = shared.checked_archive_file(archive, f"models/bank{bank}_constrained_bootstrap_seed{seed}_update0.pt")
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
        assert row["historical"] == (condition == "constrained_bootstrap")
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["optimizer_updates"] == checkpoint and state["observation_size"] == 92
        assert replay_audit.tensor_hash(state["online"]) == state["parameter_hash"] == row["online_hash"]
        assert replay_audit.tensor_hash(state["target"]) == state["target_parameter_hash"] == row["target_hash"]
        if condition == "constrained_bootstrap":
            assert common.sha(path) == common.sha(shared.checked_archive_file(archive, name))
        elif checkpoint == 0:
            assert state["parameter_hash"] == initial[bank, seed]["parameter_hash"]
            assert state["target_parameter_hash"] == initial[bank, seed]["target_parameter_hash"]
        if condition == "constrained_bootstrap" or checkpoint == final:
            states[bank, condition, seed] = state
            agents[bank, condition, seed] = bank_audit.bare_network(state["online"])
    records = read_json(study / "models.json")
    assert records == result["provenance"]["models"]
    expected_models = {(b, c, s) for b in banks for c in conditions for s in seeds}
    assert {(r["bank_id"], r["condition"], r["seed"]) for r in records} == expected_models and len(records) == len(expected_models)
    for row in records:
        bank, condition, seed = row["bank_id"], row["condition"], row["seed"]
        state = states[bank, condition, seed]
        assert row["checkpoint"] == state["optimizer_updates"] == (30000 if condition == "constrained_bootstrap" else final)
        assert row["initial_online_hash"] == initial[bank, seed]["parameter_hash"]
        assert row["initial_target_hash"] == initial[bank, seed]["target_parameter_hash"]
        source = archive / row["saved"] if condition == "constrained_bootstrap" else study / row["saved"]
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
                assert row["condition"] == "logged_graph"
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
        assert row["condition"] == "logged_graph" and row["seeds"] == len(group) == len(protocol["seeds"])
        assert all(row["updates_in_window"] == int(r["updates_in_window"]) for r in group)
        close(row["mean_loss"], np.mean([float(r["mean_loss"]) for r in group]))
    assert result["provenance"]["loss_integrity"] == {"expected_windows": len(expected), "actual_windows": len(rows), "complete": True, "finite": True}
    return len(rows)


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
            baseline, treatment, prefix = within_audit.reconstruct_streams(supports[bank, CONDITIONS[0]], supports[bank, CONDITIONS[1]], data["map_seeds"], seed, updates)
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
                    assert all(record[k] == v for k, v in original.items() if k not in ("historical", "new_updates")), "archived baseline metadata changed"
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
            assert all(agreement[k] for k in ("map_digest_identical", "map_counts_identical", "local_digest_identical", "local_counts_identical", "global_digest_identical", "global_counts_identical", "complete"))
    return counts


def validate_provenance(study, archives, result, data, transitions, supports, initial, snapshots, counts, rows, refs):
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
        metadata_name = "panels.json" if name in ("panel_evaluation", "bank_replication", "map_replay", "within_map", "recorded_actions", "constrained_bootstrap") else "dataset_metadata.json"
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
    distance = np.min(np.where(data["winnable"], data["remaining"], 33).reshape(-1, 32), axis=1)
    near = np.repeat(distance <= 2, 32)
    for bank in banks:
        bank_id = bank["bank_id"]
        original = supports[bank_id, CONDITIONS[0]]
        old = next(r for r in old_banks if r["bank_id"] == bank_id)
        assert bank["collection"] == {**old["collection"], "source": "inherited_archive_not_new_collection"}
        assert bank["status"] == "complete" and bank["support_size"] == len(original)
        assert bank["intersection"] == {"states": len(original), "union_states": len(original), "fraction_of_each": 1., "jaccard": 1.}
        assert {r["condition"] for r in bank["coverage"]["per_condition"]} == set(CONDITIONS)
        assert len(bank["coverage"]["per_condition"]) == len(CONDITIONS)
        for condition in CONDITIONS:
            bank_audit.check_array(supports[bank_id, condition], bank["arrays"][condition])
            current = next(r for r in bank["coverage"]["per_condition"] if r["condition"] == condition)
            bank_audit.check_composition(current, data, transitions, original, near)
            old_coverage = next(r for r in old["coverage"]["per_condition"] if r["condition"] == "collected_unique")
            assert {k: v for k,v in current.items() if k != "condition"} == {k: v for k,v in old_coverage.items() if k != "condition"}
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
        assert row["sha256"] == common.sha(archives["constrained_bootstrap"] / f"models/bank{bank}_constrained_bootstrap_seed{seed}_update0.pt")
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


def validate_exposure(study, result, data, transitions, supports, counts, recorded):
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
        assert summary["source"] == ("archived_baseline" if condition == CONDITIONS[0] else "new_treatment")
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
        structural = summary["all_action_structural_successor_queries"]
        for field in ("nonterminal_queries", "outside_support_queries", "outside_support_fraction"):
            close(structural[field], actual[field])
        if condition in CONDITIONS:
            observed = recorded_exposure(recorded[bank], supports[bank, condition], counts[group_key])
            live = observed["nonterminal_action_presentations"] if condition == CONDITIONS[0] else 0
            assert observed["outside_support_successor_presentations"] == 0
            assert summary["successor_queries"]["nonterminal_queries"] == live
            assert summary["successor_queries"]["outside_support_queries"] == 0
            if live:
                close(summary["successor_queries"]["outside_support_fraction"], 0.)
            else:
                assert summary["successor_queries"]["outside_support_fraction"] is None
        assert result["sampling"][str(bank)][condition][str(seed)]["successor_queries"] == summary["successor_queries"]
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


def validate_recorded_tables(study, archives, result, data, metadata, transitions, supports):
    coverage = read_json(study / "action_coverage.json")
    assert coverage == result["action_coverage"]
    banks = result["protocol"]["bank_ids"]
    assert [r["bank_id"] for r in coverage["per_bank"]] == banks
    saved = dict(np.load(study / "recorded_transitions.npz", allow_pickle=False))
    fields = {"observed", "rewards", "ends", "terminated", "truncated", "successor_indices", "occurrences"}
    assert set(saved) == {f"bank{b}_{k}" for b in banks for k in fields}
    integrity = result["provenance"]["recorded_table_integrity"]
    assert {r["bank_id"] for r in integrity} == set(banks) and len(integrity) == len(banks)
    root = Path(__file__).resolve().parents[1]
    tables, collector_reports = {}, []
    for summary in coverage["per_bank"]:
        bank = summary["bank_id"]
        archive = archives["bank_replication"]
        filename = f"banks/bank{bank}/collection_steps.csv"
        source = shared.checked_archive_file(archive, filename)
        assert summary["copied"] == filename
        assert (root / summary["source"]).resolve() == source.resolve()
        assert common.sha(study / filename) == common.sha(source) == summary["source_sha256"] == summary["copied_sha256"]
        for filename in (f"banks/bank{bank}/collection_episodes.csv", f"banks/bank{bank}/collection.npz"):
            shared.checked_archive_file(archive, filename)
        # Replay the already recorded collector action/RNG/state sequence as an
        # integrity check, using independently validated world transitions.
        visits, original, collector = bank_audit.reconstruct_bank(archive / f"banks/bank{bank}", bank, data, metadata["train"], transitions, 16)
        assert np.array_equal(original, supports[bank, CONDITIONS[0]])
        table = reconstruct_recorded_edges(source, data, transitions, original)
        assert np.array_equal(table["occurrences"].sum(1), visits)
        for name, value in table.items():
            actual = saved[f"bank{bank}_{name}"]
            assert actual.dtype == value.dtype and actual.shape == value.shape
            assert np.array_equal(actual, value, equal_nan=True), (bank, name)
        expected = validate_membership(summary, table, original)
        assert expected["logged_occurrences"] == collector["collection_steps"]
        assert summary["observed_outcomes_match_integrity_reference"]
        bank_info = next(b for b in result["banks"] if b["bank_id"] == bank)
        assert bank_info["recorded_actions"] == summary and bank_info["identical_state_support"]
        row = next(r for r in integrity if r["bank_id"] == bank)
        assert row["before"] == row["after"] == summary["arrays"]
        assert row["unchanged"] and row["read_only"] and row["optimizer_tables_unchanged"]
        assert row["tensor_before"] == row["tensor_after"] and set(row["tensor_before"]) == fields
        for name, value in table.items():
            bank_audit.check_array(value.astype(np.float32) if name == "rewards" else value, row["tensor_before"][name])
        tables[bank] = table
        collector_reports.append({**collector, "deduplicated_recorded_edges": expected["observed_edges"],
                                  "states_by_action_count": expected["states_by_action_count"]})
    return tables, collector_reports


def independent_logged_graph(table, remaining, map_seeds, support, gamma=.97):
    """Solve only recorded, strictly clock-decreasing edges; no production DP."""
    observed, ended, successors = (table[k] for k in ("observed", "ends", "successor_indices"))
    rewards = table["rewards"]
    assert observed.shape == rewards.shape == (len(remaining), 4)
    assert np.array_equal(np.flatnonzero(observed.any(1)), support)
    assert np.isfinite(rewards[observed]).all() and np.isnan(rewards[~observed]).all()
    assert ((remaining[support] >= 1) & (remaining[support] <= 32)).all()
    targets = np.full_like(rewards, np.nan, dtype=np.float64)
    state_values = np.full(len(remaining), np.nan, np.float64)
    live_edges, terminal_edges = 0, 0
    for clock in range(1, 33):
        rows = support[remaining[support] == clock]
        for action in range(4):
            current = rows[observed[rows, action]]
            terminal = ended[current, action]
            terminal_rows = current[terminal]
            assert (successors[terminal_rows, action] == -1).all()
            targets[terminal_rows, action] = rewards[terminal_rows, action]
            live = current[~terminal]
            following = successors[live, action]
            assert ((following >= 0) & (following < len(remaining))).all()
            assert observed[following].any(1).all()
            assert (remaining[following] == clock - 1).all()
            assert np.array_equal(map_seeds[following], map_seeds[live])
            assert np.isfinite(state_values[following]).all()
            targets[live, action] = rewards[live, action] + gamma * state_values[following]
            live_edges += len(live)
            terminal_edges += len(terminal_rows)
        assert np.isfinite(targets[rows][observed[rows]]).all()
        state_values[rows] = np.max(np.where(observed[rows], targets[rows], -np.inf), axis=1)
    assert np.isfinite(targets[observed]).all() and np.isnan(targets[~observed]).all()
    assert live_edges + terminal_edges == int(observed.sum())
    # Independently test every resulting recurrence as well as induction order.
    source, action = np.nonzero(observed & ~ended)
    following = successors[source, action]
    continuation = np.max(np.where(observed[following], targets[following], -np.inf), axis=1)
    assert np.array_equal(targets[source, action], rewards[source, action] + gamma * continuation)
    return targets, {"observed_edges": live_edges + terminal_edges, "nonterminal_edges": live_edges,
                     "terminal_edges": terminal_edges, "states": len(support)}


def independent_fit(predictions, target, observed):
    """State-weighted and edge-weighted errors are deliberately separate."""
    assert predictions.shape == target.shape == observed.shape
    assert np.isfinite(predictions).all() and observed.any(1).all()
    assert np.isfinite(target[observed]).all() and np.isnan(target[~observed]).all()
    error = np.where(observed, predictions.astype(np.float64) - target, 0.)
    absolute, squared = np.abs(error), error ** 2
    counts = observed.sum(1)
    selected = np.where(observed, predictions, -np.inf).argmax(1)
    target_max = np.max(np.where(observed, target, -np.inf), axis=1)
    regret = target_max - target[np.arange(len(selected)), selected]
    agreement = np.isclose(target[np.arange(len(selected)), selected], target_max, atol=1e-6, rtol=0)
    unrestricted = predictions.argmax(1)
    outside = ~observed[np.arange(len(selected)), unrestricted]
    assert (regret >= 0).all()
    return {"states": len(predictions), "observed_edges": int(counts.sum()),
            "state_mean_abs_error": float((absolute.sum(1) / counts).mean()),
            "state_mean_squared_error": float((squared.sum(1) / counts).mean()),
            "edge_mean_abs_error": float(absolute.sum() / counts.sum()),
            "edge_max_abs_error": float(absolute.max()), "restricted_action_agreement": float(agreement.mean()),
            "mean_graph_regret": float(regret.mean()), "max_graph_regret": float(regret.max()),
            "unrestricted_argmax_outside_logged_fraction": float(outside.mean()),
            "optimal_actions": int(agreement.sum()), "unrestricted_argmax_outside_logged_states": int(outside.sum())}


def validate_action_exposure(study, result, data, transitions, supports, counts, tables):
    exposure = read_json(study / "action_exposure.json")
    assert exposure == result["action_exposure"]
    expected = set(counts)
    key = lambda r: (r["bank_id"], r["condition"], r["seed"])
    assert {key(r) for r in exposure["per_seed"]} == expected and len(exposure["per_seed"]) == len(expected)
    fits = {(b, s) for b, c, s in expected if c == CONDITIONS[1]}
    saved_queries = dict(np.load(study / "recorded_query_counts.npz", allow_pickle=False))
    assert set(saved_queries) == {f"bank{b}_{c}_seed{s}" for b, s in fits for c in CONDITIONS}
    assert {(r["bank_id"], r["seed"]) for r in exposure["integrity"]} == fits and len(exposure["integrity"]) == len(fits)
    assert result["provenance"]["recorded_query_integrity"] == exposure["integrity"]
    totals = {"action_target_presentations": 0, "terminal_action_targets": 0, "nonterminal_action_targets": 0,
              "nonterminal_target_queries": 0, "outside_support_target_queries": 0}
    for row in exposure["per_seed"]:
        bank, condition, seed = key(row)
        count, support = counts[key(row)], supports[bank, condition]
        allowed = np.zeros(len(count), bool)
        allowed[support] = True
        table = tables[bank]
        observed, ended, successor = table["observed"], table["ends"], table["successor_indices"]
        live = observed & ~ended
        outside = live & ~allowed[np.maximum(successor, 0)]
        weights = count.astype(np.uint64)
        targets = int(np.dot(weights, observed.sum(1).astype(np.uint64)))
        live_presentations = int(np.dot(weights, live.sum(1).astype(np.uint64)))
        queries = live_presentations if condition == CONDITIONS[0] else 0
        outside_queries = int(np.dot(weights, outside.sum(1).astype(np.uint64)))
        assert row["source"] == ("archived_baseline" if condition == CONDITIONS[0] else "new_treatment")
        assert row["updates"] == result["sampling"][str(bank)][condition][str(seed)]["updates"]
        assert row["state_presentations"] == int(count.sum()) == row["updates"] * 64
        assert row["action_target_presentations"] == targets
        assert row["terminal_action_targets"] == targets - live_presentations
        assert row["nonterminal_action_targets"] == live_presentations
        assert row["nonterminal_target_queries"] == queries
        assert row["outside_support_target_queries"] == outside_queries
        if queries:
            close(row["outside_support_fraction"], outside_queries / queries)
        else:
            assert row["outside_support_fraction"] is None
        assert outside_queries == 0
        source_rows, actions = np.nonzero(live)
        query_counts = np.zeros(len(count), np.uint64)
        if condition == CONDITIONS[0]:
            np.add.at(query_counts, successor[source_rows, actions], count[source_rows].astype(np.uint64))
        actual = saved_queries[f"bank{bank}_{condition}_seed{seed}"]
        assert actual.dtype == np.uint32 and np.array_equal(actual, query_counts)
        assert int(actual.sum()) == queries and not actual[~allowed].any()
        if condition == CONDITIONS[1]:
            for field in totals:
                totals[field] += row[field]
        else:
            assert targets < int(count.sum()) * 4
    for row in exposure["integrity"]:
        assert row["tracked_exposure_matches"] and row["recorded_query_counts_match"] and row["no_outside_support_queries"]
    assert result["run"]["action_target_presentations"] == totals["action_target_presentations"]
    assert result["run"]["nonterminal_target_queries"] == totals["nonterminal_target_queries"]
    return totals

def validate_control_identity(study, archive, result, tables, supports, counts):
    names = ("recorded_transitions.npz", "recorded_query_counts.npz", "map_counts.npz", "action_exposure.json",
             "action_coverage.json", "supports.npz", "protocol.json", "protocol.md")
    for name in names:
        source = shared.checked_archive_file(archive, name)
        assert common.sha(source) == common.sha(study / ("control_" + name))
        assert result["provenance"]["archives"]["constrained_bootstrap"]["files"][name] == common.sha(source)
    old_tables = dict(np.load(study / "control_recorded_transitions.npz", allow_pickle=False))
    old_supports = dict(np.load(study / "control_supports.npz", allow_pickle=False))
    old_queries = dict(np.load(study / "control_recorded_query_counts.npz", allow_pickle=False))
    old_maps = dict(np.load(study / "control_map_counts.npz", allow_pickle=False))
    queries = dict(np.load(study / "recorded_query_counts.npz", allow_pickle=False))
    maps = dict(np.load(study / "map_counts.npz", allow_pickle=False))
    old_exposure = read_json(study / "control_action_exposure.json")["per_seed"]
    provenance = result["provenance"]
    assert result["protocol"]["control_archive"] == "constrained_bootstrap/pilot_v1"
    banks, seeds = result["protocol"]["bank_ids"], result["protocol"]["seeds"]
    assert len(provenance["control_recorded_tables"]) == len(banks)
    assert {r["bank_id"] for r in provenance["control_recorded_tables"]} == set(banks)
    for bank in banks:
        for name, value in tables[bank].items():
            assert np.array_equal(value, old_tables[f"bank{bank}_{name}"], equal_nan=True)
        assert np.array_equal(old_supports[f"{CONDITIONS[0]}_bank{bank}"], supports[bank, CONDITIONS[0]])
        row = next(r for r in provenance["control_recorded_tables"] if r["bank_id"] == bank)
        assert row["recorded_arrays_identical"] and row["support_identical"]
    fits = {(b, s) for b in banks for s in seeds}
    identity = provenance["paired_target_count_integrity"]
    assert len(identity) == len(fits) and {(r["bank_id"], r["seed"]) for r in identity} == fits
    exposure = result["action_exposure"]["per_seed"]
    target_fields = ("state_presentations", "action_target_presentations", "terminal_action_targets", "nonterminal_action_targets")
    for bank, seed in fits:
        control_name = f"bank{bank}_{CONDITIONS[0]}_seed{seed}"
        treatment_name = f"bank{bank}_{CONDITIONS[1]}_seed{seed}"
        assert np.array_equal(queries[control_name], old_queries[control_name])
        assert np.array_equal(maps[control_name], old_maps[control_name])
        assert queries[treatment_name].dtype == np.uint32 and not queries[treatment_name].any()
        key = lambda row: (row["bank_id"], row["condition"], row["seed"])
        old = next(r for r in old_exposure if key(r) == (bank, CONDITIONS[0], seed))
        control = next(r for r in exposure if key(r) == (bank, CONDITIONS[0], seed))
        treatment = next(r for r in exposure if key(r) == (bank, CONDITIONS[1], seed))
        assert all(old[k] == control[k] for k in (*target_fields, "nonterminal_target_queries"))
        assert treatment["nonterminal_target_queries"] == 0
        row = next(r for r in identity if (r["bank_id"], r["seed"]) == (bank, seed))
        assert row["control_target_and_query_counts_match_archive"] and row["control_query_vector_matches_archive"] and row["treatment_neural_queries_zero"]
        same_budget = result["protocol"]["budget"]["updates_per_fit"] == 30000
        assert row["same_budget"] == same_budget
        if same_budget:
            assert all(control[k] == treatment[k] for k in target_fields)
            assert np.array_equal(counts[bank, CONDITIONS[0], seed], counts[bank, CONDITIONS[1], seed])
            assert row["same_budget_target_counts_identical"]
        else:
            assert row["same_budget_target_counts_identical"] is None
    return len(names)


def validate_logged_targets(study, result, data, tables, supports):
    graph = read_json(study / "logged_graph.json")
    assert graph == result["logged_graph"]
    banks = result["protocol"]["bank_ids"]
    rows = graph["per_bank"]
    assert len(rows) == len(banks) and {r["bank_id"] for r in rows} == set(banks)
    saved = dict(np.load(study / "logged_graph_targets.npz", allow_pickle=False))
    assert set(saved) == {f"bank{b}_targets_{dtype}" for b in banks for dtype in ("float64", "float32")}
    integrity = result["provenance"]["logged_graph_integrity"]
    assert len(integrity) == len(banks) and {r["bank_id"] for r in integrity} == set(banks)
    targets, all_edges, all_live = {}, 0, 0
    for bank in banks:
        table, support = tables[bank], supports[bank, CONDITIONS[0]]
        expected, details = independent_logged_graph(table, data["remaining"], data["map_seeds"], support)
        double, single = saved[f"bank{bank}_targets_float64"], saved[f"bank{bank}_targets_float32"]
        assert double.dtype == np.float64 and single.dtype == np.float32
        assert np.array_equal(double, expected, equal_nan=True), (bank, "independent logged DP")
        assert np.array_equal(single, double.astype(np.float32), equal_nan=True), (bank, "exact float32 cast")
        row = next(r for r in rows if r["bank_id"] == bank)
        for field in ("observed_edges", "terminal_edges", "nonterminal_edges"):
            assert row[field] == details[field]
        assert row["supported_states"] == details["states"]
        assert row["graph_preparation_successor_backups"] == row["residual_validation_successor_lookups"] == details["nonterminal_edges"]
        assert row["neural_successor_queries"] == 0 and row["gamma"] == .97 and row["all_successors_closed"]
        assert row["clock_levels"] == len(row["per_clock"]) == 32
        assert row["bellman_residual_tolerance"] == 1e-12
        assert row["bellman_residual_max"] == row["bellman_residual_mean"] == 0.
        assert 0 < row["preparation_wall_seconds"] < result["run"]["wall_seconds"]
        for clock, detail in enumerate(row["per_clock"], 1):
            mask = data["remaining"] == clock
            observed = table["observed"][mask]
            live = observed & ~table["ends"][mask]
            assert detail == {"remaining": clock, "supported_states": int(observed.any(1).sum()),
                              "observed_edges": int(observed.sum()), "terminal_edges": int((observed & ~live).sum()),
                              "nonterminal_edges": int(live.sum()), "successor_backups": int(live.sum())}
        assert set(row["arrays"]) == {"targets_float64", "targets_float32"}
        bank_audit.check_array(double, row["arrays"]["targets_float64"])
        bank_audit.check_array(single, row["arrays"]["targets_float32"])
        check = next(r for r in integrity if r["bank_id"] == bank)
        assert check["before"] == check["after"] == row["arrays"]
        assert all(check[k] for k in ("unchanged", "read_only_reference", "read_only_float32_cast", "tensor_matches_saved_cast"))
        targets[bank] = expected
        all_edges += details["observed_edges"]
        all_live += details["nonterminal_edges"]
    assert result["run"]["graph_preparation_edges"] == all_edges
    assert result["run"]["graph_preparation_successor_backups"] == all_live
    return targets, all_edges, all_live


def validate_graph_fit(study, result, data, tables, supports, targets, states, agents, *, skip_forward):
    fit = read_json(study / "fit_diagnostics.json")
    assert fit == result["fit_diagnostics"]
    assert "not a gate" in fit["classification"]
    protocol, spec = result["protocol"], result["protocol"]["fit_diagnostics"]
    assert all(spec[k] for k in ("final_only", "after_all_treatment_fits", "both_conditions", "no_new_gate"))
    assert not spec["optimizer_successor_queries"] and not spec["checkpoint_probes"]
    assert spec["batch_size"] == 1024
    expected = set(states)
    rows = fit["per_seed"]
    key = lambda r: (r["bank_id"], r["condition"], r["seed"])
    assert len(rows) == len(expected) and {key(r) for r in rows} == expected
    saved = dict(np.load(study / "fit_predictions.npz", allow_pickle=False))
    keys = {f"bank{b}_{c}_seed{s}" for b, c, s in expected} | {f"bank{b}_state_rows" for b in protocol["bank_ids"]}
    assert set(saved) == keys
    pooled_inputs = {}
    root = Path(__file__).resolve().parents[1]
    for row in rows:
        bank, condition, seed = key(row)
        support = supports[bank, condition]
        assert np.array_equal(saved[f"bank{bank}_state_rows"], support)
        name = f"bank{bank}_{condition}_seed{seed}"
        assert row["prediction_key"] == name
        prediction = saved[name]
        assert prediction.shape == (len(support), 4) and prediction.dtype == np.float32
        bank_audit.check_array(prediction, row["predictions"])
        checkpoint = states[key(row)]["optimizer_updates"]
        assert row["checkpoint"] == checkpoint
        path = study / f"models/{name}_update{checkpoint}.pt"
        assert row["snapshot"] == path.relative_to(study).as_posix() and row["snapshot_sha256"] == common.sha(path)
        record = next(r for r in result["provenance"]["models"] if key(r) == key(row))
        for kind in ("online", "target", "file", "source"):
            assert row["before"][kind + "_before"] == row["after"][kind + "_after"] == record[kind + "_before"] == record[kind + "_after"]
        assert row["parameters_sources_rng_unchanged"]
        assert row["inference_state_rows"] == len(support) and row["online_output_values"] == 4 * len(support)
        assert row["target_network_queries"] == 0
        if not skip_forward:
            with torch.inference_mode():
                for start in range(0, len(support), 1024):
                    indices = support[start:start + 1024]
                    actual = agents[key(row)].online(torch.from_numpy(data["observations"][indices])).numpy()
                    assert np.array_equal(actual, prediction[start:start + len(indices)]), (key(row), start, "full-support forward")
        computed = independent_fit(prediction, targets[bank][support], tables[bank]["observed"][support])
        for field in ("states", "observed_edges"):
            assert row[field] == computed[field]
        for field in ("state_mean_abs_error", "state_mean_squared_error", "edge_mean_abs_error", "edge_max_abs_error", "restricted_action_agreement", "mean_graph_regret", "max_graph_regret", "unrestricted_argmax_outside_logged_fraction"):
            close(row[field], computed[field], (key(row), field))
        assert row["restricted_ranking_atol"] == 1e-6 and row["restricted_ranking_rtol"] == 0
        pooled_inputs.setdefault(condition, []).append(computed)
    pooled = fit["pooled"]
    assert len(pooled) == 2 and {r["condition"] for r in pooled} == set(CONDITIONS)
    for row in pooled:
        group = pooled_inputs[row["condition"]]
        n, edges = sum(r["states"] for r in group), sum(r["observed_edges"] for r in group)
        assert row["models"] == len(group) and row["states"] == n and row["observed_edges"] == edges
        for metric in ("state_mean_abs_error", "state_mean_squared_error", "restricted_action_agreement", "mean_graph_regret", "unrestricted_argmax_outside_logged_fraction"):
            close(row[metric], sum(r[metric] * r["states"] for r in group) / n)
        close(row["edge_mean_abs_error"], sum(r["edge_mean_abs_error"] * r["observed_edges"] for r in group) / edges)
        for metric in ("edge_max_abs_error", "max_graph_regret"):
            close(row[metric], max(r[metric] for r in group))
    n = sum(r["states"] for r in rows)
    edges = sum(r["observed_edges"] for r in rows)
    assert fit["inference_state_rows"] == fit["completed_prediction_state_rows"] == result["run"]["fit_diagnostic_state_rows"] == n
    assert fit["online_output_values"] == 4 * n
    assert fit["observed_edge_comparisons"] == result["run"]["fit_diagnostic_observed_edge_comparisons"] == edges
    assert fit["online_forward_batches"] == sum((r["states"] + 1023) // 1024 for r in rows)
    assert fit["neural_successor_queries"] == 0
    assert fit["wall_seconds"] == result["run"]["fit_diagnostic_wall_seconds"] and 0 < fit["wall_seconds"] < result["run"]["wall_seconds"]
    assert fit["sampled_process_peak_rss_bytes"] == result["run"]["fit_diagnostic_sampled_peak_rss_bytes"] <= result["run"]["peak_rss_bytes"]
    assert fit["integrity"] == result["provenance"]["fit_diagnostic_integrity"] == {"expected_models": len(expected), "actual_models": len(expected), "complete": True}
    assert result["run"]["checkpoint_probes"] == result["run"]["probe_query_count"] == 0
    assert not (study / "bootstrap_probes.json").exists() and not (study / "bootstrap_diagnostics.json").exists()
    return len(rows), n, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/logged_graph/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true")
    archive_names = ("supervised", "fixed_targets", "coverage", "equal_support", "panel_evaluation", "bank_replication", "map_replay", "within_map", "recorded_actions", "constrained_bootstrap")
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
    assert protocol["id"] == "logged-graph-v1" and result["run"]["status"] == "complete"
    assert result["artifacts"]["report"] == "docs/experiments/logged_graph_results_v1.md"
    assert protocol["bank_ids"] == [1, 2, 3]
    assert [r["id"] for r in protocol["conditions"]] == list(CONDITIONS)
    assert protocol["primary_comparison"] == "logged_graph_minus_constrained"
    final = max(protocol["checkpoints"])
    assert protocol["evaluation_checkpoints"] == {"constrained_bootstrap": 30000, "logged_graph": final}
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2] and protocol["git"]["dirty"] is False
        assert protocol["runtime"]["python"].startswith("3.12.")
        assert protocol["runtime"]["torch"].split("+")[0] == "2.8.0" and protocol["runtime"]["numpy"] == "2.0.2"
        assert protocol["panel_selection"] == {"count": 8, "maps_per_panel": 64, "start": 1120000, "stride": 1000}
        assert protocol["checkpoints"] == [0, 1000, 3000, 10000, 30000]
    assert protocol["control_archive"] == "constrained_bootstrap/pilot_v1"
    assert protocol["optimizer"]["implementation"] == "logged_graph.logged_graph_update"
    assert protocol["optimizer"]["successor_neural_queries"] == 0
    assert protocol["logged_graph"]["gamma"] == .97
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig()) and protocol["rule_visibility"] == "observed"
    assert all(protocol["collection"][k] == 0 for k in ("new_steps", "new_episodes"))
    assert protocol["evaluation"]["final_only"] and protocol["evaluation"]["after_all_treatment_fits"]
    assert protocol["evaluation"]["greedy_repetitions"] == 1 and protocol["evaluation"]["epsilon_0_1_repetitions"] == 2
    assert protocol["evaluation"]["optimal_q_atol"] == 1e-6 and protocol["evaluation"]["optimal_q_rtol"] == 0
    assert protocol["sampling"]["rng"] == "SeedSequence([seed,66301])"
    assert protocol["sampling"]["implementation"] == "unchanged coverage.SupportSampler"
    assert protocol["sampling"]["paired_local_global_map_schedule_within_bank"]
    assert protocol["optimizer"]["observed_only"]
    assert protocol["recorded_actions"]["equal_action_target_budget"] is True
    assert protocol["recorded_actions"]["all_tables_frozen_before_training"]
    assert protocol["collection"]["new_support_draws"] == 0
    for key, value in {"learning_rate": .001, "gradient_norm_cap": 5., "gamma": .97, "target_tau": .01, "batch_size": 64}.items():
        close(protocol["optimizer"][key], value)
    assert protocol["dataset"]["fresh_state_enumeration"] is False
    report = {"study": str(args.study), "smoke_validation_only": smoke,
              "manifest_files": common.audit_manifest(study, protocol, result)}
    data, arrays, metadata, transitions, supports = validate_inputs(study, archives, protocol)
    transition_data = dict(arrays)
    for key, value in data.items():
        transition_data["heldout_" + key] = np.empty((0,) + value.shape[1:], dtype=value.dtype)
    transition_report = common.audit_transitions(transition_data, {**metadata, "heldout": []}, transitions, config, CollectionWorld)
    tables, collector_reports = validate_recorded_tables(study, archives, result, data, metadata, transitions, supports)
    selection, task_hashes, planner = reconstruct_panels(study, archives, protocol, result, metadata, CollectionWorld, config)
    states, agents, initial, snapshots = validate_models(study, archives["constrained_bootstrap"], result)
    counts = validate_sampling(study, archives["constrained_bootstrap"], result, data, supports)
    exposure_maps, exposure_clocks = validate_exposure(study, result, data, transitions, supports, counts, tables)
    action_totals = validate_action_exposure(study, result, data, transitions, supports, counts, tables)
    control_inputs = validate_control_identity(study, archives["constrained_bootstrap"], result, tables, supports, counts)
    targets, graph_edges, graph_backups = validate_logged_targets(study, result, data, tables, supports)
    fit_models, fit_states, fit_edges = validate_graph_fit(study, result, data, tables, supports, targets, states, agents,
                                                        skip_forward=args.skip_forward_inference)
    if not smoke:
        assert action_totals == {"action_target_presentations": 24273701, "terminal_action_targets": 1062190,
                                "nonterminal_action_targets": 23211511, "nonterminal_target_queries": 0,
                                "outside_support_target_queries": 0}
        assert graph_edges == 252329 and graph_backups == 241295
        assert (fit_models, fit_states, fit_edges) == (18, 1077966, 1513974)
    loss_rows = validate_losses(study, result)
    rows, refs, paired = shared.validate_episode_tables(study, result, states, task_hashes, planner)
    shared.validate_thresholds(result, eligible=not smoke)
    source_inputs = validate_provenance(study, archives, result, data, transitions, supports, initial, snapshots, counts, rows, refs)
    replay_count, replay_steps = shared.validate_replays(study, result, rows, refs, agents, CollectionWorld, config,
                                                skip_forward=args.skip_forward_inference)
    run, budget = result["run"], protocol["budget"]
    fits = {(b, s) for b in protocol["bank_ids"] for s in protocol["seeds"]}
    assert run["fits"] == run["frozen_baselines"] == len(fits) and len(states) == 2 * len(fits)
    assert run["banks"] == len(protocol["bank_ids"]) and run["model_parameter_count"] == 20420
    assert run["train_updates"] == len(fits) * final == budget["maximum_updates"] and budget["updates_per_fit"] == final
    assert run["training_examples"] == run["train_updates"] * 64 and run["historical_baseline_updates"] == len(fits) * 30000
    assert all(run[k] == 0 for k in ("baseline_new_updates", "collection_episodes", "collection_steps"))
    assert run["support_draws"] == 0
    assert run["baseline_reconstructed_updates"] == len(fits) * 30000
    assert all(budget[k] == 0 for k in ("new_baseline_updates", "maximum_collection_steps", "maximum_collection_episodes"))
    assert run["learner_episodes"] == len(rows) and run["reference_episodes"] == run["unique_reference_episodes"] == len(refs)
    assert run["panels"] == len(selection["panels"])
    assert 0 < run["wall_seconds"] <= budget["admission_seconds"] <= 1200
    assert 0 < run["peak_rss_bytes"] <= budget["peak_process_rss_bytes"] == 4 * 1024 ** 3
    assert run["resource_checks"] > run["train_updates"] + run["baseline_reconstructed_updates"] + len(rows) + len(refs)
    assert run["resource_limits"] == {"seconds": budget["admission_seconds"], "peak_process_rss_bytes": budget["peak_process_rss_bytes"], "torch_threads": 1}
    assert protocol["runtime"]["torch_threads"] == 1 and budget["all_phases_included"]
    assert run["interpretation"] == ("smoke_or_deviation_descriptive_only" if smoke else "logged_graph_descriptive_only") and run["stop_reason"] is None
    completed = [r for r in run["progress"] if r.get("training_complete")]
    assert {(r["bank_id"], r["seed"]) for r in completed} == fits and len(completed) == len(fits)
    assert all(r["updates"] == final for r in completed)
    progress = [r for r in run["progress"] if r.get("snapshot_saved")]
    assert {(r["bank_id"], r["seed"], r["checkpoint"]) for r in progress} == {(b, s, cp) for b, s in fits for cp in protocol["checkpoints"]}
    assert len(progress) == len(fits) * len(protocol["checkpoints"]) and all(r["condition"] == "logged_graph" for r in progress)
    evaluated = [r for r in run["progress"] if r.get("evaluation_complete")]
    expected_evaluated = {(b, c, s, p["id"]) for b, s in fits for c in CONDITIONS for p in result["panels"]}
    assert {(r["bank_id"], r["condition"], r["seed"], r["panel"]) for r in evaluated} == expected_evaluated
    assert len(evaluated) == len(expected_evaluated)
    assert len(run["progress"]) == len(progress) + len(completed) + len(evaluated)
    report.update(training_states=len(data["observations"]), reused_support_sizes={str(b): len(supports[b, CONDITIONS[0]]) for b in protocol["bank_ids"]}, recorded_bank_tables_checked=len(tables),
        selected_layouts=len(task_hashes), rejected_candidates=len(selection["collision_skips"]), transition_validation=transition_report,
        archive_input_hashes_checked=source_inputs, inherited_counts_and_controls_identical=True,
        treatment_sampling_streams_checked=3 * len(fits), paired_local_and_map_streams_checked=len(fits), historical_sampler_updates_reconstructed=len(fits) * 30000,
        independent_logged_graph_edges_checked=graph_edges, graph_preparation_successor_backups=graph_backups,
        float64_labels_and_float32_casts_checked=True, optimizer_neural_successor_queries=0,
        final_supported_state_models_checked=fit_models, final_supported_state_predictions_checked=fit_states,
        final_supported_observed_edge_errors_checked=fit_edges, final_support_forward_inference_checked=not args.skip_forward_inference,
        control_recorded_input_copies_checked=control_inputs,
        recorded_action_exposure=action_totals, historical_collector_logs_checked=collector_reports,
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
