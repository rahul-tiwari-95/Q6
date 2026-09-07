"""Independently validate same-support map-balanced replay evidence.

Reconstruct training sampler streams, archive/model identity, panel admission,
raw statistics and preselected saved recordings. No optimization or second full
policy evaluation occurs. Portable mode skips only neural forward regeneration
for the already saved recordings.
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
import audit_panel_evaluation as replay_audit
import audit_bank_replication as bank_audit
from audit_fixed_targets_study import close, csv_rows, read_json

METRICS = replay_audit.DELTA_METRICS


def reconstruct_sampler(support, map_seeds, seed, updates):
    """Recreate the declared scalar calls without importing the new sampler."""
    support = np.asarray(support)
    assert support.dtype == np.int32 and np.all(np.diff(support) > 0)
    unique_maps = np.unique(map_seeds)
    assert np.array_equal(unique_maps, np.arange(300000, 300256))
    grouped = [support[map_seeds[support] == map_seed] for map_seed in unique_maps]
    assert all(len(group) for group in grouped)
    global_to_local = np.full(len(map_seeds), -1, np.int64)
    global_to_local[support] = np.arange(len(support))
    map_rng = np.random.default_rng(np.random.SeedSequence([seed, 99301]))
    row_rng = np.random.default_rng(np.random.SeedSequence([seed, 99302]))
    counts = np.zeros(len(map_seeds), np.uint32)
    local_counts = np.zeros(len(support), np.uint32)
    map_counts = np.zeros(len(unique_maps), np.uint32)
    rank_counts = np.zeros(max(map(len, grouped)), np.uint32)
    digests = {k: hashlib.sha256() for k in ("map", "rank", "local", "global")}
    for _ in range(updates):
        map_indices = map_rng.choice(len(unique_maps), size=64, replace=False)
        ranks = np.asarray([int(row_rng.integers(len(grouped[index]))) for index in map_indices], np.int64)
        rows = np.asarray([grouped[index][rank] for index, rank in zip(map_indices, ranks)], np.int64)
        local = global_to_local[rows]
        assert len(np.unique(map_indices)) == len(np.unique(rows)) == 64 and np.all(local >= 0)
        counts[rows] += 1
        local_counts[local] += 1
        map_counts[map_indices] += 1
        rank_counts += np.bincount(ranks, minlength=len(rank_counts)).astype(np.uint32)
        for key, values in (("map", unique_maps[map_indices]), ("rank", ranks), ("local", local), ("global", rows)):
            digests[key].update(values.astype("<i8").tobytes())
    assert int(counts.sum()) == int(local_counts.sum()) == int(map_counts.sum()) == updates * 64
    assert np.array_equal(counts[support], local_counts) and np.all(counts[global_to_local < 0] == 0)
    return {"global_counts": counts, "local_counts": local_counts, "map_counts": map_counts,
            "rank_counts": rank_counts, "map_seeds": unique_maps,
            "digests": {k: v.hexdigest() for k, v in digests.items()}}


def independent_exposure(data, transitions, support, counts):
    """Training presentations, separate from unique-state membership."""
    assert len(counts) == len(data["observations"]) and np.issubdtype(counts.dtype, np.integer)
    supported = np.zeros(len(counts), bool)
    supported[support] = True
    assert np.all(counts[~supported] == 0)
    total = int(counts.sum())
    map_seeds = np.unique(data["map_seeds"])
    by_map = np.asarray([counts[data["map_seeds"] == m].sum() for m in map_seeds], np.uint64)
    near = np.zeros(len(counts), bool)
    for m in map_seeds:
        map_rows = np.flatnonzero(data["map_seeds"] == m)
        for position in np.unique(data["positions"][map_rows], axis=0):
            rows = map_rows[np.all(data["positions"][map_rows] == position, axis=1)]
            possible = rows[data["winnable"][rows]]
            assert len(possible)
            if data["remaining"][possible].min() <= 2:
                near[rows] = True
    successor = transitions["successor_indices"]
    live = ~transitions["ends"]
    outside = live & ~supported[np.maximum(successor, 0)]
    live_queries = int((counts[:, None].astype(np.uint64) * live).sum())
    outside_queries = int((counts[:, None].astype(np.uint64) * outside).sum())
    def category(mask):
        return {"presentations": int(counts[mask].sum()), "support_states": int((supported & mask).sum()),
                "unique_sampled_states": int(((counts > 0) & mask).sum())}
    buckets = {"all": (1, 32), "1-8": (1, 8), "9-16": (9, 16), "17-24": (17, 24), "25-32": (25, 32)}
    return {"presentations": total, "unique_sampled_states": int(np.count_nonzero(counts)),
        "map_seeds": map_seeds, "map_counts": by_map, "map_min": int(by_map.min()), "map_max": int(by_map.max()),
        "map_cv": float(by_map.std() / by_map.mean()) if total else None,
        "by_time_bucket": {k: category((data["remaining"] >= low) & (data["remaining"] <= high)) for k, (low, high) in buckets.items()},
        "winnable": category(data["winnable"]), "impossible": category(~data["winnable"]),
        "goal_near": category(near), "goal_far": category(~near),
        "nonterminal_queries": live_queries, "outside_support_queries": outside_queries,
        "outside_support_fraction": outside_queries / live_queries if live_queries else None}


def checked_archive_file(directory, name):
    manifest = read_json(directory / "manifest.json")
    assert manifest["algorithm"] == "sha256" and manifest["status"] == "complete"
    assert common.sha(directory / name) == manifest["files"][name], ("archive identity", str(directory), name)
    return directory / name


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
        old_metadata = read_json(checked_archive_file(archive, "dataset_metadata.json"))
        assert metadata["train"] == old_metadata["train"]
        with np.load(checked_archive_file(archive, "dataset.npz"), allow_pickle=False) as old:
            for name, value in arrays.items():
                assert np.array_equal(value, old[name]), ("unchanged training data", name)
        with np.load(checked_archive_file(archive, "transitions.npz"), allow_pickle=False) as old:
            for name, value in transitions.items():
                assert np.array_equal(value, old[name]), ("unchanged transitions", name)
    assert [r["map_seed"] for r in metadata["train"]] == list(range(300000, 300256))
    data = {k.removeprefix("train_"): v for k, v in arrays.items()}
    assert len(data["observations"]) == 163840
    saved = dict(np.load(study / "supports.npz", allow_pickle=False))
    old = dict(np.load(checked_archive_file(archives["bank_replication"], "supports.npz"), allow_pickle=False))
    conditions = [r["id"] for r in protocol["conditions"]]
    assert conditions == ["collected_unique", "map_balanced"]
    assert set(saved) == {f"{c}_bank{b}" for c in conditions for b in protocol["bank_ids"]}
    supports = {}
    for bank, size in zip((1, 2, 3), (59839, 59984, 59838)):
        original = old[f"collected_unique_bank{bank}"]
        assert original.dtype == np.int32 and len(original) == size and np.all(np.diff(original) > 0)
        with np.load(checked_archive_file(archives["bank_replication"], f"banks/bank{bank}/collection.npz"), allow_pickle=False) as collection:
            assert np.array_equal(original, collection["support_indices"])
            assert np.array_equal(original, np.flatnonzero(collection["visited_counts"]))
        for condition in conditions:
            actual = saved[f"{condition}_bank{bank}"]
            assert actual.dtype == np.int32 and np.array_equal(actual, original)
        assert np.array_equal(np.unique(data["map_seeds"][original]), np.arange(300000, 300256))
        supports[bank] = original
    for name in ("q6/world.py", "q6/learning.py", "q6/fixed_targets.py", "q6/competence.py", "q6/supervised.py"):
        for archive in (archives["coverage"], archives["bank_replication"]):
            old_protocol = read_json(checked_archive_file(archive, "protocol.json"))
            assert protocol["source_sha256"][name] == old_protocol["source_sha256"][name]
            assert common.sha(checked_archive_file(archive, "source/" + name)) == old_protocol["source_sha256"][name]
    return data, arrays, metadata, transitions, supports


def reconstruct_panels(study, archives, protocol, result, metadata, World, config):
    excluded = {}
    for name, directory in archives.items():
        if name in ("panel_evaluation", "bank_replication"):
            selection = read_json(checked_archive_file(directory, "panels.json"))
            layouts = [r for p in selection["panels"] for r in p["layouts"]]
        else:
            saved = read_json(checked_archive_file(directory, "dataset_metadata.json"))
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


def validate_sampling(study, archive, result, data, supports):
    sampling = read_json(study / "sampling.json")
    assert sampling == result["sampling"]
    old_sampling = read_json(checked_archive_file(archive, "sampling.json"))
    saved_global = dict(np.load(study / "sample_counts.npz", allow_pickle=False))
    saved_local = dict(np.load(study / "local_sample_counts.npz", allow_pickle=False))
    saved_map = dict(np.load(study / "map_counts.npz", allow_pickle=False))
    old_global = dict(np.load(checked_archive_file(archive, "sample_counts.npz"), allow_pickle=False))
    old_local = dict(np.load(checked_archive_file(archive, "local_sample_counts.npz"), allow_pickle=False))
    protocol = result["protocol"]
    banks, seeds, updates = protocol["bank_ids"], protocol["seeds"], max(protocol["checkpoints"])
    conditions = [r["id"] for r in protocol["conditions"]]
    expected = {f"bank{b}_{c}_seed{s}" for b in banks for c in conditions for s in seeds}
    assert set(saved_global) == set(saved_local) == set(saved_map) == expected
    assert set(sampling) == set(map(str, banks))
    counts, map_digests = {}, {}
    for bank in banks:
        assert set(sampling[str(bank)]) == set(conditions)
        for condition in conditions:
            assert set(sampling[str(bank)][condition]) == set(map(str, seeds))
            for seed in seeds:
                key = f"bank{bank}_{condition}_seed{seed}"
                record = sampling[str(bank)][condition][str(seed)]
                global_counts, local_counts, per_map = saved_global[key], saved_local[key], saved_map[key]
                assert global_counts.dtype == local_counts.dtype == per_map.dtype == np.uint32
                assert global_counts.shape == (len(data["observations"]),) and local_counts.shape == (len(supports[bank]),) and per_map.shape == (256,)
                assert np.array_equal(global_counts[supports[bank]], local_counts)
                assert int(global_counts.sum()) == int(local_counts.sum()) == int(per_map.sum())
                if condition == "collected_unique":
                    assert record == {**old_sampling[str(bank)][condition][str(seed)], "historical": True, "new_updates": 0}
                    assert np.array_equal(global_counts, old_global[key]) and np.array_equal(local_counts, old_local[key])
                    expected_map = np.asarray([global_counts[data["map_seeds"] == m].sum() for m in np.arange(300000, 300256)], np.uint32)
                    assert np.array_equal(per_map, expected_map)
                    assert record["updates"] == 30000 and record["examples_seen"] == 1920000
                else:
                    reconstructed = reconstruct_sampler(supports[bank], data["map_seeds"], seed, updates)
                    assert np.array_equal(global_counts, reconstructed["global_counts"])
                    assert np.array_equal(local_counts, reconstructed["local_counts"])
                    assert np.array_equal(per_map, reconstructed["map_counts"])
                    for field, digest in (("map_index_sha256", "map"), ("within_map_rank_sha256", "rank"),
                                          ("local_batch_index_sha256", "local"), ("global_batch_index_sha256", "global")):
                        assert record[field] == reconstructed["digests"][digest], ("sampling digest", bank, seed, field)
                    assert record["updates"] == record["new_updates"] == updates and record["historical"] is False
                    assert record["examples_seen"] == updates * 64
                    if seed in map_digests:
                        assert map_digests[seed] == reconstructed["digests"]["map"]
                    map_digests[seed] = reconstructed["digests"]["map"]
                assert record["outside_support_direct_samples"] == 0
                assert record["support_states"] == len(supports[bank])
                assert record["unique_states_sampled"] == int(np.count_nonzero(global_counts))
                assert int(global_counts.sum()) == record["examples_seen"]
                mask = np.zeros(len(global_counts), bool)
                mask[supports[bank]] = True
                assert not global_counts[~mask].any()
                counts[bank, condition, seed] = global_counts
    return counts, map_digests


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
        actual = independent_exposure(data, transitions, supports[bank], counts[group_key])
        total = actual["presentations"]
        assert summary["source"] == ("archived_baseline" if condition == "collected_unique" else "new_treatment")
        assert summary["updates"] == result["sampling"][str(bank)][condition][str(seed)]["updates"]
        assert summary["presentations"] == total and summary["unique_states_sampled"] == actual["unique_sampled_states"]
        close(summary["map_fraction_min"], actual["map_min"] / total)
        close(summary["map_fraction_max"], actual["map_max"] / total)
        close(summary["map_fraction_cv"], actual["map_cv"])
        state_counts = counts[group_key][supports[bank]]
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
            check_category(row, {"presentations": int(counts[group_key][mask].sum()), "support_states": int(np.count_nonzero(mask[supports[bank]])),
                                 "unique_sampled_states": int(np.count_nonzero(counts[group_key][mask]))})
        for row in [r for r in exposure["per_clock"] + exposure["per_map"] if key(r) == group_key]:
            assert row["source"] == summary["source"] and row["updates"] == summary["updates"]
    return len(exposure["per_map"]), len(exposure["per_clock"])


def validate_models(study, archive, result):
    protocol = result["protocol"]
    banks, seeds, schedule = protocol["bank_ids"], protocol["seeds"], protocol["checkpoints"]
    final = max(schedule)
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: Path(p).resolve() if Path(p).is_absolute() else (root / p).resolve()
    conditions = [r["id"] for r in protocol["conditions"]]
    snapshot_rows = read_json(study / "snapshots.json")
    expected_snapshots = {(b, "map_balanced", s, cp) for b in banks for s in seeds for cp in schedule}
    expected_snapshots |= {(b, "collected_unique", s, 30000) for b in banks for s in seeds}
    assert {(r["bank_id"], r["condition"], r["seed"], r["checkpoint"]) for r in snapshot_rows} == expected_snapshots
    assert len(snapshot_rows) == len(expected_snapshots)
    files = {f"models/bank{b}_{c}_seed{s}_update{cp}.pt" for b, c, s, cp in expected_snapshots}
    assert {p.relative_to(study).as_posix() for p in (study / "models").glob("*.pt")} == files
    initial, states, agents = {}, {}, {}
    for bank in banks:
        for seed in seeds:
            path = checked_archive_file(archive, f"models/bank{bank}_collected_unique_seed{seed}_update0.pt")
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
            assert common.sha(path) == common.sha(checked_archive_file(archive, name))
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
                assert row["condition"] == "map_balanced"
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
        assert row["condition"] == "map_balanced" and row["seeds"] == len(group) == len(protocol["seeds"])
        assert all(row["updates_in_window"] == int(r["updates_in_window"]) for r in group)
        close(row["mean_loss"], np.mean([float(r["mean_loss"]) for r in group]))
    assert result["provenance"]["loss_integrity"] == {"expected_windows": len(expected), "actual_windows": len(rows), "complete": True, "finite": True}
    return len(rows)


def validate_aggregates(result, rows, refs, planner):
    """Raw episode reductions with explicit complete keys and ratio denominators."""
    protocol = result["protocol"]
    banks, seeds = protocol["bank_ids"], protocol["seeds"]
    conditions = [r["id"] for r in protocol["conditions"]]
    panels = [r["id"] for r in result["panels"]] + ["all"]
    expected = {(b, c, p, m) for b in banks for c in conditions for p in panels for m in ("greedy", "epsilon_0_1")}
    key = lambda r: (r["bank_id"], r["condition"], r["panel"], r["mode"])
    assert {key(r) for r in result["aggregate"]} == expected and len(result["aggregate"]) == len(expected)
    assert {key(r) + (r["seed"],) for r in result["seed_results"]} == {k + (s,) for k in expected for s in seeds}
    assert len(result["seed_results"]) == len(expected) * len(seeds)
    for summary in result["aggregate"] + result["seed_results"]:
        group = [r for r in rows if int(r["bank_id"]) == summary["bank_id"] and r["condition"] == summary["condition"]
                 and r["mode"] == summary["mode"] and (summary["panel"] == "all" or r["panel"] == summary["panel"])
                 and ("seed" not in summary or int(r["seed"]) == summary["seed"])]
        replay_audit.audit_summary(summary, group, planner)
    assert {(r["policy"], r["panel"]) for r in result["references"]} == {(p, q) for p in ("random_actions", "shortest_path") for q in panels}
    assert len(result["references"]) == 2 * len(panels)
    for summary in result["references"]:
        group = [r for r in refs if r["policy"] == summary["policy"] and (summary["panel"] == "all" or r["panel"] == summary["panel"])]
        replay_audit.audit_summary(summary, group, planner)


def validate_pairs(study, result, rows, planner):
    protocol = result["protocol"]
    banks, seeds, panels = protocol["bank_ids"], protocol["seeds"], result["panels"]
    conditions = [r["id"] for r in protocol["conditions"]]
    comparison = protocol["primary_comparison"]
    paired = read_json(study / "paired_differences.json")
    assert paired == result["paired_differences"]
    assert paired["comparisons"] == [{"id": comparison, "left": conditions[0], "right": conditions[1]}]
    key = lambda r: (int(r["bank_id"]), r["condition"], int(r["seed"]), r["panel"], r["mode"], int(r["map_seed"]), int(r["repetition"]))
    lookup = {key(r): r for r in rows}
    expected = {(b, s, p["id"], m) for b in banks for s in seeds for p in panels for m in p["map_seeds"]}
    assert {(r["bank_id"], r["seed"], r["panel"], r["map_seed"]) for r in paired["per_layout"]} == expected
    assert len(paired["per_layout"]) == len(expected)
    csv_pairs = csv_rows(study / "paired_layouts.csv")
    assert len(csv_pairs) == len(expected)
    for row, saved in zip(paired["per_layout"], csv_pairs):
        assert row["comparison"] == comparison and row["mode"] == "greedy" and row["repetition"] == 0
        assert set(row) == set(saved)
        for k, value in row.items():
            if isinstance(value, str):
                assert saved[k] == value
            else:
                close(saved[k], value)
        left, right = (lookup[row["bank_id"], c, row["seed"], row["panel"], "greedy", row["map_seed"], 0] for c in conditions)
        close(row["success_delta"], int(right["success"]) - int(left["success"]))
        close(row["steps_delta"], int(right["steps"]) - int(left["steps"]))
        close(row["efficient_success_delta"], replay_audit.efficient_rate([right], planner) - replay_audit.efficient_rate([left], planner))
        close(row["noop_rate_delta"], int(right["noop_steps"]) / int(right["steps"]) - int(left["noop_steps"]) / int(left["steps"]))
    panel_ids = [p["id"] for p in panels] + ["all"]
    lookup = {(r["bank_id"], r["condition"], r["panel"], r["seed"]): r for r in result["seed_results"] if r["mode"] == "greedy"}
    assert {(r["bank_id"], r["seed"], r["panel"]) for r in paired["per_seed"]} == {(b, s, p) for b in banks for s in seeds for p in panel_ids}
    assert len(paired["per_seed"]) == len(banks) * len(seeds) * len(panel_ids)
    for row in paired["per_seed"]:
        assert row["comparison"] == comparison and row["mode"] == "greedy"
        left, right = (lookup[row["bank_id"], c, row["panel"], row["seed"]] for c in conditions)
        for metric in METRICS:
            close(row[metric + "_delta"], right[metric] - left[metric])
    lookup = {(r["bank_id"], r["condition"], r["panel"]): r for r in result["aggregate"] if r["mode"] == "greedy"}
    assert {(r["bank_id"], r["panel"]) for r in paired["aggregate"]} == {(b, p) for b in banks for p in panel_ids}
    assert len(paired["aggregate"]) == len(banks) * len(panel_ids)
    for row in paired["aggregate"]:
        assert row["comparison"] == comparison and row["mode"] == "greedy" and row["seeds"] == len(seeds)
        group = [r for r in paired["per_seed"] if r["bank_id"] == row["bank_id"] and r["panel"] == row["panel"]]
        for metric in METRICS:
            close(row["mean_seed_" + metric + "_delta"], np.mean([r[metric + "_delta"] for r in group]))
        left, right = (lookup[row["bank_id"], c, row["panel"]] for c in conditions)
        close(row["pooled_noop_rate_delta"], right["noop_rate"] - left["noop_rate"])
    return paired


def validate_pooled(result, paired):
    protocol = result["protocol"]
    banks, seeds = protocol["bank_ids"], protocol["seeds"]
    conditions = [r["id"] for r in protocol["conditions"]]
    panels = [r["id"] for r in result["panels"]] + ["all"]
    pooled, robustness = result["pooled"], result["robustness"]
    expected = {(c, p, m) for c in conditions for p in panels for m in ("greedy", "epsilon_0_1")}
    assert {(r["condition"], r["panel"], r["mode"]) for r in pooled["aggregate"]} == expected and len(pooled["aggregate"]) == len(expected)
    for row in pooled["aggregate"]:
        group = [r for r in result["aggregate"] if (r["condition"], r["panel"], r["mode"]) == (row["condition"], row["panel"], row["mode"])]
        assert row["banks"] == len(group) == len(banks) and row["seeds_per_bank"] == len(seeds)
        assert row["episodes"] == sum(r["episodes"] for r in group)
        for metric in METRICS:
            if metric != "noop_rate":
                close(row[metric], np.mean([r[metric] for r in group]))
        close(row["mean_bank_noop_rate"], np.mean([r["noop_rate"] for r in group]))
        for field in ("noop_steps", "evaluation_steps", "successful_episodes", "successful_episode_steps", "efficient_success_episodes"):
            assert row[field] == sum(r[field] for r in group)
        close(row["noop_rate"], row["noop_steps"] / row["evaluation_steps"])
        if row["successful_episodes"]:
            close(row["successful_mean_steps"], row["successful_episode_steps"] / row["successful_episodes"])
        else:
            assert row["successful_mean_steps"] is None
    assert {r["panel"] for r in pooled["paired"]} == set(panels) and len(pooled["paired"]) == len(panels)
    for row in pooled["paired"]:
        group = [r for r in paired["aggregate"] if r["panel"] == row["panel"]]
        assert row["banks"] == len(banks) == len(group) and row["mode"] == "greedy"
        for metric in METRICS:
            if metric != "noop_rate":
                close(row["mean_bank_" + metric + "_delta"], np.mean([r["mean_seed_" + metric + "_delta"] for r in group]))
        close(row["mean_bank_noop_rate_delta"], np.mean([r["pooled_noop_rate_delta"] for r in group]))
        close(row["mean_bank_mean_seed_noop_rate_delta"], np.mean([r["mean_seed_noop_rate_delta"] for r in group]))
    effects = [r for r in paired["aggregate"] if r["panel"] == "all"]
    assert robustness["bank_effects"] == effects and len(effects) == len(banks)
    assert robustness["primary_comparison"] == protocol["primary_comparison"]
    assert "no new gates" in robustness["classification"]
    assert set(robustness["metrics"]) == {m + "_delta" for m in METRICS}
    for metric in METRICS:
        values = [r["mean_seed_" + metric + "_delta"] for r in effects]
        row = robustness["metrics"][metric + "_delta"]
        assert row["banks"] == len(banks) and row["sign_tolerance"] == 1e-12
        close(row["minimum"], min(values)); close(row["maximum"], max(values)); close(row["mean"], np.mean(values))
        assert row["positive_banks"] == sum(v > 1e-12 for v in values)
        assert row["negative_banks"] == sum(v < -1e-12 for v in values)
        assert row["zero_banks"] == sum(abs(v) <= 1e-12 for v in values)


def validate_episode_tables(study, result, states, task_hashes, planner):
    protocol = result["protocol"]
    banks, seeds, panels = protocol["bank_ids"], protocol["seeds"], result["panels"]
    conditions = [r["id"] for r in protocol["conditions"]]
    rows, refs = csv_rows(study / "evaluations.csv"), csv_rows(study / "references.csv")
    key = lambda r: (int(r["bank_id"]), r["condition"], int(r["seed"]), r["panel"], r["mode"], int(r["map_seed"]), int(r["repetition"]))
    expected = {(b, c, s, p["id"], mode, m, rep) for b in banks for c in conditions for s in seeds for p in panels
                for mode in ("greedy", "epsilon_0_1") for m in p["map_seeds"] for rep in range(1 if mode == "greedy" else 2)}
    assert {key(r) for r in rows} == expected and len(rows) == len(expected)
    expected_refs = {(policy, s, p["id"], m, rep) for policy in ("random_actions", "shortest_path")
                     for s in (seeds if policy == "random_actions" else [0]) for p in panels for m in p["map_seeds"]
                     for rep in range(2 if policy == "random_actions" else 1)}
    assert {(r["policy"], int(r["seed"]), r["panel"], int(r["map_seed"]), int(r["repetition"])) for r in refs} == expected_refs
    assert len(refs) == len(expected_refs) == len({r["reference_sample_id"] for r in refs})
    for row in rows:
        assert row["policy"] == "learner"
        state = states[int(row["bank_id"]), row["condition"], int(row["seed"])]
        bank_audit.check_episode(row, states, task_hashes, planner, state["optimizer_updates"])
    for row in refs:
        assert row["bank_id"] == row["condition"] == "shared" and row["mode"] == "reference"
        bank_audit.check_episode(row, states, task_hashes, planner, max(protocol["checkpoints"]))
    validate_aggregates(result, rows, refs, planner)
    paired = validate_pairs(study, result, rows, planner)
    validate_pooled(result, paired)
    return rows, refs, paired


def validate_replays(study, result, rows, refs, agents, World, config, *, skip_forward):
    protocol = result["protocol"]
    conditions = [r["id"] for r in protocol["conditions"]]
    traces = read_json(study / "trajectories.json")
    assert traces == result["trajectories"]
    key = lambda r: (str(r["bank_id"]), r["condition"], r["policy"], int(r["seed"]), r["panel"], r["mode"], int(r["map_seed"]), int(r["repetition"]))
    expected = {(str(b), c, "learner", s, p["id"], mode, p["map_seeds"][0], 0) for b in protocol["bank_ids"]
                for c in conditions for s in protocol["seeds"] for p in result["panels"] for mode in ("greedy", "epsilon_0_1")}
    expected |= {("shared", "shared", policy, protocol["seeds"][0] if policy == "random_actions" else 0, p["id"], "reference", p["map_seeds"][0], 0)
                 for policy in ("random_actions", "shortest_path") for p in result["panels"]}
    assert {key(r) for r in traces} == expected and len(traces) == len(expected)
    lookup = {key(r): r for r in rows + refs}
    steps = 0
    for trace in traces:
        row = lookup[key(trace)]
        assert int(trace["checkpoint"]) == int(row["checkpoint"])
        agent = agents[trace["bank_id"], trace["condition"], trace["seed"]] if trace["policy"] == "learner" else None
        steps += replay_audit.check_saved_replay(trace, row, World(config), agent, skip_forward=skip_forward)
    return len(traces), steps


def validate_thresholds(result, *, eligible):
    protocol = result["protocol"]
    banks, seeds, panels = protocol["bank_ids"], protocol["seeds"], result["panels"]
    conditions = [r["id"] for r in protocol["conditions"]]
    thresholds = result["descriptive_thresholds"]
    assert "not new competence gates" in thresholds["classification"] and "gates" not in result
    assert thresholds["eligible"] == result["robustness"]["eligible"] == eligible and protocol["new_competence_gates"] is False
    expected = {(b, c, s, p["id"]) for b in banks for c in conditions for s in seeds for p in panels}
    key = lambda r: (r["bank_id"], r["condition"], r["seed"], r["panel"])
    assert {key(r) for r in thresholds["per_seed"]} == expected and len(thresholds["per_seed"]) == len(expected)
    lookup = {key(r): r for r in result["seed_results"] if r["mode"] == "greedy"}
    random = {r["panel"]: r["success_rate"] for r in result["references"] if r["policy"] == "random_actions"}
    for row in thresholds["per_seed"]:
        source = lookup[key(row)]
        close(row["success_rate"], source["success_rate"])
        close(row["efficient_success_rate"], source["efficient_success_rate"])
        close(row["panel_random_success"], random[row["panel"]])
        assert row["success_reference_met"] == bool(eligible and source["success_rate"] >= .7 and source["success_rate"] > random[row["panel"]])
        assert row["efficiency_reference_met"] == bool(eligible and source["efficient_success_rate"] >= .8)
    assert {(r["bank_id"], r["condition"]) for r in thresholds["per_condition"]} == {(b, c) for b in banks for c in conditions}
    assert len(thresholds["per_condition"]) == len(banks) * len(conditions)
    for row in thresholds["per_condition"]:
        assert row["panels"] == len(panels) and len(row["per_panel"]) == len(panels)
        assert {r["panel"] for r in row["per_panel"]} == {p["id"] for p in panels}
        for panel in row["per_panel"]:
            group = [r for r in thresholds["per_seed"] if (r["bank_id"], r["condition"], r["panel"]) == (row["bank_id"], row["condition"], panel["panel"])]
            assert panel["complete_seeds"] == len(group) == len(seeds)
            assert panel["all_seed_success_reference_met"] == all(r["success_reference_met"] for r in group)
            assert panel["all_seed_efficiency_reference_met"] == all(r["efficiency_reference_met"] for r in group)
        assert row["success_reference_panels"] == sum(r["all_seed_success_reference_met"] for r in row["per_panel"])
        assert row["efficiency_reference_panels"] == sum(r["all_seed_efficiency_reference_met"] for r in row["per_panel"])


def validate_provenance(study, archives, result, supports, initial, snapshots, counts, rows, refs):
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
        metadata_name = "panels.json" if name in ("panel_evaluation", "bank_replication") else "dataset_metadata.json"
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
    old_banks = read_json(archives["bank_replication"] / "banks.json")
    for bank in banks:
        bank_id = bank["bank_id"]
        old = next(r for r in old_banks if r["bank_id"] == bank_id)
        assert bank["collection"] == {**old["collection"], "source": "inherited_archive_not_new_collection"}
        assert bank["status"] == "complete" and bank["support_size"] == len(supports[bank_id])
        assert bank["intersection"] == {"states": len(supports[bank_id]), "union_states": len(supports[bank_id]), "fraction_of_each": 1., "jaccard": 1.}
        for condition in ("collected_unique", "map_balanced"):
            bank_audit.check_array(supports[bank_id], bank["arrays"][condition])
            current = next(r for r in bank["coverage"]["per_condition"] if r["condition"] == condition)
            original = next(r for r in old["coverage"]["per_condition"] if r["condition"] == "collected_unique")
            assert {k: v for k, v in current.items() if k != "condition"} == {k: v for k, v in original.items() if k != "condition"}
    assert {r["bank_id"] for r in provenance["support_integrity"]} == set(protocol["bank_ids"])
    assert len(provenance["support_integrity"]) == len(protocol["bank_ids"])
    for row in provenance["support_integrity"]:
        assert row["unchanged"] and row["read_only"] and row["before"] == row["after"]
        for spec in row["before"].values():
            bank_audit.check_array(supports[row["bank_id"]], spec)
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
    assert {r["seed"] for r in provenance["paired_map_sampling_consistency"]} == set(protocol["seeds"])
    assert len(provenance["paired_map_sampling_consistency"]) == len(protocol["seeds"])
    for row in provenance["paired_map_sampling_consistency"]:
        assert row["fits"] == len(protocol["bank_ids"])
        assert row["complete"] and row["map_digest_identical"] and row["map_counts_identical"]
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
    parser.add_argument("--study", type=Path, default=Path("experiments/map_replay/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true")
    archive_names = ("supervised", "fixed_targets", "coverage", "equal_support", "panel_evaluation", "bank_replication")
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
    assert protocol["id"] == "map-replay-v1" and result["run"]["status"] == "complete"
    assert protocol["bank_ids"] == [1, 2, 3]
    assert [r["id"] for r in protocol["conditions"]] == ["collected_unique", "map_balanced"]
    assert protocol["primary_comparison"] == "balanced_minus_collected"
    final = max(protocol["checkpoints"])
    assert protocol["evaluation_checkpoints"] == {"collected_unique": 30000, "map_balanced": final}
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2] and protocol["git"]["dirty"] is False
        assert protocol["runtime"]["python"].startswith("3.12.")
        assert protocol["runtime"]["torch"].split("+")[0] == "2.8.0" and protocol["runtime"]["numpy"] == "2.0.2"
        assert protocol["panel_selection"] == {"count": 8, "maps_per_panel": 64, "start": 1040000, "stride": 1000}
        assert protocol["checkpoints"] == [0, 1000, 3000, 10000, 30000]
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig()) and protocol["rule_visibility"] == "observed"
    assert all(protocol["collection"][k] == 0 for k in ("new_steps", "new_episodes", "new_support_draws"))
    assert protocol["evaluation"]["final_only"] and protocol["evaluation"]["after_all_treatment_fits"]
    assert protocol["evaluation"]["greedy_repetitions"] == 1 and protocol["evaluation"]["epsilon_0_1_repetitions"] == 2
    assert protocol["evaluation"]["optimal_q_atol"] == 1e-6 and protocol["evaluation"]["optimal_q_rtol"] == 0
    assert protocol["sampling"]["map_rng"] == "SeedSequence([seed,99301])"
    assert protocol["sampling"]["state_rng"] == "SeedSequence([seed,99302])"
    assert protocol["sampling"]["map_schedule_shared_across_banks"] and protocol["sampling"]["full_bank_detached_successors"]
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
    selection, task_hashes, planner = reconstruct_panels(study, archives, protocol, result, metadata, CollectionWorld, config)
    states, agents, initial, snapshots = validate_models(study, archives["bank_replication"], result)
    counts, map_digests = validate_sampling(study, archives["bank_replication"], result, data, supports)
    exposure_maps, exposure_clocks = validate_exposure(study, result, data, transitions, supports, counts)
    loss_rows = validate_losses(study, result)
    rows, refs, paired = validate_episode_tables(study, result, states, task_hashes, planner)
    validate_thresholds(result, eligible=not smoke)
    source_inputs = validate_provenance(study, archives, result, supports, initial, snapshots, counts, rows, refs)
    replay_count, replay_steps = validate_replays(study, result, rows, refs, agents, CollectionWorld, config,
                                                skip_forward=args.skip_forward_inference)
    run, budget = result["run"], protocol["budget"]
    fits = {(b, s) for b in protocol["bank_ids"] for s in protocol["seeds"]}
    assert run["fits"] == run["frozen_baselines"] == len(fits) and len(states) == 2 * len(fits)
    assert run["banks"] == len(supports) and run["model_parameter_count"] == 20420
    assert run["train_updates"] == len(fits) * final == budget["maximum_updates"] and budget["updates_per_fit"] == final
    assert run["training_examples"] == run["train_updates"] * 64 and run["historical_baseline_updates"] == len(fits) * 30000
    assert all(run[k] == 0 for k in ("baseline_new_updates", "collection_episodes", "collection_steps", "support_draws"))
    assert all(budget[k] == 0 for k in ("new_baseline_updates", "maximum_collection_steps", "maximum_collection_episodes"))
    assert run["learner_episodes"] == len(rows) and run["reference_episodes"] == run["unique_reference_episodes"] == len(refs)
    assert run["panels"] == len(selection["panels"])
    assert 0 < run["wall_seconds"] <= budget["admission_seconds"] <= 1200
    assert 0 < run["peak_rss_bytes"] <= budget["peak_process_rss_bytes"] == 4 * 1024 ** 3
    assert run["resource_checks"] > run["train_updates"] + len(rows) + len(refs)
    assert run["resource_limits"] == {"seconds": budget["admission_seconds"], "peak_process_rss_bytes": budget["peak_process_rss_bytes"], "torch_threads": 1}
    assert protocol["runtime"]["torch_threads"] == 1 and budget["all_phases_included"]
    assert run["interpretation"] == ("smoke_or_deviation_descriptive_only" if smoke else "map_replay_descriptive_only") and run["stop_reason"] is None
    completed = [r for r in run["progress"] if r.get("training_complete")]
    assert {(r["bank_id"], r["seed"]) for r in completed} == fits and len(completed) == len(fits)
    assert all(r["updates"] == final for r in completed)
    progress = [r for r in run["progress"] if r.get("snapshot_saved")]
    assert {(r["bank_id"], r["seed"], r["checkpoint"]) for r in progress} == {(b, s, cp) for b, s in fits for cp in protocol["checkpoints"]}
    assert len(progress) == len(fits) * len(protocol["checkpoints"]) and all(r["condition"] == "map_balanced" for r in progress)
    evaluated = [r for r in run["progress"] if r.get("evaluation_complete")]
    expected_evaluated = {(b, c, s, p["id"]) for b, s in fits for c in ("collected_unique", "map_balanced") for p in result["panels"]}
    assert {(r["bank_id"], r["condition"], r["seed"], r["panel"]) for r in evaluated} == expected_evaluated
    assert len(evaluated) == len(expected_evaluated)
    assert len(run["progress"]) == len(progress) + len(completed) + len(evaluated)
    report.update(training_states=len(data["observations"]), reused_support_sizes={str(b): len(s) for b, s in supports.items()},
        selected_layouts=len(task_hashes), rejected_candidates=len(selection["collision_skips"]), transition_validation=transition_report,
        archive_input_hashes_checked=source_inputs, inherited_counts_and_controls_identical=True,
        treatment_sampling_streams_checked=4 * len(fits), map_schedule_groups_checked=len(map_digests),
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
