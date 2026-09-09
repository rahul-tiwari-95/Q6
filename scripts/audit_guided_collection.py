"""Independent audit of frozen-guided versus original random experience.

Reconstructs saved collection decisions and recorded-only learning inputs,
checks sampler exposure and frozen controls, and audits saved evaluation
evidence. No model fitting or full policy evaluation is repeated.
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
import audit_panel_evaluation as replay_audit
import audit_logged_graph as graph_audit
import audit_familiar_starts as familiar_audit
import audit_constrained_bootstrap as constrained_audit
from audit_fixed_targets_study import close, csv_rows, read_json

CONDITIONS = ("constrained_bootstrap", "guided_collection")


def reconstruct_sampling(support, map_seeds, seed, updates):
    """Uniform distinct local rows per batch; different supports need not pair."""
    support = np.asarray(support)
    assert support.dtype == np.int32 and support.ndim == 1 and len(support) >= 64
    assert np.all(support[1:] > support[:-1]) and support[0] >= 0 and support[-1] < len(map_seeds)
    assert updates >= 0 and int(updates) == updates and seed >= 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
    map_ids = np.unique(map_seeds)
    local_counts = np.zeros(len(support), np.uint32)
    global_counts = np.zeros(len(map_seeds), np.uint32)
    map_counts = np.zeros(len(map_ids), np.uint32)
    digests = {name: hashlib.sha256() for name in ("local", "global", "map")}
    for _ in range(updates):
        local = rng.choice(len(support), 64, replace=False)
        rows = support[local]
        maps = map_seeds[rows]
        local_counts[local] += 1
        global_counts[rows] += 1
        np.add.at(map_counts, np.searchsorted(map_ids, maps), 1)
        for name, values in (("local", local), ("global", rows), ("map", maps)):
            digests[name].update(values.astype("<i8").tobytes())
    assert np.array_equal(global_counts[support], local_counts)
    assert int(global_counts.sum()) == int(local_counts.sum()) == int(map_counts.sum()) == updates * 64
    return {"local_counts": local_counts, "global_counts": global_counts, "map_counts": map_counts,
            "map_ids": map_ids, "digests": {name: digest.hexdigest() for name, digest in digests.items()}}


def episode_initial_rows(data, layouts):
    lookup = {(int(m), int(p[0]), int(p[1]), int(t)): i for i, (m, p, t) in
              enumerate(zip(data["map_seeds"], data["positions"], data["remaining"]))}
    assert len(lookup) == len(data["observations"])
    rows = {item["map_seed"]: lookup[(item["map_seed"], *item["original_start"], 32)] for item in layouts}
    assert len(rows) == len(layouts)
    return rows


STEP_FIELDS = ("map_seed", "repetition", "step", "current_row", "action", "reward", "next_row", "terminated", "truncated", "remaining")


def typed_step(row):
    return {k: float(row[k]) if k == "reward" else int(row[k]) for k in STEP_FIELDS}


def audit_mixed_collection(bank_path, original_path, bank_id, data, layouts, transitions, worlds, collector, canonical_guided, *, skip_forward):
    """Replay saved evidence arithmetically; never select a new trajectory."""
    initial = episode_initial_rows(data, layouts)
    selected_random = {}
    with Path(original_path).open(newline="") as source:
        for raw in csv.DictReader(source):
            step = typed_step(raw)
            if step["repetition"] < 8:
                key = step["map_seed"], step["repetition"], step["step"]
                assert key not in selected_random
                selected_random[key] = step
    summary = read_json(bank_path / "collection.json")
    episodes = csv_rows(bank_path / "collection_episodes.csv")
    maps = [r["map_seed"] for r in layouts]
    expected = [(m, rep) for m in maps for rep in range(16)]
    assert [(int(r["map_seed"]), int(r["repetition"])) for r in episodes] == expected
    episode_lookup = {(int(r["map_seed"]), int(r["repetition"])): r for r in episodes}
    visited = np.zeros(len(data["observations"]), np.uint32)
    component_visits = {kind: np.zeros_like(visited) for kind in ("random", "guided")}
    calculated, collection_order, old_steps_seen = {}, [], set()
    next_rows, rng, digests, guided_steps = {}, {}, {}, {}
    total_steps, new_forward_rows = 0, 0
    with (bank_path / "collection_steps.csv").open(newline="") as stream:
        for raw in csv.DictReader(stream):
            step = typed_step(raw)
            map_seed, rep = step["map_seed"], step["repetition"]
            key = map_seed, rep
            assert key in episode_lookup
            policy = "random" if rep < 8 else "guided"
            assert raw["policy"] == policy
            if key not in calculated:
                collection_order.append(key)
                calculated[key] = {"map_seed": map_seed, "repetition": rep, "policy": policy, "steps": 0, "success": 0, "terminated": 0,
                    "truncated": 0, "complete": 0, "base_return": 0., "shaped_return": 0., "noop_steps": 0}
                next_rows[key] = initial[map_seed]
                rng[key] = np.random.default_rng(np.random.SeedSequence([map_seed, rep, 77301, bank_id])) if policy == "random" else None
                digests[key] = hashlib.sha256()
                guided_steps[key] = []
            current = next_rows[key]
            item = calculated[key]
            assert current >= 0 and step["current_row"] == current and step["step"] == item["steps"] + 1
            assert step["remaining"] == int(data["remaining"][current]) == 33 - step["step"]
            assert data["map_seeds"][current] == map_seed
            action = step["action"]
            assert 0 <= action < 4
            if policy == "random":
                assert action == int(rng[key].integers(4))
                assert all(raw[f"q{a}"] == "" for a in range(4))
                old_key = map_seed, rep, step["step"]
                assert old_key not in old_steps_seen and step == selected_random[old_key]
                old_steps_seen.add(old_key)
            else:
                saved_q = np.asarray([float(raw[f"q{a}"]) for a in range(4)], np.float32)
                assert np.isfinite(saved_q).all() and action == int(saved_q.argmax())
                proof = (current, action, tuple(float(q) for q in saved_q))
                guided_steps[key].append(proof)
                if rep == 8 and map_seed not in canonical_guided and not skip_forward:
                    with torch.no_grad():
                        q = collector.online(torch.from_numpy(data["observations"][current]).unsqueeze(0))[0].numpy()
                    assert np.array_equal(q, saved_q), ("saved collector forward", bank_id, map_seed, step["step"])
                    new_forward_rows += 1
            terminated, truncated = bool(transitions["terminated"][current, action]), bool(transitions["truncated"][current, action])
            following = int(transitions["successor_indices"][current, action])
            assert step["terminated"] == int(terminated) and step["truncated"] == int(truncated) and not (terminated and truncated)
            assert step["next_row"] == following and step["reward"] == float(transitions["rewards"][current, action])
            position = tuple(int(v) for v in data["positions"][current])
            after = worlds[map_seed]["following"][position, action]
            item.update(steps=step["step"], success=int(terminated), terminated=int(terminated), truncated=int(truncated), complete=int(terminated or truncated))
            item["base_return"] += -.01 + float(terminated)
            item["shaped_return"] += step["reward"]
            item["noop_steps"] += int(position == after)
            visited[current] += 1
            component_visits[policy][current] += 1
            next_rows[key] = following
            total_steps += 1
            digests[key].update(json.dumps({k: step[k] for k in STEP_FIELDS if k != "repetition"}, sort_keys=True, separators=(",", ":")).encode())
            if terminated or truncated:
                item["trajectory_sha256"] = digests[key].hexdigest()
                for field, value in item.items():
                    saved = episode_lookup[key][field]
                    if isinstance(value, str):
                        assert saved == value
                    else:
                        close(saved, value, ("collection episode", bank_id, key, field))
                if policy == "guided":
                    route = guided_steps[key]
                    if map_seed not in canonical_guided:
                        assert rep == 8
                        canonical_guided[map_seed] = {"proof": route, "trajectory_sha256": item["trajectory_sha256"]}
                    assert canonical_guided[map_seed]["proof"] == route and canonical_guided[map_seed]["trajectory_sha256"] == item["trajectory_sha256"]
    assert collection_order == expected and set(old_steps_seen) == set(selected_random)
    assert all(r == -1 for r in next_rows.values()) and all(r["complete"] == 1 for r in calculated.values())
    assert np.array_equal(visited, component_visits["random"] + component_visits["guided"])
    with np.load(bank_path / "collection.npz") as saved:
        assert set(saved.files) == {"visited_counts", "support_indices"}
        assert np.array_equal(saved["visited_counts"], visited) and saved["visited_counts"].dtype == np.uint32
        support = np.flatnonzero(visited).astype(np.int32)
        assert np.array_equal(saved["support_indices"], support) and saved["support_indices"].dtype == np.int32
    for name, value in (("visited_counts", visited), ("support_indices", support)):
        bank_audit.check_array(value, summary["arrays"][name])
    assert summary["status"] == "complete" and summary["stop_reason"] is None and summary["bank_id"] == bank_id
    assert summary["collection_episodes"] == summary["complete_collection_episodes"] == len(expected)
    assert summary["collection_steps"] == total_steps == int(visited.sum())
    assert summary["collection_successes"] == sum(r["success"] for r in calculated.values())
    assert summary["collection_noop_steps"] == sum(r["noop_steps"] for r in calculated.values())
    assert summary["episodes_per_map"] == 16 and summary["random_episodes_per_map"] == 8
    assert summary["collector_before"] == summary["collector_after"] and summary["collector_parameters_rng_unchanged"]
    assert {r["policy"] for r in summary["per_policy"]} == {"random", "guided"} and len(summary["per_policy"]) == 2
    for part in summary["per_policy"]:
        group = [r for r in calculated.values() if r["policy"] == part["policy"]]
        expected_part = {"policy": part["policy"], "episodes": len(group), "complete_episodes": len(group), "steps": sum(r["steps"] for r in group),
            "successes": sum(r["success"] for r in group), "noop_steps": sum(r["noop_steps"] for r in group),
            "unique_trajectories": len({r["trajectory_sha256"] for r in group}), "mean_steps": sum(r["steps"] for r in group) / len(group)}
        familiar_audit.compare(part, expected_part)
    assert len(summary["guided_duplicate_checks"]) == len(maps)
    assert {r["map_seed"] for r in summary["guided_duplicate_checks"]} == set(maps)
    for item in summary["guided_duplicate_checks"]:
        assert item["map_seed"] in maps and item["episodes"] == 8 and item["unique_trajectories"] == 1
    return {"visited_counts": visited, "support_indices": support, "episodes": list(calculated.values()), "component_visits": component_visits,
            "random_steps_checked": len(selected_random), "steps_checked": total_steps, "unique_collector_forward_rows": new_forward_rows}


def reconstruct_panels(study, archives, protocol, result, metadata, World, config):
    excluded = {}
    for name, directory in archives.items():
        if (directory / "panels.json").is_file():
            previous = read_json(shared.checked_archive_file(directory, "panels.json"))
            layouts = [item for panel in previous["panels"] for item in panel["layouts"]]
        else:
            previous = read_json(shared.checked_archive_file(directory, "dataset_metadata.json"))
            layouts = previous["heldout"]
        excluded[f"previous_{name}_fresh_layout"] = {item["layout_hash"] for item in layouts}
        if name == "coverage":
            excluded["training_layout"] = {item["layout_hash"] for item in metadata["train"]}
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
                payload = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(), "position": list(env.position), "config": asdict(config)}
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
    final, control, treatment = max(schedule), *CONDITIONS
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: Path(p).resolve() if Path(p).is_absolute() else (root / p).resolve()
    snapshot_rows = read_json(study / "snapshots.json")
    expected_snapshots = {(b, treatment, s, cp) for b in banks for s in seeds for cp in schedule}
    expected_snapshots |= {(b, control, s, 30000) for b in banks for s in seeds}
    assert {(r["bank_id"], r["condition"], r["seed"], r["checkpoint"]) for r in snapshot_rows} == expected_snapshots
    assert len(snapshot_rows) == len(expected_snapshots)
    files = {f"models/bank{b}_{c}_seed{s}_update{cp}.pt" for b, c, s, cp in expected_snapshots}
    assert {p.relative_to(study).as_posix() for p in (study / "models").glob("*.pt")} == files
    initial, states, agents = {}, {}, {}
    for bank in banks:
        for seed in seeds:
            path = shared.checked_archive_file(archive, f"models/bank{bank}_{control}_seed{seed}_update0.pt")
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
        assert row["historical"] == (condition == control)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["optimizer_updates"] == checkpoint and state["observation_size"] == 92
        assert replay_audit.tensor_hash(state["online"]) == state["parameter_hash"] == row["online_hash"]
        assert replay_audit.tensor_hash(state["target"]) == state["target_parameter_hash"] == row["target_hash"]
        if condition == control:
            assert common.sha(path) == common.sha(shared.checked_archive_file(archive, name))
        elif checkpoint == 0:
            assert state["parameter_hash"] == initial[bank, seed]["parameter_hash"]
            assert state["target_parameter_hash"] == initial[bank, seed]["target_parameter_hash"]
        if condition == control or checkpoint == final:
            states[bank, condition, seed] = state
            agents[bank, condition, seed] = bank_audit.bare_network(state["online"])
    records = read_json(study / "models.json")
    assert records == result["provenance"]["models"]
    expected_models = {(b, c, s) for b in banks for c in CONDITIONS for s in seeds}
    assert {(r["bank_id"], r["condition"], r["seed"]) for r in records} == expected_models and len(records) == len(expected_models)
    for row in records:
        bank, condition, seed = row["bank_id"], row["condition"], row["seed"]
        state = states[bank, condition, seed]
        assert row["checkpoint"] == state["optimizer_updates"] == (30000 if condition == control else final)
        assert row["initial_online_hash"] == initial[bank, seed]["parameter_hash"]
        assert row["initial_target_hash"] == initial[bank, seed]["target_parameter_hash"]
        source = archive / row["saved"] if condition == control else study / row["saved"]
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
                assert row["condition"] == CONDITIONS[1]
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
        assert row["condition"] == CONDITIONS[1] and row["seeds"] == len(group) == len(protocol["seeds"])
        assert all(row["updates_in_window"] == int(r["updates_in_window"]) for r in group)
        close(row["mean_loss"], np.mean([float(r["mean_loss"]) for r in group]))
    assert result["provenance"]["loss_integrity"] == {"expected_windows": len(expected), "actual_windows": len(rows), "complete": True, "finite": True}
    return len(rows)


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
            observed = constrained_audit.recorded_exposure(recorded[bank, condition], supports[bank, condition], counts[group_key])
            live, outside = observed["nonterminal_action_presentations"], observed["outside_support_successor_presentations"]
            assert summary["successor_queries"]["nonterminal_queries"] == live
            assert summary["successor_queries"]["outside_support_queries"] == outside == 0
            close(summary["successor_queries"]["outside_support_fraction"], outside / live)
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
        table = tables[bank, condition]
        observed, ended, successor = table["observed"], table["ends"], table["successor_indices"]
        live = observed & ~ended
        outside = live & ~allowed[np.maximum(successor, 0)]
        weights = count.astype(np.uint64)
        targets = int(np.dot(weights, observed.sum(1).astype(np.uint64)))
        queries = int(np.dot(weights, live.sum(1).astype(np.uint64)))
        outside_queries = int(np.dot(weights, outside.sum(1).astype(np.uint64)))
        assert row["source"] == ("archived_baseline" if condition == CONDITIONS[0] else "new_treatment")
        assert row["updates"] == result["sampling"][str(bank)][condition][str(seed)]["updates"]
        assert row["state_presentations"] == int(count.sum()) == row["updates"] * 64
        assert row["action_target_presentations"] == targets
        assert row["terminal_action_targets"] == targets - queries
        assert row["nonterminal_action_targets"] == row["nonterminal_target_queries"] == queries
        assert row["outside_support_target_queries"] == outside_queries
        close(row["outside_support_fraction"], outside_queries / queries)
        assert outside_queries == 0
        source_rows, actions = np.nonzero(live)
        query_counts = np.zeros(len(count), np.uint64)
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

def validate_inputs(study, archives, protocol):
    metadata = read_json(study / 'dataset_metadata.json')
    arrays = dict(np.load(study / 'dataset.npz', allow_pickle=False))
    transitions = dict(np.load(study / 'transitions.npz', allow_pickle=False))
    assert metadata['status'] == 'complete' and set(arrays) == set(metadata['arrays'])
    for name, value in arrays.items():
        assert name.startswith('train_')
        bank_audit.check_array(value, metadata['arrays'][name])
    assert set(transitions) == set(metadata['transition_arrays'])
    for name, value in transitions.items():
        bank_audit.check_array(value, metadata['transition_arrays'][name])
    for archive in (archives['coverage'], archives['bank_replication']):
        assert metadata['train'] == read_json(shared.checked_archive_file(archive, 'dataset_metadata.json'))['train']
        with np.load(shared.checked_archive_file(archive, 'dataset.npz')) as old:
            assert all(np.array_equal(value, old[name]) for name, value in arrays.items())
        with np.load(shared.checked_archive_file(archive, 'transitions.npz')) as old:
            assert all(np.array_equal(value, old[name]) for name, value in transitions.items())
        old_protocol = read_json(shared.checked_archive_file(archive, 'protocol.json'))
        for name in ('world', 'learning', 'fixed_targets', 'competence', 'supervised', 'optimal'):
            key = f'q6/{name}.py'
            assert protocol['source_sha256'][key] == old_protocol['source_sha256'][key] == common.sha(shared.checked_archive_file(archive, 'source/' + key))
    key = 'q6/constrained_bootstrap.py'
    assert protocol['source_sha256'][key] == common.sha(shared.checked_archive_file(archives['constrained_bootstrap'], 'source/' + key))
    assert [r['map_seed'] for r in metadata['train']] == list(range(300000, 300256))
    data = {k.removeprefix('train_'): v for k, v in arrays.items()}
    assert len(data['observations']) == 163840
    saved = dict(np.load(study / 'supports.npz'))
    assert set(saved) == {f'{c}_bank{b}' for c in CONDITIONS for b in protocol['bank_ids']}
    supports = {(b, c): saved[f'{c}_bank{b}'] for b in protocol['bank_ids'] for c in CONDITIONS}
    for value in supports.values():
        assert value.dtype == np.int32 and np.all(value[1:] > value[:-1]) and value[0] >= 0 and value[-1] < len(data['observations'])
    with np.load(shared.checked_archive_file(archives['bank_replication'], 'supports.npz')) as old:
        for bank in protocol['bank_ids']:
            assert np.array_equal(supports[bank, CONDITIONS[0]], old[f'collected_unique_bank{bank}'])
    return data, metadata, transitions, supports


def validate_collector(study, archive, result):
    record = read_json(study / 'collector.json')
    assert record == result['collector'] == result['provenance']['collector']
    name = 'models/bank1_constrained_bootstrap_seed0_update30000.pt'
    source = shared.checked_archive_file(archive, name)
    root = Path(__file__).resolve().parents[1]
    assert (root / record['source']).resolve() == source.resolve() and record['saved'] == 'collector.pt'
    digest = common.sha(source)
    for field in ('source_before', 'copy_before', 'source_after', 'copy_after', 'source_final', 'copy_final'):
        assert record[field] == digest
    assert common.sha(study / 'collector.pt') == digest and record['unchanged'] and record['files_unchanged_through_training']
    state = torch.load(study / 'collector.pt', map_location='cpu', weights_only=True)
    assert state['optimizer_updates'] == 30000 and state['observation_size'] == 92
    for side in ('online', 'target'):
        assert record[side + '_before'] == record[side + '_after'] == replay_audit.tensor_hash(state[side])
    old = next(r for r in read_json(shared.checked_archive_file(archive, 'action_exposure.json'))['per_seed']
               if (r['bank_id'], r['condition'], r['seed']) == (1, CONDITIONS[0], 0))
    assert record['historical_interactions'] == 100878 and record['historical_updates'] == 30000
    assert record['historical_state_presentations'] == old['state_presentations'] == 1920000
    assert record['historical_action_targets'] == old['action_target_presentations'] == 2705715
    assert record['historical_terminal_action_targets'] == old['terminal_action_targets'] == 118412
    assert record['historical_nonterminal_queries'] == old['nonterminal_target_queries'] == 2587303
    assert record['knowledge_overlaps_replaced_random_slots'] is True
    return bank_audit.bare_network(state['online'])


def validate_collections(study, archives, result, data, metadata, transitions, supports, worlds, collector, skip_forward):
    banks = result['protocol']['bank_ids']
    coverage = read_json(study / 'action_coverage.json')
    assert coverage == result['action_coverage']
    fields = {'observed', 'rewards', 'ends', 'terminated', 'truncated', 'successor_indices', 'occurrences'}
    new_saved = dict(np.load(study / 'recorded_transitions.npz'))
    old_saved = dict(np.load(study / 'control_recorded_transitions.npz'))
    assert set(new_saved) == {f'bank{b}_{k}' for b in banks for k in fields}
    assert common.sha(study / 'control_recorded_transitions.npz') == common.sha(shared.checked_archive_file(archives['constrained_bootstrap'], 'recorded_transitions.npz'))
    assert {(r['bank_id'], r['condition']) for r in coverage['per_condition']} == set(supports)
    assert len(coverage['per_condition']) == len(supports)
    assert coverage['per_bank'] == [r for r in coverage['per_condition'] if r['condition'] == CONDITIONS[1]]
    tables, reports, canonical = {}, [], {}
    for bank in banks:
        old_path = archives['bank_replication'] / f'banks/bank{bank}'
        for filename in ('collection_steps.csv', 'collection_episodes.csv'):
            source = shared.checked_archive_file(archives['bank_replication'], f'banks/bank{bank}/' + filename)
            assert common.sha(study / f'control_banks/bank{bank}' / filename) == common.sha(source)
        visits, old_support, old_report = bank_audit.reconstruct_bank(old_path, bank, data, metadata['train'], transitions, 16)
        assert np.array_equal(old_support, supports[bank, CONDITIONS[0]])
        new_report = audit_mixed_collection(study / f'banks/bank{bank}', old_path / 'collection_steps.csv', bank, data, metadata['train'], transitions, worlds, collector, canonical, skip_forward=skip_forward)
        assert np.array_equal(new_report['support_indices'], supports[bank, CONDITIONS[1]])
        for condition, path, saved, frequencies in ((CONDITIONS[0], old_path, old_saved, visits),
                (CONDITIONS[1], study / f'banks/bank{bank}', new_saved, new_report['visited_counts'])):
            support = supports[bank, condition]
            table = graph_audit.reconstruct_recorded_edges(path / 'collection_steps.csv', data, transitions, support)
            assert np.array_equal(table['occurrences'].sum(1), frequencies)
            for flag in ('terminated', 'truncated'):
                assert np.array_equal(table[flag][table['observed']], transitions[flag][table['observed']])
            for name, value in table.items():
                actual = saved[f'bank{bank}_{name}']
                assert actual.dtype == value.dtype and np.array_equal(actual, value, equal_nan=True)
            summary = next(r for r in coverage['per_condition'] if (r['bank_id'], r['condition']) == (bank, condition))
            constrained_audit.validate_membership(summary, table, support)
            assert summary['source_sha256'] == common.sha(path / 'collection_steps.csv')
            tables[bank, condition] = table
        integrity = next(r for r in result['provenance']['recorded_table_integrity'] if r['bank_id'] == bank)
        summary = next(r for r in coverage['per_bank'] if r['bank_id'] == bank)
        assert integrity['before'] == integrity['after'] == summary['arrays']
        assert integrity['unchanged'] and integrity['read_only'] and integrity['optimizer_tables_unchanged']
        assert integrity['tensor_before'] == integrity['tensor_after']
        for name, value in tables[bank, CONDITIONS[1]].items():
            bank_audit.check_array(value.astype(np.float32) if name == 'rewards' else value, integrity['tensor_before'][name])
        reports.append({'bank_id': bank, 'historical_collection_steps': old_report['collection_steps'], 'new_collection_steps': new_report['steps_checked'],
            'new_complete_episodes': len(new_report['episodes']), 'random_steps_reproduced': new_report['random_steps_checked'],
            'unique_collector_forward_rows': new_report['unique_collector_forward_rows'], 'old_support': len(old_support), 'new_support': len(new_report['support_indices'])})
    assert len(canonical) == 256
    return tables, reports


def validate_routes(study, result, data, metadata, supports, tables, worlds):
    routes = read_json(study / 'route_ceilings.json')
    assert routes == result['route_ceilings'] and routes['computed_before_training'] is True
    arrays = dict(np.load(study / 'route_ceilings.npz'))
    assert set(arrays) == {f'bank{b}_{c}_{k}' for b, c in supports for k in ('reachable', 'shortest_steps')}
    starts = episode_initial_rows(data, metadata['train'])
    assert len(routes['by_start']) == len(supports) * len(starts)
    expected = []
    for (bank, condition), support in supports.items():
        shortest, _ = familiar_audit.independent_reachability(tables[bank, condition], data['remaining'], support)
        reachable = shortest >= 0
        # Independent audit uses -1 for impossible; production archive uses
        # the same explicit sentinel, including unsupported rows.
        for key, value in (('reachable', reachable), ('shortest_steps', shortest.astype(np.int32))):
            actual = arrays[f'bank{bank}_{condition}_{key}']
            assert actual.dtype == value.dtype and np.array_equal(actual, value), (bank, condition, key)
            bank_audit.check_array(actual, routes['arrays'][f'bank{bank}_{condition}_{key}'])
        by_start = []
        for map_seed, row in starts.items():
            length = int(shortest[row]) if reachable[row] else None
            planner = worlds[map_seed]['distance'][tuple(worlds[map_seed]['start'])]
            by_start.append({'bank_id': bank, 'condition': condition, 'map_seed': map_seed, 'state_row': row,
                'logged_success_reachable': bool(reachable[row]), 'logged_shortest_steps': length, 'planner_steps': planner,
                'logged_efficient_success_reachable': bool(length is not None and length <= 2 * planner)})
        expected.extend(by_start)
        row = next(r for r in routes['per_condition'] if (r['bank_id'], r['condition']) == (bank, condition))
        assert row['supported_states'] == len(support) and row['reachable_states'] == int(reachable[support].sum())
        assert row['unreachable_supported_states'] == len(support) - row['reachable_states'] and row['closure_verified']
        assert row['starts'] == len(starts)
        for prefix, field in (('success', 'logged_success_reachable'), ('efficient_success', 'logged_efficient_success_reachable')):
            assert row[prefix + '_reachable_starts'] == sum(r[field] for r in by_start)
            close(row[prefix + '_ceiling'], row[prefix + '_reachable_starts'] / len(starts))
    assert len(routes['per_condition']) == len(supports)
    assert sorted(routes['by_start'], key=lambda r: (r['bank_id'], r['condition'], r['map_seed'])) == sorted(expected, key=lambda r: (r['bank_id'], r['condition'], r['map_seed']))
    return len(expected)


def validate_sampling(study, archive, result, data, supports):
    saved = {kind: dict(np.load(study / filename)) for kind, filename in
             (('global', 'sample_counts.npz'), ('local', 'local_sample_counts.npz'), ('map', 'map_counts.npz'))}
    original = {kind: dict(np.load(shared.checked_archive_file(archive, filename))) for kind, filename in
             (('global', 'sample_counts.npz'), ('local', 'local_sample_counts.npz'), ('map', 'map_counts.npz'))}
    assert read_json(study / 'sampling.json') == result['sampling']
    keys = {(b, c, s) for b, c in supports for s in result['protocol']['seeds']}
    assert all(set(values) == {f'bank{b}_{c}_seed{s}' for b, c, s in keys} for values in saved.values())
    counts = {}
    old_sampling = read_json(shared.checked_archive_file(archive, 'sampling.json'))
    for bank, condition, seed in sorted(keys):
        name = f'bank{bank}_{condition}_seed{seed}'
        row = result['sampling'][str(bank)][condition][str(seed)]
        updates = 30000 if condition == CONDITIONS[0] else max(result['protocol']['checkpoints'])
        rebuilt = reconstruct_sampling(supports[bank, condition], data['map_seeds'], seed, updates)
        assert row['updates'] == updates and row['historical'] == (condition == CONDITIONS[0])
        assert row['new_updates'] == (0 if condition == CONDITIONS[0] else updates)
        assert row['examples_seen'] == updates * 64 and row['support_states'] == len(supports[bank, condition])
        assert row['rng_seed_tuple'] == [seed, 66301] and row['map_ids'] == rebuilt['map_ids'].tolist()
        for kind, digest_field in (('local', 'local_batch_index_sha256'), ('global', 'global_batch_index_sha256'), ('map', 'map_index_sha256')):
            actual, expected = saved[kind][name], rebuilt[kind + '_counts']
            assert actual.dtype == np.uint32 and np.array_equal(actual, expected)
            assert row[digest_field] == rebuilt['digests'][kind]
            if condition == CONDITIONS[0]:
                assert np.array_equal(actual, original[kind][name])
                assert row[digest_field] == old_sampling[str(bank)][condition][str(seed)][digest_field]
        assert row['unique_states_sampled'] == int(np.count_nonzero(rebuilt['global_counts']))
        assert row['outside_support_direct_samples'] == 0
        counts[bank, condition, seed] = rebuilt['global_counts']
    assert set(result['sampling']) == {str(b) for b in result['protocol']['bank_ids']}
    reconstructions = result['provenance']['baseline_sampling_reconstruction']
    expected_fits = {(b, s) for b in result['protocol']['bank_ids'] for s in result['protocol']['seeds']}
    assert len(reconstructions) == len(expected_fits) and {(r['bank_id'], r['seed']) for r in reconstructions} == expected_fits
    for record in reconstructions:
        bank, seed = record['bank_id'], record['seed']
        name = f'bank{bank}_{CONDITIONS[0]}_seed{seed}'
        assert record['reconstructed_updates'] == 30000 and record['verified']
        assert record['map_index_sha256'] == result['sampling'][str(bank)][CONDITIONS[0]][str(seed)]['map_index_sha256']
        for field in ('local_digest_identical', 'global_digest_identical', 'local_counts_identical', 'global_counts_identical', 'map_digest_identical', 'map_counts_identical'):
            assert record[field] is True
        assert set(record['count_arrays']) == {'local', 'global', 'map'}
        for kind, spec in record['count_arrays'].items():
            bank_audit.check_array(saved[kind][name], spec)
    for bank in result['protocol']['bank_ids']:
        assert set(result['sampling'][str(bank)]) == set(CONDITIONS)
        assert all(set(v) == {str(s) for s in result['protocol']['seeds']} for v in result['sampling'][str(bank)].values())
    return counts

def validate_collection_replays(study, result, data, worlds, transitions):
    traces = read_json(study / 'collection_trajectories.json')
    assert traces == result['collection_trajectories']
    selected = [300000 + 32 * i for i in range(8)]
    expected = {(b, c, m, rep) for b in result['protocol']['bank_ids'] for c in CONDITIONS for m in selected for rep in (0, 8)}
    key = lambda r: (r['bank_id'], r['condition'], r['map_seed'], r['repetition'])
    assert {key(r) for r in traces} == expected and len(traces) == len(expected)
    raw = {}
    for bank in result['protocol']['bank_ids']:
        for condition in CONDITIONS:
            folder = 'control_banks' if condition == CONDITIONS[0] else 'banks'
            with (study / folder / f'bank{bank}/collection_steps.csv').open(newline='') as f:
                for row in csv.DictReader(f):
                    k = bank, condition, int(row['map_seed']), int(row['repetition'])
                    if k in expected:
                        raw.setdefault(k, []).append(row)
    steps = 0
    for trace in traces:
        bank, condition, map_seed, rep = key(trace)
        world = worlds[map_seed]
        component = 'guided' if condition == CONDITIONS[1] and rep == 8 else 'random'
        assert trace['policy'] == 'collector_' + component and trace['policy_component'] == component
        assert trace['seed'] == bank and trace['checkpoint'] == 0 and trace['mode'] == 'collection' and trace['panel'] == 'training_collection'
        assert trace['source'] == ('archived_collection' if condition == CONDITIONS[0] else 'new_collection')
        assert trace['world'] == 'A' and trace['grid_size'] == 5 and trace['action_mapping'] == [0, 1, 2, 3]
        assert trace['walls'] == world['walls'] and trace['pellets_initial'] == world['pellets'] and trace['start'] == world['start']
        rows = raw[key(trace)]
        assert len(trace['steps']) == len(rows) and trace['complete'] is True
        for frame, row in zip(trace['steps'], rows):
            current, action = int(row['current_row']), int(row['action'])
            position = tuple(map(int, data['positions'][current]))
            after = world['following'][position, action]
            success = bool(transitions['terminated'][current, action])
            assert frame['position_before'] == list(position) and frame['position'] == list(after)
            assert frame['action'] == action and frame['current_row'] == current and frame['successor_row'] == int(row['next_row'])
            assert frame['remaining_before'] == int(data['remaining'][current])
            assert frame['terminated'] == success and frame['truncated'] == bool(transitions['truncated'][current, action])
            assert frame['pellets'] == ([] if success else world['pellets'])
            close(frame['reward'], float(row['reward']))
            close(frame['base_reward'], -.01 + success)
            if component == 'guided':
                assert frame['q_values'] == [float(row[f'q{a}']) for a in range(4)]
            else:
                assert 'q_values' not in frame
            steps += 1
        assert trace['success'] == bool(int(rows[-1]['terminated']))
    return len(traces), steps


def validate_collection_summaries(study, result, reports):
    collection = read_json(study / 'collection.json')
    assert collection == result['collection']
    banks = result['protocol']['bank_ids']
    assert [r['bank_id'] for r in collection['per_bank']] == banks
    assert len(collection['per_condition']) == len(banks) * 2
    for bank in banks:
        saved = read_json(study / f'banks/bank{bank}/collection.json')
        assert saved == next(r for r in collection['per_bank'] if r['bank_id'] == bank)
        new = next(r for r in collection['per_condition'] if (r['bank_id'], r['condition']) == (bank, CONDITIONS[1]))
        assert new == {**saved, 'condition': CONDITIONS[1], 'source': 'new_mixed_collection'}
        old = next(r for r in collection['per_condition'] if (r['bank_id'], r['condition']) == (bank, CONDITIONS[0]))
        episodes = csv_rows(study / f'control_banks/bank{bank}/collection_episodes.csv')
        for field, column in (('collection_steps','steps'), ('collection_successes','success'), ('collection_noop_steps','noop_steps'), ('complete_collection_episodes','complete')):
            assert old[field] == sum(int(r[column]) for r in episodes)
        assert old['collection_episodes'] == len(episodes) == 4096
        routes, digest, previous = set(), None, None
        for raw in csv_rows(study / f'control_banks/bank{bank}/collection_steps.csv'):
            step = typed_step(raw)
            key = step['map_seed'], step['repetition']
            if key != previous:
                if digest is not None:
                    routes.add(digest.hexdigest())
                digest, previous = hashlib.sha256(), key
            digest.update(json.dumps({k: step[k] for k in STEP_FIELDS if k != 'repetition'}, sort_keys=True, separators=(',', ':')).encode())
        routes.add(digest.hexdigest())
        expected = {'policy': 'random', 'episodes': len(episodes), 'complete_episodes': len(episodes),
            'steps': old['collection_steps'], 'successes': old['collection_successes'], 'noop_steps': old['collection_noop_steps'],
            'unique_trajectories': len(routes), 'mean_steps': old['collection_steps'] / len(episodes)}
        familiar_audit.compare(old['per_policy'], [expected])
    checks = result['provenance']['collection_integrity']
    assert checks['expected_banks'] == checks['completed_banks'] == len(banks)
    assert checks['expected_episodes'] == checks['actual_episodes'] == checks['complete_episodes'] == len(banks) * 4096
    assert checks['expected_collection_recordings'] == checks['actual_collection_recordings'] == len(banks) * 32
    for field in ('all_recordings_complete', 'all_guided_slots_identical_within_map', 'guided_routes_identical_across_banks', 'all_random_slots_match', 'route_arrays_unchanged', 'complete'):
        assert checks[field] is True
    replication = result['provenance']['random_slot_replication']
    assert len(replication) == len(banks) and {r['bank_id'] for r in replication} == set(banks)
    for row in replication:
        report = next(r for r in reports if r['bank_id'] == row['bank_id'])
        assert row['identical'] and row['compared_steps'] == report['random_steps_reproduced']
        assert row['random_slots'] == list(range(8))


def validate_provenance(study, archives, result, data, transitions, supports, initial, snapshots, counts, rows, refs):
    protocol, provenance = result['protocol'], result['provenance']
    assert read_json(study / 'provenance.json') == provenance
    assert set(provenance['archives']) == set(archives)
    source_records = {}
    for name, directory in archives.items():
        record = provenance['archives'][name]
        manifest = read_json(directory / 'manifest.json')
        assert common.sha(study / f'{name}_manifest.json') == common.sha(directory / 'manifest.json') == record['files']['manifest.json']
        for filename, digest in record['files'].items():
            assert common.sha(directory / filename) == digest
            if filename != 'manifest.json':
                assert manifest['files'][filename] == digest
            source_records[name, filename] = digest
        metadata_name = 'panels.json' if (directory / 'panels.json').exists() else 'dataset_metadata.json'
        assert common.sha(study / f'{name}_{metadata_name}') == common.sha(directory / metadata_name)
    assert {(r['archive'], r['file']) for r in provenance['source_integrity']} == set(source_records)
    assert len(provenance['source_integrity']) == len(source_records)
    for row in provenance['source_integrity']:
        assert row['before'] == row['after'] == source_records[row['archive'], row['file']] and row['unchanged']
    for field in ('training_arrays_identical', 'transition_arrays_identical', 'all_supports_frozen_before_training', 'all_treatment_fits_finished_before_evaluation'):
        assert provenance[field]
    assert provenance['snapshot_integrity'] == {'expected': len(snapshots), 'actual': len(snapshots), 'complete': True, 'unchanged': True}
    assert read_json(study / 'banks.json') == result['banks']
    assert [r['bank_id'] for r in result['banks']] == protocol['bank_ids']
    near = np.repeat(np.min(np.where(data['winnable'], data['remaining'], 33).reshape(-1, 32), axis=1) <= 2, 32)
    for bank in result['banks']:
        b = bank['bank_id']
        first, second = (supports[b, c] for c in CONDITIONS)
        assert bank['status'] == 'complete' and bank['support_size'] == len(first)
        assert bank['support_sizes'] == {c: len(supports[b, c]) for c in CONDITIONS}
        assert bank['identical_state_support'] == bool(np.array_equal(first, second))
        intersection = len(np.intersect1d(first, second))
        union = len(first) + len(second) - intersection
        familiar_audit.compare(bank['intersection'], {'states': intersection, 'union_states': union,
            'fraction_of_control': intersection / len(first), 'fraction_of_treatment': intersection / len(second), 'jaccard': intersection / union})
        assert bank['collection'] == next(r for r in result['collection']['per_bank'] if r['bank_id'] == b)
        assert set(bank['arrays']) == set(CONDITIONS)
        assert {r['condition'] for r in bank['coverage']['per_condition']} == set(CONDITIONS)
        for c in CONDITIONS:
            bank_audit.check_array(supports[b, c], bank['arrays'][c])
            coverage = next(r for r in bank['coverage']['per_condition'] if r['condition'] == c)
            bank_audit.check_composition(coverage, data, transitions, supports[b, c], near)
    assert len(provenance['support_integrity']) == len(protocol['bank_ids'])
    for row in provenance['support_integrity']:
        assert row['unchanged'] and row['read_only'] and row['before'] == row['after']
        assert set(row['before']) == set(CONDITIONS)
        for c, spec in row['before'].items():
            bank_audit.check_array(supports[row['bank_id'], c], spec)
    fits = {(b, s) for b in protocol['bank_ids'] for s in protocol['seeds']}
    assert set(provenance['prior_initial_models']) == {f'{b}:{s}' for b, s in fits}
    for b, s in fits:
        row = provenance['prior_initial_models'][f'{b}:{s}']
        assert row['online_hash'] == initial[b, s]['parameter_hash'] and row['target_hash'] == initial[b, s]['target_parameter_hash']
        assert row['sha256'] == common.sha(shared.checked_archive_file(archives['constrained_bootstrap'], f'models/bank{b}_constrained_bootstrap_seed{s}_update0.pt'))
    assert len(provenance['initialization_consistency']) == len(fits)
    for row in provenance['initialization_consistency']:
        assert (row['bank_id'], row['seed']) in fits and row['models'] == 2 and row['online_identical'] and row['target_identical']
    assert len(provenance['replay_sampling_integrity']) == len(fits)
    for row in provenance['replay_sampling_integrity']:
        b, s = row['bank_id'], row['seed']
        assert (b, s) in fits and row['paired_with_control'] is False and row['complete'] is True
        for field in ('local_global_counts_agree', 'no_direct_off_support_samples', 'map_counts_agree', 'state_presentations_match'):
            assert row[field]
        assert row['all_supported_states_sampled'] == bool(np.all(counts[b, CONDITIONS[1], s][supports[b, CONDITIONS[1]]] > 0))
    assert len(provenance['count_integrity']) == len(counts)
    for row in provenance['count_integrity']:
        assert (row['bank_id'], row['condition'], row['seed']) in counts
        assert all(row[k] for k in ('sum_matches', 'local_matches', 'map_sum_matches', 'on_support'))
    assert len(provenance['expected_cells']) == len(protocol['bank_ids'])
    for row in provenance['expected_cells']:
        assert row['complete'] and row['bank_id'] in protocol['bank_ids']
        assert row['expected_learner_cells'] == row['complete_learner_cells'] == row['unique_complete_learner_cells'] == len(rows) // len(protocol['bank_ids'])
        assert row['expected_reference_cells'] == row['complete_reference_cells'] == row['unique_complete_reference_cells'] == len(refs)
    for row in provenance['control_exposure_identity']:
        assert row['action_counts_match_archive'] and row['query_vector_matches_archive']
    assert len(provenance['control_exposure_identity']) == len(fits)
    assert len(provenance['control_recorded_identity']) == 2 * len(protocol['bank_ids'])
    for bank in protocol['bank_ids']:
        records = [r for r in provenance['control_recorded_identity'] if r['bank_id'] == bank]
        assert {tuple(sorted(r)) for r in records} == {('bank_id', 'recorded_tables_identical', 'support_identical'), ('bank_id', 'read_only', 'unchanged_after_preparation_training')}
    for row in provenance['control_recorded_identity']:
        for key, value in row.items():
            if key != 'bank_id':
                assert value is True
    return len(source_records)


def audit(study, *, allow_smoke=False, skip_forward=False):
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    study = Path(study).resolve()
    names = ('supervised', 'fixed_targets', 'coverage', 'equal_support', 'panel_evaluation', 'bank_replication', 'map_replay', 'within_map', 'recorded_actions', 'constrained_bootstrap', 'logged_graph')
    archives = {name: root / f'experiments/{name}/pilot_v1' for name in names}
    sys.path.insert(0, str(study / 'source'))
    from q6.world import CollectionWorld, WorldConfig
    torch.set_num_threads(1)
    result, protocol = read_json(study / 'results.json'), read_json(study / 'protocol.json')
    smoke = bool(protocol['smoke'] or protocol['deviations'])
    assert not smoke or allow_smoke, 'Smoke/deviating artifacts require explicit --allow-smoke.'
    assert protocol['id'] == 'guided-collection-v1' and result['run']['status'] == 'complete'
    assert [r['id'] for r in protocol['conditions']] == list(CONDITIONS)
    assert protocol['primary_comparison'] == 'guided_minus_random' and protocol['new_competence_gates'] is False
    final = max(protocol['checkpoints'])
    assert protocol['evaluation_checkpoints'] == {CONDITIONS[0]: 30000, CONDITIONS[1]: final}
    if not smoke:
        assert protocol['bank_ids'] == [1, 2, 3] and protocol['seeds'] == [0, 1, 2] and protocol['git']['dirty'] is False
        assert protocol['runtime']['python'].startswith('3.12.') and protocol['runtime']['torch'].split('+')[0] == '2.8.0' and protocol['runtime']['numpy'] == '2.0.2'
        assert protocol['panel_selection'] == {'count': 8, 'maps_per_panel': 64, 'start': 1140000, 'stride': 1000}
        assert protocol['checkpoints'] == [0, 1000, 3000, 10000, 30000]
    config = WorldConfig(**{**protocol['world'], 'action_mapping': tuple(protocol['world']['action_mapping'])})
    assert asdict(config) == asdict(WorldConfig()) and protocol['rule_visibility'] == 'observed'
    assert protocol['collection']['random_slots'] == list(range(8)) and protocol['collection']['guided_slots'] == list(range(8, 16))
    assert protocol['collection']['episodes_per_map'] == 16 and protocol['collection']['maps'] == list(range(300000, 300256))
    assert protocol['sampling']['rng'] == 'SeedSequence([seed,66301])' and protocol['sampling']['paired_local_global_map_schedule_within_bank'] is False
    assert protocol['recorded_actions']['equal_action_target_budget'] is False and protocol['recorded_actions']['all_tables_frozen_before_training']
    assert protocol['optimizer']['implementation'] == 'unchanged constrained_bootstrap.constrained_update' and protocol['optimizer']['observed_only']
    for k, v in {'learning_rate': .001, 'gradient_norm_cap': 5., 'gamma': .97, 'target_tau': .01, 'batch_size': 64}.items():
        close(protocol['optimizer'][k], v)
    assert protocol['evaluation']['final_only'] and protocol['evaluation']['after_all_treatment_fits']
    assert protocol['evaluation']['greedy_repetitions'] == 1 and protocol['evaluation']['epsilon_0_1_repetitions'] == 2
    assert protocol['evaluation']['optimal_q_atol'] == 1e-6 and protocol['evaluation']['optimal_q_rtol'] == 0
    assert protocol['dataset']['fresh_state_enumeration'] is False
    report = {'study': str(study), 'smoke_validation_only': smoke, 'manifest_files': common.audit_manifest(study, protocol, result)}
    data, metadata, saved_transitions, supports = validate_inputs(study, archives, protocol)
    transitions, worlds, lookup, clone_checks = familiar_audit.geometry(data, metadata, config, CollectionWorld)
    for key, value in saved_transitions.items():
        assert np.array_equal(value, transitions[key]), key
    collector = validate_collector(study, archives['constrained_bootstrap'], result)
    tables, collections = validate_collections(study, archives, result, data, metadata, transitions, supports, worlds, collector, skip_forward)
    route_starts = validate_routes(study, result, data, metadata, supports, tables, worlds)
    validate_collection_summaries(study, result, collections)
    selection, task_hashes, planner = reconstruct_panels(study, archives, protocol, result, metadata, CollectionWorld, config)
    states, agents, initial, snapshots = validate_models(study, archives['constrained_bootstrap'], result)
    counts = validate_sampling(study, archives['constrained_bootstrap'], result, data, supports)
    exposure_maps, exposure_clocks = validate_exposure(study, result, data, transitions, supports, counts, tables)
    totals = validate_action_exposure(study, result, data, transitions, supports, counts, tables)
    old_actions = read_json(archives['constrained_bootstrap'] / 'action_exposure.json')['per_seed']
    old_queries = dict(np.load(archives['constrained_bootstrap'] / 'recorded_query_counts.npz'))
    new_queries = dict(np.load(study / 'recorded_query_counts.npz'))
    for row in result['action_exposure']['per_seed']:
        if row['condition'] == CONDITIONS[0]:
            old = next(v for v in old_actions if all(v[k] == row[k] for k in ('bank_id', 'condition', 'seed')))
            for k in ('updates', 'state_presentations', 'action_target_presentations', 'terminal_action_targets', 'nonterminal_target_queries'):
                assert row[k] == old[k]
            key = f"bank{row['bank_id']}_{row['condition']}_seed{row['seed']}"
            assert np.array_equal(new_queries[key], old_queries[key])
    losses = validate_losses(study, result)
    rows, refs, paired = shared.validate_episode_tables(study, result, states, task_hashes, planner)
    shared.validate_thresholds(result, eligible=not smoke)
    source_inputs = validate_provenance(study, archives, result, data, transitions, supports, initial, snapshots, counts, rows, refs)
    replay_count, replay_steps = shared.validate_replays(study, result, rows, refs, agents, CollectionWorld, config, skip_forward=skip_forward)
    collection_replays, collection_replay_steps = validate_collection_replays(study, result, data, worlds, transitions)
    run, budget = result['run'], protocol['budget']
    fits = len(protocol['bank_ids']) * len(protocol['seeds'])
    assert run['resource_checks'] > run['train_updates'] + run['baseline_reconstructed_updates'] + len(rows) + len(refs)
    assert run['new_recorded_supports'] == run['banks'] == len(protocol['bank_ids']) and run['panels'] == len(selection['panels'])
    assert run['fits'] == run['frozen_baselines'] == fits and len(states) == 2 * fits and run['model_parameter_count'] == 20420
    assert run['train_updates'] == fits * final == budget['maximum_updates'] and budget['updates_per_fit'] == final
    assert run['training_examples'] == run['train_updates'] * 64
    assert run['historical_baseline_updates'] == run['baseline_reconstructed_updates'] == fits * 30000
    assert run['baseline_new_updates'] == run['support_draws'] == 0
    assert run['collection_episodes'] == run['complete_collection_episodes'] == len(protocol['bank_ids']) * 4096
    assert run['collection_steps'] == sum(r['new_collection_steps'] for r in collections)
    assert run['collection_episodes'] == budget['maximum_collection_episodes'] and run['collection_steps'] <= budget['maximum_collection_steps'] == run['collection_episodes'] * 32
    assert run['learner_episodes'] == len(rows) and run['reference_episodes'] == run['unique_reference_episodes'] == len(refs)
    assert run['collection_recordings'] == collection_replays and run['fresh_recordings'] == replay_count and run['total_recordings'] == replay_count + collection_replays
    assert 0 < run['wall_seconds'] <= budget['admission_seconds'] <= 1200
    assert 0 < run['peak_rss_bytes'] <= budget['peak_process_rss_bytes'] == 4 * 1024 ** 3
    assert run['resource_limits'] == {'seconds': budget['admission_seconds'], 'peak_process_rss_bytes': budget['peak_process_rss_bytes'], 'torch_threads': 1}
    assert protocol['runtime']['torch_threads'] == 1 and budget['all_phases_included']
    assert run['interpretation'] == ('smoke_or_deviation_descriptive_only' if smoke else 'guided_collection_descriptive_only') and run['stop_reason'] is None
    progress = run['progress']
    trained = [i for i, r in enumerate(progress) if r.get('training_complete')]
    evaluated = [i for i, r in enumerate(progress) if r.get('evaluation_complete')]
    assert len(trained) == fits and max(trained) < min(evaluated)
    report.update(status='passed', source_inputs=source_inputs, training_states=len(data['observations']), independently_checked_transitions=len(data['observations']) * 4,
        transition_clone_crosschecks=clone_checks, collection=collections, logged_route_starts=route_starts, model_snapshots=len(snapshots), frozen_and_new_final_models=len(states),
        independently_reconstructed_sampler_updates=(30000 + final) * fits, new_training_updates=run['train_updates'], action_exposure=totals,
        learner_episodes=len(rows), reference_episodes=len(refs), loss_windows=losses, exposure_map_rows=exposure_maps, exposure_clock_rows=exposure_clocks,
        fresh_replays=replay_count, fresh_replay_steps=replay_steps, collection_replays=collection_replays, collection_replay_steps=collection_replay_steps,
        forward_inference_checked=not skip_forward, forward_scope='unique saved guided collector route per training map and preselected saved fresh recordings only' if not skip_forward else 'omitted; saved Q/action/transition and all arithmetic/hash checks retained',
        elapsed_seconds=time.monotonic() - started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('experiments/guided_collection/pilot_v1'))
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--skip-forward-inference', action='store_true')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Assertions must be enabled; do not use python -O.')
    print(json.dumps(audit(args.study, allow_smoke=args.allow_smoke, skip_forward=args.skip_forward_inference), indent=2))


if __name__ == '__main__':
    main()
