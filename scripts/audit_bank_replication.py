"""Independent audit of paired support-bank replications, without training.

Reconstructs original collector traces from verified transition tables, uniform
support draws, local/global sampling and saved model/replay evidence. No second
complete learned-policy evaluation is performed. Portable mode skips only the
regeneration of neural outputs for preselected saved recordings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.dont_write_bytecode = True
import audit_fixed_targets_study as common
import audit_panel_evaluation as panel_audit
from audit_fixed_targets_study import close, read_json, csv_rows

CONDITIONS = ("collected_unique", "uniform_subset")
BANKS = (1, 2, 3)
DELTAS = panel_audit.DELTAS
METRICS = panel_audit.DELTA_METRICS


def check_array(array, spec):
    assert list(array.shape) == spec["shape"] and str(array.dtype) == spec["dtype"]
    assert common.array_sha(array) == spec["sha256"]


def bare_network(state):
    network = torch.nn.Sequential(torch.nn.Linear(92, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64),
                                  torch.nn.ReLU(), torch.nn.Linear(64, 4))
    network.load_state_dict(state)
    network.eval().requires_grad_(False)
    assert sum(p.numel() for p in network.parameters()) == 20420
    return SimpleNamespace(online=network)


def reconstruct_bank(bank_path, bank_id, data, layouts, transitions, episodes_per_map):
    """Check the already collected stream, using independently validated dynamics."""
    counts = np.zeros(len(data["observations"]), np.uint32)
    initial = {}
    for layout in layouts:
        rows = np.flatnonzero((data["map_seeds"] == layout["map_seed"]) & (data["remaining"] == 32)
                             & np.all(data["positions"] == layout["original_start"], axis=1))
        assert len(rows) == 1
        initial[layout["map_seed"]] = int(rows[0])
    steps, successes, noops, episodes = 0, 0, 0, 0
    with (bank_path / "collection_steps.csv").open(newline="") as sf, (bank_path / "collection_episodes.csv").open(newline="") as ef:
        recorded_steps, recorded_episodes = csv.DictReader(sf), csv.DictReader(ef)
        for layout in layouts:
            map_seed = layout["map_seed"]
            for rep in range(episodes_per_map):
                rng = np.random.default_rng(np.random.SeedSequence([map_seed, rep, 77301, bank_id]))
                row, t, shaped, episode_noops = initial[map_seed], 0, 0.0, 0
                while True:
                    action = int(rng.integers(0, 4))
                    following = int(transitions["successor_indices"][row, action])
                    reward = float(transitions["rewards"][row, action])
                    ended = bool(transitions["ends"][row, action])
                    layers = data["observations"][row, :75].reshape(3, 5, 5)
                    r, c = map(int, data["positions"][row])
                    mapping = data["observations"][row, -16:].reshape(4, 4).argmax(1)
                    dr, dc = DELTAS[mapping[action]]
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < 5 and 0 <= cc < 5) or layers[0, rr, cc]:
                        rr, cc = r, c
                    success = bool(layers[1, rr, cc])
                    expected = {"map_seed": map_seed, "repetition": rep, "step": t + 1, "current_row": row,
                        "action": action, "reward": reward, "next_row": following, "terminated": int(success),
                        "truncated": int(ended and not success), "remaining": 32 - t}
                    actual = next(recorded_steps)
                    assert set(actual) == set(expected)
                    for name, value in expected.items():
                        close(actual[name], value, "collector " + name)
                    counts[row] += 1
                    steps += 1
                    t += 1
                    episode_noops += (rr, cc) == (r, c)
                    shaped += reward
                    assert ended == (success or t == 32)
                    if ended:
                        actual = next(recorded_episodes)
                        expected = {"map_seed": map_seed, "repetition": rep, "steps": t, "success": int(success),
                            "terminated": int(success), "truncated": int(not success), "complete": 1,
                            "base_return": int(success) - .01 * t, "shaped_return": shaped, "noop_steps": episode_noops}
                        assert set(actual) == set(expected)
                        for name, value in expected.items():
                            close(actual[name], value, "collector episode " + name)
                        episodes += 1
                        successes += success
                        noops += episode_noops
                        break
                    row = following
        assert next(recorded_steps, None) is None and next(recorded_episodes, None) is None
    support = np.flatnonzero(counts).astype(np.int32)
    saved = dict(np.load(bank_path / "collection.npz", allow_pickle=False))
    assert np.array_equal(saved["visited_counts"], counts) and np.array_equal(saved["support_indices"], support)
    assert episodes == len(layouts) * episodes_per_map and steps <= episodes * 32
    assert counts.sum() == steps and len(support) >= 64
    return counts, support, {"bank_id": bank_id, "collection_episodes": episodes, "collection_steps": steps,
        "collection_successes": int(successes), "collection_noop_steps": int(noops), "unique_current_states": len(support)}


def check_composition(record, data, transitions, support, near, counts=None):
    n = len(data["observations"])
    mask = np.zeros(n, bool)
    mask[support] = True
    assert record["total_training_states"] == n and record["unique_current_states"] == len(support)
    def fraction(row, selected):
        win, nearby = selected & data["winnable"], selected & near
        expected = {"states": int(selected.sum()), "visited_states": int((selected & mask).sum()),
            "winnable_states": int(win.sum()), "visited_winnable_states": int((win & mask).sum()),
            "goal_near_states": int(nearby.sum()), "visited_goal_near_states": int((nearby & mask).sum())}
        for key, value in expected.items():
            assert row[key] == value, key
        if counts is None:
            assert "visits" not in row
        else:
            assert row["visits"] == int(counts[selected].sum())
        for selected_group, key in ((selected, "coverage_rate"), (win, "winnable_coverage_rate"), (nearby, "goal_near_coverage_rate")):
            if selected_group.any():
                close(row[key], mask[selected_group].mean(), key)
            else:
                assert row[key] is None
    fraction(record["overall"], np.ones(n, bool))
    assert {r["map_seed"] for r in record["by_map"]} == set(map(int, np.unique(data["map_seeds"])))
    assert {r["time_bucket"] for r in record["by_time_bucket"]} == set(common.BUCKETS)
    for row in record["by_map"]:
        fraction(row, data["map_seeds"] == row["map_seed"])
    for row in record["by_time_bucket"]:
        low, high = (1, 32) if row["time_bucket"] == "all" else map(int, row["time_bucket"].split("-"))
        fraction(row, (data["remaining"] >= low) & (data["remaining"] <= high))
    for key, child in (("current_state_fraction", "coverage_rate"), ("winnable_current_state_fraction", "winnable_coverage_rate"),
                       ("goal_near_current_state_fraction", "goal_near_coverage_rate")):
        close(record[key], record["overall"][child])
    live = transitions["successor_indices"][support][~transitions["ends"][support]]
    outside = live[~mask[live]]
    expected = {"all_action_transitions": len(support) * 4, "nonterminal_transitions": len(live),
        "outside_support_nonterminal_transitions": len(outside), "unique_nonterminal_destinations": len(np.unique(live)),
        "unique_outside_support_destinations": len(np.unique(outside))}
    for key, value in expected.items():
        assert record["successor_queries"][key] == value, key
    if "terminal_transitions" in record["successor_queries"]:
        assert record["successor_queries"]["terminal_transitions"] == len(support) * 4 - len(live)
    close(record["successor_queries"]["outside_support_fraction"], len(outside) / len(live))


def audit_snapshots(study, prior, protocol):
    bank_ids, seeds, checkpoints = protocol["bank_ids"], protocol["seeds"], protocol["checkpoints"]
    expected = {f"bank{b}_{c}_seed{s}_update{cp}.pt" for b in bank_ids for c in CONDITIONS for s in seeds for cp in checkpoints}
    assert {p.name for p in (study / "models").glob("*.pt")} == expected
    final, states, agents, initial_hashes = max(checkpoints), {}, {}, {}
    for seed in seeds:
        initial_path = prior / "models" / f"collected_unique_seed{seed}_update0.pt"
        assert common.sha(initial_path) == read_json(prior / "manifest.json")["files"][initial_path.relative_to(prior).as_posix()]
        archived_initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        initial_hashes[seed] = archived_initial["parameter_hash"]
        for bank in bank_ids:
            for condition in CONDITIONS:
                for checkpoint in checkpoints:
                    path = study / "models" / f"bank{bank}_{condition}_seed{seed}_update{checkpoint}.pt"
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    assert state["optimizer_updates"] == checkpoint and state["observation_size"] == 92
                    assert state["purpose"] == "inference_only_not_resumable"
                    assert panel_audit.tensor_hash(state["online"]) == state["parameter_hash"]
                    assert panel_audit.tensor_hash(state["target"]) == state["target_parameter_hash"]
                    if checkpoint == 0:
                        assert state["parameter_hash"] == initial_hashes[seed] == state["target_parameter_hash"]
                        assert all(torch.equal(state["online"][key], archived_initial["online"][key]) for key in state["online"])
                    if checkpoint == final:
                        states[bank, condition, seed] = state
                        agents[bank, condition, seed] = bare_network(state["online"])
    return states, agents, initial_hashes, len(expected)


def reconstruct_sampling(study, data, supports, protocol, result, transitions):
    sampling = read_json(study / "sampling.json")
    assert sampling == result["sampling"]
    local_arrays = dict(np.load(study / "local_sample_counts.npz", allow_pickle=False))
    global_arrays = dict(np.load(study / "sample_counts.npz", allow_pickle=False))
    expected = {f"bank{b}_{c}_seed{s}" for b in protocol["bank_ids"] for c in CONDITIONS for s in protocol["seeds"]}
    assert set(local_arrays) == set(global_arrays) == expected
    n, final = len(data["observations"]), max(protocol["checkpoints"])
    for bank in protocol["bank_ids"]:
        for seed in protocol["seeds"]:
            for condition in CONDITIONS:
                support = supports[bank, condition]
                local_counts, global_counts = np.zeros(len(support), np.uint32), np.zeros(n, np.uint32)
                local_digest, global_digest = hashlib.sha256(), hashlib.sha256()
                rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
                for _ in range(final):
                    local = rng.choice(len(support), 64, replace=False)
                    actual = support[local].astype(np.int64)
                    local_counts[local] += 1
                    global_counts[actual] += 1
                    local_digest.update(local.astype("<i8").tobytes())
                    global_digest.update(actual.astype("<i8").tobytes())
                key = f"bank{bank}_{condition}_seed{seed}"
                assert local_arrays[key].dtype == np.uint32 and global_arrays[key].dtype == np.uint32
                assert np.array_equal(local_arrays[key], local_counts) and np.array_equal(global_arrays[key], global_counts)
                row = sampling[str(bank)][condition][str(seed)]
                assert row["updates"] == final and row["examples_seen"] == final * 64
                assert row["support_states"] == len(support) and row["unique_states_sampled"] == np.count_nonzero(global_counts)
                assert row["local_batch_index_sha256"] == local_digest.hexdigest()
                assert row["global_batch_index_sha256"] == global_digest.hexdigest()
                if "batch_index_sha256" in row:
                    assert row["batch_index_sha256"] == global_digest.hexdigest()
                assert row["rng_seed_tuple"] == [seed, 66301] and row["outside_support_direct_samples"] == 0
                mask = np.zeros(n, bool)
                mask[support] = True
                assert not global_counts[~mask].any(), "direct sampling outside own support"
                live = ~transitions["ends"]
                outside = live & ~mask[np.maximum(transitions["successor_indices"], 0)]
                total = int(np.dot(global_counts.astype(np.uint64), live.sum(1).astype(np.uint64)))
                external = int(np.dot(global_counts.astype(np.uint64), outside.sum(1).astype(np.uint64)))
                assert row["successor_queries"]["nonterminal_queries"] == total
                assert row["successor_queries"]["outside_support_queries"] == external
                close(row["successor_queries"]["outside_support_fraction"], external / total)
            left, right = (sampling[str(bank)][condition][str(seed)] for condition in CONDITIONS)
            assert left["local_batch_index_sha256"] == right["local_batch_index_sha256"]
            assert left["global_batch_index_sha256"] != right["global_batch_index_sha256"]
            assert np.array_equal(local_arrays[f"bank{bank}_collected_unique_seed{seed}"], local_arrays[f"bank{bank}_uniform_subset_seed{seed}"])
    return 4 * len(protocol["bank_ids"]) * len(protocol["seeds"])


def check_episode(row, states, task_hashes, planner, final):
    steps, success, map_seed = int(row["steps"]), int(row["success"]), int(row["map_seed"])
    assert 1 <= steps <= 32 and success in (0, 1) and int(row["checkpoint_complete"]) == 1
    assert row["task_hash"] == task_hashes[map_seed] and int(row["checkpoint"]) == final
    assert int(row["noop_steps"]) + int(row["moving_steps"]) == steps
    assert 0 <= int(row["moved_revisits"]) <= int(row["moving_steps"])
    assert 1 <= int(row["unique_positions"]) <= int(row["moving_steps"]) + 1
    assert 0 <= int(row["optimal_winnable_actions"]) <= int(row["winnable_steps"]) <= steps
    assert int(row["avoidable_failure_actions"]) == 1 - success
    assert float(row["action_regret_sum"]) >= 0 and float(row["q_abs_error_sum"]) >= 0
    close(row["base_return"], success - .01 * steps)
    if row["policy"] == "learner":
        model = states[int(row["bank_id"]), row["condition"], int(row["seed"])]
        assert row["parameter_hash"] == model["parameter_hash"] and int(row["q_error_steps"]) == steps
    else:
        assert row["parameter_hash"] == row["policy"] and int(row["q_error_steps"]) == 0
        assert row["reference_sample_id"] == f"{row['policy']}:{row['panel']}:{row['seed']}:{map_seed}:{row['repetition']}"
        if row["policy"] == "shortest_path":
            assert success and steps == planner[map_seed] and int(row["noop_steps"]) == int(row["moved_revisits"]) == 0


def check_loss_windows(study, protocol, result):
    losses = csv_rows(study / "losses.csv")
    final = max(protocol["checkpoints"])
    expected_steps = sorted(set(range(100, final + 1, 100)) | {cp for cp in protocol["checkpoints"] if cp})
    expected_keys = {(b, c, s, cp) for b in protocol["bank_ids"] for c in CONDITIONS for s in protocol["seeds"] for cp in expected_steps}
    keys = [(int(r["bank_id"]), r["condition"], int(r["seed"]), int(r["checkpoint"])) for r in losses]
    assert len(keys) == len(set(keys)) and set(keys) == expected_keys
    for bank in protocol["bank_ids"]:
        for condition in CONDITIONS:
            for seed in protocol["seeds"]:
                group = [r for r in losses if int(r["bank_id"]) == bank and r["condition"] == condition and int(r["seed"]) == seed]
                assert [int(r["checkpoint"]) for r in group] == expected_steps
                previous = 0
                for row in group:
                    checkpoint = int(row["checkpoint"])
                    assert int(row["updates_in_window"]) == checkpoint - previous
                    assert int(row["training_examples"]) == checkpoint * 64
                    assert np.isfinite(float(row["mean_loss"])) and float(row["mean_loss"]) >= 0
                    assert np.isfinite(float(row["last_loss"])) and float(row["last_loss"]) >= 0
                    previous = checkpoint
    expected_aggregate = {(b, c, cp) for b in protocol["bank_ids"] for c in CONDITIONS for cp in expected_steps}
    key = lambda r: (int(r["bank_id"]), r["condition"], int(r["checkpoint"]))
    assert {key(r) for r in result["loss_aggregate"]} == expected_aggregate
    assert len(result["loss_aggregate"]) == len(expected_aggregate)
    for row in result["loss_aggregate"]:
        group = [r for r in losses if key(r) == key(row)]
        assert row["seeds"] == len(group) == len(protocol["seeds"])
        assert all(row["updates_in_window"] == int(r["updates_in_window"]) for r in group)
        close(row["mean_loss"], np.mean([float(r["mean_loss"]) for r in group]))
    return len(losses)


def audit_inputs_and_panels(study, archives, protocol, result, World, config, *, smoke):
    provenance = read_json(study / "provenance.json")
    assert provenance == result["provenance"]
    metadata = read_json(study / "dataset_metadata.json")
    arrays = dict(np.load(study / "dataset.npz", allow_pickle=False))
    transitions = dict(np.load(study / "transitions.npz", allow_pickle=False))
    assert metadata["status"] == "complete" and set(arrays) == set(metadata["arrays"])
    for key, spec in metadata["arrays"].items():
        check_array(arrays[key], spec)
    for key, spec in metadata["transition_arrays"].items():
        check_array(transitions[key], spec)
    data = {key.removeprefix("train_"): value for key, value in arrays.items()}
    excluded = {}
    for name, archive in archives.items():
        record = provenance["archives"][name]
        manifest = read_json(archive / "manifest.json")
        assert manifest["status"] == "complete" and manifest["algorithm"] == "sha256"
        for path, digest in record["files"].items():
            assert common.sha(archive / path) == digest
            if path != "manifest.json":
                assert manifest["files"][path] == digest
        filename = "panels.json" if name == "panel_evaluation" else "dataset_metadata.json"
        saved = read_json(archive / filename)
        assert common.sha(study / f"{name}_{filename}") == common.sha(archive / filename)
        layouts = [r for p in saved["panels"] for r in p["layouts"]] if name == "panel_evaluation" else saved["heldout"]
        excluded[f"previous_{name}_fresh_layout"] = {r["layout_hash"] for r in layouts}
        if name == "coverage":
            excluded["training_layout"] = {r["layout_hash"] for r in saved["train"]}
            old_metadata = saved
    excluded["training_layout"].update(r["layout_hash"] for r in metadata["train"])
    if not smoke:
        assert metadata["train"] == old_metadata["train"]
        assert metadata["training_arrays_identical"] and metadata["transition_arrays_identical"]
        with np.load(archives["coverage"] / "dataset.npz", allow_pickle=False) as original:
            for key, value in arrays.items():
                assert np.array_equal(value, original[key]), key
        with np.load(archives["coverage"] / "transitions.npz", allow_pickle=False) as original:
            for key, value in transitions.items():
                assert np.array_equal(value, original[key]), key
        assert [r["map_seed"] for r in metadata["train"]] == list(range(300000, 300256))
    else:
        assert [r["map_seed"] for r in metadata["train"]] == [740000, 740001]
    old_protocol = read_json(archives["coverage"] / "protocol.json")
    for name in ("q6/world.py", "q6/learning.py", "q6/fixed_targets.py", "q6/competence.py", "q6/supervised.py"):
        assert protocol["source_sha256"][name] == old_protocol["source_sha256"][name], name
    # The collector gained a suffix; the replay sampler itself must not change.
    import ast
    class_tree = lambda text, name: ast.dump(next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == name), include_attributes=False)
    assert class_tree((study / "source/q6/coverage.py").read_text(), "SupportSampler") == class_tree((archives["coverage"] / "source/q6/coverage.py").read_text(), "SupportSampler")
    # Supply an empty heldout panel to the established train-transition auditor.
    # No fresh-state bank is enumerated or loaded by this study/audit.
    transition_data = dict(arrays)
    for key, value in data.items():
        transition_data["heldout_" + key] = np.empty((0,) + value.shape[1:], dtype=value.dtype)
    transition_report = common.audit_transitions(transition_data, {**metadata, "heldout": []}, transitions, config, World)
    selection = read_json(study / "panels.json")
    assert selection["status"] == "complete" and not selection["selection_uses_outcomes"]
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
                    for dr, dc in DELTAS:
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
    return data, metadata, transitions, selection, task_hashes, planner, transition_report


def audit_banks(study, protocol, result, data, metadata, transitions):
    banks = read_json(study / "banks.json")
    assert banks == result["banks"] and [b["bank_id"] for b in banks] == protocol["bank_ids"]
    saved = dict(np.load(study / "supports.npz", allow_pickle=False))
    assert set(saved) == {f"{c}_bank{b}" for b in protocol["bank_ids"] for c in CONDITIONS}
    near = np.zeros(len(data["observations"]), bool)
    for map_seed in np.unique(data["map_seeds"]):
        map_rows = np.flatnonzero(data["map_seeds"] == map_seed)
        for position in np.unique(data["positions"][map_rows], axis=0):
            rows = map_rows[np.all(data["positions"][map_rows] == position, axis=1)]
            if data["remaining"][rows[data["winnable"][rows]]].min() <= 2:
                near[rows] = True
    supports, reports = {}, []
    for bank in banks:
        bank_id = bank["bank_id"]
        directory = study / "banks" / f"bank{bank_id}"
        assert read_json(directory / "bank.json") == bank
        counts, collected, report = reconstruct_bank(directory, bank_id, data, metadata["train"], transitions, protocol["collection"]["episodes_per_map"])
        reports.append(report)
        assert bank["status"] == bank["collection"]["status"] == "complete"
        assert bank["support_size"] == len(collected)
        for key in ("collection_episodes", "collection_steps", "collection_successes", "collection_noop_steps", "unique_current_states"):
            assert bank["collection"][key] == report[key]
        assert bank["collection"]["complete_collection_episodes"] == report["collection_episodes"]
        assert bank["collection"]["collection_rng"] == f"SeedSequence([map_seed,repetition,77301,{bank_id}])"
        assert read_json(directory / "coverage.json") == bank["collection"]
        for name, array in (("visited_counts", counts), ("support_indices", collected)):
            check_array(array, bank["collection"]["arrays"][name])
        check_composition(bank["collection"], data, transitions, collected, near, counts)
        uniform = np.sort(np.random.default_rng(np.random.SeedSequence([88301, bank_id])).choice(len(counts), size=len(collected), replace=False)).astype(np.int32)
        for condition, expected in (("collected_unique", collected), ("uniform_subset", uniform)):
            support = saved[f"{condition}_bank{bank_id}"]
            assert support.dtype == np.int32 and np.array_equal(support, expected)
            check_array(support, bank["arrays"][condition])
            supports[bank_id, condition] = support
            record = next(r for r in bank["coverage"]["per_condition"] if r["condition"] == condition)
            check_composition(record, data, transitions, support, near)
        intersection = len(np.intersect1d(collected, uniform))
        overlap = bank["intersection"]
        assert overlap["states"] == intersection and overlap["union_states"] == 2 * len(collected) - intersection
        close(overlap["fraction_of_each"], intersection / len(collected))
        close(overlap["jaccard"], intersection / overlap["union_states"])
    return supports, reports


def audit_raw_results(study, protocol, result, states, task_hashes, planner):
    rows, refs = csv_rows(study / "evaluations.csv"), csv_rows(study / "references.csv")
    banks, seeds, panels, final = protocol["bank_ids"], protocol["seeds"], result["panels"], max(protocol["checkpoints"])
    raw_key = lambda r: (int(r["bank_id"]), r["condition"], int(r["seed"]), r["panel"], r["mode"], int(r["map_seed"]), int(r["repetition"]))
    expected = {(b, c, s, p["id"], mode, m, rep) for b in banks for c in CONDITIONS for s in seeds for p in panels
                for mode in ("greedy", "epsilon_0_1") for m in p["map_seeds"] for rep in range(1 if mode == "greedy" else 2)}
    actual = [raw_key(r) for r in rows]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    expected_refs = {(policy, s, p["id"], m, rep) for policy in ("random_actions", "shortest_path") for s in (seeds if policy == "random_actions" else [0])
                     for p in panels for m in p["map_seeds"] for rep in range(2 if policy == "random_actions" else 1)}
    ref_keys = [(r["policy"], int(r["seed"]), r["panel"], int(r["map_seed"]), int(r["repetition"])) for r in refs]
    assert len(ref_keys) == len(set(ref_keys)) and set(ref_keys) == expected_refs
    assert len({r["reference_sample_id"] for r in refs}) == len(refs)
    for row in rows + refs:
        check_episode(row, states, task_hashes, planner, final)
    assert all(r["bank_id"] == "shared" and r["condition"] == "shared" and r["mode"] == "reference" for r in refs)
    assert all(r["policy"] == "learner" for r in rows)
    panel_ids = [p["id"] for p in panels] + ["all"]
    expected_aggregate = {(b, c, p, m) for b in banks for c in CONDITIONS for p in panel_ids for m in ("greedy", "epsilon_0_1")}
    key = lambda r: (r["bank_id"], r["condition"], r["panel"], r["mode"])
    assert {key(r) for r in result["aggregate"]} == expected_aggregate and len(result["aggregate"]) == len(expected_aggregate)
    assert {key(r) + (r["seed"],) for r in result["seed_results"]} == {k + (s,) for k in expected_aggregate for s in seeds}
    assert len(result["seed_results"]) == len(expected_aggregate) * len(seeds)
    for summary in result["aggregate"] + result["seed_results"]:
        group = [r for r in rows if int(r["bank_id"]) == summary["bank_id"] and r["condition"] == summary["condition"] and r["mode"] == summary["mode"]
                 and (summary["panel"] == "all" or r["panel"] == summary["panel"]) and ("seed" not in summary or int(r["seed"]) == summary["seed"])]
        panel_audit.audit_summary(summary, group, planner)
    assert {(r["policy"], r["panel"]) for r in result["references"]} == {(p, q) for p in ("random_actions", "shortest_path") for q in panel_ids}
    assert len(result["references"]) == 2 * len(panel_ids)
    for summary in result["references"]:
        group = [r for r in refs if r["policy"] == summary["policy"] and (summary["panel"] == "all" or r["panel"] == summary["panel"])]
        panel_audit.audit_summary(summary, group, planner)
    paired = read_json(study / "paired_differences.json")
    assert paired == result["paired_differences"]
    assert paired["comparisons"] == [{"id": "uniform_minus_collected", "left": "collected_unique", "right": "uniform_subset"}]
    expected_pairs = {(b, s, p["id"], m) for b in banks for s in seeds for p in panels for m in p["map_seeds"]}
    pair_key = lambda r: (r["bank_id"], r["seed"], r["panel"], r["map_seed"])
    assert {pair_key(r) for r in paired["per_layout"]} == expected_pairs and len(paired["per_layout"]) == len(expected_pairs)
    lookup = {raw_key(r): r for r in rows}
    csv_pairs = csv_rows(study / "paired_layouts.csv")
    assert len(csv_pairs) == len(paired["per_layout"])
    for row, csv_row in zip(paired["per_layout"], csv_pairs):
        assert row["comparison"] == "uniform_minus_collected" and row["mode"] == "greedy" and row["repetition"] == 0
        assert set(row) == set(csv_row)
        for k, value in row.items():
            if isinstance(value, str):
                assert csv_row[k] == value
            else:
                close(csv_row[k], value, k)
        left, right = (lookup[row["bank_id"], c, row["seed"], row["panel"], "greedy", row["map_seed"], 0] for c in CONDITIONS)
        close(row["success_delta"], int(right["success"]) - int(left["success"]))
        close(row["steps_delta"], int(right["steps"]) - int(left["steps"]))
        close(row["efficient_success_delta"], panel_audit.efficient_rate([right], planner) - panel_audit.efficient_rate([left], planner))
        close(row["noop_rate_delta"], int(right["noop_steps"]) / int(right["steps"]) - int(left["noop_steps"]) / int(left["steps"]))
    seed_lookup = {(r["bank_id"], r["condition"], r["panel"], r["seed"]): r for r in result["seed_results"] if r["mode"] == "greedy"}
    expected_seed = {(b, s, p) for b in banks for s in seeds for p in panel_ids}
    assert {(r["bank_id"], r["seed"], r["panel"]) for r in paired["per_seed"]} == expected_seed and len(paired["per_seed"]) == len(expected_seed)
    for row in paired["per_seed"]:
        assert row["mode"] == "greedy" and row["comparison"] == "uniform_minus_collected"
        left, right = (seed_lookup[row["bank_id"], c, row["panel"], row["seed"]] for c in CONDITIONS)
        for metric in METRICS:
            close(row[metric + "_delta"], right[metric] - left[metric])
    aggregate_lookup = {(r["bank_id"], r["condition"], r["panel"]): r for r in result["aggregate"] if r["mode"] == "greedy"}
    assert {(r["bank_id"], r["panel"]) for r in paired["aggregate"]} == {(b, p) for b in banks for p in panel_ids}
    assert len(paired["aggregate"]) == len(banks) * len(panel_ids)
    for row in paired["aggregate"]:
        assert row["mode"] == "greedy" and row["comparison"] == "uniform_minus_collected" and row["seeds"] == len(seeds)
        group = [r for r in paired["per_seed"] if r["bank_id"] == row["bank_id"] and r["panel"] == row["panel"]]
        for metric in METRICS:
            close(row["mean_seed_" + metric + "_delta"], np.mean([r[metric + "_delta"] for r in group]))
        left, right = (aggregate_lookup[row["bank_id"], c, row["panel"]] for c in CONDITIONS)
        close(row["pooled_noop_rate_delta"], right["noop_rate"] - left["noop_rate"])
    return rows, refs, paired


def audit_equal_bank_aggregation(protocol, result, paired):
    pooled, robustness = result["pooled"], result["robustness"]
    panels, banks = [r["id"] for r in result["panels"]] + ["all"], protocol["bank_ids"]
    expected = {(c, p, m) for c in CONDITIONS for p in panels for m in ("greedy", "epsilon_0_1")}
    assert {(r["condition"], r["panel"], r["mode"]) for r in pooled["aggregate"]} == expected and len(pooled["aggregate"]) == len(expected)
    for row in pooled["aggregate"]:
        group = [r for r in result["aggregate"] if (r["condition"], r["panel"], r["mode"]) == (row["condition"], row["panel"], row["mode"])]
        assert row["banks"] == len(group) == len(banks) and row["seeds_per_bank"] == len(protocol["seeds"])
        assert row["episodes"] == sum(r["episodes"] for r in group)
        for metric in METRICS:
            if metric != "noop_rate":
                close(row[metric], np.mean([r[metric] for r in group]))
        close(row["mean_bank_noop_rate"], np.mean([r["noop_rate"] for r in group]))
        for count in ("noop_steps", "evaluation_steps", "successful_episodes", "successful_episode_steps", "efficient_success_episodes"):
            assert row[count] == sum(r[count] for r in group)
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
    assert robustness["primary_comparison"] == "uniform_minus_collected" and "no new competence gates" in robustness["classification"]
    assert set(robustness["metrics"]) == {m + "_delta" for m in METRICS}
    for metric in METRICS:
        values = [r["mean_seed_" + metric + "_delta"] for r in effects]
        row = robustness["metrics"][metric + "_delta"]
        assert row["banks"] == len(banks) and row["sign_tolerance"] == 1e-12
        close(row["minimum"], min(values)); close(row["maximum"], max(values)); close(row["mean"], np.mean(values))
        assert row["positive_banks"] == sum(v > 1e-12 for v in values)
        assert row["negative_banks"] == sum(v < -1e-12 for v in values)
        assert row["zero_banks"] == sum(abs(v) <= 1e-12 for v in values)


def audit_integrity_records(study, prior, protocol, result, states, initial_hashes, supports, snapshot_count):
    provenance = result["provenance"]
    records = read_json(study / "models.json")
    assert records == provenance["models"]
    expected = {(b, c, s) for b in protocol["bank_ids"] for c in CONDITIONS for s in protocol["seeds"]}
    assert {(r["bank_id"], r["condition"], r["seed"]) for r in records} == expected and len(records) == len(expected)
    for row in records:
        key = row["bank_id"], row["condition"], row["seed"]
        state = states[key]
        assert row["initial_online_hash"] == row["initial_target_hash"] == initial_hashes[row["seed"]]
        assert row["online_before"] == row["online_after"] == state["parameter_hash"]
        assert row["target_before"] == row["target_after"] == state["target_parameter_hash"]
        assert row["file_before"] == row["file_after"] == common.sha(study / row["saved"])
        assert row["unchanged_during_evaluation"]
        assert [r["panel"] for r in row["panel_checks"]] == [r["id"] for r in result["panels"]]
        for check in row["panel_checks"]:
            assert check["unchanged"]
            for name in ("online", "target", "file"):
                assert check[name + "_before"] == check[name + "_after"] == row[name + "_before"]
    snapshots = read_json(study / "snapshots.json")
    keys = [(r["bank_id"], r["condition"], r["seed"], r["checkpoint"]) for r in snapshots]
    assert len(keys) == len(set(keys)) == snapshot_count
    assert set(keys) == {(b, c, s, cp) for b, c, s in expected for cp in protocol["checkpoints"]}
    for row in snapshots:
        path = study / row["saved"]
        assert row["sha256"] == common.sha(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert row["online_hash"] == state["parameter_hash"] and row["target_hash"] == state["target_parameter_hash"]
    assert provenance["snapshot_integrity"] == {"expected": snapshot_count, "actual": snapshot_count, "complete": True, "unchanged": True}
    assert set(provenance["prior_initial_models"]) == set(map(str, protocol["seeds"]))
    for seed in protocol["seeds"]:
        row = provenance["prior_initial_models"][str(seed)]
        path = prior / "models" / f"collected_unique_seed{seed}_update0.pt"
        assert row["sha256"] == common.sha(path) and row["online_hash"] == row["target_hash"] == initial_hashes[seed]
    assert {r["seed"] for r in provenance["initialization_consistency"]} == set(protocol["seeds"])
    for row in provenance["initialization_consistency"]:
        assert row["fits"] == 2 * len(protocol["bank_ids"]) and row["online_matches_prior"] and row["target_matches_prior"]
    assert {(r["bank_id"], r["seed"]) for r in provenance["paired_sampling_consistency"]} == {(b, s) for b in protocol["bank_ids"] for s in protocol["seeds"]}
    for row in provenance["paired_sampling_consistency"]:
        assert all(row[k] for k in ("complete", "local_digest_identical", "local_counts_identical", "initial_online_identical"))
    assert {r["bank_id"] for r in provenance["support_integrity"]} == set(protocol["bank_ids"])
    for row in provenance["support_integrity"]:
        assert row["unchanged"] and row["read_only"] and row["before"] == row["after"]
        for condition in CONDITIONS:
            check_array(supports[row["bank_id"], condition], row["before"][condition])
    assert provenance["all_banks_frozen_before_training"] and provenance["all_fits_finished_before_evaluation"]
    intersections = read_json(study / "support_intersections.json")
    assert intersections == result["support_intersections"]
    expected = {(a, b, c, d) for a in protocol["bank_ids"] for b in protocol["bank_ids"] if a < b for c in CONDITIONS for d in CONDITIONS}
    assert {(r["left_bank"], r["right_bank"], r["left_condition"], r["right_condition"]) for r in intersections} == expected
    assert len(intersections) == len(expected)
    for row in intersections:
        left, right = supports[row["left_bank"], row["left_condition"]], supports[row["right_bank"], row["right_condition"]]
        intersection = len(np.intersect1d(left, right))
        assert row["states"] == intersection and row["union_states"] == len(left) + len(right) - intersection
        close(row["jaccard"], intersection / row["union_states"])
        close(row["fraction_of_left"], intersection / len(left)); close(row["fraction_of_right"], intersection / len(right))
    return len(intersections)


def audit_thresholds(protocol, result, *, smoke):
    thresholds = result["descriptive_thresholds"]
    assert "not new competence gates" in thresholds["classification"]
    assert thresholds["eligible"] == result["robustness"]["eligible"] == (not smoke)
    assert protocol["new_competence_gates"] is False and "gates" not in result
    banks, seeds, panels = protocol["bank_ids"], protocol["seeds"], result["panels"]
    expected = {(b, c, s, p["id"]) for b in banks for c in CONDITIONS for s in seeds for p in panels}
    key = lambda r: (r["bank_id"], r["condition"], r["seed"], r["panel"])
    assert {key(r) for r in thresholds["per_seed"]} == expected and len(thresholds["per_seed"]) == len(expected)
    lookup = {key(r): r for r in result["seed_results"] if r["mode"] == "greedy"}
    random = {r["panel"]: r["success_rate"] for r in result["references"] if r["policy"] == "random_actions"}
    for row in thresholds["per_seed"]:
        source = lookup[key(row)]
        close(row["success_rate"], source["success_rate"])
        close(row["efficient_success_rate"], source["efficient_success_rate"])
        close(row["panel_random_success"], random[row["panel"]])
        assert row["success_reference_met"] == bool(not smoke and source["success_rate"] >= .7 and source["success_rate"] > random[row["panel"]])
        assert row["efficiency_reference_met"] == bool(not smoke and source["efficient_success_rate"] >= .8)
    assert {(r["bank_id"], r["condition"]) for r in thresholds["per_condition"]} == {(b, c) for b in banks for c in CONDITIONS}
    assert len(thresholds["per_condition"]) == 2 * len(banks)
    for row in thresholds["per_condition"]:
        assert row["panels"] == len(panels) and len(row["per_panel"]) == len(panels)
        assert {r["panel"] for r in row["per_panel"]} == {p["id"] for p in panels}
        for item in row["per_panel"]:
            group = [r for r in thresholds["per_seed"] if (r["bank_id"], r["condition"], r["panel"]) == (row["bank_id"], row["condition"], item["panel"])]
            assert item["complete_seeds"] == len(seeds) == len(group)
            assert item["all_seed_success_reference_met"] == all(r["success_reference_met"] for r in group)
            assert item["all_seed_efficiency_reference_met"] == all(r["efficiency_reference_met"] for r in group)
        assert row["success_reference_panels"] == sum(r["all_seed_success_reference_met"] for r in row["per_panel"])
        assert row["efficiency_reference_panels"] == sum(r["all_seed_efficiency_reference_met"] for r in row["per_panel"])


def audit_replays(study, protocol, result, rows, refs, agents, World, config, *, skip_forward):
    traces = read_json(study / "trajectories.json")
    assert traces == result["trajectories"]
    key = lambda r: (str(r["bank_id"]), r["condition"], r["policy"], int(r["seed"]), r["panel"], r["mode"], int(r["map_seed"]), int(r["repetition"]))
    expected = {(str(b), c, "learner", s, p["id"], mode, p["map_seeds"][0], 0) for b in protocol["bank_ids"] for c in CONDITIONS
                for s in protocol["seeds"] for p in result["panels"] for mode in ("greedy", "epsilon_0_1")}
    expected |= {("shared", "shared", policy, protocol["seeds"][0] if policy == "random_actions" else 0, p["id"], "reference", p["map_seeds"][0], 0)
                 for policy in ("random_actions", "shortest_path") for p in result["panels"]}
    assert {key(t) for t in traces} == expected and len(traces) == len(expected)
    lookup = {key(row): row for row in rows + refs}
    steps = 0
    for trace in traces:
        agent = agents[trace["bank_id"], trace["condition"], trace["seed"]] if trace["policy"] == "learner" else None
        steps += panel_audit.check_saved_replay(trace, lookup[key(trace)], World(config), agent, skip_forward=skip_forward)
    return len(traces), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/bank_replication/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true")
    archive_names = ("supervised", "fixed_targets", "coverage", "equal_support", "panel_evaluation")
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
    assert protocol["id"] == "bank-replication-v1" and result["run"]["status"] == "complete"
    assert protocol["bank_ids"] == list(BANKS)
    assert [r["id"] for r in protocol["conditions"]] == list(CONDITIONS)
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2] and protocol["git"]["dirty"] is False
        assert protocol["runtime"]["python"].startswith("3.12.")
        assert protocol["runtime"]["torch"].split("+")[0] == "2.8.0" and protocol["runtime"]["numpy"] == "2.0.2"
        assert protocol["panel_selection"] == {"count": 8, "maps_per_panel": 64, "start": 1020000, "stride": 1000}
        assert protocol["checkpoints"] == [0, 1000, 3000, 10000, 30000]
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig())
    assert protocol["rule_visibility"] == "observed"
    assert protocol["collection"]["episodes_per_map"] == 16 and protocol["collection"]["all_pairs_frozen_before_optimization"]
    assert protocol["evaluation"]["final_only"] and protocol["evaluation"]["after_all_fits"]
    assert protocol["evaluation"]["greedy_repetitions"] == 1 and protocol["evaluation"]["epsilon_0_1_repetitions"] == 2
    assert protocol["evaluation"]["optimal_q_atol"] == 1e-6 and protocol["evaluation"]["optimal_q_rtol"] == 0
    assert protocol["evaluation_checkpoints"] == [max(protocol["checkpoints"])]
    optimizer = protocol["optimizer"]
    for key, value in {"learning_rate": .001, "gradient_norm_cap": 5., "gamma": .97, "target_tau": .01, "batch_size": 64}.items():
        close(optimizer[key], value)
    assert optimizer["paired_local_counts_within_bank"] and protocol["dataset"]["fresh_state_enumeration"] is False
    report = {"study": str(args.study), "smoke_validation_only": smoke,
              "manifest_files": common.audit_manifest(study, protocol, result)}
    data, metadata, transitions, selection, task_hashes, planner, transition_report = audit_inputs_and_panels(
        study, archives, protocol, result, CollectionWorld, config, smoke=smoke)
    supports, collection_reports = audit_banks(study, protocol, result, data, metadata, transitions)
    states, agents, initial_hashes, snapshots = audit_snapshots(study, archives["coverage"], protocol)
    sampling_streams = reconstruct_sampling(study, data, supports, protocol, result, transitions)
    loss_rows = check_loss_windows(study, protocol, result)
    rows, refs, paired = audit_raw_results(study, protocol, result, states, task_hashes, planner)
    audit_equal_bank_aggregation(protocol, result, paired)
    intersection_count = audit_integrity_records(study, archives["coverage"], protocol, result, states, initial_hashes, supports, snapshots)
    audit_thresholds(protocol, result, smoke=smoke)
    replay_count, replay_steps = audit_replays(study, protocol, result, rows, refs, agents, CollectionWorld, config,
                                              skip_forward=args.skip_forward_inference)
    run, budget = result["run"], protocol["budget"]
    banks, seeds, final = protocol["bank_ids"], protocol["seeds"], max(protocol["checkpoints"])
    fit_keys = {(b, c, s) for b in banks for c in CONDITIONS for s in seeds}
    assert run["banks"] == len(banks) and run["fits"] == len(states) == len(fit_keys)
    assert run["model_parameter_count"] == 20420
    assert run["train_updates"] == len(fit_keys) * final == budget["maximum_updates"]
    assert run["training_examples"] == run["train_updates"] * 64 and budget["updates_per_fit"] == final
    assert run["collection_episodes"] == sum(r["collection_episodes"] for r in collection_reports) == budget["maximum_collection_episodes"]
    assert run["collection_steps"] == sum(r["collection_steps"] for r in collection_reports) <= budget["maximum_collection_steps"]
    assert budget["maximum_collection_steps"] == budget["maximum_collection_episodes"] * config.horizon
    assert run["learner_episodes"] == len(rows) and run["reference_episodes"] == run["unique_reference_episodes"] == len(refs)
    assert run["panels"] == len(selection["panels"])
    assert 0 < run["wall_seconds"] <= budget["admission_seconds"] <= 1200
    assert 0 < run["peak_rss_bytes"] <= budget["peak_process_rss_bytes"] == 4 * 1024 ** 3
    assert run["resource_checks"] > run["train_updates"] + run["collection_steps"] + len(rows) + len(refs)
    assert run["resource_limits"] == {"seconds": budget["admission_seconds"], "peak_process_rss_bytes": budget["peak_process_rss_bytes"], "torch_threads": 1}
    assert protocol["runtime"]["torch_threads"] == 1 and budget["all_phases_included"]
    assert run["interpretation"] == ("smoke_or_deviation_descriptive_only" if smoke else "independent_bank_replication_descriptive_only")
    assert run["stop_reason"] is None
    completed = [r for r in run["progress"] if r.get("training_complete")]
    assert {(r["bank_id"], r["condition"], r["seed"]) for r in completed} == fit_keys
    assert len(completed) == len(fit_keys) and all(r["updates"] == final for r in completed)
    snapshot_progress = [r for r in run["progress"] if r.get("snapshot_saved")]
    assert {(r["bank_id"], r["condition"], r["seed"], r["checkpoint"]) for r in snapshot_progress} == {
        (b, c, s, cp) for b, c, s in fit_keys for cp in protocol["checkpoints"]}
    assert len(snapshot_progress) == snapshots and len(run["progress"]) == len(fit_keys) + snapshots
    cells = result["provenance"]["expected_cells"]
    assert {r["bank_id"] for r in cells} == set(banks) and len(cells) == len(banks)
    for cell in cells:
        assert cell["complete"]
        assert cell["expected_learner_cells"] == cell["complete_learner_cells"] == cell["unique_complete_learner_cells"] == len(rows) // len(banks)
        assert cell["expected_reference_cells"] == cell["complete_reference_cells"] == cell["unique_complete_reference_cells"] == len(refs)
    report.update(training_states=len(data["observations"]), selected_layouts=len(task_hashes),
        rejected_candidates=len(selection["collision_skips"]), transition_validation=transition_report,
        collected_trace_reconstruction=collection_reports, collection_audit_is_reconstruction_not_new_collection=True,
        support_intersections_checked=intersection_count, sampling_streams_checked=sampling_streams,
        model_snapshots_checked=snapshots, final_models_checked=len(states), initial_models_match_archive=True,
        frozen_evaluation_identity_unchanged=True, loss_rows_checked=loss_rows, learner_episode_rows=len(rows),
        reference_episode_rows=len(refs), paired_layout_rows=len(paired["per_layout"]), paired_seed_rows=len(paired["per_seed"]),
        paired_aggregate_rows=len(paired["aggregate"]), recorded_replays_checked=replay_count,
        recorded_steps_checked=replay_steps, replay_forward_inference_checked=not args.skip_forward_inference,
        forward_inference_checked=not args.skip_forward_inference, full_policy_evaluation_repeated=False,
        new_training_updates=0, new_collection_steps=0, bank_effects=result["robustness"]["bank_effects"],
        audit_seconds=time.monotonic() - started)
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
