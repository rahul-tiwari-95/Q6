"""Independent audit of frozen policies on original familiar starts.

Verifies graph reachability, saved prior predictions, step-level evidence and
paired outcomes without fitting or repeating the full learned-policy study.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
import audit_fixed_targets_study as common
import audit_bank_replication as bank_audit
import audit_panel_evaluation as replay_audit
import audit_logged_graph as graph_audit
import audit_map_replay as shared
from audit_fixed_targets_study import close, read_json, csv_rows

CONDITIONS = ("constrained_bootstrap", "logged_graph")
ACTION_SETS = ("unrestricted", "logged")
FLOAT32_MEAN_ULPS = 2


def assert_float32_mean(actual, values, label=""):
    """Check the archived prediction-gap mean against a float64 reference.

    The archived runner reduced nonnegative float32 per-state gaps in float32.
    SIMD reduction order differs across platforms. Use fsum for the reference
    and allow two float32 rounding units at its magnitude, only for this mean.
    This archival tolerance covers all 180 saved slices (maximum 1.518 ULP);
    it is not a universal error bound for arbitrary float32 summations.
    Exact zero stays exact, and other audit metrics retain their own checks.
    """
    values = np.asarray(values)
    assert values.dtype == np.float32 and values.ndim == 1 and values.size > 0, (label, "nonempty float32 gap vector required")
    assert np.isfinite(values).all() and (values >= 0).all(), (label, "finite nonnegative gaps required")
    actual = float(actual)
    assert math.isfinite(actual) and actual >= 0, (label, "invalid saved mean", actual)
    canonical = math.fsum(map(float, values)) / values.size
    if canonical == 0:
        ulp = 0.0
    else:
        # The smallest float32 spacing is 2**-149. frexp avoids overflow in
        # np.spacing at the maximum finite float32 value.
        rounded = float(np.float32(canonical))
        exponent = math.frexp(rounded)[1]
        ulp = math.ldexp(1.0, -149 if rounded == 0 else max(-149, exponent - 24))
    tolerance = FLOAT32_MEAN_ULPS * ulp
    error = abs(actual - canonical)
    assert error <= tolerance, (label, actual, canonical, "float32 ULP tolerance", tolerance)
    return {"canonical_mean": canonical, "absolute_tolerance": tolerance,
            "absolute_error": error, "error_float32_ulps": error / ulp if ulp else 0.0}


def independent_reachability(table, remaining, support):
    """Min-plus dynamic program over recorded edges, independent of value DP."""
    observed, ended, successor = (table[k] for k in ("observed", "ends", "successor_indices"))
    terminated, truncated = table["terminated"], table["truncated"]
    assert np.array_equal(ended[observed], (terminated | truncated)[observed])
    assert not (terminated & truncated).any()
    assert np.array_equal(np.flatnonzero(observed.any(1)), support)
    n = len(remaining)
    action_steps = np.full((n, 4), -1, np.int16)
    state_steps = np.full(n, -1, np.int16)
    for clock in range(1, 33):
        states = support[remaining[support] == clock]
        for action in range(4):
            current = states[observed[states, action]]
            success = current[terminated[current, action]]
            action_steps[success, action] = 1
            live = current[~ended[current, action]]
            following = successor[live, action]
            assert ((following >= 0) & (following < n)).all()
            assert observed[following].any(1).all()
            assert (remaining[following] == clock - 1).all()
            reachable = state_steps[following] >= 1
            action_steps[live[reachable], action] = state_steps[following[reachable]] + 1
        best = np.where(action_steps[states] >= 1, action_steps[states], 32767).min(1)
        state_steps[states] = np.where(best == 32767, -1, best)
    assert not np.any(action_steps[~observed] != -1)
    assert np.all(state_steps[state_steps >= 1] <= remaining[state_steps >= 1])
    return state_steps, action_steps


def prediction_arrays(predictions, target, observed):
    """Per-state statistics of the existing predictions; no neural inference."""
    counts = observed.sum(1)
    assert predictions.shape == target.shape == observed.shape and observed.any(1).all()
    assert np.isfinite(predictions).all() and np.isfinite(target[observed]).all()
    error = np.where(observed, predictions.astype(np.float64) - target, 0.)
    bias = error.sum(1) / counts
    centered = np.where(observed, error - bias[:, None], 0.)
    graph = np.where(observed, target, -np.inf)
    selected = np.where(observed, predictions, -np.inf).argmax(1)
    optimum = graph.max(1)
    regret = optimum - target[np.arange(len(target)), selected]
    unrestricted = predictions.argmax(1)
    outside = ~observed[np.arange(len(target)), unrestricted]
    graph_sorted = np.sort(graph, axis=1)
    margin = np.full(len(target), np.nan)
    multiple = counts > 1
    margin[multiple] = graph_sorted[multiple, -1] - graph_sorted[multiple, -2]
    prediction_gap = predictions.max(1).astype(np.float64) - np.where(observed, predictions, -np.inf).max(1).astype(np.float64)
    return {"counts": counts, "state_signed_bias": bias, "state_abs_error": np.abs(error).sum(1) / counts,
            "state_squared_error": (error ** 2).sum(1) / counts,
            "state_centered_abs_error": np.abs(centered).sum(1) / counts,
            "state_centered_squared_error": (centered ** 2).sum(1) / counts,
            "restricted_agreement": np.abs(regret) <= 1e-6, "graph_regret": regret,
            "outside_argmax": outside, "target_action_gap": margin, "prediction_outside_gap": prediction_gap}


def reduce_prediction_arrays(arrays, include):
    n = int(include.sum())
    multiple = include & (arrays["counts"] > 1)
    m = int(multiple.sum())
    result = {"states": n, "observed_edges": int(arrays["counts"][include].sum()), "multiple_action_states": m}
    for key in ("state_signed_bias", "state_abs_error", "state_squared_error", "restricted_agreement", "graph_regret", "outside_argmax", "prediction_outside_gap"):
        result[key] = float(arrays[key][include].mean()) if n else None
    for key in ("state_centered_abs_error", "state_centered_squared_error", "target_action_gap"):
        result[key] = float(arrays[key][multiple].mean()) if m else None
    result["target_action_gap_zero_fraction"] = float((arrays["target_action_gap"][multiple] <= 1e-6).mean()) if m else None
    return result


def compare(actual, expected, label="value"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (label, set(actual) ^ set(expected))
        for key in expected:
            compare(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (label, len(actual), len(expected))
        for i, value in enumerate(expected):
            compare(actual[i], value, f"{label}[{i}]")
    elif isinstance(expected, float):
        close(actual, expected, label)
    else:
        assert actual == expected, (label, actual, expected)


def geometry(data, metadata, config, World):
    """Reconstruct full row identities and one-step outcomes, not policy runs."""
    n = len(data["observations"])
    arrays = {"successor_indices": np.full((n, 4), -1, np.int32),
              "rewards": np.empty((n, 4), np.float64), "ends": np.empty((n, 4), bool),
              "terminated": np.empty((n, 4), bool), "truncated": np.empty((n, 4), bool)}
    worlds, lookup = {}, {}
    offset, clone_checks = 0, 0
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for entry in metadata["train"]:
        env = World(config)
        env.reset(seed=entry["map_seed"])
        original = env.position
        goal = tuple(int(x) for x in np.argwhere(env.pellets)[0])
        distance, queue = {goal: 0}, deque([goal])
        while queue:
            p = queue.popleft()
            for dr, dc in directions:
                q = p[0] + dr, p[1] + dc
                if 0 <= q[0] < config.size and 0 <= q[1] < config.size and not env.walls[q] and q not in distance:
                    distance[q] = distance[p] + 1
                    queue.append(q)
        positions = [tuple(int(x) for x in p) for p in np.argwhere(~(env.walls | env.pellets))]
        section = slice(offset, offset + len(positions) * config.horizon)
        assert np.array_equal(data["positions"][section], np.repeat(positions, config.horizon, axis=0))
        assert np.array_equal(data["remaining"][section], np.tile(np.arange(1, config.horizon + 1), len(positions)))
        assert (data["map_seeds"][section] == entry["map_seed"]).all()
        assert entry["original_start"] == list(original)
        layout = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(), "config": asdict(config)}
        assert hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest() == entry["layout_hash"]
        task = {**layout, "position": list(original)}
        following = {}
        for p in positions:
            for a, physical in enumerate(config.action_mapping):
                dr, dc = directions[physical]
                q = p[0] + dr, p[1] + dc
                following[p, a] = q if q in distance else p
        worlds[entry["map_seed"]] = {"walls": np.argwhere(env.walls).tolist(), "pellets": np.argwhere(env.pellets).tolist(),
            "start": list(original), "goal": goal, "distance": distance, "following": following,
            "task_hash": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(), "layout_hash": entry["layout_hash"]}
        position_index = {p: i for i, p in enumerate(positions)}
        for pi, p in enumerate(positions):
            rows = offset + pi * config.horizon + np.arange(config.horizon)
            remaining = np.arange(1, config.horizon + 1)
            env.position, env.elapsed = p, 0
            expected = np.repeat(env.observe()[None, :], config.horizon, axis=0)
            expected[:, 3 * config.size**2] = remaining / config.horizon
            assert np.array_equal(data["observations"][rows], expected)
            for index, t in zip(rows, remaining):
                lookup[entry["map_seed"], p[0], p[1], int(t)] = int(index)
            for action in range(4):
                q = following[p, action]
                success = q == goal
                ends = success | (remaining == 1)
                next_rows = offset + position_index.get(q, 0) * config.horizon + remaining - 2
                rewards = config.step_cost + config.pellet_reward * success + config.shaping_weight * (
                    config.gamma * np.where(ends, 0., -distance[q] / config.size) + distance[p] / config.size)
                arrays["successor_indices"][rows, action] = np.where(ends, -1, next_rows)
                arrays["rewards"][rows, action] = rewards
                arrays["ends"][rows, action] = ends
                arrays["terminated"][rows, action] = success
                arrays["truncated"][rows, action] = ends & ~np.asarray(success)
                if len(worlds) <= 2:
                    for remaining_value in (1, 2, 32):
                        env.elapsed = config.horizon - remaining_value
                        clone = env.clone()
                        _, reward, terminated, truncated, _ = clone.step(action)
                        close(reward, rewards[remaining_value - 1], "independent reward/clone")
                        assert clone.position == q and terminated == success and truncated == bool(remaining_value == 1 and not success)
                        clone_checks += 1
        offset += len(positions) * config.horizon
    assert offset == n and len(lookup) == n
    return arrays, worlds, lookup, clone_checks


def audit_inputs(study, result, root):
    provenance = result["provenance"]
    prior = root / provenance["source_archive"]
    assert prior.resolve() == (root / "experiments/logged_graph/pilot_v1").resolve()
    manifest = read_json(study / "prior_manifest.json")
    assert manifest == read_json(prior / "manifest.json") and manifest["status"] == "complete"
    required = {"manifest.json", "dataset.npz", "dataset_metadata.json", "supports.npz", "recorded_transitions.npz", "logged_graph_targets.npz",
        "fit_predictions.npz", "fit_diagnostics.json", "protocol.json", "protocol.md"}
    required.update(f"source/q6/{m}.py" for m in ("world", "learning", "diagnostics", "logged_graph", "recorded_actions", "panel_evaluation"))
    required.update(f"banks/bank{b}/collection_steps.csv" for b in result["protocol"]["bank_ids"])
    required.update(f"models/bank{b}_{c}_seed{s}_update30000.pt" for b in result["protocol"]["bank_ids"] for c in CONDITIONS for s in result["protocol"]["seeds"])
    assert {r["file"] for r in provenance["inputs"]} == required and len(provenance["inputs"]) == len(required)
    for row in provenance["inputs"]:
        name = row["file"]
        digest = common.sha(prior / name)
        assert row["before"] == row["after"] == digest and row["unchanged"]
        if name != "manifest.json":
            assert manifest["files"][name] == digest
        copied = "prior_" + name if name in ("manifest.json", "protocol.json", "protocol.md") else name
        assert common.sha(study / copied) == digest, ("copied input", copied)
    for module in ("world", "learning", "diagnostics", "logged_graph", "recorded_actions", "panel_evaluation"):
        name = f"source/q6/{module}.py"
        assert common.sha(study / name) == manifest["files"][name]
    records = read_json(study / "models.json")
    assert records == provenance["models"]
    expected = {(b, c, s) for b in result["protocol"]["bank_ids"] for c in CONDITIONS for s in result["protocol"]["seeds"]}
    assert {(r["bank_id"], r["condition"], r["seed"]) for r in records} == expected and len(records) == len(expected)
    networks = {}
    for record in records:
        key = record["bank_id"], record["condition"], record["seed"]
        name = f"models/bank{key[0]}_{key[1]}_seed{key[2]}_update30000.pt"
        assert record["saved"] == name and (root / record["source"]).resolve() == (prior / name).resolve()
        snapshot = torch.load(study / name, map_location="cpu", weights_only=True)
        assert record["checkpoint"] == snapshot["optimizer_updates"] == 30000 and snapshot["observation_size"] == 92
        expected_boundary = {"online": replay_audit.tensor_hash(snapshot["online"]), "target": replay_audit.tensor_hash(snapshot["target"]),
            "source": common.sha(prior / name), "copy": common.sha(study / name)}
        assert expected_boundary["source"] == expected_boundary["copy"] == manifest["files"][name]
        assert expected_boundary["online"] == snapshot["parameter_hash"] and expected_boundary["target"] == snapshot["target_parameter_hash"]
        assert record["before"] == record["after"] == expected_boundary and record["unchanged"]
        assert record["initial_source_copy_identical"]
        assert {r["action_set"] for r in record["action_set_checks"]} == set(ACTION_SETS) and len(record["action_set_checks"]) == 2
        for check in record["action_set_checks"]:
            assert check["before"] == check["after"] == expected_boundary and check["unchanged"]
        networks[key] = bank_audit.bare_network(snapshot["online"])
    assert provenance["torch_rng_unchanged"]
    return networks, len(provenance["inputs"])


def audit_slices(study, result, data, supports, tables, targets, shortest):
    saved = read_json(study / "prediction_slices.json")
    assert saved == result["prediction_slices"] and saved["new_inference_rows"] == 0
    buckets = {"1_8": (1, 8), "9_16": (9, 16), "17_24": (17, 24), "25_32": (25, 32)}
    assert saved["clock_groups"] == {k: list(v) for k, v in buckets.items()}
    expected_keys = set()
    rows = {(r["bank_id"], r["condition"], r["seed"], r["axis"], r["group"]): r for r in saved["per_slice"]}
    assert len(rows) == len(saved["per_slice"])
    with np.load(study / "fit_predictions.npz") as predictions:
        for bank, support in supports.items():
            assert np.array_equal(predictions[f"bank{bank}_state_rows"], support)
            mask = tables[bank]["observed"][support]
            count = mask.sum(1)
            clock = data["remaining"][support]
            selectors = {("all", "all"): np.ones(len(support), bool), ("action_count", "one"): count == 1,
                ("action_count", "multiple"): count > 1, ("reachability", "reachable"): shortest[bank][support] > 0,
                ("reachability", "unreachable"): shortest[bank][support] < 0,
                **{("remaining", k): (clock >= low) & (clock <= high) for k, (low, high) in buckets.items()}}
            selectors["original_start", "clock32"] = clock == 32
            for condition in CONDITIONS:
                for seed in result["protocol"]["seeds"]:
                    pred = predictions[f"bank{bank}_{condition}_seed{seed}"]
                    assert pred.dtype == np.float32 and pred.shape == (len(support), 4)
                    arrays = prediction_arrays(pred, targets[bank][support], mask)
                    for (axis, group), selected in selectors.items():
                        if not selected.any():
                            continue
                        key = bank, condition, seed, axis, group
                        expected_keys.add(key)
                        item = rows[key]
                        independent = graph_audit.independent_fit(pred[selected], targets[bank][support][selected], mask[selected])
                        independent.pop("optimal_actions")
                        independent.pop("unrestricted_argmax_outside_logged_states")
                        assert item["restricted_ranking_atol"] == 1e-6 and item["restricted_ranking_rtol"] == 0
                        for name, value in independent.items():
                            close(item[name], value, f"prior slice {key} {name}")
                        multiple = selected & (count > 1)
                        expected_metrics = {
                            "state_mean_signed_error": float(arrays["state_signed_bias"][selected].mean()),
                            "state_mean_squared_offset": float(np.square(arrays["state_signed_bias"][selected]).mean()),
                            "multiple_action_states": int(multiple.sum()),
                            "centered_state_mean_abs_error": float(arrays["state_centered_abs_error"][multiple].mean()) if multiple.any() else None,
                            "centered_state_mean_squared_error": float(arrays["state_centered_squared_error"][multiple].mean()) if multiple.any() else None,
                            "mean_target_top_two_gap": float(arrays["target_action_gap"][multiple].mean()) if multiple.any() else None}
                        for name, value in expected_metrics.items():
                            compare(item[name], value, f"prior slice {key} {name}")
                        gaps = pred[selected].max(1) - np.where(mask[selected], pred[selected], -np.inf).max(1)
                        assert_float32_mean(item["mean_unrestricted_prediction_gap"], gaps, f"prior slice {key} mean_unrestricted_prediction_gap")
                        if axis == "original_start":
                            assert item["classification"].startswith("exploratory")
    assert set(rows) == expected_keys
    return len(rows)


def audit_steps(study, result, data, worlds, lookup, transitions, tables, graph, shortest, networks, config, skip_forward):
    episodes = read_json(study / "episodes.json")
    references = read_json(study / "references.json")
    all_rows = episodes + references
    by_id = {r["episode_id"]: r for r in all_rows}
    assert len(by_id) == len(all_rows)
    protocol = result["protocol"]
    banks, seeds, maps = protocol["bank_ids"], protocol["seeds"], protocol["map_seeds"]
    expected = {(b, c, s, a, m) for b in banks for c in CONDITIONS for s in seeds for a in ACTION_SETS for m in maps}
    assert {(r["bank_id"], r["condition"], r["seed"], r["action_set"], r["map_seed"]) for r in episodes} == expected and len(episodes) == len(expected)
    ref_expected = {(str(b), "logged_q", m) for b in banks for m in maps} | {("shared", "shortest_path", m) for m in maps}
    assert {(str(r["bank_id"]), r["policy"], r["map_seed"]) for r in references} == ref_expected and len(references) == len(ref_expected)
    compare(result["provenance"]["expected_cells"], {"learner_expected": len(expected), "learner_actual": len(episodes), "reference_expected": len(ref_expected),
        "reference_actual": len(references), "learner_unique": len(expected), "reference_unique": len(ref_expected), "complete": True})
    traces = {t["episode_id"]: t for t in read_json(study / "trajectories.json")}
    assert list(traces.values()) == result["trajectories"]
    replay_ids = {r["episode_id"] for r in all_rows if r["map_seed"] in protocol["replay_map_seeds"]}
    assert set(traces) == replay_ids and len(traces) == result["provenance"]["expected_recordings"] == result["provenance"]["actual_recordings"]
    totals, next_states, occupancy = {}, {}, defaultdict(lambda: defaultdict(int))
    base_keys = ("episode_id", "bank_id", "condition", "seed", "checkpoint", "policy", "action_set", "mode", "panel", "block", "map_seed", "repetition")
    count_keys = ("noop_steps", "supported_steps", "unsupported_steps", "off_mask_actions", "unrestricted_argmax_outside_steps", "support_exits", "support_reentries",
        "logged_regret_steps", "restricted_agreement_steps", "restricted_agreement_count", "logged_reachable_steps", "logged_reachability_losses")
    trace_steps, forward_rows = 0, 0
    with gzip.open(study / "steps.jsonl.gz", "rt") as stream:
        for line in stream:
            frame = json.loads(line)
            episode_id = frame["episode_id"]
            assert episode_id in by_id
            episode = by_id[episode_id]
            for key in base_keys:
                assert frame[key] == episode[key]
            bank, condition, seed, action_set, map_seed, policy = (episode[k] for k in ("bank_id", "condition", "seed", "action_set", "map_seed", "policy"))
            world = worlds[map_seed]
            if episode_id not in totals:
                assert episode["task_hash"] == world["task_hash"] and episode["block"] == episode["panel"] == f"block_{(map_seed - 300000) // 32}"
                assert episode["mode"] == "greedy" and episode["repetition"] == 0 and episode["checkpoint_complete"] == 1
                assert episode["checkpoint"] == (30000 if policy == "learner" else 0)
                assert episode_id == f"{bank}:{condition}:{seed}:{action_set}:{map_seed}:{policy}"
                totals[episode_id] = {"steps": 0, "base_return": 0., "shaped_return": 0., "discounted_return": 0., "logged_regret_sum": 0.,
                    "first_off_mask_step": None, "first_support_exit_step": None, **{k: 0 for k in count_keys}}
                next_states[episode_id] = lookup[(map_seed, *world["start"], 32)]
            total = totals[episode_id]
            step, current = total["steps"] + 1, next_states[episode_id]
            assert current >= 0 and frame["step"] == step and frame["current_row"] == current
            position = tuple(int(x) for x in data["positions"][current])
            remaining = int(data["remaining"][current])
            assert remaining == 33 - step and frame["remaining_before"] == remaining and frame["position_before"] == list(position)
            table = None if policy == "shortest_path" else tables[bank]
            mask = table["observed"][current] if table is not None else np.zeros(4, bool)
            supported = bool(mask.any()) if table is not None else None
            assert frame["current_supported"] == supported and frame["recorded_mask"] == (mask.tolist() if table is not None else None)
            logged_values = [float(graph[bank][current, a]) if supported and mask[a] else None for a in range(4)]
            compare(frame["logged_q_values"], logged_values)
            action = frame["action"]
            assert isinstance(action, int) and not isinstance(action, bool) and 0 <= action < 4
            q = None
            if policy == "learner":
                q = np.asarray(frame["q_values"], np.float32)
                assert q.shape == (4,) and np.isfinite(q).all()
                assert action == int(q.argmax() if action_set == "unrestricted" else np.where(mask, q, -np.inf).argmax())
                if episode_id in replay_ids and not skip_forward:
                    with torch.no_grad():
                        regenerated = networks[bank, condition, seed].online(torch.from_numpy(data["observations"][current]).unsqueeze(0))[0].numpy()
                    assert np.array_equal(q, regenerated), ("saved replay inference", episode_id, step)
                    forward_rows += 1
            elif policy == "logged_q":
                assert frame["q_values"] is None and supported and action_set == "logged"
                assert action == int(np.where(mask, graph[bank][current], -np.inf).argmax())
            else:
                assert policy == "shortest_path" and bank == "shared" and action_set == "unrestricted" and frame["q_values"] is None
                candidates = [a for a in range(4) if world["distance"][world["following"][position, a]] == world["distance"][position] - 1]
                # World A action order equals the reference's physical BFS order.
                assert action == min(candidates)
            selected = bool(mask[action]) if supported is not None else None
            off_mask = bool(supported and not selected)
            outside = bool(supported and q is not None and not mask[q.argmax()])
            following = int(transitions["successor_indices"][current, action])
            ended = bool(transitions["ends"][current, action])
            terminated = bool(transitions["terminated"][current, action])
            truncated = bool(transitions["truncated"][current, action])
            successor_supported = bool(table["observed"][following].any()) if table is not None and not ended else None
            exit_support = bool(supported and not ended and successor_supported is False)
            reentry = bool(supported is False and not ended and successor_supported)
            was_reachable = bool(shortest[bank][current] > 0) if table is not None and supported else None
            lost = bool(was_reachable and ((ended and not terminated) or (not ended and successor_supported and shortest[bank][following] < 0)))
            regret, agreement = None, None
            if supported:
                optimum = np.max(graph[bank][current, mask])
                if selected:
                    regret = float(max(0., optimum - graph[bank][current, action]))
                if q is not None:
                    restricted = np.where(mask, q, -np.inf).argmax()
                    agreement = bool(abs(optimum - graph[bank][current, restricted]) <= 1e-6)
            expected_fields = {"selected_action_recorded": selected, "off_mask_action": off_mask, "unrestricted_argmax_outside": outside,
                "logged_value_regret": regret, "restricted_action_agreement": agreement, "logged_success_reachable": was_reachable, "logged_reachability_lost": lost,
                "successor_row": following, "successor_supported": successor_supported, "support_exit": exit_support, "support_reentry": reentry,
                "position": list(world["following"][position, action]), "reward": float(transitions["rewards"][current, action]),
                "base_reward": config.step_cost + config.pellet_reward * terminated,
                "pellets": [] if terminated else world["pellets"], "terminated": terminated, "truncated": truncated}
            for key, value in expected_fields.items():
                compare(frame[key], value, f"step {episode_id} {step} {key}")
            if action_set == "logged":
                assert supported and selected and (ended or successor_supported)
            if selected:
                assert table["successor_indices"][current, action] == following and table["terminated"][current, action] == terminated and table["truncated"][current, action] == truncated
            total["steps"] = step
            total["base_return"] += expected_fields["base_reward"]
            total["shaped_return"] += expected_fields["reward"]
            total["discounted_return"] += config.gamma ** (step - 1) * expected_fields["reward"]
            increments = {"noop_steps": position == world["following"][position, action], "supported_steps": supported is True, "unsupported_steps": supported is False,
                "off_mask_actions": off_mask, "unrestricted_argmax_outside_steps": outside, "support_exits": exit_support, "support_reentries": reentry,
                "logged_regret_steps": regret is not None, "restricted_agreement_steps": agreement is not None, "restricted_agreement_count": agreement is True,
                "logged_reachable_steps": was_reachable is True, "logged_reachability_losses": lost}
            for key, value in increments.items():
                total[key] += int(value)
            total["logged_regret_sum"] += regret or 0.
            for key, event in (("first_off_mask_step", off_mask), ("first_support_exit_step", exit_support)):
                if event and total[key] is None:
                    total[key] = step
            if policy == "learner":
                occupancy_key = bank, condition, seed, action_set, remaining
                for name, value in (("decisions", 1), ("supported", supported), ("unsupported", not supported), ("off_mask", off_mask), ("exits", exit_support), ("reentries", reentry)):
                    occupancy[occupancy_key][name] += int(value)
            if episode_id in traces:
                trace = traces[episode_id]
                assert trace["steps"][step - 1] == {k: v for k, v in frame.items() if k not in base_keys}
                for key in base_keys:
                    assert trace[key] == episode[key]
                assert trace["walls"] == world["walls"] and trace["pellets_initial"] == world["pellets"] and trace["start"] == world["start"]
                assert trace["grid_size"] == config.size and trace["action_mapping"] == list(config.action_mapping) and trace["world"] == "A"
                trace_steps += 1
            if ended:
                total["success"] = int(terminated)
                total["efficient_success"] = int(terminated and step <= 2 * world["distance"][tuple(world["start"])])
                for key, value in total.items():
                    compare(episode[key], value, f"episode {episode_id} {key}")
                if episode_id in traces:
                    assert len(traces[episode_id]["steps"]) == step and traces[episode_id]["success"] == terminated
                if policy == "logged_q":
                    start = lookup[(map_seed, *world["start"], 32)]
                    close(total["discounted_return"], float(np.nanmax(graph[bank][start])), "reference return equals graph value")
                if action_set == "logged" and terminated:
                    start = lookup[(map_seed, *world["start"], 32)]
                    assert 0 < shortest[bank][start] <= step
                if policy == "shortest_path":
                    assert terminated and step == world["distance"][tuple(world["start"])]
            next_states[episode_id] = following
    assert set(totals) == set(by_id) and all(value == -1 for value in next_states.values())
    steps = sum(r["steps"] for r in all_rows)
    assert steps == result["run"]["step_records"]
    assert len(episodes) == result["run"]["learner_episodes"] and len(references) == result["run"]["reference_episodes"]
    occupancy_rows = [{"bank_id": b, "condition": c, "seed": s, "action_set": a, "remaining": t, **dict(v)} for (b, c, s, a, t), v in sorted(occupancy.items())]
    compare(result["occupancy"], occupancy_rows)
    return episodes, references, {"all_episode_steps_checked": steps, "saved_recordings": len(traces), "recording_steps_checked": trace_steps, "recording_forward_rows_checked": forward_rows}


def reduced(rows):
    n = len(rows)
    sums = {k: sum(r[k] for r in rows) for k in ("steps", "success", "efficient_success", "base_return", "shaped_return", "noop_steps", "supported_steps", "unsupported_steps",
        "off_mask_actions", "unrestricted_argmax_outside_steps", "support_exits", "support_reentries", "logged_regret_steps", "logged_regret_sum",
        "restricted_agreement_steps", "restricted_agreement_count", "logged_reachable_steps", "logged_reachability_losses")}
    value = {"episodes": n, "success_rate": sums.pop("success") / n, "efficient_success_rate": sums.pop("efficient_success") / n,
        "mean_steps": sums["steps"] / n, "evaluation_steps": sums.pop("steps"),
        "base_return": sums.pop("base_return") / n, "shaped_return": sums.pop("shaped_return") / n, **sums}
    successful = [r for r in rows if r["success"]]
    value.update(successful_episodes=len(successful), successful_mean_steps=sum(r["steps"] for r in successful) / len(successful) if successful else None,
        off_mask_episodes=sum(r["first_off_mask_step"] is not None for r in rows), support_exit_episodes=sum(r["first_support_exit_step"] is not None for r in rows))
    for key, numerator, denominator in (("noop_rate", "noop_steps", "evaluation_steps"), ("off_mask_rate", "off_mask_actions", "supported_steps"),
        ("unrestricted_argmax_outside_rate", "unrestricted_argmax_outside_steps", "supported_steps"), ("unsupported_step_fraction", "unsupported_steps", "evaluation_steps"),
        ("off_mask_episode_rate", "off_mask_episodes", "episodes"), ("support_exit_episode_rate", "support_exit_episodes", "episodes"),
        ("mean_logged_regret", "logged_regret_sum", "logged_regret_steps"), ("restricted_action_agreement", "restricted_agreement_count", "restricted_agreement_steps")):
        value[key] = value[numerator] / value[denominator] if value[denominator] else None
    return value


def audit_summaries(study, result, rows, references):
    protocol = result["protocol"]
    blocks = ["all"] + [b["id"] for b in protocol["blocks"]]
    bcs = [(b, c, a) for b in protocol["bank_ids"] for c in CONDITIONS for a in ACTION_SETS]
    targets = ((result["aggregate"], {(b, c, a, block) for b, c, a in bcs for block in blocks}, ("bank_id", "condition", "action_set", "block")),
        (result["seed_results"], {(b, c, a, block, s) for b, c, a in bcs for block in blocks for s in protocol["seeds"]}, ("bank_id", "condition", "action_set", "block", "seed")),
        (result["pooled"]["aggregate"], {(c, a, block) for c in CONDITIONS for a in ACTION_SETS for block in blocks}, ("condition", "action_set", "block")))
    for saved, expected, fields in targets:
        assert {tuple(r[k] for k in fields) for r in saved} == expected and len(saved) == len(expected)
        for summary in saved:
            group = [r for r in rows if all((summary[k] == "all" or r[k] == summary[k]) if k == "block" else r[k] == summary[k] for k in fields)]
            compare({k: v for k, v in summary.items() if k not in fields}, reduced(group), "episode aggregate")
    ref_expected = {(str(b), p, block) for b, p in {(r["bank_id"], r["policy"]) for r in references} for block in blocks}
    assert {(str(r["bank_id"]), r["policy"], r["block"]) for r in result["references"]} == ref_expected and len(result["references"]) == len(ref_expected)
    for summary in result["references"]:
        group = [r for r in references if r["bank_id"] == summary["bank_id"] and r["policy"] == summary["policy"] and (summary["block"] == "all" or r["block"] == summary["block"])]
        compare({k: v for k, v in summary.items() if k not in ("bank_id", "policy", "block")}, reduced(group))
    # Cell order: control unrestricted/logged, exact unrestricted/logged.
    weights = {"mask_effect_constrained": (-1, 1, 0, 0), "mask_effect_graph": (0, 0, -1, 1),
        "graph_minus_constrained_unrestricted": (-1, 0, 1, 0), "graph_minus_constrained_logged": (0, -1, 0, 1), "interaction": (1, -1, -1, 1)}
    metrics = ("success", "efficient_success", "steps", "base_return", "shaped_return", "noop_steps")
    cells = {(r["bank_id"], r["seed"], r["map_seed"], r["condition"], r["action_set"]): r for r in rows}
    calculated = {}
    for bank in protocol["bank_ids"]:
        for seed in protocol["seeds"]:
            for map_seed in protocol["map_seeds"]:
                four = [cells[bank, seed, map_seed, c, a] for c in CONDITIONS for a in ACTION_SETS]
                for name, vector in weights.items():
                    calculated[bank, seed, map_seed, name] = {m + "_delta": sum(weight * r[m] for weight, r in zip(vector, four)) for m in metrics}
    paired = result["paired_differences"]
    assert paired == read_json(study / "paired_differences.json")
    assert len(paired["per_start"]) == len(calculated)
    assert {(r["bank_id"], r["seed"], r["map_seed"], r["comparison"]) for r in paired["per_start"]} == set(calculated)
    for item in paired["per_start"]:
        compare({k: item[k] for k in next(iter(calculated.values()))}, calculated[item["bank_id"], item["seed"], item["map_seed"], item["comparison"]])
        assert item["block"] == f"block_{(item['map_seed'] - 300000) // 32}"
    for kind in ("per_seed", "aggregate"):
        expected = {(b, block, c, s) for b in protocol["bank_ids"] for block in blocks for c in weights for s in (protocol["seeds"] if kind == "per_seed" else [None])}
        assert {(r["bank_id"], r["block"], r["comparison"], r.get("seed")) for r in paired[kind]} == expected and len(paired[kind]) == len(expected)
        for item in paired[kind]:
            group = [v for (b, s, m, c), v in calculated.items() if b == item["bank_id"] and c == item["comparison"] and (kind != "per_seed" or s == item["seed"])
                and (item["block"] == "all" or item["block"] == f"block_{(m - 300000) // 32}")]
            assert item["starts"] == len(group)
            for m in metrics:
                close(item[m + "_delta"], sum(r[m + "_delta"] for r in group) / len(group), "paired reduction")
    assert {(r["block"], r["comparison"]) for r in result["pooled"]["paired"]} == {(b, c) for b in blocks for c in weights}
    for item in result["pooled"]["paired"]:
        group = [r for r in paired["aggregate"] if r["block"] == item["block"] and r["comparison"] == item["comparison"]]
        assert item["banks"] == len(group) == len(protocol["bank_ids"])
        for m in metrics:
            close(item[m + "_delta"], sum(r[m + "_delta"] for r in group) / len(group), "equal bank effect")
    robustness = result["robustness"]
    effects = [r for r in paired["aggregate"] if r["comparison"] == "interaction" and r["block"] == "all"]
    compare(robustness["bank_effects"], effects)
    values = [r["efficient_success_delta"] for r in effects]
    for key, value in {"mean": sum(values) / len(values), "minimum": min(values), "maximum": max(values),
        "positive_banks": sum(x > 1e-12 for x in values), "negative_banks": sum(x < -1e-12 for x in values), "zero_banks": sum(abs(x) <= 1e-12 for x in values)}.items():
        compare(robustness[key], value)
    assert robustness["primary_comparison"] == "interaction" and robustness["primary_metric"] == "efficient_success_delta" and robustness["sign_tolerance"] == 1e-12
    return {"paired_start_rows": len(calculated), "paired_seed_rows": len(paired["per_seed"]), "paired_bank_rows": len(paired["aggregate"]), "interaction_by_bank": effects}


def audit(study, allow_smoke=False, skip_forward=False):
    started = time.monotonic()
    torch.set_num_threads(1)
    study = Path(study).resolve()
    root = Path(__file__).resolve().parents[1]
    result, protocol = read_json(study / "results.json"), read_json(study / "protocol.json")
    assert result["run"]["status"] == "complete"
    manifest_files = common.audit_manifest(study, protocol, result)
    assert read_json(study / "manifest.json")["status"] == "complete"
    assert protocol["id"] == "familiar-starts-v1" and protocol["conditions"] == list(CONDITIONS) and protocol["action_sets"] == list(ACTION_SETS)
    assert protocol["evaluation_only"] and protocol["mode"] == "greedy" and protocol["repetitions"] == 1 and not protocol["new_competence_gates"]
    assert protocol["primary_comparison"] == "interaction" and protocol["primary_metric"] == "efficient_success_delta"
    eligible = not protocol["smoke"] and not protocol["deviations"]
    assert result["robustness"]["eligible"] == eligible
    if not allow_smoke:
        assert eligible and protocol["git"]["dirty"] is False and protocol["git"]["revision"]
        assert protocol["seeds"] == [0, 1, 2] and protocol["bank_ids"] == [1, 2, 3] and protocol["map_seeds"] == list(range(300000, 300256))
        assert protocol["runtime"]["python"].startswith("3.12.") and protocol["runtime"]["numpy"] == "2.0.2" and protocol["runtime"]["torch"].split("+")[0] == "2.8.0"
    assert protocol["replay_map_seeds"] == [m for m in protocol["map_seeds"] if (m - 300000) % 32 == 0]
    block_ids = sorted({(m - 300000) // 32 for m in protocol["map_seeds"]})
    assert protocol["blocks"] == [{"id": f"block_{b}", "map_seeds": [m for m in protocol["map_seeds"] if (m - 300000) // 32 == b]} for b in block_ids]
    assert protocol["runtime"]["torch_threads"] == 1 and protocol["runtime"]["device"] == "cpu"
    for key in ("train_updates", "collection_steps", "collection_episodes", "new_support_draws", "new_labels_fitted"):
        assert result["run"][key] == 0
    assert 0 < result["run"]["wall_seconds"] <= protocol["budget"]["admission_seconds"]
    assert 0 < result["run"]["peak_rss_bytes"] <= protocol["budget"]["peak_process_rss_bytes"]
    assert protocol["budget"]["all_phases_included"]
    for name, key in (("provenance", "provenance"), ("reachability", "reachability"), ("occupancy", "occupancy")):
        assert read_json(study / f"{name}.json") == result[key]
    sys.path.insert(0, str(study / "source"))
    from q6.world import CollectionWorld, WorldConfig
    cfg = dict(protocol["world"])
    cfg["action_mapping"] = tuple(cfg["action_mapping"])
    config = WorldConfig(**cfg)
    assert asdict(config) == asdict(WorldConfig())
    networks, input_files = audit_inputs(study, result, root)
    assert len(networks) == result["run"]["frozen_models"]
    metadata = read_json(study / "dataset_metadata.json")
    with np.load(study / "dataset.npz") as saved:
        data = {k.removeprefix("train_"): saved[k] for k in saved.files if k.startswith("train_")}
    for name, spec in metadata["arrays"].items():
        if name.startswith("train_"):
            bank_audit.check_array(data[name.removeprefix("train_")], spec)
    assert [r["map_seed"] for r in metadata["train"]] == list(range(300000, 300256))
    assert len(data["observations"]) == 163840
    transitions, worlds, lookup, clone_checks = geometry(data, metadata, config, CollectionWorld)
    supports, tables, targets, shortest = {}, {}, {}, {}
    with np.load(study / "supports.npz") as support_file, np.load(study / "recorded_transitions.npz") as table_file, np.load(study / "logged_graph_targets.npz") as target_file, np.load(study / "reachability.npz") as reach_file:
        for bank in protocol["bank_ids"]:
            support = support_file[f"{CONDITIONS[0]}_bank{bank}"]
            assert support.dtype == np.int32 and np.all(support[1:] > support[:-1])
            assert np.array_equal(support, support_file[f"{CONDITIONS[1]}_bank{bank}"])
            table = graph_audit.reconstruct_recorded_edges(study / f"banks/bank{bank}/collection_steps.csv", data, transitions, support)
            for flag in ("terminated", "truncated"):
                assert np.array_equal(table[flag][table["observed"]], transitions[flag][table["observed"]])
            for key, array in table.items():
                assert np.array_equal(array, table_file[f"bank{bank}_{key}"], equal_nan=True), (bank, key)
            expected_graph, _ = graph_audit.independent_logged_graph(table, data["remaining"], data["map_seeds"], support, config.gamma)
            assert np.array_equal(expected_graph, target_file[f"bank{bank}_targets_float64"], equal_nan=True)
            assert np.array_equal(expected_graph.astype(np.float32), target_file[f"bank{bank}_targets_float32"], equal_nan=True)
            steps, _ = independent_reachability(table, data["remaining"], support)
            assert np.array_equal(reach_file[f"bank{bank}_reachable"], steps > 0) and np.array_equal(reach_file[f"bank{bank}_shortest_steps"], steps)
            supports[bank], tables[bank], targets[bank], shortest[bank] = support, table, expected_graph, steps
            check = next(r for r in result["provenance"]["logged_graph_checks"] if r["bank_id"] == bank)
            assert check["logged_targets_reproduced"]
            for name, array in table.items():
                bank_audit.check_array(array, check["table_arrays"][name])
            bank_audit.check_array(expected_graph, check["arrays"]["targets"])
            bank_audit.check_array(support, check["support_arrays"]["support"])
            starts32 = np.sort(np.asarray([lookup[(m, *worlds[m]["start"], 32)] for m in worlds], np.int64))
            assert np.array_equal(support[data["remaining"][support] == 32], starts32)
            assert check["clock32_support_is_original_start_rows"]
            bank_audit.check_array(starts32, check["original_start_rows"])
            summary = next(r for r in result["reachability"]["graph_summaries"] if r["bank_id"] == bank)
            assert summary["supported_states"] == len(support) and summary["reachable_states"] == int((steps > 0).sum())
            assert summary["unreachable_supported_states"] == len(support) - int((steps > 0).sum()) and summary["closure_verified"]
    assert len(result["provenance"]["graph_integrity"]) == len(supports)
    for item in result["provenance"]["graph_integrity"]:
        assert item["unchanged"] and item["read_only"]
    starts = read_json(study / "starts.json")
    assert starts == result["reachability"]["by_start"]
    assert {(r["bank_id"], r["map_seed"]) for r in starts} == {(b, m) for b in supports for m in protocol["map_seeds"]}
    assert len(starts) == len(supports) * len(protocol["map_seeds"])
    for start in starts:
        bank, map_seed = start["bank_id"], start["map_seed"]
        world = worlds[map_seed]
        row = lookup[(map_seed, *world["start"], 32)]
        distance = int(shortest[bank][row])
        planner_steps = world["distance"][tuple(world["start"])]
        compare(start, {"bank_id": bank, "map_seed": map_seed, "block": f"block_{(map_seed - 300000) // 32}", "state_row": row, "position": world["start"], "remaining": 32,
            "layout_hash": world["layout_hash"], "logged_success_reachable": distance > 0, "logged_shortest_steps": distance if distance > 0 else None,
            "planner_steps": planner_steps, "logged_efficient_success_reachable": bool(0 < distance <= 2 * planner_steps)})
    for bank in supports:
        group = [r for r in starts if r["bank_id"] == bank]
        reach, efficient = (sum(r[k] for r in group) for k in ("logged_success_reachable", "logged_efficient_success_reachable"))
        compare(next(r for r in result["reachability"]["per_bank"] if r["bank_id"] == bank), {"bank_id": bank, "starts": len(group),
            "success_reachable_starts": reach, "success_ceiling": reach / len(group), "efficient_success_reachable_starts": efficient, "efficient_success_ceiling": efficient / len(group)})
    slices = audit_slices(study, result, data, supports, tables, targets, shortest)
    rows, references, step_summary = audit_steps(study, result, data, worlds, lookup, transitions, tables, targets, shortest, networks, config, skip_forward)
    summary = audit_summaries(study, result, rows, references)
    return {"study": str(study), "status": "pass", "forward_inference_checked": not skip_forward,
        "forward_scope": "only prospectively selected saved recordings; no full policy reevaluation", "manifest_files": manifest_files,
        "archive_inputs": input_files, "frozen_models": len(networks), "new_updates": 0, "original_state_rows": len(data["observations"]),
        "independent_transition_entries": len(data["observations"]) * 4, "kernel_clone_checks": clone_checks,
        "recorded_edges": sum(int(t["observed"].sum()) for t in tables.values()), "graph_reachability_states": sum(len(s) for s in supports.values()),
        "prior_prediction_slices": slices, "learner_episodes": len(rows), "reference_episodes": len(references), **step_summary, **summary,
        "prediction_gap_mean_float32_ulp_tolerance": FLOAT32_MEAN_ULPS,
        "eligible_descriptive_study": eligible, "wall_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/familiar_starts/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true", help="Retain saved-Q/transition/arithmetic checks; skip neural regeneration of selected recordings only.")
    args = parser.parse_args()
    print(json.dumps(audit(args.study, args.allow_smoke, args.skip_forward_inference), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
