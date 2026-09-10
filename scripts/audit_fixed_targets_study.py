"""Read-only independent audit of the fixed-data target comparison.

No optimizer updates or policy rollouts are performed. The audit reconstructs
all Bellman transitions, sampling digests, primary gates and final prediction
metrics from archived inputs. Forward inference and bounded single-transition
world checks verify the saved evidence. Use the recorded Python/Torch/NumPy
environment for bit-identical dense predictions.

    python scripts/audit_fixed_targets_study.py --study experiments/fixed_targets/pilot_v1

An explicit --allow-smoke permits structural validation of a small completed
smoke; it never turns smoke outcomes into research evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

CONDITIONS = ("exact_q", "double_dqn")
PANELS = ("train", "heldout")
BUCKETS = ("all", "1-8", "9-16", "17-24", "25-32")
ADDITIVE = ("states", "action_values", "winnable_states", "impossible_states", "optimal_winnable_actions",
            "abs_error_sum", "squared_error_sum", "signed_error_sum", "action_regret_sum",
            "winnable_regret_sum", "impossible_regret_sum")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label="", *, atol=1e-8):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=atol), (label, actual, expected)


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def layout_hash(env):
    payload = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(),
               "config": asdict(env.config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def unique_file(directory, patterns):
    matches = set()
    for pattern in patterns:
        matches.update(directory.glob(pattern))
    assert len(matches) == 1, ("expected exactly one artifact", str(directory), patterns, sorted(map(str, matches)))
    return next(iter(matches))


def audit_manifest(study, protocol, result):
    manifest = read_json(study / "manifest.json")
    assert manifest["algorithm"] == "sha256"
    for name, digest in manifest["files"].items():
        assert sha(study / name) == digest, ("artifact hash", name)
    actual = {p.relative_to(study).as_posix() for p in study.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert actual == set(manifest["files"]), "manifest must cover every archived file"
    assert result["protocol"] == protocol
    assert sha(study / "protocol.md") == protocol["protocol_sha256"]
    for name, digest in protocol["source_sha256"].items():
        assert sha(study / "source" / name) == digest, ("source hash", name)
    return len(actual)


def audit_dataset(study, prior, protocol, metadata, data, config, World, *, smoke):
    assert metadata["status"] == "complete"
    for name, spec in metadata["arrays"].items():
        assert list(data[name].shape) == spec["shape"] and str(data[name].dtype) == spec["dtype"]
        assert array_sha(data[name]) == spec["sha256"], ("array hash", name)
    prior_metadata = read_json(prior / "dataset_metadata.json")
    previous_fresh = {row["layout_hash"] for row in prior_metadata["heldout"]}
    env = World(config)
    for panel in PANELS:
        for row in metadata[panel]:
            env.reset(seed=row["map_seed"])
            assert layout_hash(env) == row["layout_hash"]
            assert list(env.position) == row["original_start"]
    train_hashes = {row["layout_hash"] for row in metadata["train"]}
    fresh_hashes = {row["layout_hash"] for row in metadata["heldout"]}
    assert not train_hashes & fresh_hashes and not previous_fresh & fresh_hashes
    assert len(fresh_hashes) == len(metadata["heldout"])
    duplicate_counts = {key: count for key, count in Counter(row["layout_hash"] for row in metadata["train"]).items() if count > 1}
    assert duplicate_counts == metadata["train_duplicate_layouts"]
    observation_hashes = {panel: {hashlib.sha256(row.tobytes()).digest() for row in data[f"{panel}_observations"]}
                          for panel in PANELS}
    assert not observation_hashes["train"] & observation_hashes["heldout"]
    if not smoke:
        assert [row["map_seed"] for row in metadata["train"]] == list(range(300000, 300256))
        assert len(metadata["heldout"]) == 64
        with np.load(prior / "dataset.npz", allow_pickle=False) as old_data:
            for name, array in data.items():
                if name.startswith("train_"):
                    assert np.array_equal(array, old_data[name]), ("historical training-array identity", name)
        previous_protocol = read_json(prior / "protocol.json")
        for name in ("q6/world.py", "q6/learning.py"):
            assert protocol["source_sha256"][name] == previous_protocol["source_sha256"][name]
        # Reconstruct candidate admission independently of the saved decisions.
        accepted, rejected, seen = [], [], set()
        candidate = 940000
        while len(accepted) < 64:
            env.reset(seed=candidate)
            key = layout_hash(env)
            if key in train_hashes or key in previous_fresh or key in seen:
                rejected.append((candidate, key))
            else:
                accepted.append((candidate, key))
                seen.add(key)
            candidate += 1
        assert accepted == [(row["map_seed"], row["layout_hash"]) for row in metadata["heldout"]]
        assert rejected == [(row["map_seed"], row["layout_hash"]) for row in metadata["collision_skips"]]
        assert all(row["reason"] for row in metadata["collision_skips"])
    return {"train_layouts": len(train_hashes), "fresh_layouts": len(fresh_hashes),
            "train_fresh_observation_intersections": 0, "previous_fresh_layout_intersections": 0,
            "training_arrays_equal_previous_supervised": None if smoke else True}


def audit_transitions(data, metadata, transitions, config, World):
    """Finite-horizon recurrence is unique after successor time ordering is checked."""
    env = World(config)
    rewards, ends, successors = (transitions[key] for key in ("rewards", "ends", "successor_indices"))
    for name, spec in metadata["transition_arrays"].items():
        assert list(transitions[name].shape) == spec["shape"] and str(transitions[name].dtype) == spec["dtype"]
        assert array_sha(transitions[name]) == spec["sha256"]
    count = len(data["train_observations"])
    assert rewards.shape == ends.shape == successors.shape == (count, 4)
    assert np.isfinite(rewards).all() and np.isin(ends, (0, 1)).all()
    assert successors.dtype.kind in "iu"
    assert np.all(successors[np.asarray(ends, bool)] == -1)
    assert np.all((successors[~np.asarray(ends, bool)] >= 0) & (successors[~np.asarray(ends, bool)] < count))
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    target_entries, clone_checks, transition_entries = 0, 0, 0
    max_bellman_error = max_reward_error = 0.0
    for panel in PANELS:
        observations, targets = data[f"{panel}_observations"], data[f"{panel}_targets"]
        n, horizon = config.size, config.horizon
        positions_per_map = n * n - config.wall_count - 1
        rows_per_map = positions_per_map * horizon
        assert observations.shape == (len(metadata[panel]) * rows_per_map, 3 * n * n + 17)
        assert targets.shape == (len(observations), 4)
        for map_index, layout in enumerate(metadata[panel]):
            env.reset(seed=layout["map_seed"])
            positions = [tuple(map(int, p)) for p in np.argwhere(~(env.walls | env.pellets))]
            position_index = {position: index for index, position in enumerate(positions)}
            goal = tuple(map(int, np.argwhere(env.pellets)[0]))
            distance, queue = {goal: 0}, deque([goal])
            while queue:
                cell = queue.popleft()
                for dr, dc in directions:
                    nxt = cell[0] + dr, cell[1] + dc
                    if 0 <= nxt[0] < n and 0 <= nxt[1] < n and not env.walls[nxt] and nxt not in distance:
                        distance[nxt] = distance[cell] + 1
                        queue.append(nxt)
            assert len(distance) == n * n - config.wall_count
            start = map_index * rows_per_map
            section = slice(start, start + rows_per_map)
            assert np.array_equal(data[f"{panel}_positions"][section], np.repeat(positions, horizon, axis=0))
            assert np.array_equal(data[f"{panel}_remaining"][section], np.tile(np.arange(1, horizon + 1), len(positions)))
            assert np.array_equal(data[f"{panel}_map_seeds"][section], np.full(rows_per_map, layout["map_seed"]))
            for pi, position in enumerate(positions):
                for remaining in range(1, horizon + 1):
                    index = start + pi * horizon + remaining - 1
                    env.position, env.elapsed = position, horizon - remaining
                    assert np.array_equal(observations[index], env.observe())
                    assert bool(data[f"{panel}_winnable"][index]) == (distance[position] <= remaining)
                    for action, physical in enumerate(config.action_mapping):
                        dr, dc = directions[physical]
                        nxt = position[0] + dr, position[1] + dc
                        if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or env.walls[nxt]:
                            nxt = position
                        collected = nxt == goal
                        ended = collected or remaining == 1
                        next_index = -1 if ended else start + position_index[nxt] * horizon + remaining - 2
                        before, after = -distance[position] / n, 0.0 if ended else -distance[nxt] / n
                        reward = config.step_cost + config.pellet_reward * collected + config.shaping_weight * (config.gamma * after - before)
                        expected = reward + (0.0 if ended else config.gamma * targets[next_index].max())
                        max_bellman_error = max(max_bellman_error, abs(expected - targets[index, action]))
                        target_entries += 1
                        if panel == "train":
                            assert bool(ends[index, action]) == ended
                            assert int(successors[index, action]) == next_index
                            max_reward_error = max(max_reward_error, abs(float(rewards[index, action]) - reward))
                            transition_entries += 1
                        if map_index < 2 and remaining in (1, 2, horizon):
                            clone = env.clone()
                            _, actual_reward, terminated, truncated, _ = clone.step(action)
                            close(actual_reward, reward, "clone reward")
                            assert clone.position == nxt and (terminated or truncated) == ended
                            clone_checks += 1
    assert max_bellman_error < 1e-12, max_bellman_error
    assert max_reward_error < 1e-7, max_reward_error
    return {"exact_target_entries_checked": target_entries, "saved_transition_entries_checked": transition_entries,
            "max_bellman_error": max_bellman_error, "max_saved_reward_error": max_reward_error,
            "single_transition_clone_checks": clone_checks}


def audit_sampling(study, prior, protocol, data, result, *, smoke):
    sampling = read_json(study / "sampling.json")
    assert sampling == result["sampling"]
    counts = dict(np.load(study / "sample_counts.npz", allow_pickle=False))
    n = len(data["train_observations"])
    prior_sampling = read_json(prior / "sampling.json")
    checked = 0
    for seed in protocol["seeds"]:
        left, right = (sampling[condition][str(seed)] for condition in CONDITIONS)
        assert left["updates"] == right["updates"]
        updates = left["updates"]
        assert updates == max(protocol["checkpoints"])
        expected = np.zeros(n, np.uint32)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
        digest = hashlib.sha256()
        for _ in range(updates):
            indices = rng.choice(n, 64, replace=False)
            expected[indices] += 1
            digest.update(indices.astype("<i8").tobytes())
        for condition in CONDITIONS:
            row = sampling[condition][str(seed)]
            assert row["examples_seen"] == updates * 64
            assert row["unique_states_sampled"] == np.count_nonzero(expected)
            assert row["batch_index_sha256"] == digest.hexdigest()
            assert np.array_equal(counts[f"{condition}_seed{seed}"], expected)
            checked += 1
        if not smoke:
            assert left["batch_index_sha256"] == prior_sampling[str(seed)]["batch_index_sha256"]
    return {"paired_sampler_digests_reconstructed": checked, "samplers_equal_previous_supervised": None if smoke else True}


def audit_models(study, prior, protocol, data, result, DQN, *, smoke, skip_forward=False):
    model_count = 0
    predictions = {}
    final = max(protocol["checkpoints"])
    for condition in CONDITIONS:
        for seed in protocol["seeds"]:
            initial_hash = None
            for checkpoint in protocol["checkpoints"]:
                path = unique_file(study / "models", [f"{condition}_seed{seed}_update{checkpoint}.pt"])
                saved = torch.load(path, map_location="cpu", weights_only=True)
                model = DQN(saved["observation_size"], seed=seed)
                model.online.load_state_dict(saved["online"])
                assert model.parameter_hash() == saved["parameter_hash"]
                assert sum(p.numel() for p in model.online.parameters()) == 20420
                assert "target" in saved, "target-network snapshot required by protocol"
                assert all(torch.isfinite(tensor).all() for tensor in saved["target"].values())
                target_digest = hashlib.sha256()
                for tensor in saved["target"].values():
                    target_digest.update(tensor.numpy().tobytes())
                assert target_digest.hexdigest() == saved["target_parameter_hash"]
                assert saved["purpose"] == "inference_only_not_resumable"
                assert saved["optimizer_updates"] == checkpoint
                if checkpoint == 0:
                    initial_hash = model.parameter_hash()
                    assert initial_hash == DQN(92, seed=seed).parameter_hash()
                    assert initial_hash == result["run"]["initial_policy_hashes"][condition][str(seed)]
                    for key in saved["online"]:
                        assert torch.equal(saved["online"][key], saved["target"][key])
                if condition == "exact_q":
                    original_target = DQN(92, seed=seed).target.state_dict()
                    for key in original_target:
                        assert torch.equal(original_target[key], saved["target"][key])
                if checkpoint == final:
                    dense_path = unique_file(study, [f"predictions_{condition}_seed{seed}.npz", f"{condition}_predictions_seed{seed}.npz"])
                    predictions[condition, seed] = dict(np.load(dense_path, allow_pickle=False))
                    for panel in PANELS:
                        observations = data[f"{panel}_observations"]
                        assert predictions[condition, seed][panel].shape == (len(observations), 4)
                        assert np.isfinite(predictions[condition, seed][panel]).all()
                        if not skip_forward:
                            actual = np.empty((len(observations), 4), np.float32)
                            with torch.no_grad():
                                for start in range(0, len(observations), 4096):
                                    actual[start:start + 4096] = model.online(torch.from_numpy(observations[start:start + 4096])).numpy()
                            assert np.array_equal(actual, predictions[condition, seed][panel]), ("dense predictions", condition, seed, panel)
                    if condition == "exact_q" and not smoke:
                        previous = torch.load(prior / "models" / f"supervised_q_seed{seed}_update30000.pt", map_location="cpu", weights_only=True)
                        assert saved["parameter_hash"] == previous["parameter_hash"], ("historical exact-arm identity", seed)
                model_count += 1
    assert model_count == 2 * len(protocol["seeds"]) * len(protocol["checkpoints"])
    return predictions, {"model_snapshots": model_count, "forward_inference_checked": not skip_forward,
                         "dense_prediction_arrays_reproduced": 0 if skip_forward else 4 * len(protocol["seeds"]),
                         "exact_final_weights_equal_previous_supervised": None if smoke else True}


def audit_episode_summary(summary, rows):
    assert summary["episodes"] == len(rows)
    for raw, metric in (("success", "success_rate"), ("steps", "mean_steps"), ("base_return", "mean_return"), ("shaped_return", "mean_shaped_return")):
        close(summary[metric], np.mean([float(row[raw]) for row in rows]), metric)
    for numerator, denominator, metric in (("noop_steps", "steps", "noop_rate"), ("moved_revisits", "moving_steps", "moved_revisit_rate"),
        ("optimal_winnable_actions", "winnable_steps", "optimal_action_rate"), ("action_regret_sum", "steps", "mean_action_regret"),
        ("q_abs_error_sum", "q_error_steps", "mean_abs_q_error")):
        denom = sum(float(row[denominator]) for row in rows)
        if denom:
            close(summary[metric], sum(float(row[numerator]) for row in rows) / denom, metric)
        else:
            assert summary[metric] is None


def audit_raw_tables(study, metadata, data, protocol, result, predictions, *, smoke):
    episodes, references, states, losses = [csv_rows(study / name) for name in
        ("evaluations.csv", "references.csv", "state_metrics.csv", "losses.csv")]
    seeds, checkpoints = protocol["seeds"], protocol["checkpoints"]
    final = max(checkpoints)
    expected_episodes = {(condition, seed, checkpoint, panel, mode, layout["map_seed"], rep)
        for condition in CONDITIONS for seed in seeds for checkpoint in checkpoints for panel in PANELS
        for mode in ("greedy", "epsilon_0_1") for layout in metadata[panel] for rep in range(1 if mode == "greedy" else 2)}
    episode_key = lambda r: (r["condition"], int(r["seed"]), int(r["checkpoint"]), r["panel"], r["mode"], int(r["map_seed"]), int(r["repetition"]))
    actual_keys = [episode_key(row) for row in episodes]
    assert len(set(actual_keys)) == len(actual_keys) and set(actual_keys) == expected_episodes
    for row in episodes + references:
        assert int(row["checkpoint_complete"]) == 1
        steps, success = int(row["steps"]), int(row["success"])
        assert 1 <= steps <= 32 and success in (0, 1)
        close(row["base_return"], success - .01 * steps)
        assert int(row["noop_steps"]) + int(row["moving_steps"]) == steps
        assert 0 <= int(row["moved_revisits"]) <= int(row["moving_steps"])
        assert 0 <= int(row["optimal_winnable_actions"]) <= int(row["winnable_steps"]) <= steps
        assert int(row["avoidable_failure_actions"]) == 1 - success
    expected_refs = {(policy, seed, panel, layout["map_seed"], rep)
        for policy in ("random_actions", "shortest_path") for seed in ([0] if policy == "shortest_path" else seeds)
        for panel in PANELS for layout in metadata[panel] for rep in range(1 if policy == "shortest_path" else 2)}
    ref_keys = [(r["policy"], int(r["seed"]), r["panel"], int(r["map_seed"]), int(r["repetition"])) for r in references]
    assert len(set(ref_keys)) == len(ref_keys) and set(ref_keys) == expected_refs
    assert len({r["reference_sample_id"] for r in references}) == len(references)
    assert result["run"]["reference_episodes"] == result["run"]["unique_reference_episodes"] == len(references)
    planner = {(r["panel"], int(r["map_seed"])): int(r["steps"]) for r in references if r["policy"] == "shortest_path"}
    for panel in PANELS:
        for layout in metadata[panel]:
            mask = ((data[f"{panel}_map_seeds"] == layout["map_seed"]) &
                    np.all(data[f"{panel}_positions"] == layout["original_start"], axis=1) & data[f"{panel}_winnable"])
            # Reachability labels were independently checked against BFS above.
            distance = int(data[f"{panel}_remaining"][mask].min())
            assert planner[panel, layout["map_seed"]] == distance
    assert all(int(r["success"]) == 1 for r in references if r["policy"] == "shortest_path")

    def efficient(rows):
        return sum(int(r["success"]) == 1 and int(r["steps"]) <= 2 * planner[r["panel"], int(r["map_seed"])]
                   for r in rows) / len(rows)

    for summary in result["aggregate"] + result["seed_results"]:
        selected = [r for r in episodes if all(str(r[k]) == str(summary[k]) for k in ("condition", "checkpoint", "panel", "mode"))
                    and ("seed" not in summary or int(r["seed"]) == summary["seed"])]
        audit_episode_summary(summary, selected)
        successful = [r for r in selected if int(r["success"])]
        successful_steps = sum(int(r["steps"]) for r in successful)
        successful_noops = sum(int(r["noop_steps"]) for r in successful)
        assert summary["successful_episodes"] == len(successful)
        assert summary["successful_episode_steps"] == successful_steps
        assert summary["successful_noop_steps"] == successful_noops
        if successful:
            close(summary["successful_mean_steps"], successful_steps / len(successful))
            close(summary["successful_noop_rate"], successful_noops / successful_steps)
        else:
            assert summary["successful_mean_steps"] is summary["successful_noop_rate"] is None
        close(summary["efficient_success_rate"], efficient(selected), "efficient success / all episodes")
        if "seed" not in summary:
            rates = [np.mean([int(r["success"]) for r in selected if int(r["seed"]) == seed]) for seed in seeds]
            assert summary["seeds"] == len(seeds)
            close(summary["seed_success_min"], min(rates)); close(summary["seed_success_max"], max(rates))
    for summary in result["references"]:
        selected = [r for r in references if (r["policy"], r["panel"]) == (summary["policy"], summary["panel"])]
        audit_episode_summary(summary, selected)
        close(summary["efficient_success_rate"], efficient(selected))
    for condition in CONDITIONS:
        for seed in seeds:
            rows = [r for r in losses if r["condition"] == condition and int(r["seed"]) == seed]
            expected_loss_steps = sorted(set(range(100, final + 1, 100)) | {cp for cp in checkpoints if cp > 0})
            assert [int(r["checkpoint"]) for r in rows] == expected_loss_steps
            previous = 0
            for row in rows:
                checkpoint = int(row["checkpoint"])
                assert int(row["updates_in_window"]) == checkpoint - previous
                assert int(row["training_examples"]) == checkpoint * 64
                assert np.isfinite(float(row["mean_loss"])) and float(row["mean_loss"]) >= 0
                previous = checkpoint
    for summary in result["loss_aggregate"]:
        selected = [r for r in losses if r["condition"] == summary["condition"] and int(r["checkpoint"]) == summary["checkpoint"]]
        values = [float(r["mean_loss"]) for r in selected]
        close(summary["mean_loss"], np.mean(values)); close(summary["seed_loss_min"], min(values)); close(summary["seed_loss_max"], max(values))
        assert summary["seeds"] == len(selected)
        assert all(int(r["updates_in_window"]) == summary["updates_in_window"] for r in selected)
    expected_states = {(condition, seed, checkpoint, panel, layout["map_seed"], bucket) for condition in CONDITIONS
        for seed in seeds for checkpoint in checkpoints for panel in PANELS for layout in metadata[panel] for bucket in BUCKETS}
    state_key = lambda r: (r["condition"], int(r["seed"]), int(r["checkpoint"]), r["panel"], int(r["map_seed"]), r["time_bucket"])
    state_keys = [state_key(r) for r in states]
    assert len(set(state_keys)) == len(state_keys) and set(state_keys) == expected_states
    state_groups = defaultdict(list)
    final_states = defaultdict(list)
    for row in states:
        state_groups[row["condition"], int(row["checkpoint"]), row["panel"], row["time_bucket"]].append(row)
        if int(row["checkpoint"]) == final:
            final_states[row["condition"], int(row["seed"]), row["panel"]].append(row)
    errors, final_agreement = {}, {}
    indices_cache = {}
    for panel in PANELS:
        for layout in metadata[panel]:
            map_indices = np.flatnonzero(data[f"{panel}_map_seeds"] == layout["map_seed"])
            for bucket in BUCKETS:
                low, high = (1, 32) if bucket == "all" else map(int, bucket.split("-"))
                remaining = data[f"{panel}_remaining"][map_indices]
                indices_cache[panel, layout["map_seed"], bucket] = map_indices[(remaining >= low) & (remaining <= high)]
    for (condition, seed), dense in predictions.items():
        for panel in PANELS:
            target = data[f"{panel}_targets"]
            prediction = dense[panel].astype(np.float64)
            error = prediction - target
            regret = np.maximum(0.0, target.max(1) - target[np.arange(len(target)), prediction.argmax(1)])
            win = data[f"{panel}_winnable"]
            errors[condition, seed, panel] = np.abs(error).mean(1)
            final_agreement[condition, seed, panel] = float((regret[win] <= 1e-6).mean())
            for row in final_states[condition, seed, panel]:
                index = indices_cache[panel, int(row["map_seed"]), row["time_bucket"]]
                e, r, w = error[index], regret[index], win[index]
                expected = {"states": len(e), "action_values": e.size, "winnable_states": int(w.sum()),
                    "impossible_states": int((~w).sum()), "optimal_winnable_actions": int(((r <= 1e-6) & w).sum()),
                    "abs_error_sum": np.abs(e).sum(), "squared_error_sum": np.square(e).sum(), "signed_error_sum": e.sum(),
                    "action_regret_sum": r.sum(), "winnable_regret_sum": r[w].sum(), "impossible_regret_sum": r[~w].sum(),
                    "mean_abs_q_error": np.abs(e).mean(), "rmse_q_error": np.sqrt(np.square(e).mean()),
                    "q95_abs_q_error": np.quantile(np.abs(e).mean(1), .95), "mean_signed_q_bias": e.mean(), "mean_action_regret": r.mean()}
                for name, value in expected.items():
                    close(row[name], value, name)
                if w.any():
                    close(row["optimal_action_rate"], (r[w] <= 1e-6).mean())
                    close(row["winnable_mean_action_regret"], r[w].mean())
                else:
                    assert row["optimal_action_rate"] == row["winnable_mean_action_regret"] == ""
                if (~w).any():
                    close(row["impossible_action_regret"], r[~w].mean())
                else:
                    assert row["impossible_action_regret"] == ""
    for summary in result["state_aggregate"]:
        selected = state_groups[summary["condition"], summary["checkpoint"], summary["panel"], summary["time_bucket"]]
        for metric in ADDITIVE:
            close(summary[metric], sum(float(r[metric]) for r in selected), metric)
        ratios = (("abs_error_sum", "action_values", "mean_abs_q_error"), ("signed_error_sum", "action_values", "mean_signed_q_bias"),
            ("action_regret_sum", "states", "mean_action_regret"), ("optimal_winnable_actions", "winnable_states", "optimal_action_rate"),
            ("winnable_regret_sum", "winnable_states", "winnable_mean_action_regret"), ("impossible_regret_sum", "impossible_states", "impossible_action_regret"))
        for numerator, denominator, metric in ratios:
            if summary[denominator]:
                close(summary[metric], summary[numerator] / summary[denominator], metric)
            else:
                assert summary[metric] is None
        close(summary["rmse_q_error"], np.sqrt(summary["squared_error_sum"] / summary["action_values"]))
        per_seed = []
        for seed in seeds:
            rows = [r for r in selected if int(r["seed"]) == seed]
            win = sum(int(r["winnable_states"]) for r in rows)
            if win:
                per_seed.append(sum(int(r["optimal_winnable_actions"]) for r in rows) / win)
        assert summary["seeds"] == len(seeds)
        if per_seed:
            close(summary["seed_optimal_action_min"], min(per_seed)); close(summary["seed_optimal_action_max"], max(per_seed))
        if summary["checkpoint"] == final:
            panel, bucket = summary["panel"], summary["time_bucket"]
            low, high = (1, 32) if bucket == "all" else map(int, bucket.split("-"))
            mask = (data[f"{panel}_remaining"] >= low) & (data[f"{panel}_remaining"] <= high)
            values = np.concatenate([errors[summary["condition"], seed, panel][mask] for seed in seeds])
            close(summary["q95_abs_q_error"], np.quantile(values, .95))

    random_rows = [r for r in references if r["policy"] == "random_actions" and r["panel"] == "heldout"]
    random_rate = float(np.mean([int(r["success"]) for r in random_rows]))
    assert result["gates"]["eligible"] == (not smoke)
    close(result["gates"]["pooled_random_fresh_success"], random_rate)
    final_episode_lookup = {(r["condition"], int(r["seed"]), r["panel"], int(r["map_seed"])): r
                           for r in episodes if int(r["checkpoint"]) == final and r["mode"] == "greedy"}
    for condition_gate in result["gates"]["per_condition"]:
        condition = condition_gate["condition"]
        assert {r["seed"] for r in condition_gate["per_seed"]} == set(seeds)
        for gate in condition_gate["per_seed"]:
            seed = gate["seed"]
            groups = {panel: [final_episode_lookup[condition, seed, panel, layout["map_seed"]] for layout in metadata[panel]] for panel in PANELS}
            rates = {panel: float(np.mean([int(r["success"]) for r in rows])) for panel, rows in groups.items()}
            agreement = final_agreement[condition, seed, "train"]
            efficiency = efficient(groups["heldout"])
            close(gate["train_success"], rates["train"]); close(gate["fresh_success"], rates["heldout"])
            close(gate["train_state_optimal"], agreement); close(gate["fresh_efficient_success"], efficiency)
            assert gate["training_fit"] == bool(not smoke and rates["train"] >= .9 and agreement >= .9)
            assert gate["fresh"] == bool(not smoke and rates["heldout"] >= .7 and rates["heldout"] > random_rate)
            assert gate["efficient"] == bool(not smoke and efficiency >= .8)
        for gate_name in ("training_fit", "fresh", "efficient"):
            assert condition_gate[gate_name] == all(r[gate_name] for r in condition_gate["per_seed"])
    assert {g["condition"] for g in result["gates"]["per_condition"]} == set(CONDITIONS)
    gates = {row["condition"]: row for row in result["gates"]["per_condition"]}
    expected_interpretation = ("smoke_or_protocol_deviation_not_gate_evidence" if smoke else
        "both_fresh_gates_met" if all(g["fresh"] for g in gates.values()) else "exact_only_fresh_gate_met" if gates["exact_q"]["fresh"] else
        "bootstrap_only_fresh_gate_met" if gates["double_dqn"]["fresh"] else "neither_fresh_gate_met")
    assert result["run"]["interpretation"] == expected_interpretation
    paired = read_json(study / "paired_differences.json")
    assert paired == result["paired_differences"]
    assert paired["direction"] == "double_dqn minus exact_q"
    assert len(paired["per_layout"]) == len(seeds) * sum(len(metadata[p]) for p in PANELS)
    assert len(paired["per_seed"]) == len(seeds) * 2
    for row in paired["per_layout"]:
        left, right = (final_episode_lookup[c, row["seed"], row["panel"], row["map_seed"]] for c in CONDITIONS)
        for metric in ("success", "steps", "noop_steps"):
            close(row[f"{metric}_delta"], int(right[metric]) - int(left[metric]))
        close(row["noop_rate_delta"], int(right["noop_steps"]) / int(right["steps"]) - int(left["noop_steps"]) / int(left["steps"]))
    for row in paired["per_seed"]:
        pair = {c: [final_episode_lookup[c, row["seed"], row["panel"], layout["map_seed"]] for layout in metadata[row["panel"]]] for c in CONDITIONS}
        values = {c: {"success_rate": np.mean([int(r["success"]) for r in rows]), "mean_steps": np.mean([int(r["steps"]) for r in rows]),
            "noop_rate": sum(int(r["noop_steps"]) for r in rows) / sum(int(r["steps"]) for r in rows), "efficient_success_rate": efficient(rows)} for c, rows in pair.items()}
        for metric in values["exact_q"]:
            close(row[f"{metric}_delta"], values["double_dqn"][metric] - values["exact_q"][metric])
    for row in paired["aggregate"]:
        selected = [r for r in paired["per_seed"] if r["panel"] == row["panel"]]
        assert row["seeds"] == len(seeds)
        for metric in ("success_rate", "mean_steps", "noop_rate", "efficient_success_rate"):
            close(row[f"mean_seed_{metric}_delta"], np.mean([r[f"{metric}_delta"] for r in selected]))
    return {"evaluation_rows": len(episodes), "reference_rows": len(references), "state_metric_rows": len(states),
            "loss_rows": len(losses), "paired_layout_rows": len(paired["per_layout"]), "gates": result["gates"]}


def audit_provenance(study, prior, result, protocol, *, smoke):
    provenance = read_json(study / "provenance.json")
    assert provenance == result["provenance"]
    for name, digest in provenance["files"].items():
        assert sha(prior / name) == digest, ("historical archive hash", name)
    for name in ("dataset_metadata.json", "sampling.json", "protocol.json"):
        assert sha(study / f"prior_{name}") == sha(prior / name)
    assert {r["seed"] for r in provenance["paired_consistency"]} == set(protocol["seeds"])
    for row in provenance["paired_consistency"]:
        assert all(row[key] for key in ("complete", "initial_weights_identical", "batch_index_sha256_identical", "per_row_counts_identical"))
    if not smoke:
        assert provenance["replication_required"] and provenance["training_arrays_identical"]
        assert {r["seed"] for r in provenance["exact_replication"]} == set(protocol["seeds"])
        for row in provenance["exact_replication"]:
            assert row["applicable"] and row["final_weights_identical"] and row["batch_index_sha256_identical"] and row["initial_weights_identical"]
            path = prior / "models" / f"supervised_q_seed{row['seed']}_update30000.pt"
            assert sha(path) == row["archive_model_sha256"]
    maximum = 2 * len(protocol["seeds"]) * max(protocol["checkpoints"])
    assert result["run"]["train_updates"] == maximum
    assert result["run"]["training_examples"] == maximum * 64
    assert result["run"]["model_parameter_count"] == 20420
    assert result["run"]["resource_checks"] > maximum
    assert result["run"]["peak_rss_bytes"] <= protocol["budget"]["peak_process_rss_bytes"]
    assert result["run"]["wall_seconds"] < protocol["budget"]["admission_seconds"]
    progress = result["run"]["progress"]
    assert all(row["evaluation_complete"] for row in progress)
    assert {(r["condition"], r["seed"], r["checkpoint"]) for r in progress} == {
        (c, s, cp) for c in CONDITIONS for s in protocol["seeds"] for cp in protocol["checkpoints"]}
    assert len(progress) == 2 * len(protocol["seeds"]) * len(protocol["checkpoints"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, default=Path("experiments/fixed_targets/pilot_v1"))
    parser.add_argument("--prior-study", type=Path, default=Path("experiments/supervised/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true",
                        help="Omit only bit-exact regenerated model predictions; retain saved-prediction metric checks.")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Assertions must be enabled; do not run python -O.")
    root = Path(__file__).resolve().parents[1]
    study = (args.study if args.study.is_absolute() else root / args.study).resolve()
    prior = (args.prior_study if args.prior_study.is_absolute() else root / args.prior_study).resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(study / "source"))
    from q6.learning import DQN
    from q6.world import CollectionWorld, WorldConfig
    torch.set_num_threads(1)
    started = time.monotonic()
    result, protocol, metadata = (read_json(study / name) for name in ("results.json", "protocol.json", "dataset_metadata.json"))
    smoke = bool(protocol["smoke"] or protocol["deviations"])
    assert not smoke or args.allow_smoke, "Smoke/deviating runs require explicit --allow-smoke and cannot support gates."
    assert result["run"]["status"] == "complete", "This audit requires completed artifacts."
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2]
        assert protocol["checkpoints"] == [0, 1000, 3000, 10000, 30000]
        assert protocol["runtime"]["python"].startswith("3.12.")
        assert protocol["runtime"]["torch"].split("+")[0] == "2.8.0"
        assert protocol["runtime"]["numpy"] == "2.0.2"
        assert protocol["git"]["dirty"] is False
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig())
    data = dict(np.load(study / "dataset.npz", allow_pickle=False))
    transitions = dict(np.load(study / "transitions.npz", allow_pickle=False))
    report = {"study": str(args.study), "smoke_validation_only": smoke,
              "manifest_files": audit_manifest(study, protocol, result)}
    report.update(audit_dataset(study, prior, protocol, metadata, data, config, CollectionWorld, smoke=smoke))
    report.update(audit_transitions(data, metadata, transitions, config, CollectionWorld))
    report.update(audit_sampling(study, prior, protocol, data, result, smoke=smoke))
    predictions, model_report = audit_models(study, prior, protocol, data, result, DQN, smoke=smoke, skip_forward=args.skip_forward_inference)
    report.update(model_report)
    audit_provenance(study, prior, result, protocol, smoke=smoke)
    report.update(audit_raw_tables(study, metadata, data, protocol, result, predictions, smoke=smoke))
    report["audit_seconds"] = time.monotonic() - started
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
