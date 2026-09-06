"""Audit frozen-policy panel evidence without repeating the full evaluation.

Reconstructs panel admission, artifact/model identity, raw arithmetic and only
outcome-independent saved replay traces. --skip-forward-inference skips model
forward reproduction, while retaining saved-Q action/metric and model-hash checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
import audit_fixed_targets_study as common
from audit_fixed_targets_study import close, csv_rows, read_json

CONDITIONS = ("collected_unique", "uniform_subset", "exhaustive")
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def tensor_hash(state):
    digest = hashlib.sha256()
    for tensor in state.values():
        assert torch.isfinite(tensor).all()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def independent_world_values(env):
    """BFS and finite-horizon recurrence, independent of the archived oracle."""
    config = env.config
    goal = tuple(map(int, np.argwhere(env.pellets)[0]))
    distance, queue = {goal: 0}, deque([goal])
    while queue:
        p = queue.popleft()
        for dr, dc in DELTAS:
            q = p[0] + dr, p[1] + dc
            if 0 <= q[0] < config.size and 0 <= q[1] < config.size and not env.walls[q] and q not in distance:
                distance[q] = distance[p] + 1
                queue.append(q)
    assert len(distance) == config.size * config.size - config.wall_count
    positions = [p for p in distance if p != goal]
    following = {}
    for p in positions:
        for action, physical in enumerate(config.action_mapping):
            dr, dc = DELTAS[physical]
            q = p[0] + dr, p[1] + dc
            following[p, action] = q if q in distance else p
    values = np.zeros((config.horizon + 1, config.size, config.size, 4), np.float64)
    for remaining in range(1, config.horizon + 1):
        for p in positions:
            before = -distance[p] / config.size
            for action in range(4):
                q = following[p, action]
                collected = q == goal
                ended = collected or remaining == 1
                after = 0.0 if ended else -distance[q] / config.size
                reward = config.step_cost + config.pellet_reward * collected + config.shaping_weight * (config.gamma * after - before)
                values[remaining, *p, action] = reward + (0.0 if ended else config.gamma * values[remaining - 1, *q].max())
    return distance, following, values, goal


def check_saved_replay(trace, row, env, agent, *, skip_forward):
    """Validate actions and transitions already recorded; do not choose new traces."""
    config = env.config
    state, _ = env.reset(seed=trace["map_seed"])
    distance, following, exact_values, goal = independent_world_values(env)
    assert trace["grid_size"] == config.size and trace["action_mapping"] == list(config.action_mapping)
    assert trace["walls"] == np.argwhere(env.walls).tolist() and trace["pellets_initial"] == np.argwhere(env.pellets).tolist()
    assert trace["start"] == list(env.position)
    payload = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(),
               "position": list(env.position), "config": asdict(config)}
    assert row["task_hash"] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    rng = np.random.default_rng(np.random.SeedSequence([trace["seed"], trace["map_seed"], trace.get("repetition", 0), 55219]))
    coins, random_actions = rng.random(config.horizon), rng.integers(0, 4, config.horizon)
    visited = {env.position}
    counts = {key: 0 for key in ("noop_steps", "moving_steps", "moved_revisits", "winnable_steps", "optimal_winnable_actions", "avoidable_failure_actions", "q_error_steps")}
    sums = {key: 0.0 for key in ("base_return", "shaped_return", "action_regret_sum", "q_abs_error_sum")}
    initial_error = None
    for step, frame in enumerate(trace["steps"]):
        remaining, before = config.horizon - step, env.position
        optimal = exact_values[remaining, *before]
        assert np.allclose(frame["optimal_q_values"], optimal, atol=1e-12, rtol=0)
        possible = np.asarray([following[before, a] == goal or distance[following[before, a]] <= remaining - 1 for a in range(4)])
        winnable = distance[before] <= remaining
        assert frame["winnable"] == winnable and frame["action_can_finish"] == possible.tolist()
        action = int(frame["action"])
        assert 0 <= action < 4
        if trace["policy"] == "learner":
            saved_q = np.asarray(frame["q_values"], np.float32)
            assert saved_q.shape == (4,) and np.isfinite(saved_q).all()
            if not skip_forward:
                with torch.no_grad():
                    actual = agent.online(torch.from_numpy(state).unsqueeze(0))[0].numpy()
                assert np.array_equal(saved_q, actual), "saved replay network values"
            expected_action = int(saved_q.argmax())
            if trace["mode"] == "epsilon_0_1" and coins[step] < .1:
                expected_action = int(random_actions[step])
            assert action == expected_action, "saved action and independent evaluation draws"
            error = float(np.abs(saved_q.astype(np.float64) - optimal).mean())
            sums["q_abs_error_sum"] += error
            counts["q_error_steps"] += 1
            if initial_error is None:
                initial_error = error
        elif trace["policy"] == "random_actions":
            assert action == int(random_actions[step])
        elif trace["policy"] == "shortest_path":
            # BFS ties use physical up/down/left/right order in the shared policy.
            improving = [a for physical in range(4) for a, direction in enumerate(config.action_mapping)
                         if direction == physical and distance[following[before, a]] == distance[before] - 1]
            assert improving and action == improving[0]
        else:
            raise AssertionError("unknown saved policy")
        regret = max(0.0, float(optimal.max() - optimal[action]))
        close(frame["action_regret"], regret)
        sums["action_regret_sum"] += regret
        if winnable:
            counts["winnable_steps"] += 1
            counts["optimal_winnable_actions"] += int(regret <= 1e-6)
            counts["avoidable_failure_actions"] += int(not possible[action])
        after = following[before, action]
        success = after == goal
        ended = success or remaining == 1
        potential_next = 0.0 if ended else -distance[after] / config.size
        base_reward = config.step_cost + config.pellet_reward * success
        reward = base_reward + config.shaping_weight * (config.gamma * potential_next + distance[before] / config.size)
        assert frame["position"] == list(after) and frame["terminated"] == success and frame["truncated"] == (ended and not success)
        assert frame["pellets"] == ([] if success else [list(goal)])
        close(frame["reward"], reward); close(frame["base_reward"], base_reward)
        assert ended == (step == len(trace["steps"]) - 1)
        if before == after:
            counts["noop_steps"] += 1
        else:
            counts["moving_steps"] += 1
            counts["moved_revisits"] += int(after in visited)
        visited.add(after)
        sums["base_return"] += base_reward
        sums["shaped_return"] += reward
        # Feed the validated recorded action to the kernel solely to reconstruct
        # its next observation for saved-model forward checks.
        state, actual_reward, terminated, truncated, _ = env.step(action)
        close(actual_reward, reward)
        assert env.position == after and terminated == success and truncated == (ended and not success)
    assert trace["success"] == bool(int(row["success"])) == env.terminated
    assert len(trace["steps"]) == int(row["steps"])
    assert int(row["unique_positions"]) == len(visited)
    for key, value in {**counts, **sums}.items():
        close(row[key], value, f"saved replay {key}")
    if initial_error is None:
        assert row["initial_abs_q_error"] == ""
    else:
        close(row["initial_abs_q_error"], initial_error)
    return len(trace["steps"])


def efficient_rate(rows, planner):
    return sum(int(r["success"]) == 1 and int(r["steps"]) <= 2 * planner[int(r["map_seed"])] for r in rows) / len(rows)


def audit_summary(summary, rows, planner):
    assert rows
    common.audit_episode_summary(summary, rows)
    totals = {key: sum(int(r[key]) for r in rows) for key in ("steps", "moving_steps", "winnable_steps", "q_error_steps", "avoidable_failure_actions")}
    assert summary["evaluation_steps"] == totals["steps"]
    for key in ("moving_steps", "winnable_steps", "q_error_steps", "avoidable_failure_actions"):
        assert summary[key] == totals[key]
    successful = [r for r in rows if int(r["success"])]
    success_steps = sum(int(r["steps"]) for r in successful)
    success_noops = sum(int(r["noop_steps"]) for r in successful)
    assert summary["successful_episodes"] == len(successful)
    assert summary["successful_episode_steps"] == success_steps
    assert summary["successful_noop_steps"] == success_noops
    if successful:
        close(summary["successful_mean_steps"], success_steps / len(successful))
        close(summary["successful_noop_rate"], success_noops / success_steps)
    else:
        assert summary["successful_mean_steps"] is summary["successful_noop_rate"] is None
    close(summary["efficient_success_rate"], efficient_rate(rows, planner))
    assert summary["noop_steps"] == sum(int(r["noop_steps"]) for r in rows)
    assert summary["efficient_success_episodes"] == sum(int(r["success"]) == 1 and int(r["steps"]) <= 2 * planner[int(r["map_seed"])] for r in rows)
    seeds = sorted({int(r["seed"]) for r in rows})
    if "seeds" in summary:
        assert summary["seeds"] == len(seeds)
    if "seed_success_min" in summary:
        rates = [np.mean([int(r["success"]) for r in rows if int(r["seed"]) == seed]) for seed in seeds]
        close(summary["seed_success_min"], min(rates)); close(summary["seed_success_max"], max(rates))


def audit_episode_invariants(rows, expected_keys, models, task_hashes, planner):
    key = lambda r: (r["condition"], r["policy"], int(r["seed"]), r["panel"], r["mode"], int(r["map_seed"]), int(r["repetition"]))
    actual = [key(r) for r in rows]
    assert len(actual) == len(set(actual)) and set(actual) == expected_keys
    for row in rows:
        steps, success = int(row["steps"]), int(row["success"])
        assert int(row["checkpoint_complete"]) == 1 and 1 <= steps <= 32 and success in (0, 1)
        assert row["task_hash"] == task_hashes[int(row["map_seed"])]
        close(row["base_return"], success - .01 * steps)
        assert int(row["noop_steps"]) + int(row["moving_steps"]) == steps
        assert 0 <= int(row["moved_revisits"]) <= int(row["moving_steps"])
        assert 0 <= int(row["optimal_winnable_actions"]) <= int(row["winnable_steps"]) <= steps
        assert int(row["avoidable_failure_actions"]) == 1 - success
        assert 1 <= int(row["unique_positions"]) <= int(row["moving_steps"]) + 1
        assert float(row["action_regret_sum"]) >= 0 and float(row["q_abs_error_sum"]) >= 0
        if row["policy"] == "learner":
            assert int(row["checkpoint"]) == 30000 and int(row["q_error_steps"]) == steps
            assert row["parameter_hash"] == models[row["condition"], int(row["seed"])]["parameter_hash"]
        else:
            assert row["parameter_hash"] == row["policy"] and int(row["q_error_steps"]) == 0
            assert row["reference_sample_id"] == f"{row['policy']}:{row['panel']}:{row['seed']}:{row['map_seed']}:{row['repetition']}"
            if row["policy"] == "shortest_path":
                assert success == 1 and steps == planner[int(row["map_seed"])]
                assert int(row["noop_steps"]) == int(row["moved_revisits"]) == 0


def load_bare_network(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["optimizer_updates"] == 30000 and saved["observation_size"] == 92
    assert saved["purpose"] == "inference_only_not_resumable"
    assert tensor_hash(saved["online"]) == saved["parameter_hash"]
    assert tensor_hash(saved["target"]) == saved["target_parameter_hash"]
    network = torch.nn.Sequential(torch.nn.Linear(92, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64),
                                  torch.nn.ReLU(), torch.nn.Linear(64, 4))
    network.load_state_dict(saved["online"])
    network.eval().requires_grad_(False)
    assert sum(p.numel() for p in network.parameters()) == 20420
    assert not network.training and not any(p.requires_grad for p in network.parameters())
    from types import SimpleNamespace
    return saved, SimpleNamespace(online=network)


def audit_panels(study, archives, protocol, result, World, config):
    selection = read_json(study / "panels.json")
    assert selection["status"] == "complete" and not selection["selection_uses_outcomes"]
    assert selection["panels"] == result["panels"] == protocol["panels"]
    provenance = read_json(study / "provenance.json")
    assert provenance == result["provenance"]
    excluded = {}
    for name, archive in archives.items():
        record = provenance["archives"][name]
        for path, digest in record["files"].items():
            assert common.sha(archive / path) == digest, (name, path)
            if path != "manifest.json":
                assert read_json(archive / "manifest.json")["files"][path] == digest
        metadata = read_json(archive / "dataset_metadata.json")
        assert common.sha(study / f"{name}_dataset_metadata.json") == common.sha(archive / "dataset_metadata.json")
        excluded[f"previous_{name}_fresh_layout"] = {r["layout_hash"] for r in metadata["heldout"]}
        if name == "coverage":
            excluded["training_layout"] = {r["layout_hash"] for r in metadata["train"]}
        old_protocol = read_json(archive / "protocol.json")
        for source in ("q6/world.py", "q6/competence.py", "q6/supervised.py"):
            assert protocol["source_sha256"][source] == old_protocol["source_sha256"][source], ("unchanged rollout procedure", name, source)
    env = World(config)
    selected, rejected, used = [], [], set()
    task_hashes, planner = {}, {}
    spec = protocol["panel_selection"]
    assert spec["selected_before_evaluation"]
    for index in range(spec["count"]):
        panel = {"id": f"panel_{index}", "scan_start": spec["start"] + index * spec["stride"], "layouts": [], "map_seeds": []}
        candidate = panel["scan_start"]
        while len(panel["layouts"]) < spec["maps_per_panel"]:
            env.reset(seed=candidate)
            key = common.layout_hash(env)
            reason = next((name for name, hashes in excluded.items() if key in hashes), None)
            if reason is None and key in used:
                reason = "earlier_selected_layout"
            row = {"map_seed": candidate, "layout_hash": key, "original_start": list(env.position)}
            if reason:
                rejected.append({"panel": panel["id"], **row, "reason": reason})
            else:
                panel["layouts"].append(row)
                panel["map_seeds"].append(candidate)
                used.add(key)
                payload = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(),
                           "position": list(env.position), "config": asdict(config)}
                task_hashes[candidate] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                goal = tuple(map(int, np.argwhere(env.pellets)[0]))
                queue, distance = deque([goal]), {goal: 0}
                while queue:
                    p = queue.popleft()
                    for dr, dc in DELTAS:
                        q = p[0] + dr, p[1] + dc
                        if 0 <= q[0] < config.size and 0 <= q[1] < config.size and not env.walls[q] and q not in distance:
                            distance[q] = distance[p] + 1
                            queue.append(q)
                planner[candidate] = distance[env.position]
            candidate += 1
        selected.append(panel)
    assert selected == selection["panels"] and rejected == selection["collision_skips"]
    assert len(used) == selection["unique_selected_layouts"] == spec["count"] * spec["maps_per_panel"]
    assert selection["excluded_layout_counts"] == {name: len(hashes) for name, hashes in excluded.items()}
    return selection, task_hashes, planner


def audit_models(study, archives, protocol, result):
    records = read_json(study / "models.json")
    assert records == result["provenance"]["models"]
    expected = {(c, s) for c in CONDITIONS for s in protocol["seeds"]}
    assert {(r["condition"], r["seed"]) for r in records} == expected and len(records) == len(expected)
    states, agents = {}, {}
    for row in records:
        condition, seed = row["condition"], row["seed"]
        archive = "coverage" if condition == "exhaustive" else "equal_support"
        assert row["archive"] == archive and row["source_optimizer_updates"] == 30000
        path = archives[archive] / "models" / f"{condition}_seed{seed}_update30000.pt"
        saved = study / "models" / f"{condition}_seed{seed}.pt"
        digest = common.sha(path)
        assert read_json(archives[archive] / "manifest.json")["files"][path.relative_to(archives[archive]).as_posix()] == digest
        assert common.sha(saved) == row["source_sha256"] == row["saved_sha256"] == row["saved_sha256_after"] == digest
        state, agent = load_bare_network(saved)
        assert row["online_before"] == row["online_after"] == state["parameter_hash"]
        assert row["target_before"] == row["target_after"] == state["target_parameter_hash"]
        assert row["unchanged"] and not row["requires_grad"] and not row["requires_grad_after"] and row["model_parameter_count"] == 20420
        assert row["saved"] == f"models/{condition}_seed{seed}.pt"
        assert row["source_sha256_after"] == digest
        assert [check["panel"] for check in row["panel_checks"]] == [p["id"] for p in result["panels"]]
        for check in row["panel_checks"]:
            assert check["unchanged"]
            assert check["online_before"] == state["parameter_hash"] and check["target_before"] == state["target_parameter_hash"]
            assert check["saved_sha256_before"] == check["source_sha256_before"] == digest
            assert check["online_hash"] == state["parameter_hash"] and check["target_hash"] == state["target_parameter_hash"]
            assert check["saved_sha256"] == check["source_sha256"] == digest
        states[condition, seed], agents[condition, seed] = state, agent
    assert result["provenance"]["all_loaded_models_unchanged"] and result["run"]["all_models_unchanged"]
    return states, agents


def audit_tables(study, protocol, result, selection, states, task_hashes, planner, *, smoke):
    rows, refs = csv_rows(study / "evaluations.csv"), csv_rows(study / "references.csv")
    panels, seeds = selection["panels"], protocol["seeds"]
    expected = {(c, "learner", s, p["id"], mode, m, rep) for c in CONDITIONS for s in seeds for p in panels
                for mode in ("greedy", "epsilon_0_1") for m in p["map_seeds"] for rep in range(1 if mode == "greedy" else 2)}
    audit_episode_invariants(rows, expected, states, task_hashes, planner)
    expected_refs = {("shared", policy, s, p["id"], "reference", m, rep) for policy in ("random_actions", "shortest_path")
                     for s in (seeds if policy == "random_actions" else [0]) for p in panels for m in p["map_seeds"]
                     for rep in range(2 if policy == "random_actions" else 1)}
    audit_episode_invariants(refs, expected_refs, states, task_hashes, planner)
    assert len({r["reference_sample_id"] for r in refs}) == len(refs)
    panel_ids = [p["id"] for p in panels] + ["all"]
    summary_keys = {(c, p, mode) for c in CONDITIONS for p in panel_ids for mode in ("greedy", "epsilon_0_1")}
    assert {(r["condition"], r["panel"], r["mode"]) for r in result["aggregate"]} == summary_keys
    assert len(result["aggregate"]) == len(summary_keys)
    assert {(r["condition"], r["panel"], r["mode"], r["seed"]) for r in result["seed_results"]} == {
        (*key, s) for key in summary_keys for s in seeds}
    assert len(result["seed_results"]) == len(summary_keys) * len(seeds)
    for summary in result["aggregate"] + result["seed_results"]:
        group = [r for r in rows if r["condition"] == summary["condition"] and r["mode"] == summary["mode"]
                 and (summary["panel"] == "all" or r["panel"] == summary["panel"])
                 and ("seed" not in summary or int(r["seed"]) == summary["seed"])]
        audit_summary(summary, group, planner)
    assert {(r["policy"], r["panel"]) for r in result["references"]} == {(p, q) for p in ("random_actions", "shortest_path") for q in panel_ids}
    assert len(result["references"]) == 2 * len(panel_ids)
    for summary in result["references"]:
        group = [r for r in refs if r["policy"] == summary["policy"] and (summary["panel"] == "all" or r["panel"] == summary["panel"])]
        audit_summary(summary, group, planner)
    return rows, refs


COMPARISONS = (("uniform_minus_collected", "collected_unique", "uniform_subset"),
               ("exhaustive_minus_collected", "collected_unique", "exhaustive"),
               ("exhaustive_minus_uniform", "uniform_subset", "exhaustive"))
DELTA_METRICS = ("success_rate", "efficient_success_rate", "mean_steps", "noop_rate")


def audit_pairs(study, protocol, result, rows, planner):
    paired = read_json(study / "paired_differences.json")
    assert paired == result["paired_differences"]
    assert paired["comparisons"] == [{"id": c, "left": l, "right": r} for c, l, r in COMPARISONS]
    raw = {(r["condition"], r["panel"], int(r["seed"]), int(r["map_seed"]), r["mode"], int(r["repetition"])): r for r in rows}
    comparison_lookup = {c: (l, r) for c, l, r in COMPARISONS}
    # Pairing is declared for greedy only; secondary epsilon outcomes are
    # retained separately in rollout summaries.
    modes = ("greedy",)
    expected = {(c, p["id"], s, m, mode, rep) for c, _, _ in COMPARISONS for p in result["panels"] for s in protocol["seeds"]
                for m in p["map_seeds"] for mode in modes for rep in range(1)}
    pair_key = lambda r: (r["comparison"], r["panel"], int(r["seed"]), int(r["map_seed"]), r["mode"], int(r["repetition"]))
    assert {pair_key(r) for r in paired["per_layout"]} == expected and len(paired["per_layout"]) == len(expected)
    csv_pairs = csv_rows(study / "paired_layouts.csv")
    assert len(csv_pairs) == len(paired["per_layout"])
    for item, csv_item in zip(paired["per_layout"], csv_pairs):
        assert set(item) == set(csv_item)
        for key, value in item.items():
            if isinstance(value, str):
                assert csv_item[key] == value
            else:
                close(csv_item[key], value, key)
        left_c, right_c = comparison_lookup[item["comparison"]]
        key = (item["panel"], item["seed"], item["map_seed"], item["mode"], item["repetition"])
        left, right = raw[left_c, *key], raw[right_c, *key]
        close(item["success_delta"], int(right["success"]) - int(left["success"]))
        close(item["steps_delta"], int(right["steps"]) - int(left["steps"]))
        close(item["efficient_success_delta"], efficient_rate([right], planner) - efficient_rate([left], planner))
        close(item["noop_rate_delta"], int(right["noop_steps"]) / int(right["steps"]) - int(left["noop_steps"]) / int(left["steps"]))
    lookup = {(r["condition"], r["panel"], r["seed"], r["mode"]): r for r in result["seed_results"]}
    panel_ids = [p["id"] for p in result["panels"]] + ["all"]
    expected_seed = {(c, p, s, mode) for c, _, _ in COMPARISONS for p in panel_ids for s in protocol["seeds"] for mode in modes}
    assert {(r["comparison"], r["panel"], r["seed"], r["mode"]) for r in paired["per_seed"]} == expected_seed
    assert len(paired["per_seed"]) == len(expected_seed)
    for row in paired["per_seed"]:
        left, right = (lookup[c, row["panel"], row["seed"], row["mode"]] for c in comparison_lookup[row["comparison"]])
        for metric in DELTA_METRICS:
            close(row[metric + "_delta"], right[metric] - left[metric])
    expected_aggregate = {(c, p, mode) for c, _, _ in COMPARISONS for p in panel_ids for mode in modes}
    assert {(r["comparison"], r["panel"], r["mode"]) for r in paired["aggregate"]} == expected_aggregate
    assert len(paired["aggregate"]) == len(expected_aggregate)
    for row in paired["aggregate"]:
        group = [r for r in paired["per_seed"] if (r["comparison"], r["panel"], r["mode"]) == (row["comparison"], row["panel"], row["mode"])]
        assert row["seeds"] == len(protocol["seeds"]) == len(group)
        for metric in DELTA_METRICS:
            close(row["mean_seed_" + metric + "_delta"], np.mean([r[metric + "_delta"] for r in group]))
        pooled = {r["condition"]: r for r in result["aggregate"] if r["panel"] == row["panel"] and r["mode"] == row["mode"]}
        left, right = comparison_lookup[row["comparison"]]
        close(row["pooled_noop_rate_delta"], pooled[right]["noop_rate"] - pooled[left]["noop_rate"])
    return paired


def audit_descriptive(protocol, result, paired, *, smoke):
    thresholds, robustness = result["descriptive_thresholds"], result["robustness"]
    assert "not new competence gates" in thresholds["classification"]
    assert thresholds["eligible"] == robustness["eligible"] == (not smoke)
    assert "gates" not in result, "this evaluation must not introduce competence gates"
    close(thresholds["success_reference"], .7); close(thresholds["efficient_success_reference"], .8)
    panels, seeds = result["panels"], protocol["seeds"]
    expected = {(c, p["id"], s) for c in CONDITIONS for p in panels for s in seeds}
    actual = {(r["condition"], r["panel"], r["seed"]) for r in thresholds["per_seed"]}
    assert actual == expected and len(thresholds["per_seed"]) == len(expected)
    lookup = {(r["condition"], r["panel"], r["seed"]): r for r in result["seed_results"] if r["mode"] == "greedy"}
    random = {r["panel"]: r["success_rate"] for r in result["references"] if r["policy"] == "random_actions"}
    for row in thresholds["per_seed"]:
        raw = lookup[row["condition"], row["panel"], row["seed"]]
        close(row["success_rate"], raw["success_rate"]); close(row["efficient_success_rate"], raw["efficient_success_rate"])
        close(row["panel_random_success"], random[row["panel"]])
        assert row["success_reference_met"] == bool(not smoke and raw["success_rate"] >= .7 and raw["success_rate"] > random[row["panel"]])
        assert row["efficiency_reference_met"] == bool(not smoke and raw["efficient_success_rate"] >= .8)
    assert {r["condition"] for r in thresholds["per_condition"]} == set(CONDITIONS) and len(thresholds["per_condition"]) == 3
    for row in thresholds["per_condition"]:
        assert row["panels"] == len(panels)
        assert {r["panel"] for r in row["per_panel"]} == {p["id"] for p in panels} and len(row["per_panel"]) == len(panels)
        for outcome in row["per_panel"]:
            group = [r for r in thresholds["per_seed"] if (r["condition"], r["panel"]) == (row["condition"], outcome["panel"])]
            assert outcome["complete_seeds"] == len(seeds) == len(group)
            assert outcome["all_seed_success_reference_met"] == all(r["success_reference_met"] for r in group)
            assert outcome["all_seed_efficiency_reference_met"] == all(r["efficiency_reference_met"] for r in group)
        assert row["success_reference_panels"] == sum(r["all_seed_success_reference_met"] for r in row["per_panel"])
        assert row["efficiency_reference_panels"] == sum(r["all_seed_efficiency_reference_met"] for r in row["per_panel"])
    assert robustness["primary_comparison"] == "uniform_minus_collected"
    assert robustness["primary_metric"] == "efficient_success_rate_delta" and robustness["primary_mode"] == "greedy"
    assert {(r["comparison"], r["mode"]) for r in robustness["per_comparison"]} == {(c, "greedy") for c, _, _ in COMPARISONS}
    assert len(robustness["per_comparison"]) == 3
    for row in robustness["per_comparison"]:
        group = [r for r in paired["aggregate"] if r["comparison"] == row["comparison"] and r["mode"] == row["mode"] and r["panel"] != "all"]
        assert row["panels"] == len(panels) == len(group)
        assert set(row["metrics"]) == {m + "_delta" for m in DELTA_METRICS}
        for metric in DELTA_METRICS:
            values = [r["mean_seed_" + metric + "_delta"] for r in group]
            record = row["metrics"][metric + "_delta"]
            close(record["minimum"], min(values)); close(record["maximum"], max(values)); close(record["mean"], np.mean(values))
            assert record["panels"] == len(panels) and record["sign_zero_tolerance"] == 1e-12
            assert record["positive_panels"] == sum(v > 1e-12 for v in values)
            assert record["negative_panels"] == sum(v < -1e-12 for v in values)
            assert record["zero_panels"] == sum(abs(v) <= 1e-12 for v in values)
    assert {(r["condition"], r["mode"]) for r in robustness["per_condition"]} == {(c, m) for c in CONDITIONS for m in ("greedy", "epsilon_0_1")}
    assert len(robustness["per_condition"]) == 6
    for row in robustness["per_condition"]:
        group = [r for r in result["aggregate"] if r["condition"] == row["condition"] and r["mode"] == row["mode"] and r["panel"] != "all"]
        assert row["panels"] == len(panels) == len(group) and set(row["metrics"]) == set(DELTA_METRICS)
        for metric in DELTA_METRICS:
            close(row["metrics"][metric]["minimum"], min(r[metric] for r in group))
            close(row["metrics"][metric]["maximum"], max(r[metric] for r in group))


def audit_replays(study, protocol, result, rows, refs, agents, World, config, *, skip_forward):
    trajectories = read_json(study / "trajectories.json")
    assert trajectories == result["trajectories"]
    key = lambda r: (r["condition"], r["policy"], int(r["seed"]), r["panel"], r["mode"], int(r["map_seed"]), int(r.get("repetition", 0)))
    expected = {(c, "learner", s, p["id"], m, p["map_seeds"][0], 0) for c in CONDITIONS for s in protocol["seeds"]
                for p in result["panels"] for m in ("greedy", "epsilon_0_1")}
    expected |= {("shared", policy, 0 if policy == "shortest_path" else protocol["seeds"][0], p["id"], "reference", p["map_seeds"][0], 0)
                 for policy in ("random_actions", "shortest_path") for p in result["panels"]}
    actual = [key(r) for r in trajectories]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    lookup = {key(r): r for r in rows + refs}
    recorded_steps = 0
    for trace in trajectories:
        agent = agents[trace["condition"], trace["seed"]] if trace["policy"] == "learner" else None
        recorded_steps += check_saved_replay(trace, lookup[key(trace)], World(config), agent, skip_forward=skip_forward)
    return len(trajectories), recorded_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/panel_evaluation/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true")
    for name in ("supervised", "fixed-targets", "coverage", "equal-support"):
        parser.add_argument("--" + name + "-study", type=Path)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Assertions must be enabled; do not use python -O.")
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p.resolve() if p.is_absolute() else (root / p).resolve()
    study = resolve(args.study)
    archives = {name: resolve(getattr(args, name + "_study") or Path(f"experiments/{name}/pilot_v1"))
                for name in ("supervised", "fixed_targets", "coverage", "equal_support")}
    sys.path.insert(0, str(study / "source"))
    from q6.world import CollectionWorld, WorldConfig
    torch.set_num_threads(1)
    started = time.monotonic()
    result, protocol = read_json(study / "results.json"), read_json(study / "protocol.json")
    smoke = bool(protocol["smoke"] or protocol["deviations"])
    assert not smoke or args.allow_smoke, "Smoke/deviating artifacts require explicit --allow-smoke."
    assert protocol["id"] == "panel-evaluation-v1" and result["run"]["status"] == "complete"
    assert [r["id"] for r in protocol["conditions"]] == list(CONDITIONS)
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2] and protocol["git"]["dirty"] is False
        assert protocol["runtime"]["python"].startswith("3.12.") and protocol["runtime"]["torch"].split("+")[0] == "2.8.0"
        assert protocol["runtime"]["numpy"] == "2.0.2"
        assert protocol["panel_selection"] == {"count": 8, "maps_per_panel": 64, "start": 970000, "stride": 1000, "selected_before_evaluation": True}
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig())
    assert protocol["evaluation"]["greedy_repetitions"] == 1 and protocol["evaluation"]["epsilon_0_1_repetitions"] == 2
    assert protocol["evaluation"]["optimal_q_agreement_atol"] == 1e-6 and protocol["evaluation"]["optimal_q_agreement_rtol"] == 0
    report = {"study": str(args.study), "smoke_validation_only": smoke, "manifest_files": common.audit_manifest(study, protocol, result)}
    selection, task_hashes, planner = audit_panels(study, archives, protocol, result, CollectionWorld, config)
    states, agents = audit_models(study, archives, protocol, result)
    rows, refs = audit_tables(study, protocol, result, selection, states, task_hashes, planner, smoke=smoke)
    paired = audit_pairs(study, protocol, result, rows, planner)
    audit_descriptive(protocol, result, paired, smoke=smoke)
    replay_count, replay_steps = audit_replays(study, protocol, result, rows, refs, agents, CollectionWorld, config, skip_forward=args.skip_forward_inference)
    run = result["run"]
    for group in (protocol, run, result["provenance"]):
        assert all(group[k] == 0 for k in ("new_training_updates", "new_collection_steps", "new_support_draws"))
    assert run["frozen_models"] == len(states) == 3 * len(protocol["seeds"]) and run["model_parameter_count"] == 20420
    assert run["learner_episodes"] == run["completed_learner_episodes"] == len(rows) == protocol["budget"]["maximum_learner_episodes"]
    assert run["reference_episodes"] == run["unique_reference_episodes"] == len(refs) == protocol["budget"]["maximum_reference_episodes"]
    assert run["panels"] == len(selection["panels"]) and run["unique_layouts"] == len(task_hashes)
    assert run["wall_seconds"] <= protocol["budget"]["admission_seconds"]
    assert run["peak_rss_bytes"] <= protocol["budget"]["peak_process_rss_bytes"]
    assert run["resource_checks"] > len(rows) + len(refs)
    assert run["interpretation"] == ("smoke_or_deviation_descriptive_only" if smoke else "frozen_panel_robustness_descriptive_only")
    assert run["stop_reason"] is None
    expected_progress = {(c, s, p["id"]) for c in CONDITIONS for s in protocol["seeds"] for p in selection["panels"]}
    assert len(run["progress"]) == len(expected_progress) and all(r["evaluation_complete"] for r in run["progress"])
    assert {(r["condition"], r["seed"], r["panel"]) for r in run["progress"]} == expected_progress
    cells = result["provenance"]["expected_cells"]
    assert cells["complete"]
    assert cells["expected_learner_cells"] == cells["complete_learner_cells"] == cells["unique_complete_learner_cells"] == len(rows)
    assert cells["expected_reference_cells"] == cells["complete_reference_cells"] == cells["unique_complete_reference_cells"] == len(refs)
    report.update(selected_layouts=len(task_hashes), rejected_candidates=len(selection["collision_skips"]),
        frozen_models_checked=len(states), model_panel_checks=len(expected_progress), model_identity_unchanged=True,
        learner_episode_rows=len(rows), reference_episode_rows=len(refs), paired_layout_rows=len(paired["per_layout"]),
        paired_seed_rows=len(paired["per_seed"]), paired_aggregate_rows=len(paired["aggregate"]),
        recorded_replays_checked=replay_count, recorded_steps_checked=replay_steps,
        replay_forward_inference_checked=not args.skip_forward_inference, forward_inference_checked=not args.skip_forward_inference,
        full_policy_evaluation_repeated=False, new_training_updates=0, new_collection_steps=0,
        descriptive_thresholds=result["descriptive_thresholds"]["per_condition"], audit_seconds=time.monotonic() - started)
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
