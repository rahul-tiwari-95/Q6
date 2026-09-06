"""Independently validate an archived fixed-coverage comparison, without training.

The original random collection is reconstructed from its declared RNG and the
independently checked transition table. This is collection-trace validation,
not a new learned-policy evaluation. Model forward inference can be omitted
for dependency-range CI; saved-prediction/metric checks remain enabled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
import audit_fixed_targets_study as common
from audit_fixed_targets_study import ADDITIVE, close, csv_rows, read_json

CONDITIONS = ("exhaustive", "collected_unique")

PANELS = ("train", "heldout")
BUCKETS = common.BUCKETS


def reconstruct_collection(data, metadata, transitions, episodes_per_map):
    """Rebuild the original random traces with no newly collected information."""
    n = len(data["train_observations"])
    counts = np.zeros(n, np.uint32)
    traces, episodes = [], []
    next_rows, rewards, ends = (transitions[key] for key in ("successor_indices", "rewards", "ends"))
    for layout in metadata["train"]:
        map_seed = layout["map_seed"]
        initial = np.flatnonzero((data["train_map_seeds"] == map_seed) &
            np.all(data["train_positions"] == layout["original_start"], axis=1) & (data["train_remaining"] == 32))
        assert len(initial) == 1
        for repetition in range(episodes_per_map):
            rng = np.random.default_rng(np.random.SeedSequence([map_seed, repetition, 77301]))
            row, step, shaped_return, noops = int(initial[0]), 0, 0.0, 0
            while True:
                action = int(rng.integers(0, 4))
                following = int(next_rows[row, action])
                ended = bool(ends[row, action])
                reward = float(rewards[row, action])
                # Identify collection independently from the terminal reward.
                layers = data["train_observations"][row, :75].reshape(3, 5, 5)
                r, c = map(int, data["train_positions"][row])
                mapping = data["train_observations"][row, -16:].reshape(4, 4).argmax(1)
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[mapping[action]]
                rr, cc = r + dr, c + dc
                if not (0 <= rr < 5 and 0 <= cc < 5) or layers[0, rr, cc]:
                    rr, cc = r, c
                success = bool(layers[1, rr, cc])
                counts[row] += 1
                traces.append({"map_seed": map_seed, "repetition": repetition, "step": step + 1,
                    "current_row": row, "action": action, "next_row": following, "remaining": 32 - step,
                    "reward": reward, "terminated": int(success), "truncated": int(ended and not success)})
                shaped_return += reward
                noops += (rr, cc) == (r, c)
                step += 1
                assert ended == (success or step == 32)
                if ended:
                    episodes.append({"map_seed": map_seed, "repetition": repetition, "steps": step,
                        "success": int(success), "shaped_return": shaped_return, "base_return": int(success) - .01 * step,
                        "terminated": int(success), "truncated": int(not success), "complete": 1, "noop_steps": noops})
                    break
                assert following >= 0
                row = following
    support = np.flatnonzero(counts).astype(np.int64)
    assert sum(row["steps"] for row in episodes) == int(counts.sum()) == len(traces)
    assert len(traces) <= len(metadata["train"]) * episodes_per_map * 32
    return counts, support, traces, episodes


def coverage_counts(data, support, transitions):
    mask = np.zeros(len(data["train_observations"]), bool)
    mask[support] = True
    distance = np.empty(len(mask), np.int16)
    for map_seed in np.unique(data["train_map_seeds"]):
        map_rows = np.flatnonzero(data["train_map_seeds"] == map_seed)
        for position in np.unique(data["train_positions"][map_rows], axis=0):
            rows = map_rows[np.all(data["train_positions"][map_rows] == position, axis=1)]
            nearest = data["train_remaining"][rows[data["train_winnable"][rows]]].min()
            distance[rows] = nearest
    successors = transitions["successor_indices"][support]
    live = ~transitions["ends"][support]
    live_destinations = successors[live]
    outside = live_destinations[~mask[live_destinations]]
    return {"support_states": len(support), "full_states": len(mask),
        "winnable_states": int(data["train_winnable"][support].sum()),
        "impossible_states": int((~data["train_winnable"][support]).sum()),
        "near_goal_distance1_states": int((distance[support] == 1).sum()),
        "near_goal_distance_le2_states": int((distance[support] <= 2).sum()),
        "nonterminal_successor_edges": len(live_destinations), "outside_support_successor_edges": len(outside),
        "unique_outside_support_successors": len(np.unique(outside)),
        "terminal_edges": int((~live).sum()),
        "by_remaining": {str(t): int((data["train_remaining"][support] == t).sum()) for t in range(1, 33)},
        "by_map": {str(seed): int((data["train_map_seeds"][support] == seed).sum()) for seed in np.unique(data["train_map_seeds"])}}


def audit_model_predictions(study, data, protocol, conditions, result, DQN, *, skip_forward):
    predictions = {}
    final = max(protocol["checkpoints"])
    count = 0
    for condition in conditions:
        for seed in protocol["seeds"]:
            for checkpoint in protocol["checkpoints"]:
                path = study / "models" / f"{condition}_seed{seed}_update{checkpoint}.pt"
                state = torch.load(path, map_location="cpu", weights_only=True)
                agent = DQN(state["observation_size"], seed=seed)
                agent.online.load_state_dict(state["online"])
                assert agent.parameter_hash() == state["parameter_hash"]
                target_digest = hashlib.sha256()
                for tensor in state["target"].values():
                    assert torch.isfinite(tensor).all()
                    target_digest.update(tensor.numpy().tobytes())
                assert target_digest.hexdigest() == state["target_parameter_hash"]
                assert state["optimizer_updates"] == checkpoint and state["purpose"] == "inference_only_not_resumable"
                if checkpoint == 0:
                    assert agent.parameter_hash() == DQN(92, seed=seed).parameter_hash()
                    assert agent.parameter_hash() == result["run"]["initial_policy_hashes"][condition][str(seed)]
                    assert all(torch.equal(state["online"][key], state["target"][key]) for key in state["online"])
                if checkpoint == final:
                    predictions[condition, seed] = dict(np.load(study / f"predictions_{condition}_seed{seed}.npz", allow_pickle=False))
                    for panel in PANELS:
                        observation = data[f"{panel}_observations"]
                        assert predictions[condition, seed][panel].shape == (len(observation), 4)
                        assert np.isfinite(predictions[condition, seed][panel]).all()
                        if not skip_forward:
                            actual = np.empty((len(observation), 4), np.float32)
                            with torch.no_grad():
                                for start in range(0, len(observation), 4096):
                                    actual[start:start + 4096] = agent.online(torch.from_numpy(observation[start:start + 4096])).numpy()
                            assert np.array_equal(actual, predictions[condition, seed][panel]), (condition, seed, panel)
                count += 1
    return predictions, count



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
        common.audit_episode_summary(summary, selected)
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
        common.audit_episode_summary(summary, selected)
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
        "both_fresh_gates_met" if all(g["fresh"] for g in gates.values()) else "exhaustive_only_fresh_gate_met" if gates["exhaustive"]["fresh"] else
        "collected_only_fresh_gate_met" if gates["collected_unique"]["fresh"] else "neither_fresh_gate_met")
    assert result["run"]["interpretation"] == expected_interpretation
    paired = read_json(study / "paired_differences.json")
    assert paired == result["paired_differences"]
    assert paired["direction"] == "collected_unique minus exhaustive"
    assert len(paired["per_layout"]) == len(seeds) * sum(len(metadata[p]) for p in PANELS)
    assert len(paired["per_seed"]) == len(seeds) * 2
    for row in paired["per_layout"]:
        left, right = (final_episode_lookup[c, row["seed"], row["panel"], row["map_seed"]] for c in CONDITIONS)
        for metric in ("success", "steps", "noop_steps"):
            close(row[f"{metric}_delta"], int(right[metric]) - int(left[metric]))
        close(row["noop_rate_delta"], int(right["noop_steps"]) / int(right["steps"]) - int(left["noop_steps"]) / int(left["steps"]))
        close(row["efficient_success_delta"], efficient([right]) - efficient([left]))
    for row in paired["per_seed"]:
        pair = {c: [final_episode_lookup[c, row["seed"], row["panel"], layout["map_seed"]] for layout in metadata[row["panel"]]] for c in CONDITIONS}
        values = {c: {"success_rate": np.mean([int(r["success"]) for r in rows]), "mean_steps": np.mean([int(r["steps"]) for r in rows]),
            "noop_rate": sum(int(r["noop_steps"]) for r in rows) / sum(int(r["steps"]) for r in rows), "efficient_success_rate": efficient(rows)} for c, rows in pair.items()}
        for metric in values["exhaustive"]:
            close(row[f"{metric}_delta"], values["collected_unique"][metric] - values["exhaustive"][metric])
    for row in paired["aggregate"]:
        selected = [r for r in paired["per_seed"] if r["panel"] == row["panel"]]
        assert row["seeds"] == len(seeds)
        for metric in ("success_rate", "mean_steps", "noop_rate", "efficient_success_rate"):
            close(row[f"mean_seed_{metric}_delta"], np.mean([r[f"{metric}_delta"] for r in selected]))
    return {"evaluation_rows": len(episodes), "reference_rows": len(references), "state_metric_rows": len(states),
            "loss_rows": len(losses), "paired_layout_rows": len(paired["per_layout"]), "gates": result["gates"]}


def audit_collection(study, data, metadata, transitions, protocol, result):
    coverage = read_json(study / "coverage.json")
    assert coverage == result["coverage"] and coverage["status"] == "complete"
    arrays = dict(np.load(study / "collection.npz", allow_pickle=False))
    for key, spec in coverage["arrays"].items():
        assert list(arrays[key].shape) == spec["shape"] and str(arrays[key].dtype) == spec["dtype"]
        assert common.array_sha(arrays[key]) == spec["sha256"]
    counts, support, traces, episodes = reconstruct_collection(data, metadata, transitions, protocol["collection"]["episodes_per_map"])
    assert np.array_equal(counts, arrays["visited_counts"])
    assert np.array_equal(support, arrays["support_indices"])
    assert len(support) >= 64
    actual_steps = csv_rows(study / "collection_steps.csv")
    actual_episodes = csv_rows(study / "collection_episodes.csv")
    for expected_rows, actual_rows in ((traces, actual_steps), (episodes, actual_episodes)):
        assert len(expected_rows) == len(actual_rows)
        for expected, actual in zip(expected_rows, actual_rows):
            assert set(expected) == set(actual)
            for key, value in expected.items():
                close(actual[key], value, f"collection {key}")
    assert coverage["collection_episodes"] == coverage["complete_collection_episodes"] == len(episodes)
    assert coverage["collection_steps"] == len(traces)
    assert coverage["collection_successes"] == sum(r["success"] for r in episodes)
    assert coverage["collection_noop_steps"] == sum(r["noop_steps"] for r in episodes)
    assert len(traces) <= protocol["budget"]["maximum_collection_steps"]
    diagnostics = coverage_counts(data, support, transitions)
    assert coverage["total_training_states"] == diagnostics["full_states"]
    assert coverage["unique_current_states"] == diagnostics["support_states"]
    mask = counts > 0
    # Derive distance from time-ordered reachability labels, independently checked
    # against BFS by the transition audit; it never chooses collection actions.
    near = np.zeros(len(counts), bool)
    for seed in np.unique(data["train_map_seeds"]):
        rows = np.flatnonzero(data["train_map_seeds"] == seed)
        for position in np.unique(data["train_positions"][rows], axis=0):
            local = rows[np.all(data["train_positions"][rows] == position, axis=1)]
            if data["train_remaining"][local[data["train_winnable"][local]]].min() <= 2:
                near[local] = True
    def check_fraction(record, selected):
        winnable = selected & data["train_winnable"]
        nearby = selected & near
        expected = {"states": int(selected.sum()), "visited_states": int((selected & mask).sum()),
            "winnable_states": int(winnable.sum()), "visited_winnable_states": int((winnable & mask).sum()),
            "goal_near_states": int(nearby.sum()), "visited_goal_near_states": int((nearby & mask).sum()), "visits": int(counts[selected].sum())}
        for key, value in expected.items():
            assert record[key] == value, ("coverage denominator", key)
        for selected_group, metric in ((selected, "coverage_rate"), (winnable, "winnable_coverage_rate"), (nearby, "goal_near_coverage_rate")):
            if selected_group.any():
                close(record[metric], mask[selected_group].mean(), metric)
            else:
                assert record[metric] is None
    check_fraction(coverage["overall"], np.ones(len(counts), bool))
    for row in coverage["by_map"]:
        check_fraction(row, data["train_map_seeds"] == row["map_seed"])
    assert {r["map_seed"] for r in coverage["by_map"]} == {r["map_seed"] for r in metadata["train"]}
    for row in coverage["by_time_bucket"]:
        low, high = (1, 32) if row["time_bucket"] == "all" else map(int, row["time_bucket"].split("-"))
        check_fraction(row, (data["train_remaining"] >= low) & (data["train_remaining"] <= high))
    assert {r["time_bucket"] for r in coverage["by_time_bucket"]} == set(BUCKETS)
    for top, detailed in (("current_state_fraction", "coverage_rate"), ("winnable_current_state_fraction", "winnable_coverage_rate"),
                          ("goal_near_current_state_fraction", "goal_near_coverage_rate")):
        close(coverage[top], coverage["overall"][detailed])
    access = coverage["successor_queries"]
    live = transitions["successor_indices"][support][~transitions["ends"][support]]
    outside = live[~mask[live]]
    for key, expected in {"all_action_transitions": len(support) * 4, "nonterminal_transitions": len(live),
        "outside_support_nonterminal_transitions": len(outside), "unique_nonterminal_destinations": len(np.unique(live)),
        "unique_outside_support_destinations": len(np.unique(outside))}.items():
        assert access[key] == expected
    close(access["outside_support_fraction"], len(outside) / len(live))
    return support, {"original_collection_episodes_reconstructed": len(episodes), "original_collection_steps_reconstructed": len(traces),
                     "collected_unique_states": len(support), "coverage_fraction": len(support) / len(counts),
                     "outside_support_successor_edges": len(outside), "unique_outside_support_successors": len(np.unique(outside))}


def audit_inputs(study, prior, earlier, data, transitions, metadata, protocol, result, World, config, *, smoke):
    prior_meta, earlier_meta = read_json(prior / "dataset_metadata.json"), read_json(earlier / "dataset_metadata.json")
    provenance = read_json(study / "provenance.json")
    assert provenance == result["provenance"]
    assert metadata["status"] == "complete"
    for name, spec in metadata["arrays"].items():
        assert list(data[name].shape) == spec["shape"] and str(data[name].dtype) == spec["dtype"]
        assert common.array_sha(data[name]) == spec["sha256"]
    for name, digest in provenance["files"].items():
        assert common.sha(prior / name) == digest
    for name in ("dataset_metadata.json", "sampling.json", "protocol.json"):
        assert common.sha(study / f"prior_{name}") == common.sha(prior / name)
    assert common.sha(earlier / "dataset_metadata.json") == provenance["earlier_supervised_metadata_sha256"]
    assert common.sha(study / "earlier_supervised_dataset_metadata.json") == common.sha(earlier / "dataset_metadata.json")
    old_fresh = {r["layout_hash"] for r in prior_meta["heldout"]}
    earlier_fresh = {r["layout_hash"] for r in earlier_meta["heldout"]}
    train = {r["layout_hash"] for r in metadata["train"]}
    fresh = {r["layout_hash"] for r in metadata["heldout"]}
    assert not fresh & (train | old_fresh | earlier_fresh)
    assert len(fresh) == len(metadata["heldout"])
    assert metadata["train_duplicate_layouts"] == {k: v for k, v in Counter(r["layout_hash"] for r in metadata["train"]).items() if v > 1}
    observed = {p: {hashlib.sha256(row.tobytes()).digest() for row in data[f"{p}_observations"]} for p in PANELS}
    assert not observed["train"] & observed["heldout"]
    assert set(metadata["split_check"].values()) == {0}
    env = World(config)
    for panel in PANELS:
        for row in metadata[panel]:
            env.reset(seed=row["map_seed"])
            assert common.layout_hash(env) == row["layout_hash"] and list(env.position) == row["original_start"]
    if not smoke:
        assert [r["map_seed"] for r in metadata["train"]] == list(range(300000, 300256))
        assert len(metadata["heldout"]) == 64
        candidate, accepted, rejected, used = 950000, [], [], set()
        while len(accepted) < 64:
            env.reset(seed=candidate)
            key = common.layout_hash(env)
            reason = ("training_layout" if key in train else "previous_fixed_targets_fresh_layout" if key in old_fresh else
                      "previous_supervised_fresh_layout" if key in earlier_fresh else "earlier_fresh_layout" if key in used else None)
            row = {"map_seed": candidate, "layout_hash": key, "original_start": list(env.position)}
            if reason:
                rejected.append({**row, "reason": reason})
            else:
                accepted.append(row)
                used.add(key)
            candidate += 1
        assert accepted == metadata["heldout"] and rejected == metadata["collision_skips"]
        with np.load(prior / "dataset.npz", allow_pickle=False) as old:
            for name, array in data.items():
                if name.startswith("train_"):
                    assert np.array_equal(array, old[name]), ("historical data identity", name)
        with np.load(prior / "transitions.npz", allow_pickle=False) as old:
            assert set(transitions) == set(old.files)
            for key, array in transitions.items():
                assert np.array_equal(array, old[key]), ("historical transition identity", key)
        assert provenance["training_arrays_identical"] and provenance["transition_arrays_identical"] and provenance["replication_required"]
        old_protocol = read_json(prior / "protocol.json")
        for key in ("q6/world.py", "q6/learning.py", "q6/fixed_targets.py"):
            assert protocol["source_sha256"][key] == old_protocol["source_sha256"][key]
    return provenance


def audit_sampling(study, prior, data, transitions, support, protocol, result, provenance, *, smoke):
    sampling = read_json(study / "sampling.json")
    assert sampling == result["sampling"]
    global_arrays = dict(np.load(study / "sample_counts.npz", allow_pickle=False))
    local_arrays = dict(np.load(study / "local_sample_counts.npz", allow_pickle=False))
    n, final = len(data["train_observations"]), max(protocol["checkpoints"])
    supports = {"exhaustive": np.arange(n, dtype=np.int64), "collected_unique": support}
    old_sampling = read_json(prior / "sampling.json")
    visited_mask = np.zeros(n, bool)
    visited_mask[support] = True
    live = ~transitions["ends"]
    outside = live & ~visited_mask[np.maximum(transitions["successor_indices"], 0)]
    for condition in CONDITIONS:
        indices = supports[condition]
        for seed in protocol["seeds"]:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
            local_counts, global_counts = np.zeros(len(indices), np.uint32), np.zeros(n, np.uint32)
            local_digest, global_digest = hashlib.sha256(), hashlib.sha256()
            for _ in range(final):
                local = rng.choice(len(indices), 64, replace=False)
                actual = indices[local]
                local_counts[local] += 1
                global_counts[actual] += 1
                local_digest.update(local.astype("<i8").tobytes())
                global_digest.update(actual.astype("<i8").tobytes())
            row = sampling[condition][str(seed)]
            assert row["updates"] == final and row["examples_seen"] == final * 64
            assert row["support_states"] == len(indices) and row["unique_states_sampled"] == np.count_nonzero(global_counts)
            assert row["local_batch_index_sha256"] == local_digest.hexdigest()
            assert row["batch_index_sha256"] == row["global_batch_index_sha256"] == global_digest.hexdigest()
            assert np.array_equal(global_arrays[f"{condition}_seed{seed}"], global_counts)
            assert np.array_equal(local_arrays[f"{condition}_seed{seed}"], local_counts)
            if condition == "collected_unique":
                assert not global_counts[~visited_mask].any(), "unvisited current state entered training"
            total_queries = int(np.dot(global_counts.astype(np.uint64), live.sum(1).astype(np.uint64)))
            external_queries = int(np.dot(global_counts.astype(np.uint64), outside.sum(1).astype(np.uint64)))
            assert row["successor_queries"]["nonterminal_queries"] == total_queries
            assert row["successor_queries"]["outside_collected_support_queries"] == external_queries
            close(row["successor_queries"]["outside_collected_support_fraction"], external_queries / total_queries)
            if condition == "exhaustive" and not smoke:
                assert local_digest.hexdigest() == global_digest.hexdigest() == old_sampling["double_dqn"][str(seed)]["batch_index_sha256"]
    assert {r["seed"] for r in provenance["paired_consistency"]} == set(protocol["seeds"])
    for row in provenance["paired_consistency"]:
        assert row["complete"] and row["initial_weights_identical"] and row["same_update_count"]
        assert row["same_batch_indices_required"] is False
    if not smoke:
        assert {r["seed"] for r in provenance["exhaustive_replication"]} == set(protocol["seeds"])
        for row in provenance["exhaustive_replication"]:
            assert row["applicable"] and row["final_weights_identical"] and row["initial_weights_identical"] and row["batch_index_sha256_identical"]
            previous_path = prior / "models" / f"double_dqn_seed{row['seed']}_update30000.pt"
            assert common.sha(previous_path) == row["archive_model_sha256"]
            previous = torch.load(previous_path, map_location="cpu", weights_only=True)
            actual = torch.load(study / "models" / f"exhaustive_seed{row['seed']}_update30000.pt", map_location="cpu", weights_only=True)
            assert actual["parameter_hash"] == previous["parameter_hash"]
            assert actual["target_parameter_hash"] == previous["target_parameter_hash"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/coverage/pilot_v1"))
    parser.add_argument("--prior-study", type=Path, default=Path("experiments/fixed_targets/pilot_v1"))
    parser.add_argument("--supervised-study", type=Path, default=Path("experiments/supervised/pilot_v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--skip-forward-inference", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Assertions must be enabled; do not run python -O.")
    root = Path(__file__).resolve().parents[1]
    resolve = lambda p: p.resolve() if p.is_absolute() else (root / p).resolve()
    study, prior, earlier = map(resolve, (args.study, args.prior_study, args.supervised_study))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(study / "source"))
    from q6.learning import DQN
    from q6.world import CollectionWorld, WorldConfig
    torch.set_num_threads(1)
    started = time.monotonic()
    result, protocol, metadata = [read_json(study / name) for name in ("results.json", "protocol.json", "dataset_metadata.json")]
    smoke = bool(protocol["smoke"] or protocol["deviations"])
    assert not smoke or args.allow_smoke, "Smoke/deviating artifacts require explicit --allow-smoke."
    assert result["run"]["status"] == "complete"
    assert [c["id"] for c in protocol["conditions"]] == list(CONDITIONS)
    if not smoke:
        assert protocol["seeds"] == [0, 1, 2] and protocol["checkpoints"] == [0, 1000, 3000, 10000, 30000]
        assert protocol["runtime"]["python"].startswith("3.12.") and protocol["runtime"]["torch"].split("+")[0] == "2.8.0"
        assert protocol["runtime"]["numpy"] == "2.0.2" and protocol["git"]["dirty"] is False
        assert protocol["collection"]["episodes_per_map"] == 16
    config = WorldConfig(**{**protocol["world"], "action_mapping": tuple(protocol["world"]["action_mapping"])})
    assert asdict(config) == asdict(WorldConfig())
    data = dict(np.load(study / "dataset.npz", allow_pickle=False))
    transitions = dict(np.load(study / "transitions.npz", allow_pickle=False))
    report = {"study": str(args.study), "smoke_validation_only": smoke,
        "manifest_files": common.audit_manifest(study, protocol, result)}
    provenance = audit_inputs(study, prior, earlier, data, transitions, metadata, protocol, result, CollectionWorld, config, smoke=smoke)
    report.update(common.audit_transitions(data, metadata, transitions, config, CollectionWorld))
    support, collection_report = audit_collection(study, data, metadata, transitions, protocol, result)
    report.update(collection_report)
    audit_sampling(study, prior, data, transitions, support, protocol, result, provenance, smoke=smoke)
    predictions, model_count = audit_model_predictions(study, data, protocol, CONDITIONS, result, DQN, skip_forward=args.skip_forward_inference)
    report.update(audit_raw_tables(study, metadata, data, protocol, result, predictions, smoke=smoke))
    maximum = 2 * len(protocol["seeds"]) * max(protocol["checkpoints"])
    assert result["run"]["train_updates"] == maximum and result["run"]["training_examples"] == maximum * 64
    assert result["run"]["model_parameter_count"] == 20420 and result["run"]["resource_checks"] > maximum
    assert result["run"]["peak_rss_bytes"] <= protocol["budget"]["peak_process_rss_bytes"]
    assert result["run"]["wall_seconds"] <= protocol["budget"]["admission_seconds"]
    progress = result["run"]["progress"]
    assert len(progress) == 2 * len(protocol["seeds"]) * len(protocol["checkpoints"]) and all(r["evaluation_complete"] for r in progress)
    assert {(r["condition"], r["seed"], r["checkpoint"]) for r in progress} == {
        (c, s, cp) for c in CONDITIONS for s in protocol["seeds"] for cp in protocol["checkpoints"]}
    report.update(model_snapshots=model_count, forward_inference_checked=not args.skip_forward_inference,
        dense_prediction_arrays_reproduced=0 if args.skip_forward_inference else 4 * len(protocol["seeds"]),
        local_and_global_sampler_streams_reconstructed=4 * len(protocol["seeds"]),
        exhaustive_historical_identity_checked=not smoke, no_unvisited_current_state_samples=True,
        audit_seconds=time.monotonic() - started)
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
