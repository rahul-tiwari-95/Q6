"""Frozen familiar-start action-support diagnosis, without training or collection."""
from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import platform
import shlex
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .adaptation import write_json
from .competence import BudgetReached, ROOT, artifact_path, canonical_task, git_info
from .diagnostics import ShortestPathPolicy
from .fixed_targets import ConsistencyError, MemoryReached, array_metadata, guard, module_hash, peak_rss_bytes, sha
from .logged_graph import logged_graph_metrics, solve_logged_graph
from .map_replay import checked_input
from .panel_evaluation import FrozenPolicy
from .recorded_actions import build_recorded_table
from .supervised import layout_key, save_manifest
from .world import CollectionWorld, WorldConfig

CONDITIONS = ("constrained_bootstrap", "logged_graph")
ACTION_SETS = ("unrestricted", "logged")
COMPARISONS = (
    {"id": "mask_effect_constrained", "label": "Mask effect: constrained DDQN", "weights": [[0, 0, -1], [0, 1, 1]]},
    {"id": "mask_effect_graph", "label": "Mask effect: exact logged graph", "weights": [[1, 0, -1], [1, 1, 1]]},
    {"id": "graph_minus_constrained_unrestricted", "label": "Exact minus DDQN: unrestricted", "weights": [[0, 0, -1], [1, 0, 1]]},
    {"id": "graph_minus_constrained_logged", "label": "Exact minus DDQN: logged", "weights": [[0, 1, -1], [1, 1, 1]]},
    {"id": "interaction", "label": "Exact masking effect minus DDQN masking effect", "weights": [[1, 1, 1], [1, 0, -1], [0, 1, -1], [0, 0, 1]]},
)
METRICS = ("success", "efficient_success", "steps", "base_return", "shaped_return", "noop_steps")
CLOCK_GROUPS = {"1_8": (1, 8), "9_16": (9, 16), "17_24": (17, 24), "25_32": (25, 32)}


def logged_success_paths(recorded, remaining, map_seeds, enforce=None):
    """Reachability minimizes successful path length, independently of return Q."""
    enforce = enforce or (lambda: None)
    _, validation = solve_logged_graph(recorded, remaining, map_seeds, enforce=enforce)
    mask = recorded["observed"]
    rows, actions = np.nonzero(mask)
    shortest = np.full(len(mask), -1, np.int32)
    for clock in np.unique(remaining[rows]):
        enforce()
        selected = remaining[rows] == clock
        source, action = rows[selected], actions[selected]
        terminated = recorded["terminated"][source, action]
        ended = recorded["ends"][source, action]
        distance = np.full(len(source), np.iinfo(np.int32).max, np.int32)
        distance[terminated] = 1
        live = ~ended
        successors = recorded["successor_indices"][source[live], action[live]]
        downstream = shortest[successors]
        distance[np.flatnonzero(live)[downstream >= 0]] = downstream[downstream >= 0] + 1
        current = np.unique(source)
        minimum = np.full(len(mask), np.iinfo(np.int32).max, np.int32)
        np.minimum.at(minimum, source, distance)
        shortest[current] = np.where(minimum[current] < np.iinfo(np.int32).max, minimum[current], -1)
    reachable = shortest >= 0
    shortest.flags.writeable = reachable.flags.writeable = False
    return reachable, shortest, {"supported_states": validation["supported_states"], "reachable_states": int(reachable.sum()),
        "unreachable_supported_states": int(mask.any(1).sum() - reachable.sum()), "definition": "shortest successful path over logged edges, not return-optimal Q policy", "closure_verified": True}


def make_row_lookup(data):
    keys = [(int(m), int(p[0]), int(p[1]), int(t)) for m, p, t in zip(data["map_seeds"], data["positions"], data["remaining"])]
    lookup = {key: i for i, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ConsistencyError("duplicate map/position/clock state rows")
    return lookup


def select_action(q_values, recorded_mask, action_set):
    q_values, recorded_mask = np.asarray(q_values), np.asarray(recorded_mask, bool)
    if q_values.shape != (4,) or recorded_mask.shape != (4,) or not np.all(np.isfinite(q_values)):
        raise ConsistencyError("invalid frozen predictions/action mask")
    if action_set == "unrestricted":
        return int(q_values.argmax())
    if action_set != "logged" or not recorded_mask.any():
        raise ConsistencyError("logged action selection requires nonempty recorded mask; no fallback")
    return int(np.where(recorded_mask, q_values, -np.inf).argmax())


def evaluate_start(model, config, data, row_lookup, table, graph_targets, *, bank_id, condition, seed, map_seed, action_set,
                   enforce=None, policy="learner", reachable=None, step_sink=None):
    enforce = enforce or (lambda: None)
    env = CollectionWorld(config)
    state, _ = env.reset(seed=map_seed)
    block = f"block_{(map_seed - 300000) // 32}"
    episode_id = f"{bank_id}:{condition}:{seed}:{action_set}:{map_seed}:{policy}"
    base = {"episode_id": episode_id, "bank_id": bank_id, "condition": condition, "seed": seed, "checkpoint": 30000 if policy == "learner" else 0,
        "policy": policy, "action_set": action_set, "mode": "greedy", "panel": block, "block": block, "map_seed": map_seed, "repetition": 0}
    replay = {**base, "world": "A", "grid_size": config.size, "action_mapping": list(config.action_mapping), "walls": np.argwhere(env.walls).tolist(),
        "pellets_initial": np.argwhere(env.pellets).tolist(), "start": list(env.position), "steps": []}
    row = {**base, "task_hash": canonical_task(env), "steps": 0, "success": 0, "base_return": 0., "shaped_return": 0., "discounted_return": 0.,
        "noop_steps": 0, "supported_steps": 0, "unsupported_steps": 0, "off_mask_actions": 0, "unrestricted_argmax_outside_steps": 0,
        "first_off_mask_step": None, "first_support_exit_step": None, "support_exits": 0, "support_reentries": 0,
        "logged_regret_steps": 0, "logged_regret_sum": 0., "restricted_agreement_steps": 0, "restricted_agreement_count": 0,
        "logged_reachable_steps": 0, "logged_reachability_losses": 0, "checkpoint_complete": 1}
    planner = ShortestPathPolicy(config.size) if policy == "shortest_path" else None
    transitions = []
    while True:
        enforce()
        current = row_lookup.get((map_seed, *env.position, config.horizon - env.elapsed))
        if current is None or not np.array_equal(state, data["observations"][current]):
            raise ConsistencyError("current observation differs from archived state identity")
        mask = table["observed"][current] if table is not None else np.zeros(4, bool)
        supported = bool(mask.any()) if table is not None else None
        q, logged_q = None, [None] * 4
        if supported:
            logged_q = [float(graph_targets[current, a]) if mask[a] else None for a in range(4)]
        if policy == "learner":
            with torch.no_grad():
                q = model.online(torch.from_numpy(state).unsqueeze(0))[0].numpy()
            action = select_action(q, mask, action_set)
        elif policy == "logged_q":
            if not supported:
                raise ConsistencyError("logged-Q reference left closed support")
            action = int(np.where(mask, graph_targets[current], -np.inf).argmax())
        elif policy == "shortest_path":
            action = planner.act(state)
        else:
            raise ConsistencyError("unknown frozen evaluation policy")
        before, remaining = list(env.position), config.horizon - env.elapsed
        selected_recorded = bool(mask[action]) if supported is not None else None
        off_mask = bool(supported and not mask[action])
        unrestricted_outside = bool(supported and q is not None and not mask[q.argmax()])
        regret, agreement = None, None
        if supported:
            optimum = float(np.max(graph_targets[current, mask]))
            if selected_recorded:
                regret = max(0., optimum - float(graph_targets[current, action]))
            if q is not None:
                chosen_restricted = select_action(q, mask, "logged")
                agreement = optimum - float(graph_targets[current, chosen_restricted]) <= 1e-6
        was_reachable = bool(reachable[current]) if reachable is not None and supported else None
        state, reward, terminated, truncated, info = env.step(action)
        ended = terminated or truncated
        successor = -1 if ended else row_lookup.get((map_seed, *env.position, config.horizon - env.elapsed), -1)
        if not ended and successor < 0:
            raise ConsistencyError("nonterminal successor missing from full observation rows")
        successor_supported = bool(table["observed"][successor].any()) if table is not None and not ended else None
        if selected_recorded:
            if (not np.isclose(reward, table["rewards"][current, action], atol=1e-12, rtol=0)
                    or terminated != bool(table["terminated"][current, action]) or truncated != bool(table["truncated"][current, action])
                    or successor != int(table["successor_indices"][current, action])):
                raise ConsistencyError("selected recorded transition differs from logged outcome")
        if (action_set == "logged" or policy == "logged_q") and (not supported or not selected_recorded or (not ended and not successor_supported)):
            raise ConsistencyError("masked trajectory violated logged closure")
        exited = bool(supported and not ended and successor_supported is False)
        reentered = bool(supported is False and not ended and successor_supported)
        lost_reachability = bool(was_reachable and ((ended and not terminated) or (not ended and successor_supported and not reachable[successor])))
        step = env.elapsed
        frame = {"step": step, "current_row": current, "current_supported": supported, "remaining_before": remaining,
            "position_before": before, "recorded_mask": mask.tolist() if table is not None else None, "q_values": q.tolist() if q is not None else None,
            "logged_q_values": logged_q, "action": action, "selected_action_recorded": selected_recorded, "off_mask_action": off_mask,
            "unrestricted_argmax_outside": unrestricted_outside, "logged_value_regret": regret, "restricted_action_agreement": agreement,
            "logged_success_reachable": was_reachable, "logged_reachability_lost": lost_reachability,
            "successor_row": successor, "successor_supported": successor_supported, "support_exit": exited, "support_reentry": reentered,
            "position": list(env.position), "reward": reward, "base_reward": info["base_reward"], "pellets": np.argwhere(env.pellets).tolist(),
            "terminated": terminated, "truncated": truncated}
        transitions.append({**base, **frame})
        if step_sink:
            step_sink(transitions[-1])
        replay["steps"].append(frame)
        row["steps"] = step
        row["base_return"] += info["base_reward"]
        row["shaped_return"] += reward
        row["discounted_return"] += config.gamma ** (step - 1) * reward
        row["noop_steps"] += int(before == list(env.position))
        row["supported_steps"] += int(supported is True)
        row["unsupported_steps"] += int(supported is False)
        row["off_mask_actions"] += int(off_mask)
        row["unrestricted_argmax_outside_steps"] += int(unrestricted_outside)
        row["support_exits"] += int(exited)
        row["support_reentries"] += int(reentered)
        if off_mask and row["first_off_mask_step"] is None:
            row["first_off_mask_step"] = step
        if exited and row["first_support_exit_step"] is None:
            row["first_support_exit_step"] = step
        if regret is not None:
            row["logged_regret_sum"] += regret
            row["logged_regret_steps"] += 1
        if agreement is not None:
            row["restricted_agreement_steps"] += 1
            row["restricted_agreement_count"] += int(agreement)
        row["logged_reachable_steps"] += int(was_reachable is True)
        row["logged_reachability_losses"] += int(lost_reachability)
        if ended:
            break
    row["success"] = int(terminated)
    replay["success"] = bool(terminated)
    return row, transitions, replay


def reduce_episodes(rows):
    if not rows:
        return {}
    n, steps = len(rows), sum(r["steps"] for r in rows)
    supported = sum(r["supported_steps"] for r in rows)
    successful = [r for r in rows if r["success"]]
    result = {"episodes": n, "success_rate": sum(r["success"] for r in rows) / n,
        "efficient_success_rate": sum(r["efficient_success"] for r in rows) / n, "mean_steps": steps / n,
        "base_return": sum(r["base_return"] for r in rows) / n, "shaped_return": sum(r["shaped_return"] for r in rows) / n,
        "noop_steps": sum(r["noop_steps"] for r in rows), "evaluation_steps": steps,
        "supported_steps": supported, "unsupported_steps": sum(r["unsupported_steps"] for r in rows),
        "off_mask_actions": sum(r["off_mask_actions"] for r in rows), "unrestricted_argmax_outside_steps": sum(r["unrestricted_argmax_outside_steps"] for r in rows),
        "off_mask_episodes": sum(r["first_off_mask_step"] is not None for r in rows), "support_exit_episodes": sum(r["first_support_exit_step"] is not None for r in rows),
        "support_exits": sum(r["support_exits"] for r in rows), "support_reentries": sum(r["support_reentries"] for r in rows),
        "logged_regret_steps": sum(r["logged_regret_steps"] for r in rows), "logged_regret_sum": sum(r["logged_regret_sum"] for r in rows),
        "restricted_agreement_steps": sum(r["restricted_agreement_steps"] for r in rows), "restricted_agreement_count": sum(r["restricted_agreement_count"] for r in rows),
        "logged_reachable_steps": sum(r["logged_reachable_steps"] for r in rows), "logged_reachability_losses": sum(r["logged_reachability_losses"] for r in rows),
        "successful_episodes": len(successful), "successful_mean_steps": sum(r["steps"] for r in successful) / len(successful) if successful else None}
    result.update(noop_rate=result["noop_steps"] / steps, off_mask_rate=result["off_mask_actions"] / supported if supported else None,
        unrestricted_argmax_outside_rate=result["unrestricted_argmax_outside_steps"] / supported if supported else None,
        unsupported_step_fraction=result["unsupported_steps"] / steps, off_mask_episode_rate=result["off_mask_episodes"] / n, support_exit_episode_rate=result["support_exit_episodes"] / n,
        mean_logged_regret=result["logged_regret_sum"] / result["logged_regret_steps"] if result["logged_regret_steps"] else None,
        restricted_action_agreement=result["restricted_agreement_count"] / result["restricted_agreement_steps"] if result["restricted_agreement_steps"] else None)
    return result


def summarize_crossed(rows, references, banks, seeds, blocks):
    seed_results, aggregate = [], []
    for bank in banks:
        for block in ["all", *blocks]:
            for condition in CONDITIONS:
                for action_set in ACTION_SETS:
                    group = [r for r in rows if r["bank_id"] == bank and r["condition"] == condition and r["action_set"] == action_set and (block == "all" or r["block"] == block)]
                    if group:
                        aggregate.append({"bank_id": bank, "condition": condition, "action_set": action_set, "block": block, **reduce_episodes(group)})
                    for seed in seeds:
                        sub = [r for r in group if r["seed"] == seed]
                        if sub:
                            seed_results.append({"bank_id": bank, "condition": condition, "action_set": action_set, "block": block, "seed": seed, **reduce_episodes(sub)})
    reference_aggregate = []
    for bank, policy in sorted({(str(r["bank_id"]), r["policy"]) for r in references}):
        for block in ["all", *blocks]:
            group = [r for r in references if str(r["bank_id"]) == bank and r["policy"] == policy and (block == "all" or r["block"] == block)]
            if group:
                reference_aggregate.append({"bank_id": int(bank) if bank != "shared" else bank, "policy": policy, "block": block, **reduce_episodes(group)})
    cells = {(r["bank_id"], r["seed"], r["map_seed"], r["condition"], r["action_set"]): r for r in rows}
    per_start = []
    for bank, seed, map_seed in sorted({(r["bank_id"], r["seed"], r["map_seed"]) for r in rows}):
        if not all((bank, seed, map_seed, c, a) in cells for c in CONDITIONS for a in ACTION_SETS):
            continue
        for comparison in COMPARISONS:
            terms = [(weight, cells[(bank, seed, map_seed, CONDITIONS[c], ACTION_SETS[a])]) for c, a, weight in comparison["weights"]]
            per_start.append({"bank_id": bank, "seed": seed, "map_seed": map_seed, "block": terms[0][1]["block"], "comparison": comparison["id"],
                **{metric + "_delta": sum(weight * episode[metric] for weight, episode in terms) for metric in METRICS}})
    paired = {"comparisons": list(COMPARISONS), "per_start": per_start, "per_seed": [], "aggregate": []}
    for bank in banks:
        for block in ["all", *blocks]:
            for comparison in COMPARISONS:
                group = [r for r in per_start if r["bank_id"] == bank and r["comparison"] == comparison["id"] and (block == "all" or r["block"] == block)]
                for seed in seeds:
                    sub = [r for r in group if r["seed"] == seed]
                    if sub:
                        paired["per_seed"].append({"bank_id": bank, "block": block, "comparison": comparison["id"], "seed": seed, "starts": len(sub),
                            **{m + "_delta": float(np.mean([r[m + "_delta"] for r in sub])) for m in METRICS}})
                if group:
                    paired["aggregate"].append({"bank_id": bank, "block": block, "comparison": comparison["id"], "starts": len(group),
                        **{m + "_delta": float(np.mean([r[m + "_delta"] for r in group])) for m in METRICS}})
    pooled = {"aggregate": [], "paired": [], "definition": "outcome means over equal-sized bank cells; paired effects equal-bank means; blocked ratio pools actual steps"}
    for block in ["all", *blocks]:
        for condition in CONDITIONS:
            for action_set in ACTION_SETS:
                group = [r for r in rows if r["condition"] == condition and r["action_set"] == action_set and (block == "all" or r["block"] == block)]
                if group:
                    pooled["aggregate"].append({"condition": condition, "action_set": action_set, "block": block, **reduce_episodes(group)})
        for comparison in COMPARISONS:
            group = [r for r in paired["aggregate"] if r["block"] == block and r["comparison"] == comparison["id"]]
            if group:
                pooled["paired"].append({"block": block, "comparison": comparison["id"], "banks": len(group),
                    **{m + "_delta": float(np.mean([r[m + "_delta"] for r in group])) for m in METRICS}})
    return seed_results, aggregate, reference_aggregate, paired, pooled


def prediction_slice_metrics(predictions, targets, observed):
    result = logged_graph_metrics(predictions, targets, observed)
    rows, actions = np.nonzero(observed)
    counts = observed.sum(1)
    errors = predictions[rows, actions].astype(np.float64) - targets[rows, actions]
    state_signed = np.bincount(rows, weights=errors, minlength=len(observed)) / counts
    result["state_mean_signed_error"] = float(state_signed.mean())
    result["state_mean_squared_offset"] = float(np.mean(state_signed**2))
    multiple = counts > 1
    centered = errors - state_signed[rows]
    centered_abs = np.bincount(rows, weights=np.abs(centered), minlength=len(observed)) / counts
    centered_sq = np.bincount(rows, weights=centered**2, minlength=len(observed)) / counts
    sorted_target = np.sort(np.where(observed, targets, -np.inf), axis=1)
    result.update(multiple_action_states=int(multiple.sum()),
        centered_state_mean_abs_error=float(centered_abs[multiple].mean()) if multiple.any() else None,
        centered_state_mean_squared_error=float(centered_sq[multiple].mean()) if multiple.any() else None,
        mean_target_top_two_gap=float(np.mean(sorted_target[multiple, -1] - sorted_target[multiple, -2])) if multiple.any() else None,
        mean_unrestricted_prediction_gap=float(np.mean(predictions.max(1) - np.where(observed, predictions, -np.inf).max(1))))
    return result


def prior_prediction_slices(prior_predictions, supports, tables, graph, reachability, data, banks, seeds, enforce=None):
    enforce = enforce or (lambda: None)
    result = []
    for bank in banks:
        support, table = supports[bank], tables[bank]
        mask = table["observed"][support]
        targets = graph[bank][support]
        count = mask.sum(1)
        selectors = [("all", "all", np.ones(len(support), bool)), ("action_count", "one", count == 1), ("action_count", "multiple", count > 1),
            ("reachability", "reachable", reachability[bank][support]), ("reachability", "unreachable", ~reachability[bank][support])]
        selectors.append(("original_start", "clock32", data["remaining"][support] == 32))
        selectors += [("remaining", name, (data["remaining"][support] >= low) & (data["remaining"][support] <= high)) for name, (low, high) in CLOCK_GROUPS.items()]
        if not np.array_equal(prior_predictions[f"bank{bank}_state_rows"], support):
            raise ConsistencyError("archived prediction order differs from recorded support")
        for condition in CONDITIONS:
            for seed in seeds:
                predictions = prior_predictions[f"bank{bank}_{condition}_seed{seed}"]
                if predictions.shape != (len(support), 4):
                    raise ConsistencyError("archived supported prediction shape differs")
                for axis, group, selected in selectors:
                    enforce()
                    if selected.any():
                        result.append({"bank_id": bank, "condition": condition, "seed": seed, "axis": axis, "group": group,
                            "classification": "exploratory prior-data original-start slice" if axis == "original_start" else "predeclared descriptive prior-prediction slice",
                            **prediction_slice_metrics(predictions[selected], targets[selected], mask[selected])})
    return {"classification": "predeclared descriptive slices of archived predictions; no new inference or gates",
        "per_slice": result, "clock_groups": CLOCK_GROUPS, "new_inference_rows": 0,
        "centered_definition": "subtract each state's mean observed-action prediction error; average absolute/squared centered error within multiple-action states then across those states",
        "top_two_gap_definition": "largest minus second-largest logged target, including ties, multiple-action states only"}


def verify_cells(rows, references, banks, seeds, maps):
    expected = {(b, c, s, a, m) for b in banks for c in CONDITIONS for s in seeds for a in ACTION_SETS for m in maps}
    actual = [(r["bank_id"], r["condition"], r["seed"], r["action_set"], r["map_seed"]) for r in rows]
    ref_expected = {(str(b), "logged_q", m) for b in banks for m in maps} | {("shared", "shortest_path", m) for m in maps}
    ref_actual = [(str(r["bank_id"]), r["policy"], r["map_seed"]) for r in references]
    return {"learner_expected": len(expected), "learner_actual": len(actual), "reference_expected": len(ref_expected), "reference_actual": len(ref_actual),
        "learner_unique": len(set(actual)), "reference_unique": len(set(ref_actual)),
        "complete": len(actual) == len(expected) and set(actual) == expected and len(ref_actual) == len(ref_expected) and set(ref_actual) == ref_expected}


def model_boundary(record, model, output):
    return {"online": model.parameter_hash(), "target": module_hash(model.target), "copy": sha(output / record["saved"]), "source": sha(ROOT / record["source"])}


def run_study(output, protocol_file, *, banks=(1, 2, 3), seeds=(0, 1, 2), map_seeds=None, prior_dir=None,
              max_seconds=1200, max_rss_bytes=4 * 1024**3, dashboard=None, smoke=False):
    output, protocol_file = Path(output), Path(protocol_file)
    banks, seeds = list(banks), list(seeds)
    maps = list(map_seeds if map_seeds is not None else range(300000, 300256))
    prior_dir = Path(prior_dir) if prior_dir is not None else ROOT / "experiments/logged_graph/pilot_v1"
    if not protocol_file.is_file() or not banks or not seeds or not maps or len(set(banks)) != len(banks) or len(set(seeds)) != len(seeds) or len(set(maps)) != len(maps):
        raise ValueError("protocol and distinct nonempty banks/seeds/maps required")
    if min(banks) < 1 or min(seeds) < 0 or not set(maps) <= set(range(300000, 300256)) or maps != sorted(maps) or min(max_seconds, max_rss_bytes) <= 0:
        raise ValueError("invalid frozen study configuration")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    started, checks = time.monotonic(), 0
    deadline = started + max_seconds
    def enforce():
        nonlocal checks
        checks += 1
        guard(deadline, max_rss_bytes)
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "torch_threads": 1, "device": "cpu"}
    expected = {"banks": [1, 2, 3], "seeds": [0, 1, 2], "map_seeds": list(range(300000, 300256)), "max_seconds": 1200, "max_rss_bytes": 4 * 1024**3}
    actual = dict(banks=banks, seeds=seeds, map_seeds=maps, max_seconds=max_seconds, max_rss_bytes=max_rss_bytes)
    deviations = [{"field": k, "actual": v, "declared": expected[k]} for k, v in actual.items() if v != expected[k]]
    if not runtime["python"].startswith("3.12.") or runtime["torch"].split("+")[0] != "2.8.0" or runtime["numpy"] != "2.0.2":
        deviations.append({"field": "runtime", "actual": runtime, "declared": "Python3.12/Torch2.8.0/NumPy2.0.2"})
    if prior_dir.resolve() != (ROOT / "experiments/logged_graph/pilot_v1").resolve():
        deviations.append({"field": "prior_dir", "actual": artifact_path(prior_dir), "declared": "experiments/logged_graph/pilot_v1"})
    git = git_info()
    if git["dirty"] is not False or not git["revision"]:
        deviations.append({"field": "git", "actual": git, "declared": "clean frozen revision"})
    blocks = {f"block_{(m - 300000) // 32}": [] for m in maps}
    for m in maps:
        blocks[f"block_{(m - 300000) // 32}"].append(m)
    replay_maps = [m for m in maps if (m - 300000) % 32 == 0]
    config = WorldConfig()
    source_files = sorted((ROOT / "q6").glob("*.py"))
    protocol = {"id": "familiar-starts-v1", "world": asdict(config), "rule_visibility": "observed", "conditions": list(CONDITIONS), "action_sets": list(ACTION_SETS),
        "bank_ids": banks, "seeds": seeds, "map_seeds": maps, "blocks": [{"id": k, "map_seeds": v} for k, v in blocks.items()], "replay_map_seeds": replay_maps,
        "checkpoint": 30000, "evaluation_only": True, "mode": "greedy", "repetitions": 1, "new_competence_gates": False,
        "runtime": runtime, "git": git, "smoke": smoke, "deviations": deviations, "protocol_sha256": sha(protocol_file),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}, "created_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": list(COMPARISONS), "primary_comparison": "interaction", "primary_metric": "efficient_success_delta",
        "diagnostics": {"first_step_numbering": "1-based action/transition index", "off_mask_denominator": "supported current decisions", "terminal_is_support_exit": False,
            "missing_logged_values": None, "ranking_atol": 1e-6, "ranking_rtol": 0, "clock_groups": CLOCK_GROUPS},
        "budget": {"admission_seconds": max_seconds, "peak_process_rss_bytes": max_rss_bytes, "all_phases_included": True}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    shutil.copy2(protocol_file, output / "protocol.md")
    for p in source_files:
        dest = output / "source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    write_json(output / "protocol.json", protocol)
    command = ["python", "-m", "q6.familiar_starts", "--output", str(output), "--protocol-file", str(protocol_file), "--banks", ",".join(map(str, banks)),
        "--seeds", ",".join(map(str, seeds)), "--map-seeds", ",".join(map(str, maps)), "--prior-dir", artifact_path(prior_dir), "--max-seconds", str(max_seconds), "--max-rss-bytes", str(max_rss_bytes)]
    if smoke:
        command.append("--smoke")
    (output / "command.txt").write_text(shlex.join(command) + "\n# Use a new output path for reproduction.\n")
    (output / "environment.txt").write_text("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")})) + "\n")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    status, reason = "complete", None
    data, tables, supports, graph, reachable, shortest, models = {}, {}, {}, {}, {}, {}, {}
    records, rows, references, trajectories, starts, graph_checks, reach_summaries = [], [], [], [], [], [], []
    prediction_slices = {"per_slice": [], "new_inference_rows": 0}
    input_hashes = {}
    timings = {"preparation": 0., "references": 0., "evaluation": 0., "aggregation": 0.}
    step_count = 0
    occupancy = {}
    rng_initial = torch.get_rng_state().clone()
    with gzip.open(output / "steps.jsonl.gz", "wt") as trace:
        def step_sink(row):
            nonlocal step_count
            step_count += 1
            trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            if row["policy"] == "learner":
                key = row["bank_id"], row["condition"], row["seed"], row["action_set"], row["remaining_before"]
                item = occupancy.setdefault(key, {"decisions": 0, "supported": 0, "unsupported": 0, "off_mask": 0, "exits": 0, "reentries": 0})
                for field, value in (("decisions", 1), ("supported", row["current_supported"]), ("unsupported", not row["current_supported"]), ("off_mask", row["off_mask_action"]), ("exits", row["support_exit"]), ("reentries", row["support_reentry"])):
                    item[field] += int(value)
        try:
            phase = time.monotonic()
            try:
                enforce()
                manifest = json.loads((prior_dir / "manifest.json").read_text())
                if manifest["status"] != "complete":
                    raise ConsistencyError("prior logged-graph archive is incomplete")
                input_hashes["manifest.json"] = sha(prior_dir / "manifest.json")
                shutil.copy2(prior_dir / "manifest.json", output / "prior_manifest.json")
                names = ("dataset.npz", "dataset_metadata.json", "supports.npz", "recorded_transitions.npz", "logged_graph_targets.npz", "fit_predictions.npz", "fit_diagnostics.json", "protocol.json", "protocol.md")
                for name in names:
                    enforce()
                    checked_input(prior_dir, manifest, name, input_hashes)
                    shutil.copy2(prior_dir / name, output / ("prior_" + name if name.startswith("protocol") else name))
                for name in ("world", "learning", "diagnostics", "logged_graph", "recorded_actions", "panel_evaluation"):
                    filename = f"source/q6/{name}.py"
                    checked_input(prior_dir, manifest, filename, input_hashes)
                    if sha(prior_dir / filename) != sha(ROOT / f"q6/{name}.py"):
                        raise ConsistencyError("frozen inference/world/graph source differs from archived source")
                metadata = json.loads((prior_dir / "dataset_metadata.json").read_text())
                with np.load(output / "dataset.npz") as saved:
                    data = {k.removeprefix("train_"): saved[k] for k in saved.files if k.startswith("train_")}
                lookup = make_row_lookup(data)
                metadata_by_map = {r["map_seed"]: r for r in metadata["train"]}
                if set(metadata_by_map) != set(range(300000, 300256)):
                    raise ConsistencyError("archived training metadata must contain all256 declared maps")
                original_start_rows = np.sort(np.asarray([lookup[(m, *metadata_by_map[m]["original_start"], config.horizon)] for m in sorted(metadata_by_map)], np.int64))
                with np.load(output / "supports.npz") as support_file, np.load(output / "recorded_transitions.npz") as table_file, np.load(output / "logged_graph_targets.npz") as target_file:
                    for bank in banks:
                        enforce()
                        support = support_file[f"{CONDITIONS[0]}_bank{bank}"]
                        if not np.array_equal(support, support_file[f"{CONDITIONS[1]}_bank{bank}"]):
                            raise ConsistencyError("frozen arms have different support")
                        filename = f"banks/bank{bank}/collection_steps.csv"
                        checked_input(prior_dir, manifest, filename, input_hashes)
                        (output / filename).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(prior_dir / filename, output / filename)
                        table, _ = build_recorded_table(output / filename, data, support, enforce)
                        if array_metadata(table) != array_metadata({k: table_file[f"bank{bank}_{k}"] for k in table}):
                            raise ConsistencyError("recorded graph differs from copied original logs")
                        exact, _ = solve_logged_graph(table, data["remaining"], data["map_seeds"], enforce=enforce)
                        graph[bank] = target_file[f"bank{bank}_targets_float64"]
                        matches = np.array_equal(exact, graph[bank], equal_nan=True)
                        graph_checks.append({"bank_id": bank, "logged_targets_reproduced": matches, "arrays": array_metadata({"targets": graph[bank]})})
                        if not matches:
                            raise ConsistencyError("archived logged targets fail recorded-only recurrence")
                        reachable[bank], shortest[bank], summary = logged_success_paths(table, data["remaining"], data["map_seeds"], enforce)
                        reach_summaries.append({"bank_id": bank, **summary})
                        clock32_rows = np.sort(support[data["remaining"][support] == config.horizon])
                        if not np.array_equal(clock32_rows, original_start_rows):
                            raise ConsistencyError("clock32 support rows differ from all256 original metadata spawns")
                        graph_checks[-1].update(clock32_support_is_original_start_rows=True, original_start_rows=array_metadata({"rows": original_start_rows})["rows"])
                        table_before = array_metadata(table)
                        support.flags.writeable = graph[bank].flags.writeable = False
                        tables[bank], supports[bank] = table, support
                        graph_checks[-1].update(table_arrays=table_before, support_arrays=array_metadata({"support": support}))
                np.savez_compressed(output / "reachability.npz", **{f"bank{b}_{name}": values for b in banks for name, values in (("reachable", reachable[b]), ("shortest_steps", shortest[b]))})
                env = CollectionWorld(config)
                for map_seed in maps:
                    enforce()
                    observation, _ = env.reset(seed=map_seed)
                    state_row = lookup[(map_seed, *env.position, config.horizon)]
                    old = metadata_by_map[map_seed]
                    if old["original_start"] != list(env.position) or old["layout_hash"] != layout_key(env) or not np.array_equal(observation, data["observations"][state_row]):
                        raise ConsistencyError("original start/layout/observation differs from archive")
                    for bank in banks:
                        if not tables[bank]["observed"][state_row].any():
                            raise ConsistencyError("original clock32 spawn is outside recorded support")
                        starts.append({"bank_id": bank, "map_seed": map_seed, "block": f"block_{(map_seed - 300000) // 32}", "state_row": state_row, "position": list(env.position),
                            "remaining": config.horizon, "layout_hash": old["layout_hash"], "logged_success_reachable": bool(reachable[bank][state_row]),
                            "logged_shortest_steps": int(shortest[bank][state_row]) if reachable[bank][state_row] else None})
                write_json(output / "starts.json", starts)
                for bank in banks:
                    for condition in CONDITIONS:
                        for seed in seeds:
                            enforce()
                            filename = f"models/bank{bank}_{condition}_seed{seed}_update30000.pt"
                            checked_input(prior_dir, manifest, filename, input_hashes)
                            shutil.copy2(prior_dir / filename, output / filename)
                            snapshot = torch.load(output / filename, map_location="cpu", weights_only=True)
                            if snapshot["optimizer_updates"] != 30000:
                                raise ConsistencyError("familiar policies must use frozen final30000 weights")
                            model = FrozenPolicy(snapshot)
                            record = {"bank_id": bank, "condition": condition, "seed": seed, "checkpoint": 30000, "saved": filename,
                                "source": artifact_path(prior_dir / filename), "action_set_checks": []}
                            record["before"] = model_boundary(record, model, output)
                            record["initial_source_copy_identical"] = record["before"]["copy"] == record["before"]["source"]
                            if not record["initial_source_copy_identical"]:
                                raise ConsistencyError("frozen model copy differs from manifest-verified source")
                            records.append(record)
                            models[(bank, condition, seed)] = model
                with np.load(output / "fit_predictions.npz") as predictions:
                    prediction_slices = prior_prediction_slices(predictions, supports, tables, graph, reachable, data, banks, seeds, enforce)
                write_json(output / "prediction_slices.json", prediction_slices)
                enforce()
            finally:
                timings["preparation"] += time.monotonic() - phase
            planner_steps = {}
            phase = time.monotonic()
            try:
                for map_seed in maps:
                    episode, _, replay = evaluate_start(None, config, data, lookup, None, None, bank_id="shared", condition="shared", seed=0,
                        map_seed=map_seed, action_set="unrestricted", policy="shortest_path", enforce=enforce, step_sink=step_sink)
                    if not episode["success"]:
                        raise ConsistencyError("full-world planner failed original connected start")
                    planner_steps[map_seed] = episode["steps"]
                    episode["efficient_success"] = episode["success"]
                    references.append(episode)
                    if map_seed in replay_maps:
                        trajectories.append(replay)
                for start in starts:
                    start["planner_steps"] = planner_steps[start["map_seed"]]
                    start["logged_efficient_success_reachable"] = start["logged_success_reachable"] and start["logged_shortest_steps"] <= 2 * start["planner_steps"]
                write_json(output / "starts.json", starts)
                for bank in banks:
                    for map_seed in maps:
                        episode, _, replay = evaluate_start(None, config, data, lookup, tables[bank], graph[bank], bank_id=bank, condition="shared", seed=0,
                            map_seed=map_seed, action_set="logged", policy="logged_q", enforce=enforce, reachable=reachable[bank], step_sink=step_sink)
                        start = next(r for r in starts if r["bank_id"] == bank and r["map_seed"] == map_seed)
                        value = float(np.nanmax(graph[bank][start["state_row"]]))
                        if abs(episode["discounted_return"] - value) > 1e-10 or (episode["success"] and not start["logged_success_reachable"]):
                            raise ConsistencyError("logged-Q reference violates return or reachability")
                        episode["efficient_success"] = int(episode["success"] and episode["steps"] <= 2 * planner_steps[map_seed])
                        references.append(episode)
                        if map_seed in replay_maps:
                            trajectories.append(replay)
            finally:
                timings["references"] += time.monotonic() - phase
            phase = time.monotonic()
            try:
                for record in records:
                    bank, condition, seed = record["bank_id"], record["condition"], record["seed"]
                    model = models[(bank, condition, seed)]
                    for action_set in ACTION_SETS:
                        before, rng_before = model_boundary(record, model, output), torch.get_rng_state().clone()
                        if before != record["before"]:
                            raise ConsistencyError("frozen model changed before action-set evaluation")
                        check = {"action_set": action_set, "before": before}
                        record["action_set_checks"].append(check)
                        try:
                            for map_seed in maps:
                                episode, _, replay = evaluate_start(model, config, data, lookup, tables[bank], graph[bank], bank_id=bank, condition=condition, seed=seed,
                                    map_seed=map_seed, action_set=action_set, enforce=enforce, reachable=reachable[bank], step_sink=step_sink)
                                episode["efficient_success"] = int(episode["success"] and episode["steps"] <= 2 * planner_steps[map_seed])
                                rows.append(episode)
                                if map_seed in replay_maps:
                                    trajectories.append(replay)
                        finally:
                            check["after"] = model_boundary(record, model, output)
                            check["unchanged"] = before == check["after"] and torch.equal(rng_before, torch.get_rng_state())
                            if not check["unchanged"]:
                                raise ConsistencyError("frozen model/source/RNG changed during action-set evaluation")
                    print(f"bank{bank} {condition} seed{seed}: both action sets evaluated", flush=True)
            finally:
                timings["evaluation"] += time.monotonic() - phase
        except (BudgetReached, ConsistencyError) as exc:
            status = "inconsistent_not_evidence" if isinstance(exc, ConsistencyError) else "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
            reason = str(exc)
    phase = time.monotonic()
    cells = verify_cells(rows, references, banks, seeds, maps)
    input_integrity = [{"file": name, "before": digest, "after": sha(prior_dir / name), "unchanged": sha(prior_dir / name) == digest} for name, digest in input_hashes.items()]
    for record in records:
        record["after"] = model_boundary(record, models[(record["bank_id"], record["condition"], record["seed"])], output)
        record["unchanged"] = record["before"] == record["after"] and len(record["action_set_checks"]) == 2 and all(r.get("unchanged", False) for r in record["action_set_checks"])
    graph_integrity = [{"bank_id": b, "unchanged": array_metadata(tables[b]) == next(r["table_arrays"] for r in graph_checks if r["bank_id"] == b)
        and array_metadata({"targets": graph[b]}) == next(r["arrays"] for r in graph_checks if r["bank_id"] == b)
        and array_metadata({"support": supports[b]}) == next(r["support_arrays"] for r in graph_checks if r["bank_id"] == b),
        "read_only": not graph[b].flags.writeable and not supports[b].flags.writeable and not reachable[b].flags.writeable and not shortest[b].flags.writeable} for b in tables]
    expected_recordings = len(replay_maps) * (len(banks) * len(seeds) * 4 + len(banks) + 1)
    complete = cells["complete"] and len(records) == len(banks) * len(seeds) * 2 and len(starts) == len(banks) * len(maps) and len(trajectories) == expected_recordings
    complete = complete and all(r["unchanged"] for r in records + input_integrity) and all(r["unchanged"] and r["read_only"] for r in graph_integrity) and torch.equal(rng_initial, torch.get_rng_state())
    if status == "complete" and not complete:
        status, reason = "inconsistent_not_evidence", "expected cells/identities/recordings were not complete"
    seed_results, aggregate, ref_aggregate, paired, pooled = summarize_crossed(rows, references, banks, seeds, blocks)
    reach_result = {"per_bank": [], "by_start": starts, "graph_summaries": reach_summaries, "classification": "logged success/path ceiling distinct from return-optimal Q reference"}
    for bank in banks:
        group = [r for r in starts if r["bank_id"] == bank and "planner_steps" in r]
        if group:
            reach_result["per_bank"].append({"bank_id": bank, "starts": len(group), "success_reachable_starts": sum(r["logged_success_reachable"] for r in group),
                "success_ceiling": float(np.mean([r["logged_success_reachable"] for r in group])), "efficient_success_reachable_starts": sum(r["logged_efficient_success_reachable"] for r in group),
                "efficient_success_ceiling": float(np.mean([r["logged_efficient_success_reachable"] for r in group]))})
    effects = [r for r in paired["aggregate"] if r["comparison"] == "interaction" and r["block"] == "all"]
    robustness = {"classification": "descriptive familiar-start selection interaction; no new gates or significance claims", "primary_comparison": "interaction", "primary_metric": "efficient_success_delta", "bank_effects": effects}
    values = [r["efficient_success_delta"] for r in effects]
    robustness.update(mean=float(np.mean(values)) if values else None, minimum=min(values) if values else None, maximum=max(values) if values else None,
        positive_banks=sum(v > 1e-12 for v in values), negative_banks=sum(v < -1e-12 for v in values), zero_banks=sum(abs(v) <= 1e-12 for v in values), sign_tolerance=1e-12)
    occupancy_rows = [{"bank_id": b, "condition": c, "seed": s, "action_set": a, "remaining": t, **v} for (b, c, s, a, t), v in sorted(occupancy.items())]
    timings["aggregation"] = time.monotonic() - phase
    try:
        enforce()
    except BudgetReached as exc:
        status, reason = "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap", str(exc)
    eligible = status == "complete" and not smoke and not deviations
    robustness["eligible"] = eligible
    provenance = {"source_archive": artifact_path(prior_dir), "inputs": input_integrity, "models": records, "expected_cells": cells,
        "logged_graph_checks": graph_checks, "graph_integrity": graph_integrity, "expected_recordings": expected_recordings,
        "actual_recordings": len(trajectories), "torch_rng_unchanged": torch.equal(rng_initial, torch.get_rng_state())}
    artifacts = {name: artifact_path(output / file) for name, file in {"episodes": "episodes.json", "references": "references.json", "steps": "steps.jsonl.gz", "starts": "starts.json",
        "reachability": "reachability.json", "reachability_arrays": "reachability.npz", "prediction_slices": "prediction_slices.json", "paired_differences": "paired_differences.json",
        "trajectories": "trajectories.json", "models": "models.json", "provenance": "provenance.json", "occupancy": "occupancy.json", "manifest": "manifest.json"}.items()}
    artifacts["report"] = "docs/experiments/familiar_starts_results_v1.md"
    result = {"schema_version": 1, "protocol": protocol, "run": {"id": output.name, "status": status, "stop_reason": reason,
        "interpretation": "familiar_starts_descriptive_only" if eligible else "smoke_or_deviation_descriptive_only" if status == "complete" else status,
        "train_updates": 0, "collection_steps": 0, "collection_episodes": 0, "new_support_draws": 0, "new_labels_fitted": 0,
        "learner_episodes": len(rows), "reference_episodes": len(references), "step_records": step_count, "frozen_models": len(records),
        "wall_seconds": time.monotonic() - started, "phase_wall_seconds": timings, "peak_rss_bytes": peak_rss_bytes(), "resource_checks": checks,
        "resource_limits": protocol["budget"], "limitations": ["Privileged logged masks diagnose familiar-state action selection; they are not deployable fresh-map masks.",
            "Familiar blocks, banks and reused learner seeds are not independent replications; no new competence or significance claim.",
            "Return-optimal logged-Q reference and successful-path reachability answer different questions; fresh panels remain separate."]},
        "aggregate": aggregate, "seed_results": seed_results, "references": ref_aggregate, "paired_differences": paired, "pooled": pooled,
        "robustness": robustness, "reachability": reach_result, "prediction_slices": prediction_slices, "occupancy": occupancy_rows,
        "provenance": provenance, "trajectories": trajectories, "artifacts": artifacts}
    for name, value in (("episodes", rows), ("references", references), ("starts", starts), ("reachability", reach_result), ("prediction_slices", prediction_slices),
        ("paired_differences", paired), ("trajectories", trajectories), ("models", records), ("provenance", provenance), ("occupancy", occupancy_rows), ("results", result)):
        write_json(output / f"{name}.json", value)
    save_manifest(output, status)
    if dashboard:
        write_json(Path(dashboard), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--banks", default="1,2,3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--map-seeds")
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    kwargs = {"banks": [int(x) for x in args.banks.split(",")], "seeds": [int(x) for x in args.seeds.split(",")],
        "map_seeds": [int(x) for x in args.map_seeds.split(",")] if args.map_seeds else None,
        "prior_dir": args.prior_dir, "max_seconds": args.max_seconds, "max_rss_bytes": args.max_rss_bytes, "dashboard": args.dashboard, "smoke": args.smoke}
    if args.smoke:
        kwargs.update(seeds=[0], map_seeds=[300000, 300032])
    result = run_study(args.output, args.protocol_file, **kwargs)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
