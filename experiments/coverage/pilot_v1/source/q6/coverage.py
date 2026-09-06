"""Fixed Double-DQN learning under exhaustive versus randomly collected support.

Only current-state support changes. All four transition targets and full-bank
successor queries remain privileged; this is not trajectory-only online RL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from .competence import BudgetReached, EVAL_FIELDS, ROOT, artifact_path, check_budget, git_info, summarize
from .fixed_targets import (ConsistencyError, MemoryReached, array_metadata, build_transitions,
    efficient_rate, fixed_update, guard, module_hash, peak_rss_bytes, select_fixed_layouts, sha, validate_transitions)
from .learning import DQN
from .optimal import VisibleOptimalQ
from .supervised import (BUCKETS, CHECKPOINTS, LOSS_FIELDS, STATE_FIELDS, TIE_ATOL,
                         BatchSampler, enumerate_panel, full_state_metrics, reduce_states, rollout, save_manifest)
from .world import CollectionWorld, WorldConfig

CONDITIONS = ("exhaustive", "collected_unique")
COLLECTION_STEP_FIELDS = ["map_seed", "repetition", "step", "current_row", "action", "reward", "next_row",
                          "terminated", "truncated", "remaining"]
COLLECTION_EPISODE_FIELDS = ["map_seed", "repetition", "steps", "success", "terminated", "truncated",
                             "complete", "base_return", "shaped_return", "noop_steps"]


class SupportSampler:
    """Uniform unique-support draws; preserve local draws and global exposure."""
    def __init__(self, support, state_count, seed, batch_size=64):
        self.support = np.asarray(support, dtype=np.int32)
        if not len(self.support) or np.any(self.support < 0) or np.any(self.support >= state_count) or np.any(np.diff(self.support) <= 0):
            raise ValueError("support must contain distinct sorted valid global row indices")
        self.local = BatchSampler(len(self.support), seed, batch_size)
        self.counts = np.zeros(state_count, np.uint32)
        self.digest = hashlib.sha256()

    @property
    def updates(self):
        return self.local.updates

    def next_batch(self):
        indices = self.support[self.local.next_batch()].astype(np.int64)
        self.counts[indices] += 1
        self.digest.update(indices.astype("<i8").tobytes())
        return indices


def coverage_metrics(data, transitions, visited_counts):
    visited = visited_counts > 0
    support = np.flatnonzero(visited)
    # Diagnostic distance comes from the already archived reachability labels,
    # after collection. It never informs collector action choices.
    goal_near = np.zeros(len(visited), bool)
    for seed in np.unique(data["map_seeds"]):
        rows = np.flatnonzero(data["map_seeds"] == seed)
        positions = data["positions"][rows]
        for position in np.unique(positions, axis=0):
            local = rows[np.all(positions == position, axis=1)]
            possible = local[data["winnable"][local]]
            if len(possible) and data["remaining"][possible].min() <= 2:
                goal_near[local] = True
    def fraction(mask):
        n = int(mask.sum())
        winnable = mask & data["winnable"]
        near = mask & goal_near
        return {"states": n, "visited_states": int((mask & visited).sum()),
            "coverage_rate": float(visited[mask].mean()) if n else None,
            "winnable_states": int(winnable.sum()), "visited_winnable_states": int((winnable & visited).sum()),
            "winnable_coverage_rate": float(visited[winnable].mean()) if winnable.any() else None,
            "goal_near_states": int(near.sum()), "visited_goal_near_states": int((near & visited).sum()),
            "goal_near_coverage_rate": float(visited[near].mean()) if near.any() else None,
            "visits": int(visited_counts[mask].sum())}
    overall = fraction(np.ones(len(visited), bool))
    live_indices = transitions["successor_indices"][support][~transitions["ends"][support]]
    outside = int((~visited[live_indices]).sum())
    return {"total_training_states": len(visited), "unique_current_states": len(support),
        "current_state_fraction": overall["coverage_rate"], "winnable_current_state_fraction": overall["winnable_coverage_rate"],
        "goal_near_current_state_fraction": overall["goal_near_coverage_rate"], "goal_near_definition": "shortest-path distance at most 2 moves, all remaining clocks",
        "overall": overall, "by_map": [{"map_seed": int(seed), **fraction(data["map_seeds"] == seed)} for seed in np.unique(data["map_seeds"])],
        "by_time_bucket": [{"time_bucket": bucket, **fraction((data["remaining"] >= low) & (data["remaining"] <= high))}
                            for bucket, (low, high) in BUCKETS.items()],
        "successor_queries": {"all_action_transitions": len(support) * 4, "nonterminal_transitions": len(live_indices),
            "outside_support_nonterminal_transitions": outside,
            "unique_nonterminal_destinations": int(len(np.unique(live_indices))),
            "unique_outside_support_destinations": int(len(np.unique(live_indices[~visited[live_indices]]))),
            "outside_support_fraction": outside / len(live_indices) if len(live_indices) else None,
            "denominator": "all four nonterminal transitions from unique visited current states, before training"}}


def select_coverage_layouts(config, train_seeds, fresh_count, fresh_start, prior_fresh, earlier_fresh, deadline=float("inf")):
    result = select_fixed_layouts(config, train_seeds, fresh_count, fresh_start, prior_fresh + earlier_fresh, deadline)
    previous = {r["layout_hash"] for r in prior_fresh}
    for row in result["collision_skips"]:
        if row["reason"] == "previous_supervised_fresh_layout":
            row["reason"] = "previous_fixed_targets_fresh_layout" if row["layout_hash"] in previous else "previous_supervised_fresh_layout"
    return result


def collect_support(config, data, transitions, output, *, episodes_per_map=16, deadline=float("inf"), enforce=None):
    """Collect once with random actions; no learner or oracle is consulted."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    enforce = enforce or (lambda: check_budget(deadline))
    visited = np.zeros(len(data["observations"]), np.uint32)
    lookup = {(int(seed), tuple(map(int, position)), int(remaining)): i
              for i, (seed, position, remaining) in enumerate(zip(data["map_seeds"], data["positions"], data["remaining"]))}
    episodes, total_steps, status, reason = [], 0, "complete", None
    env = CollectionWorld(config)
    current = None
    with (output / "collection_steps.csv").open("w", newline="") as sf, (output / "collection_episodes.csv").open("w", newline="") as ef:
        sw, ew = csv.DictWriter(sf, COLLECTION_STEP_FIELDS), csv.DictWriter(ef, COLLECTION_EPISODE_FIELDS)
        sw.writeheader()
        ew.writeheader()
        try:
            for seed in np.unique(data["map_seeds"]):
                for repetition in range(episodes_per_map):
                    enforce()
                    env.reset(seed=int(seed))
                    rng = np.random.default_rng(np.random.SeedSequence([int(seed), repetition, 77301]))
                    current = {"map_seed": int(seed), "repetition": repetition, "steps": 0, "success": 0,
                        "terminated": 0, "truncated": 0, "complete": 0, "base_return": 0., "shaped_return": 0., "noop_steps": 0}
                    while not (env.terminated or env.truncated):
                        enforce()
                        remaining = config.horizon - env.elapsed
                        row = lookup[(int(seed), env.position, remaining)]
                        action = int(rng.integers(4))
                        before = env.position
                        _, reward, terminated, truncated, info = env.step(action)
                        following = -1 if terminated or truncated else lookup[(int(seed), env.position, config.horizon - env.elapsed)]
                        if (abs(reward - transitions["rewards"][row, action]) > 1e-12 or
                                bool(terminated or truncated) != bool(transitions["ends"][row, action]) or
                                following != int(transitions["successor_indices"][row, action])):
                            raise ConsistencyError("collected transition differs from archived transition bank")
                        visited[row] += 1
                        total_steps += 1
                        current["steps"] += 1
                        current["shaped_return"] += reward
                        current["base_return"] += info["base_reward"]
                        current["noop_steps"] += env.position == before
                        current.update(success=int(terminated), terminated=int(terminated), truncated=int(truncated))
                        sw.writerow({"map_seed": int(seed), "repetition": repetition, "step": current["steps"], "current_row": row,
                            "action": action, "reward": reward, "next_row": following, "terminated": int(terminated),
                            "truncated": int(truncated), "remaining": remaining})
                    current["complete"] = 1
                    ew.writerow(current)
                    episodes.append(current)
                    current = None
        except (BudgetReached, ConsistencyError) as exc:
            status = ("inconsistent_not_gate_evidence" if isinstance(exc, ConsistencyError) else
                      "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap")
            reason = str(exc)
            if current is not None:
                ew.writerow(current)
                episodes.append(current)
    arrays = {"visited_counts": visited, "support_indices": np.flatnonzero(visited).astype(np.int32)}
    np.savez_compressed(output / "collection.npz", **arrays)
    coverage = {"status": status, "stop_reason": reason, "collection_policy": "uniform random actions",
        "collection_rng": "SeedSequence([map_seed,repetition,77301])", "episodes_per_map": episodes_per_map,
        "collection_episodes": len(episodes), "complete_collection_episodes": sum(r["complete"] for r in episodes),
        "collection_steps": total_steps, "collection_successes": sum(r["success"] for r in episodes),
        "collection_noop_steps": sum(r["noop_steps"] for r in episodes),
        "arrays": array_metadata(arrays), **coverage_metrics(data, transitions, visited)}
    write_json(output / "coverage.json", coverage)
    return coverage, arrays


def run_study(output, protocol_file, *, seeds=(0, 1, 2), updates=30000, train_maps=256, fresh_maps=64,
              train_seed_start=300000, fresh_seed_start=950000, max_seconds=1200, max_rss_bytes=4 * 1024**3,
              dashboard=None, prior_dir=None, supervised_dir=None, collection_episodes_per_map=16, smoke=False, checkpoints=None):
    started = time.monotonic()
    output, protocol_file = Path(output), Path(protocol_file)
    prior_dir = Path(prior_dir) if prior_dir is not None else ROOT / "experiments/fixed_targets/pilot_v1"
    supervised_dir = Path(supervised_dir) if supervised_dir is not None else ROOT / "experiments/supervised/pilot_v1"
    seeds = list(seeds)
    if not protocol_file.is_file():
        raise ValueError("an existing predeclared protocol is required")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("distinct nonnegative learner seeds are required")
    if min(updates, train_maps, fresh_maps, max_seconds, max_rss_bytes, collection_episodes_per_map) <= 0:
        raise ValueError("positive budgets are required")
    schedule = sorted(set(checkpoints if checkpoints is not None else [c for c in CHECKPOINTS if c <= updates] + [updates]))
    if schedule[0] != 0 or schedule[-1] != updates or min(schedule) < 0:
        raise ValueError("checkpoints must span zero through the update budget")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    prior_metadata = json.loads((prior_dir / "dataset_metadata.json").read_text())
    prior_sampling = json.loads((prior_dir / "sampling.json").read_text())
    prior_results = json.loads((prior_dir / "results.json").read_text())
    if prior_results["run"]["status"] != "complete":
        raise ValueError("a complete archived fixed-target study is required")
    earlier_metadata = json.loads((supervised_dir / "dataset_metadata.json").read_text())
    config = WorldConfig()
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "platform": platform.platform(), "machine": platform.machine(), "torch_threads": 1, "device": "cpu"}
    actual = dict(seeds=seeds, updates=updates, train_maps=train_maps, fresh_maps=fresh_maps,
                  train_seed_start=train_seed_start, fresh_seed_start=fresh_seed_start, max_seconds=max_seconds,
                  max_rss_bytes=max_rss_bytes, checkpoints=schedule, collection_episodes_per_map=collection_episodes_per_map)
    declared = dict(seeds=[0, 1, 2], updates=30000, train_maps=256, fresh_maps=64, train_seed_start=300000,
                    fresh_seed_start=950000, max_seconds=1200, max_rss_bytes=4 * 1024**3, checkpoints=list(CHECKPOINTS), collection_episodes_per_map=16)
    deviations = [{"field": k, "actual": v, "declared": declared[k]} for k, v in actual.items() if v != declared[k]]
    if not runtime["python"].startswith("3.12.") or runtime["torch"].split("+")[0] != "2.8.0" or runtime["numpy"] != "2.0.2":
        deviations.append({"field": "runtime", "actual": runtime, "declared": "Python3.12/Torch2.8.0/NumPy2.0.2"})
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    protocol = {"id": "coverage-v1", "question": "With the target procedure fixed, how does collected current-state support compare with exhaustive coverage?",
        "privileged_offline_coverage": True, "rule_visibility": "observed", "seeds": seeds, "checkpoints": schedule,
        "checkpoint_units": "optimizer updates", "conditions": [{"id": "exhaustive", "label": "Exhaustive support"},
            {"id": "collected_unique", "label": "Collected unique support"}], "world": asdict(config), "runtime": runtime,
        "git": git_info(), "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
        "protocol_sha256": sha(protocol_file), "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(smoke), "deviations": deviations,
        "budget": {"maximum_updates": 2 * len(seeds) * updates, "updates_per_seed": updates, "batch_size": 64,
            "admission_seconds": max_seconds, "peak_process_rss_bytes": max_rss_bytes, "dataset_and_evaluation_included": True, "collection_included": True,
            "maximum_collection_episodes": train_maps * collection_episodes_per_map,
            "maximum_collection_steps": train_maps * collection_episodes_per_map * config.horizon},
        "dataset": {"train_map_seeds": list(range(train_seed_start, train_seed_start + train_maps)),
            "heldout_map_seeds": [], "train_states": 0, "heldout_states": 0, "collision_skips": []},
        "optimizer": {"name": "Adam", "learning_rate": .001, "loss": "all-four-action SmoothL1 mean", "beta": 1,
            "gradient_norm_cap": 5., "sampling_rng": "SeedSequence([seed,66301])", "batch_size": 64,
            "paired_batch_schedule": False, "sampling": "uniform distinct local support positions mapped to global rows", "double_dqn_gamma": .97, "target_tau": .01,
            "target_update_timing": "after every optimizer step in both arms"},
        "gates": {"training_success": .9, "training_state_optimal": .9, "heldout_success": .7,
            "efficient_success": .8, "planner_step_multiplier": 2, "scope": "each seed, final greedy; fresh above pooled random"},
        "collection": {"policy": "uniform random actions", "episodes_per_map": collection_episodes_per_map,
            "rng": "SeedSequence([map_seed,repetition,77301])", "order": "map ascending then repetition ascending",
            "support": "sorted unique pre-action current-state global row IDs; visitation frequency is not sampling weight",
            "successor_access": "full training bank for detached queries; outside-support successors never enter current-state support"},
        "evaluation": {"greedy_repetitions": 1, "epsilon_0_1_repetitions": 2,
            "rng": "competence.evaluation_draws excludes condition/checkpoint", "tie_atol": TIE_ATOL, "tie_rtol": 0}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    write_json(output / "protocol.json", protocol)
    shutil.copy2(protocol_file, output / "protocol.md")
    for path in source_files:
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    (output / "environment.txt").write_text("\n".join(packages) + "\n")
    command = ["python", "-m", "q6.coverage", "--output", str(output), "--protocol-file", str(protocol_file),
        "--prior-dir", artifact_path(prior_dir), "--supervised-dir", artifact_path(supervised_dir),
        "--collection-episodes-per-map", str(collection_episodes_per_map), "--seeds", ",".join(map(str, seeds)), "--updates", str(updates),
        "--train-maps", str(train_maps), "--fresh-maps", str(fresh_maps), "--train-seed-start", str(train_seed_start),
        "--fresh-seed-start", str(fresh_seed_start), "--max-seconds", str(max_seconds), "--max-rss-bytes", str(max_rss_bytes),
        "--checkpoints", ",".join(map(str, schedule))]
    if smoke:
        command.append("--smoke")
    (output / "command.txt").write_text(shlex.join(command) + "\n# Reproduction needs a new output path.\n")
    provenance = {"archive": artifact_path(prior_dir), "files": {name: sha(prior_dir / name) for name in
        ("dataset.npz", "transitions.npz", "dataset_metadata.json", "sampling.json", "results.json", "protocol.json", "manifest.json")},
        "archive_initial_policy_hashes": prior_results["run"]["initial_policy_hashes"], "exhaustive_replication": [],
        "training_arrays_identical": None, "transition_arrays_identical": None, "earlier_supervised_archive": artifact_path(supervised_dir),
        "earlier_supervised_metadata_sha256": sha(supervised_dir / "dataset_metadata.json"), "replication_required": not smoke and not deviations}
    for name in ("dataset_metadata.json", "sampling.json", "protocol.json"):
        shutil.copy2(prior_dir / name, output / ("prior_" + name))
    shutil.copy2(supervised_dir / "dataset_metadata.json", output / "earlier_supervised_dataset_metadata.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    deadline = started + max_seconds
    resource_checks = 0
    def enforce():
        nonlocal resource_checks
        resource_checks += 1
        guard(deadline, max_rss_bytes)
    oracle = VisibleOptimalQ(config, max_cached_maps=train_maps + fresh_maps)
    data, selection, metadata = {}, {}, {"status": "not_generated"}
    coverage, collection_arrays, supports = {"status": "not_generated"}, {}, {}
    all_rows, state_rows, reference_rows, trajectories, losses, progress = [], [], [], [], [], []
    errors, initial_hashes, samplers, samples, timings = {}, {}, {}, {}, {}
    dataset_seconds = reference_seconds = collection_seconds = 0.
    total_updates, model_parameters, status = 0, None, "complete"
    active_window, active_key = [], None
    with (output / "evaluations.csv").open("w", newline="") as ef, (output / "state_metrics.csv").open("w", newline="") as sf, \
         (output / "references.csv").open("w", newline="") as rf, (output / "losses.csv").open("w", newline="") as lf:
        ew, sw = csv.DictWriter(ef, EVAL_FIELDS), csv.DictWriter(sf, ["condition", *STATE_FIELDS])
        rw, lw = csv.DictWriter(rf, EVAL_FIELDS), csv.DictWriter(lf, ["condition", *LOSS_FIELDS])
        for writer in (ew, sw, rw, lw):
            writer.writeheader()
        try:
            phase = time.monotonic()
            try:
                enforce()
                selection = select_coverage_layouts(config, protocol["dataset"]["train_map_seeds"], fresh_maps,
                                                 fresh_seed_start, prior_metadata["heldout"], earlier_metadata["heldout"], deadline)
                metadata = {"status": "generating", **selection, "complete_panels": [],
                    "enumeration_order": "map selection order, row-major non-wall/non-pellet position, remaining1..32 innermost"}
                for panel in ("train", "heldout"):
                    data[panel] = enumerate_panel(config, selection[panel], oracle, deadline)
                    metadata["complete_panels"].append(panel)
                    enforce()
                arrays = {f"{panel}_{k}": v for panel, values in data.items() for k, v in values.items()}
                metadata["arrays"] = array_metadata(arrays)
                np.savez_compressed(output / "dataset.npz", **arrays)
                transitions = build_transitions(config, data["train"], deadline)
                np.savez_compressed(output / "transitions.npz", **transitions)
                metadata["transition_arrays"] = array_metadata(transitions)
                metadata["transition_validation"] = validate_transitions(config, data["train"], transitions, deadline)
                train_hashes = {hashlib.sha256(v.tobytes()).digest() for v in data["train"]["observations"]}
                fresh_hashes = {hashlib.sha256(v.tobytes()).digest() for v in data["heldout"]["observations"]}
                metadata["split_check"] = {"layout_intersections": len({r["layout_hash"] for r in selection["train"]} &
                    {r["layout_hash"] for r in selection["heldout"]}), "observation_intersections": len(train_hashes & fresh_hashes),
                    "previous_fixed_targets_fresh_layout_intersections": len({r["layout_hash"] for r in prior_metadata["heldout"]} &
                    {r["layout_hash"] for r in selection["heldout"]}),
                    "previous_supervised_fresh_layout_intersections": len({r["layout_hash"] for r in earlier_metadata["heldout"]} &
                    {r["layout_hash"] for r in selection["heldout"]})}
                del train_hashes, fresh_hashes
                if any(metadata["split_check"].values()):
                    raise ConsistencyError("layout/observation leakage detected")
                provenance["training_arrays_identical"] = all(metadata["arrays"][f"train_{k}"] == prior_metadata["arrays"][f"train_{k}"] for k in data["train"])
                if provenance["replication_required"] and not provenance["training_arrays_identical"]:
                    raise ConsistencyError("training bank differs from the archived fixed-target bank")
                provenance["transition_arrays_identical"] = metadata["transition_arrays"] == prior_metadata["transition_arrays"]
                if provenance["replication_required"] and not provenance["transition_arrays_identical"]:
                    raise ConsistencyError("transition bank differs from the archived fixed-target bank")
                metadata["status"] = "complete"
                protocol["dataset"].update(heldout_map_seeds=[r["map_seed"] for r in selection["heldout"]],
                    train_states=len(data["train"]["observations"]), heldout_states=len(data["heldout"]["observations"]),
                    collision_skips=selection["collision_skips"])
                write_json(output / "dataset_metadata.json", metadata)
                write_json(output / "protocol.json", protocol)
                train_obs = torch.from_numpy(data["train"]["observations"])
                exact_targets = torch.from_numpy(data["train"]["targets"].astype(np.float32))
                transition_tensors = {k: torch.from_numpy(v.astype(np.float32) if k == "rewards" else v) for k, v in transitions.items()}
            finally:
                dataset_seconds += time.monotonic() - phase
            phase = time.monotonic()
            try:
                coverage, collection_arrays = collect_support(config, data["train"], transitions, output,
                    episodes_per_map=collection_episodes_per_map, deadline=deadline, enforce=enforce)
                supports = {"exhaustive": np.arange(len(train_obs), dtype=np.int32),
                            "collected_unique": collection_arrays["support_indices"]}
                if coverage["status"] == "inconsistent_not_gate_evidence":
                    raise ConsistencyError(coverage["stop_reason"])
                if coverage["status"] != "complete":
                    raise MemoryReached(coverage["stop_reason"]) if coverage["status"] == "incomplete_memory_cap" else BudgetReached(coverage["stop_reason"])
                if len(supports["collected_unique"]) < 64:
                    raise ConsistencyError("configuration failure: collected support cannot supply 64 distinct current states")
            finally:
                collection_seconds += time.monotonic() - phase
            for condition in CONDITIONS:
                for seed in seeds:
                    enforce()
                    active_key = (condition, seed)
                    agent = DQN(train_obs.shape[1], seed=seed)
                    initial_hashes.setdefault(condition, {})[str(seed)] = agent.parameter_hash()
                    model_parameters = sum(p.numel() for p in agent.online.parameters())
                    sampler = samplers[active_key] = SupportSampler(supports[condition], len(train_obs), seed)
                    timing = timings[f"{condition}:seed{seed}"] = {"training_wall_seconds": 0., "evaluation_wall_seconds": 0.}
                    active_window = []
                    for checkpoint in schedule:
                        phase = time.monotonic()
                        try:
                            while sampler.updates < checkpoint:
                                enforce()
                                loss = fixed_update(agent, train_obs, exact_targets, transition_tensors, sampler.next_batch(), "double_dqn")
                                total_updates += 1
                                active_window.append(loss)
                                if sampler.updates % 100 == 0 or sampler.updates == checkpoint:
                                    row = {"condition": condition, "seed": seed, "checkpoint": sampler.updates,
                                        "updates_in_window": len(active_window), "mean_loss": float(np.mean(active_window)),
                                        "last_loss": loss, "training_examples": sampler.updates * 64}
                                    losses.append(row)
                                    lw.writerow(row)
                                    active_window = []
                        finally:
                            timing["training_wall_seconds"] += time.monotonic() - phase
                            lf.flush()
                        torch.save({"online": agent.online.state_dict(), "target": agent.target.state_dict(),
                            "observation_size": agent.observation_size, "parameter_hash": agent.parameter_hash(),
                            "target_parameter_hash": module_hash(agent.target), "optimizer_updates": checkpoint,
                            "purpose": "inference_only_not_resumable"}, output / "models" / f"{condition}_seed{seed}_update{checkpoint}.pt")
                        cp_rows, cp_states, cp_trajectories, cp_errors, final_predictions = [], [], [], {}, {}
                        phase = time.monotonic()
                        try:
                            for panel in ("train", "heldout"):
                                enforce()
                                rows, predictions, per_errors = full_state_metrics(agent, data[panel], seed=seed,
                                    checkpoint=checkpoint, panel=panel, deadline=deadline)
                                cp_states.extend({"condition": condition, **row} for row in rows)
                                cp_errors[(condition, seed, checkpoint, panel)] = per_errors
                                if checkpoint == updates:
                                    final_predictions[panel] = predictions
                                for mode in ("greedy", "epsilon_0_1"):
                                    for layout in selection[panel]:
                                        enforce()
                                        for rep in range(1 if mode == "greedy" else 2):
                                            row, trajectory = rollout(agent, config, oracle, condition=condition, seed=seed,
                                                checkpoint=checkpoint, mode=mode, panel=panel, map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                            cp_rows.append(row)
                                            if layout == selection[panel][0] and rep == 0:
                                                cp_trajectories.append(trajectory)
                        except BudgetReached:
                            for row in cp_rows:
                                row["checkpoint_complete"] = 0
                            ew.writerows(cp_rows)
                            all_rows.extend(cp_rows)
                            progress.append({"condition": condition, "seed": seed, "checkpoint": checkpoint, "evaluation_complete": False})
                            raise
                        finally:
                            timing["evaluation_wall_seconds"] += time.monotonic() - phase
                        ew.writerows(cp_rows)
                        sw.writerows(cp_states)
                        all_rows.extend(cp_rows)
                        state_rows.extend(cp_states)
                        errors.update(cp_errors)
                        trajectories.extend(cp_trajectories)
                        progress.append({"condition": condition, "seed": seed, "checkpoint": checkpoint, "evaluation_complete": True})
                        if final_predictions:
                            np.savez_compressed(output / f"predictions_{condition}_seed{seed}.npz", **final_predictions)
                        ef.flush()
                        sf.flush()
                        rates = [np.mean([r["success"] for r in cp_rows if r["panel"] == panel and r["mode"] == "greedy"]) for panel in ("train", "heldout")]
                        print(f"{condition} seed={seed} update={checkpoint}: train={rates[0]:.3f} fresh={rates[1]:.3f}", flush=True)
                    if condition == "exhaustive":
                        archive_model = prior_dir / "models" / f"double_dqn_seed{seed}_update30000.pt"
                        check = {"seed": seed, "applicable": provenance["replication_required"], "final_weights_identical": None,
                                 "batch_index_sha256_identical": None, "initial_weights_identical": None}
                        if provenance["replication_required"]:
                            previous = torch.load(archive_model, map_location="cpu", weights_only=True)
                            check.update(archive_model=artifact_path(archive_model), archive_model_sha256=sha(archive_model),
                                final_weights_identical=agent.parameter_hash() == previous["parameter_hash"],
                                batch_index_sha256_identical=sampler.digest.hexdigest() == prior_sampling["double_dqn"][str(seed)]["batch_index_sha256"],
                                initial_weights_identical=initial_hashes[condition][str(seed)] == prior_results["run"]["initial_policy_hashes"]["double_dqn"][str(seed)])
                            if not all(check[k] for k in ("final_weights_identical", "batch_index_sha256_identical", "initial_weights_identical")):
                                provenance["exhaustive_replication"].append(check)
                                raise ConsistencyError("exhaustive arm failed archived deterministic replication")
                        provenance["exhaustive_replication"].append(check)
            phase = time.monotonic()
            try:
                for policy in ("random_actions", "shortest_path"):
                    for seed in ([0] if policy == "shortest_path" else seeds):
                        for panel in ("train", "heldout"):
                            for layout in selection[panel]:
                                enforce()
                                for rep in range(1 if policy == "shortest_path" else 2):
                                    row, trajectory = rollout(None, config, oracle, condition="shared", policy=policy, seed=seed,
                                        checkpoint=0, mode="reference", panel=panel, map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                    row["reference_sample_id"] = f"{policy}:{seed}:{layout['map_seed']}:{rep}"
                                    reference_rows.append(row)
                                    rw.writerow(row)
                                    if seed == (0 if policy == "shortest_path" else seeds[0]) and rep == 0 and layout == selection[panel][0]:
                                        trajectories.append(trajectory)
            finally:
                reference_seconds += time.monotonic() - phase
            enforce()
        except (BudgetReached, ConsistencyError) as exc:
            status = ("inconsistent_not_gate_evidence" if isinstance(exc, ConsistencyError) else
                      "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap")
            provenance["stop_reason"] = str(exc)
            if active_window and active_key is not None:
                row = {"condition": active_key[0], "seed": active_key[1], "checkpoint": sampler.updates,
                    "updates_in_window": len(active_window), "mean_loss": float(np.mean(active_window)),
                    "last_loss": active_window[-1], "training_examples": sampler.updates * 64}
                losses.append(row)
                lw.writerow(row)
    for (condition, seed), sampler in samplers.items():
        samples.setdefault(condition, {})[str(seed)] = {"updates": sampler.updates, "examples_seen": int(sampler.counts.sum()),
            "unique_states_sampled": int(np.count_nonzero(sampler.counts)), "batch_index_sha256": sampler.digest.hexdigest(), "global_batch_index_sha256": sampler.digest.hexdigest(),
            "local_batch_index_sha256": sampler.local.digest.hexdigest(), "support_states": len(sampler.support), "rng_seed_tuple": [seed, 66301]}
        live = ~transitions["ends"]
        visited = collection_arrays["visited_counts"] > 0
        outside = live & ~visited[np.maximum(transitions["successor_indices"], 0)]
        queries = int(np.dot(sampler.counts.astype(np.uint64), live.sum(axis=1).astype(np.uint64)))
        external = int(np.dot(sampler.counts.astype(np.uint64), outside.sum(axis=1).astype(np.uint64)))
        samples[condition][str(seed)]["successor_queries"] = {"nonterminal_queries": queries,
            "outside_collected_support_queries": external, "outside_collected_support_fraction": external / queries if queries else None}
    provenance["paired_consistency"] = []
    for seed in seeds:
        a, b = samplers.get(("exhaustive", seed)), samplers.get(("collected_unique", seed))
        check = {"seed": seed, "complete": a is not None and b is not None and a.updates == b.updates == updates,
            "initial_weights_identical": initial_hashes.get("exhaustive", {}).get(str(seed)) == initial_hashes.get("collected_unique", {}).get(str(seed))
                if a is not None and b is not None else None,
            "same_update_count": a.updates == b.updates if a is not None and b is not None else None,
            "same_batch_indices_required": False}
        provenance["paired_consistency"].append(check)
        if status == "complete" and not all(check[k] for k in ("complete", "initial_weights_identical", "same_update_count")):
            status = "inconsistent_not_gate_evidence"
            provenance["stop_reason"] = "paired initialization or update-count consistency failed"
    np.savez_compressed(output / "sample_counts.npz", **{f"{c}_seed{s}": v.counts for (c, s), v in samplers.items()})
    np.savez_compressed(output / "local_sample_counts.npz", **{f"{c}_seed{s}": v.local.counts for (c, s), v in samplers.items()})
    if metadata["status"] != "complete":
        metadata["status"] = status
    for name, value in (("sampling", samples), ("dataset_metadata", metadata), ("protocol", protocol), ("provenance", provenance)):
        write_json(output / f"{name}.json", value)
    planner_steps = {(r["panel"], r["map_seed"]): r["steps"] for r in reference_rows if r["policy"] == "shortest_path"}
    completed = [r for r in all_rows if r["checkpoint_complete"]]
    def summary(rows):
        successful = [r for r in rows if r["success"]]
        successful_steps = sum(r["steps"] for r in successful)
        return {**summarize(rows), "successful_episodes": len(successful), "successful_episode_steps": successful_steps,
                "successful_noop_steps": sum(r["noop_steps"] for r in successful),
                "successful_mean_steps": successful_steps / len(successful) if successful else None,
                "successful_noop_rate": sum(r["noop_steps"] for r in successful) / successful_steps if successful_steps else None,
                "efficient_success_rate": efficient_rate(rows, planner_steps)
                if all((r["panel"], r["map_seed"]) in planner_steps for r in rows) else None}
    seed_results, aggregate, state_aggregate, references, loss_aggregate = [], [], [], [], []
    for condition in CONDITIONS:
        for checkpoint in schedule:
            for panel in ("train", "heldout"):
                for mode in ("greedy", "epsilon_0_1"):
                    selected = [r for r in completed if (r["condition"], r["checkpoint"], r["panel"], r["mode"]) == (condition, checkpoint, panel, mode)]
                    if selected:
                        rates = []
                        for seed in sorted({r["seed"] for r in selected}):
                            item = {"condition": condition, "seed": seed, "checkpoint": checkpoint, "panel": panel, "mode": mode,
                                    **summary([r for r in selected if r["seed"] == seed])}
                            seed_results.append(item)
                            rates.append(item["success_rate"])
                        aggregate.append({"condition": condition, "checkpoint": checkpoint, "panel": panel, "mode": mode,
                            **summary(selected), "seeds": len(rates), "seed_success_min": min(rates), "seed_success_max": max(rates)})
                for bucket, (low, high) in BUCKETS.items():
                    selected = [r for r in state_rows if (r["condition"], r["checkpoint"], r["panel"], r["time_bucket"]) == (condition, checkpoint, panel, bucket)]
                    if selected:
                        present = sorted({r["seed"] for r in selected})
                        mask = (data[panel]["remaining"] >= low) & (data[panel]["remaining"] <= high)
                        pooled = np.concatenate([errors[(condition, seed, checkpoint, panel)][mask] for seed in present])
                        rates = []
                        for seed in present:
                            items = [r for r in selected if r["seed"] == seed]
                            win = sum(r["winnable_states"] for r in items)
                            if win:
                                rates.append(sum(r["optimal_winnable_actions"] for r in items) / win)
                        state_aggregate.append({"condition": condition, "checkpoint": checkpoint, "panel": panel, "time_bucket": bucket,
                            **reduce_states(selected, pooled), "seeds": len(present), "seed_optimal_action_min": min(rates) if rates else None,
                            "seed_optimal_action_max": max(rates) if rates else None})
        for checkpoint in sorted({r["checkpoint"] for r in losses if r["condition"] == condition}):
            selected = [r for r in losses if r["condition"] == condition and r["checkpoint"] == checkpoint]
            loss_aggregate.append({"condition": condition, "checkpoint": checkpoint, "mean_loss": float(np.mean([r["mean_loss"] for r in selected])),
                "seed_loss_min": min(r["mean_loss"] for r in selected), "seed_loss_max": max(r["mean_loss"] for r in selected),
                "seeds": len(selected), "updates_in_window": selected[0]["updates_in_window"]})
    for policy, panel in sorted({(r["policy"], r["panel"]) for r in reference_rows}):
        references.append({"condition": "shared", "policy": policy, "panel": panel, "mode": "reference",
            **summary([r for r in reference_rows if r["policy"] == policy and r["panel"] == panel])})
    eligible = status == "complete" and not smoke and not deviations
    random_fresh = next((r["success_rate"] for r in references if r["policy"] == "random_actions" and r["panel"] == "heldout"), None)
    per_condition = []
    for condition in CONDITIONS:
        gate_rows = []
        for seed in seeds:
            final = {r["panel"]: r for r in seed_results if r["condition"] == condition and r["seed"] == seed and r["checkpoint"] == updates and r["mode"] == "greedy"}
            states = [r for r in state_rows if r["condition"] == condition and r["seed"] == seed and r["checkpoint"] == updates and r["panel"] == "train" and r["time_bucket"] == "all"]
            win = sum(r["winnable_states"] for r in states)
            optimal = sum(r["optimal_winnable_actions"] for r in states) / win if win else None
            known, fresh = final.get("train", {}).get("success_rate"), final.get("heldout", {}).get("success_rate")
            efficient = final.get("heldout", {}).get("efficient_success_rate")
            gate_rows.append({"seed": seed, "train_success": known, "train_state_optimal": optimal,
                "fresh_success": fresh, "fresh_efficient_success": efficient,
                "training_fit": bool(eligible and known is not None and known >= .9 and optimal is not None and optimal >= .9),
                "fresh": bool(eligible and fresh is not None and fresh >= .7 and random_fresh is not None and fresh > random_fresh),
                "efficient": bool(eligible and efficient is not None and efficient >= .8)})
        per_condition.append({"condition": condition, "per_seed": gate_rows,
            **{key: all(r[key] for r in gate_rows) for key in ("training_fit", "fresh", "efficient")}})
    paired = {"direction": "collected_unique minus exhaustive", "scope": "final greedy, matched seed and layout", "per_seed": [], "per_layout": [], "aggregate": []}
    final_rows = {(r["condition"], r["seed"], r["panel"], r["map_seed"]): r for r in completed if r["checkpoint"] == updates and r["mode"] == "greedy"}
    for seed in seeds:
        for panel in ("train", "heldout"):
            pair = {r["condition"]: r for r in seed_results if r["seed"] == seed and r["panel"] == panel and r["checkpoint"] == updates and r["mode"] == "greedy"}
            if set(pair) == set(CONDITIONS):
                paired["per_seed"].append({"seed": seed, "panel": panel, **{key + "_delta": pair["collected_unique"][key] - pair["exhaustive"][key]
                    if pair["collected_unique"][key] is not None and pair["exhaustive"][key] is not None else None
                    for key in ("success_rate", "mean_steps", "noop_rate", "efficient_success_rate")}})
            for layout in selection.get(panel, []):
                left = final_rows.get(("exhaustive", seed, panel, layout["map_seed"]))
                right = final_rows.get(("collected_unique", seed, panel, layout["map_seed"]))
                if left is not None and right is not None:
                    paired["per_layout"].append({"seed": seed, "panel": panel, "map_seed": layout["map_seed"],
                        "noop_rate_delta": right["noop_steps"] / right["steps"] - left["noop_steps"] / left["steps"],
                        "efficient_success_delta": int(bool(right["success"]) and right["steps"] <= 2 * planner_steps[(panel, layout["map_seed"])])
                            - int(bool(left["success"]) and left["steps"] <= 2 * planner_steps[(panel, layout["map_seed"])])
                            if (panel, layout["map_seed"]) in planner_steps else None,
                        **{key + "_delta": int(right[key]) - int(left[key]) for key in ("success", "steps", "noop_steps")}})
    for panel in ("train", "heldout"):
        selected = [r for r in paired["per_seed"] if r["panel"] == panel]
        if selected:
            paired["aggregate"].append({"panel": panel, "seeds": len(selected), **{f"mean_seed_{key}_delta": float(np.mean([r[f"{key}_delta"] for r in selected]))
                if all(r[f"{key}_delta"] is not None for r in selected) else None for key in ("success_rate", "mean_steps", "noop_rate", "efficient_success_rate")}})
    try:
        enforce()
    except BudgetReached as exc:
        status = "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
        provenance["stop_reason"] = str(exc)
        eligible = False
        for condition_gates in per_condition:
            for key in ("training_fit", "fresh", "efficient"):
                condition_gates[key] = False
                for seed_gates in condition_gates["per_seed"]:
                    seed_gates[key] = False
    write_json(output / "provenance.json", provenance)
    interpretation = ("smoke_or_protocol_deviation_not_gate_evidence" if smoke or deviations else "incomplete_not_gate_evidence" if status != "complete"
        else "both_fresh_gates_met" if all(c["fresh"] for c in per_condition) else "exhaustive_only_fresh_gate_met" if per_condition[0]["fresh"]
        else "collected_only_fresh_gate_met" if per_condition[1]["fresh"] else "neither_fresh_gate_met")
    artifacts = {name: artifact_path(output / filename) for name, filename in {
        "protocol": "protocol.json", "evaluations": "evaluations.csv", "training": "losses.csv", "state_metrics": "state_metrics.csv",
        "dataset": "dataset.npz", "transitions": "transitions.npz", "dataset_metadata": "dataset_metadata.json", "references": "references.csv",
        "sample_counts": "sample_counts.npz", "sampling": "sampling.json", "provenance": "provenance.json", "paired_differences": "paired_differences.json", "manifest": "manifest.json",
        "coverage": "coverage.json", "collection": "collection.npz", "collection_steps": "collection_steps.csv",
        "collection_episodes": "collection_episodes.csv", "local_sample_counts": "local_sample_counts.npz"}.items()}
    artifacts.update(directory=artifact_path(output), protocol_document=artifact_path(protocol_file), report="docs/experiments/coverage_results_v1.md")
    result = {"schema_version": 1, "protocol": protocol, "run": {"id": output.name, "status": status, "interpretation": interpretation,
        "train_updates": total_updates, "training_examples": sum(r["examples_seen"] for c in samples.values() for r in c.values()),
        "wall_seconds": time.monotonic() - started, "dataset_wall_seconds": dataset_seconds, "reference_wall_seconds": reference_seconds, "collection_wall_seconds": collection_seconds,
        "training_wall_seconds": sum(t["training_wall_seconds"] for t in timings.values()),
        "evaluation_wall_seconds": sum(t["evaluation_wall_seconds"] for t in timings.values()), "per_condition_seed_timing": timings,
        "peak_rss_bytes": peak_rss_bytes(), "resource_checks": resource_checks, "resource_limits": {"peak_process_rss_bytes": max_rss_bytes, "seconds": max_seconds, "torch_threads": 1,
            "check_cadence": "each collection step, each optimizer update, each evaluation layout, phase boundaries and after aggregation; sampled stop guard"},
        "model_parameter_count": model_parameters, "reference_episodes": len(reference_rows),
        "unique_reference_episodes": len({r["reference_sample_id"] for r in reference_rows}), "initial_policy_hashes": initial_hashes, "progress": progress,
        "limitations": ["Both arms receive privileged all-action transitions and full-bank detached successor queries; this is not trajectory-only online RL.",
            "Collected current states are sampled uniformly over unique support, not weighted by visitation frequency.",
            "The three initializations share one collection bank; these are not independent collection replications.",
            "A support effect does not isolate online exploration, replay freshness or visitation weighting."]},
        "aggregate": aggregate, "seed_results": seed_results, "state_aggregate": state_aggregate, "state_metrics": artifacts["state_metrics"],
        "loss_aggregate": loss_aggregate, "references": references, "gates": {"eligible": eligible, "pooled_random_fresh_success": random_fresh, "per_condition": per_condition},
        "sampling": samples, "provenance": provenance, "coverage": coverage, "paired_differences": paired, "trajectories": trajectories, "artifacts": artifacts}
    write_json(output / "paired_differences.json", paired)
    write_json(output / "trajectories.json", trajectories)
    write_json(output / "results.json", result)
    save_manifest(output, status)
    if dashboard:
        write_json(Path(dashboard), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--supervised-dir", type=Path)
    parser.add_argument("--collection-episodes-per-map", type=int, default=16)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--train-maps", type=int, default=256)
    parser.add_argument("--fresh-maps", type=int, default=64)
    parser.add_argument("--train-seed-start", type=int, default=300000)
    parser.add_argument("--fresh-seed-start", type=int, default=950000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.updates, args.train_maps, args.fresh_maps, args.seeds = 24, 2, 2, "0"
        args.train_seed_start, args.fresh_seed_start = 720000, 990000
    result = run_study(args.output, args.protocol_file, seeds=[int(v) for v in args.seeds.split(",")], updates=args.updates,
        train_maps=args.train_maps, fresh_maps=args.fresh_maps, train_seed_start=args.train_seed_start, fresh_seed_start=args.fresh_seed_start,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, prior_dir=args.prior_dir, supervised_dir=args.supervised_dir,
        collection_episodes_per_map=args.collection_episodes_per_map, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
