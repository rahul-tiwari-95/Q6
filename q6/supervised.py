"""Bounded exact-Q supervision diagnostic; no replay or temporal-difference learning.

The labels are privileged offline targets. This tests the existing network and
supervised optimizer, and does not establish sample-efficient RL or adaptation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .adaptation import write_json
from .competence import (BudgetReached, EVAL_FIELDS, ROOT, artifact_path, check_budget,
                         evaluate_episode, git_info, summarize)
from .learning import DQN
from .optimal import VisibleOptimalQ
from .world import CollectionWorld, WorldConfig

CHECKPOINTS = (0, 1000, 3000, 10000, 30000)
BUCKETS = {"all": (1, 32), "1-8": (1, 8), "9-16": (9, 16), "17-24": (17, 24), "25-32": (25, 32)}
TIE_ATOL = 1e-6
LOSS_FIELDS = ["seed", "checkpoint", "updates_in_window", "mean_loss", "last_loss", "training_examples"]
STATE_FIELDS = ["seed", "checkpoint", "panel", "map_seed", "time_bucket", "states", "action_values",
                "winnable_states", "impossible_states", "optimal_winnable_actions", "abs_error_sum",
                "squared_error_sum", "signed_error_sum", "action_regret_sum", "winnable_regret_sum",
                "impossible_regret_sum", "mean_abs_q_error", "rmse_q_error", "q95_abs_q_error",
                "mean_signed_q_bias", "optimal_action_rate", "mean_action_regret",
                "winnable_mean_action_regret", "impossible_action_regret"]


def layout_key(env):
    """A labeled layout ignores spawn and clock, but includes goal and rules."""
    payload = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(),
               "config": asdict(env.config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def select_layouts(config, train_seeds, fresh_count, fresh_start, deadline=float("inf")):
    env = CollectionWorld(config)
    train, fresh, skipped = [], [], []
    for seed in train_seeds:
        check_budget(deadline)
        env.reset(seed=int(seed))
        train.append({"map_seed": int(seed), "layout_hash": layout_key(env), "original_start": list(env.position)})
    train_hashes = {row["layout_hash"] for row in train}
    accepted_hashes = set()
    candidate = fresh_start
    while len(fresh) < fresh_count:
        check_budget(deadline)
        env.reset(seed=candidate)
        key = layout_key(env)
        if key in train_hashes or key in accepted_hashes:
            skipped.append({"map_seed": candidate, "layout_hash": key,
                            "reason": "training_layout" if key in train_hashes else "earlier_fresh_layout"})
        else:
            fresh.append({"map_seed": candidate, "layout_hash": key, "original_start": list(env.position)})
            accepted_hashes.add(key)
        candidate += 1
    counts = Counter(row["layout_hash"] for row in train)
    return {"train": train, "heldout": fresh, "collision_skips": skipped,
            "train_duplicate_layouts": {key: count for key, count in counts.items() if count > 1}}


def enumerate_panel(config, layouts, oracle, deadline=float("inf")):
    """Enumerate valid nonterminal position/clock states, independent of rollouts."""
    per_map = (config.size ** 2 - config.wall_count - config.pellet_count) * config.horizon
    count = len(layouts) * per_map
    env = CollectionWorld(config)
    data = {"observations": np.empty((count, env.observation_size), np.float32),
            "targets": np.empty((count, 4), np.float64), "winnable": np.empty(count, bool),
            "map_seeds": np.empty(count, np.int64), "remaining": np.empty(count, np.int16),
            "positions": np.empty((count, 2), np.int16)}
    index = 0
    for layout in layouts:
        check_budget(deadline)
        env.reset(seed=layout["map_seed"])
        for row, column in np.argwhere(~(env.walls | env.pellets)):
            for remaining in range(1, config.horizon + 1):
                env.position = (int(row), int(column))
                env.elapsed = config.horizon - remaining
                observation = env.observe()
                data["observations"][index] = observation
                data["targets"][index] = oracle.q_values(observation)
                data["winnable"][index] = oracle.can_finish(observation)
                data["map_seeds"][index] = layout["map_seed"]
                data["remaining"][index] = remaining
                data["positions"][index] = env.position
                index += 1
    assert index == count
    return data


class BatchSampler:
    def __init__(self, state_count, seed, batch_size=64):
        if not 0 < batch_size <= state_count:
            raise ValueError("batch_size must fit the dataset")
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 66301]))
        self.counts = np.zeros(state_count, np.uint32)
        self.batch_size, self.updates = batch_size, 0
        self.digest = hashlib.sha256()

    def next_batch(self):
        indices = self.rng.choice(len(self.counts), self.batch_size, replace=False)
        self.counts[indices] += 1
        self.digest.update(indices.astype("<i8").tobytes())
        self.updates += 1
        return indices


def supervised_update(agent, observations, targets, indices):
    predicted = agent.online(observations[indices])
    loss = F.smooth_l1_loss(predicted, targets[indices], reduction="mean")
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 5.0)
    agent.optimizer.step()
    return float(loss.item())


def reduce_states(rows, per_state_errors):
    additive = ("states", "action_values", "winnable_states", "impossible_states", "optimal_winnable_actions",
                "abs_error_sum", "squared_error_sum", "signed_error_sum", "action_regret_sum",
                "winnable_regret_sum", "impossible_regret_sum")
    result = {key: sum(row[key] for row in rows) for key in additive}
    n, actions, win, impossible = (result[k] for k in ("states", "action_values", "winnable_states", "impossible_states"))
    result.update(mean_abs_q_error=result["abs_error_sum"] / actions,
                  rmse_q_error=float(np.sqrt(result["squared_error_sum"] / actions)),
                  q95_abs_q_error=float(np.quantile(per_state_errors, 0.95)),
                  mean_signed_q_bias=result["signed_error_sum"] / actions,
                  optimal_action_rate=result["optimal_winnable_actions"] / win if win else None,
                  mean_action_regret=result["action_regret_sum"] / n,
                  winnable_mean_action_regret=result["winnable_regret_sum"] / win if win else None,
                  impossible_action_regret=result["impossible_regret_sum"] / impossible if impossible else None)
    return result


def full_state_metrics(agent, data, *, seed, checkpoint, panel, deadline=float("inf")):
    predictions = np.empty_like(data["targets"], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(predictions), 4096):
            check_budget(deadline)
            predictions[start:start + 4096] = agent.online(torch.from_numpy(data["observations"][start:start + 4096])).numpy()
    error = predictions.astype(np.float64) - data["targets"]
    per_state_errors = np.abs(error).mean(axis=1)
    chosen = predictions.argmax(axis=1)
    regret = np.maximum(0.0, data["targets"].max(axis=1) - data["targets"][np.arange(len(chosen)), chosen])
    optimal = regret <= TIE_ATOL
    rows = []
    for map_seed in np.unique(data["map_seeds"]):
        check_budget(deadline)
        for bucket, (low, high) in BUCKETS.items():
            selected = (data["map_seeds"] == map_seed) & (data["remaining"] >= low) & (data["remaining"] <= high)
            if not selected.any():
                continue
            winnable = selected & data["winnable"]
            impossible = selected & ~data["winnable"]
            raw = {"states": int(selected.sum()), "action_values": int(selected.sum()) * 4,
                   "winnable_states": int(winnable.sum()), "impossible_states": int(impossible.sum()),
                   "optimal_winnable_actions": int((winnable & optimal).sum()),
                   "abs_error_sum": float(np.abs(error[selected]).sum()),
                   "squared_error_sum": float(np.square(error[selected]).sum()),
                   "signed_error_sum": float(error[selected].sum()), "action_regret_sum": float(regret[selected].sum()),
                   "winnable_regret_sum": float(regret[winnable].sum()), "impossible_regret_sum": float(regret[impossible].sum())}
            rows.append({"seed": seed, "checkpoint": checkpoint, "panel": panel, "map_seed": int(map_seed),
                         "time_bucket": bucket, **reduce_states([raw], per_state_errors[selected])})
    return rows, predictions, per_state_errors


def rollout(agent, config, oracle, *, condition="supervised_q", policy="learner", **kwargs):
    row, trajectory = evaluate_episode(agent, config, oracle, condition=condition,
        policy="learner" if policy == "prior_stream" else policy, optimal_tolerance=TIE_ATOL, **kwargs)
    row["policy"] = trajectory["policy"] = policy
    return row, trajectory


def prior_overlap(config, prior_dir, seeds, fresh_layouts, deadline):
    path = prior_dir / "task_exposure.json"
    exposures = [item for item in json.loads(path.read_text()) if item["condition"] == "stream" and item["seed"] in seeds]
    cached = {}
    fresh_hashes = {row["layout_hash"] for row in fresh_layouts}
    env = CollectionWorld(config)
    result = []
    for item in exposures:
        hashes = set()
        for task in item["tasks"]:
            check_budget(deadline)
            map_seed = task["map_seed"]
            if map_seed not in cached:
                env.reset(seed=map_seed)
                cached[map_seed] = layout_key(env)
            hashes.add(cached[map_seed])
        overlap = sorted(hashes & fresh_hashes)
        result.append({"seed": item["seed"], "training_task_ids": len(item["tasks"]),
                       "unique_training_layouts": len(hashes), "fresh_layout_overlap": overlap,
                       "fresh_layout_overlap_count": len(overlap)})
    prior_protocol_path = prior_dir / "protocol.json"
    prior_protocol = json.loads(prior_protocol_path.read_text())
    return {"input": artifact_path(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_protocol": artifact_path(prior_protocol_path),
            "source_protocol_sha256": hashlib.sha256(prior_protocol_path.read_bytes()).hexdigest(),
            "historical_source_sha256": prior_protocol["source_sha256"],
            "historical_source_directory": artifact_path(prior_dir / "source"),
            "panel_unchanged_by_overlap": True, "unique_task_ids_regenerated": len(cached), "seeds": result}


def save_manifest(output, status):
    write_json(output / "manifest.json", {"schema_version": 1, "algorithm": "sha256", "status": status,
        "files": {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}})


def run_study(output, protocol_file, *, seeds=(0, 1, 2), updates=30000, train_maps=256, fresh_maps=64,
              train_seed_start=300000, fresh_seed_start=930000, max_seconds=900, dashboard=None,
              prior_dir=None, smoke=False, checkpoints=None):
    output, protocol_file = Path(output), Path(protocol_file)
    prior_dir = Path(prior_dir) if prior_dir is not None else ROOT / "experiments/competence/pilot_v1"
    seeds = list(seeds)
    if not protocol_file.is_file():
        raise ValueError("an existing predeclared protocol is required")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("distinct nonnegative learner seeds are required")
    if min(updates, train_maps, fresh_maps, max_seconds) <= 0:
        raise ValueError("positive budgets are required")
    schedule = sorted(set(checkpoints if checkpoints is not None else [c for c in CHECKPOINTS if c <= updates] + [updates]))
    if not schedule or schedule[0] != 0 or schedule[-1] != updates or min(schedule) < 0:
        raise ValueError("checkpoints must span zero through the update budget")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    config = WorldConfig()
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "platform": platform.platform(), "machine": platform.machine(), "torch_threads": 1, "device": "cpu"}
    actual = {"seeds": seeds, "updates": updates, "train_maps": train_maps, "fresh_maps": fresh_maps,
              "train_seed_start": train_seed_start, "fresh_seed_start": fresh_seed_start,
              "max_seconds": max_seconds, "checkpoints": schedule}
    declared = {"seeds": [0, 1, 2], "updates": 30000, "train_maps": 256, "fresh_maps": 64,
                "train_seed_start": 300000, "fresh_seed_start": 930000, "max_seconds": 900, "checkpoints": list(CHECKPOINTS)}
    deviations = [{"field": key, "actual": value, "declared": declared[key]} for key, value in actual.items() if value != declared[key]]
    if not runtime["python"].startswith("3.12.") or runtime["torch"].split("+")[0] != "2.8.0" or runtime["numpy"] != "2.0.2":
        deviations.append({"field": "runtime", "actual": runtime, "declared": "Python3.12/Torch2.8.0/NumPy2.0.2"})
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    source_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    protocol_text = protocol_file.read_bytes()
    protocol = {"id": "supervised-v1", "question": "Can the same network learn exact action values on known and fresh layouts?",
        "hypothesis": "Removing exploration and bootstrapped targets may expose a different fitting or transfer limit.",
        "privileged_offline_supervision": True, "seeds": seeds, "checkpoints": schedule, "checkpoint_units": "optimizer updates",
        "conditions": [{"id": "supervised_q", "label": "Supervised network"}, {"id": "prior_stream", "label": "Historical RL stream"}],
        "world": {**asdict(config), "action_mapping": list(config.action_mapping)}, "rule_visibility": "observed", "runtime": runtime, "git": git_info(),
        "source_sha256": source_hashes, "protocol_sha256": hashlib.sha256(protocol_text).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(), "smoke": bool(smoke), "deviations": deviations,
        "budget": {"maximum_updates": len(seeds) * updates, "updates_per_seed": updates, "batch_size": 64,
                   "admission_seconds": max_seconds, "dataset_and_evaluation_included": True},
        "dataset": {"train_map_seeds": list(range(train_seed_start, train_seed_start + train_maps)), "heldout_map_seeds": [],
                    "train_states": 0, "heldout_states": 0, "layout_key_description": "walls + pellet + rules, ignoring spawn and clock",
                    "collision_skips": []},
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "loss": "all-four-action SmoothL1 mean",
                      "gradient_norm_cap": 5.0, "sampling": "64 uniform states without replacement within each batch",
                      "sampling_rng": "SeedSequence([seed,66301])", "replay_or_td_targets": False},
        "gates": {"training_success": 0.9, "training_state_optimal": 0.9, "heldout_success": 0.7,
                  "scope": "each seed, final checkpoint, greedy only; fresh above pooled random"},
        "evaluation": {"greedy_repetitions": 1, "epsilon_0_1_repetitions": 2,
                       "rng": "competence.evaluation_draws; excludes condition/checkpoint", "tie_atol": TIE_ATOL, "tie_rtol": 0},
        "metric_definitions": {"q95_abs_q_error": "95th percentile of per-state mean absolute error across four actions",
             "optimal_action_rate": "optimal greedy actions among winnable states; tie tolerance1e-6",
             "mean_signed_q_bias": "prediction minus exact value, averaged over all four actions and states",
             "mean_action_regret": "unconditional mean exact regret of learned greedy argmax",
             "loss_aggregate": "mean pre-update minibatch SmoothL1 over preceding updates_in_window (normally100)"}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", protocol)
    (output / "protocol.md").write_bytes(protocol_text)
    for path in source_files:
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    (output / "environment.txt").write_text("\n".join(packages) + "\n")
    command = ["python", "-m", "q6.supervised", "--output", str(output), "--protocol-file", str(protocol_file),
               "--prior-dir", artifact_path(prior_dir), "--seeds", ",".join(map(str, seeds)), "--updates", str(updates),
               "--train-maps", str(train_maps), "--fresh-maps", str(fresh_maps), "--train-seed-start", str(train_seed_start),
               "--fresh-seed-start", str(fresh_seed_start), "--max-seconds", str(max_seconds),
               "--checkpoints", ",".join(map(str, schedule))]
    if smoke:
        command.append("--smoke")
    (output / "command.txt").write_text(shlex.join(command) + "\n# Reproduction needs a new output path.\n")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    deadline = started + max_seconds
    oracle = VisibleOptimalQ(config, max_cached_maps=train_maps + fresh_maps)
    data, selection, dataset_metadata = {}, {}, {"status": "not_generated"}
    all_rows, state_rows, reference_rows, trajectories, losses, progress = [], [], [], [], [], []
    errors, initial_hashes, samples = {}, {}, {}
    samplers = {}
    total_updates = 0
    model_parameter_count = None
    dataset_seconds = train_seconds = evaluation_seconds = reference_seconds = 0.0
    prior_metadata = {}
    complete = True
    with (output / "evaluations.csv").open("w", newline="") as eval_file, \
         (output / "state_metrics.csv").open("w", newline="") as state_file, \
         (output / "references.csv").open("w", newline="") as ref_file, \
         (output / "losses.csv").open("w", newline="") as loss_file:
        eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_FIELDS)
        state_writer = csv.DictWriter(state_file, fieldnames=STATE_FIELDS)
        ref_writer = csv.DictWriter(ref_file, fieldnames=EVAL_FIELDS)
        loss_writer = csv.DictWriter(loss_file, fieldnames=LOSS_FIELDS)
        for writer in (eval_writer, state_writer, ref_writer, loss_writer):
            writer.writeheader()
        try:
            dataset_start = time.monotonic()
            try:
                selection = select_layouts(config, protocol["dataset"]["train_map_seeds"], fresh_maps, fresh_seed_start, deadline)
                dataset_metadata = {"status": "generating", **selection, "complete_panels": [],
                    "enumeration_order": "training/fresh map selection order, row-major non-wall non-pellet positions, remaining1..32 increasing innermost"}
                write_json(output / "dataset_metadata.json", dataset_metadata)
                for panel in ("train", "heldout"):
                    data[panel] = enumerate_panel(config, selection[panel], oracle, deadline)
                    dataset_metadata["complete_panels"].append(panel)
                np.savez_compressed(output / "dataset.npz", **{f"{panel}_{key}": value for panel, values in data.items() for key, value in values.items()})
                dataset_metadata.update(status="complete", arrays={f"{panel}_{key}": {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for panel, values in data.items() for key, value in values.items()})
                layout_overlap = {r["layout_hash"] for r in selection["train"]} & {r["layout_hash"] for r in selection["heldout"]}
                observed_hashes = {hashlib.sha256(row.tobytes()).digest() for row in data["train"]["observations"]}
                fresh_hashes = {hashlib.sha256(row.tobytes()).digest() for row in data["heldout"]["observations"]}
                dataset_metadata["split_check"] = {"layout_intersections": len(layout_overlap),
                    "observation_intersections": len(observed_hashes & fresh_hashes),
                    "unique_training_observations": len(observed_hashes), "unique_fresh_observations": len(fresh_hashes)}
                if layout_overlap or observed_hashes & fresh_hashes:
                    raise ValueError("training/fresh leakage detected")
                del observed_hashes, fresh_hashes
                protocol["dataset"].update(heldout_map_seeds=[row["map_seed"] for row in selection["heldout"]],
                    train_states=len(data["train"]["observations"]), heldout_states=len(data["heldout"]["observations"]),
                    collision_skips=selection["collision_skips"])
                write_json(output / "dataset_metadata.json", dataset_metadata)
                write_json(output / "protocol.json", protocol)
                prior_metadata = prior_overlap(config, prior_dir, seeds, selection["heldout"], deadline)
                shutil.copy2(prior_dir / "protocol.json", output / "prior_protocol.json")
                prior_metadata["models"] = {}
                (output / "models").mkdir(exist_ok=True)
                for seed in seeds:
                    source = prior_dir / "models" / f"stream_seed{seed}_step120000.pt"
                    destination = output / "models" / f"prior_stream_seed{seed}.pt"
                    shutil.copy2(source, destination)
                    prior_metadata["models"][str(seed)] = {"source": artifact_path(source), "source_training_steps": 120000,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "saved": destination.relative_to(output).as_posix()}
                write_json(output / "prior_reference_metadata.json", prior_metadata)
                write_json(output / "dataset_metadata.json", dataset_metadata)
                write_json(output / "protocol.json", protocol)
            finally:
                dataset_seconds += time.monotonic() - dataset_start
            train_obs = torch.from_numpy(data["train"]["observations"])
            train_targets = torch.from_numpy(data["train"]["targets"].astype(np.float32))
            for seed in seeds:
                check_budget(deadline)
                agent = DQN(train_obs.shape[1], seed=seed)
                initial_hashes[str(seed)] = agent.parameter_hash()
                model_parameter_count = sum(p.numel() for p in agent.online.parameters())
                sampler = BatchSampler(len(train_obs), seed)
                samplers[seed] = sampler
                window_losses = []
                for checkpoint in schedule:
                    phase_start = time.monotonic()
                    try:
                        while sampler.updates < checkpoint:
                            check_budget(deadline)
                            indices = sampler.next_batch()
                            loss = supervised_update(agent, train_obs, train_targets, indices)
                            total_updates += 1
                            window_losses.append(loss)
                            if sampler.updates % 100 == 0 or sampler.updates == checkpoint:
                                row = {"seed": seed, "checkpoint": sampler.updates, "updates_in_window": len(window_losses),
                                    "mean_loss": float(np.mean(window_losses)), "last_loss": loss, "training_examples": sampler.updates * 64}
                                losses.append(row)
                                loss_writer.writerow(row)
                                window_losses = []
                    finally:
                        train_seconds += time.monotonic() - phase_start
                        loss_file.flush()
                    snapshot = output / "models" / f"supervised_q_seed{seed}_update{checkpoint}.pt"
                    torch.save({"online": agent.online.state_dict(), "observation_size": agent.observation_size,
                                "parameter_hash": agent.parameter_hash(), "optimizer_updates": checkpoint,
                                "purpose": "inference_only_not_resumable"}, snapshot)
                    checkpoint_rows, checkpoint_states, checkpoint_trajectories = [], [], []
                    checkpoint_errors, final_predictions = {}, {}
                    eval_start = time.monotonic()
                    try:
                        for panel in ("train", "heldout"):
                            rows, predictions, per_errors = full_state_metrics(agent, data[panel], seed=seed,
                                checkpoint=checkpoint, panel=panel, deadline=deadline)
                            checkpoint_states.extend(rows)
                            checkpoint_errors[(seed, checkpoint, panel)] = per_errors
                            if checkpoint == updates:
                                final_predictions[panel] = predictions
                            for mode in ("greedy", "epsilon_0_1"):
                                for layout in selection[panel]:
                                    for rep in range(1 if mode == "greedy" else 2):
                                        row, trajectory = rollout(agent, config, oracle, seed=seed, checkpoint=checkpoint,
                                            mode=mode, panel=panel, map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                        checkpoint_rows.append(row)
                                        if layout == selection[panel][0] and rep == 0:
                                            checkpoint_trajectories.append(trajectory)
                    except BudgetReached:
                        for row in checkpoint_rows:
                            row["checkpoint_complete"] = 0
                        eval_writer.writerows(checkpoint_rows)
                        all_rows.extend(checkpoint_rows)
                        progress.append({"seed": seed, "checkpoint": checkpoint, "evaluation_complete": False})
                        raise
                    finally:
                        evaluation_seconds += time.monotonic() - eval_start
                    eval_writer.writerows(checkpoint_rows)
                    state_writer.writerows(checkpoint_states)
                    all_rows.extend(checkpoint_rows)
                    state_rows.extend(checkpoint_states)
                    errors.update(checkpoint_errors)
                    trajectories.extend(checkpoint_trajectories)
                    progress.append({"seed": seed, "checkpoint": checkpoint, "evaluation_complete": True})
                    if final_predictions:
                        np.savez_compressed(output / f"predictions_seed{seed}.npz", **final_predictions)
                    for handle in (eval_file, state_file):
                        handle.flush()
                    rates = [np.mean([r["success"] for r in checkpoint_rows if r["panel"] == panel and r["mode"] == "greedy"])
                             for panel in ("train", "heldout")]
                    print(f"supervised_q seed={seed} update={checkpoint}: train={rates[0]:.3f} fresh={rates[1]:.3f}", flush=True)
            ref_start = time.monotonic()
            try:
                for policy in ("random_actions", "shortest_path", "prior_stream"):
                    for seed in ([0] if policy == "shortest_path" else seeds):
                        prior_agent = None
                        if policy == "prior_stream":
                            loaded = torch.load(output / "models" / f"prior_stream_seed{seed}.pt", map_location="cpu", weights_only=True)
                            prior_agent = DQN(loaded["observation_size"], seed=seed)
                            prior_agent.online.load_state_dict(loaded["online"])
                            if prior_agent.parameter_hash() != loaded["parameter_hash"]:
                                raise ValueError("historical checkpoint parameter hash does not match")
                        for panel in (("heldout",) if policy == "prior_stream" else ("train", "heldout")):
                            modes = ("greedy", "epsilon_0_1") if policy == "prior_stream" else ("reference",)
                            for mode in modes:
                                for layout in selection[panel]:
                                    repetitions = 1 if policy == "shortest_path" or mode == "greedy" else 2
                                    for rep in range(repetitions):
                                        row, trajectory = rollout(prior_agent, config, oracle,
                                            condition="prior_stream" if policy == "prior_stream" else "supervised_q", policy=policy,
                                            seed=seed, checkpoint=0, mode=mode, panel=panel, map_seed=layout["map_seed"], repetition=rep,
                                            deadline=deadline)
                                        row["reference_sample_id"] = f"{policy}:{mode}:{seed}:{layout['map_seed']}:{rep}"
                                        reference_rows.append(row)
                                        ref_writer.writerow(row)
                                        if (policy in ("shortest_path", "prior_stream") or seed == seeds[0]) and rep == 0 and layout == selection[panel][0]:
                                            trajectories.append(trajectory)
            finally:
                reference_seconds += time.monotonic() - ref_start
        except BudgetReached:
            complete = False
            if samplers and window_losses:
                row = {"seed": seed, "checkpoint": sampler.updates, "updates_in_window": len(window_losses),
                    "mean_loss": float(np.mean(window_losses)), "last_loss": window_losses[-1], "training_examples": sampler.updates * 64}
                losses.append(row)
                loss_writer.writerow(row)
    for seed, sampler in samplers.items():
        samples[str(seed)] = {"updates": sampler.updates, "examples_seen": int(sampler.counts.sum()),
                              "unique_states_sampled": int(np.count_nonzero(sampler.counts)),
                              "batch_index_sha256": sampler.digest.hexdigest(), "rng_seed_tuple": [seed, 66301]}
    np.savez_compressed(output / "sample_counts.npz", **{f"seed{seed}": sampler.counts for seed, sampler in samplers.items()})
    write_json(output / "sampling.json", samples)
    if dataset_metadata["status"] != "complete":
        dataset_metadata["status"] = "incomplete_admission_cap"
    write_json(output / "dataset_metadata.json", dataset_metadata)
    write_json(output / "protocol.json", protocol)
    seed_results, aggregate, state_aggregate, references = [], [], [], []
    completed_rows = [row for row in all_rows if row["checkpoint_complete"]]
    for checkpoint in schedule:
        for panel in ("train", "heldout"):
            for mode in ("greedy", "epsilon_0_1"):
                selected = [r for r in completed_rows if (r["checkpoint"], r["panel"], r["mode"]) == (checkpoint, panel, mode)]
                if not selected:
                    continue
                per_seed = []
                for seed in sorted({r["seed"] for r in selected}):
                    summary = {"condition": "supervised_q", "seed": seed, "checkpoint": checkpoint, "panel": panel, "mode": mode,
                               **summarize([r for r in selected if r["seed"] == seed])}
                    seed_results.append(summary)
                    per_seed.append(summary["success_rate"])
                aggregate.append({"condition": "supervised_q", "checkpoint": checkpoint, "panel": panel, "mode": mode,
                    **summarize(selected), "seeds": len(per_seed), "seed_success_min": min(per_seed), "seed_success_max": max(per_seed)})
            for bucket, (low, high) in BUCKETS.items():
                selected = [r for r in state_rows if (r["checkpoint"], r["panel"], r["time_bucket"]) == (checkpoint, panel, bucket)]
                if not selected:
                    continue
                present_seeds = sorted({r["seed"] for r in selected})
                mask = (data[panel]["remaining"] >= low) & (data[panel]["remaining"] <= high)
                pooled_errors = np.concatenate([errors[(seed, checkpoint, panel)][mask] for seed in present_seeds])
                per_seed_rates = []
                for seed in present_seeds:
                    items = [r for r in selected if r["seed"] == seed]
                    win = sum(r["winnable_states"] for r in items)
                    if win:
                        per_seed_rates.append(sum(r["optimal_winnable_actions"] for r in items) / win)
                state_aggregate.append({"condition": "supervised_q", "checkpoint": checkpoint, "panel": panel, "time_bucket": bucket,
                    **reduce_states(selected, pooled_errors), "seeds": len(present_seeds),
                    "seed_optimal_action_min": min(per_seed_rates) if per_seed_rates else None,
                    "seed_optimal_action_max": max(per_seed_rates) if per_seed_rates else None})
    for policy, panel, mode in sorted({(r["policy"], r["panel"], r["mode"]) for r in reference_rows}):
        selected = [r for r in reference_rows if (r["policy"], r["panel"], r["mode"]) == (policy, panel, mode)]
        references.append({"condition": "prior_stream" if policy == "prior_stream" else "supervised_q",
            "policy": policy, "panel": panel, "mode": mode, **summarize(selected)})
    eligible = complete and not smoke and not deviations
    random_fresh = next((r["success_rate"] for r in references if r["policy"] == "random_actions" and r["panel"] == "heldout"), None)
    gate_rows = []
    for seed in seeds:
        def success(panel):
            return next((r["success_rate"] for r in seed_results if r["seed"] == seed and r["checkpoint"] == updates
                         and r["mode"] == "greedy" and r["panel"] == panel), None)
        known = [r for r in state_rows if r["seed"] == seed and r["checkpoint"] == updates and r["panel"] == "train" and r["time_bucket"] == "all"]
        win = sum(r["winnable_states"] for r in known)
        state_optimal = sum(r["optimal_winnable_actions"] for r in known) / win if win else None
        train_success, fresh_success = success("train"), success("heldout")
        gate_rows.append({"seed": seed, "train_success": train_success, "train_state_optimal": state_optimal, "fresh_success": fresh_success,
            "training_fit": bool(eligible and train_success is not None and train_success >= 0.9 and state_optimal is not None and state_optimal >= 0.9),
            "fresh": bool(eligible and fresh_success is not None and fresh_success >= 0.7 and random_fresh is not None and fresh_success > random_fresh)})
    gates = {"eligible": eligible, "training_fit": all(r["training_fit"] for r in gate_rows),
             "fresh": all(r["fresh"] for r in gate_rows), "per_seed": gate_rows, "pooled_random_fresh_success": random_fresh}
    interpretation = ("smoke_or_protocol_deviation_not_gate_evidence" if smoke or deviations else "incomplete_not_gate_evidence" if not complete else
                      "fresh_pass_fit_unresolved" if gates["fresh"] and not gates["training_fit"] else
                      "fresh_competence_gate_met" if gates["fresh"] else "training_fit_only" if gates["training_fit"] else "training_fit_not_established")
    loss_aggregate = []
    for checkpoint in sorted({r["checkpoint"] for r in losses}):
        selected = [r for r in losses if r["checkpoint"] == checkpoint]
        loss_aggregate.append({"checkpoint": checkpoint, "mean_loss": float(np.mean([r["mean_loss"] for r in selected])),
            "seed_loss_min": min(r["mean_loss"] for r in selected), "seed_loss_max": max(r["mean_loss"] for r in selected),
            "seeds": len(selected), "updates_in_window": selected[0]["updates_in_window"]})
    artifacts = {"directory": artifact_path(output), "protocol": artifact_path(output / "protocol.json"),
        "protocol_document": artifact_path(protocol_file), "report": "docs/experiments/supervised_results_v1.md",
        "evaluations": artifact_path(output / "evaluations.csv"), "training": artifact_path(output / "losses.csv"),
        "state_metrics": artifact_path(output / "state_metrics.csv"), "dataset": artifact_path(output / "dataset.npz") if (output / "dataset.npz").exists() else None,
        "dataset_metadata": artifact_path(output / "dataset_metadata.json"), "references": artifact_path(output / "references.csv"),
        "sample_counts": artifact_path(output / "sample_counts.npz"), "sampling": artifact_path(output / "sampling.json"),
        "prior_reference_metadata": artifact_path(output / "prior_reference_metadata.json") if prior_metadata else None,
        "manifest": artifact_path(output / "manifest.json")}
    result = {"schema_version": 1, "protocol": protocol,
        "run": {"id": output.name, "status": "complete" if complete else "incomplete_admission_cap", "interpretation": interpretation,
                "train_updates": total_updates, "training_examples": sum(row["examples_seen"] for row in samples.values()),
                "wall_seconds": time.monotonic() - started, "dataset_wall_seconds": dataset_seconds,
                "training_wall_seconds": train_seconds, "evaluation_wall_seconds": evaluation_seconds,
                "reference_wall_seconds": reference_seconds, "model_parameter_count": model_parameter_count,
                "reference_episodes": len(reference_rows),
                "unique_reference_episodes": len({r["reference_sample_id"] for r in reference_rows}),
                "initial_policy_hashes": initial_hashes, "progress": progress,
                "limitations": ["Privileged exact labels remove both exploration and bootstrapped targets; this does not isolate a single RL failure cause.",
                    "All possible position/clock states on the selected layouts are supervised; this is not an RL sample-efficiency comparison.",
                    "Historical RL receives the same fresh panel; physical overlap with its training layouts is reported without replacing the panel.",
                    "Three initializations share one selected training bank; seed ranges are not confidence intervals."]},
        "aggregate": aggregate, "seed_results": seed_results, "state_aggregate": state_aggregate,
        "state_metrics": artifact_path(output / "state_metrics.csv"), "loss_aggregate": loss_aggregate,
        "references": references, "gates": gates, "sampling": samples, "trajectories": trajectories, "artifacts": artifacts}
    write_json(output / "trajectories.json", trajectories)
    write_json(output / "results.json", result)
    save_manifest(output, result["run"]["status"])
    if dashboard:
        write_json(Path(dashboard), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--train-maps", type=int, default=256)
    parser.add_argument("--fresh-maps", type=int, default=64)
    parser.add_argument("--train-seed-start", type=int, default=300000)
    parser.add_argument("--fresh-seed-start", type=int, default=930000)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.updates, args.train_maps, args.fresh_maps, args.seeds = 24, 2, 2, "0"
        args.train_seed_start, args.fresh_seed_start = 700000, 970000
    result = run_study(args.output, args.protocol_file, seeds=[int(v) for v in args.seeds.split(",")],
        updates=args.updates, train_maps=args.train_maps, fresh_maps=args.fresh_maps, train_seed_start=args.train_seed_start,
        fresh_seed_start=args.fresh_seed_start, max_seconds=args.max_seconds, dashboard=args.dashboard, prior_dir=args.prior_dir,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None, smoke=args.smoke)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
