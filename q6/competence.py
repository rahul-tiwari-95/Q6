"""A-only competence diagnosis with fixed-support and streaming task arms.

Learning is unchanged Double DQN. Evaluation owns its randomness, does not
reset training episodes, and uses the exact reference only for diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .adaptation import write_json
from .diagnostics import ShortestPathPolicy
from .learning import DQN
from .optimal import VisibleOptimalQ
from .world import CollectionWorld, WorldConfig

ROOT = Path(__file__).resolve().parent.parent
FIXED_TASKS = tuple(range(200000, 200016))
DEFAULT_CHECKPOINTS = (0, 4000, 12000, 40000, 120000)
CONDITIONS = (
    {"id": "fixed_1", "label": "One repeated task", "probe_seeds": [200000],
     "description": "The same walls, spawn and pellet at every reset."},
    {"id": "fixed_16", "label": "Sixteen repeated tasks", "probe_seeds": list(FIXED_TASKS),
     "description": "Balanced shuffled epochs of sixteen fixed task instances."},
    {"id": "stream", "label": "Task stream", "probe_seeds": list(FIXED_TASKS),
     "description": "The sixteen probes once, then uniform task-ID draws with replacement. Probes are not a memorization ceiling."},
)
TRAIN_FIELDS = ["condition", "seed", "episode", "map_seed", "task_hash", "start_step", "end_step",
                "steps", "completed", "success", "base_return", "shaped_return", "noop_steps",
                "moving_steps", "moved_revisits", "epsilon_start", "epsilon_end", "last_loss", "stop_reason"]
EVAL_FIELDS = ["condition", "policy", "seed", "checkpoint", "checkpoint_complete", "mode", "panel",
               "map_seed", "task_hash", "repetition", "success", "steps", "base_return", "shaped_return",
               "noop_steps", "moving_steps", "moved_revisits", "unique_positions", "winnable_steps",
               "optimal_winnable_actions", "avoidable_failure_actions", "action_regret_sum", "q_abs_error_sum",
               "q_error_steps", "initial_abs_q_error", "parameter_hash", "reference_sample_id"]


class BudgetReached(Exception):
    pass


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetReached("wall-clock admission cap reached")


def canonical_task(env: CollectionWorld) -> str:
    payload = {"walls": env.walls.astype(int).tolist(), "pellets": env.pellets.astype(int).tolist(),
               "position": [int(v) for v in env.position], "config": asdict(env.config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class TaskSampler:
    """Owns map selection RNG, independently of worlds and learner replay."""
    def __init__(self, condition: str, seed: int):
        if condition not in {c["id"] for c in CONDITIONS}:
            raise ValueError("unknown condition")
        self.condition = condition
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 77103]))
        self.draws = 0
        self.epoch = []

    def next_seed(self) -> int:
        if self.condition == "fixed_1":
            result = FIXED_TASKS[0]
        elif self.condition == "fixed_16":
            if not self.epoch:
                self.epoch = self.rng.permutation(FIXED_TASKS).tolist()
            result = self.epoch.pop()
        elif self.draws < len(FIXED_TASKS):
            result = FIXED_TASKS[self.draws]
        else:
            result = int(self.rng.integers(0, 900000))
            while result in FIXED_TASKS:
                result = int(self.rng.integers(0, 900000))
        self.draws += 1
        return int(result)


class Trainer:
    """A continuous training trajectory that can pause without resetting."""
    def __init__(self, agent: DQN, condition: str, seed: int, budget_steps: int, writer,
                 config: WorldConfig | None = None):
        self.agent, self.condition, self.seed = agent, condition, seed
        self.budget_steps, self.writer = budget_steps, writer
        self.env = CollectionWorld(config or WorldConfig())
        self.sampler = TaskSampler(condition, seed)
        self.state = None
        self.episode = 0
        self.current = None
        self.exposure = {}

    def _reset(self):
        map_seed = self.sampler.next_seed()
        self.state, _ = self.env.reset(seed=map_seed)
        task_hash = canonical_task(self.env)
        self.episode += 1
        self.visited = {self.env.position}
        self.current = {"condition": self.condition, "seed": self.seed, "episode": self.episode,
                        "map_seed": map_seed, "task_hash": task_hash, "start_step": self.agent.steps,
                        "steps": 0, "base_return": 0.0, "shaped_return": 0.0, "noop_steps": 0,
                        "moving_steps": 0, "moved_revisits": 0, "epsilon_start": None}
        if map_seed not in self.exposure:
            self.exposure[map_seed] = {"map_seed": map_seed, "task_hash": task_hash,
                                       "episode_starts": 0, "completed_episodes": 0, "transitions": 0}
        self.exposure[map_seed]["episode_starts"] += 1

    def _write_episode(self, completed: bool, reason: str):
        row = {**self.current, "end_step": self.agent.steps, "completed": int(completed),
               "success": int(self.env.terminated), "epsilon_end": self.epsilon,
               "last_loss": self.agent.last_loss, "stop_reason": reason}
        self.writer.writerow(row)
        if completed:
            self.exposure[row["map_seed"]]["completed_episodes"] += 1
        self.current = None
        self.state = None

    def advance_to(self, checkpoint: int, deadline: float):
        if checkpoint < self.agent.steps or checkpoint > self.budget_steps:
            raise ValueError("checkpoint must be between current step and training budget")
        while self.agent.steps < checkpoint:
            check_budget(deadline)
            if self.state is None:
                self._reset()
            self.epsilon = max(0.1, 1 - 0.9 * self.agent.steps / max(1, 0.7 * self.budget_steps))
            if self.current["epsilon_start"] is None:
                self.current["epsilon_start"] = self.epsilon
            action = self.agent.act(self.state, self.epsilon)
            before = self.env.position
            nxt, reward, terminated, truncated, info = self.env.step(action)
            self.agent.observe(self.state, action, reward, nxt, terminated or truncated)
            self.state = nxt
            self.current["steps"] += 1
            self.current["base_return"] += info["base_reward"]
            self.current["shaped_return"] += reward
            self.exposure[self.current["map_seed"]]["transitions"] += 1
            if before == self.env.position:
                self.current["noop_steps"] += 1
            else:
                self.current["moving_steps"] += 1
                self.current["moved_revisits"] += int(self.env.position in self.visited)
            self.visited.add(self.env.position)
            if terminated or truncated:
                self._write_episode(True, "success" if terminated else "horizon")

    def finish(self, reason="training_budget"):
        if self.current is not None:
            self._write_episode(False, reason)


def evaluation_draws(seed: int, map_seed: int, repetition: int, horizon: int):
    # Excludes condition/checkpoint. Draw both arrays in advance so trajectories
    # share the same per-time-step exploration schedule, even when they diverge.
    rng = np.random.default_rng(np.random.SeedSequence([seed, map_seed, repetition, 55219]))
    return rng.random(horizon), rng.integers(0, 4, horizon)


def evaluate_episode(agent, config, oracle, *, condition, seed, checkpoint, mode, panel,
                     map_seed, repetition=0, policy="learner", deadline=float("inf")):
    check_budget(deadline)
    env = CollectionWorld(config)
    state, _ = env.reset(seed=map_seed)
    task_hash = canonical_task(env)
    coins, random_actions = evaluation_draws(seed, map_seed, repetition, config.horizon)
    planner = ShortestPathPolicy(config.size) if policy == "shortest_path" else None
    policy_hash = agent.parameter_hash() if agent is not None else policy
    trajectory = {"condition": condition, "policy": policy, "seed": seed, "checkpoint": checkpoint,
                  "mode": mode, "panel": panel, "world": "A", "map_seed": map_seed,
                  "repetition": repetition, "grid_size": config.size, "action_mapping": list(config.action_mapping),
                  "walls": np.argwhere(env.walls).tolist(), "pellets_initial": np.argwhere(env.pellets).tolist(),
                  "start": list(env.position), "steps": []}
    row = {"condition": condition, "policy": policy, "seed": seed, "checkpoint": checkpoint,
           "checkpoint_complete": 1, "mode": mode, "panel": panel, "map_seed": map_seed,
           "task_hash": task_hash, "repetition": repetition, "base_return": 0.0, "shaped_return": 0.0,
           "noop_steps": 0, "moving_steps": 0, "moved_revisits": 0, "winnable_steps": 0,
           "optimal_winnable_actions": 0, "avoidable_failure_actions": 0, "action_regret_sum": 0.0,
           "q_abs_error_sum": 0.0, "q_error_steps": 0, "initial_abs_q_error": None,
           "parameter_hash": policy_hash,
           "reference_sample_id": "" if policy == "learner" else f"{policy}:{seed}:{map_seed}:{repetition}"}
    visited = {env.position}
    while True:
        optimal_q = oracle.q_values(state)
        winnable = oracle.can_finish(state)
        can_finish_action = oracle.action_can_finish(state)
        learned_q = None
        if policy == "learner":
            with torch.no_grad():
                learned_q = agent.online(torch.from_numpy(state).unsqueeze(0))[0].numpy()
            action = int(learned_q.argmax())
            if mode == "epsilon_0_1" and coins[env.elapsed] < 0.1:
                action = int(random_actions[env.elapsed])
            error = float(np.abs(learned_q - optimal_q).mean())
            row["q_abs_error_sum"] += error
            row["q_error_steps"] += 1
            if row["initial_abs_q_error"] is None:
                row["initial_abs_q_error"] = error
        elif policy == "random_actions":
            action = int(random_actions[env.elapsed])
        elif policy == "shortest_path":
            action = planner.act(state)
        else:
            raise ValueError("unknown evaluation policy")
        regret = float(max(0.0, optimal_q.max() - optimal_q[action]))
        row["action_regret_sum"] += regret
        if winnable:
            row["winnable_steps"] += 1
            row["optimal_winnable_actions"] += int(np.isclose(optimal_q[action], optimal_q.max(), atol=1e-8, rtol=0))
            row["avoidable_failure_actions"] += int(not can_finish_action[action])
        before = env.position
        state, reward, terminated, truncated, info = env.step(action)
        if before == env.position:
            row["noop_steps"] += 1
        else:
            row["moving_steps"] += 1
            row["moved_revisits"] += int(env.position in visited)
        visited.add(env.position)
        row["base_return"] += info["base_reward"]
        row["shaped_return"] += reward
        frame = {"position": list(env.position), "action": action, "reward": reward,
                 "base_reward": info["base_reward"], "pellets": np.argwhere(env.pellets).tolist(),
                 "terminated": terminated, "truncated": truncated,
                 "optimal_q_values": optimal_q.tolist(), "action_regret": regret,
                 "winnable": winnable, "action_can_finish": can_finish_action.tolist()}
        if learned_q is not None:
            frame["q_values"] = learned_q.tolist()
        trajectory["steps"].append(frame)
        if terminated or truncated:
            break
    row.update(success=int(terminated), steps=env.elapsed, unique_positions=len(visited))
    trajectory["success"] = terminated
    return row, trajectory


def summarize(rows):
    total_steps = sum(r["steps"] for r in rows)
    winnable = sum(r["winnable_steps"] for r in rows)
    q_steps = sum(r["q_error_steps"] for r in rows)
    moving = sum(r["moving_steps"] for r in rows)
    return {"episodes": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "mean_return": float(np.mean([r["base_return"] for r in rows])),
            "mean_shaped_return": float(np.mean([r["shaped_return"] for r in rows])),
            "noop_rate": sum(r["noop_steps"] for r in rows) / total_steps,
            "moved_revisit_rate": sum(r["moved_revisits"] for r in rows) / moving if moving else None,
            "moving_steps": moving, "evaluation_steps": total_steps, "winnable_steps": winnable, "q_error_steps": q_steps,
            "optimal_action_rate": sum(r["optimal_winnable_actions"] for r in rows) / winnable if winnable else None,
            "mean_action_regret": sum(r["action_regret_sum"] for r in rows) / total_steps,
            "mean_abs_q_error": sum(r["q_abs_error_sum"] for r in rows) / q_steps if q_steps else None,
            "avoidable_failure_actions": sum(r["avoidable_failure_actions"] for r in rows)}


def artifact_path(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def git_info():
    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
        return {"revision": revision, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"revision": None, "dirty": None}


def run_study(output: Path, protocol_file: Path, *, seeds=(0, 1, 2), phase_steps=120000,
              eval_episodes=64, max_seconds=900, dashboard=None, checkpoints=None, smoke=False):
    output, protocol_file = Path(output), Path(protocol_file)
    seeds = list(seeds)
    if not protocol_file.is_file():
        raise ValueError("an existing predeclared protocol file is required")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct nonnegative integers")
    if min(phase_steps, eval_episodes, max_seconds) <= 0:
        raise ValueError("all budgets must be positive")
    chosen_checkpoints = sorted(set(checkpoints if checkpoints is not None else
                                    [c for c in DEFAULT_CHECKPOINTS if c <= phase_steps] + [phase_steps]))
    if not chosen_checkpoints or chosen_checkpoints[0] != 0 or chosen_checkpoints[-1] != phase_steps:
        raise ValueError("checkpoints must include zero and the final training budget")
    if min(chosen_checkpoints) < 0 or max(chosen_checkpoints) > phase_steps:
        raise ValueError("checkpoints outside training budget")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    git = git_info()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = WorldConfig()
    heldout = list(range(920000, 920000 + eval_episodes))
    deviations = []
    for label, actual, expected in (("seeds", seeds, [0, 1, 2]), ("phase_steps", phase_steps, 120000),
                                    ("eval_episodes", eval_episodes, 64), ("max_seconds", max_seconds, 900),
                                    ("checkpoints", chosen_checkpoints, list(DEFAULT_CHECKPOINTS))):
        if actual != expected:
            deviations.append({"field": label, "actual": actual, "declared": expected})
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "platform": platform.platform(), "machine": platform.machine(), "torch_threads": 1, "device": "cpu"}
    if not runtime["python"].startswith("3.12.") or runtime["torch"].split("+")[0] != "2.8.0" or runtime["numpy"] != "2.0.2":
        deviations.append({"field": "runtime", "actual": runtime, "declared": "Python3.12/Torch2.8.0/NumPy2.0.2"})
    protocol_text = protocol_file.read_bytes()
    protocol = {"id": "competence-v1", "question": "Can the current DQN learn repeated tasks before a task stream?",
                "hypothesis": "Repeated support may be learnable while transfer to fresh tasks remains weak.",
                "seeds": seeds, "conditions": copy.deepcopy(list(CONDITIONS)), "checkpoints": chosen_checkpoints,
                "heldout_seeds": heldout, "world": asdict(config), "rule_visibility": "observed",
                "gates": {"memorization": 0.9, "heldout": 0.7, "scope": "each learner seed; greedy only; heldout must exceed random"},
                "budget": {"steps_per_condition_seed": phase_steps, "maximum_training_steps": len(seeds) * 3 * phase_steps,
                           "admission_seconds": max_seconds, "evaluation_included": True},
                "epsilon_evaluation_repetitions": {"probe": 8, "heldout": 2},
                "rng": {"learner": "owned shared exploration/replay RNG", "tasks": "owned SeedSequence([seed,77103])",
                        "evaluation": "SeedSequence([learnerseed,mapseed,repetition,55219]); common across conditions/checkpoints"},
                "source_sha256": source_hashes, "protocol_sha256": hashlib.sha256(protocol_text).hexdigest(),
                "git": git, "runtime": runtime, "smoke": bool(smoke), "deviations": deviations,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metric_definitions": {"optimal_action_rate": "optimal selections / winnable visited states only",
                                       "mean_action_regret": "unconditional trajectory-step average of Q*max - Q*chosen",
                                       "mean_abs_q_error": "unconditional visited-step average of four-action absolute Q error",
                                       "moved_revisit_rate": "moves returning to any previously visited position / moves, excluding no-ops"}}
    write_json(output / "protocol.json", protocol)
    (output / "protocol.md").write_bytes(protocol_text)
    for path in source_files:
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    (output / "environment.txt").write_text("\n".join(packages) + "\n")
    command = ["python", "-m", "q6.competence", "--output", str(output), "--protocol-file", str(protocol_file),
               "--seeds", ",".join(map(str, seeds)), "--phase-steps", str(phase_steps), "--eval-episodes", str(eval_episodes),
               "--max-seconds", str(max_seconds)]
    if smoke:
        command.append("--smoke")
    if checkpoints is not None:
        command.extend(["--checkpoints", ",".join(map(str, chosen_checkpoints))])
    (output / "command.txt").write_text(shlex.join(command) + "\n# Reproduction needs a new output path.\n")
    start = time.monotonic()
    deadline = start + max_seconds
    oracle = VisibleOptimalQ(config, max_cached_maps=128)
    raw, trajectories, summaries, reference_rows, exposures, progress = [], [], [], [], [], []
    total_train_steps = 0
    train_seconds = eval_seconds = reference_seconds = 0.0
    complete = True
    initial_hashes = {}
    model_parameter_count = None
    active_trainer = None
    reference_cache = {}

    with gzip.open(output / "training.csv.gz", "wt", newline="", encoding="utf-8") as train_handle, \
            (output / "evaluations.csv").open("w", newline="") as eval_handle, \
            (output / "references.csv").open("w", newline="") as ref_handle:
        train_writer = csv.DictWriter(train_handle, fieldnames=TRAIN_FIELDS)
        eval_writer = csv.DictWriter(eval_handle, fieldnames=EVAL_FIELDS)
        ref_writer = csv.DictWriter(ref_handle, fieldnames=EVAL_FIELDS)
        for writer in (train_writer, eval_writer, ref_writer):
            writer.writeheader()
        try:
            for condition_spec in CONDITIONS:
                condition = condition_spec["id"]
                panels = {"probe": condition_spec["probe_seeds"], "heldout": heldout}
                for seed in seeds:
                    check_budget(deadline)
                    agent = DQN(CollectionWorld(config).observation_size, seed=seed, gamma=config.gamma)
                    model_parameter_count = sum(parameter.numel() for parameter in agent.online.parameters())
                    initial = agent.parameter_hash()
                    if seed in initial_hashes:
                        assert initial_hashes[seed] == initial, "conditions did not share initialization"
                    initial_hashes[seed] = initial
                    trainer = Trainer(agent, condition, seed, phase_steps, train_writer, config)
                    active_trainer = trainer
                    for checkpoint in chosen_checkpoints:
                        before_steps, phase_start = agent.steps, time.monotonic()
                        try:
                            trainer.advance_to(checkpoint, deadline)
                        finally:
                            total_train_steps += agent.steps - before_steps
                            train_seconds += time.monotonic() - phase_start
                            train_handle.flush()
                        snapshot = output / "models" / f"{condition}_seed{seed}_step{checkpoint}.pt"
                        snapshot.parent.mkdir(exist_ok=True)
                        torch.save({"online": agent.online.state_dict(), "observation_size": agent.observation_size,
                                    "parameter_hash": agent.parameter_hash(), "training_steps": agent.steps,
                                    "purpose": "inference_only_not_resumable"}, snapshot)
                        checkpoint_rows, checkpoint_trajectories = [], []
                        eval_start = time.monotonic()
                        try:
                            for panel, map_seeds in panels.items():
                                for mode in ("greedy", "epsilon_0_1"):
                                    repetitions = 1 if mode == "greedy" else (8 if panel == "probe" else 2)
                                    for map_seed in map_seeds:
                                        for repetition in range(repetitions):
                                            row, trajectory = evaluate_episode(agent, config, oracle, condition=condition, seed=seed,
                                                checkpoint=checkpoint, mode=mode, panel=panel, map_seed=map_seed,
                                                repetition=repetition, deadline=deadline)
                                            checkpoint_rows.append(row)
                                            if map_seed == map_seeds[0] and repetition == 0:
                                                checkpoint_trajectories.append(trajectory)
                        except BudgetReached:
                            for row in checkpoint_rows:
                                row["checkpoint_complete"] = 0
                            eval_writer.writerows(checkpoint_rows)
                            raw.extend(checkpoint_rows)
                            progress.append({"condition": condition, "seed": seed, "checkpoint": checkpoint, "evaluation_complete": False})
                            raise
                        finally:
                            eval_seconds += time.monotonic() - eval_start
                        eval_writer.writerows(checkpoint_rows)
                        eval_handle.flush()
                        raw.extend(checkpoint_rows)
                        trajectories.extend(checkpoint_trajectories)
                        progress.append({"condition": condition, "seed": seed, "checkpoint": checkpoint, "evaluation_complete": True})
                        for panel in panels:
                            for mode in ("greedy", "epsilon_0_1"):
                                selected = [r for r in checkpoint_rows if r["panel"] == panel and r["mode"] == mode]
                                summaries.append({"condition": condition, "seed": seed, "checkpoint": checkpoint,
                                                  "panel": panel, "mode": mode, **summarize(selected)})
                        print(f"{condition} seed={seed} step={checkpoint}: "
                              f"probe={summaries[-4]['success_rate']:.3f} heldout={summaries[-2]['success_rate']:.3f}", flush=True)
                    trainer.finish()
                    exposures.append({"condition": condition, "seed": seed, "tasks": list(trainer.exposure.values())})
                    active_trainer = None
            # Shared references are evaluated once per unique random stream and
            # map, then presented across panels/conditions without new evidence.
            ref_start = time.monotonic()
            try:
                for condition_spec in CONDITIONS:
                    condition = condition_spec["id"]
                    for panel, map_seeds in {"probe": condition_spec["probe_seeds"], "heldout": heldout}.items():
                        for policy in ("random_actions", "shortest_path"):
                            repetitions = (8 if panel == "probe" else 2) if policy == "random_actions" else 1
                            reference_seeds = seeds if policy == "random_actions" else [0]
                            for seed in reference_seeds:
                                for map_seed in map_seeds:
                                    for repetition in range(repetitions):
                                        check_budget(deadline)
                                        key = (policy, seed, map_seed, repetition)
                                        if key not in reference_cache:
                                            reference_cache[key] = evaluate_episode(None, config, oracle, condition=condition, seed=seed,
                                                checkpoint=0, mode="reference", panel=panel, map_seed=map_seed, repetition=repetition,
                                                policy=policy, deadline=deadline)
                                        cached_row, cached_trajectory = reference_cache[key]
                                        row = {**cached_row, "condition": condition, "panel": panel}
                                        reference_rows.append(row)
                                        ref_writer.writerow(row)
                                        if seed == reference_seeds[0] and repetition == 0 and map_seed == map_seeds[0]:
                                            trajectories.append({**copy.deepcopy(cached_trajectory), "condition": condition, "panel": panel})
            finally:
                reference_seconds += time.monotonic() - ref_start
        except BudgetReached:
            complete = False
            if active_trainer is not None:
                active_trainer.finish("admission_cap")
                exposures.append({"condition": active_trainer.condition, "seed": active_trainer.seed,
                                  "tasks": list(active_trainer.exposure.values())})
    aggregate = []
    keys = sorted({(r["condition"], r["checkpoint"], r["panel"], r["mode"]) for r in summaries})
    for condition, checkpoint, panel, mode in keys:
        selected = [r for r in raw if r["checkpoint_complete"] and
                    (r["condition"], r["checkpoint"], r["panel"], r["mode"]) == (condition, checkpoint, panel, mode)]
        per_seed = [r for r in summaries if (r["condition"], r["checkpoint"], r["panel"], r["mode"]) ==
                    (condition, checkpoint, panel, mode)]
        aggregate.append({"condition": condition, "checkpoint": checkpoint, "panel": panel, "mode": mode,
                          **summarize(selected), "seeds": len(per_seed),
                          "seed_success_min": min(r["success_rate"] for r in per_seed),
                          "seed_success_max": max(r["success_rate"] for r in per_seed)})
    references = []
    for condition, policy, panel in sorted({(r["condition"], r["policy"], r["panel"]) for r in reference_rows}):
        selected = [r for r in reference_rows if (r["condition"], r["policy"], r["panel"]) == (condition, policy, panel)]
        references.append({"condition": condition, "policy": policy, "panel": panel, **summarize(selected),
                           "reused_across_conditions": True})
    # Training/evaluation overlap is based on canonical physical instances, not
    # merely disjoint RNG seed IDs. Fresh seeds can generate identical tasks.
    heldout_hashes = {r["task_hash"] for r in raw if r["panel"] == "heldout"}
    exposure_summary = []
    for item in exposures:
        hashes = {t["task_hash"] for t in item["tasks"]}
        exposure_summary.append({"condition": item["condition"], "seed": item["seed"],
                                 "unique_task_ids": len(item["tasks"]), "unique_canonical_tasks": len(hashes),
                                 "episode_starts": sum(t["episode_starts"] for t in item["tasks"]),
                                 "transitions": sum(t["transitions"] for t in item["tasks"]),
                                 "heldout_canonical_overlap": sorted(hashes & heldout_hashes)})
    eligible = complete and not smoke and not deviations
    gates = {"eligible": eligible, "repeated_support": {}, "heldout": {}}
    for spec in CONDITIONS:
        condition = spec["id"]
        for panel, threshold, destination in (("probe", 0.9, "repeated_support"), ("heldout", 0.7, "heldout")):
            if panel == "probe" and condition == "stream":
                continue
            selected = [r for r in summaries if r["condition"] == condition and r["checkpoint"] == phase_steps
                        and r["panel"] == panel and r["mode"] == "greedy"]
            passed = eligible and len(selected) == len(seeds) and all(r["success_rate"] >= threshold for r in selected)
            if panel == "heldout":
                random_ref = next((r["success_rate"] for r in references if r["condition"] == condition
                                   and r["panel"] == "heldout" and r["policy"] == "random_actions"), None)
                passed = passed and random_ref is not None and all(r["success_rate"] > random_ref for r in selected)
            gates[destination][condition] = bool(passed)
    interpretation = ("smoke_or_protocol_deviation_not_gate_evidence" if smoke or deviations else
                      "incomplete_not_gate_evidence" if not complete else
                      "fresh_competence_gate_met" if any(gates["heldout"].values()) else
                      "repeated_support_only" if any(gates["repeated_support"].values()) else "known_task_competence_not_established")
    artifacts = {"directory": artifact_path(output), "protocol": artifact_path(output / "protocol.json"),
                 "protocol_document": artifact_path(protocol_file), "report": "docs/experiments/competence_results_v1.md",
                 "evaluations": artifact_path(output / "evaluations.csv"), "training": artifact_path(output / "training.csv.gz"),
                 "references": artifact_path(output / "references.csv"), "exposure": artifact_path(output / "task_exposure.json"),
                 "manifest": artifact_path(output / "manifest.json")}
    result = {"schema_version": 1, "protocol": protocol,
              "run": {"id": output.name, "status": "complete" if complete else "incomplete_admission_cap",
                      "interpretation": interpretation, "train_steps": total_train_steps, "wall_seconds": time.monotonic() - start,
                      "training_wall_seconds": train_seconds, "evaluation_wall_seconds": eval_seconds,
                      "reference_wall_seconds": reference_seconds, "model_parameter_count": model_parameter_count,
                      "evaluation_steps": sum(r["steps"] for r in raw),
                      "unique_reference_steps": sum(r[0]["steps"] for r in reference_cache.values()),
                      "initial_policy_hashes": initial_hashes, "progress": progress,
                      "limitations": ["Three initializations share one selected task bank.",
                                      "Balanced task episodes do not balance transition exposure.",
                                      "Stream probes are selected once initially; at early checkpoints some have not yet been encountered.",
                                      "Greedy and epsilon evaluation use identical tasks; Q errors remain trajectory-weighted.",
                                      "Curves show seed ranges, not confidence intervals."]},
              "aggregate": aggregate, "seed_results": summaries, "references": references, "gates": gates,
              "task_exposure_summary": exposure_summary, "trajectories": trajectories, "artifacts": artifacts}
    write_json(output / "task_exposure.json", exposures)
    write_json(output / "trajectories.json", trajectories)
    write_json(output / "results.json", result)
    manifest = {"schema_version": 1, "algorithm": "sha256", "run_id": output.name, "status": result["run"]["status"],
                "files": {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}}
    write_json(output / "manifest.json", manifest)
    if dashboard:
        write_json(Path(dashboard), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--phase-steps", type=int, default=120000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--checkpoints", help="comma-separated steps; nonstandard schedules are marked as protocol deviations")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.phase_steps, args.eval_episodes, args.seeds = 64, 2, "0"
    result = run_study(args.output, args.protocol_file, seeds=[int(v) for v in args.seeds.split(",")],
                       phase_steps=args.phase_steps, eval_episodes=args.eval_episodes, max_seconds=args.max_seconds,
                       dashboard=args.dashboard, smoke=args.smoke,
                       checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
