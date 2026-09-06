"""Run the preregistered-in-repository, exploratory A→B→A baseline pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shlex
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .learning import DQN
from .world import CollectionWorld, WorldConfig

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ("untrained", "after_a", "after_b", "after_return_a")
MAPPINGS = {"A": (0, 1, 2, 3), "B": (3, 2, 0, 1)}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(agent: DQN, config: WorldConfig, panel: list[int], *, seed: int,
             condition: str, checkpoint: str, world: str):
    """Separate deterministic environments; no replay, gradients, or RNG draws."""
    rows, illustration = [], None
    policy_hash = agent.parameter_hash()
    for map_seed in panel:
        env = CollectionWorld(config)
        state, _ = env.reset(seed=map_seed)
        trajectory = {"seed": seed, "condition": condition, "checkpoint": checkpoint,
                      "world": world, "map_seed": map_seed, "grid_size": config.size,
                      "action_mapping": list(config.action_mapping),
                      "walls": np.argwhere(env.walls).tolist(),
                      "pellets_initial": np.argwhere(env.pellets).tolist(),
                      "start": list(env.position), "steps": []}
        shaped, base = 0.0, 0.0
        while True:
            action = agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            shaped += reward
            base += info["base_reward"]
            trajectory["steps"].append({"position": info["position"], "action": action,
                                         "reward": reward, "base_reward": info["base_reward"],
                                         "pellets": np.argwhere(env.pellets).tolist(),
                                         "terminated": terminated, "truncated": truncated})
            if terminated or truncated:
                break
        trajectory["success"] = terminated
        if illustration is None:
            illustration = trajectory
        rows.append({"seed": seed, "condition": condition, "checkpoint": checkpoint,
                     "world": world, "map_seed": map_seed, "success": int(terminated),
                     "steps": env.elapsed, "base_return": base, "shaped_return": shaped,
                     "parameter_hash": policy_hash})
    assert policy_hash == agent.parameter_hash(), "evaluation changed policy parameters"
    return rows, illustration


def summarize(rows: list[dict]) -> dict:
    return {"episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_return": float(np.mean([r["base_return"] for r in rows])),
            "mean_shaped_return": float(np.mean([r["shaped_return"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows]))}


def train_phase(agent: DQN, config: WorldConfig, *, seed: int, phase_index: int,
                phase_steps: int, first_phase_steps: int, deadline: float, writer):
    env = CollectionWorld(config)
    map_rng = np.random.default_rng(np.random.SeedSequence([seed, phase_index, 73021]))
    local_steps, episode, completed_episodes = 0, 0, 0
    while local_steps < phase_steps:
        if time.monotonic() >= deadline:
            return {"complete": False, "steps": local_steps, "episodes": completed_episodes}
        map_seed = int(map_rng.integers(0, 900000))
        state, _ = env.reset(seed=map_seed)
        episode += 1
        base_return, shaped_return = 0.0, 0.0
        while local_steps < phase_steps:
            epsilon = max(0.1, 1.0 - 0.9 * agent.steps / max(1, 0.7 * first_phase_steps))
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.observe(state, action, reward, next_state, terminated or truncated)
            state = next_state
            local_steps += 1
            shaped_return += reward
            base_return += info["base_reward"]
            if terminated or truncated:
                completed_episodes += 1
                break
        writer.writerow({"seed": seed, "phase": ("A", "B", "A_return")[phase_index],
                         "episode": episode, "map_seed": map_seed, "steps": env.elapsed,
                         "global_train_steps": agent.steps, "success": int(env.terminated),
                         "completed": int(env.terminated or env.truncated),
                         "base_return": base_return, "shaped_return": shaped_return,
                         "epsilon": epsilon, "last_loss": agent.last_loss})
    return {"complete": True, "steps": local_steps, "episodes": completed_episodes}


def run_study(output: Path, seeds: list[int], phase_steps: int = 12000,
              eval_episodes: int = 16, max_seconds: float = 300,
              dashboard: Path | None = None) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if min(phase_steps, eval_episodes, max_seconds) <= 0:
        raise ValueError("budgets must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output is nonempty; choose a new run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = WorldConfig()
    panel = list(range(900000, 900000 + eval_episodes))
    protocol_path = ROOT / "docs/experiments/adaptation_protocol_v1.md"
    protocol_text = protocol_path.read_text() if protocol_path.exists() else ""
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    protocol = {"id": "adaptation-v1", "status": "exploratory_pilot",
                "question": "Adaptation and retention under an observed movement-rule change",
                "scenario": "movement_rule", "rule_visibility": "observed",
                "world": asdict(config), "rules": [
                    {"world": "A", "label": "Normal controls", "mapping": list(MAPPINGS["A"])},
                    {"world": "B", "label": "Clockwise-rotated controls", "mapping": list(MAPPINGS["B"])}],
                "sequence": ["A", "B", "A"], "seeds": seeds,
                "phase_steps": phase_steps, "eval_panel": panel,
                "max_seconds": max_seconds, "primary_metric": "greedy evaluation success rate",
                "competence_gate": {"metric": "mean A success after first A phase", "minimum": 0.7,
                                    "failure_label": "baseline_underlearned", "statistical_test": False},
                "algorithm": {"name": "Compact Double DQN", "hidden_sizes": [128, 64],
                              "learning_rate": 0.001, "batch_size": 64, "replay_capacity": 12000,
                              "update_every": 4, "warmup_transitions": 256, "target_tau": 0.01},
                "evaluation_return": "base reward; shaping reported separately",
                "continued_state": "retain weights, Adam, target, replay and RNG; reset episode only",
                "frozen_state": "clone after A; no further training or RNG draws",
                "missing_comparisons": ["recurrent memory", "memory-reset ablation", "multiple movement changes"],
                "protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
                "source_sha256": source_hashes,
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "device": "cpu", "torch_threads": 1},
                "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "protocol.json", protocol)  # Written before learning.
    if protocol_text:
        (output / "protocol.md").write_text(protocol_text)
    command = ["python3", "-m", "q6.adaptation", "--output", str(output), "--seeds",
               ",".join(map(str, seeds)), "--phase-steps", str(phase_steps),
               "--eval-episodes", str(eval_episodes), "--max-seconds", str(max_seconds)]
    (output / "command.txt").write_text(shlex.join(command) + "\n# Use a NEW output path when reproducing.\n")
    start = time.monotonic()
    deadline = start + max_seconds
    raw_evaluations, runs, trajectories, phase_records = [], [], [], []
    training_seconds = evaluation_seconds = 0.0
    complete = True

    def panel_evaluation(agent, seed, condition, checkpoint):
        nonlocal evaluation_seconds
        eval_start = time.monotonic()
        for world in ("A", "B"):
            world_config = replace(config, action_mapping=MAPPINGS[world])
            rows, trajectory = evaluate(agent, world_config, panel, seed=seed,
                                        condition=condition, checkpoint=checkpoint, world=world)
            raw_evaluations.extend(rows)
            trajectories.append(trajectory)
            runs.append({"seed": seed, "condition": condition, "checkpoint": checkpoint,
                         "world": world, **summarize(rows),
                         "parameter_hash": agent.parameter_hash(), "train_steps": agent.steps})
        evaluation_seconds += time.monotonic() - eval_start

    training_fields = ["seed", "phase", "episode", "map_seed", "steps", "global_train_steps",
                       "success", "completed", "base_return", "shaped_return", "epsilon", "last_loss"]
    with (output / "training.csv").open("w", newline="") as training_file:
        writer = csv.DictWriter(training_file, fieldnames=training_fields)
        writer.writeheader()
        for seed in seeds:
            if time.monotonic() >= deadline:
                complete = False
                break
            agent = DQN(CollectionWorld(config).observation_size, seed=seed, gamma=config.gamma)
            panel_evaluation(agent, seed, "untrained", "untrained")
            frozen = None
            for phase_index, world in enumerate(("A", "B", "A")):
                phase_start = time.monotonic()
                phase_result = train_phase(agent, replace(config, action_mapping=MAPPINGS[world]),
                                           seed=seed, phase_index=phase_index, phase_steps=phase_steps,
                                           first_phase_steps=phase_steps, deadline=deadline, writer=writer)
                duration = time.monotonic() - phase_start
                training_seconds += duration
                training_file.flush()
                phase_records.append({"seed": seed, "phase_index": phase_index, "world": world,
                                      "wall_seconds": duration, **phase_result})
                if not phase_result["complete"]:
                    complete = False
                    break
                checkpoint = CHECKPOINTS[phase_index + 1]
                if phase_index == 0:
                    frozen = DQN.from_state(agent.state_dict())
                panel_evaluation(agent, seed, "continued", checkpoint)
                panel_evaluation(frozen, seed, "frozen_after_a", checkpoint)
                model_path = output / "models" / f"seed_{seed}_{checkpoint}.pt"
                model_path.parent.mkdir(exist_ok=True)
                # Inference snapshots; exact continuation is supported by
                # DQN.state_dict(), but large replay arrays are not published.
                torch.save({"online": agent.online.state_dict(),
                            "observation_size": agent.observation_size,
                            "parameter_hash": agent.parameter_hash(),
                            "purpose": "inference_only_not_resumable"}, model_path)
                print(f"seed={seed} {checkpoint}: A={runs[-4]['success_rate']:.3f} "
                      f"B={runs[-3]['success_rate']:.3f} ({duration:.1f}s train)", flush=True)
            if not complete:
                break
    if raw_evaluations:
        with (output / "evaluations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(raw_evaluations[0]))
            writer.writeheader()
            writer.writerows(raw_evaluations)
    aggregate = []
    for condition, checkpoint, world in sorted({(r["condition"], r["checkpoint"], r["world"]) for r in runs}):
        selected = [r for r in runs if (r["condition"], r["checkpoint"], r["world"]) == (condition, checkpoint, world)]
        rows = [r for r in raw_evaluations if (r["condition"], r["checkpoint"], r["world"]) == (condition, checkpoint, world)]
        aggregate.append({"condition": condition, "checkpoint": checkpoint, "world": world,
                          **summarize(rows), "seeds": len(selected),
                          "seed_success_min": min(r["success_rate"] for r in selected),
                          "seed_success_max": max(r["success_rate"] for r in selected)})
    training_steps = sum(p["steps"] for p in phase_records)
    initial_a = [r["success_rate"] for r in runs if r["condition"] == "continued"
                 and r["checkpoint"] == "after_a" and r["world"] == "A"]
    elapsed = time.monotonic() - start
    status = "complete" if complete else "incomplete_budget_cap"
    competence = float(np.mean(initial_a)) if initial_a else None
    interpretation_status = ("incomplete" if not complete else
                             "baseline_underlearned" if competence is None or competence < 0.7
                             else "competence_gate_met_exploratory")
    try:
        artifact_directory = output.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        artifact_directory = str(output)
    result = {"schema_version": 1, "protocol": protocol,
              "run": {"id": output.name, "status": status, "exploratory": True,
                      "wall_seconds": elapsed, "training_wall_seconds": training_seconds,
                      "evaluation_wall_seconds": evaluation_seconds, "training_steps": training_steps,
                      "evaluation_steps": sum(r["steps"] for r in raw_evaluations),
                      "phase_records": phase_records, "checkpoint_order": list(CHECKPOINTS),
                      "conditions": ["untrained", "continued", "frozen_after_a"],
                      "after_a_mean_success": competence,
                      "interpretation_status": interpretation_status,
                      "competence_note": "The preregistered feasibility gate is mean A success >=70% after A. Below it, retention is not interpreted as forgetting.",
                      "limitations": ["Three seeds by default; descriptive pilot only.",
                                      "The movement rule is visible, not inferred from hidden dynamics.",
                                      "Frozen control has no post-A training compute.",
                                      "No recurrent-memory or reset-state comparison in this milestone."]},
              "artifacts": {"directory": artifact_directory, "protocol": artifact_directory + "/protocol.json",
                            "protocol_document": "docs/experiments/adaptation_protocol_v1.md",
                            "training": artifact_directory + "/training.csv", "evaluations": artifact_directory + "/evaluations.csv",
                            "results": artifact_directory + "/results.json", "trajectories": artifact_directory + "/trajectories.json"},
              "aggregate": aggregate, "runs": runs, "trajectories": trajectories}
    write_json(output / "trajectories.json", trajectories)
    write_json(output / "results.json", result)
    if dashboard:
        write_json(dashboard, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--phase-steps", type=int, default=12000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    result = run_study(args.output, [int(s) for s in args.seeds.split(",")],
                       args.phase_steps, args.eval_episodes, args.max_seconds, args.dashboard)
    print(json.dumps(result["run"], indent=2))


if __name__ == "__main__":
    main()
