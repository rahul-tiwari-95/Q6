"""Post-hoc, evaluation-only random and shortest-path reference policies.

These checks are deliberately stored separately from the predeclared pilot.
Neither policy sees environment internals: both receive the same observation
vector as the DQN. The planner is a model-based reference, not a learned method.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from .adaptation import evaluate, summarize, write_json
from .world import DELTAS, WorldConfig


class ShortestPathPolicy:
    def __init__(self, size: int):
        self.size = size

    def act(self, observation: np.ndarray) -> int:
        n = self.size
        layers = observation[:3 * n * n].reshape(3, n, n)
        walls, pellets, occupancy = layers > 0.5
        positions = np.argwhere(occupancy)
        if len(positions) != 1:
            raise ValueError("observation must contain one agent")
        start = tuple(int(v) for v in positions[0])
        queue, seen = deque([(start, None)]), {start}
        while queue:
            (r, c), first_direction = queue.popleft()
            if pellets[r, c]:
                if first_direction is None:
                    raise ValueError("agent cannot stand on an uncollected pellet")
                # Each row is an action label; columns are physical directions.
                mapping = observation[3 * n * n + 1:].reshape(4, 4)
                return int(np.argmax(mapping[:, first_direction]))
            for direction, (dr, dc) in enumerate(DELTAS):
                candidate = r + dr, c + dc
                if (0 <= candidate[0] < n and 0 <= candidate[1] < n
                        and not walls[candidate] and candidate not in seen):
                    seen.add(candidate)
                    queue.append((candidate, direction if first_direction is None else first_direction))
        raise ValueError("no reachable pellet in observation")

    def parameter_hash(self):
        return f"shortest-path-visible-observation-size-{self.size}"


class RandomPolicy:
    def __init__(self, seed: int, map_seed: int):
        self.seed, self.map_seed = seed, map_seed
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, map_seed, 42019]))

    def act(self, observation: np.ndarray) -> int:
        return int(self.rng.integers(4))

    def parameter_hash(self):
        return f"random-actions-seed-{self.seed}-map-{self.map_seed}"


def run_diagnostics(study: Path, output: Path | None = None) -> dict:
    protocol = json.loads((study / "protocol.json").read_text())
    raw_config = dict(protocol["world"])
    raw_config["action_mapping"] = tuple(raw_config["action_mapping"])
    base = WorldConfig(**raw_config)
    rows, trajectories = [], []
    for rule in protocol["rules"]:
        world = rule["world"]
        config = replace(base, action_mapping=tuple(rule["mapping"]))
        for policy_name in ("shortest_path", "random_actions"):
            # Deterministic planner is evaluated once, not counted as three
            # independent replications of an identical policy.
            seeds = [0] if policy_name == "shortest_path" else protocol["seeds"]
            for seed in seeds:
                for map_seed in protocol["eval_panel"]:
                    policy = (ShortestPathPolicy(config.size) if policy_name == "shortest_path"
                              else RandomPolicy(seed, map_seed))
                    episodes, trajectory = evaluate(policy, config, [map_seed], seed=seed,
                                                     condition=policy_name, checkpoint="posthoc_reference", world=world)
                    rows.extend(episodes)
                    if seed == seeds[0] and map_seed == protocol["eval_panel"][0]:
                        trajectories.append(trajectory)
    aggregate = []
    for policy_name in ("shortest_path", "random_actions"):
        for world in ("A", "B"):
            selected = [row for row in rows if row["condition"] == policy_name and row["world"] == world]
            aggregate.append({"condition": policy_name, "world": world, **summarize(selected)})
    with (study / "training.csv").open() as handle:
        training = list(csv.DictReader(handle))
    training_tail = []
    for seed in protocol["seeds"]:
        for phase in ("A", "B", "A_return"):
            selected = [row for row in training if int(row["seed"]) == seed
                        and row["phase"] == phase and row["completed"] == "1"][-100:]
            if selected:
                training_tail.append({"seed": seed, "phase": phase, "completed_episodes": len(selected),
                                      "success_rate": float(np.mean([int(row["success"]) for row in selected]))})
    result = {"schema_version": 1, "status": "posthoc_diagnostic_not_predeclared_outcome",
              "study_id": study.name, "protocol_id": protocol["id"],
              "notes": ["No training or model changes were performed.",
                        "Shortest path uses only visible walls, pellets, agent position and action mapping.",
                        "The planner still acts one step at a time under the same horizon and collision rules.",
                        "Random actions use independent per-map RNG and three seeds by default.",
                        "Training-tail and held-out success differ in both map samples and exploration; they do not isolate generalization."],
              "aggregate": aggregate, "training_tail": training_tail,
              "episodes": rows, "trajectories": trajectories}
    target = output or study / "diagnostics.json"
    write_json(target, result)
    with (study / "diagnostics_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_diagnostics(args.study, args.output)
    print(json.dumps({"aggregate": result["aggregate"], "training_tail": result["training_tail"]}, indent=2))


if __name__ == "__main__":
    main()
