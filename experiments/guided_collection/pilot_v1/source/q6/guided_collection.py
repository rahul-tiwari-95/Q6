"""A fixed random/greedy episode mixture followed by unchanged constrained DDQN."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import map_replay as shared
from .adaptation import write_json
from .competence import BudgetReached, artifact_path
from .constrained_bootstrap import constrained_update
from .familiar_starts import logged_success_paths, make_row_lookup
from .fixed_targets import ConsistencyError, MemoryReached, array_metadata, module_hash
from .panel_evaluation import FrozenPolicy
from .recorded_actions import RecordedActionsComparison, build_recorded_table
from .world import CollectionWorld, DELTAS, WorldConfig

CONDITIONS = ("constrained_bootstrap", "guided_collection")
COMPARISONS = ({"id": "guided_minus_random", "left": CONDITIONS[0], "right": CONDITIONS[1]},)
CHECKPOINTS = (0, 1000, 3000, 10000, 30000)
STEP_FIELDS = ("map_seed", "repetition", "step", "current_row", "action", "reward", "next_row", "terminated", "truncated", "remaining")
EPISODE_FIELDS = ("map_seed", "repetition", "policy", "steps", "success", "terminated", "truncated", "complete", "base_return", "shaped_return", "noop_steps", "trajectory_sha256")


def collect_mixture(config, data, collector, output, *, bank_id, enforce=None, episodes_per_map=16, random_episodes=8):
    """Record every real interaction; fixed greedy slots receive no masks/outcomes."""
    requested_enforce = enforce or (lambda: None)
    resource_checks = 0
    def enforce():
        nonlocal resource_checks
        resource_checks += 1
        requested_enforce()
    if bank_id < 1 or episodes_per_map < 1 or not 0 <= random_episodes <= episodes_per_map:
        raise ValueError("valid bank and episode mixture required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lookup = make_row_lookup(data)
    env, visited = CollectionWorld(config), np.zeros(len(data["observations"]), np.uint32)
    episodes, current = [], None
    total_steps, started = 0, time.monotonic()
    before = {"online": collector.parameter_hash(), "target": module_hash(collector.target)}
    rng_before = torch.get_rng_state().clone()
    status, reason = "complete", None
    with (output / "collection_steps.csv").open("w", newline="") as sf, (output / "collection_episodes.csv").open("w", newline="") as ef:
        sw = csv.DictWriter(sf, [*STEP_FIELDS, "policy", "q0", "q1", "q2", "q3"])
        ew = csv.DictWriter(ef, EPISODE_FIELDS)
        sw.writeheader()
        ew.writeheader()
        try:
            for map_seed in np.unique(data["map_seeds"]):
                for repetition in range(episodes_per_map):
                    enforce()
                    observation, _ = env.reset(seed=int(map_seed))
                    component = "random" if repetition < random_episodes else "guided"
                    rng = np.random.default_rng(np.random.SeedSequence([int(map_seed), repetition, 77301, bank_id])) if component == "random" else None
                    digest = hashlib.sha256()
                    current = {"map_seed": int(map_seed), "repetition": repetition, "policy": component, "steps": 0, "success": 0,
                        "terminated": 0, "truncated": 0, "complete": 0, "base_return": 0., "shaped_return": 0., "noop_steps": 0, "trajectory_sha256": digest.hexdigest()}
                    while not (env.terminated or env.truncated):
                        enforce()
                        remaining = config.horizon - env.elapsed
                        row = lookup[(int(map_seed), *env.position, remaining)]
                        if not np.array_equal(observation, data["observations"][row]):
                            raise ConsistencyError("collector observation differs from archived row identity")
                        q = None
                        if component == "random":
                            action = int(rng.integers(4))
                        else:
                            with torch.no_grad():
                                q = collector.online(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
                            if not np.all(np.isfinite(q)):
                                raise ConsistencyError("nonfinite frozen collector predictions")
                            action = int(q.argmax())
                        previous_position = env.position
                        observation, reward, terminated, truncated, info = env.step(action)
                        following = -1 if terminated or truncated else lookup[(int(map_seed), *env.position, config.horizon - env.elapsed)]
                        visited[row] += 1
                        total_steps += 1
                        current.update(steps=env.elapsed, success=int(terminated), terminated=int(terminated), truncated=int(truncated))
                        current["base_return"] += info["base_reward"]
                        current["shaped_return"] += reward
                        current["noop_steps"] += int(previous_position == env.position)
                        step = {"map_seed": int(map_seed), "repetition": repetition, "step": env.elapsed, "current_row": row, "action": action,
                            "reward": reward, "next_row": following, "terminated": int(terminated), "truncated": int(truncated), "remaining": remaining}
                        digest.update(json.dumps({k: step[k] for k in STEP_FIELDS if k != "repetition"}, sort_keys=True, separators=(",", ":")).encode())
                        current["trajectory_sha256"] = digest.hexdigest()
                        sw.writerow({**step, "policy": component, **{f"q{a}": float(q[a]) if q is not None else "" for a in range(4)}})
                    current["complete"] = 1
                    ew.writerow(current)
                    episodes.append(current)
                    current = None
        except (BudgetReached, ConsistencyError) as exc:
            status = "inconsistent_not_evidence" if isinstance(exc, ConsistencyError) else "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
            reason = str(exc)
            if current is not None:
                ew.writerow(current)
                episodes.append(current)
    after = {"online": collector.parameter_hash(), "target": module_hash(collector.target)}
    unchanged = before == after and torch.equal(rng_before, torch.get_rng_state())
    if not unchanged:
        status, reason = "inconsistent_not_evidence", "frozen collector parameters or torch RNG changed"
    arrays = {"visited_counts": visited, "support_indices": np.flatnonzero(visited).astype(np.int32)}
    for a in arrays.values():
        a.flags.writeable = False
    np.savez_compressed(output / "collection.npz", **arrays)
    components = []
    for policy in ("random", "guided"):
        group = [r for r in episodes if r["policy"] == policy]
        components.append({"policy": policy, "episodes": len(group), "complete_episodes": sum(r["complete"] for r in group),
            "steps": sum(r["steps"] for r in group), "successes": sum(r["success"] for r in group), "noop_steps": sum(r["noop_steps"] for r in group),
            "unique_trajectories": len({r["trajectory_sha256"] for r in group if r["complete"]}),
            "mean_steps": float(np.mean([r["steps"] for r in group])) if group else None})
    guided_checks = [{"map_seed": int(m), "episodes": int(sum(r["map_seed"] == m and r["policy"] == "guided" and r["complete"] for r in episodes)),
        "unique_trajectories": len({r["trajectory_sha256"] for r in episodes if r["map_seed"] == m and r["policy"] == "guided" and r["complete"]})} for m in np.unique(data["map_seeds"])]
    summary = {"bank_id": bank_id, "status": status, "stop_reason": reason, "collection_episodes": len(episodes), "complete_collection_episodes": sum(r["complete"] for r in episodes),
        "collection_steps": total_steps, "collection_successes": sum(r["success"] for r in episodes), "collection_noop_steps": sum(r["noop_steps"] for r in episodes),
        "episodes_per_map": episodes_per_map, "random_episodes_per_map": random_episodes, "per_policy": components,
        "guided_duplicate_checks": guided_checks, "collector_before": before, "collector_after": after, "collector_parameters_rng_unchanged": unchanged,
        "arrays": array_metadata(arrays), "wall_seconds": time.monotonic() - started, "resource_checks": resource_checks}
    write_json(output / "collection.json", summary)
    return summary, arrays


def verify_random_slots(original_path, mixed_path, random_episodes=8):
    """Numeric records must exactly reproduce archived random slot streams."""
    def records(path):
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                if int(row["repetition"]) < random_episodes:
                    yield tuple(float(row[k]) if k == "reward" else int(row[k]) for k in STEP_FIELDS)
    from itertools import zip_longest
    count, identical = 0, True
    for old, new in zip_longest(records(original_path), records(mixed_path)):
        count += 1
        identical &= old == new
    return {"random_slots": list(range(random_episodes)), "compared_steps": count, "identical": identical}


def physical_shortest_steps(observation, size=5):
    walls, pellets, occupancy = observation[:3 * size * size].reshape(3, size, size) > .5
    start = tuple(np.argwhere(occupancy)[0])
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        position, distance = queue.popleft()
        if pellets[position]:
            return distance
        for dr, dc in DELTAS:
            following = position[0] + dr, position[1] + dc
            if 0 <= following[0] < size and 0 <= following[1] < size and not walls[following] and following not in seen:
                seen.add(following)
                queue.append((following, distance + 1))
    raise ConsistencyError("physical shortest-path diagnostic found unreachable pellet")


def collection_replays(path, config, data, bank, condition, selected_maps):
    """Render recorded outcomes; no additional collection steps or predictions."""
    grouped = {}
    with Path(path).open(newline="") as f:
        for record in csv.DictReader(f):
            key = int(record["map_seed"]), int(record["repetition"])
            if key[0] in selected_maps and key[1] in (0, 8):
                grouped.setdefault(key, []).append(record)
    result = []
    for (map_seed, slot), records in sorted(grouped.items()):
        env = CollectionWorld(config)
        env.reset(seed=map_seed)
        component = "guided" if condition == CONDITIONS[1] and slot >= 8 else "random"
        replay = {"bank_id": bank, "condition": condition, "policy": "collector_" + component, "policy_component": component,
            "seed": bank, "checkpoint": 0, "mode": "collection", "panel": "training_collection", "map_seed": map_seed, "repetition": slot,
            "source": "new_collection" if condition == CONDITIONS[1] else "archived_collection", "world": "A", "grid_size": config.size,
            "action_mapping": list(config.action_mapping), "walls": np.argwhere(env.walls).tolist(), "pellets_initial": np.argwhere(env.pellets).tolist(), "start": list(env.position), "steps": []}
        for row in records:
            current, action, following = int(row["current_row"]), int(row["action"]), int(row["next_row"])
            terminated, truncated = bool(int(row["terminated"])), bool(int(row["truncated"]))
            position = list(map(int, data["positions"][following])) if following >= 0 else list(map(int, np.argwhere(env.pellets)[0])) if terminated else list(map(int, data["positions"][current]))
            if truncated:
                dr, dc = DELTAS[config.action_mapping[action]]
                candidate = position[0] + dr, position[1] + dc
                if 0 <= candidate[0] < config.size and 0 <= candidate[1] < config.size and not env.walls[candidate]:
                    position = list(candidate)
            frame = {"position": position, "position_before": list(map(int, data["positions"][current])), "action": action, "reward": float(row["reward"]),
                "base_reward": config.step_cost + config.pellet_reward * terminated, "pellets": [] if terminated else np.argwhere(env.pellets).tolist(),
                "terminated": terminated, "truncated": truncated, "current_row": current, "successor_row": following, "remaining_before": int(row["remaining"])}
            if row.get("q0", ""):
                frame["q_values"] = [float(row[f"q{a}"]) for a in range(4)]
            replay["steps"].append(frame)
        replay["success"] = bool(records and int(records[-1]["terminated"]))
        replay["complete"] = bool(records and (int(records[-1]["terminated"]) or int(records[-1]["truncated"])))
        result.append(replay)
    return result


class GuidedCollectionComparison(RecordedActionsComparison):
    conditions, comparisons = CONDITIONS, COMPARISONS
    module, panel_seed_start = "q6.guided_collection", 1140000
    extra_archives = ("map_replay", "within_map", "recorded_actions", "constrained_bootstrap", "logged_graph")
    control_archive, control_uses_recorded, requires_paired_replay = "constrained_bootstrap", True, False

    def __init__(self):
        super().__init__()
        self.control_tables, self.control_summaries, self.treatment_supports = {}, [], {}
        self.control_queries, self.control_maps, self.control_actions = {}, {}, []
        self.collections, self.collector, self.random_checks = [], {}, []
        self.control_collection_summaries = []
        self.collector_checks, self.control_checks, self.control_identity = [], [], []
        self.route_starts, self.route_rows, self.route_arrays, self.route_before = [], [], {}, {}
        self.collection_trajectories, self.guided_routes = [], {}

    def configure_protocol(self, protocol):
        super().configure_protocol(protocol)
        self.seeds, self.updates = protocol["seeds"], protocol["budget"]["updates_per_fit"]
        protocol.update(id="guided-collection-v1", question="Does a fixed random/greedy collection mixture improve fresh behavior with constrained DDQN unchanged?",
            conditions=[{"id": CONDITIONS[0], "label": "Random collection (frozen control)"}, {"id": CONDITIONS[1], "label": "Eight random plus eight fixed greedy episodes"}])
        protocol["collection"] = {"episodes_per_map": 16, "maps": list(range(300000, 300256)), "random_slots": list(range(8)), "guided_slots": list(range(8, 16)),
            "random_rng": "SeedSequence([map_seed,repetition,77301,bank_id])", "guided_action_selection": "all4 lowest-label greedy; no mask, epsilon or oracle",
            "collector": "constrained_bootstrap bank1 seed0 update30000, fixed by index", "guided_routes_shared_across_banks": True,
            "equal_episode_not_interaction_budget": True, "new_support_draws": 0, "support_source": "sorted unique recorded current rows"}
        protocol["optimizer"]["implementation"] = "unchanged constrained_bootstrap.constrained_update"
        protocol["sampling"].update(paired_local_global_map_schedule_within_bank=False, baseline_reconstruction="full30000 draws checked only against archived control",
            target_access="own-arm recorded outcomes and successor logged masks only", changed_support_changes_realized_stream=True)
        protocol["budget"].update(maximum_collection_episodes=len(self.bank_ids) * 4096, maximum_collection_steps=len(self.bank_ids) * 4096 * 32)
        protocol["recorded_actions"]["equal_action_target_budget"] = False
        protocol["collector_recordings"] = {"map_seeds": list(range(300000, 300256, 32)), "slots": [0, 8], "both_arms": True,
            "expected": len(self.bank_ids) * 8 * 2 * 2, "archived_control_rendering": "from saved rows only, without additional world.step or predictions"}
        protocol["evaluation"]["action_selection"] = "all4 fresh greedy and epsilon0.1; no logged masks at deployment"

    def prepare_inputs(self, directories, manifests, archives_meta, data, transitions, output, enforce):
        self.enforce = enforce
        super().prepare_inputs(directories, manifests, archives_meta, data, transitions, output, enforce)
        self.control_tables, self.control_summaries = self.tables.copy(), [dict(r, condition=CONDITIONS[0]) for r in self.table_summaries]
        directory, manifest = directories[self.control_archive], manifests[self.control_archive]
        for filename in ("recorded_transitions.npz", "recorded_query_counts.npz", "map_counts.npz", "action_exposure.json", "action_coverage.json", "supports.npz", "protocol.json", "protocol.md"):
            enforce()
            shared.checked_input(directory, manifest, filename, archives_meta[self.control_archive]["files"])
            shutil.copy2(directory / filename, output / ("control_" + filename))
        with np.load(directory / "recorded_transitions.npz") as saved, np.load(directory / "supports.npz") as support_file:
            for bank, table in self.control_tables.items():
                matches = array_metadata(table) == array_metadata({k: saved[f"bank{bank}_{k}"] for k in table})
                same_support = np.array_equal(support_file[f"{CONDITIONS[0]}_bank{bank}"], self.original_supports[bank])
                self.control_identity.append({"bank_id": bank, "recorded_tables_identical": matches, "support_identical": same_support})
                if not matches or not same_support:
                    raise ConsistencyError("archived constrained control differs from original random logs")
        with np.load(directory / "recorded_query_counts.npz") as q, np.load(directory / "map_counts.npz") as m:
            for bank in self.bank_ids:
                for seed in self.seeds:
                    name = f"bank{bank}_{CONDITIONS[0]}_seed{seed}"
                    self.control_queries[(bank, seed)], self.control_maps[(bank, seed)] = q[name], m[name]
        self.control_actions = json.loads((directory / "action_exposure.json").read_text())["per_seed"]
        filename = "models/bank1_constrained_bootstrap_seed0_update30000.pt"
        shared.checked_input(directory, manifest, filename, archives_meta[self.control_archive]["files"])
        shutil.copy2(directory / filename, output / "collector.pt")
        collector_snapshot = torch.load(output / "collector.pt", map_location="cpu", weights_only=True)
        if collector_snapshot["optimizer_updates"] != 30000:
            raise ConsistencyError("frozen collector must be the final30000 checkpoint")
        collector = FrozenPolicy(collector_snapshot)
        teacher_counts = next(r for r in self.control_actions if (r["bank_id"], r["condition"], r["seed"]) == (1, CONDITIONS[0], 0))
        self.collector = {"source": artifact_path(directory / filename), "saved": "collector.pt", "source_before": shared.sha(directory / filename),
            "copy_before": shared.sha(output / "collector.pt"), "online_before": collector.parameter_hash(), "target_before": module_hash(collector.target),
            "selection": "fixed bank1/seed0/final30000, not selected by performance", "historical_interactions": 100878,
            "historical_updates": 30000, "historical_state_presentations": 1920000, "historical_action_targets": teacher_counts["action_target_presentations"],
            "historical_nonterminal_queries": teacher_counts["nonterminal_target_queries"], "historical_terminal_action_targets": teacher_counts["terminal_action_targets"], "knowledge_overlaps_replaced_random_slots": True}
        if self.collector["source_before"] != self.collector["copy_before"]:
            raise ConsistencyError("collector copy differs from source")
        for bank in self.bank_ids:
            enforce()
            old_path = output / f"control_banks/bank{bank}/collection_steps.csv"
            old_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / f"banks/bank{bank}/collection_steps.csv", old_path)
            old_summary = next(r for r in self.control_summaries if r["bank_id"] == bank)
            old_summary.update(copied=old_path.relative_to(output).as_posix(), copied_sha256=shared.sha(old_path))
            filename = f"banks/bank{bank}/collection_episodes.csv"
            shared.checked_input(directories["bank_replication"], manifests["bank_replication"], filename, archives_meta["bank_replication"]["files"])
            shutil.copy2(directories["bank_replication"] / filename, old_path.parent / "collection_episodes.csv")
        self.control_collection_summaries = self.collection_result(capture_controls=True)["per_condition"]
        self.tables, self.tensors, self.table_summaries, self.tensor_before = {}, {}, [], {}
        for bank in self.bank_ids:
            enforce()
            directory_out = output / f"banks/bank{bank}"
            summary, arrays = collect_mixture(WorldConfig(), data, collector, directory_out, bank_id=bank, enforce=enforce)
            self.collections.append(summary)
            write_json(output / "collection.json", {"per_bank": self.collections})
            if summary["status"] != "complete":
                if summary["status"] == "inconsistent_not_evidence":
                    raise ConsistencyError(summary["stop_reason"])
                if summary["status"] == "incomplete_memory_cap":
                    raise MemoryReached(summary["stop_reason"])
                raise BudgetReached(summary["stop_reason"])
            check = {"bank_id": bank, **verify_random_slots(output / f"control_banks/bank{bank}/collection_steps.csv", directory_out / "collection_steps.csv")}
            self.random_checks.append(check)
            if not check["identical"]:
                raise ConsistencyError("new random slots0..7 differ from archived original streams")
            table, table_summary = build_recorded_table(directory_out / "collection_steps.csv", data, arrays["support_indices"], enforce)
            mask = table["observed"]
            if not table_summary["all_nonterminal_successors_in_support_verified"] or not all(np.array_equal(table[k][mask], transitions[k][mask]) for k in ("rewards", "ends", "successor_indices")):
                raise ConsistencyError("new logged graph fails closure or transition integrity check")
            self.tables[bank], self.treatment_supports[bank] = table, arrays["support_indices"]
            self.tensors[bank] = {k: torch.from_numpy(v.astype(np.float32) if k == "rewards" else v.copy()) for k, v in table.items()}
            self.tensor_before[bank] = array_metadata({k: v.numpy() for k, v in self.tensors[bank].items()})
            self.table_summaries.append({"bank_id": bank, "condition": CONDITIONS[1], **table_summary, "source": artifact_path(directory_out / "collection_steps.csv"),
                "source_sha256": shared.sha(directory_out / "collection_steps.csv"), "optimizer_inputs": "only own recorded outcomes/successor masks, never all-action oracle targets"})
            with (directory_out / "collection_episodes.csv").open(newline="") as f:
                self.guided_routes[bank] = {int(r["map_seed"]): r["trajectory_sha256"] for r in csv.DictReader(f) if int(r["repetition"]) == 8}
            for condition, path in ((CONDITIONS[0], output / f"control_banks/bank{bank}/collection_steps.csv"), (CONDITIONS[1], directory_out / "collection_steps.csv")):
                self.collection_trajectories.extend(collection_replays(path, WorldConfig(), data, bank, condition, set(range(300000, 300256, 32))))
            np.savez_compressed(output / "recorded_transitions.npz", **{f"bank{b}_{k}": v for b, t in self.tables.items() for k, v in t.items()})
            write_json(output / "collection_trajectories.json", self.collection_trajectories)
            self.prepare_routes(bank, data, enforce)
        self.collector.update(source_after=shared.sha(shared.ROOT / self.collector["source"]), copy_after=shared.sha(output / "collector.pt"),
            online_after=collector.parameter_hash(), target_after=module_hash(collector.target))
        self.collector["unchanged"] = all(self.collector[k + "_before"] == self.collector[k + "_after"] for k in ("source", "copy", "online", "target"))
        if not self.collector["unchanged"] or any(routes != next(iter(self.guided_routes.values())) for routes in self.guided_routes.values()):
            raise ConsistencyError("collector changed or guided routes differ across banks")
        write_json(output / "collector.json", self.collector)
        enforce()

    def prepare_routes(self, bank, data, enforce):
        env, lookup = CollectionWorld(), make_row_lookup(data)
        for condition in self.conditions:
            table = self.table_for_condition(bank, condition)
            reachable, shortest, summary = logged_success_paths(table, data["remaining"], data["map_seeds"], enforce)
            arrays = {f"bank{bank}_{condition}_reachable": reachable, f"bank{bank}_{condition}_shortest_steps": shortest}
            self.route_arrays.update(arrays)
            self.route_before.update(array_metadata(arrays))
            starts = []
            for map_seed in np.unique(data["map_seeds"]):
                enforce()
                observation, _ = env.reset(seed=int(map_seed))
                row = lookup[(int(map_seed), *env.position, env.config.horizon)]
                if not table["observed"][row].any():
                    raise ConsistencyError("collected graph omits original start")
                physical = physical_shortest_steps(observation)
                starts.append({"bank_id": bank, "condition": condition, "map_seed": int(map_seed), "state_row": row,
                    "logged_success_reachable": bool(reachable[row]), "logged_shortest_steps": int(shortest[row]) if reachable[row] else None,
                    "planner_steps": physical, "logged_efficient_success_reachable": bool(reachable[row] and shortest[row] <= 2 * physical)})
            self.route_starts.extend(starts)
            self.route_rows.append({"bank_id": bank, "condition": condition, **summary, "starts": len(starts),
                "success_reachable_starts": sum(r["logged_success_reachable"] for r in starts), "success_ceiling": float(np.mean([r["logged_success_reachable"] for r in starts])),
                "efficient_success_reachable_starts": sum(r["logged_efficient_success_reachable"] for r in starts),
                "efficient_success_ceiling": float(np.mean([r["logged_efficient_success_reachable"] for r in starts]))})
        np.savez_compressed(self.output / "route_ceilings.npz", **self.route_arrays)
        write_json(self.output / "route_ceilings.json", {"per_condition": self.route_rows, "by_start": self.route_starts})

    def table_for_condition(self, bank, condition):
        return self.control_tables[bank] if condition == CONDITIONS[0] else self.tables[bank]

    def prepare_support(self, bank, support, data, enforce):
        if not np.array_equal(support, self.original_supports[bank]):
            raise ConsistencyError("historical random support differs")
        return {CONDITIONS[0]: support, CONDITIONS[1]: self.treatment_supports[bank]}

    def bank_metadata(self, bank, pair, data):
        left, right = pair[CONDITIONS[0]], pair[CONDITIONS[1]]
        intersection = len(np.intersect1d(left, right))
        return {"support_sizes": {c: len(v) for c, v in pair.items()}, "support_size_definition": "inherited support_size is historical control; use support_sizes for each arm",
            "identical_state_support": np.array_equal(left, right),
            "intersection": {"states": intersection, "union_states": len(left) + len(right) - intersection, "fraction_of_control": intersection / len(left),
                "fraction_of_treatment": intersection / len(right), "jaccard": intersection / (len(left) + len(right) - intersection)},
            "collection": next(r for r in self.collections if r["bank_id"] == bank),
            "recorded_actions": {"per_condition": [r for r in self.control_summaries + self.table_summaries if r["bank_id"] == bank]}}

    def record_baseline(self, bank, seed, support, data, global_counts, local_counts, archived, enforce, updates):
        old_map = archived["map_index_sha256"]
        super().record_baseline(bank, seed, support, data, global_counts, local_counts, archived, enforce, updates)
        check = self.reconstruction[-1]
        check["map_digest_identical"] = check["map_index_sha256"] == old_map
        check["map_counts_identical"] = check["count_arrays"]["map"] == array_metadata({"map": self.control_maps[(bank, seed)]})["map"]
        check["verified"] = check["verified"] and check["map_digest_identical"] and check["map_counts_identical"]
        if not check["verified"]:
            raise ConsistencyError("historical random-control map stream differs from archive")
        archived.pop("compared_prefix", None)

    def compute_update(self, agent, observations, recorded, indices, bank, seed):
        return constrained_update(agent, observations, recorded, indices)[0]

    def consistency(self, samplers, bank_ids, seeds, updates):
        results = []
        for bank in bank_ids:
            for seed in seeds:
                sampler = samplers.get((bank, CONDITIONS[1], seed))
                check = {"bank_id": bank, "seed": seed, "paired_with_control": False, "complete": False}
                if sampler:
                    support = sampler.support
                    actual_maps = np.bincount(np.searchsorted(sampler.map_ids, sampler.map_seeds), weights=sampler.counts, minlength=len(sampler.map_ids)).astype(np.uint64)
                    check.update(updates=sampler.updates, support_states=len(support), local_global_counts_agree=np.array_equal(sampler.local_counts, sampler.counts[support]),
                        no_direct_off_support_samples=int(sampler.counts.sum()) == int(sampler.counts[support].sum()), map_counts_agree=np.array_equal(actual_maps, sampler.map_counts),
                        state_presentations_match=int(sampler.counts.sum()) == updates * 64,
                        all_supported_states_sampled=bool(np.all(sampler.counts[support] > 0)))
                    check["complete"] = sampler.updates == updates and all(check[k] for k in ("local_global_counts_agree", "no_direct_off_support_samples", "map_counts_agree", "state_presentations_match"))
                    if updates == 30000:
                        check["complete"] &= check["all_supported_states_sampled"]
                results.append(check)
        return results

    def finalize_exposure(self, data, transitions, supports, counts, samples, exposure, output):
        self.output = output
        records, checks = [], []
        for (bank, condition, seed), count in counts.items():
            support = supports[bank][condition]
            membership = np.zeros(len(count), bool)
            membership[support] = True
            table = self.table_for_condition(bank, condition)
            recorded_condition = condition == self.conditions[1] or self.control_uses_recorded
            if recorded_condition:
                mask, ended, successor = table["observed"], table["ends"], table["successor_indices"]
            else:
                mask, ended, successor = np.ones_like(transitions["ends"]), transitions["ends"], transitions["successor_indices"]
            live = mask & ~ended
            outside = live & ~membership[np.maximum(successor, 0)]
            target_count = int(np.dot(count.astype(np.uint64), mask.sum(1).astype(np.uint64)))
            nonterminal_count = int(np.dot(count.astype(np.uint64), live.sum(1).astype(np.uint64)))
            neural_queries = condition == self.conditions[0] or self.uses_neural_successors
            query_count = nonterminal_count if neural_queries else 0
            outside_count = int(np.dot(count.astype(np.uint64), outside.sum(1).astype(np.uint64))) if neural_queries else 0
            record = {"bank_id": bank, "condition": condition, "seed": seed, "source": "archived_baseline" if condition == self.conditions[0] else "new_treatment",
                "updates": samples[str(bank)][condition][str(seed)]["updates"], "state_presentations": int(count.sum()), "action_target_presentations": target_count,
                "terminal_action_targets": target_count - nonterminal_count, "nonterminal_action_targets": nonterminal_count, "nonterminal_target_queries": query_count,
                "outside_support_target_queries": outside_count, "outside_support_fraction": outside_count / query_count if query_count else None,
                "action_target_definition": "four outcomes perstate" if not recorded_condition else "distinct logged outcomes perstate; perstate mean loss",
                "query_provenance": "historical counterfactual all-action transitions" if not recorded_condition else "only nonterminal successors from manifest-verified logged edges"}
            records.append(record)
            if condition == self.conditions[1]:
                actual = self.optimizer_counts.get((bank, seed), {})
                expected_queries = np.zeros(len(count), np.uint64)
                source_rows, actions = np.nonzero(live)
                if neural_queries:
                    np.add.at(expected_queries, successor[source_rows, actions], count[source_rows].astype(np.uint64))
                check = {"bank_id": bank, "seed": seed,
                    "tracked_exposure_matches": all(actual.get(k) == record[k] for k in ("updates", "state_presentations", "action_target_presentations", "nonterminal_target_queries")),
                    "recorded_query_counts_match": np.array_equal(self.query_counts.get((bank, seed)), expected_queries),
                    "no_outside_support_queries": outside_count == 0}
                checks.append(check)
            if recorded_condition:
                summary = next(r for r in exposure["summaries"] if (r["bank_id"], r["condition"], r["seed"]) == (bank, condition, seed))
                summary["all_action_structural_successor_queries"] = summary["successor_queries"]
                summary["successor_queries"] = {"nonterminal_queries": query_count, "outside_support_queries": outside_count,
                    "outside_support_fraction": outside_count / query_count if query_count else None, "source": "recorded edges only"}
        self.action_exposure = {"classification": "actual supervised action targets; baseline historical, treatment newly recorded-only",
            "per_seed": records, "integrity": checks}
        for bank, table in self.tables.items():
            before = next(r["arrays"] for r in self.table_summaries if r["bank_id"] == bank)
            after = array_metadata(table)
            tensor_after = array_metadata({k: v.numpy() for k, v in self.tensors[bank].items()})
            self.table_integrity.append({"bank_id": bank, "before": before, "after": after, "unchanged": before == after,
                "read_only": all(not v.flags.writeable for v in table.values()), "tensor_before": self.tensor_before[bank],
                "tensor_after": tensor_after, "optimizer_tables_unchanged": self.tensor_before[bank] == tensor_after})
        np.savez_compressed(output / "recorded_query_counts.npz", **{f"bank{b}_{self.conditions[1]}_seed{s}": v for (b, s), v in self.query_counts.items()})
        write_json(output / "action_exposure.json", self.action_exposure)
        integrity = {"complete": len(self.tables) == len(self.bank_ids) and all(all(r[k] for k in ("tracked_exposure_matches", "recorded_query_counts_match", "no_outside_support_queries")) for r in checks)
            and all(r["unchanged"] and r["read_only"] and r["optimizer_tables_unchanged"] for r in self.table_integrity)}

        vectors = {f"bank{b}_{CONDITIONS[0]}_seed{s}": v for (b, s), v in self.control_queries.items()}
        vectors.update({f"bank{b}_{CONDITIONS[1]}_seed{s}": v for (b, s), v in self.query_counts.items()})
        np.savez_compressed(output / "recorded_query_counts.npz", **vectors)
        for row in self.action_exposure["per_seed"]:
            if row["condition"] != CONDITIONS[0]:
                continue
            bank, seed = row["bank_id"], row["seed"]
            old = next(r for r in self.control_actions if (r["bank_id"], r["condition"], r["seed"]) == (bank, CONDITIONS[0], seed))
            table = self.control_tables[bank]
            source, actions = np.nonzero(table["observed"] & ~table["ends"])
            expected = np.zeros(len(data["observations"]), np.uint64)
            np.add.at(expected, table["successor_indices"][source, actions], counts[(bank, CONDITIONS[0], seed)][source].astype(np.uint64))
            fields = ("updates", "state_presentations", "action_target_presentations", "terminal_action_targets", "nonterminal_target_queries")
            self.control_checks.append({"bank_id": bank, "seed": seed, "action_counts_match_archive": all(row[k] == old[k] for k in fields),
                "query_vector_matches_archive": np.array_equal(expected, self.control_queries[(bank, seed)])})
        for bank, table in self.control_tables.items():
            before = next(r["arrays"] for r in self.control_summaries if r["bank_id"] == bank)
            self.control_identity.append({"bank_id": bank, "unchanged_after_preparation_training": before == array_metadata(table),
                "read_only": all(not v.flags.writeable for v in table.values())})
        self.collection_integrity = {"expected_banks": len(self.bank_ids), "completed_banks": len(self.collections),
            "expected_episodes": len(self.bank_ids) * 4096, "actual_episodes": sum(r["collection_episodes"] for r in self.collections),
            "complete_episodes": sum(r["complete_collection_episodes"] for r in self.collections),
            "expected_collection_recordings": len(self.bank_ids) * 32, "actual_collection_recordings": len(self.collection_trajectories),
            "all_recordings_complete": all(r["complete"] for r in self.collection_trajectories),
            "all_guided_slots_identical_within_map": all(c["episodes"] == 8 and c["unique_trajectories"] == 1 for r in self.collections for c in r["guided_duplicate_checks"]),
            "guided_routes_identical_across_banks": bool(self.guided_routes) and all(v == next(iter(self.guided_routes.values())) for v in self.guided_routes.values()),
            "all_random_slots_match": len(self.random_checks) == len(self.bank_ids) and all(r["identical"] for r in self.random_checks),
            "route_arrays_unchanged": self.route_before == array_metadata(self.route_arrays)}
        if self.collector:
            self.collector["source_final"] = shared.sha(shared.ROOT / self.collector["source"])
            self.collector["copy_final"] = shared.sha(output / self.collector["saved"])
            self.collector["files_unchanged_through_training"] = self.collector["source_final"] == self.collector["source_before"] == self.collector["copy_final"] == self.collector["copy_before"]
        ci = self.collection_integrity
        ci["complete"] = (ci["completed_banks"] == ci["expected_banks"] and ci["actual_episodes"] == ci["complete_episodes"] == ci["expected_episodes"]
            and ci["actual_collection_recordings"] == ci["expected_collection_recordings"]
            and all(ci[k] for k in ("all_recordings_complete", "all_guided_slots_identical_within_map", "guided_routes_identical_across_banks", "all_random_slots_match", "route_arrays_unchanged"))
            and self.collector.get("unchanged", False) and self.collector.get("files_unchanged_through_training", False) and all(r["collector_parameters_rng_unchanged"] for r in self.collections))
        complete = (integrity["complete"] and ci["complete"] and len(self.control_checks) == len(self.bank_ids) * len(self.seeds)
            and all(r["action_counts_match_archive"] and r["query_vector_matches_archive"] for r in self.control_checks)
            and all(r.get("recorded_tables_identical", True) and r.get("support_identical", True) and r.get("unchanged_after_preparation_training", True) and r.get("read_only", True) for r in self.control_identity))
        self.action_exposure["classification"] = "own-arm recorded targets; equal updates/state presentations, intentionally unequal action targets and successor counts"
        write_json(output / "action_exposure.json", self.action_exposure)
        return {"complete": complete}

    def provenance(self):
        return {**super().provenance(), "collector": self.collector, "collection_integrity": getattr(self, "collection_integrity", {}),
            "random_slot_replication": self.random_checks, "control_recorded_identity": self.control_identity, "control_exposure_identity": self.control_checks,
            "optimization_access": "unchanged constrained_update using treatment's recorded outcomes and successor masks; no oracle targets",
            "replay_pairing_claim": "none across arms: changed support sizes alter realized local/global/map streams"}

    def collection_result(self, capture_controls=False):
        if not capture_controls:
            return {"classification": "equal complete episode allocation, unequal interactions and prior collector knowledge",
                "per_bank": self.collections, "per_condition": self.control_collection_summaries + [{**r, "condition": CONDITIONS[1], "source": "new_mixed_collection"} for r in self.collections]}
        controls = []
        for bank in self.bank_ids:
            path = self.output / f"control_banks/bank{bank}/collection_episodes.csv"
            if not path.exists():
                continue
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            route_hashes = set()
            path_steps = path.parent / "collection_steps.csv"
            previous, digest = None, None
            with path_steps.open(newline="") as f:
                for index, row in enumerate(csv.DictReader(f)):
                    if index % 256 == 0:
                        self.enforce()
                    key = int(row["map_seed"]), int(row["repetition"])
                    if key != previous:
                        if digest is not None:
                            route_hashes.add(digest.hexdigest())
                        digest, previous = hashlib.sha256(), key
                    step = {k: float(row[k]) if k == "reward" else int(row[k]) for k in STEP_FIELDS if k != "repetition"}
                    digest.update(json.dumps(step, sort_keys=True, separators=(",", ":")).encode())
            if digest is not None:
                route_hashes.add(digest.hexdigest())
            controls.append({"bank_id": bank, "condition": CONDITIONS[0], "source": "archived_random_collection", "collection_episodes": len(rows),
                "complete_collection_episodes": sum(int(r["complete"]) for r in rows), "collection_steps": sum(int(r["steps"]) for r in rows),
                "collection_successes": sum(int(r["success"]) for r in rows), "collection_noop_steps": sum(int(r["noop_steps"]) for r in rows),
                "per_policy": [{"policy": "random", "episodes": len(rows), "complete_episodes": sum(int(r["complete"]) for r in rows),
                    "steps": sum(int(r["steps"]) for r in rows), "successes": sum(int(r["success"]) for r in rows), "noop_steps": sum(int(r["noop_steps"]) for r in rows),
                    "unique_trajectories": len(route_hashes), "mean_steps": float(np.mean([int(r["steps"]) for r in rows])) if rows else None}]})
        return {"classification": "equal complete episode allocation, unequal interactions and prior collector knowledge",
            "per_bank": self.collections, "per_condition": controls + [{**r, "condition": CONDITIONS[1], "source": "new_mixed_collection"} for r in self.collections]}

    def configure_result(self, result):
        super().configure_result(result)
        result["run"]["interpretation"] = "guided_collection_descriptive_only" if result["run"]["interpretation"] == "recorded_actions_descriptive_only" else result["run"]["interpretation"]
        result["run"].update(collection_steps=sum(r["collection_steps"] for r in self.collections), collection_episodes=sum(r["collection_episodes"] for r in self.collections),
            complete_collection_episodes=sum(r["complete_collection_episodes"] for r in self.collections), collection_wall_seconds=sum(r["wall_seconds"] for r in self.collections),
            new_recorded_supports=len(self.treatment_supports), support_draws=0, collection_recordings=len(self.collection_trajectories),
            fresh_recordings=len(result["trajectories"]), total_recordings=len(result["trajectories"]) + len(self.collection_trajectories))
        result["run"]["limitations"] = ["Fixed8random+8greedy episodes per map changes interactions, states, actions, route availability and replay distribution together.",
            "Guidance comes from one pretrained bank1/seed0 constrained model with prior100878 interactions/30000updates; this is extra overlapping historical knowledge.",
            "Eight greedy slots duplicate one route per map and guided routes are shared across banks; duplicates add no loss weight.",
            "Actual action-target and neural-query counts differ; only learner updates and sampled-state presentations are matched.",
            "Frozen-policy collection followed by offline learning is not continuous online adaptation or memory; no new competence gate or significance claim."]
        result["robustness"]["classification"] = "descriptive total collection-policy effect; no new gates or significance claims"
        result["provenance"]["replay_sampling_integrity"] = result["provenance"].pop("paired_map_sampling_consistency")
        result["collection"], result["collector"] = self.collection_result(), self.collector
        result["route_ceilings"] = {"classification": "recorded successful-path ceilings at original training starts; not fresh-policy performance", "per_condition": self.route_rows, "by_start": self.route_starts,
            "arrays": self.route_before, "computed_before_training": True}
        result["collection_trajectories"] = self.collection_trajectories
        result["action_coverage"] = {"per_bank": self.table_summaries, "per_condition": self.control_summaries + self.table_summaries}
        for name in ("collection", "collector", "route_ceilings", "collection_trajectories", "action_coverage"):
            write_json(self.output / f"{name}.json", result[name])
            result["artifacts"][name] = artifact_path(self.output / f"{name}.json")
        result["artifacts"]["route_ceiling_arrays"] = artifact_path(self.output / "route_ceilings.npz")


def run_study(output, protocol_file, **kwargs):
    kwargs.setdefault("panel_seed_start", 1140000)
    comparison = GuidedCollectionComparison()
    comparison.output = Path(output)
    return shared.run_study(output, protocol_file, _comparison=comparison, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--bank-ids", default="1,2,3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--panel-count", type=int, default=8)
    parser.add_argument("--maps-per-panel", type=int, default=64)
    parser.add_argument("--panel-seed-start", type=int, default=1140000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    for name in shared.ARCHIVES + GuidedCollectionComparison.extra_archives:
        parser.add_argument("--" + name.replace("_", "-") + "-dir", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.bank_ids, args.seeds, args.updates, args.panel_count, args.maps_per_panel, args.panel_seed_start = "1", "0", 24, 2, 2, 1200000
    archives = {name: getattr(args, name + "_dir") for name in shared.ARCHIVES + GuidedCollectionComparison.extra_archives if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, bank_ids=[int(v) for v in args.bank_ids.split(",")], seeds=[int(v) for v in args.seeds.split(",")], updates=args.updates,
        panel_count=args.panel_count, maps_per_panel=args.maps_per_panel, panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
