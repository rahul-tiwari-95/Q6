"""Evaluate nine frozen policies across preselected disjoint fresh panels.

No optimizer, replay, collection, support selection or learner update is created.
Panel ranges and threshold counts are descriptive; no competence gate is added.
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
from torch import nn

from .adaptation import write_json
from .competence import BudgetReached, EVAL_FIELDS, ROOT, artifact_path, git_info, summarize
from .fixed_targets import ConsistencyError, MemoryReached, efficient_rate, guard, module_hash, peak_rss_bytes, sha
from .optimal import VisibleOptimalQ
from .supervised import layout_key, rollout, save_manifest
from .world import CollectionWorld, WorldConfig

CONDITIONS = ("collected_unique", "uniform_subset", "exhaustive")
COMPARISONS = ({"id": "uniform_minus_collected", "left": "collected_unique", "right": "uniform_subset"},
               {"id": "exhaustive_minus_collected", "left": "collected_unique", "right": "exhaustive"},
               {"id": "exhaustive_minus_uniform", "left": "uniform_subset", "right": "exhaustive"})
DELTA_METRICS = ("success_rate", "efficient_success_rate", "mean_steps", "noop_rate")
PAIR_FIELDS = ["comparison", "panel", "seed", "map_seed", "mode", "repetition", "success_delta",
               "efficient_success_delta", "steps_delta", "noop_rate_delta"]


class FrozenPolicy:
    """Inference-only architecture matching the archived 92→128→64→4 weights."""
    def __init__(self, snapshot):
        if snapshot["observation_size"] != 92:
            raise ConsistencyError("frozen policy observation size differs from World A")
        self.observation_size = snapshot["observation_size"]
        with torch.random.fork_rng(devices=[]):
            self.online = nn.Sequential(nn.Linear(92, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
            self.target = nn.Sequential(nn.Linear(92, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
        self.online.load_state_dict(snapshot["online"])
        self.target.load_state_dict(snapshot["target"])
        self.online.eval().requires_grad_(False)
        self.target.eval().requires_grad_(False)
        if self.parameter_hash() != snapshot["parameter_hash"] or module_hash(self.target) != snapshot["target_parameter_hash"]:
            raise ConsistencyError("frozen policy tensor hashes differ from checkpoint metadata")

    def parameter_hash(self):
        return module_hash(self.online)


def select_panels(config, excluded, panel_count, maps_per_panel, start, stride, enforce):
    env = CollectionWorld(config)
    accepted, panels, skips = set(), [], []
    for index in range(panel_count):
        panel = {"id": f"panel_{index}", "scan_start": start + index * stride, "layouts": [], "map_seeds": []}
        candidate = panel["scan_start"]
        while len(panel["layouts"]) < maps_per_panel:
            enforce()
            env.reset(seed=candidate)
            key = layout_key(env)
            reason = next((name for name, hashes in excluded.items() if key in hashes), None)
            if reason is None and key in accepted:
                reason = "earlier_selected_layout"
            record = {"map_seed": candidate, "layout_hash": key, "original_start": list(env.position)}
            if reason:
                skips.append({"panel": panel["id"], **record, "reason": reason})
            else:
                panel["layouts"].append(record)
                panel["map_seeds"].append(candidate)
                accepted.add(key)
            candidate += 1
        panels.append(panel)
    pairs = [{"left": left["id"], "right": right["id"], "intersection": len(
        {r["layout_hash"] for r in left["layouts"]} & {r["layout_hash"] for r in right["layouts"]})}
        for index, left in enumerate(panels) for right in panels[index + 1:]]
    return {"status": "complete", "panels": panels, "collision_skips": skips,
        "pairwise_panel_intersections": pairs, "exclusion_intersections": {name: len(accepted & hashes) for name, hashes in excluded.items()},
        "unique_selected_layouts": len(accepted), "excluded_layout_counts": {k: len(v) for k, v in excluded.items()},
        "selection_uses_outcomes": False, "layout_identity": "walls + pellet + rules, ignoring start, clock and seed"}


def reduce_episodes(rows, planner):
    successful = [r for r in rows if r["success"]]
    successful_steps = sum(r["steps"] for r in successful)
    return {**summarize(rows), "efficient_success_rate": efficient_rate(rows, planner)
        if all((r["panel"], r["map_seed"]) in planner for r in rows) else None,
        "efficient_success_episodes": sum(bool(r["success"]) and r["steps"] <= 2 * planner[(r["panel"], r["map_seed"])] for r in rows)
            if all((r["panel"], r["map_seed"]) in planner for r in rows) else None,
        "noop_steps": sum(r["noop_steps"] for r in rows),
        "successful_episodes": len(successful), "successful_episode_steps": successful_steps,
        "successful_noop_steps": sum(r["noop_steps"] for r in successful),
        "successful_mean_steps": successful_steps / len(successful) if successful else None,
        "successful_noop_rate": sum(r["noop_steps"] for r in successful) / successful_steps if successful_steps else None}


def summarize_panels(rows, refs, panels, seeds, *, conditions=CONDITIONS, comparisons=COMPARISONS):
    rows = [r for r in rows if r["checkpoint_complete"]]
    refs = [r for r in refs if r["checkpoint_complete"]]
    planner = {(r["panel"], r["map_seed"]): r["steps"] for r in refs if r["policy"] == "shortest_path"}
    seed_results, aggregate, references = [], [], []
    for panel in [p["id"] for p in panels] + ["all"]:
        selected = [r for r in rows if panel == "all" or r["panel"] == panel]
        for condition in conditions:
            for mode in ("greedy", "epsilon_0_1"):
                group = [r for r in selected if r["condition"] == condition and r["mode"] == mode]
                if not group:
                    continue
                summaries = []
                for seed in seeds:
                    per_seed = [r for r in group if r["seed"] == seed]
                    if per_seed:
                        item = {"condition": condition, "panel": panel, "seed": seed, "mode": mode, **reduce_episodes(per_seed, planner)}
                        seed_results.append(item)
                        summaries.append(item)
                aggregate.append({"condition": condition, "panel": panel, "mode": mode, **reduce_episodes(group, planner),
                    "seeds": len(summaries), "seed_success_min": min(r["success_rate"] for r in summaries),
                    "seed_success_max": max(r["success_rate"] for r in summaries)})
        for policy in ("random_actions", "shortest_path"):
            group = [r for r in refs if r["policy"] == policy and (panel == "all" or r["panel"] == panel)]
            if group:
                references.append({"condition": "shared", "policy": policy, "panel": panel, "mode": "reference", **reduce_episodes(group, planner)})
    paired = {"comparisons": list(comparisons), "scope": "greedy only; right minus left on matched panel, seed and map",
        "per_seed": [], "aggregate": [], "per_layout": []}
    lookup = {(r["condition"], r["panel"], r["seed"], r["mode"]): r for r in seed_results}
    pooled_lookup = {(r["condition"], r["panel"], r["mode"]): r for r in aggregate}
    for comparison in comparisons:
        for panel in [p["id"] for p in panels] + ["all"]:
            for mode in ("greedy",):
                values = []
                for seed in seeds:
                    left = lookup.get((comparison["left"], panel, seed, mode))
                    right = lookup.get((comparison["right"], panel, seed, mode))
                    if left and right:
                        item = {"comparison": comparison["id"], "panel": panel, "seed": seed, "mode": mode,
                            **{k + "_delta": right[k] - left[k] if right[k] is not None and left[k] is not None else None for k in DELTA_METRICS}}
                        paired["per_seed"].append(item)
                        values.append(item)
                if values:
                    pooled_left = pooled_lookup[(comparison["left"], panel, mode)]
                    pooled_right = pooled_lookup[(comparison["right"], panel, mode)]
                    paired["aggregate"].append({"comparison": comparison["id"], "panel": panel, "mode": mode, "seeds": len(values),
                        "pooled_noop_rate_delta": pooled_right["noop_rate"] - pooled_left["noop_rate"],
                        **{"mean_seed_" + k + "_delta": float(np.mean([r[k + "_delta"] for r in values]))
                           if all(r[k + "_delta"] is not None for r in values) else None for k in DELTA_METRICS}})
        raw_lookup = {(r["condition"], r["panel"], r["seed"], r["map_seed"], r["mode"], r["repetition"]): r for r in rows if r["mode"] == "greedy"}
        for key, left in raw_lookup.items():
            if key[0] != comparison["left"]:
                continue
            right = raw_lookup.get((comparison["right"], *key[1:]))
            if right is None:
                continue
            length = planner.get((left["panel"], left["map_seed"]))
            paired["per_layout"].append({"comparison": comparison["id"], "panel": left["panel"], "seed": left["seed"],
                "map_seed": left["map_seed"], "mode": left["mode"], "repetition": left["repetition"],
                "success_delta": int(right["success"]) - int(left["success"]),
                "efficient_success_delta": int(bool(right["success"]) and right["steps"] <= 2 * length)
                    - int(bool(left["success"]) and left["steps"] <= 2 * length) if length is not None else None,
                "steps_delta": right["steps"] - left["steps"],
                "noop_rate_delta": right["noop_steps"] / right["steps"] - left["noop_steps"] / left["steps"]})
    return seed_results, aggregate, references, paired


def descriptive_summaries(seed_results, aggregate, references, paired, panels, seeds, eligible, *, conditions=CONDITIONS, comparisons=COMPARISONS):
    thresholds = {"classification": "descriptive historical threshold counts, not new competence gates",
        "eligible": eligible, "success_reference": .7, "efficient_success_reference": .8,
        "per_seed": [], "per_condition": []}
    random = {r["panel"]: r["success_rate"] for r in references if r["policy"] == "random_actions"}
    for row in seed_results:
        if row["panel"] == "all" or row["mode"] != "greedy":
            continue
        efficiency = row["efficient_success_rate"]
        thresholds["per_seed"].append({"condition": row["condition"], "panel": row["panel"], "seed": row["seed"],
            "success_rate": row["success_rate"], "efficient_success_rate": efficiency,
            "panel_random_success": random.get(row["panel"]),
            "success_reference_met": bool(eligible and row["success_rate"] >= .7 and row["panel"] in random and row["success_rate"] > random[row["panel"]]),
            "efficiency_reference_met": bool(eligible and efficiency is not None and efficiency >= .8)})
    for condition in conditions:
        outcomes = []
        for panel in panels:
            group = [r for r in thresholds["per_seed"] if r["condition"] == condition and r["panel"] == panel["id"]]
            outcomes.append({"panel": panel["id"], "complete_seeds": len(group),
                "all_seed_success_reference_met": len(group) == len(seeds) and all(r["success_reference_met"] for r in group),
                "all_seed_efficiency_reference_met": len(group) == len(seeds) and all(r["efficiency_reference_met"] for r in group)})
        thresholds["per_condition"].append({"condition": condition, "panels": len(panels),
            "success_reference_panels": sum(r["all_seed_success_reference_met"] for r in outcomes),
            "efficiency_reference_panels": sum(r["all_seed_efficiency_reference_met"] for r in outcomes), "per_panel": outcomes})
    robustness = {"classification": "descriptive panel ranges and sign counts; no confidence intervals, significance or equivalence claims",
        "primary_comparison": "uniform_minus_collected", "primary_metric": "efficient_success_rate_delta", "primary_mode": "greedy",
        "paired_metric_aggregation": "arithmetic mean of seed-wise deltas; pooled no-op-rate differences are recorded separately in paired aggregates",
        "per_comparison": [], "per_condition": []}
    for comparison in comparisons:
        for mode in ("greedy",):
            groups = [r for r in paired["aggregate"] if r["comparison"] == comparison["id"] and r["panel"] != "all" and r["mode"] == mode]
            item = {"comparison": comparison["id"], "mode": mode, "panels": len(groups), "metrics": {}}
            for metric in DELTA_METRICS:
                values = [r["mean_seed_" + metric + "_delta"] for r in groups if r["mean_seed_" + metric + "_delta"] is not None]
                if values:
                    item["metrics"][metric + "_delta"] = {"minimum": min(values), "maximum": max(values), "mean": float(np.mean(values)),
                        "positive_panels": sum(v > 1e-12 for v in values), "negative_panels": sum(v < -1e-12 for v in values),
                        "zero_panels": sum(abs(v) <= 1e-12 for v in values), "panels": len(values), "sign_zero_tolerance": 1e-12}
            robustness["per_comparison"].append(item)
    for condition in conditions:
        for mode in ("greedy", "epsilon_0_1"):
            groups = [r for r in aggregate if r["condition"] == condition and r["panel"] != "all" and r["mode"] == mode]
            robustness["per_condition"].append({"condition": condition, "mode": mode, "panels": len(groups),
                "metrics": {metric: {"minimum": min(r[metric] for r in groups), "maximum": max(r[metric] for r in groups)}
                    for metric in DELTA_METRICS if groups and all(r[metric] is not None for r in groups)}})
    return thresholds, robustness


def verify_expected_cells(rows, refs, panels, seeds, *, conditions=CONDITIONS):
    expected = {(condition, panel["id"], seed, layout["map_seed"], mode, rep)
        for condition in conditions for panel in panels for seed in seeds for layout in panel["layouts"]
        for mode in ("greedy", "epsilon_0_1") for rep in range(1 if mode == "greedy" else 2)}
    actual = [(r["condition"], r["panel"], r["seed"], r["map_seed"], r["mode"], r["repetition"])
        for r in rows if r["checkpoint_complete"]]
    expected_refs = {(policy, panel["id"], seed, layout["map_seed"], rep)
        for policy in ("random_actions", "shortest_path") for panel in panels
        for seed in (seeds if policy == "random_actions" else [0]) for layout in panel["layouts"]
        for rep in range(2 if policy == "random_actions" else 1)}
    actual_refs = [(r["policy"], r["panel"], r["seed"], r["map_seed"], r["repetition"])
        for r in refs if r["checkpoint_complete"]]
    return {"expected_learner_cells": len(expected), "complete_learner_cells": len(actual),
        "unique_complete_learner_cells": len(set(actual)), "expected_reference_cells": len(expected_refs),
        "complete_reference_cells": len(actual_refs), "unique_complete_reference_cells": len(set(actual_refs)),
        "complete": set(actual) == expected and len(actual) == len(expected) and
                    set(actual_refs) == expected_refs and len(actual_refs) == len(expected_refs)}


def run_study(output, protocol_file, *, seeds=(0, 1, 2), panel_count=8, maps_per_panel=64,
              panel_seed_start=970000, panel_stride=1000, max_seconds=1200, max_rss_bytes=4 * 1024**3,
              dashboard=None, archives=None, smoke=False):
    started = time.monotonic()
    output, protocol_file = Path(output), Path(protocol_file)
    seeds = list(seeds)
    if not protocol_file.is_file() or not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("existing protocol and distinct nonnegative seeds are required")
    if min(panel_count, maps_per_panel, panel_stride, max_seconds, max_rss_bytes) <= 0 or panel_seed_start < 0:
        raise ValueError("positive budgets and nonnegative panel start are required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    directories = {name: ROOT / f"experiments/{name}/pilot_v1" for name in ("supervised", "fixed_targets", "coverage", "equal_support")}
    if archives:
        directories.update({name: Path(path) for name, path in archives.items()})
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "torch_threads": 1}
    actual = dict(seeds=seeds, panel_count=panel_count, maps_per_panel=maps_per_panel,
        panel_seed_start=panel_seed_start, panel_stride=panel_stride, max_seconds=max_seconds, max_rss_bytes=max_rss_bytes)
    declared = dict(seeds=[0, 1, 2], panel_count=8, maps_per_panel=64, panel_seed_start=970000,
        panel_stride=1000, max_seconds=1200, max_rss_bytes=4 * 1024**3)
    deviations = [{"field": k, "actual": v, "declared": declared[k]} for k, v in actual.items() if v != declared[k]]
    if not runtime["python"].startswith("3.12.") or runtime["torch"].split("+")[0] != "2.8.0" or runtime["numpy"] != "2.0.2":
        deviations.append({"field": "runtime", "actual": runtime, "declared": "Python3.12/Torch2.8.0/NumPy2.0.2"})
    for name, directory in directories.items():
        expected = ROOT / f"experiments/{name}/pilot_v1"
        if directory.resolve() != expected.resolve():
            deviations.append({"field": "archive_" + name, "actual": artifact_path(directory), "declared": artifact_path(expected)})
    config = WorldConfig()
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    protocol = {"id": "panel-evaluation-v1", "question": "How consistently do these frozen policy differences recur across fresh panels?",
        "conditions": [{"id": c, "label": {"collected_unique": "Collected unique", "uniform_subset": "Uniform subset", "exhaustive": "Exhaustive"}[c]} for c in CONDITIONS],
        "seeds": seeds, "panels": [], "panel_selection": {"count": panel_count, "maps_per_panel": maps_per_panel,
            "start": panel_seed_start, "stride": panel_stride, "selected_before_evaluation": True},
        "world": asdict(config), "rule_visibility": "observed", "frozen_policies": True,
        "new_training_updates": 0, "new_collection_steps": 0, "new_support_draws": 0,
        "evaluation": {"greedy_repetitions": 1, "epsilon_0_1_repetitions": 2,
            "rng": "SeedSequence([seed,map_seed,repetition,55219]); shared across policies",
            "greedy_argmax_ties": "lowest action label", "optimal_q_agreement_atol": 1e-6, "optimal_q_agreement_rtol": 0},
        "primary_comparison": "uniform_minus_collected", "primary_metric": "greedy efficient-success difference",
        "descriptive_reference_lines": {"success": .7, "efficient_success": .8, "planner_step_multiplier": 2, "not_new_gates": True},
        "budget": {"admission_seconds": max_seconds, "peak_process_rss_bytes": max_rss_bytes,
            "maximum_learner_episodes": 3 * len(seeds) * panel_count * maps_per_panel * 3,
            "maximum_reference_episodes": (2 * len(seeds) + 1) * panel_count * maps_per_panel,
            "preparation_and_aggregation_included": True},
        "runtime": runtime, "git": git_info(), "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
        "protocol_sha256": sha(protocol_file), "created_at": datetime.now(timezone.utc).isoformat(), "smoke": bool(smoke), "deviations": deviations}
    if protocol["git"]["dirty"] is not False or not protocol["git"]["revision"]:
        deviations.append({"field": "source_git", "actual": protocol["git"], "declared": "clean captured git revision"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    shutil.copy2(protocol_file, output / "protocol.md")
    for path in source_files:
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    packages = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    (output / "environment.txt").write_text("\n".join(packages) + "\n")
    command = ["python", "-m", "q6.panel_evaluation", "--output", str(output), "--protocol-file", str(protocol_file),
        "--seeds", ",".join(map(str, seeds)), "--panel-count", str(panel_count), "--maps-per-panel", str(maps_per_panel),
        "--panel-seed-start", str(panel_seed_start), "--panel-stride", str(panel_stride), "--max-seconds", str(max_seconds), "--max-rss-bytes", str(max_rss_bytes)]
    for name, directory in directories.items():
        command.extend(["--" + name.replace("_", "-") + "-dir", artifact_path(directory)])
    if smoke:
        command.append("--smoke")
    (output / "command.txt").write_text(shlex.join(command) + "\n# Reproduction needs a new output path.\n")
    write_json(output / "protocol.json", protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    deadline = started + max_seconds
    resource_checks = 0
    def enforce():
        nonlocal resource_checks
        resource_checks += 1
        guard(deadline, max_rss_bytes)
    models, model_records, archive_records, source_manifests = {}, [], {}, {}
    selection = {"status": "not_generated", "panels": [], "collision_skips": []}
    rows, references_raw, trajectories, progress = [], [], [], []
    status, stop_reason = "complete", None
    prep_seconds = reference_seconds = evaluation_seconds = 0.
    timings = {}
    oracle = VisibleOptimalQ(config, max_cached_maps=panel_count * maps_per_panel)
    with (output / "evaluations.csv").open("w", newline="") as ef, (output / "references.csv").open("w", newline="") as rf:
        ew, rw = csv.DictWriter(ef, EVAL_FIELDS), csv.DictWriter(rf, EVAL_FIELDS)
        ew.writeheader()
        rw.writeheader()
        try:
            phase = time.monotonic()
            try:
                enforce()
                excluded = {}
                for name, directory in directories.items():
                    metadata = json.loads((directory / "dataset_metadata.json").read_text())
                    manifest = source_manifests[name] = json.loads((directory / "manifest.json").read_text())
                    if manifest["status"] != "complete" or any(sha(directory / f) != manifest["files"][f] for f in ("dataset_metadata.json", "protocol.json")):
                        raise ConsistencyError("archive metadata does not match its complete manifest")
                    archive_records[name] = {"directory": artifact_path(directory), "files": {f: sha(directory / f) for f in ("dataset_metadata.json", "protocol.json", "manifest.json")}}
                    shutil.copy2(directory / "dataset_metadata.json", output / f"{name}_dataset_metadata.json")
                    excluded[f"previous_{name}_fresh_layout"] = {r["layout_hash"] for r in metadata["heldout"]}
                    if name == "coverage":
                        excluded["training_layout"] = {r["layout_hash"] for r in metadata["train"]}
                selection = select_panels(config, excluded, panel_count, maps_per_panel, panel_seed_start, panel_stride, enforce)
                protocol["panels"] = selection["panels"]
                write_json(output / "panels.json", selection)
                write_json(output / "protocol.json", protocol)
                for condition in CONDITIONS:
                    archive = "coverage" if condition == "exhaustive" else "equal_support"
                    for seed in seeds:
                        enforce()
                        source = directories[archive] / "models" / f"{condition}_seed{seed}_update30000.pt"
                        destination = output / "models" / f"{condition}_seed{seed}.pt"
                        if sha(source) != source_manifests[archive]["files"][f"models/{source.name}"]:
                            raise ConsistencyError("source checkpoint file differs from its archive manifest")
                        shutil.copy2(source, destination)
                        snapshot = torch.load(destination, map_location="cpu", weights_only=True)
                        if snapshot["optimizer_updates"] != 30000:
                            raise ConsistencyError("source checkpoint is not the declared final frozen model")
                        model = FrozenPolicy(snapshot)
                        models[(condition, seed)] = model
                        model_records.append({"condition": condition, "seed": seed, "archive": archive,
                            "source": artifact_path(source), "saved": destination.relative_to(output).as_posix(),
                            "source_sha256": sha(source), "saved_sha256": sha(destination),
                            "online_before": model.parameter_hash(), "target_before": module_hash(model.target),
                            "source_optimizer_updates": 30000, "panel_checks": [], "model_parameter_count": sum(p.numel() for p in model.online.parameters()),
                            "requires_grad": any(p.requires_grad for network in (model.online, model.target) for p in network.parameters())})
                write_json(output / "models.json", model_records)
            finally:
                prep_seconds += time.monotonic() - phase
            for panel in selection["panels"]:
                phase = time.monotonic()
                try:
                    for policy in ("random_actions", "shortest_path"):
                        for seed in (seeds if policy == "random_actions" else [0]):
                            group, replays = [], []
                            try:
                                for layout in panel["layouts"]:
                                    for rep in range(2 if policy == "random_actions" else 1):
                                        enforce()
                                        row, replay = rollout(None, config, oracle, condition="shared", policy=policy, seed=seed,
                                            checkpoint=30000, mode="reference", panel=panel["id"], map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                        row["reference_sample_id"] = f"{policy}:{panel['id']}:{seed}:{layout['map_seed']}:{rep}"
                                        group.append(row)
                                        if layout == panel["layouts"][0] and rep == 0 and seed == (seeds[0] if policy == "random_actions" else 0):
                                            replays.append(replay)
                            except BudgetReached:
                                for row in group:
                                    row["checkpoint_complete"] = 0
                                references_raw.extend(group)
                                rw.writerows(group)
                                raise
                            references_raw.extend(group)
                            rw.writerows(group)
                            trajectories.extend(replays)
                    rf.flush()
                finally:
                    reference_seconds += time.monotonic() - phase
                for condition in CONDITIONS:
                    for seed in seeds:
                        model = models[(condition, seed)]
                        record = next(r for r in model_records if r["condition"] == condition and r["seed"] == seed)
                        panel_check = {"panel": panel["id"], "online_before": model.parameter_hash(), "target_before": module_hash(model.target),
                            "saved_sha256_before": sha(output / record["saved"]), "source_sha256_before": sha(ROOT / record["source"])}
                        record["panel_checks"].append(panel_check)
                        panel_check["unchanged"] = (panel_check["online_before"] == record["online_before"] and panel_check["target_before"] == record["target_before"]
                            and panel_check["saved_sha256_before"] == panel_check["source_sha256_before"] == record["source_sha256"])
                        if not panel_check["unchanged"]:
                            raise ConsistencyError("frozen model changed before panel evaluation")
                        group, replays = [], []
                        phase = time.monotonic()
                        try:
                            for mode in ("greedy", "epsilon_0_1"):
                                for layout in panel["layouts"]:
                                    for rep in range(1 if mode == "greedy" else 2):
                                        enforce()
                                        row, replay = rollout(model, config, oracle, condition=condition, seed=seed, checkpoint=30000,
                                            mode=mode, panel=panel["id"], map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                        group.append(row)
                                        if layout == panel["layouts"][0] and rep == 0:
                                            replays.append(replay)
                        except BudgetReached:
                            for row in group:
                                row["checkpoint_complete"] = 0
                            rows.extend(group)
                            ew.writerows(group)
                            progress.append({"condition": condition, "seed": seed, "panel": panel["id"], "evaluation_complete": False})
                            raise
                        finally:
                            duration = time.monotonic() - phase
                            evaluation_seconds += duration
                            timings[f"{condition}:seed{seed}:{panel['id']}"] = duration
                            record = next(r for r in model_records if r["condition"] == condition and r["seed"] == seed)
                            check = {"panel": panel["id"], "online_hash": model.parameter_hash(), "target_hash": module_hash(model.target),
                                "saved_sha256": sha(output / record["saved"]), "source_sha256": sha(ROOT / record["source"])}
                            check["unchanged"] = (check["online_hash"] == record["online_before"] and check["target_hash"] == record["target_before"]
                                and check["saved_sha256"] == check["source_sha256"] == record["source_sha256"])
                            panel_check.update(check)
                            if not check["unchanged"]:
                                raise ConsistencyError("frozen model changed during panel evaluation")
                        rows.extend(group)
                        ew.writerows(group)
                        trajectories.extend(replays)
                        progress.append({"condition": condition, "seed": seed, "panel": panel["id"], "evaluation_complete": True})
                        ef.flush()
                        rate = np.mean([r["success"] for r in group if r["mode"] == "greedy"])
                        print(f"{panel['id']} {condition} seed={seed}: greedy_success={rate:.3f}", flush=True)
            enforce()
        except (BudgetReached, ConsistencyError) as exc:
            status = "inconsistent_not_evidence" if isinstance(exc, ConsistencyError) else "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
            stop_reason = str(exc)
    for record in model_records:
        model = models[(record["condition"], record["seed"])]
        record.update(online_after=model.parameter_hash(), target_after=module_hash(model.target),
            saved_sha256_after=sha(output / record["saved"]), source_sha256_after=sha(ROOT / record["source"]),
            requires_grad_after=any(p.requires_grad for network in (model.online, model.target) for p in network.parameters()))
        record["unchanged"] = (record["online_before"] == record["online_after"] and record["target_before"] == record["target_after"]
            and record["source_sha256"] == record["source_sha256_after"] == record["saved_sha256"] == record["saved_sha256_after"]
            and all(check["unchanged"] for check in record["panel_checks"]) and not record["requires_grad"] and not record["requires_grad_after"])
        if not record["unchanged"]:
            status, stop_reason = "inconsistent_not_evidence", "frozen model or copied checkpoint changed"
    cell_check = verify_expected_cells(rows, references_raw, selection["panels"], seeds)
    expected_models = len(CONDITIONS) * len(seeds)
    if status == "complete" and (not cell_check["complete"] or len(model_records) != expected_models or
            any(len(r["panel_checks"]) != panel_count for r in model_records)):
        status, stop_reason = "inconsistent_not_evidence", "expected frozen-model/panel/evaluation cells are incomplete or duplicated"
    aggregation_start = time.monotonic()
    seed_results, aggregate, references, paired = summarize_panels(rows, references_raw, selection["panels"], seeds)
    try:
        enforce()
    except BudgetReached as exc:
        status = "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
        stop_reason = str(exc)
    eligible = status == "complete" and not smoke and not deviations
    thresholds, robustness = descriptive_summaries(seed_results, aggregate, references, paired, selection["panels"], seeds, eligible)
    try:
        enforce()
    except BudgetReached as exc:
        status = "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
        stop_reason, eligible = str(exc), False
        thresholds, robustness = descriptive_summaries(seed_results, aggregate, references, paired, selection["panels"], seeds, False)
    robustness["eligible"] = eligible
    aggregation_seconds = time.monotonic() - aggregation_start
    provenance = {"archives": archive_records, "models": model_records, "all_loaded_models_unchanged": len(model_records) == expected_models and all(r["unchanged"] for r in model_records),
        "expected_cells": cell_check,
        "new_training_updates": 0, "new_collection_steps": 0, "new_support_draws": 0}
    artifacts = {name: artifact_path(output / path) for name, path in {"protocol": "protocol.json", "panels": "panels.json",
        "evaluations": "evaluations.csv", "references": "references.csv", "models": "models.json", "provenance": "provenance.json",
        "paired_differences": "paired_differences.json", "paired_layouts": "paired_layouts.csv", "trajectories": "trajectories.json", "manifest": "manifest.json"}.items()}
    artifacts.update(directory=artifact_path(output), protocol_document=artifact_path(protocol_file), report="docs/experiments/panel_evaluation_results_v1.md")
    result = {"schema_version": 1, "protocol": protocol, "run": {"id": output.name, "status": status, "stop_reason": stop_reason,
        "interpretation": "smoke_or_deviation_descriptive_only" if smoke or deviations else "frozen_panel_robustness_descriptive_only" if status == "complete" else "incomplete_not_evidence",
        "new_training_updates": 0, "new_collection_steps": 0, "new_support_draws": 0, "frozen_models": len(models),
        "model_parameter_count": 20420, "learner_episodes": len(rows), "completed_learner_episodes": sum(r["checkpoint_complete"] for r in rows),
        "reference_episodes": len(references_raw), "unique_reference_episodes": len({r["reference_sample_id"] for r in references_raw}),
        "panels": len(selection["panels"]), "unique_layouts": selection.get("unique_selected_layouts", 0),
        "wall_seconds": time.monotonic() - started, "preparation_wall_seconds": prep_seconds,
        "evaluation_wall_seconds": evaluation_seconds, "reference_wall_seconds": reference_seconds, "aggregation_wall_seconds": aggregation_seconds,
        "model_panel_wall_seconds": timings, "peak_rss_bytes": peak_rss_bytes(), "resource_checks": resource_checks,
        "resource_limits": {"peak_process_rss_bytes": max_rss_bytes, "seconds": max_seconds, "torch_threads": 1,
            "check_cadence": "each selected layout, model loading, evaluation episode and phase boundaries; sampled stop guard"},
        "progress": progress, "all_models_unchanged": provenance["all_loaded_models_unchanged"],
        "limitations": ["These panels vary task selection for the same fixed policies, not training or support draws.",
            "Panel ranges and signs are descriptive; no confidence, significance or equivalence claim is made.",
            "Historical reference-line counts do not create new competence gates or alter preceding results.",
            "Episode length must be read with success because failures consume the full horizon."]},
        "panels": selection["panels"], "aggregate": aggregate, "seed_results": seed_results, "references": references,
        "paired_differences": paired, "robustness": robustness, "descriptive_thresholds": thresholds,
        "provenance": provenance, "trajectories": trajectories, "artifacts": artifacts}
    write_json(output / "protocol.json", protocol)
    write_json(output / "panels.json", selection)
    write_json(output / "models.json", model_records)
    write_json(output / "provenance.json", provenance)
    write_json(output / "paired_differences.json", paired)
    with (output / "paired_layouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(paired["per_layout"])
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
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--panel-count", type=int, default=8)
    parser.add_argument("--maps-per-panel", type=int, default=64)
    parser.add_argument("--panel-seed-start", type=int, default=970000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    for name in ("supervised", "fixed-targets", "coverage", "equal-support"):
        parser.add_argument("--" + name + "-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.panel_count, args.maps_per_panel, args.panel_seed_start = "0", 2, 2, 1010000
    archives = {name: getattr(args, name + "_dir") for name in ("supervised", "fixed_targets", "coverage", "equal_support") if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, seeds=[int(s) for s in args.seeds.split(",")], panel_count=args.panel_count,
        maps_per_panel=args.maps_per_panel, panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
