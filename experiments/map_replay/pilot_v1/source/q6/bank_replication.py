"""Three independently collected equal-size bank pairs with unchanged DDQN fits.

Freeze every bank pair before optimization. Evaluate only final models, using
shared new panels and references; bank-level differences are descriptive.
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
from .competence import BudgetReached, EVAL_FIELDS, ROOT, artifact_path, git_info
from .coverage import SupportSampler, collect_support
from .equal_support import support_metrics
from .fixed_targets import ConsistencyError, MemoryReached, array_metadata, build_transitions, fixed_update, guard, module_hash, peak_rss_bytes, sha, validate_transitions
from .learning import DQN
from .optimal import VisibleOptimalQ
from .panel_evaluation import FrozenPolicy, PAIR_FIELDS, select_panels, summarize_panels, verify_expected_cells, descriptive_summaries
from .supervised import CHECKPOINTS, LOSS_FIELDS, enumerate_panel, layout_key, rollout, save_manifest
from .world import CollectionWorld, WorldConfig

CONDITIONS = ("collected_unique", "uniform_subset")
COMPARISONS = ({"id": "uniform_minus_collected", "left": "collected_unique", "right": "uniform_subset"},)
METRICS = ("success_rate", "efficient_success_rate", "mean_steps", "noop_rate")


def choose_uniform_support(state_count, size, bank_id):
    if not 64 <= size <= state_count:
        raise ConsistencyError("configuration failure: collected support must fit the 64-state batch")
    rng = np.random.default_rng(np.random.SeedSequence([88301, int(bank_id)]))
    return np.sort(rng.choice(state_count, size=size, replace=False)).astype(np.int32)


def prepare_banks(config, data, transitions, output, bank_ids, episodes_per_map, enforce, deadline):
    banks, supports = [], {}
    for bank_id in bank_ids:
        directory = output / "banks" / f"bank{bank_id}"
        coverage, arrays = collect_support(config, data, transitions, directory,
            episodes_per_map=episodes_per_map, deadline=deadline, enforce=enforce, rng_suffix=(bank_id,))
        bank = {"bank_id": bank_id, "status": coverage["status"], "collection": coverage,
                "artifacts": {"collection": artifact_path(directory / "collection.npz"),
                    "steps": artifact_path(directory / "collection_steps.csv"), "episodes": artifact_path(directory / "collection_episodes.csv")}}
        banks.append(bank)
        if coverage["status"] != "complete":
            return banks, supports, coverage["status"], coverage["stop_reason"]
        collected = arrays["support_indices"]
        try:
            uniform = choose_uniform_support(len(data["observations"]), len(collected), bank_id)
        except ConsistencyError as exc:
            bank["status"] = "inconsistent_not_evidence"
            return banks, supports, "inconsistent_not_gate_evidence", str(exc)
        pair = {"collected_unique": collected, "uniform_subset": uniform}
        supports[bank_id] = pair
        overlap = int(len(np.intersect1d(collected, uniform)))
        bank.update(support_size=len(collected), arrays=array_metadata(pair),
            uniform_rng=f"SeedSequence([88301,{bank_id}])",
            intersection={"states": overlap, "union_states": 2 * len(collected) - overlap,
                "fraction_of_each": overlap / len(collected), "jaccard": overlap / (2 * len(collected) - overlap)},
            coverage={"per_condition": [{"condition": condition, **support_metrics(data, transitions, support)} for condition, support in pair.items()]})
        write_json(directory / "bank.json", bank)
        for support in pair.values():
            support.flags.writeable = False
        np.savez_compressed(output / "supports.npz", **{f"{condition}_bank{bank}": support for bank, pair in supports.items() for condition, support in pair.items()})
    np.savez_compressed(output / "supports.npz", **{f"{condition}_bank{bank}": support for bank, pair in supports.items() for condition, support in pair.items()})
    return banks, supports, "complete", None


def aggregate_banks(rows, refs, panels, seeds, bank_ids, *, conditions=CONDITIONS, comparisons=COMPARISONS):
    seed_results, aggregate, references = [], [], []
    paired = {"comparisons": list(comparisons), "scope": "greedy uniform minus collected within bank, seed and layout" if comparisons == COMPARISONS else "greedy right minus left within bank, seed and layout",
        "per_seed": [], "aggregate": [], "per_layout": []}
    for bank_id in bank_ids:
        subset = [r for r in rows if r["bank_id"] == bank_id]
        local_seed, local_aggregate, references, local_pairs = summarize_panels(subset, refs, panels, seeds,
            conditions=conditions, comparisons=comparisons)
        seed_results.extend({"bank_id": bank_id, **r} for r in local_seed)
        aggregate.extend({"bank_id": bank_id, **r} for r in local_aggregate)
        for key in ("per_seed", "aggregate", "per_layout"):
            paired[key].extend({"bank_id": bank_id, **r} for r in local_pairs[key])
    pooled = {"classification": "equal-bank arithmetic means; three banks are the replication units",
        "aggregate": [], "paired": []}
    for panel in [r["id"] for r in panels] + ["all"]:
        for mode in ("greedy", "epsilon_0_1"):
            for condition in conditions:
                group = [r for r in aggregate if r["panel"] == panel and r["mode"] == mode and r["condition"] == condition]
                if group:
                    noop_steps, total_steps = sum(r["noop_steps"] for r in group), sum(r["evaluation_steps"] for r in group)
                    successful_steps = sum(r["successful_episode_steps"] for r in group)
                    successful_episodes = sum(r["successful_episodes"] for r in group)
                    pooled["aggregate"].append({"condition": condition, "panel": panel, "mode": mode, "banks": len(group),
                        "seeds_per_bank": len(seeds), "episodes": sum(r["episodes"] for r in group),
                        **{k: float(np.mean([r[k] for r in group])) if all(r[k] is not None for r in group) else None for k in METRICS},
                        "mean_bank_noop_rate": float(np.mean([r["noop_rate"] for r in group])),
                        "noop_steps": noop_steps, "evaluation_steps": total_steps, "noop_rate": noop_steps / total_steps,
                        "noop_rate_definition": "pooled blocked steps divided by pooled episode steps; mean_bank_noop_rate is separate",
                        "successful_episodes": successful_episodes, "successful_episode_steps": successful_steps,
                        "successful_mean_steps": successful_steps / successful_episodes if successful_episodes else None,
                        "efficient_success_episodes": sum(r["efficient_success_episodes"] for r in group)
                            if all(r["efficient_success_episodes"] is not None for r in group) else None})
        group = [r for r in paired["aggregate"] if r["panel"] == panel]
        if group:
            pooled["paired"].append({"panel": panel, "mode": "greedy", "banks": len(group),
                **{"mean_bank_" + k + "_delta": float(np.mean([r["mean_seed_" + k + "_delta"] for r in group]))
                    if all(r["mean_seed_" + k + "_delta"] is not None for r in group) else None for k in METRICS},
                "mean_bank_mean_seed_noop_rate_delta": float(np.mean([r["mean_seed_noop_rate_delta"] for r in group])),
                "mean_bank_noop_rate_delta": float(np.mean([r["pooled_noop_rate_delta"] for r in group])),
                "noop_delta_definition": "mean_bank_noop_rate_delta averages within-bank pooled blocked-step ratios; mean_bank_mean_seed_noop_rate_delta separately averages learner ratios"})
    return seed_results, aggregate, references, paired, pooled


def run_study(output, protocol_file, *, bank_ids=(1, 2, 3), seeds=(0, 1, 2), updates=30000,
              episodes_per_map=16, panel_count=8, maps_per_panel=64, panel_seed_start=1020000,
              panel_stride=1000, train_seed_start=300000, train_maps=256, max_seconds=1200,
              max_rss_bytes=4 * 1024**3, dashboard=None, archives=None, smoke=False, checkpoints=None):
    started = time.monotonic()
    output, protocol_file = Path(output), Path(protocol_file)
    bank_ids, seeds = list(bank_ids), list(seeds)
    if not protocol_file.is_file() or not bank_ids or not seeds or len(set(bank_ids)) != len(bank_ids) or len(set(seeds)) != len(seeds):
        raise ValueError("existing protocol and distinct bank/learner identifiers are required")
    if min(bank_ids) < 1 or min(seeds) < 0 or min(updates, episodes_per_map, panel_count, maps_per_panel, train_maps, max_seconds, max_rss_bytes, panel_stride) <= 0:
        raise ValueError("positive budgets/bank IDs and nonnegative learner seeds are required")
    schedule = sorted(set(checkpoints if checkpoints is not None else [c for c in CHECKPOINTS if c <= updates] + [updates]))
    if not schedule or schedule[0] != 0 or schedule[-1] != updates:
        raise ValueError("snapshot schedule must span zero through the update budget")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    directories = {name: ROOT / f"experiments/{name}/pilot_v1" for name in ("supervised", "fixed_targets", "coverage", "equal_support", "panel_evaluation")}
    if archives:
        directories.update({name: Path(path) for name, path in archives.items()})
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "torch_threads": 1}
    actual = dict(bank_ids=bank_ids, seeds=seeds, updates=updates, episodes_per_map=episodes_per_map,
        panel_count=panel_count, maps_per_panel=maps_per_panel, panel_seed_start=panel_seed_start, panel_stride=panel_stride,
        train_seed_start=train_seed_start, train_maps=train_maps, max_seconds=max_seconds, max_rss_bytes=max_rss_bytes, checkpoints=schedule)
    declared = dict(bank_ids=[1, 2, 3], seeds=[0, 1, 2], updates=30000, episodes_per_map=16, panel_count=8, maps_per_panel=64,
        panel_seed_start=1020000, panel_stride=1000, train_seed_start=300000, train_maps=256, max_seconds=1200,
        max_rss_bytes=4 * 1024**3, checkpoints=list(CHECKPOINTS))
    deviations = [{"field": k, "actual": v, "declared": declared[k]} for k, v in actual.items() if v != declared[k]]
    if not runtime["python"].startswith("3.12.") or runtime["torch"].split("+")[0] != "2.8.0" or runtime["numpy"] != "2.0.2":
        deviations.append({"field": "runtime", "actual": runtime, "declared": "Python3.12/Torch2.8.0/NumPy2.0.2"})
    for name, directory in directories.items():
        expected = ROOT / f"experiments/{name}/pilot_v1"
        if directory.resolve() != expected.resolve():
            deviations.append({"field": "archive_" + name, "actual": artifact_path(directory), "declared": artifact_path(expected)})
    config = WorldConfig()
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    protocol = {"id": "bank-replication-v1", "question": "Does the uniform-bank efficiency difference recur across independently collected bank pairs?",
        "conditions": [{"id": "collected_unique", "label": "Collected unique"}, {"id": "uniform_subset", "label": "Uniform subset"}],
        "bank_ids": bank_ids, "seeds": seeds, "checkpoints": schedule, "evaluation_checkpoints": [updates], "panels": [],
        "world": asdict(config), "rule_visibility": "observed", "runtime": runtime, "git": git_info(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}, "protocol_sha256": sha(protocol_file),
        "created_at": datetime.now(timezone.utc).isoformat(), "smoke": bool(smoke), "deviations": deviations,
        "collection": {"episodes_per_map": episodes_per_map, "rng": "SeedSequence([map_seed,repetition,77301,bank_id])",
            "policy": "uniform random actions", "all_pairs_frozen_before_optimization": True},
        "support_selection": {"uniform_rng": "SeedSequence([88301,bank_id])", "size": "exact realized unique collected support per bank",
            "draw": "one choice(fullN,size=K,replace=False), sorted int32", "shared_across_learner_seeds": True},
        "optimizer": {"implementation": "unchanged fixed_targets.fixed_update(...,'double_dqn')", "learning_rate": .001,
            "loss": "all-four-action SmoothL1 mean beta1", "gradient_norm_cap": 5., "gamma": .97, "target_tau": .01,
            "batch_size": 64, "sampling_rng": "SeedSequence([seed,66301])", "paired_local_counts_within_bank": True},
        "evaluation": {"final_only": True, "after_all_fits": True, "greedy_repetitions": 1, "epsilon_0_1_repetitions": 2,
            "rng": "SeedSequence([seed,map_seed,repetition,55219])", "greedy_ties": "lowest label", "optimal_q_atol": 1e-6, "optimal_q_rtol": 0},
        "panel_selection": {"count": panel_count, "maps_per_panel": maps_per_panel, "start": panel_seed_start, "stride": panel_stride},
        "budget": {"maximum_updates": len(bank_ids) * len(CONDITIONS) * len(seeds) * updates, "updates_per_fit": updates,
            "maximum_collection_episodes": len(bank_ids) * train_maps * episodes_per_map,
            "maximum_collection_steps": len(bank_ids) * train_maps * episodes_per_map * config.horizon,
            "admission_seconds": max_seconds, "peak_process_rss_bytes": max_rss_bytes, "all_phases_included": True},
        "primary_comparison": "uniform_minus_collected", "primary_metric": "greedy efficient success", "new_competence_gates": False}
    if protocol["git"]["dirty"] is not False or not protocol["git"]["revision"]:
        deviations.append({"field": "source_git", "actual": protocol["git"], "declared": "clean captured revision"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    shutil.copy2(protocol_file, output / "protocol.md")
    for path in source_files:
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (output / "environment.txt").write_text("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")})) + "\n")
    command = ["python", "-m", "q6.bank_replication", "--output", str(output), "--protocol-file", str(protocol_file),
        "--bank-ids", ",".join(map(str, bank_ids)), "--seeds", ",".join(map(str, seeds)), "--updates", str(updates),
        "--episodes-per-map", str(episodes_per_map), "--panel-count", str(panel_count), "--maps-per-panel", str(maps_per_panel),
        "--panel-seed-start", str(panel_seed_start), "--panel-stride", str(panel_stride), "--train-seed-start", str(train_seed_start),
        "--train-maps", str(train_maps), "--max-seconds", str(max_seconds), "--max-rss-bytes", str(max_rss_bytes),
        "--checkpoints", ",".join(map(str, schedule))]
    for name, directory in directories.items():
        command.extend(["--" + name.replace("_", "-") + "-dir", artifact_path(directory)])
    if smoke:
        command.append("--smoke")
    (output / "command.txt").write_text(shlex.join(command) + "\n# Reproduction needs a new output path.\n")
    write_json(output / "protocol.json", protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    deadline, checks = started + max_seconds, 0
    def enforce():
        nonlocal checks
        checks += 1
        guard(deadline, max_rss_bytes)
    status, reason = "complete", None
    data, transitions, metadata, archives_meta = {}, {}, {}, {}
    banks, supports, models, sampling_objects, samples, model_records = [], {}, {}, {}, {}, []
    rows, refs, trajectories, losses, progress = [], [], [], [], []
    selection = {"status": "not_generated", "panels": []}
    timings = {"preparation": 0., "collection_support": 0., "training": 0., "references": 0., "evaluation": 0.}
    fit_timings, model_panel_timings = {}, {}
    total_updates, current_window, active = 0, [], None
    prior_initial, snapshot_records, support_intersections, support_integrity = {}, [], [], []
    with (output / "losses.csv").open("w", newline="") as lf, (output / "evaluations.csv").open("w", newline="") as ef, (output / "references.csv").open("w", newline="") as rf:
        lw = csv.DictWriter(lf, ["bank_id", "condition", *LOSS_FIELDS])
        ew, rw = csv.DictWriter(ef, ["bank_id", *EVAL_FIELDS]), csv.DictWriter(rf, ["bank_id", *EVAL_FIELDS])
        for writer in (lw, ew, rw):
            writer.writeheader()
        try:
            phase = time.monotonic()
            try:
                enforce()
                excluded = {}
                for name, directory in directories.items():
                    filename = "panels.json" if name == "panel_evaluation" else "dataset_metadata.json"
                    saved = json.loads((directory / filename).read_text())
                    manifest = json.loads((directory / "manifest.json").read_text())
                    if manifest["status"] != "complete" or manifest["files"][filename] != sha(directory / filename):
                        raise ConsistencyError("input archive metadata differs from complete manifest")
                    archives_meta[name] = {"directory": artifact_path(directory), "files": {filename: sha(directory / filename), "manifest.json": sha(directory / "manifest.json")}}
                    shutil.copy2(directory / filename, output / f"{name}_{filename}")
                    layouts = [r for panel in saved["panels"] for r in panel["layouts"]] if name == "panel_evaluation" else saved["heldout"]
                    excluded[f"previous_{name}_fresh_layout"] = {r["layout_hash"] for r in layouts}
                    if name == "coverage":
                        archived_metadata = saved
                        excluded["training_layout"] = {r["layout_hash"] for r in saved["train"]}
                        for filename in ("dataset.npz", "transitions.npz"):
                            if manifest["files"][filename] != sha(directory / filename):
                                raise ConsistencyError("training bank input differs from archive manifest")
                            archives_meta[name]["files"][filename] = sha(directory / filename)
                        for seed in seeds:
                            filename = f"models/collected_unique_seed{seed}_update0.pt"
                            if manifest["files"][filename] != sha(directory / filename):
                                raise ConsistencyError("prior initialization differs from archived manifest")
                            initial = torch.load(directory / filename, map_location="cpu", weights_only=True)
                            prior_initial[str(seed)] = {"online_hash": initial["parameter_hash"], "target_hash": initial["target_parameter_hash"],
                                "source": artifact_path(directory / filename), "sha256": sha(directory / filename)}
                if smoke:
                    env = CollectionWorld(config)
                    layouts = []
                    for seed in range(train_seed_start, train_seed_start + train_maps):
                        env.reset(seed=seed)
                        layouts.append({"map_seed": seed, "layout_hash": layout_key(env), "original_start": list(env.position)})
                    data = enumerate_panel(config, layouts, VisibleOptimalQ(config), deadline)
                    transitions = build_transitions(config, data, deadline)
                else:
                    with np.load(directories["coverage"] / "dataset.npz") as arrays:
                        data = {key.removeprefix("train_"): arrays[key] for key in arrays.files if key.startswith("train_")}
                    with np.load(directories["coverage"] / "transitions.npz") as arrays:
                        transitions = {key: arrays[key] for key in arrays.files}
                    layouts = archived_metadata["train"]
                    if [r["map_seed"] for r in layouts] != list(range(train_seed_start, train_seed_start + train_maps)):
                        raise ConsistencyError("configured training layouts differ from the archived bank")
                excluded["training_layout"].update(r["layout_hash"] for r in layouts)
                metadata = {"status": "complete", "train": layouts, "arrays": array_metadata({"train_" + k: v for k, v in data.items()}),
                    "transition_arrays": array_metadata(transitions), "training_arrays_identical": False, "transition_arrays_identical": False}
                metadata["training_arrays_identical"] = all(v == archived_metadata["arrays"][k] for k, v in metadata["arrays"].items())
                metadata["transition_arrays_identical"] = metadata["transition_arrays"] == archived_metadata["transition_arrays"]
                if not smoke and not (metadata["training_arrays_identical"] and metadata["transition_arrays_identical"]):
                    raise ConsistencyError("archived training arrays/transitions changed")
                metadata["transition_validation"] = validate_transitions(config, data, transitions, deadline)
                np.savez_compressed(output / "dataset.npz", **{"train_" + k: v for k, v in data.items()})
                np.savez_compressed(output / "transitions.npz", **transitions)
                write_json(output / "dataset_metadata.json", metadata)
                selection = select_panels(config, excluded, panel_count, maps_per_panel, panel_seed_start, panel_stride, enforce)
                protocol["panels"] = selection["panels"]
                protocol["dataset"] = {"train_states": len(data["observations"]), "train_map_seeds": [r["map_seed"] for r in layouts], "fresh_state_enumeration": False}
                write_json(output / "panels.json", selection)
                write_json(output / "protocol.json", protocol)
            finally:
                timings["preparation"] += time.monotonic() - phase
            phase = time.monotonic()
            try:
                banks, supports, bank_status, bank_reason = prepare_banks(config, data, transitions, output, bank_ids, episodes_per_map, enforce, deadline)
                write_json(output / "banks.json", banks)
                for index, left_bank in enumerate(bank_ids):
                    for right_bank in bank_ids[index + 1:]:
                        if left_bank not in supports or right_bank not in supports:
                            continue
                        for left_condition in CONDITIONS:
                            for right_condition in CONDITIONS:
                                left, right = supports[left_bank][left_condition], supports[right_bank][right_condition]
                                overlap = int(len(np.intersect1d(left, right)))
                                union = len(left) + len(right) - overlap
                                support_intersections.append({"left_bank": left_bank, "right_bank": right_bank,
                                    "left_condition": left_condition, "right_condition": right_condition, "states": overlap,
                                    "union_states": union, "jaccard": overlap / union,
                                    "fraction_of_left": overlap / len(left), "fraction_of_right": overlap / len(right)})
                write_json(output / "support_intersections.json", support_intersections)
                if bank_status != "complete":
                    raise ConsistencyError(bank_reason) if bank_status == "inconsistent_not_gate_evidence" else MemoryReached(bank_reason) if bank_status == "incomplete_memory_cap" else BudgetReached(bank_reason)
            finally:
                timings["collection_support"] += time.monotonic() - phase
            train_obs = torch.from_numpy(data["observations"])
            unused_exact = torch.from_numpy(data["targets"].astype(np.float32))
            transition_tensors = {k: torch.from_numpy(v.astype(np.float32) if k == "rewards" else v) for k, v in transitions.items()}
            for bank_id in bank_ids:
                for condition in CONDITIONS:
                    for seed in seeds:
                        enforce()
                        active = (bank_id, condition, seed)
                        agent = DQN(92, seed=seed)
                        initial_hash, initial_target = agent.parameter_hash(), module_hash(agent.target)
                        if initial_hash != prior_initial[str(seed)]["online_hash"] or initial_target != prior_initial[str(seed)]["target_hash"]:
                            raise ConsistencyError("new fit initialization differs from prior learner")
                        sampler = sampling_objects[active] = SupportSampler(supports[bank_id][condition], len(train_obs), seed)
                        current_window = []
                        phase = time.monotonic()
                        try:
                            for checkpoint in schedule:
                                while sampler.updates < checkpoint:
                                    enforce()
                                    loss = fixed_update(agent, train_obs, unused_exact, transition_tensors, sampler.next_batch(), "double_dqn")
                                    total_updates += 1
                                    if not np.isfinite(loss):
                                        raise ConsistencyError("nonfinite optimizer loss")
                                    current_window.append(loss)
                                    if sampler.updates % 100 == 0 or sampler.updates == checkpoint:
                                        row = {"bank_id": bank_id, "condition": condition, "seed": seed, "checkpoint": sampler.updates,
                                            "updates_in_window": len(current_window), "mean_loss": float(np.mean(current_window)), "last_loss": loss,
                                            "training_examples": sampler.updates * 64}
                                        losses.append(row)
                                        lw.writerow(row)
                                        current_window = []
                                if not all(torch.isfinite(p).all().item() for network in (agent.online, agent.target) for p in network.parameters()):
                                    raise ConsistencyError("nonfinite model weights")
                                path = output / "models" / f"bank{bank_id}_{condition}_seed{seed}_update{checkpoint}.pt"
                                torch.save({"online": agent.online.state_dict(), "target": agent.target.state_dict(), "observation_size": 92,
                                    "parameter_hash": agent.parameter_hash(), "target_parameter_hash": module_hash(agent.target),
                                    "optimizer_updates": checkpoint, "purpose": "inference_only_not_resumable"}, path)
                                snapshot_records.append({"bank_id": bank_id, "condition": condition, "seed": seed, "checkpoint": checkpoint,
                                    "saved": path.relative_to(output).as_posix(), "sha256": sha(path), "online_hash": agent.parameter_hash(), "target_hash": module_hash(agent.target)})
                                progress.append({"bank_id": bank_id, "condition": condition, "seed": seed, "checkpoint": checkpoint, "snapshot_saved": True})
                                print(f"bank{bank_id} {condition} seed={seed}: saved update {checkpoint}", flush=True)
                            model_records.append({"bank_id": bank_id, "condition": condition, "seed": seed,
                                "initial_online_hash": initial_hash, "initial_target_hash": initial_target, "saved": path.relative_to(output).as_posix(), "file_before": sha(path),
                                "online_before": agent.parameter_hash(), "target_before": module_hash(agent.target), "panel_checks": []})
                            progress.append({"bank_id": bank_id, "condition": condition, "seed": seed, "training_complete": True, "updates": sampler.updates})
                            print(f"bank{bank_id} {condition} seed={seed}: fit complete ({sampler.updates} updates)", flush=True)
                        finally:
                            duration = time.monotonic() - phase
                            timings["training"] += duration
                            fit_timings[f"bank{bank_id}:{condition}:seed{seed}"] = duration
                            lf.flush()
                        del agent
            # All 18 fits finish before any learned policy evaluation.
            for record in model_records:
                enforce()
                snapshot = torch.load(output / record["saved"], map_location="cpu", weights_only=True)
                models[(record["bank_id"], record["condition"], record["seed"])] = FrozenPolicy(snapshot)
            oracle = VisibleOptimalQ(config, max_cached_maps=panel_count * maps_per_panel)
            for panel in selection["panels"]:
                phase = time.monotonic()
                try:
                    for policy in ("random_actions", "shortest_path"):
                        for seed in (seeds if policy == "random_actions" else [0]):
                            for layout in panel["layouts"]:
                                for rep in range(2 if policy == "random_actions" else 1):
                                    enforce()
                                    row, replay = rollout(None, config, oracle, condition="shared", policy=policy, seed=seed, checkpoint=updates,
                                        mode="reference", panel=panel["id"], map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                    row["bank_id"] = replay["bank_id"] = "shared"
                                    row["reference_sample_id"] = f"{policy}:{panel['id']}:{seed}:{layout['map_seed']}:{rep}"
                                    refs.append(row)
                                    rw.writerow(row)
                                    if layout == panel["layouts"][0] and rep == 0 and seed == (seeds[0] if policy == "random_actions" else 0):
                                        trajectories.append(replay)
                    rf.flush()
                finally:
                    timings["references"] += time.monotonic() - phase
                for record in model_records:
                    bank_id, condition, seed = record["bank_id"], record["condition"], record["seed"]
                    model = models[(bank_id, condition, seed)]
                    check = {"panel": panel["id"], "online_before": model.parameter_hash(), "target_before": module_hash(model.target),
                        "file_before": sha(output / record["saved"])}
                    record["panel_checks"].append(check)
                    group, replays = [], []
                    phase = time.monotonic()
                    try:
                        for mode in ("greedy", "epsilon_0_1"):
                            for layout in panel["layouts"]:
                                for rep in range(1 if mode == "greedy" else 2):
                                    enforce()
                                    row, replay = rollout(model, config, oracle, condition=condition, seed=seed, checkpoint=updates,
                                        mode=mode, panel=panel["id"], map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                    row["bank_id"] = replay["bank_id"] = bank_id
                                    group.append(row)
                                    if layout == panel["layouts"][0] and rep == 0:
                                        replays.append(replay)
                    except BudgetReached:
                        for row in group:
                            row["checkpoint_complete"] = 0
                        rows.extend(group)
                        ew.writerows(group)
                        raise
                    finally:
                        duration = time.monotonic() - phase
                        timings["evaluation"] += duration
                        model_panel_timings[f"bank{bank_id}:{condition}:seed{seed}:{panel['id']}"] = duration
                        check.update(online_after=model.parameter_hash(), target_after=module_hash(model.target), file_after=sha(output / record["saved"]))
                        check["unchanged"] = check["online_before"] == check["online_after"] == record["online_before"] and check["target_before"] == check["target_after"] == record["target_before"] and check["file_before"] == check["file_after"] == record["file_before"]
                        if not check["unchanged"]:
                            raise ConsistencyError("frozen final model changed during evaluation")
                    rows.extend(group)
                    ew.writerows(group)
                    trajectories.extend(replays)
                    ef.flush()
        except (BudgetReached, ConsistencyError) as exc:
            status = "inconsistent_not_evidence" if isinstance(exc, ConsistencyError) else "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
            reason = str(exc)
            if current_window and active:
                bank_id, condition, seed = active
                row = {"bank_id": bank_id, "condition": condition, "seed": seed, "checkpoint": sampler.updates,
                    "updates_in_window": len(current_window), "mean_loss": float(np.mean(current_window)), "last_loss": current_window[-1], "training_examples": sampler.updates * 64}
                losses.append(row)
                lw.writerow(row)
    for (bank_id, condition, seed), sampler in sampling_objects.items():
        visited = np.zeros(len(data["observations"]), bool)
        visited[supports[bank_id][condition]] = True
        live = ~transitions["ends"]
        outside = live & ~visited[np.maximum(transitions["successor_indices"], 0)]
        query_count = int(np.dot(sampler.counts.astype(np.uint64), live.sum(1).astype(np.uint64)))
        outside_count = int(np.dot(sampler.counts.astype(np.uint64), outside.sum(1).astype(np.uint64)))
        samples.setdefault(str(bank_id), {}).setdefault(condition, {})[str(seed)] = {"updates": sampler.updates,
            "examples_seen": int(sampler.counts.sum()), "unique_states_sampled": int(np.count_nonzero(sampler.counts)),
            "support_states": len(sampler.support), "local_batch_index_sha256": sampler.local.digest.hexdigest(),
            "global_batch_index_sha256": sampler.digest.hexdigest(), "rng_seed_tuple": [seed, 66301],
            "outside_support_direct_samples": int(sampler.counts[~visited].sum()),
            "successor_queries": {"nonterminal_queries": query_count, "outside_support_queries": outside_count,
                "outside_support_fraction": outside_count / query_count if query_count else None}}
    np.savez_compressed(output / "sample_counts.npz", **{f"bank{b}_{c}_seed{s}": v.counts for (b, c, s), v in sampling_objects.items()})
    np.savez_compressed(output / "local_sample_counts.npz", **{f"bank{b}_{c}_seed{s}": v.local.counts for (b, c, s), v in sampling_objects.items()})
    consistency = []
    for bank_id in bank_ids:
        for seed in seeds:
            a, b = (sampling_objects.get((bank_id, condition, seed)) for condition in CONDITIONS)
            records = [r for r in model_records if r["bank_id"] == bank_id and r["seed"] == seed]
            item = {"bank_id": bank_id, "seed": seed, "complete": bool(a and b and a.updates == b.updates == updates),
                "local_digest_identical": bool(a and b and a.local.digest.hexdigest() == b.local.digest.hexdigest()),
                "local_counts_identical": bool(a and b and np.array_equal(a.local.counts, b.local.counts)),
                "initial_online_identical": len(records) == 2 and len({r["initial_online_hash"] for r in records}) == 1}
            consistency.append(item)
    initialization_consistency = []
    for seed in seeds:
        selected = [r for r in model_records if r["seed"] == seed]
        initialization_consistency.append({"seed": seed, "fits": len(selected),
            "online_matches_prior": bool(selected) and all(r["initial_online_hash"] == prior_initial[str(seed)]["online_hash"] for r in selected),
            "target_matches_prior": bool(selected) and all(r["initial_target_hash"] == prior_initial[str(seed)]["target_hash"] for r in selected)})
    for bank in banks:
        if bank["bank_id"] not in supports:
            continue
        after = array_metadata(supports[bank["bank_id"]])
        support_integrity.append({"bank_id": bank["bank_id"], "before": bank["arrays"], "after": after,
            "unchanged": after == bank["arrays"], "read_only": all(not a.flags.writeable for a in supports[bank["bank_id"]].values())})
    for record in model_records:
        model = models.get((record["bank_id"], record["condition"], record["seed"]))
        record.update(file_after=sha(output / record["saved"]), online_after=model.parameter_hash() if model else None,
            target_after=module_hash(model.target) if model else None)
        record["unchanged_during_evaluation"] = record["online_before"] == record["online_after"] and record["target_before"] == record["target_after"] and record["file_before"] == record["file_after"] and all(c.get("unchanged", False) for c in record["panel_checks"])
    cell_checks = [{"bank_id": bank_id, **verify_expected_cells([r for r in rows if r["bank_id"] == bank_id], refs,
        selection["panels"], seeds, conditions=CONDITIONS)} for bank_id in bank_ids]
    expected_snapshots = {f"models/bank{bank}_{condition}_seed{seed}_update{checkpoint}.pt"
        for bank in bank_ids for condition in CONDITIONS for seed in seeds for checkpoint in schedule}
    actual_snapshots = {r["saved"] for r in snapshot_records}
    snapshot_integrity = {"expected": len(expected_snapshots), "actual": len(snapshot_records),
        "complete": actual_snapshots == expected_snapshots and len(snapshot_records) == len(expected_snapshots),
        "unchanged": all(sha(output / r["saved"]) == r["sha256"] for r in snapshot_records)}
    complete = (snapshot_integrity["complete"] and snapshot_integrity["unchanged"] and
        all(r["unchanged"] and r["read_only"] for r in support_integrity) and
        all(r["fits"] == len(bank_ids) * 2 and r["online_matches_prior"] and r["target_matches_prior"] for r in initialization_consistency) and len(model_records) == len(bank_ids) * len(CONDITIONS) * len(seeds) and total_updates == protocol["budget"]["maximum_updates"]
        and all(all(r[k] for k in ("complete", "local_digest_identical", "local_counts_identical", "initial_online_identical")) for r in consistency)
        and all(r["unchanged_during_evaluation"] and len(r["panel_checks"]) == panel_count for r in model_records)
        and all(r["complete"] for r in cell_checks) and all(s["outside_support_direct_samples"] == 0 for b in samples.values() for c in b.values() for s in c.values()))
    if status == "complete" and not complete:
        status, reason = "inconsistent_not_evidence", "expected training/evaluation/sampling/model invariants failed"
    aggregation_start = time.monotonic()
    seed_results, aggregate, references, paired, pooled = aggregate_banks(rows, refs, selection["panels"], seeds, bank_ids)
    loss_aggregate = []
    for bank_id in bank_ids:
        for condition in CONDITIONS:
            for checkpoint in sorted({r["checkpoint"] for r in losses if r["bank_id"] == bank_id and r["condition"] == condition}):
                group = [r for r in losses if r["bank_id"] == bank_id and r["condition"] == condition and r["checkpoint"] == checkpoint]
                loss_aggregate.append({"bank_id": bank_id, "condition": condition, "checkpoint": checkpoint,
                    "mean_loss": float(np.mean([r["mean_loss"] for r in group])), "seeds": len(group), "updates_in_window": group[0]["updates_in_window"]})
    effects = [r for r in paired["aggregate"] if r["panel"] == "all"]
    robustness = {"classification": "descriptive bank replication; no new competence gates or significance claims",
        "primary_metric": "greedy efficient-success difference", "primary_comparison": "uniform_minus_collected",
        "bank_effects": effects, "metrics": {}}
    for metric in METRICS:
        values = [r["mean_seed_" + metric + "_delta"] for r in effects if r["mean_seed_" + metric + "_delta"] is not None]
        if values:
            robustness["metrics"][metric + "_delta"] = {"banks": len(values), "minimum": min(values), "maximum": max(values),
                "mean": float(np.mean(values)), "positive_banks": sum(v > 1e-12 for v in values),
                "negative_banks": sum(v < -1e-12 for v in values), "zero_banks": sum(abs(v) <= 1e-12 for v in values), "sign_tolerance": 1e-12}
    thresholds = {"classification": "descriptive historical threshold crossings, not new competence gates", "per_seed": [], "per_condition": []}
    for bank_id in bank_ids:
        local_pairs = {key: [r for r in paired[key] if r["bank_id"] == bank_id] for key in ("per_seed", "aggregate", "per_layout")}
        local, _ = descriptive_summaries([r for r in seed_results if r["bank_id"] == bank_id],
            [r for r in aggregate if r["bank_id"] == bank_id], references, local_pairs, selection["panels"], seeds,
            status == "complete" and not smoke and not deviations, conditions=CONDITIONS, comparisons=COMPARISONS)
        for key in ("per_seed", "per_condition"):
            thresholds[key].extend({"bank_id": bank_id, **r} for r in local[key])
    try:
        enforce()
    except BudgetReached as exc:
        status, reason = ("incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"), str(exc)
    eligible = status == "complete" and not smoke and not deviations
    robustness["eligible"] = thresholds["eligible"] = eligible
    if not eligible:
        for row in thresholds["per_seed"]:
            row["success_reference_met"] = row["efficiency_reference_met"] = False
        for row in thresholds["per_condition"]:
            row["success_reference_panels"] = row["efficiency_reference_panels"] = 0
            for panel in row["per_panel"]:
                panel["all_seed_success_reference_met"] = panel["all_seed_efficiency_reference_met"] = False
    provenance = {"archives": archives_meta, "paired_sampling_consistency": consistency, "expected_cells": cell_checks,
        "training_arrays_identical": metadata.get("training_arrays_identical"), "transition_arrays_identical": metadata.get("transition_arrays_identical"),
        "all_banks_frozen_before_training": len(supports) == len(bank_ids), "all_fits_finished_before_evaluation": len(model_records) == len(bank_ids) * 2 * len(seeds),
        "models": model_records, "prior_initial_models": prior_initial, "initialization_consistency": initialization_consistency,
        "snapshot_integrity": snapshot_integrity, "support_integrity": support_integrity}
    artifacts = {key: artifact_path(output / value) for key, value in {"protocol": "protocol.json", "banks": "banks.json", "panels": "panels.json",
        "dataset": "dataset.npz", "transitions": "transitions.npz", "dataset_metadata": "dataset_metadata.json", "supports": "supports.npz",
        "training": "losses.csv", "evaluations": "evaluations.csv", "references": "references.csv", "sampling": "sampling.json",
        "sample_counts": "sample_counts.npz", "local_sample_counts": "local_sample_counts.npz", "models": "models.json",
        "paired_differences": "paired_differences.json", "support_intersections": "support_intersections.json", "snapshots": "snapshots.json", "paired_layouts": "paired_layouts.csv", "provenance": "provenance.json", "manifest": "manifest.json"}.items()}
    artifacts.update(directory=artifact_path(output), protocol_document=artifact_path(protocol_file), report="docs/experiments/bank_replication_results_v1.md")
    result = {"schema_version": 1, "protocol": protocol, "run": {"id": output.name, "status": status, "stop_reason": reason,
        "interpretation": "smoke_or_deviation_descriptive_only" if smoke or deviations else "independent_bank_replication_descriptive_only" if status == "complete" else "incomplete_not_evidence",
        "train_updates": total_updates, "training_examples": sum(r["examples_seen"] for bank in samples.values() for condition in bank.values() for r in condition.values()),
        "collection_episodes": sum(b["collection"]["collection_episodes"] for b in banks), "collection_steps": sum(b["collection"]["collection_steps"] for b in banks),
        "learner_episodes": len(rows), "reference_episodes": len(refs), "unique_reference_episodes": len({r["reference_sample_id"] for r in refs}),
        "banks": len(banks), "fits": len(model_records), "model_parameter_count": 20420, "panels": len(selection["panels"]),
        "wall_seconds": time.monotonic() - started, **{k + "_wall_seconds": v for k, v in timings.items()},
        "aggregation_wall_seconds": time.monotonic() - aggregation_start, "fit_wall_seconds": fit_timings, "model_panel_wall_seconds": model_panel_timings,
        "peak_rss_bytes": peak_rss_bytes(), "resource_checks": checks, "resource_limits": {"seconds": max_seconds, "peak_process_rss_bytes": max_rss_bytes, "torch_threads": 1},
        "progress": progress, "limitations": ["Three new bank pairs share the same 256 training layouts and fresh evaluation panels.",
            "The same learner initializations are shared across bank pairs; they are not nine independent bank replications.",
            "Counterfactual all-action access and full-bank detached successor queries remain privileged.",
            "Evaluation occurs only after every fit finishes; snapshots are not evaluated or selected retrospectively."]},
        "banks": banks, "panels": selection["panels"], "aggregate": aggregate, "seed_results": seed_results,
        "references": references, "paired_differences": paired, "pooled": pooled, "robustness": robustness,
        "descriptive_thresholds": thresholds, "support_intersections": support_intersections,
        "loss_aggregate": loss_aggregate, "sampling": samples, "provenance": provenance, "trajectories": trajectories, "artifacts": artifacts}
    for name, value in (("protocol", protocol), ("banks", banks), ("panels", selection), ("sampling", samples),
        ("models", model_records), ("snapshots", snapshot_records), ("support_intersections", support_intersections), ("provenance", provenance), ("paired_differences", paired), ("trajectories", trajectories), ("results", result)):
        write_json(output / f"{name}.json", value)
    with (output / "paired_layouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["bank_id", *PAIR_FIELDS])
        writer.writeheader()
        writer.writerows(paired["per_layout"])
    save_manifest(output, status)
    if dashboard:
        write_json(Path(dashboard), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--bank-ids", default="1,2,3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--episodes-per-map", type=int, default=16)
    parser.add_argument("--panel-count", type=int, default=8)
    parser.add_argument("--maps-per-panel", type=int, default=64)
    parser.add_argument("--panel-seed-start", type=int, default=1020000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--train-seed-start", type=int, default=300000)
    parser.add_argument("--train-maps", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    for name in ("supervised", "fixed-targets", "coverage", "equal-support", "panel-evaluation"):
        parser.add_argument("--" + name + "-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.updates, args.train_maps, args.panel_count, args.maps_per_panel = "0", 24, 2, 2, 2
        args.train_seed_start, args.panel_seed_start = 740000, 1100000
    archives = {name: getattr(args, name + "_dir") for name in ("supervised", "fixed_targets", "coverage", "equal_support", "panel_evaluation") if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, bank_ids=[int(v) for v in args.bank_ids.split(",")], seeds=[int(v) for v in args.seeds.split(",")],
        updates=args.updates, episodes_per_map=args.episodes_per_map, panel_count=args.panel_count, maps_per_panel=args.maps_per_panel,
        panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride, train_seed_start=args.train_seed_start, train_maps=args.train_maps,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
