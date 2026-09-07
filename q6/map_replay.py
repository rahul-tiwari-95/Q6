"""Equal-map replay on unchanged archived collected supports; frozen controls."""
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
from .bank_replication import METRICS, aggregate_banks
from .competence import BudgetReached, EVAL_FIELDS, ROOT, artifact_path, git_info
from .equal_support import support_metrics
from .fixed_targets import ConsistencyError, MemoryReached, array_metadata, fixed_update, guard, module_hash, peak_rss_bytes, sha, validate_transitions
from .learning import DQN
from .optimal import VisibleOptimalQ
from .panel_evaluation import FrozenPolicy, PAIR_FIELDS, select_panels, verify_expected_cells, descriptive_summaries
from .supervised import CHECKPOINTS, LOSS_FIELDS, rollout, save_manifest
from .world import WorldConfig

CONDITIONS = ("collected_unique", "map_balanced")
COMPARISONS = ({"id": "balanced_minus_collected", "left": "collected_unique", "right": "map_balanced"},)
ARCHIVES = ("supervised", "fixed_targets", "coverage", "equal_support", "panel_evaluation", "bank_replication")
BUCKETS = {"1_8": (1, 8), "9_16": (9, 16), "17_24": (17, 24), "25_32": (25, 32), "all": (1, 32)}


class MapSampler:
    """Two independent streams preserve map pairing despite different row bounds."""
    def __init__(self, support, map_seeds, seed, batch_size=64):
        self.support = np.asarray(support, dtype=np.int32)
        self.map_ids = np.unique(map_seeds)
        if (batch_size > len(self.map_ids) or batch_size < 1 or not len(self.support)
                or np.any(np.diff(self.support) <= 0) or self.support[0] < 0 or self.support[-1] >= len(map_seeds)):
            raise ConsistencyError("configuration failure: invalid support or too few maps for a distinct-map batch")
        self.rows_by_map = [self.support[map_seeds[self.support] == map_seed] for map_seed in self.map_ids]
        if any(not len(rows) for rows in self.rows_by_map):
            raise ConsistencyError("configuration failure: every training map needs a supported state")
        self.map_rng = np.random.default_rng(np.random.SeedSequence([seed, 99301]))
        self.state_rng = np.random.default_rng(np.random.SeedSequence([seed, 99302]))
        self.batch_size, self.updates = batch_size, 0
        self.counts = np.zeros(len(map_seeds), np.uint32)
        self.local_counts = np.zeros(len(support), np.uint32)
        self.map_counts = np.zeros(len(self.map_ids), np.uint32)
        self.map_digest, self.rank_digest, self.local_digest, self.digest = (hashlib.sha256() for _ in range(4))

    def next_batch(self):
        maps = self.map_rng.choice(len(self.map_ids), size=self.batch_size, replace=False)
        ranks = np.asarray([int(self.state_rng.integers(len(self.rows_by_map[m]))) for m in maps], np.int64)
        rows = np.asarray([self.rows_by_map[m][r] for m, r in zip(maps, ranks)], np.int64)
        local = np.searchsorted(self.support, rows).astype(np.int64)
        self.counts[rows] += 1
        self.local_counts[local] += 1
        self.map_counts[maps] += 1
        for digest, values in ((self.map_digest, self.map_ids[maps]), (self.rank_digest, ranks), (self.local_digest, local), (self.digest, rows)):
            digest.update(values.astype("<i8").tobytes())
        self.updates += 1
        return rows


def exposure_metrics(data, transitions, supports, counts, sampling, *, control_condition="collected_unique"):
    """Archived membership and actual presentation frequency are separate facts."""
    result = {"classification": "descriptive actual training exposure; baseline counts are historical", "per_map": [], "per_clock": [], "summaries": []}
    # Enumeration clocks are innermost; a position is goal-near iff winnable at clock2.
    goal_near = np.zeros(len(data["observations"]), bool)
    for map_seed in np.unique(data["map_seeds"]):
        rows = np.flatnonzero(data["map_seeds"] == map_seed)
        for position in np.unique(data["positions"][rows], axis=0):
            local = rows[np.all(data["positions"][rows] == position, axis=1)]
            goal_near[local] = np.any(data["winnable"][local] & (data["remaining"][local] <= 2))
    for (bank, condition, seed), count in counts.items():
        support = supports[bank][condition]
        mask = np.zeros(len(count), bool)
        mask[support] = True
        sample = sampling[str(bank)][condition][str(seed)]
        total = int(count.sum())
        base = {"bank_id": bank, "condition": condition, "seed": seed, "source": "archived_baseline" if condition == control_condition else "new_treatment", "updates": sample["updates"]}
        def entry(selected):
            presentations = int(count[selected].sum())
            return {"supported_states": int((mask & selected).sum()), "presentations": presentations,
                "presentation_fraction": presentations / total if total else None, "unique_states_sampled": int(np.count_nonzero(count[selected]))}
        map_rows = [{**base, "map_seed": int(m), **entry(data["map_seeds"] == m)} for m in np.unique(data["map_seeds"])]
        result["per_map"].extend(map_rows)
        result["per_clock"].extend({**base, "bucket": label, **entry((data["remaining"] >= low) & (data["remaining"] <= high))} for label, (low, high) in BUCKETS.items())
        fractions = np.asarray([r["presentations"] for r in map_rows], np.float64) / total if total else np.zeros(len(map_rows))
        present = count[support]
        categories = {name: entry(selected) for name, selected in (("winnable", data["winnable"]), ("impossible", ~data["winnable"]), ("goal_near", goal_near), ("goal_far", ~goal_near))}
        live = ~transitions["ends"]
        outside = live & ~mask[np.maximum(transitions["successor_indices"], 0)]
        queries = int(np.dot(count.astype(np.uint64), live.sum(1).astype(np.uint64)))
        outside_queries = int(np.dot(count.astype(np.uint64), outside.sum(1).astype(np.uint64)))
        result["summaries"].append({**base, "presentations": total, "unique_states_sampled": int(np.count_nonzero(count)),
            "map_fraction_min": float(fractions.min()), "map_fraction_max": float(fractions.max()),
            "map_fraction_cv": float(fractions.std() / fractions.mean()) if total else None,
            "state_count_distribution": {"denominator": "all supported current states, including zero presentations", "states": len(present),
                "minimum": int(present.min()), "maximum": int(present.max()), "mean": float(present.mean()),
                "median": float(np.median(present)), "q95": float(np.quantile(present, .95)), "zero_states": int((present == 0).sum())},
            "category_exposure": categories, "outside_support_direct_samples": int(count[~mask].sum()),
            "successor_queries": {"nonterminal_queries": queries, "outside_support_queries": outside_queries, "outside_support_fraction": outside_queries / queries if queries else None}})
    return result


def run_study(output, protocol_file, *, bank_ids=(1, 2, 3), seeds=(0, 1, 2), updates=30000,
              panel_count=8, maps_per_panel=64, panel_seed_start=1040000, panel_stride=1000,
              max_seconds=1200, max_rss_bytes=4 * 1024**3, dashboard=None, archives=None, smoke=False, checkpoints=None, _comparison=None):
    conditions = _comparison.conditions if _comparison else CONDITIONS
    comparisons = _comparison.comparisons if _comparison else COMPARISONS
    control, treatment = conditions
    control_archive = getattr(_comparison, "control_archive", "bank_replication")
    module = _comparison.module if _comparison else "q6.map_replay"
    started = time.monotonic()
    output, protocol_file = Path(output), Path(protocol_file)
    bank_ids, seeds = list(bank_ids), list(seeds)
    schedule = sorted(set(checkpoints if checkpoints is not None else [c for c in CHECKPOINTS if c <= updates] + [updates]))
    if not protocol_file.is_file() or not bank_ids or not seeds or len(set(bank_ids)) != len(bank_ids) or len(set(seeds)) != len(seeds):
        raise ValueError("existing protocol and distinct bank/learner identifiers required")
    if not set(bank_ids).issubset({1, 2, 3}) or min(seeds) < 0 or min(updates, panel_count, maps_per_panel, panel_stride, max_seconds, max_rss_bytes) <= 0 or not schedule or schedule[0] != 0 or schedule[-1] != updates:
        raise ValueError("positive budgets and snapshot schedule spanning zero to final required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    directories = {name: ROOT / f"experiments/{name}/pilot_v1" for name in (ARCHIVES + (_comparison.extra_archives if _comparison else ())) }
    directories.update({name: Path(path) for name, path in (archives or {}).items()})
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "torch_threads": 1}
    actual = dict(bank_ids=bank_ids, seeds=seeds, updates=updates, panel_count=panel_count, maps_per_panel=maps_per_panel,
        panel_seed_start=panel_seed_start, panel_stride=panel_stride, max_seconds=max_seconds, max_rss_bytes=max_rss_bytes, checkpoints=schedule)
    declared = dict(bank_ids=[1, 2, 3], seeds=[0, 1, 2], updates=30000, panel_count=8, maps_per_panel=64,
        panel_seed_start=_comparison.panel_seed_start if _comparison else 1040000, panel_stride=1000, max_seconds=1200, max_rss_bytes=4 * 1024**3, checkpoints=list(CHECKPOINTS))
    deviations = [{"field": k, "actual": v, "declared": declared[k]} for k, v in actual.items() if v != declared[k]]
    if not runtime["python"].startswith("3.12.") or runtime["torch"].split("+")[0] != "2.8.0" or runtime["numpy"] != "2.0.2":
        deviations.append({"field": "runtime", "actual": runtime, "declared": "Python3.12/Torch2.8.0/NumPy2.0.2"})
    for name, path in directories.items():
        if path.resolve() != (ROOT / f"experiments/{name}/pilot_v1").resolve():
            deviations.append({"field": "archive_" + name, "actual": artifact_path(path), "declared": f"experiments/{name}/pilot_v1"})
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    protocol = {"id": "map-replay-v1", "question": "Does equal-map replay improve efficiency using unchanged collected experience?",
        "conditions": [{"id": control, "label": "Collected unique (frozen control)"}, {"id": treatment, "label": "Map-balanced replay"}],
        "bank_ids": bank_ids, "seeds": seeds, "checkpoints": schedule, "evaluation_checkpoints": {control: 30000, treatment: updates},
        "panels": [], "world": asdict(WorldConfig()), "rule_visibility": "observed", "runtime": runtime, "git": git_info(), "smoke": bool(smoke), "deviations": deviations,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}, "protocol_sha256": sha(protocol_file), "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": {"new_steps": 0, "new_episodes": 0, "new_support_draws": 0, "source": "bank_replication/pilot_v1 collected banks1,2,3"},
        "optimizer": {"implementation": "unchanged fixed_targets.fixed_update(...,'double_dqn')", "learning_rate": .001, "loss": "all-four-action SmoothL1 mean beta1", "gradient_norm_cap": 5., "gamma": .97, "target_tau": .01, "batch_size": 64},
        "sampling": {"map_rng": "SeedSequence([seed,99301])", "state_rng": "SeedSequence([seed,99302])", "draw": "choice(256,64,replace=False), retain order; one scalar integers(K_map) per map", "map_schedule_shared_across_banks": True, "full_bank_detached_successors": True},
        "evaluation": {"final_only": True, "after_all_treatment_fits": True, "greedy_repetitions": 1, "epsilon_0_1_repetitions": 2, "rng": "SeedSequence([seed,map_seed,repetition,55219])", "greedy_ties": "lowest label", "optimal_q_atol": 1e-6, "optimal_q_rtol": 0},
        "panel_selection": {"count": panel_count, "maps_per_panel": maps_per_panel, "start": panel_seed_start, "stride": panel_stride},
        "budget": {"maximum_updates": len(bank_ids) * len(seeds) * updates, "updates_per_fit": updates, "new_baseline_updates": 0, "maximum_collection_steps": 0, "maximum_collection_episodes": 0, "admission_seconds": max_seconds, "peak_process_rss_bytes": max_rss_bytes, "all_phases_included": True},
        "primary_comparison": comparisons[0]["id"], "primary_metric": "greedy efficient success", "new_competence_gates": False}
    if _comparison:
        _comparison.configure_protocol(protocol)
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
    command = ["python", "-m", module, "--output", str(output), "--protocol-file", str(protocol_file)]
    for key, value in actual.items():
        command.extend(["--" + key.replace("_", "-"), ",".join(map(str, value)) if isinstance(value, list) else str(value)])
    for name, path in directories.items():
        command.extend(["--" + name.replace("_", "-") + "-dir", artifact_path(path)])
    if smoke:
        command.append("--smoke")
    (output / "command.txt").write_text(shlex.join(command) + "\n# Reproduction needs a new output path.\n")
    write_json(output / "protocol.json", protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config, deadline, checks = WorldConfig(), started + max_seconds, 0
    def enforce():
        nonlocal checks
        checks += 1
        guard(deadline, max_rss_bytes)
    status, reason = "complete", None
    data, transitions, metadata, archives_meta, supports, counts, local_counts, map_counts, samples, samplers = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    banks, model_records, snapshot_records, rows, refs, trajectories, losses, progress, support_integrity = [], [], [], [], [], [], [], [], []
    selection, models, prior_initial = {"panels": []}, {}, {}
    timings = {"preparation": 0., "training": 0., "references": 0., "evaluation": 0.}
    fit_timings, total_updates, current_window, active = {}, 0, [], None
    with (output / "losses.csv").open("w", newline="") as lf, (output / "evaluations.csv").open("w", newline="") as ef, (output / "references.csv").open("w", newline="") as rf:
        lw = csv.DictWriter(lf, ["bank_id", "condition", *LOSS_FIELDS])
        ew, rw = csv.DictWriter(ef, ["bank_id", *EVAL_FIELDS]), csv.DictWriter(rf, ["bank_id", *EVAL_FIELDS])
        for writer in (lw, ew, rw):
            writer.writeheader()
        try:
            phase = time.monotonic()
            try:
                enforce()
                excluded, manifests = {}, {}
                for name, directory in directories.items():
                    manifest = manifests[name] = json.loads((directory / "manifest.json").read_text())
                    if manifest["status"] != "complete":
                        raise ConsistencyError("input archive is not complete")
                    archives_meta[name] = {"directory": artifact_path(directory), "files": {"manifest.json": sha(directory / "manifest.json")}}
                    shutil.copy2(directory / "manifest.json", output / f"{name}_manifest.json")
                    filename = "panels.json" if name in ("panel_evaluation", "bank_replication", "map_replay", "within_map", "recorded_actions", "constrained_bootstrap") else "dataset_metadata.json"
                    checked_input(directory, manifest, filename, archives_meta[name]["files"])
                    saved = json.loads((directory / filename).read_text())
                    shutil.copy2(directory / filename, output / f"{name}_{filename}")
                    layouts = [r for p in saved["panels"] for r in p["layouts"]] if filename == "panels.json" else saved["heldout"]
                    excluded[f"previous_{name}_fresh_layout"] = {r["layout_hash"] for r in layouts}
                for name in ("coverage", "bank_replication"):
                    directory, manifest = directories[name], manifests[name]
                    for filename in ("protocol.json", "protocol.md"):
                        checked_input(directory, manifest, filename, archives_meta[name]["files"])
                        shutil.copy2(directory / filename, output / f"{name}_{filename}")
                    for core_module in ("world", "learning", "competence", "supervised", "optimal", "fixed_targets"):
                        filename = f"source/q6/{core_module}.py"
                        checked_input(directory, manifest, filename, archives_meta[name]["files"])
                        if sha(directory / filename) != sha(ROOT / f"q6/{core_module}.py"):
                            raise ConsistencyError(f"core algorithm source differs from {name} archive: {core_module}")
                directory, manifest = directories["bank_replication"], manifests["bank_replication"]
                for filename in ("dataset.npz", "transitions.npz", "dataset_metadata.json", "supports.npz", "sampling.json", "sample_counts.npz", "local_sample_counts.npz", "banks.json", "models.json"):
                    checked_input(directory, manifest, filename, archives_meta["bank_replication"]["files"])
                metadata = json.loads((directory / "dataset_metadata.json").read_text())
                with np.load(directory / "dataset.npz") as saved:
                    data = {k.removeprefix("train_"): saved[k] for k in saved.files if k.startswith("train_")}
                with np.load(directory / "transitions.npz") as saved:
                    transitions = {k: saved[k] for k in saved.files}
                coverage_meta = json.loads((directories["coverage"] / "dataset_metadata.json").read_text())
                array_info = array_metadata({"train_" + k: v for k, v in data.items()})
                transition_info = array_metadata(transitions)
                if array_info != metadata["arrays"] or not all(v == coverage_meta["arrays"][k] for k, v in array_info.items()) or transition_info != metadata["transition_arrays"] or transition_info != coverage_meta["transition_arrays"]:
                    raise ConsistencyError("training arrays/transitions differ from archived identities")
                for filename in ("dataset.npz", "transitions.npz"):
                    checked_input(directories["coverage"], manifests["coverage"], filename, archives_meta["coverage"]["files"])
                    shutil.copy2(directory / filename, output / filename)
                metadata = {**metadata, "training_arrays_identical": True, "transition_arrays_identical": True,
                    "transition_validation": validate_transitions(config, data, transitions, deadline)}
                write_json(output / "dataset_metadata.json", metadata)
                if [r["map_seed"] for r in metadata["train"]] != list(range(300000, 300256)):
                    raise ConsistencyError("training bank must be the declared full 256 maps")
                if _comparison and hasattr(_comparison, "prepare_inputs"):
                    _comparison.prepare_inputs(directories, manifests, archives_meta, data, transitions, output, enforce)
                excluded["training_layout"] = {r["layout_hash"] for r in metadata["train"]}
                selection = select_panels(config, excluded, panel_count, maps_per_panel, panel_seed_start, panel_stride, enforce)
                protocol["panels"] = selection["panels"]
                protocol["dataset"] = {"train_states": len(data["observations"]), "train_map_seeds": [r["map_seed"] for r in metadata["train"]], "fresh_state_enumeration": False}
                write_json(output / "panels.json", selection)
                write_json(output / "protocol.json", protocol)
                old_banks = json.loads((directory / "banks.json").read_text())
                control_directory, control_manifest = directories[control_archive], manifests[control_archive]
                for filename in ("sampling.json", "sample_counts.npz", "local_sample_counts.npz"):
                    checked_input(control_directory, control_manifest, filename, archives_meta[control_archive]["files"])
                old_sampling = json.loads((control_directory / "sampling.json").read_text())
                with np.load(directory / "supports.npz") as arrays, np.load(control_directory / "sample_counts.npz") as global_archive, np.load(control_directory / "local_sample_counts.npz") as local_archive:
                    for bank in bank_ids:
                        support = arrays[f"collected_unique_bank{bank}"]
                        (_comparison.make_sampler(support, data["map_seeds"], 0) if _comparison else MapSampler(support, data["map_seeds"], 0))  # Validate sampler availability before any optimizer.
                        support.flags.writeable = False
                        supports[bank] = _comparison.prepare_support(bank, support, data, enforce) if _comparison else {c: support for c in conditions}
                        original = next(r for r in old_banks if r["bank_id"] == bank)
                        bank_info = {"bank_id": bank, "status": "complete", "support_size": len(support), "arrays": array_metadata(supports[bank]),
                            "collection": {**original["collection"], "source": "inherited_archive_not_new_collection"},
                            "intersection": {"states": len(support), "union_states": len(support), "fraction_of_each": 1., "jaccard": 1.},
                            "coverage": {"per_condition": [{"condition": c, **support_metrics(data, transitions, supports[bank][c])} for c in conditions]}}
                        if _comparison:
                            bank_info.update(_comparison.bank_metadata(bank, supports[bank], data))
                        banks.append(bank_info)
                        if _comparison:
                            np.savez_compressed(output / "supports.npz", **{f"{c}_bank{b}": a for b, pair in supports.items() for c, a in pair.items()})
                            write_json(output / "banks.json", banks)
                        for seed in seeds:
                            key, name = (bank, control, seed), f"bank{bank}_{control}_seed{seed}"
                            counts[key], local_counts[key] = global_archive[name], local_archive[name]
                            map_counts[key] = np.asarray([counts[key][data["map_seeds"] == m].sum() for m in np.unique(data["map_seeds"])], np.uint32)
                            samples.setdefault(str(bank), {}).setdefault(control, {})[str(seed)] = {**old_sampling[str(bank)][control][str(seed)], "historical": True, "new_updates": 0}
                            if _comparison:
                                _comparison.record_baseline(bank, seed, support, data, counts[key], local_counts[key], samples[str(bank)][control][str(seed)], enforce, updates)
                            initial_name = f"models/{name}_update0.pt"
                            checked_input(control_directory, control_manifest, initial_name, archives_meta[control_archive]["files"])
                            initial = torch.load(control_directory / initial_name, map_location="cpu", weights_only=True)
                            prior_initial[f"{bank}:{seed}"] = {"online_hash": initial["parameter_hash"], "target_hash": initial["target_parameter_hash"], "source": artifact_path(control_directory / initial_name), "sha256": sha(control_directory / initial_name)}
                            final_name = f"models/{name}_update30000.pt"
                            checked_input(control_directory, control_manifest, final_name, archives_meta[control_archive]["files"])
                            shutil.copy2(control_directory / final_name, output / final_name)
                            final = torch.load(output / final_name, map_location="cpu", weights_only=True)
                            policy = FrozenPolicy(final)
                            model_records.append({"bank_id": bank, "condition": control, "seed": seed, "checkpoint": 30000,
                                "initial_online_hash": initial["parameter_hash"], "initial_target_hash": initial["target_parameter_hash"],
                                "saved": final_name, "source": artifact_path(control_directory / final_name), "source_before": sha(control_directory / final_name),
                                "file_before": sha(output / final_name), "online_before": policy.parameter_hash(), "target_before": module_hash(policy.target), "panel_checks": []})
                            snapshot_records.append({"bank_id": bank, "condition": control, "seed": seed, "checkpoint": 30000,
                                "saved": final_name, "sha256": sha(output / final_name), "online_hash": policy.parameter_hash(), "target_hash": module_hash(policy.target), "historical": True})
                            if final["optimizer_updates"] != 30000 or sha(output / final_name) != sha(control_directory / final_name):
                                raise ConsistencyError("frozen baseline checkpoint/copy differs")
                np.savez_compressed(output / "supports.npz", **{f"{c}_bank{b}": a for b, pair in supports.items() for c, a in pair.items()})
                write_json(output / "banks.json", banks)
                enforce()
            finally:
                timings["preparation"] += time.monotonic() - phase
            custom_update = _comparison is not None and hasattr(_comparison, "update")
            train_obs = torch.from_numpy(data["observations"])
            unused_exact = None if custom_update else torch.from_numpy(data["targets"].astype(np.float32))
            tensors = {} if custom_update else {k: torch.from_numpy(v.astype(np.float32) if k == "rewards" else v) for k, v in transitions.items()}
            for bank in bank_ids:
                for seed in seeds:
                    enforce()
                    active = bank, treatment, seed
                    agent = DQN(92, seed=seed)
                    initial = prior_initial[f"{bank}:{seed}"]
                    if agent.parameter_hash() != initial["online_hash"] or module_hash(agent.target) != initial["target_hash"]:
                        raise ConsistencyError("treatment initialization differs from archived control")
                    sampler = samplers[active] = (_comparison.make_sampler(supports[bank][treatment], data["map_seeds"], seed) if _comparison else MapSampler(supports[bank][treatment], data["map_seeds"], seed))
                    phase, current_window = time.monotonic(), []
                    try:
                        for checkpoint in schedule:
                            while sampler.updates < checkpoint:
                                enforce()
                                indices = sampler.next_batch()
                                loss = (_comparison.update(agent, train_obs, indices, bank, seed) if custom_update
                                    else fixed_update(agent, train_obs, unused_exact, tensors, indices, "double_dqn"))
                                total_updates += 1
                                if not np.isfinite(loss):
                                    raise ConsistencyError("nonfinite optimizer loss")
                                current_window.append(loss)
                                if sampler.updates % 100 == 0 or sampler.updates == checkpoint:
                                    row = {"bank_id": bank, "condition": treatment, "seed": seed, "checkpoint": sampler.updates,
                                        "updates_in_window": len(current_window), "mean_loss": float(np.mean(current_window)), "last_loss": loss, "training_examples": sampler.updates * 64}
                                    losses.append(row)
                                    lw.writerow(row)
                                    current_window = []
                            if not all(torch.isfinite(p).all().item() for net in (agent.online, agent.target) for p in net.parameters()):
                                raise ConsistencyError("nonfinite model weights")
                            path = output / "models" / f"bank{bank}_{treatment}_seed{seed}_update{checkpoint}.pt"
                            torch.save({"online": agent.online.state_dict(), "target": agent.target.state_dict(), "observation_size": 92,
                                "parameter_hash": agent.parameter_hash(), "target_parameter_hash": module_hash(agent.target), "optimizer_updates": checkpoint, "purpose": "inference_only_not_resumable"}, path)
                            snapshot_records.append({"bank_id": bank, "condition": treatment, "seed": seed, "checkpoint": checkpoint,
                                "saved": path.relative_to(output).as_posix(), "sha256": sha(path), "online_hash": agent.parameter_hash(), "target_hash": module_hash(agent.target), "historical": False})
                            if _comparison and hasattr(_comparison, "checkpoint_probe"):
                                _comparison.checkpoint_probe(agent, train_obs, bank, seed, checkpoint, path)
                            progress.append({"bank_id": bank, "condition": treatment, "seed": seed, "checkpoint": checkpoint, "snapshot_saved": True})
                            print(f"bank{bank} {treatment} seed={seed}: saved update {checkpoint}", flush=True)
                        model_records.append({"bank_id": bank, "condition": treatment, "seed": seed, "checkpoint": updates,
                            "initial_online_hash": initial["online_hash"], "initial_target_hash": initial["target_hash"], "saved": path.relative_to(output).as_posix(),
                            "source": artifact_path(path), "source_before": sha(path), "file_before": sha(path), "online_before": agent.parameter_hash(), "target_before": module_hash(agent.target), "panel_checks": []})
                        progress.append({"bank_id": bank, "seed": seed, "training_complete": True, "updates": sampler.updates})
                    finally:
                        duration = time.monotonic() - phase
                        timings["training"] += duration
                        fit_timings[f"bank{bank}:{treatment}:seed{seed}"] = duration
                        lf.flush()
                    del agent
            # Every treatment fit finishes before either arm makes a learned prediction.
            for record in model_records:
                enforce()
                models[(record["bank_id"], record["condition"], record["seed"])] = FrozenPolicy(torch.load(output / record["saved"], map_location="cpu", weights_only=True))
            if _comparison and hasattr(_comparison, "final_fit_diagnostics"):
                _comparison.final_fit_diagnostics(models, model_records, data, output, enforce)
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
                    model = models[(record["bank_id"], record["condition"], record["seed"])]
                    check = {"panel": panel["id"], **model_boundary(record, model, output, "before")}
                    record["panel_checks"].append(check)
                    if not all(check[k + "_before"] == record[k + "_before"] for k in ("online", "target", "file", "source")):
                        raise ConsistencyError("frozen model/source changed before panel evaluation")
                    group, replay_group, phase = [], [], time.monotonic()
                    try:
                        for mode in ("greedy", "epsilon_0_1"):
                            for layout in panel["layouts"]:
                                for rep in range(1 if mode == "greedy" else 2):
                                    enforce()
                                    row, replay = rollout(model, config, oracle, condition=record["condition"], seed=record["seed"], checkpoint=record["checkpoint"],
                                        mode=mode, panel=panel["id"], map_seed=layout["map_seed"], repetition=rep, deadline=deadline)
                                    row["bank_id"] = replay["bank_id"] = record["bank_id"]
                                    group.append(row)
                                    if layout == panel["layouts"][0] and rep == 0:
                                        replay_group.append(replay)
                    except BudgetReached:
                        for row in group:
                            row["checkpoint_complete"] = 0
                        rows.extend(group)
                        ew.writerows(group)
                        raise
                    finally:
                        timings["evaluation"] += time.monotonic() - phase
                        check.update(model_boundary(record, model, output, "after"))
                        check["unchanged"] = all(check[k + "_before"] == check[k + "_after"] == record[k + "_before"] for k in ("online", "target", "file", "source"))
                        if not check["unchanged"]:
                            raise ConsistencyError("frozen source/copy/module identity changed")
                    rows.extend(group)
                    ew.writerows(group)
                    trajectories.extend(replay_group)
                    progress.append({"bank_id": record["bank_id"], "condition": record["condition"], "seed": record["seed"], "panel": panel["id"], "evaluation_complete": True})
                    ef.flush()
                print(f"{panel['id']}: all frozen models evaluated", flush=True)
        except (BudgetReached, ConsistencyError) as exc:
            status = "inconsistent_not_evidence" if isinstance(exc, ConsistencyError) else "incomplete_memory_cap" if isinstance(exc, MemoryReached) else "incomplete_admission_cap"
            reason = str(exc)
            if current_window and active:
                bank, condition, seed = active
                row = {"bank_id": bank, "condition": condition, "seed": seed, "checkpoint": sampler.updates, "updates_in_window": len(current_window),
                    "mean_loss": float(np.mean(current_window)), "last_loss": current_window[-1], "training_examples": sampler.updates * 64}
                losses.append(row)
                lw.writerow(row)
    aggregation_start = time.monotonic()
    for (bank, condition, seed), sampler in samplers.items():
        key = bank, condition, seed
        counts[key], local_counts[key], map_counts[key] = sampler.counts, sampler.local_counts, sampler.map_counts
        if _comparison:
            samples.setdefault(str(bank), {}).setdefault(condition, {})[str(seed)] = _comparison.sampling_metadata(seed, sampler)
        else:
            samples.setdefault(str(bank), {}).setdefault(condition, {})[str(seed)] = {"updates": sampler.updates, "new_updates": sampler.updates, "historical": False,
                "examples_seen": int(sampler.counts.sum()), "unique_states_sampled": int(np.count_nonzero(sampler.counts)), "support_states": len(sampler.support),
                "local_batch_index_sha256": sampler.local_digest.hexdigest(), "global_batch_index_sha256": sampler.digest.hexdigest(),
                "map_index_sha256": sampler.map_digest.hexdigest(), "within_map_rank_sha256": sampler.rank_digest.hexdigest(),
                "map_rng_seed_tuple": [seed, 99301], "state_rng_seed_tuple": [seed, 99302], "map_ids": sampler.map_ids.tolist(),
                "outside_support_direct_samples": int(sampler.counts.sum() - sampler.counts[sampler.support].sum())}
    for filename, arrays in (("sample_counts", counts), ("local_sample_counts", local_counts), ("map_counts", map_counts)):
        np.savez_compressed(output / f"{filename}.npz", **{f"bank{b}_{c}_seed{s}": a for (b, c, s), a in arrays.items()})
    exposure = exposure_metrics(data, transitions, supports, counts, samples, control_condition=control) if counts else {"per_map": [], "per_clock": [], "summaries": []}
    extra_integrity = (_comparison.finalize_exposure(data, transitions, supports, counts, samples, exposure, output)
        if _comparison and hasattr(_comparison, "finalize_exposure") else {"complete": True})
    for row in exposure["summaries"]:
        samples[str(row["bank_id"])][row["condition"]][str(row["seed"])]["successor_queries"] = row["successor_queries"]
    if _comparison:
        consistency = _comparison.consistency(samplers, bank_ids, seeds, updates)
    else:
        consistency = []
        for seed in seeds:
            group = [samplers[(bank, treatment, seed)] for bank in bank_ids if (bank, treatment, seed) in samplers]
            consistency.append({"seed": seed, "fits": len(group), "complete": len(group) == len(bank_ids) and all(s.updates == updates for s in group),
                "map_digest_identical": bool(group) and len({s.map_digest.hexdigest() for s in group}) == 1,
                "map_counts_identical": bool(group) and all(np.array_equal(s.map_counts, group[0].map_counts) for s in group)})
    initialization_consistency = []
    for bank in bank_ids:
        for seed in seeds:
            group = [r for r in model_records if r["bank_id"] == bank and r["seed"] == seed]
            initialization_consistency.append({"bank_id": bank, "seed": seed, "models": len(group),
                "online_identical": len(group) == 2 and len({r["initial_online_hash"] for r in group}) == 1,
                "target_identical": len(group) == 2 and len({r["initial_target_hash"] for r in group}) == 1})
    for bank in banks:
        after = array_metadata(supports[bank["bank_id"]])
        support_integrity.append({"bank_id": bank["bank_id"], "before": bank["arrays"], "after": after,
            "unchanged": after == bank["arrays"], "read_only": all(not a.flags.writeable for a in supports[bank["bank_id"]].values())})
    for record in model_records:
        model = models.get((record["bank_id"], record["condition"], record["seed"]))
        if model:
            record.update(model_boundary(record, model, output, "after"))
        else:
            record.update(source_after=sha(ROOT / record["source"]), file_after=sha(output / record["saved"]), online_after=None, target_after=None)
        record["unchanged_during_evaluation"] = all(record[k + "_before"] == record[k + "_after"] for k in ("online", "target", "file", "source")) and all(r.get("unchanged", False) for r in record["panel_checks"])
    cell_checks = [{"bank_id": bank, **verify_expected_cells([r for r in rows if r["bank_id"] == bank], refs, selection["panels"], seeds, conditions=conditions)} for bank in bank_ids]
    expected_snapshots = {f"models/bank{bank}_{treatment}_seed{seed}_update{cp}.pt" for bank in bank_ids for seed in seeds for cp in schedule}
    expected_snapshots.update(f"models/bank{bank}_{control}_seed{seed}_update30000.pt" for bank in bank_ids for seed in seeds)
    snapshot_integrity = {"expected": len(expected_snapshots), "actual": len(snapshot_records), "complete": {r["saved"] for r in snapshot_records} == expected_snapshots and len(snapshot_records) == len(expected_snapshots),
        "unchanged": all(sha(output / r["saved"]) == r["sha256"] for r in snapshot_records)}
    source_integrity = [{"archive": name, "file": filename, "before": digest, "after": sha(directories[name] / filename),
        "unchanged": sha(directories[name] / filename) == digest} for name, archive in archives_meta.items() for filename, digest in archive["files"].items()]
    count_integrity = [{"bank_id": b, "condition": c, "seed": s,
        "sum_matches": int(a.sum()) == samples[str(b)][c][str(s)]["updates"] * 64,
        "local_matches": np.array_equal(local_counts[(b, c, s)], a[supports[b][c]]),
        "map_sum_matches": int(map_counts[(b, c, s)].sum()) == int(a.sum()),
        "on_support": int(a[supports[b][c]].sum()) == int(a.sum())} for (b, c, s), a in counts.items()]
    expected_windows = len(set(range(100, updates + 1, 100)) | set(schedule[1:])) * len(bank_ids) * len(seeds)
    loss_integrity = {"expected_windows": expected_windows, "actual_windows": len(losses),
        "complete": len(losses) == expected_windows and sum(r["updates_in_window"] for r in losses) == total_updates,
        "finite": all(np.isfinite(r["mean_loss"]) and np.isfinite(r["last_loss"]) for r in losses)}
    complete = (extra_integrity["complete"] and loss_integrity["complete"] and loss_integrity["finite"] and snapshot_integrity["complete"] and snapshot_integrity["unchanged"] and len(supports) == len(bank_ids)
        and all(r["unchanged"] and r["read_only"] for r in support_integrity)
        and all(r["complete"] and r["map_digest_identical"] and r["map_counts_identical"] for r in consistency)
        and all(r["online_identical"] and r["target_identical"] for r in initialization_consistency)
        and all(r["unchanged_during_evaluation"] and len(r["panel_checks"]) == panel_count for r in model_records)
        and all(r["unchanged"] for r in source_integrity) and all(all(r[k] for k in ("sum_matches", "local_matches", "map_sum_matches", "on_support")) for r in count_integrity)
        and len(model_records) == 2 * len(bank_ids) * len(seeds) and len(counts) == len(model_records)
        and total_updates == protocol["budget"]["maximum_updates"] and all(r["complete"] for r in cell_checks)
        and len(trajectories) == panel_count * (len(bank_ids) * len(seeds) * 2 * 2 + 2))
    if status == "complete" and not complete:
        status, reason = "inconsistent_not_evidence", "expected training/evaluation/sampling/model invariants failed"
    seed_results, aggregate, references, paired, pooled = aggregate_banks(rows, refs, selection["panels"], seeds, bank_ids, conditions=conditions, comparisons=comparisons)
    loss_aggregate = []
    for bank in bank_ids:
        for cp in sorted({r["checkpoint"] for r in losses if r["bank_id"] == bank}):
            group = [r for r in losses if r["bank_id"] == bank and r["checkpoint"] == cp]
            loss_aggregate.append({"bank_id": bank, "condition": treatment, "checkpoint": cp, "mean_loss": float(np.mean([r["mean_loss"] for r in group])), "seeds": len(group), "updates_in_window": group[0]["updates_in_window"]})
    effects = [r for r in paired["aggregate"] if r["panel"] == "all"]
    robustness = {"classification": "descriptive replay intervention on three previously drawn banks; no new gates or significance claims",
        "primary_metric": "greedy efficient-success difference", "primary_comparison": comparisons[0]["id"], "bank_effects": effects, "metrics": {}}
    for metric in METRICS:
        values = [r["mean_seed_" + metric + "_delta"] for r in effects if r["mean_seed_" + metric + "_delta"] is not None]
        if values:
            robustness["metrics"][metric + "_delta"] = {"banks": len(values), "minimum": min(values), "maximum": max(values), "mean": float(np.mean(values)),
                "positive_banks": sum(v > 1e-12 for v in values), "negative_banks": sum(v < -1e-12 for v in values), "zero_banks": sum(abs(v) <= 1e-12 for v in values), "sign_tolerance": 1e-12}
    thresholds = {"classification": "descriptive historical threshold crossings, not new competence gates", "per_seed": [], "per_condition": []}
    for bank in bank_ids:
        local_pairs = {key: [r for r in paired[key] if r["bank_id"] == bank] for key in ("per_seed", "aggregate", "per_layout")}
        local, _ = descriptive_summaries([r for r in seed_results if r["bank_id"] == bank], [r for r in aggregate if r["bank_id"] == bank],
            references, local_pairs, selection["panels"], seeds, status == "complete" and not smoke and not deviations, conditions=conditions, comparisons=comparisons)
        for key in ("per_seed", "per_condition"):
            thresholds[key].extend({"bank_id": bank, **r} for r in local[key])
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
    provenance = {"archives": archives_meta, "paired_map_sampling_consistency": consistency, "expected_cells": cell_checks,
        "training_arrays_identical": metadata.get("training_arrays_identical"), "transition_arrays_identical": metadata.get("transition_arrays_identical"),
        "all_supports_frozen_before_training": len(supports) == len(bank_ids), "all_treatment_fits_finished_before_evaluation": len([r for r in model_records if r["condition"] == treatment]) == len(bank_ids) * len(seeds),
        "models": model_records, "prior_initial_models": prior_initial, "initialization_consistency": initialization_consistency,
        "snapshot_integrity": snapshot_integrity, "loss_integrity": loss_integrity, "support_integrity": support_integrity, "source_integrity": source_integrity, "count_integrity": count_integrity,
        "baseline_counts_source": f"manifest-verified {control_archive}/pilot_v1 sample_counts.npz and local_sample_counts.npz; no new baseline updates"}
    if _comparison:
        provenance.update(_comparison.provenance())
    artifacts = {key: artifact_path(output / filename) for key, filename in {"protocol": "protocol.json", "banks": "banks.json", "panels": "panels.json", "dataset": "dataset.npz",
        "transitions": "transitions.npz", "dataset_metadata": "dataset_metadata.json", "supports": "supports.npz", "training": "losses.csv", "evaluations": "evaluations.csv",
        "references": "references.csv", "sampling": "sampling.json", "sample_counts": "sample_counts.npz", "local_sample_counts": "local_sample_counts.npz", "map_counts": "map_counts.npz",
        "exposure": "exposure.json", "models": "models.json", "snapshots": "snapshots.json", "paired_differences": "paired_differences.json", "paired_layouts": "paired_layouts.csv", "provenance": "provenance.json", "manifest": "manifest.json"}.items()}
    artifacts.update(directory=artifact_path(output), protocol_document=artifact_path(protocol_file), report=f"docs/experiments/{module.split('.')[-1]}_results_v1.md")
    result = {"schema_version": 1, "protocol": protocol, "run": {"id": output.name, "status": status, "stop_reason": reason,
        "interpretation": "smoke_or_deviation_descriptive_only" if smoke or deviations else "map_replay_descriptive_only" if status == "complete" else "incomplete_not_evidence",
        "train_updates": total_updates, "training_examples": sum(int(s.counts.sum()) for s in samplers.values()), "baseline_new_updates": 0, "collection_episodes": 0, "collection_steps": 0, "support_draws": 0,
        "historical_baseline_updates": sum(s["updates"] for b in samples.values() for s in b.get(control, {}).values()),
        "learner_episodes": len(rows), "reference_episodes": len(refs), "unique_reference_episodes": len({r["reference_sample_id"] for r in refs}),
        "banks": len(banks), "fits": len([r for r in model_records if r["condition"] == treatment]), "frozen_baselines": len([r for r in model_records if r["condition"] == control]),
        "model_parameter_count": 20420, "panels": len(selection["panels"]), "wall_seconds": time.monotonic() - started, **{k + "_wall_seconds": v for k, v in timings.items()},
        "aggregation_wall_seconds": time.monotonic() - aggregation_start, "fit_wall_seconds": fit_timings, "peak_rss_bytes": peak_rss_bytes(), "resource_checks": checks,
        "resource_limits": {"seconds": max_seconds, "peak_process_rss_bytes": max_rss_bytes, "torch_threads": 1}, "progress": progress,
        "limitations": ["Same three archived collected banks; no new collection or independent bank draws.", "The same learner initializations are crossed across banks.",
            "Equal-map replay changes both map exposure and within-batch map diversity; missing states remain missing.", "Archived controls are reevaluated on new panels; their historical training counts are separate from this run's work.",
            "Counterfactual all-action supervision and full-bank detached successor queries remain privileged."]},
        "banks": banks, "panels": selection["panels"], "aggregate": aggregate, "seed_results": seed_results, "references": references, "paired_differences": paired, "pooled": pooled,
        "robustness": robustness, "descriptive_thresholds": thresholds, "loss_aggregate": loss_aggregate, "sampling": samples, "exposure": exposure, "provenance": provenance, "trajectories": trajectories, "artifacts": artifacts}
    if _comparison:
        _comparison.configure_result(result)
    for name, value in (("protocol", protocol), ("banks", banks), ("panels", selection), ("sampling", samples), ("exposure", exposure), ("models", model_records),
        ("snapshots", snapshot_records), ("provenance", provenance), ("paired_differences", paired), ("trajectories", trajectories), ("results", result)):
        write_json(output / f"{name}.json", value)
    with (output / "paired_layouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["bank_id", *PAIR_FIELDS])
        writer.writeheader()
        writer.writerows(paired["per_layout"])
    save_manifest(output, status)
    if dashboard:
        write_json(Path(dashboard), result)
    return result


def checked_input(directory, manifest, filename, evidence):
    digest = sha(directory / filename)
    evidence[filename] = digest
    if manifest["files"].get(filename) != digest:
        raise ConsistencyError(f"input archive file differs from manifest: {filename}")


def model_boundary(record, model, output, phase):
    return {"online_" + phase: model.parameter_hash(), "target_" + phase: module_hash(model.target),
        "file_" + phase: sha(output / record["saved"]), "source_" + phase: sha(ROOT / record["source"])}


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
    parser.add_argument("--panel-seed-start", type=int, default=1040000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    for name in ARCHIVES:
        parser.add_argument("--" + name.replace("_", "-") + "-dir", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.updates, args.panel_count, args.maps_per_panel, args.panel_seed_start = "0", 24, 2, 2, 1120000
    archives = {name: getattr(args, name + "_dir") for name in ARCHIVES if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, bank_ids=[int(v) for v in args.bank_ids.split(",")], seeds=[int(v) for v in args.seeds.split(",")], updates=args.updates,
        panel_count=args.panel_count, maps_per_panel=args.maps_per_panel, panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
