"""Verify shipped pilot hashes and reported aggregates without retraining.

This checks internal integrity, not scientific validity or independent replication.
Run from any directory: python /path/to/Q6/scripts/verify_pilot_artifacts.py
"""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def check_hash(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path.relative_to(ROOT)}")


def verify_adaptation(folder):
    protocol = read_json(folder / "protocol.json")
    check_hash(folder / "protocol.md", protocol["protocol_sha256"])
    for name, digest in protocol["source_sha256"].items():
        check_hash(folder / "source" / name, digest)
    result = read_json(folder / "results.json")
    if result["protocol"] != protocol:
        raise ValueError(f"Results/protocol mismatch: {folder.name}")
    groups = defaultdict(list)
    with (folder / "evaluations.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["condition"], row["checkpoint"], row["world"]].append(row)
    reported = {(r["condition"], r["checkpoint"], r["world"]): r
                for r in result["aggregate"]}
    if groups.keys() != reported.keys():
        raise ValueError(f"Missing aggregate groups: {folder.name}")
    for key, rows in groups.items():
        summary = reported[key]
        if summary["episodes"] != len(rows):
            raise ValueError(f"Episode-count mismatch: {folder.name} {key}")
        for source, metric in (("success", "success_rate"), ("steps", "mean_steps"),
                               ("base_return", "mean_return"),
                               ("shaped_return", "mean_shaped_return")):
            mean = math.fsum(float(row[source]) for row in rows) / len(rows)
            if not math.isclose(mean, summary[metric], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Aggregate mismatch: {folder.name} {key} {metric}")
    print(f"{folder.name}: source/protocol hashes and {len(groups)} CSV aggregates verified")


def verify_manifest(folder):
    manifest = read_json(folder / "manifest.json")
    if manifest["algorithm"] != "sha256":
        raise ValueError("Unsupported manifest algorithm")
    for name, digest in manifest["files"].items():
        check_hash(folder / name, digest)
    print(f"{folder.relative_to(ROOT)}: all {len(manifest['files'])} manifest files verified")


def verify_competence(folder):
    protocol = read_json(folder / "protocol.json")
    result = read_json(folder / "results.json")
    check_hash(folder / "protocol.md", protocol["protocol_sha256"])
    for name, digest in protocol["source_sha256"].items():
        check_hash(folder / "source" / name, digest)
    if result["protocol"] != protocol:
        raise ValueError(f"Results/protocol mismatch: {folder.name}")
    groups = defaultdict(list)
    with (folder / "evaluations.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["checkpoint_complete"]):
                key = (row["condition"], int(row["checkpoint"]), row["panel"], row["mode"])
                groups[key].append(row)
    reported = {(r["condition"], r["checkpoint"], r["panel"], r["mode"]): r
                for r in result["aggregate"]}
    if groups.keys() != reported.keys():
        raise ValueError(f"Missing competence groups: {folder.name}")
    for key, rows in groups.items():
        summary = reported[key]
        if summary["episodes"] != len(rows):
            raise ValueError(f"Episode-count mismatch: {key}")
        per_seed = defaultdict(list)
        for row in rows:
            per_seed[row["seed"]].append(int(row["success"]))
        rates = [sum(values) / len(values) for values in per_seed.values()]
        expected = {"success_rate": math.fsum(float(r["success"]) for r in rows) / len(rows),
                    "mean_steps": math.fsum(float(r["steps"]) for r in rows) / len(rows),
                    "seed_success_min": min(rates), "seed_success_max": max(rates),
                    "seeds": len(per_seed),
                    "winnable_steps": sum(int(r["winnable_steps"]) for r in rows)}
        for metric, value in expected.items():
            if not math.isclose(value, summary[metric], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Competence aggregate mismatch: {key} {metric}")
    log = folder / Path(result["artifacts"]["training"]).name
    opener = gzip.open if log.suffix == ".gz" else open
    with opener(log, "rt", newline="") as handle:
        logged_steps = sum(int(row["steps"]) for row in csv.DictReader(handle))
    if logged_steps != result["run"]["train_steps"]:
        raise ValueError("Training transitions do not match the episode log")
    print(f"{folder.name}: {len(groups)} competence CSV aggregates and {logged_steps} training transitions verified")


def verify_supervised(folder):
    """Reconcile exact-supervision reports with their captured inputs and rows."""
    protocol = read_json(folder / "protocol.json")
    result = read_json(folder / "results.json")
    check_hash(folder / "protocol.md", protocol["protocol_sha256"])
    for name, digest in protocol["source_sha256"].items():
        check_hash(folder / "source" / name, digest)
    if result["protocol"] != protocol:
        raise ValueError(f"Supervised results/protocol mismatch: {folder.name}")
    preceding = read_json(ROOT / "experiments/competence/pilot_v1/protocol.json")
    for source in ("q6/learning.py", "q6/world.py"):
        if protocol["source_sha256"][source] != preceding["source_sha256"][source]:
            raise ValueError(f"Same-network/world diagnostic changed {source}")
    previous_initial = read_json(ROOT / "experiments/competence/pilot_v1/results.json")["run"]["initial_policy_hashes"]
    for seed, digest in result["run"]["initial_policy_hashes"].items():
        if seed in previous_initial and digest != previous_initial[seed]:
            raise ValueError(f"Network initialization changed for seed {seed}")
    groups = defaultdict(list)
    with (folder / "evaluations.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["checkpoint_complete"]):
                groups[row["condition"], int(row["checkpoint"]), row["panel"], row["mode"]].append(row)
    reported = {(r["condition"], r["checkpoint"], r["panel"], r["mode"]): r
                for r in result["aggregate"]}
    if groups.keys() != reported.keys():
        raise ValueError(f"Missing supervised aggregate groups: {folder.name}")
    if result["run"]["status"] == "complete":
        expected_groups = {("supervised_q", checkpoint, panel, mode)
                           for checkpoint in protocol["checkpoints"] for panel in ("train", "heldout")
                           for mode in ("greedy", "epsilon_0_1")}
        if set(groups) != expected_groups:
            raise ValueError("Complete supervised run is missing declared checkpoint groups")
        for key, rows in groups.items():
            _, _, panel, mode = key
            map_seeds = protocol["dataset"][f"{panel}_map_seeds"]
            repetitions = 1 if mode == "greedy" else protocol["evaluation"]["epsilon_0_1_repetitions"]
            expected = Counter((seed, map_seed, rep) for seed in protocol["seeds"]
                               for map_seed in map_seeds for rep in range(repetitions))
            actual = Counter((int(r["seed"]), int(r["map_seed"]), int(r["repetition"])) for r in rows)
            if actual != expected:
                raise ValueError(f"Missing or duplicated supervised evaluation cells: {key}")
    for key, rows in groups.items():
        summary = reported[key]
        per_seed = defaultdict(list)
        for row in rows:
            per_seed[int(row["seed"])].append(int(row["success"]))
        rates = [sum(values) / len(values) for values in per_seed.values()]
        expected = {
            "episodes": len(rows), "seeds": len(per_seed),
            "success_rate": math.fsum(float(r["success"]) for r in rows) / len(rows),
            "mean_steps": math.fsum(float(r["steps"]) for r in rows) / len(rows),
            "mean_return": math.fsum(float(r["base_return"]) for r in rows) / len(rows),
            "mean_shaped_return": math.fsum(float(r["shaped_return"]) for r in rows) / len(rows),
            "seed_success_min": min(rates), "seed_success_max": max(rates),
            "winnable_steps": sum(int(r["winnable_steps"]) for r in rows),
        }
        for metric, value in expected.items():
            if not math.isclose(value, summary[metric], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Supervised aggregate mismatch: {key} {metric}")
    import numpy as np

    metadata = read_json(folder / "dataset_metadata.json")
    final_state_rates = {}
    if metadata["status"] == "complete":
        with np.load(folder / "dataset.npz", allow_pickle=False) as data:
            for name, expected in metadata["arrays"].items():
                array = data[name]
                if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
                    raise ValueError(f"Dataset array metadata mismatch: {name}")
                if hashlib.sha256(array.tobytes()).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Dataset array hash mismatch: {name}")
            train_layouts = {r["layout_hash"] for r in metadata["train"]}
            fresh_layouts = {r["layout_hash"] for r in metadata["heldout"]}
            if train_layouts & fresh_layouts or len(fresh_layouts) != len(metadata["heldout"]):
                raise ValueError("Supervised split contains repeated or training fresh layouts")
            # Byte-level observation identity catches position/clock split mistakes.
            observation_hashes = {}
            for panel in ("train", "heldout"):
                observation_hashes[panel] = {hashlib.sha256(row.tobytes()).digest()
                                           for row in data[f"{panel}_observations"]}
            if observation_hashes["train"] & observation_hashes["heldout"]:
                raise ValueError("Supervised train/fresh observation overlap")
            with (folder / "state_metrics.csv").open(newline="") as handle:
                state_rows = list(csv.DictReader(handle))
            last = protocol["budget"]["updates_per_seed"]
            final_errors = defaultdict(list)
            for seed in protocol["seeds"]:
                prediction_file = folder / f"predictions_seed{seed}.npz"
                if not prediction_file.exists():
                    if result["run"]["status"] == "complete":
                        raise ValueError(f"Missing final predictions for seed {seed}")
                    continue
                with np.load(prediction_file, allow_pickle=False) as predictions:
                    for panel in ("train", "heldout"):
                        target, prediction = data[f"{panel}_targets"], predictions[panel]
                        if prediction.shape != target.shape or not np.isfinite(prediction).all():
                            raise ValueError("Invalid final prediction array")
                        error = prediction.astype(np.float64) - target
                        chosen = prediction.argmax(axis=1)
                        regret = np.maximum(0, target.max(axis=1) - target[np.arange(len(target)), chosen])
                        win = data[f"{panel}_winnable"]
                        optimal = regret <= protocol["evaluation"]["tie_atol"]
                        final_state_rates[seed, panel] = float(optimal[win].mean()) if win.any() else None
                        rows = [r for r in state_rows if int(r["seed"]) == seed and int(r["checkpoint"]) == last
                                and r["panel"] == panel and r["time_bucket"] == "all"]
                        expected = {"states": len(target), "action_values": target.size,
                                    "winnable_states": int(win.sum()), "impossible_states": int((~win).sum()),
                                    "optimal_winnable_actions": int((optimal & win).sum()),
                                    "abs_error_sum": float(np.abs(error).sum()),
                                    "squared_error_sum": float(np.square(error).sum()),
                                    "signed_error_sum": float(error.sum()),
                                    "action_regret_sum": float(regret.sum()),
                                    "winnable_regret_sum": float(regret[win].sum()),
                                    "impossible_regret_sum": float(regret[~win].sum())}
                        for metric, value in expected.items():
                            actual = math.fsum(float(r[metric]) for r in rows)
                            if not math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-8):
                                raise ValueError(f"Final state metrics mismatch: {seed} {panel} {metric}")
                        remaining = data[f"{panel}_remaining"]
                        for bucket, low, high in (("all", 1, 32), ("1-8", 1, 8), ("9-16", 9, 16),
                                                  ("17-24", 17, 24), ("25-32", 25, 32)):
                            mask = (remaining >= low) & (remaining <= high)
                            final_errors[panel, bucket].append(np.abs(error[mask]).mean(axis=1))
            for row in result["state_aggregate"]:
                selected = [r for r in state_rows if int(r["checkpoint"]) == row["checkpoint"]
                            and r["panel"] == row["panel"] and r["time_bucket"] == row["time_bucket"]]
                for metric in ("states", "winnable_states", "optimal_winnable_actions", "abs_error_sum",
                               "squared_error_sum", "signed_error_sum", "action_regret_sum"):
                    if not math.isclose(math.fsum(float(r[metric]) for r in selected), row[metric],
                                        rel_tol=1e-10, abs_tol=1e-8):
                        raise ValueError(f"State aggregate mismatch: {row['checkpoint']} {metric}")
                if row["checkpoint"] == last:
                    q95 = np.quantile(np.concatenate(final_errors[row["panel"], row["time_bucket"]]), .95)
                    if not math.isclose(float(q95), row["q95_abs_q_error"], rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError("Final pooled state-error percentile mismatch")
    sampling = read_json(folder / "sampling.json")
    if result["run"]["status"] == "complete":
        if set(sampling) != {str(seed) for seed in protocol["seeds"]}:
            raise ValueError("Completed supervised run is missing training seeds")
        if any(row["updates"] != protocol["budget"]["updates_per_seed"] for row in sampling.values()):
            raise ValueError("Completed supervised run did not reach its declared update budget")
    with np.load(folder / "sample_counts.npz", allow_pickle=False) as counts:
        for seed, row in sampling.items():
            array = counts[f"seed{seed}"]
            if array.shape != (protocol["dataset"]["train_states"],):
                raise ValueError("Supervised sample count array does not cover the training dataset")
            if int(array.sum()) != row["examples_seen"] or row["examples_seen"] != row["updates"] * 64:
                raise ValueError("Supervised sample accounting mismatch")
            if np.count_nonzero(array) != row["unique_states_sampled"]:
                raise ValueError("Supervised unique sample count mismatch")
    if sum(row["updates"] for row in sampling.values()) != result["run"]["train_updates"]:
        raise ValueError("Supervised update accounting mismatch")
    if sum(row["examples_seen"] for row in sampling.values()) != result["run"]["training_examples"]:
        raise ValueError("Supervised training example accounting mismatch")
    eligible = result["run"]["status"] == "complete" and not protocol["smoke"] and not protocol["deviations"]
    if result["gates"]["eligible"] != eligible:
        raise ValueError("Incorrect supervised gate eligibility")
    with (folder / "references.csv").open(newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    random_rows = [r for r in reference_rows if r["policy"] == "random_actions" and r["panel"] == "heldout"]
    for reference in result["references"]:
        selected = [r for r in reference_rows if all(r[key] == reference[key] for key in ("policy", "panel", "mode"))]
        if reference["episodes"] != len(selected):
            raise ValueError("Supervised reference episode-count mismatch")
        rate = sum(int(r["success"]) for r in selected) / len(selected)
        if not math.isclose(rate, reference["success_rate"], abs_tol=1e-12):
            raise ValueError("Supervised reference success mismatch")
    random_rate = sum(int(r["success"]) for r in random_rows) / len(random_rows) if random_rows else None
    if Counter(gate["seed"] for gate in result["gates"]["per_seed"]) != Counter(protocol["seeds"]):
        raise ValueError("Supervised gate rows omit or repeat a declared seed")
    for gate in result["gates"]["per_seed"]:
        seed = gate["seed"]
        scores = {}
        for panel in ("train", "heldout"):
            rows = [r for r in groups.get(("supervised_q", protocol["budget"]["updates_per_seed"], panel, "greedy"), [])
                    if int(r["seed"]) == seed]
            scores[panel] = sum(int(r["success"]) for r in rows) / len(rows) if rows else None
        agreement = final_state_rates.get((seed, "train"))
        for name, expected in (("train_success", scores["train"]), ("fresh_success", scores["heldout"]),
                               ("train_state_optimal", agreement)):
            actual = gate[name]
            if (actual is None) != (expected is None) or (actual is not None and
                    not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)):
                raise ValueError(f"Supervised per-seed reported value mismatch: {seed} {name}")
        fit = bool(eligible and scores["train"] is not None and scores["train"] >= .9
                   and agreement is not None and agreement >= .9)
        fresh = bool(eligible and scores["heldout"] is not None and scores["heldout"] >= .7
                     and random_rate is not None and scores["heldout"] > random_rate)
        if gate["training_fit"] != fit or gate["fresh"] != fresh:
            raise ValueError(f"Supervised seed gate mismatch: {seed}")
    for key in ("training_fit", "fresh"):
        if result["gates"][key] != all(r[key] for r in result["gates"]["per_seed"]):
            raise ValueError(f"Supervised pooled gate mismatch: {key}")
    print(f"{folder.name}: {len(groups)} supervised rollout groups, dataset split, final predictions, exposure and gates verified")


def verify_coverage_support(path):
    """Reconcile the separate post-hoc artifact without changing its inputs."""
    from analyze_coverage_support import analyze

    saved = read_json(path)
    actual = analyze(ROOT / saved["source_directory"])

    def same(left, right):
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(
                same(left[key], right[key]) for key in left)
        if isinstance(left, list):
            return isinstance(right, list) and len(left) == len(right) and all(
                same(a, b) for a, b in zip(left, right))
        if isinstance(left, float):
            return isinstance(right, (int, float)) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        return left == right

    if not same(saved, actual):
        raise ValueError(f"Post-hoc support diagnostic mismatch: {path.relative_to(ROOT)}")
    print(f"{path.relative_to(ROOT)}: input hashes and saved-prediction support diagnostics verified")


def main():
    for version in ("v1", "v2"):
        verify_adaptation(ROOT / "experiments/adaptation" / f"pilot_{version}")
    folder = ROOT / "experiments/provenance/release-pilot-v1"
    verify_manifest(folder)
    for manifest in sorted((ROOT / "experiments/competence").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        if (manifest.parent / "protocol.json").exists():
            verify_competence(manifest.parent)
    for manifest in sorted((ROOT / "experiments/supervised").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        verify_supervised(manifest.parent)
    for manifest in sorted((ROOT / "experiments/fixed_targets").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        # Isolate archived q6 imports from the current checkout. The portable
        # lane retains raw-data/model-hash checks; exact forward reproduction
        # belongs to the separately documented, pinned-runtime audit.
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_fixed_targets_study.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/coverage").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_coverage_study.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for diagnostic in sorted((ROOT / "experiments/coverage").glob("analysis*/support_diagnostic.json")):
        verify_coverage_support(diagnostic)
    for manifest in sorted((ROOT / "experiments/equal_support").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_equal_support_study.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/panel_evaluation").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_panel_evaluation.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/bank_replication").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_bank_replication.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/map_replay").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_map_replay.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/within_map").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_within_map.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/recorded_actions").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_recorded_actions.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/constrained_bootstrap").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_constrained_bootstrap.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/logged_graph").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_logged_graph.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)
    for manifest in sorted((ROOT / "experiments/familiar_starts").glob("*/manifest.json")):
        verify_manifest(manifest.parent)
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_familiar_starts.py"),
                        "--study", str(manifest.parent), "--skip-forward-inference"], check=True)


if __name__ == "__main__":
    main()
