#!/usr/bin/env python3
"""Post-hoc support diagnostics from saved arrays; no inference or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("exhaustive", "collected_unique")
ADDITIVE = ("states", "action_values", "winnable_states", "impossible_states",
            "optimal_winnable_actions", "abs_error_sum")


def analyze(study: Path) -> dict:
    """Arithmetic over immutable final predictions and their archived labels."""
    saved = json.loads((study / "results.json").read_text())
    if saved["run"]["status"] != "complete":
        raise ValueError("support analysis requires a complete saved study")
    seeds = saved["protocol"]["seeds"]
    with np.load(study / "dataset.npz") as data:
        targets, winnable = data["train_targets"], data["train_winnable"]
    with np.load(study / "collection.npz") as data:
        visited = data["visited_counts"] > 0
    if targets.shape != (len(visited), 4) or winnable.shape != visited.shape:
        raise ValueError("dataset and support dimensions disagree")
    if not visited.any() or visited.all():
        raise ValueError("both visited and unvisited states are needed")
    names = ["results.json", "dataset.npz", "collection.npz"] + [
        f"predictions_{condition}_seed{seed}.npz" for condition in CONDITIONS for seed in seeds]
    result = {
        "classification": "post-hoc diagnostic; not a declared gate or a new evaluation",
        "method": "Arithmetic over saved final dense predictions and archived exact targets, with no inference, optimizer updates, policy rollouts or new data collection.",
        "agreement_definition": "Lowest-label prediction argmax has exact Q regret <=1e-6 among winnable states; rtol0.",
        "mae_definition": "Mean absolute prediction-minus-exact error across all four action outputs; impossible states retained.",
        "pooled_denominator": "Same states evaluated by three initializations; not three independent state banks." if len(seeds) == 3 else
            f"Same states evaluated by {len(seeds)} initializations; these are not independent state banks.",
        "source_directory": Path(os.path.relpath(study, ROOT)).as_posix(),
        "inputs": {name: hashlib.sha256((study / name).read_bytes()).hexdigest() for name in names},
        "per_seed": [], "pooled": [],
    }
    for condition in CONDITIONS:
        for seed in seeds:
            with np.load(study / f"predictions_{condition}_seed{seed}.npz") as data:
                predicted = data["train"]
            if predicted.shape != targets.shape or not np.isfinite(predicted).all():
                raise ValueError("saved prediction shape or values are invalid")
            chosen = predicted.argmax(1)
            regret = targets.max(1) - targets[np.arange(len(targets)), chosen]
            errors = np.abs(predicted.astype(np.float64) - targets)
            for label, mask in (("visited", visited), ("unvisited", ~visited)):
                win = mask & winnable
                row = {"condition": condition, "seed": seed, "support": label,
                    "states": int(mask.sum()), "action_values": int(mask.sum()) * 4,
                    "winnable_states": int(win.sum()), "impossible_states": int((mask & ~winnable).sum()),
                    "optimal_winnable_actions": int((win & (regret <= 1e-6)).sum()),
                    "abs_error_sum": float(errors[mask].sum())}
                row.update(optimal_action_rate=row["optimal_winnable_actions"] / row["winnable_states"] if row["winnable_states"] else None,
                           mean_abs_q_error=row["abs_error_sum"] / row["action_values"])
                result["per_seed"].append(row)
        for label in ("visited", "unvisited"):
            rows = [r for r in result["per_seed"] if r["condition"] == condition and r["support"] == label]
            pooled = {"condition": condition, "support": label, "seeds": len(seeds),
                      **{key: sum(r[key] for r in rows) for key in ADDITIVE}}
            pooled.update(unique_states=rows[0]["states"], unique_winnable_states=rows[0]["winnable_states"],
                optimal_action_rate=pooled["optimal_winnable_actions"] / pooled["winnable_states"] if pooled["winnable_states"] else None,
                mean_abs_q_error=pooled["abs_error_sum"] / pooled["action_values"])
            result["pooled"].append(pooled)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/coverage/pilot_v1"))
    parser.add_argument("--output", type=Path, default=Path("experiments/coverage/analysis_v1/support_diagnostic.json"))
    args = parser.parse_args()
    study = (ROOT / args.study).resolve()
    output = (ROOT / args.output).resolve()
    if output.is_relative_to(study):
        parser.error("supplemental output must be outside the immutable study directory")
    result = analyze(study)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved post-hoc support diagnostic: {os.path.relpath(output, ROOT)}")


if __name__ == "__main__":
    main()
