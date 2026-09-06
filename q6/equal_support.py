"""Equal-size collected versus uniformly drawn support with fixed DDQN learning."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import coverage as shared
from .adaptation import write_json
from .competence import ROOT, artifact_path
from .fixed_targets import ConsistencyError, array_metadata, select_fixed_layouts, sha

CONDITIONS = ("collected_unique", "uniform_subset")


def uniform_support(state_count, size):
    if not 64 <= size <= state_count:
        raise ConsistencyError("support size must fit the bank and 64-state batch")
    rng = np.random.default_rng(np.random.SeedSequence([88301]))
    return np.sort(rng.choice(state_count, size=size, replace=False)).astype(np.int32)


def support_metrics(data, transitions, support):
    membership = np.zeros(len(data["observations"]), np.uint32)
    membership[support] = 1
    metrics = shared.coverage_metrics(data, transitions, membership)
    # Binary membership is not a record of actual collector visits.
    for row in [metrics["overall"], *metrics["by_map"], *metrics["by_time_bucket"]]:
        row.pop("visits")
    edges = metrics["successor_queries"]
    edges["denominator"] = "all four nonterminal transitions from unique supported current states, before training"
    edges["terminal_transitions"] = edges["all_action_transitions"] - edges["nonterminal_transitions"]
    metrics["semantics"] = "support membership; visited_* field names denote inclusion, not new trajectory visits"
    return metrics


def final_support_diagnostics(output, data, supports, seeds):
    """Use saved dense predictions; no additional inference or policy rollouts."""
    if "train" not in data:
        return None
    targets, winnable = data["train"]["targets"], data["train"]["winnable"]
    result = {"classification": "declared descriptive diagnostic; not a gate",
        "scope": "final saved dense predictions on each arm's own current-state support and its full-training-bank complement",
        "agreement_denominator": "winnable states only; exact action regret <=1e-6, rtol0",
        "error_denominator": "all four actions and all states, including deadline-impossible states",
        "pooled_denominator": "shared states repeated across learner initializations, not independent task banks",
        "per_seed": [], "pooled": []}
    errors_by_group = {}
    for condition, support in supports.items():
        mask = np.zeros(len(targets), bool)
        mask[support] = True
        for seed in seeds:
            path = output / f"predictions_{condition}_seed{seed}.npz"
            if not path.is_file():
                continue
            with np.load(path) as arrays:
                predicted = arrays["train"]
            error = predicted.astype(np.float64) - targets
            per_state = np.abs(error).mean(axis=1)
            regret = targets.max(axis=1) - targets[np.arange(len(targets)), predicted.argmax(axis=1)]
            for subset, included in (("support", mask), ("outside_support", ~mask)):
                win = included & winnable
                raw = {"condition": condition, "seed": seed, "subset": subset, "states": int(included.sum()),
                    "action_values": 4 * int(included.sum()), "winnable_states": int(win.sum()),
                    "impossible_states": int((included & ~winnable).sum()),
                    "optimal_winnable_actions": int((win & (regret <= 1e-6)).sum()),
                    "abs_error_sum": float(np.abs(error[included]).sum()),
                    "squared_error_sum": float(np.square(error[included]).sum())}
                errors_by_group.setdefault((condition, subset), []).append(per_state[included])
                raw.update(_derived(raw, per_state[included]))
                result["per_seed"].append(raw)
        for subset in ("support", "outside_support"):
            rows = [r for r in result["per_seed"] if r["condition"] == condition and r["subset"] == subset]
            if not rows:
                continue
            pooled = {"condition": condition, "subset": subset, "seeds": len(rows),
                "unique_states": rows[0]["states"], "unique_winnable_states": rows[0]["winnable_states"],
                **{key: sum(r[key] for r in rows) for key in ("states", "action_values", "winnable_states",
                    "impossible_states", "optimal_winnable_actions", "abs_error_sum", "squared_error_sum")}}
            pooled.update(_derived(pooled, np.concatenate(errors_by_group[(condition, subset)])))
            result["pooled"].append(pooled)
    write_json(output / "support_diagnostics.json", result)
    return result


def _derived(row, errors):
    return {"optimal_action_rate": row["optimal_winnable_actions"] / row["winnable_states"] if row["winnable_states"] else None,
        "mean_abs_q_error": row["abs_error_sum"] / row["action_values"] if row["action_values"] else None,
        "rmse_q_error": float(np.sqrt(row["squared_error_sum"] / row["action_values"])) if row["action_values"] else None,
        "q95_abs_q_error": float(np.quantile(errors, .95)) if len(errors) else None}


class EqualSupportComparison:
    """Support/provenance differences around the existing coverage runner."""
    conditions = CONDITIONS
    module = "q6.equal_support"
    fresh_seed_start = 960000
    replication_condition = archive_condition = "collected_unique"
    replication_key = "collected_replication"
    first_pass_interpretation = "collected_only_fresh_gate_met"
    second_pass_interpretation = "uniform_only_fresh_gate_met"
    limitations = [
        "Both arms retain privileged all-action transitions and full-bank detached successor queries; this is not trajectory-only online RL.",
        "Equal cardinality controls support size but not map/clock composition, graph connectivity or induced sampling distribution.",
        "One archived collected bank and one uniform subset are shared by three learner initializations; these are not independent support replications.",
        "Own-support and outside-support diagnostics do not alter the full-bank or fresh-behavior gates.",
    ]

    def __init__(self, fixed_dir=None):
        self.fixed_dir = Path(fixed_dir) if fixed_dir is not None else ROOT / "experiments/fixed_targets/pilot_v1"

    def configure_protocol(self, protocol):
        self.protocol = protocol
        protocol.update(id="equal-support-v1", question="At equal support size, how do collected states compare with uniformly sampled states?",
            conditions=[{"id": "collected_unique", "label": "Collected unique support"}, {"id": "uniform_subset", "label": "Uniform subset"}])
        protocol["budget"].update(collection_included=False, maximum_collection_episodes=0, maximum_collection_steps=0,
                                   support_preparation_included=True)
        protocol["optimizer"].update(paired_local_batch_schedule=True, paired_global_batch_schedule=False)
        protocol["collection"] = {"new_episodes": 0, "new_steps": 0, "source": "archived coverage-v1 support, unchanged",
            "smoke_source": "synthetic fixture: sorted every-third row on alternate banks; no collection"}
        protocol["support_selection"] = {"uniform_rng": "SeedSequence([88301])", "uniform_draw": "choice(fullN,size=K,replace=False), sorted int32",
            "declared_support_size": 59626, "shared_across_learner_seeds": True,
            "successor_access": "full training bank, detached; outside-support successors never enter current-state support"}

    def command_args(self):
        return ["--fixed-dir", artifact_path(self.fixed_dir)]

    def configure_provenance(self, provenance, output, prior_dir):
        self.fixed_metadata = json.loads((self.fixed_dir / "dataset_metadata.json").read_text())
        provenance["files"]["sample_counts.npz"] = sha(prior_dir / "sample_counts.npz")
        provenance.update(fixed_targets_archive=artifact_path(self.fixed_dir),
            fixed_targets_metadata_sha256=sha(self.fixed_dir / "dataset_metadata.json"),
            archived_collection_files={name: sha(prior_dir / name) for name in
                ("collection.npz", "coverage.json", "collection_steps.csv", "collection_episodes.csv")},
            new_collection_episodes=0, new_collection_steps=0)
        shutil.copy2(self.fixed_dir / "dataset_metadata.json", output / "fixed_targets_dataset_metadata.json")
        shutil.copy2(prior_dir / "collection.npz", output / "prior_collection.npz")
        shutil.copy2(prior_dir / "coverage.json", output / "prior_coverage.json")

    def select_layouts(self, config, train_seeds, count, start, prior_fresh, earlier_fresh, deadline):
        result = select_fixed_layouts(config, train_seeds, count, start,
            prior_fresh + earlier_fresh + self.fixed_metadata["heldout"], deadline)
        coverage_hashes = {r["layout_hash"] for r in prior_fresh}
        fixed_hashes = {r["layout_hash"] for r in self.fixed_metadata["heldout"]}
        for row in result["collision_skips"]:
            if row["reason"] == "previous_supervised_fresh_layout":
                if row["layout_hash"] in coverage_hashes:
                    row["reason"] = "previous_coverage_fresh_layout"
                elif row["layout_hash"] in fixed_hashes:
                    row["reason"] = "previous_fixed_targets_fresh_layout"
        return result

    def split_checks(self, selection, prior_metadata):
        fresh = {r["layout_hash"] for r in selection["heldout"]}
        return {"previous_coverage_fresh_layout_intersections": len(fresh & {r["layout_hash"] for r in prior_metadata["heldout"]}),
            "previous_fixed_targets_fresh_layout_intersections": len(fresh & {r["layout_hash"] for r in self.fixed_metadata["heldout"]})}

    def prepare_supports(self, data, transitions, output, prior_dir, *, smoke, required, enforce):
        enforce()
        n = len(data["observations"])
        with np.load(prior_dir / "collection.npz") as archived:
            original = archived["support_indices"]
            archived_counts = archived["visited_counts"]
        collected = np.arange(0, n, 3, dtype=np.int32) if smoke else original.copy()
        if (collected.ndim != 1 or len(collected) < 64 or np.any(collected < 0) or np.any(collected >= n)
                or np.any(np.diff(collected) <= 0)):
            raise ConsistencyError("configuration failure: support must contain sorted distinct valid rows and fit the 64-state batch")
        if not smoke and (collected.dtype != np.int32 or len(archived_counts) != n or not np.array_equal(collected, np.flatnonzero(archived_counts))):
            raise ConsistencyError("archived collected support identity failed")
        if required and len(collected) != 59626:
            raise ConsistencyError("archived collected support declared size failed")
        supports = {"collected_unique": collected, "uniform_subset": uniform_support(n, len(collected))}
        np.savez_compressed(output / "supports.npz", **supports)
        overlap = int(len(np.intersect1d(*supports.values())))
        union = 2 * len(collected) - overlap
        coverage = {"status": "complete", "new_collection_episodes": 0, "new_collection_steps": 0,
            "support_source": "synthetic alternate-bank smoke fixture" if smoke else "unchanged archived collected support",
            "synthetic_smoke_support": bool(smoke), "total_training_states": n, "support_size": len(collected),
            "collected_support_identical_to_archive": bool(np.array_equal(collected, original)),
            "uniform_rng": "SeedSequence([88301])", "arrays": array_metadata(supports),
            "intersection": {"states": overlap, "union_states": union, "jaccard": overlap / union,
                "overlap_fraction": overlap / len(collected), "fraction_of_collected": overlap / len(collected),
                "fraction_of_uniform": overlap / len(collected)},
            "per_condition": [{"condition": condition, **support_metrics(data, transitions, support)} for condition, support in supports.items()]}
        self.protocol["support_selection"]["actual_support_size"] = len(collected)
        write_json(output / "coverage.json", coverage)
        return coverage, {}, supports

    @staticmethod
    def support_diagnostics(output, data, supports, seeds):
        return final_support_diagnostics(output, data, supports, seeds)

    @staticmethod
    def artifacts(output):
        return {"supports": artifact_path(output / "supports.npz"),
            "prior_collection": artifact_path(output / "prior_collection.npz"),
            "support_diagnostics": artifact_path(output / "support_diagnostics.json"),
            "report": "docs/experiments/equal_support_results_v1.md"}


def run_study(output, protocol_file, *, prior_dir=None, fixed_dir=None, fresh_seed_start=960000, **kwargs):
    prior_dir = Path(prior_dir) if prior_dir is not None else ROOT / "experiments/coverage/pilot_v1"
    return shared.run_study(output, protocol_file, prior_dir=prior_dir, fresh_seed_start=fresh_seed_start,
                            _comparison=EqualSupportComparison(fixed_dir), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--fixed-dir", type=Path)
    parser.add_argument("--supervised-dir", type=Path)
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--train-maps", type=int, default=256)
    parser.add_argument("--fresh-maps", type=int, default=64)
    parser.add_argument("--train-seed-start", type=int, default=300000)
    parser.add_argument("--fresh-seed-start", type=int, default=960000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--collection-episodes-per-map", type=int, default=16,
                        help=argparse.SUPPRESS)  # Shared command format; this study never collects.
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.updates, args.train_maps, args.fresh_maps, args.seeds = 24, 2, 2, "0"
        args.train_seed_start, args.fresh_seed_start = 730000, 991000
    result = run_study(args.output, args.protocol_file, prior_dir=args.prior_dir, fixed_dir=args.fixed_dir,
        supervised_dir=args.supervised_dir, dashboard=args.dashboard, seeds=[int(s) for s in args.seeds.split(",")],
        updates=args.updates, train_maps=args.train_maps, fresh_maps=args.fresh_maps, train_seed_start=args.train_seed_start,
        fresh_seed_start=args.fresh_seed_start, max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes,
        collection_episodes_per_map=args.collection_episodes_per_map, smoke=args.smoke,
        checkpoints=[int(c) for c in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
