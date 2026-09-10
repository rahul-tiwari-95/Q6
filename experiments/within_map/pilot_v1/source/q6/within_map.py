"""Change within-map state support while exactly preserving collected map quotas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import map_replay as shared
from .coverage import SupportSampler
from .fixed_targets import ConsistencyError, array_metadata

CONDITIONS = ("collected_unique", "within_map_uniform")
COMPARISONS = ({"id": "within_map_minus_collected", "left": CONDITIONS[0], "right": CONDITIONS[1]},)


def quota_support(support, map_seeds, bank_id, enforce=lambda: None, *, on_draw=None):
    """One owned uniform draw per map; no clock, target or outcome selection."""
    support = np.asarray(support, np.int32)
    if not len(support) or np.any(np.diff(support) <= 0) or support[0] < 0 or support[-1] >= len(map_seeds):
        raise ConsistencyError("quota support requires distinct sorted valid collected rows")
    if np.any(np.diff(map_seeds) < 0):
        raise ConsistencyError("exhaustive rows must form ascending contiguous map blocks")
    pieces = []
    for map_seed in np.unique(map_seeds):
        enforce()
        candidates = np.flatnonzero(map_seeds == map_seed)
        size = int(np.count_nonzero(map_seeds[support] == map_seed))
        if size == 0:
            pieces.append(np.empty(0, np.int64))
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(bank_id), int(map_seed), 105301]))
        pieces.append(candidates[rng.choice(len(candidates), size=size, replace=False)])
        if on_draw:
            on_draw()
    result = np.sort(np.concatenate(pieces)).astype(np.int32)
    result.flags.writeable = False
    return result


class QuotaSampler:
    """Unchanged SupportSampler, augmented only with ordered map evidence."""
    def __init__(self, support, map_seeds, seed, batch_size=64):
        try:
            self.base = SupportSampler(support, len(map_seeds), seed, batch_size)
        except ValueError as exc:
            raise ConsistencyError(f"invalid distinct-state replay configuration: {exc}") from exc
        self.support, self.counts, self.digest = self.base.support, self.base.counts, self.base.digest
        self.map_seeds, self.map_ids = map_seeds, np.unique(map_seeds)
        self.map_counts = np.zeros(len(self.map_ids), np.uint32)
        self.map_digest = hashlib.sha256()

    @property
    def updates(self):
        return self.base.updates

    @property
    def local_counts(self):
        return self.base.local.counts

    @property
    def local_digest(self):
        return self.base.local.digest

    def next_batch(self):
        rows = self.base.next_batch()
        maps = self.map_seeds[rows]
        self.map_digest.update(maps.astype("<i8").tobytes())
        np.add.at(self.map_counts, np.searchsorted(self.map_ids, maps), 1)
        return rows


def sampling_metadata(seed, sampler):
    return {"updates": sampler.updates, "new_updates": sampler.updates, "historical": False,
        "examples_seen": int(sampler.counts.sum()), "unique_states_sampled": int(np.count_nonzero(sampler.counts)), "support_states": len(sampler.support),
        "local_batch_index_sha256": sampler.local_digest.hexdigest(), "global_batch_index_sha256": sampler.digest.hexdigest(),
        "map_index_sha256": sampler.map_digest.hexdigest(), "rng_seed_tuple": [seed, 66301], "map_ids": sampler.map_ids.tolist(),
        "outside_support_direct_samples": int(sampler.counts.sum() - sampler.counts[sampler.support].sum())}


class WithinMapComparison:
    conditions, comparisons = CONDITIONS, COMPARISONS
    module, panel_seed_start, extra_archives = "q6.within_map", 1060000, ("map_replay",)
    make_sampler = staticmethod(QuotaSampler)
    sampling_metadata = staticmethod(sampling_metadata)

    def __init__(self):
        self.prefixes, self.reconstruction, self.draws = {}, [], 0

    def configure_protocol(self, protocol):
        protocol.update(id="within-map-v1", question="Does within-map uniform state support improve efficiency when map quotas and replay exposure are fixed?",
            conditions=[{"id": "collected_unique", "label": "Collected unique (frozen control)"}, {"id": "within_map_uniform", "label": "Within-map uniform support"}])
        protocol["collection"].update(new_support_draws=len(protocol["bank_ids"]) * 256, per_map_subset_draws=len(protocol["bank_ids"]) * 256, replacement_supports=len(protocol["bank_ids"]))
        protocol["support_selection"] = {"rng": "SeedSequence([bank_id,map_seed,105301])", "draw": "one choice(sorted global rows for map,size=collected quota,replace=False) per map; globally sorted int32", "same_map_quotas": True, "shared_across_learner_seeds": True, "clock_or_target_selection": False}
        protocol["sampling"] = {"rng": "SeedSequence([seed,66301])", "implementation": "unchanged coverage.SupportSampler", "batch_size": 64,
            "draw": "64 distinct local positions from ascending support; original map blocks retained", "paired_map_schedule_within_bank": True,
            "paired_local_schedule_within_bank": True, "full_bank_detached_successors": True,
            "baseline_reconstruction": "full30000 original draws checked; treatment matched to same-budget prefix (smoke24 only)"}

    def prepare_support(self, bank, support, data, enforce):
        def counted():
            self.draws += 1
        treatment = quota_support(support, data["map_seeds"], bank, enforce, on_draw=counted)
        return {"collected_unique": support, "within_map_uniform": treatment}

    def bank_metadata(self, bank, pair, data):
        left, right = pair["collected_unique"], pair["within_map_uniform"]
        quotas = [{"map_seed": int(m), "collected_states": int(np.count_nonzero(data["map_seeds"][left] == m)),
            "treatment_states": int(np.count_nonzero(data["map_seeds"][right] == m))} for m in np.unique(data["map_seeds"])]
        for row in quotas:
            candidates = np.flatnonzero(data["map_seeds"] == row["map_seed"])
            selected = right[data["map_seeds"][right] == row["map_seed"]]
            row.update(candidate_states=len(candidates), rng_seed_tuple=[bank, row["map_seed"], 105301],
                candidate_row_sha256=hashlib.sha256(candidates.astype("<i8").tobytes()).hexdigest(),
                selected_row_sha256=hashlib.sha256(selected.astype("<i4").tobytes()).hexdigest())
            row["identical"] = row["collected_states"] == row["treatment_states"]
        if not all(r["identical"] for r in quotas) or not np.array_equal(data["map_seeds"][left], data["map_seeds"][right]):
            raise ConsistencyError("within-map quotas or sorted map blocks differ")
        overlap = len(np.intersect1d(left, right))
        return {"quotas": {"exact": True, "by_map": quotas}, "uniform_rng": f"SeedSequence([{bank},map_seed,105301])",
            "intersection": {"states": overlap, "union_states": len(left) + len(right) - overlap, "fraction_of_each": overlap / len(left), "jaccard": overlap / (len(left) + len(right) - overlap)}}

    def record_baseline(self, bank, seed, support, data, global_counts, local_counts, archived, enforce, updates):
        if not 1 <= updates <= 30000 or archived["updates"] != 30000:
            raise ConsistencyError("treatment must fit archived30000-update baseline stream")
        sampler = QuotaSampler(support, data["map_seeds"], seed)
        prefix = None
        for _ in range(30000):
            enforce()
            sampler.next_batch()
            if sampler.updates == updates:
                prefix = {"updates": updates, "local_digest": sampler.local_digest.hexdigest(), "global_digest": sampler.digest.hexdigest(),
                    "map_digest": sampler.map_digest.hexdigest(), "local_counts": sampler.local_counts.copy(), "map_counts": sampler.map_counts.copy()}
        record = {"bank_id": bank, "seed": seed, "reconstructed_updates": sampler.updates,
            "local_digest_identical": sampler.local_digest.hexdigest() == archived["local_batch_index_sha256"],
            "global_digest_identical": sampler.digest.hexdigest() == archived["global_batch_index_sha256"],
            "local_counts_identical": np.array_equal(sampler.local_counts, local_counts), "global_counts_identical": np.array_equal(sampler.counts, global_counts),
            "map_index_sha256": sampler.map_digest.hexdigest(), "count_arrays": array_metadata({"global": sampler.counts, "local": sampler.local_counts, "map": sampler.map_counts})}
        record["verified"] = all(record[k] for k in ("local_digest_identical", "global_digest_identical", "local_counts_identical", "global_counts_identical"))
        self.reconstruction.append(record)
        if not record["verified"]:
            raise ConsistencyError("reconstructed original sampling stream differs from archived control")
        self.prefixes[(bank, seed)] = prefix
        archived.update(map_index_sha256=sampler.map_digest.hexdigest(), map_ids=sampler.map_ids.tolist(),
            compared_prefix={"updates": updates, "local_batch_index_sha256": prefix["local_digest"], "global_batch_index_sha256": prefix["global_digest"], "map_index_sha256": prefix["map_digest"],
                "count_arrays": array_metadata({"local": prefix["local_counts"], "map": prefix["map_counts"]})})

    def consistency(self, samplers, bank_ids, seeds, updates):
        rows = []
        for bank in bank_ids:
            for seed in seeds:
                sampler, prefix = samplers.get((bank, "within_map_uniform", seed)), self.prefixes.get((bank, seed))
                row = {"bank_id": bank, "seed": seed, "compared_updates": updates, "baseline_updates": 30000,
                    "map_digest_identical": bool(sampler and prefix and sampler.map_digest.hexdigest() == prefix["map_digest"]),
                    "map_counts_identical": bool(sampler and prefix and np.array_equal(sampler.map_counts, prefix["map_counts"])),
                    "local_digest_identical": bool(sampler and prefix and sampler.local_digest.hexdigest() == prefix["local_digest"]),
                    "local_counts_identical": bool(sampler and prefix and np.array_equal(sampler.local_counts, prefix["local_counts"]))}
                row["complete"] = bool(sampler and prefix and sampler.updates == updates) and all(row[k] for k in ("map_digest_identical", "map_counts_identical", "local_digest_identical", "local_counts_identical"))
                rows.append(row)
        return rows

    def provenance(self):
        return {"baseline_sampling_reconstruction": self.reconstruction,
            "map_quota_claim": "same per-map support counts and ordered map stream within each bank/seed; clock/position/goal composition may differ"}

    def configure_result(self, result):
        result["run"]["support_draws"] = self.draws
        result["run"]["per_map_subset_draws"] = self.draws
        result["run"]["replacement_supports"] = len(result["banks"])
        result["run"]["baseline_reconstructed_updates"] = sum(r["reconstructed_updates"] for r in self.reconstruction)
        result["run"]["limitations"] = ["Three quota-matched replacement supports reuse previously collected banks and training maps; no new trajectories.",
            "Local replay slots and map exposure match within each bank/seed; clock, position, goal-distance and successor structure may change together.",
            "Archived controls are reevaluated on new panels; reconstruction checks their sampling stream without refitting them.",
            "Shared learner initializations and panels do not increase the number of bank draws.", "All-action supervision and detached full-bank successors remain privileged; no online competence claim."]
        result["robustness"]["classification"] = "descriptive within-map support comparison on three quota-matched bank pairs; no new gates or significance claims"
        if result["run"]["interpretation"] == "map_replay_descriptive_only":
            result["run"]["interpretation"] = "within_map_descriptive_only"


def run_study(output, protocol_file, **kwargs):
    kwargs.setdefault("panel_seed_start", 1060000)
    return shared.run_study(output, protocol_file, _comparison=WithinMapComparison(), **kwargs)


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
    parser.add_argument("--panel-seed-start", type=int, default=1060000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    for name in shared.ARCHIVES + ("map_replay",):
        parser.add_argument("--" + name.replace("_", "-") + "-dir", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.updates, args.panel_count, args.maps_per_panel, args.panel_seed_start = "0", 24, 2, 2, 1140000
    archives = {name: getattr(args, name + "_dir") for name in shared.ARCHIVES + ("map_replay",) if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, bank_ids=[int(v) for v in args.bank_ids.split(",")], seeds=[int(v) for v in args.seeds.split(",")], updates=args.updates,
        panel_count=args.panel_count, maps_per_panel=args.maps_per_panel, panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
