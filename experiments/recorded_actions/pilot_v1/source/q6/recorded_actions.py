"""Learn only from recorded action outcomes on unchanged collected state supports."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import map_replay as shared
from .adaptation import write_json
from .competence import artifact_path
from .fixed_targets import ConsistencyError, array_metadata
from .within_map import WithinMapComparison, QuotaSampler

CONDITIONS = ("collected_unique", "recorded_actions")
COMPARISONS = ({"id": "recorded_minus_collected", "left": CONDITIONS[0], "right": CONDITIONS[1]},)


def build_recorded_table(records, data, support, enforce=lambda: None):
    """Deduplicate logged edges, retaining frequency only as a diagnostic."""
    if isinstance(records, (str, Path)):
        with Path(records).open(newline="") as handle:
            return build_recorded_table(csv.DictReader(handle), data, support, enforce)
    size = len(data["observations"])
    support = np.asarray(support, np.int32)
    if not len(support) or np.any(np.diff(support) <= 0) or support[0] < 0 or support[-1] >= size:
        raise ConsistencyError("recorded support must contain distinct sorted valid rows")
    included = np.zeros(size, bool)
    included[support] = True
    arrays = {"observed": np.zeros((size, 4), bool), "rewards": np.full((size, 4), np.nan, np.float64),
        "ends": np.ones((size, 4), bool), "terminated": np.zeros((size, 4), bool), "truncated": np.zeros((size, 4), bool), "successor_indices": np.full((size, 4), -1, np.int32), "occurrences": np.zeros((size, 4), np.uint32)}
    total = 0
    for record in records:
        if total % 256 == 0:
            enforce()
        try:
            row, action = int(record["current_row"]), int(record["action"])
            reward, successor = float(record["reward"]), int(record["next_row"])
            terminated, truncated = int(record["terminated"]), int(record["truncated"])
            map_seed = int(record["map_seed"]) if "map_seed" in record else None
            remaining = int(record["remaining"]) if "remaining" in record else None
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ConsistencyError(f"malformed recorded outcome fields: {exc}") from exc
        if not 0 <= row < size or not included[row] or not 0 <= action < 4 or not np.isfinite(reward) or terminated not in (0, 1) or truncated not in (0, 1):
            raise ConsistencyError("invalid recorded state/action/reward/end flag")
        if terminated + truncated > 1 or (truncated and int(data["remaining"][row]) != 1):
            raise ConsistencyError("invalid termination/truncation combination or timeout clock")
        ended = bool(terminated or truncated)
        if (ended and successor != -1) or (not ended and not 0 <= successor < size):
            raise ConsistencyError("recorded terminal/successor index mismatch")
        if map_seed is not None and map_seed != int(data["map_seeds"][row]):
            raise ConsistencyError("recorded source map differs from row metadata")
        if remaining is not None and remaining != int(data["remaining"][row]):
            raise ConsistencyError("recorded source clock differs from row metadata")
        if not ended and (data["map_seeds"][row] != data["map_seeds"][successor] or data["remaining"][row] - 1 != data["remaining"][successor]):
            raise ConsistencyError("recorded successor crosses map or fails clock decrement")
        if arrays["observed"][row, action] and (arrays["rewards"][row, action] != reward or arrays["ends"][row, action] != ended or arrays["terminated"][row, action] != bool(terminated) or arrays["truncated"][row, action] != bool(truncated) or arrays["successor_indices"][row, action] != successor):
            raise ConsistencyError("duplicate recorded state/action outcomes disagree")
        arrays["observed"][row, action] = True
        arrays["rewards"][row, action] = reward
        arrays["ends"][row, action] = ended
        arrays["terminated"][row, action] = bool(terminated)
        arrays["truncated"][row, action] = bool(truncated)
        arrays["successor_indices"][row, action] = successor
        arrays["occurrences"][row, action] += 1
        total += 1
    mask = arrays["observed"]
    if not np.array_equal(np.flatnonzero(mask.any(1)), support):
        raise ConsistencyError("every supported state must have at least one recorded action")
    live = mask & ~arrays["ends"]
    destinations = arrays["successor_indices"][live]
    outside = int(np.count_nonzero(~included[destinations]))
    summary = {"supported_states": len(support), "recorded_edges": int(mask.sum()), "all_action_edges": 4 * len(support),
        "observed_action_fraction": float(mask.sum() / (4 * len(support))), "collector_steps": total, "duplicate_records": total - int(mask.sum()),
        "states_by_observed_actions": [{"actions": n, "states": int(np.count_nonzero(mask[support].sum(1) == n))} for n in (1, 2, 3, 4)],
        "nonterminal_recorded_edges": int(live.sum()), "terminal_recorded_edges": int((mask & arrays["ends"]).sum()),
        "nonterminal_successors_in_support": int(live.sum()) - outside, "nonterminal_successors_outside_support": outside,
        "all_nonterminal_successors_in_support_verified": outside == 0,
        "distinct_nonterminal_successor_states": len(np.unique(destinations)),
        "unobserved_outcomes": "NaN reward, ended=true, next=-1 are absent-entry sentinels; observed mask is authoritative",
        "arrays": array_metadata(arrays)}
    for array in arrays.values():
        array.flags.writeable = False
    return arrays, summary


def recorded_update(agent, observations, recorded, indices):
    """Gather observed edges before outcome math; average actions within each state."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    observed = recorded["observed"][indices]
    action_counts = observed.sum(1)
    if not bool(torch.all(action_counts > 0)):
        raise ConsistencyError("sampled state has no recorded action")
    batch_rows, actions = observed.nonzero(as_tuple=True)
    source_rows = indices[batch_rows]
    # No absent reward, end flag or successor enters this computation.
    rewards = recorded["rewards"][source_rows, actions]
    ended = recorded["ends"][source_rows, actions]
    successors = recorded["successor_indices"][source_rows, actions]
    with torch.no_grad():
        targets = rewards.clone()
        live = ~ended
        if bool(live.any()):
            next_observations = observations[successors[live].long()]
            next_actions = agent.online(next_observations).argmax(1, keepdim=True)
            targets[live] += agent.gamma * agent.target(next_observations).gather(1, next_actions).squeeze(1)
    predicted = agent.online(observations[indices])[batch_rows, actions]
    edge_losses = F.smooth_l1_loss(predicted, targets, reduction="none", beta=1.)
    state_losses = torch.zeros(len(indices), dtype=edge_losses.dtype).index_add(0, batch_rows, edge_losses)
    loss = (state_losses / action_counts).mean()
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 5.0)
    agent.optimizer.step()
    with torch.no_grad():
        for target, online in zip(agent.target.parameters(), agent.online.parameters()):
            target.lerp_(online, 0.01)
    return float(loss.item())


class RecordedActionsComparison(WithinMapComparison):
    conditions, comparisons = CONDITIONS, COMPARISONS
    module, panel_seed_start, extra_archives = "q6.recorded_actions", 1080000, ("map_replay", "within_map")

    def __init__(self):
        super().__init__()
        self.tables, self.tensors, self.table_summaries, self.original_supports = {}, {}, [], {}
        self.tensor_before, self.optimizer_counts, self.query_counts = {}, {}, {}
        self.action_exposure = {"per_seed": [], "integrity": []}
        self.table_integrity = []

    def configure_protocol(self, protocol):
        self.bank_ids = protocol["bank_ids"]
        protocol.update(id="recorded-actions-v1", question="What changes when supervision uses only recorded actions on the same states and replay stream?",
            conditions=[{"id": "collected_unique", "label": "All four outcomes (frozen control)"}, {"id": "recorded_actions", "label": "Recorded action outcomes"}])
        protocol["optimizer"].update(implementation="recorded_actions.recorded_update; DDQN target, optimizer and target update unchanged",
            loss="mean over states of mean SmoothL1 beta1 over each state's distinct recorded actions", observed_only=True)
        protocol["sampling"] = {"rng": "SeedSequence([seed,66301])", "implementation": "unchanged coverage.SupportSampler", "batch_size": 64,
            "paired_local_global_map_schedule_within_bank": True, "baseline_reconstruction": "full30000 draws checked; same-budget prefix used in smoke",
            "target_access": "recorded rewards, ends and successors only; all4 model predictions at recorded successors allowed"}
        protocol["recorded_actions"] = {"deduplication": "state/action repeats must agree on reward, terminated, truncated and successor; no frequency weighting",
            "absent_entries": "observed=false,reward=NaN,ends=true,terminated=false,truncated=false,next=-1,occurrences=0",
            "equal_action_target_budget": False, "all_tables_frozen_before_training": True}

    def prepare_inputs(self, directories, manifests, archives_meta, data, transitions, output, enforce):
        self.output = output
        directory, manifest = directories["bank_replication"], manifests["bank_replication"]
        with np.load(directory / "supports.npz") as saved:
            for bank in self.bank_ids:
                enforce()
                filename = f"banks/bank{bank}/collection_steps.csv"
                shared.checked_input(directory, manifest, filename, archives_meta["bank_replication"]["files"])
                destination = output / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(directory / filename, destination)
                support = saved[f"collected_unique_bank{bank}"]
                table, summary = build_recorded_table(destination, data, support, enforce)
                mask = table["observed"]
                if not all(np.array_equal(table[k][mask], transitions[k][mask]) for k in ("rewards", "ends", "successor_indices")):
                    raise ConsistencyError("logged outcome disagrees with archived transition integrity reference")
                if not summary["all_nonterminal_successors_in_support_verified"]:
                    raise ConsistencyError("complete collector logs have a nonterminal successor missing from original support")
                support.flags.writeable = False
                self.tables[bank], self.original_supports[bank] = table, support
                self.tensors[bank] = {k: torch.from_numpy(v.astype(np.float32) if k == "rewards" else v.copy()) for k, v in table.items()}
                self.tensor_before[bank] = array_metadata({k: v.numpy() for k, v in self.tensors[bank].items()})
                self.table_summaries.append({"bank_id": bank, **summary, "source": artifact_path(directory / filename),
                    "source_sha256": shared.sha(directory / filename), "copied": filename, "copied_sha256": shared.sha(destination),
                    "observed_outcomes_match_integrity_reference": True, "optimizer_inputs": "observations, recorded tensors, sampled state indices; no exact or full-transition argument"})
                np.savez_compressed(output / "recorded_transitions.npz", **{f"bank{b}_{k}": v for b, t in self.tables.items() for k, v in t.items()})
                write_json(output / "action_coverage.json", {"per_bank": self.table_summaries})

    def prepare_support(self, bank, support, data, enforce):
        if not np.array_equal(support, self.original_supports[bank]):
            raise ConsistencyError("recorded action table support differs from original control")
        return {condition: support for condition in CONDITIONS}

    def bank_metadata(self, bank, pair, data):
        return {"recorded_actions": next(r for r in self.table_summaries if r["bank_id"] == bank), "identical_state_support": True}

    def update(self, agent, observations, indices, bank, seed):
        table = self.tables[bank]
        observed = table["observed"][indices]
        live = observed & ~table["ends"][indices]
        targets, queries = int(observed.sum()), int(live.sum())
        key = bank, seed
        if key not in self.optimizer_counts:
            self.optimizer_counts[key] = {"updates": 0, "state_presentations": 0, "action_target_presentations": 0, "nonterminal_target_queries": 0}
            self.query_counts[key] = np.zeros(len(observations), np.uint32)
        counts = self.optimizer_counts[key]
        counts["updates"] += 1
        counts["state_presentations"] += len(indices)
        counts["action_target_presentations"] += targets
        counts["nonterminal_target_queries"] += queries
        np.add.at(self.query_counts[key], table["successor_indices"][indices][live], 1)
        return recorded_update(agent, observations, self.tensors[bank], indices)

    def consistency(self, samplers, bank_ids, seeds, updates):
        # Reuse the quota study's tested local/map checks under its expected key.
        translated = {(b, "within_map_uniform", s): sampler for (b, _, s), sampler in samplers.items()}
        rows = super().consistency(translated, bank_ids, seeds, updates)
        for row in rows:
            bank, seed = row["bank_id"], row["seed"]
            sampler, prefix = samplers.get((bank, "recorded_actions", seed)), self.prefixes.get((bank, seed))
            row["global_digest_identical"] = bool(sampler and prefix and sampler.digest.hexdigest() == prefix["global_digest"])
            row["global_counts_identical"] = bool(sampler and prefix and np.array_equal(sampler.counts[sampler.support], prefix["local_counts"]) and int(sampler.counts.sum()) == int(prefix["local_counts"].sum()))
            row["complete"] = row["complete"] and row["global_digest_identical"] and row["global_counts_identical"]
        return rows

    def finalize_exposure(self, data, transitions, supports, counts, samples, exposure, output):
        self.output = output
        records, checks = [], []
        for (bank, condition, seed), count in counts.items():
            support = supports[bank][condition]
            membership = np.zeros(len(count), bool)
            membership[support] = True
            table = self.tables[bank]
            if condition == "recorded_actions":
                mask, ended, successor = table["observed"], table["ends"], table["successor_indices"]
            else:
                mask, ended, successor = np.ones_like(transitions["ends"]), transitions["ends"], transitions["successor_indices"]
            live = mask & ~ended
            outside = live & ~membership[np.maximum(successor, 0)]
            target_count = int(np.dot(count.astype(np.uint64), mask.sum(1).astype(np.uint64)))
            query_count = int(np.dot(count.astype(np.uint64), live.sum(1).astype(np.uint64)))
            outside_count = int(np.dot(count.astype(np.uint64), outside.sum(1).astype(np.uint64)))
            record = {"bank_id": bank, "condition": condition, "seed": seed, "source": "archived_baseline" if condition == "collected_unique" else "new_treatment",
                "updates": samples[str(bank)][condition][str(seed)]["updates"], "state_presentations": int(count.sum()), "action_target_presentations": target_count,
                "terminal_action_targets": target_count - query_count, "nonterminal_action_targets": query_count, "nonterminal_target_queries": query_count,
                "outside_support_target_queries": outside_count, "outside_support_fraction": outside_count / query_count if query_count else None,
                "action_target_definition": "four outcomes perstate" if condition == "collected_unique" else "distinct logged outcomes perstate; perstate mean loss",
                "query_provenance": "historical counterfactual all-action transitions" if condition == "collected_unique" else "only nonterminal successors from manifest-verified logged edges"}
            records.append(record)
            if condition == "recorded_actions":
                actual = self.optimizer_counts.get((bank, seed), {})
                expected_queries = np.zeros(len(count), np.uint64)
                source_rows, actions = np.nonzero(live)
                np.add.at(expected_queries, successor[source_rows, actions], count[source_rows].astype(np.uint64))
                check = {"bank_id": bank, "seed": seed,
                    "tracked_exposure_matches": all(actual.get(k) == record[k] for k in ("updates", "state_presentations", "action_target_presentations", "nonterminal_target_queries")),
                    "recorded_query_counts_match": np.array_equal(self.query_counts.get((bank, seed)), expected_queries),
                    "no_outside_support_queries": outside_count == 0}
                checks.append(check)
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
        np.savez_compressed(output / "recorded_query_counts.npz", **{f"bank{b}_recorded_actions_seed{s}": v for (b, s), v in self.query_counts.items()})
        write_json(output / "action_exposure.json", self.action_exposure)
        return {"complete": len(self.tables) == len(self.bank_ids) and all(all(r[k] for k in ("tracked_exposure_matches", "recorded_query_counts_match", "no_outside_support_queries")) for r in checks)
            and all(r["unchanged"] and r["read_only"] and r["optimizer_tables_unchanged"] for r in self.table_integrity)}

    def provenance(self):
        return {"baseline_sampling_reconstruction": self.reconstruction, "recorded_table_integrity": self.table_integrity,
            "recorded_query_integrity": self.action_exposure["integrity"], "optimization_access": "recorded_update(agent,observations,recorded,indices); no exact/fulltransition input"}

    def configure_result(self, result):
        result["run"]["support_draws"] = 0
        result["run"]["baseline_reconstructed_updates"] = sum(r["reconstructed_updates"] for r in self.reconstruction)
        result["run"]["action_target_presentations"] = sum(r["action_target_presentations"] for r in self.action_exposure["per_seed"] if r["condition"] == "recorded_actions")
        result["run"]["nonterminal_target_queries"] = sum(r["nonterminal_target_queries"] for r in self.action_exposure["per_seed"] if r["condition"] == "recorded_actions")
        result["run"]["limitations"] = ["Offline deduplicated state replay with observed-action masks; no new collection, online adaptation or memory.",
            "State supports, replay draws and optimizer budgets match; supervised action counts and effective per-action weights change.",
            "DDQN argmax predicts all four actions at recorded successors, including actions without recorded outcomes there.",
            "Recorded successor access differs from the control's counterfactual all-action queries; no isolated extrapolation or target-count mechanism claim.",
            "Three banks share maps, learner seeds and panels; crossed cells do not increase the number of bank draws."]
        result["robustness"]["classification"] = "descriptive recorded-action supervision comparison on fixed collected state supports; no new gates or significance claims"
        if result["run"]["interpretation"] == "map_replay_descriptive_only":
            result["run"]["interpretation"] = "recorded_actions_descriptive_only"
        result["action_coverage"] = {"per_bank": self.table_summaries}
        result["action_exposure"] = self.action_exposure
        for name in ("action_coverage", "action_exposure"):
            write_json(self.output / f"{name}.json", result[name])
            result["artifacts"][name] = artifact_path(self.output / f"{name}.json")
        result["artifacts"]["recorded_transitions"] = artifact_path(self.output / "recorded_transitions.npz")
        result["artifacts"]["recorded_query_counts"] = artifact_path(self.output / "recorded_query_counts.npz")


def run_study(output, protocol_file, **kwargs):
    kwargs.setdefault("panel_seed_start", 1080000)
    comparison = RecordedActionsComparison()
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
    parser.add_argument("--panel-seed-start", type=int, default=1080000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    for name in shared.ARCHIVES + ("map_replay", "within_map"):
        parser.add_argument("--" + name.replace("_", "-") + "-dir", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.updates, args.panel_count, args.maps_per_panel, args.panel_seed_start = "0", 24, 2, 2, 1160000
    archives = {name: getattr(args, name + "_dir") for name in shared.ARCHIVES + ("map_replay", "within_map") if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, bank_ids=[int(v) for v in args.bank_ids.split(",")], seeds=[int(v) for v in args.seeds.split(",")], updates=args.updates,
        panel_count=args.panel_count, maps_per_panel=args.maps_per_panel, panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
