"""Restrict training-time DDQN successor argmax to logged actions only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import map_replay as shared
from .adaptation import write_json
from .competence import artifact_path
from .fixed_targets import ConsistencyError, array_metadata, module_hash
from .recorded_actions import RecordedActionsComparison
from .within_map import QuotaSampler

CONDITIONS = ("recorded_actions", "constrained_bootstrap")
COMPARISONS = ({"id": "constrained_minus_recorded", "left": CONDITIONS[0], "right": CONDITIONS[1]},)
SIGN_TOLERANCE = 1e-12
COUNT_FIELDS = ("query_count", "outside_count", "positive_target_deltas", "negative_target_deltas", "zero_target_deltas")
SUM_FIELDS = ("online_gap_sum", "online_gap_squared_sum", "target_delta_sum", "target_delta_squared_sum")
EXTREMA_FIELDS = ("online_gap_min", "online_gap_max", "target_delta_min", "target_delta_max")


def empty_diagnostics():
    return {**{k: 0 for k in COUNT_FIELDS}, **{k: 0. for k in SUM_FIELDS}, **{k: None for k in EXTREMA_FIELDS}}


def combine_diagnostics(rows):
    result = empty_diagnostics()
    for row in rows:
        for key in COUNT_FIELDS + SUM_FIELDS:
            result[key] += row[key]
        for key in EXTREMA_FIELDS:
            if row[key] is not None:
                result[key] = row[key] if result[key] is None else (min if key.endswith("_min") else max)(result[key], row[key])
    n = result["query_count"]
    result.update(outside_fraction=result["outside_count"] / n if n else None,
        online_gap_mean=result["online_gap_sum"] / n if n else None,
        target_delta_mean=result["target_delta_sum"] / n if n else None, sign_tolerance=SIGN_TOLERANCE)
    return result


def constrained_targets(agent, observations, recorded, indices, include_edges=False):
    """Pure pre-update predictions; unrestricted alternatives are diagnostics only."""
    indices = torch.as_tensor(indices.copy() if isinstance(indices, np.ndarray) and not indices.flags.writeable else indices, dtype=torch.long)
    observed = recorded["observed"][indices]
    action_counts = observed.sum(1)
    if not bool(torch.all(action_counts > 0)):
        raise ConsistencyError("sampled state has no recorded action")
    batch_rows, actions = observed.nonzero(as_tuple=True)
    sources = indices[batch_rows]
    rewards = recorded["rewards"][sources, actions]
    ended = recorded["ends"][sources, actions]
    successors = recorded["successor_indices"][sources, actions]
    diagnostics, edges = empty_diagnostics(), []
    with torch.no_grad():
        targets = rewards.clone()
        live = ~ended
        if bool(live.any()):
            next_rows = successors[live].long()
            if bool(torch.any((next_rows < 0) | (next_rows >= len(observations)))):
                raise ConsistencyError("nonterminal recorded successor index is invalid")
            allowed = recorded["observed"][next_rows]
            if not bool(torch.all(allowed.any(1))):
                raise ConsistencyError("recorded nonterminal successor has no logged action; no fallback")
            online_q = agent.online(observations[next_rows])
            target_q = agent.target(observations[next_rows])
            unrestricted = online_q.argmax(1)
            restricted = online_q.masked_fill(~allowed, -torch.inf).argmax(1)
            query_rows = torch.arange(len(next_rows))
            unrestricted_targets = rewards[live] + agent.gamma * target_q[query_rows, unrestricted]
            restricted_targets = rewards[live] + agent.gamma * target_q[query_rows, restricted]
            targets[live] = restricted_targets
            outside = ~allowed[query_rows, unrestricted]
            gap = (online_q[query_rows, unrestricted] - online_q[query_rows, restricted]).double()
            delta = (restricted_targets - unrestricted_targets).double()
            diagnostics.update(query_count=len(next_rows), outside_count=int(outside.sum()),
                positive_target_deltas=int((delta > SIGN_TOLERANCE).sum()), negative_target_deltas=int((delta < -SIGN_TOLERANCE).sum()),
                zero_target_deltas=int((delta.abs() <= SIGN_TOLERANCE).sum()), online_gap_sum=float(gap.sum()), online_gap_squared_sum=float(gap.square().sum()),
                online_gap_min=float(gap.min()), online_gap_max=float(gap.max()), target_delta_sum=float(delta.sum()), target_delta_squared_sum=float(delta.square().sum()),
                target_delta_min=float(delta.min()), target_delta_max=float(delta.max()))
            if any(not math.isfinite(v) for v in diagnostics.values() if isinstance(v, float)):
                raise ConsistencyError("nonfinite bootstrap diagnostic")
            if include_edges:
                live_sources, live_actions = sources[live], actions[live]
                for i in range(len(next_rows)):
                    edges.append({"current_row": int(live_sources[i]), "action": int(live_actions[i]), "successor_row": int(next_rows[i]),
                        "reward": float(rewards[live][i]), "successor_mask": allowed[i].tolist(), "online_q": online_q[i].tolist(), "target_q": target_q[i].tolist(),
                        "unrestricted_action": int(unrestricted[i]), "restricted_action": int(restricted[i]), "unrestricted_argmax_outside": bool(outside[i]),
                        "online_gap": float(gap[i]), "unrestricted_target": float(unrestricted_targets[i]), "restricted_target": float(restricted_targets[i]),
                        "target_delta": float(delta[i])})
    return {"targets": targets.detach(), "batch_rows": batch_rows, "actions": actions, "action_counts": action_counts,
        "diagnostics": combine_diagnostics([diagnostics]), "edges": edges}


def constrained_update(agent, observations, recorded, indices):
    result = constrained_targets(agent, observations, recorded, indices)
    indices = torch.as_tensor(indices.copy() if isinstance(indices, np.ndarray) and not indices.flags.writeable else indices, dtype=torch.long)
    prediction = agent.online(observations[indices])[result["batch_rows"], result["actions"]]
    edge_loss = F.smooth_l1_loss(prediction, result["targets"], reduction="none", beta=1.)
    state_loss = torch.zeros(len(indices), dtype=edge_loss.dtype).index_add(0, result["batch_rows"], edge_loss)
    loss = (state_loss / result["action_counts"]).mean()
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 5.)
    agent.optimizer.step()
    with torch.no_grad():
        for target, online in zip(agent.target.parameters(), agent.online.parameters()):
            target.lerp_(online, .01)
    return float(loss.item()), result["diagnostics"]


class ConstrainedBootstrapComparison(RecordedActionsComparison):
    conditions, comparisons = CONDITIONS, COMPARISONS
    module, panel_seed_start = "q6.constrained_bootstrap", 1100000
    extra_archives = ("map_replay", "within_map", "recorded_actions")
    control_archive, control_uses_recorded = "recorded_actions", True

    def __init__(self):
        super().__init__()
        self.windows, self.pending, self.probes, self.probe_indices = [], {}, [], {}
        self.control_queries, self.control_map_counts, self.control_action_rows = {}, {}, []
        self.control_table_checks, self.paired_target_checks = [], []
        self.diagnostic_integrity = {"complete": False}
        self.probe_integrity = {"complete": False}

    def configure_protocol(self, protocol):
        super().configure_protocol(protocol)
        self.seeds, self.updates, self.checkpoints = protocol["seeds"], protocol["budget"]["updates_per_fit"], protocol["checkpoints"]
        protocol.update(id="constrained-bootstrap-v1", question="Does limiting bootstrap action selection to logged successor actions help under matched recorded data and target count?",
            conditions=[{"id": "recorded_actions", "label": "Recorded outcomes, unrestricted backup (frozen)"}, {"id": "constrained_bootstrap", "label": "Logged-successor constrained backup"}])
        protocol["optimizer"].update(implementation="constrained_bootstrap.constrained_update", successor_argmax="logged successor action mask only; lowest allowed label on ties; no fallback")
        protocol["recorded_actions"]["equal_action_target_budget"] = True
        protocol["control_archive"] = "recorded_actions/pilot_v1"
        protocol["bootstrap_diagnostics"] = {"scope": "same pre-update treatment weights; unrestricted alternative never enters loss", "window_updates": 100,
            "query_denominator": "optimizer nonterminal recorded-edge presentations", "online_gap": "online unrestricted max minus online restricted selected value",
            "target_delta": "complete restricted float32 DDQN target minus complete unrestricted float32 DDQN target", "sign_tolerance": SIGN_TOLERANCE,
            "empty_query_windows": "zero counts/sums; null means/extrema", "historical_control_diagnostics": False}
        protocol["probes"] = {"selection": "first64 original SupportSampler batch per bank/seed, independently reconstructed and frozen before fits",
            "checkpoints": self.checkpoints, "expected": len(self.bank_ids) * len(self.seeds) * len(self.checkpoints),
            "no_optimizer_or_policy_evaluation": True, "full_online_target_vectors": True}
        protocol["evaluation"]["action_selection"] = "all four actions at fresh states; no recorded masks at deployment"

    def prepare_inputs(self, directories, manifests, archives_meta, data, transitions, output, enforce):
        super().prepare_inputs(directories, manifests, archives_meta, data, transitions, output, enforce)
        directory, manifest = directories["recorded_actions"], manifests["recorded_actions"]
        names = ("recorded_transitions.npz", "recorded_query_counts.npz", "map_counts.npz", "action_exposure.json", "action_coverage.json", "supports.npz", "protocol.json", "protocol.md")
        for filename in names:
            enforce()
            shared.checked_input(directory, manifest, filename, archives_meta["recorded_actions"]["files"])
            shutil.copy2(directory / filename, output / ("control_" + filename))
        with np.load(directory / "recorded_transitions.npz") as saved, np.load(directory / "supports.npz") as support_file:
            for bank, table in self.tables.items():
                archive_table = {k: saved[f"bank{bank}_{k}"] for k in table}
                identity = array_metadata(archive_table) == array_metadata(table)
                support_identity = np.array_equal(support_file[f"recorded_actions_bank{bank}"], self.original_supports[bank])
                self.control_table_checks.append({"bank_id": bank, "recorded_arrays_identical": identity, "support_identical": support_identity})
                if not identity or not support_identity:
                    raise ConsistencyError("recorded-control tables/support differ from reconstructed original logs")
        with np.load(directory / "recorded_query_counts.npz") as query_file, np.load(directory / "map_counts.npz") as map_file:
            for bank in self.bank_ids:
                for seed in self.seeds:
                    name = f"bank{bank}_recorded_actions_seed{seed}"
                    self.control_queries[(bank, seed)], self.control_map_counts[(bank, seed)] = query_file[name], map_file[name]
        self.control_action_rows = json.loads((directory / "action_exposure.json").read_text())["per_seed"]

    def record_baseline(self, bank, seed, support, data, global_counts, local_counts, archived, enforce, updates):
        map_digest = archived["map_index_sha256"]
        super().record_baseline(bank, seed, support, data, global_counts, local_counts, archived, enforce, updates)
        record = self.reconstruction[-1]
        record["map_digest_identical"] = record["map_index_sha256"] == map_digest
        record["map_counts_identical"] = record["count_arrays"]["map"] == array_metadata({"map": self.control_map_counts[(bank, seed)]})["map"]
        record["verified"] = record["verified"] and record["map_digest_identical"] and record["map_counts_identical"]
        if not record["verified"]:
            raise ConsistencyError("recorded-control ordered map stream differs from archive")
        probe = QuotaSampler(support, data["map_seeds"], seed).next_batch()
        probe.flags.writeable = False
        self.probe_indices[(bank, seed)] = probe
        np.savez_compressed(self.output / "probe_indices.npz", **{f"bank{b}_seed{s}": indices for (b, s), indices in self.probe_indices.items()})

    def compute_update(self, agent, observations, recorded, indices, bank, seed):
        loss, diagnostics = constrained_update(agent, observations, recorded, indices)
        key = bank, seed
        self.pending.setdefault(key, []).append(diagnostics)
        update = self.optimizer_counts[key]["updates"]
        if update % 100 == 0 or update == self.updates:
            self.flush_window(bank, seed, update)
        return loss

    def flush_window(self, bank, seed, checkpoint):
        rows = self.pending.get((bank, seed), [])
        if rows:
            self.windows.append({"bank_id": bank, "condition": "constrained_bootstrap", "seed": seed, "checkpoint": checkpoint,
                "updates_in_window": len(rows), **combine_diagnostics(rows)})
            self.pending[(bank, seed)] = []

    def checkpoint_probe(self, agent, observations, bank, seed, checkpoint, path):
        before = {"online": agent.parameter_hash(), "target": module_hash(agent.target), "updates": self.optimizer_counts.get((bank, seed), {}).get("updates", 0)}
        rng_before = torch.get_rng_state().clone()
        result = constrained_targets(agent, observations, self.tensors[bank], self.probe_indices[(bank, seed)], include_edges=True)
        after = {"online": agent.parameter_hash(), "target": module_hash(agent.target), "updates": self.optimizer_counts.get((bank, seed), {}).get("updates", 0)}
        unchanged = before == after and torch.equal(rng_before, torch.get_rng_state()) and not self.probe_indices[(bank, seed)].flags.writeable
        record = {"bank_id": bank, "condition": "constrained_bootstrap", "seed": seed, "checkpoint": checkpoint,
            "snapshot": path.relative_to(self.output).as_posix(), "snapshot_sha256": shared.sha(path), "state_rows": self.probe_indices[(bank, seed)].tolist(),
            "before": before, "after": after, "parameters_rng_counters_unchanged": unchanged,
            "diagnostics": result["diagnostics"], "edges": result["edges"]}
        self.probes.append(record)
        write_json(self.output / "bootstrap_probes.json", self.probe_result())
        if not unchanged:
            raise ConsistencyError("diagnostic probe changed parameters, RNG or optimizer counters")

    def probe_result(self):
        return {"classification": "fixed inference-only checkpoint probes; excluded from optimizer and policy totals",
            "per_checkpoint": self.probes, "probes": len(self.probes), "state_presentations": sum(len(r["state_rows"]) for r in self.probes),
            "query_count": sum(r["diagnostics"]["query_count"] for r in self.probes)}

    def finalize_exposure(self, data, transitions, supports, counts, samples, exposure, output):
        integrity = super().finalize_exposure(data, transitions, supports, counts, samples, exposure, output)
        for (bank, seed), pending in self.pending.items():
            if pending:
                self.flush_window(bank, seed, self.optimizer_counts[(bank, seed)]["updates"])
        per_seed = []
        for bank in self.bank_ids:
            for seed in self.seeds:
                windows = [r for r in self.windows if r["bank_id"] == bank and r["seed"] == seed]
                if windows:
                    per_seed.append({"bank_id": bank, "condition": "constrained_bootstrap", "seed": seed,
                        "updates": sum(r["updates_in_window"] for r in windows), **combine_diagnostics(windows)})
        self.diagnostics = {"classification": "same-weight alternatives during treatment optimization; no historical-control diagnostic curve",
            "per_seed": per_seed, "windows": self.windows, "query_denominator": "optimizer recorded nonterminal query presentations"}
        query_vectors = {f"bank{b}_constrained_bootstrap_seed{s}": v for (b, s), v in self.query_counts.items()}
        query_vectors.update({f"bank{b}_recorded_actions_seed{s}": v for (b, s), v in self.control_queries.items()})
        np.savez_compressed(output / "recorded_query_counts.npz", **query_vectors)
        for row in self.action_exposure["per_seed"]:
            bank, seed = row["bank_id"], row["seed"]
            if row["condition"] != "recorded_actions":
                continue
            old = next(r for r in self.control_action_rows if (r["bank_id"], r["condition"], r["seed"]) == (bank, "recorded_actions", seed))
            mask, ended, successor = (self.tables[bank][k] for k in ("observed", "ends", "successor_indices"))
            source_rows, actions = np.nonzero(mask & ~ended)
            expected = np.zeros(len(data["observations"]), np.uint64)
            np.add.at(expected, successor[source_rows, actions], counts[(bank, "recorded_actions", seed)][source_rows].astype(np.uint64))
            new = next((r for r in self.action_exposure["per_seed"] if (r["bank_id"], r["condition"], r["seed"]) == (bank, "constrained_bootstrap", seed)), None)
            target_fields = ("state_presentations", "action_target_presentations", "terminal_action_targets", "nonterminal_target_queries")
            check = {"bank_id": bank, "seed": seed,
                "control_action_counts_match_archive": all(row[k] == old[k] for k in target_fields),
                "control_query_vector_matches_archive": np.array_equal(expected, self.control_queries[(bank, seed)]),
                "same_budget": self.updates == 30000,
                "same_budget_target_counts_identical": bool(new) and all(row[k] == new[k] for k in target_fields) if self.updates == 30000 else None,
                "same_budget_query_vector_identical": np.array_equal(self.query_counts.get((bank, seed)), self.control_queries[(bank, seed)]) if self.updates == 30000 else None}
            self.paired_target_checks.append(check)
        self.diagnostic_integrity = {"expected_windows": math.ceil(self.updates / 100) * len(self.bank_ids) * len(self.seeds), "actual_windows": len(self.windows),
            "query_count": sum(r["query_count"] for r in per_seed), "optimizer_query_count": sum(v["nonterminal_target_queries"] for v in self.optimizer_counts.values())}
        self.diagnostic_integrity["complete"] = (len(self.windows) == self.diagnostic_integrity["expected_windows"]
            and self.diagnostic_integrity["query_count"] == self.diagnostic_integrity["optimizer_query_count"]
            and sum(r["updates"] for r in per_seed) == len(self.bank_ids) * len(self.seeds) * self.updates
            and all(r["positive_target_deltas"] + r["negative_target_deltas"] + r["zero_target_deltas"] == r["query_count"] for r in self.windows))
        expected_probes = {(b, s, cp) for b in self.bank_ids for s in self.seeds for cp in self.checkpoints}
        self.probe_integrity = {"expected": len(expected_probes), "actual": len(self.probes),
            "complete": {(r["bank_id"], r["seed"], r["checkpoint"]) for r in self.probes} == expected_probes and len(self.probes) == len(expected_probes),
            "unchanged": all(r["parameters_rng_counters_unchanged"] for r in self.probes)}
        write_json(output / "bootstrap_diagnostics.json", self.diagnostics)
        write_json(output / "bootstrap_probes.json", self.probe_result())
        with (output / "bootstrap_diagnostics.csv").open("w", newline="") as handle:
            fields = ["bank_id", "condition", "seed", "checkpoint", "updates_in_window", *combine_diagnostics([])]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(self.windows)
        integrity["complete"] = integrity["complete"] and self.diagnostic_integrity["complete"] and self.probe_integrity["complete"] and self.probe_integrity["unchanged"] and all(
            r["control_action_counts_match_archive"] and r["control_query_vector_matches_archive"] and
            (not r["same_budget"] or (r["same_budget_target_counts_identical"] and r["same_budget_query_vector_identical"])) for r in self.paired_target_checks)
        return integrity

    def provenance(self):
        return {**super().provenance(), "control_archive": "recorded_actions/pilot_v1", "control_table_identity": self.control_table_checks,
            "paired_target_query_identity": self.paired_target_checks, "bootstrap_diagnostic_integrity": self.diagnostic_integrity, "probe_integrity": self.probe_integrity,
            "optimization_access": "constrained_update(agent,observations,recorded,indices); only successor selection mask differs"}

    def configure_result(self, result):
        super().configure_result(result)
        result["run"]["interpretation"] = "constrained_bootstrap_descriptive_only" if result["run"]["interpretation"] == "recorded_actions_descriptive_only" else result["run"]["interpretation"]
        result["run"].update(probe_evaluations=len(self.probes), probe_state_presentations=sum(len(r["state_rows"]) for r in self.probes),
            probe_nonterminal_queries=sum(r["diagnostics"]["query_count"] for r in self.probes))
        result["run"]["limitations"] = ["Only the training bootstrap action set is constrained; fresh-world action selection remains unrestricted.",
            "Restricting the backup can exclude useful actions and changes the backup operator; no sole-cause extrapolation or overestimation claim.",
            "Both arms use the same recorded masks, states, outcomes, replay and target counts; no new collection, online learning or memory.",
            "Window diagnostics describe same-weight alternatives in treatment; fixed probes are separate inference-only observations, not checkpoint selection.",
            "Three banks share training maps, learner initializations and panels; crossed cells do not increase independent replication count."]
        result["robustness"]["classification"] = "descriptive constrained-bootstrap comparison; no new gates or significance claims"
        result["bootstrap_diagnostics"] = getattr(self, "diagnostics", {"per_seed": [], "windows": []})
        result["bootstrap_probes"] = self.probe_result()
        for name, filename in (("bootstrap_diagnostics", "bootstrap_diagnostics.json"), ("bootstrap_diagnostic_windows", "bootstrap_diagnostics.csv"),
                ("bootstrap_probes", "bootstrap_probes.json"), ("probe_indices", "probe_indices.npz")):
            result["artifacts"][name] = artifact_path(self.output / filename)


def run_study(output, protocol_file, **kwargs):
    kwargs.setdefault("panel_seed_start", 1100000)
    comparison = ConstrainedBootstrapComparison()
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
    parser.add_argument("--panel-seed-start", type=int, default=1100000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    for name in shared.ARCHIVES + ("map_replay", "within_map", "recorded_actions"):
        parser.add_argument("--" + name.replace("_", "-") + "-dir", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.updates, args.panel_count, args.maps_per_panel, args.panel_seed_start = "0", 24, 2, 2, 1180000
    archives = {name: getattr(args, name + "_dir") for name in shared.ARCHIVES + ("map_replay", "within_map", "recorded_actions") if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, bank_ids=[int(v) for v in args.bank_ids.split(",")], seeds=[int(v) for v in args.seeds.split(",")], updates=args.updates,
        panel_count=args.panel_count, maps_per_panel=args.maps_per_panel, panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
