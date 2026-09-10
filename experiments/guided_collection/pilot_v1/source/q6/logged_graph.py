"""Fit exact finite-horizon values using only the closed graph of logged edges."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import map_replay as shared
from .adaptation import write_json
from .competence import artifact_path
from .fixed_targets import ConsistencyError, array_metadata, module_hash
from .recorded_actions import RecordedActionsComparison

CONDITIONS = ("constrained_bootstrap", "logged_graph")
COMPARISONS = ({"id": "logged_graph_minus_constrained", "left": CONDITIONS[0], "right": CONDITIONS[1]},)
RANK_ATOL = 1e-6


def solve_logged_graph(recorded, remaining, map_seeds, gamma=.97, enforce=None):
    """Backward DP over logged edges only; no environment or exhaustive Q input."""
    enforce = enforce or (lambda: None)
    observed = np.asarray(recorded["observed"], dtype=bool)
    remaining, map_seeds = np.asarray(remaining), np.asarray(map_seeds)
    n = len(observed)
    if observed.shape != (n, 4) or remaining.shape != (n,) or map_seeds.shape != (n,) or not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ConsistencyError("invalid logged graph shape or discount")
    if not np.all(np.isfinite(remaining)) or not np.all(remaining == remaining.astype(np.int64)) or np.any(remaining < 1):
        raise ConsistencyError("logged graph clocks must be positive integers")
    for name in ("rewards", "ends", "terminated", "truncated", "successor_indices"):
        if np.shape(recorded[name]) != observed.shape:
            raise ConsistencyError("logged edge array shape mismatch")
    rows, actions = np.nonzero(observed)
    if not len(rows):
        raise ConsistencyError("empty logged graph")
    rewards = np.asarray(recorded["rewards"])[rows, actions].astype(np.float64)
    ended = np.asarray(recorded["ends"])[rows, actions].astype(bool)
    terminated = np.asarray(recorded["terminated"])[rows, actions].astype(bool)
    truncated = np.asarray(recorded["truncated"])[rows, actions].astype(bool)
    successor = np.asarray(recorded["successor_indices"])[rows, actions]
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(successor)) or not np.all(successor == successor.astype(np.int64)):
        raise ConsistencyError("nonfinite logged reward or invalid successor index")
    successor = successor.astype(np.int64)
    if np.any(terminated & truncated) or not np.array_equal(ended, terminated | truncated) or np.any(truncated & (remaining[rows] != 1)):
        raise ConsistencyError("logged terminal flags disagree")
    if np.any(successor[ended] != -1):
        raise ConsistencyError("terminal logged edge must have successor -1")
    live = ~ended
    nxt = successor[live]
    if np.any((nxt < 0) | (nxt >= n)):
        raise ConsistencyError("nonterminal logged successor index invalid")
    if np.any(~observed[nxt].any(1)) or np.any(map_seeds[nxt] != map_seeds[rows[live]]) or np.any(remaining[nxt] != remaining[rows[live]] - 1):
        raise ConsistencyError("logged graph is not closed within map and decreasing clock")
    targets = np.full(observed.shape, np.nan, np.float64)
    values = np.full(n, np.nan, np.float64)
    per_clock = []
    for clock in np.unique(remaining[rows]):
        enforce()
        select = remaining[rows] == clock
        current, action = rows[select], actions[select]
        q = rewards[select].copy()
        current_live = live[select]
        if current_live.any():
            downstream = values[successor[select][current_live]]
            if not np.all(np.isfinite(downstream)):
                raise ConsistencyError("logged backward DP queried an unresolved successor")
            q[current_live] += gamma * downstream
        if not np.all(np.isfinite(q)):
            raise ConsistencyError("nonfinite logged graph target")
        targets[current, action] = q
        unique = np.unique(current)
        values[unique] = np.where(observed[unique], targets[unique], -np.inf).max(1)
        per_clock.append({"remaining": int(clock), "supported_states": len(unique), "observed_edges": int(select.sum()),
            "terminal_edges": int((~current_live).sum()), "nonterminal_edges": int(current_live.sum()), "successor_backups": int(current_live.sum())})
    enforce()
    reconstructed = rewards.copy()
    reconstructed[live] += gamma * values[nxt]
    residual = np.abs(targets[rows, actions] - reconstructed)
    max_residual = float(residual.max())
    if max_residual > 1e-12:
        raise ConsistencyError("logged graph Bellman residual exceeds declared tolerance")
    targets.flags.writeable = False
    return targets, {"supported_states": int(observed.any(1).sum()), "observed_edges": len(rows),
        "terminal_edges": int(ended.sum()), "nonterminal_edges": int(live.sum()), "graph_preparation_successor_backups": int(live.sum()),
        "neural_successor_queries": 0, "clock_levels": len(np.unique(remaining[rows])), "gamma": gamma,
        "all_successors_closed": True, "per_clock": per_clock, "bellman_residual_max": max_residual,
        "bellman_residual_mean": float(residual.mean()), "bellman_residual_tolerance": 1e-12, "residual_validation_successor_lookups": int(live.sum()), "optimizer_access": "float32 cast of fixed logged-only float64 DP labels; absent actions excluded"}


def logged_graph_update(agent, observations, observed_mask, targets, indices):
    """Only sampled current observations and fixed observed labels enter the loss."""
    indices = torch.as_tensor(indices.copy() if isinstance(indices, np.ndarray) and not indices.flags.writeable else indices, dtype=torch.long)
    mask = observed_mask[indices]
    action_counts = mask.sum(1)
    if not bool(torch.all(action_counts > 0)):
        raise ConsistencyError("sampled state has no logged target")
    rows, actions = mask.nonzero(as_tuple=True)
    labels = targets[indices[rows], actions].detach()
    if not bool(torch.isfinite(labels).all()):
        raise ConsistencyError("nonfinite observed logged graph target")
    predictions = agent.online(observations[indices])[rows, actions]
    edge_loss = F.smooth_l1_loss(predictions, labels, reduction="none", beta=1.)
    state_loss = torch.zeros(len(indices), dtype=edge_loss.dtype).index_add(0, rows, edge_loss)
    loss = (state_loss / action_counts).mean()
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 5.)
    agent.optimizer.step()
    with torch.no_grad():
        for target, online in zip(agent.target.parameters(), agent.online.parameters()):
            target.lerp_(online, .01)
    return float(loss.item())


def logged_graph_metrics(predictions, targets, observed):
    """Equal state weighting for fit/ranking; edge-pooled errors are separate."""
    predictions, targets, observed = np.asarray(predictions), np.asarray(targets), np.asarray(observed, bool)
    if predictions.shape != targets.shape or observed.shape != targets.shape or observed.ndim != 2 or observed.shape[1] != 4 or not len(observed) or not np.all(observed.any(1)):
        raise ConsistencyError("logged fit requires nonempty supported rows with four actions")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets[observed])):
        raise ConsistencyError("nonfinite logged fit prediction or observed label")
    rows, actions = np.nonzero(observed)
    error = predictions[rows, actions].astype(np.float64) - targets[rows, actions].astype(np.float64)
    counts = observed.sum(1)
    abs_sum = np.bincount(rows, weights=np.abs(error), minlength=len(observed))
    squared_sum = np.bincount(rows, weights=error**2, minlength=len(observed))
    selected = np.where(observed, predictions, -np.inf).argmax(1)
    graph_max = np.where(observed, targets, -np.inf).max(1)
    regret = graph_max - targets[np.arange(len(observed)), selected]
    return {"states": len(observed), "observed_edges": len(error),
        "state_mean_abs_error": float(np.mean(abs_sum / counts)), "state_mean_squared_error": float(np.mean(squared_sum / counts)),
        "edge_mean_abs_error": float(np.abs(error).mean()), "edge_max_abs_error": float(np.abs(error).max()),
        "restricted_action_agreement": float(np.mean(regret <= RANK_ATOL)), "mean_graph_regret": float(regret.mean()), "max_graph_regret": float(regret.max()),
        "unrestricted_argmax_outside_logged_fraction": float(np.mean(~observed[np.arange(len(observed)), predictions.argmax(1)])),
        "restricted_ranking_atol": RANK_ATOL, "restricted_ranking_rtol": 0}


class LoggedGraphComparison(RecordedActionsComparison):
    conditions, comparisons = CONDITIONS, COMPARISONS
    module, panel_seed_start = "q6.logged_graph", 1120000
    extra_archives = ("map_replay", "within_map", "recorded_actions", "constrained_bootstrap")
    control_archive, control_uses_recorded, uses_neural_successors = "constrained_bootstrap", True, False

    def __init__(self):
        super().__init__()
        self.graphs, self.graph_casts, self.graph_tensors, self.graph_before, self.graph_records = {}, {}, {}, {}, []
        self.control_queries, self.control_map_counts, self.control_action_rows = {}, {}, []
        self.control_table_checks, self.paired_target_checks, self.graph_integrity = [], [], []
        self.fit_records, self.fit_predictions = [], {}
        self.fit_seconds = 0.
        self.fit_peak_rss_bytes = 0
        self.fit_inference_rows, self.fit_forward_batches = 0, 0
        self.fit_integrity = {"complete": False}

    def configure_protocol(self, protocol):
        super().configure_protocol(protocol)
        self.seeds, self.updates = protocol["seeds"], protocol["budget"]["updates_per_fit"]
        protocol.update(id="logged-graph-v1", question="Does exact evaluation of the same logged-action graph improve learning over constrained bootstrapping?",
            conditions=[{"id": CONDITIONS[0], "label": "Logged-successor constrained backup (frozen)"}, {"id": CONDITIONS[1], "label": "Exact logged-graph targets"}])
        protocol["optimizer"].update(implementation="logged_graph.logged_graph_update", target="fixed float32 cast of backward float64 DP over logged edges only",
            successor_neural_queries=0, target_network="tau .01 maintained after each optimizer update; never queried by treatment optimizer")
        protocol["sampling"]["target_access"] = "sampled current observations, observed-action masks and fixed logged-graph labels only"
        protocol["recorded_actions"]["equal_action_target_budget"] = True
        protocol["control_archive"] = "constrained_bootstrap/pilot_v1"
        protocol["logged_graph"] = {"precision": "float64 recurrence, float32 optimizer labels", "gamma": .97,
            "absent_actions": "NaN, excluded before max or loss", "terminal": "logged reward, successor -1", "order": "ascending remaining clock",
            "inputs": "logged observed/reward/end/successor tables plus remaining clock and map identity; no exhaustive Q or world transition calls"}
        protocol["fit_diagnostics"] = {"final_only": True, "after_all_treatment_fits": True, "both_conditions": True, "no_new_gate": True,
            "population": "every supported current state, including states without a successful logged path", "reference": "float64 logged graph values",
            "prediction_order": "saved sorted support rows; four online Q outputs per row", "batch_size": 1024,
            "state_errors": "mean over states of mean observed-action absolute/squared error", "edge_errors": "pooled observed-action absolute mean and maximum",
            "restricted_ranking": "lowest-label predicted argmax within logged mask belongs to graph max set within atol1e-6 rtol0",
            "graph_regret": "logged graph maximum minus graph value of predicted restricted argmax", "unrestricted_argmax_outside_logged_fraction": "all supported states denominator",
            "optimizer_successor_queries": False, "checkpoint_probes": False}
        protocol["evaluation"]["action_selection"] = "all four actions on fresh maps; no logged masks at deployment"

    def prepare_inputs(self, directories, manifests, archives_meta, data, transitions, output, enforce):
        super().prepare_inputs(directories, manifests, archives_meta, data, transitions, output, enforce)
        directory, manifest = directories[self.control_archive], manifests[self.control_archive]
        for filename in ("recorded_transitions.npz", "recorded_query_counts.npz", "map_counts.npz", "action_exposure.json", "action_coverage.json", "supports.npz", "protocol.json", "protocol.md"):
            enforce()
            shared.checked_input(directory, manifest, filename, archives_meta[self.control_archive]["files"])
            shutil.copy2(directory / filename, output / ("control_" + filename))
        with np.load(directory / "recorded_transitions.npz") as saved, np.load(directory / "supports.npz") as support_file:
            for bank, table in self.tables.items():
                archived = {k: saved[f"bank{bank}_{k}"] for k in table}
                same_table = array_metadata(archived) == array_metadata(table)
                same_support = np.array_equal(support_file[f"{CONDITIONS[0]}_bank{bank}"], self.original_supports[bank])
                check = {"bank_id": bank, "recorded_arrays_identical": same_table, "support_identical": same_support}
                self.control_table_checks.append(check)
                if not same_table or not same_support:
                    raise ConsistencyError("constrained control recorded graph or support differs")
        with np.load(directory / "recorded_query_counts.npz") as queries, np.load(directory / "map_counts.npz") as maps:
            for bank in self.bank_ids:
                for seed in self.seeds:
                    name = f"bank{bank}_{CONDITIONS[0]}_seed{seed}"
                    self.control_queries[(bank, seed)], self.control_map_counts[(bank, seed)] = queries[name], maps[name]
        self.control_action_rows = json.loads((directory / "action_exposure.json").read_text())["per_seed"]
        for bank, table in self.tables.items():
            phase = time.monotonic()
            targets, summary = solve_logged_graph(table, data["remaining"], data["map_seeds"], .97, enforce)
            cast = targets.astype(np.float32)
            cast.flags.writeable = False
            tensor = torch.from_numpy(cast.copy())
            self.graphs[bank], self.graph_casts[bank], self.graph_tensors[bank] = targets, cast, tensor
            metadata = array_metadata({"targets_float64": targets, "targets_float32": tensor.numpy()})
            self.graph_before[bank] = metadata
            self.graph_records.append({"bank_id": bank, **summary, "arrays": metadata, "preparation_wall_seconds": time.monotonic() - phase})
            self.save_graphs()
            enforce()

    def save_graphs(self):
        arrays = {}
        for bank, target in self.graphs.items():
            arrays[f"bank{bank}_targets_float64"] = target
            arrays[f"bank{bank}_targets_float32"] = self.graph_casts[bank]
        np.savez_compressed(self.output / "logged_graph_targets.npz", **arrays)
        write_json(self.output / "logged_graph.json", {"classification": "exact values of the closed recorded-action graph; not full-world optimal values", "per_bank": self.graph_records})

    def record_baseline(self, bank, seed, support, data, global_counts, local_counts, archived, enforce, updates):
        old_map_digest = archived["map_index_sha256"]
        super().record_baseline(bank, seed, support, data, global_counts, local_counts, archived, enforce, updates)
        check = self.reconstruction[-1]
        check["map_digest_identical"] = check["map_index_sha256"] == old_map_digest
        check["map_counts_identical"] = check["count_arrays"]["map"] == array_metadata({"map": self.control_map_counts[(bank, seed)]})["map"]
        check["verified"] = check["verified"] and check["map_digest_identical"] and check["map_counts_identical"]
        if not check["verified"]:
            raise ConsistencyError("constrained control map replay differs from archive")

    def compute_update(self, agent, observations, recorded, indices, bank, seed):
        return logged_graph_update(agent, observations, recorded["observed"], self.graph_tensors[bank], indices)

    def final_fit_diagnostics(self, models, model_records, data, output, enforce):
        expected = {(b, c, s) for b in self.bank_ids for c in self.conditions for s in self.seeds}
        actual = {(r["bank_id"], r["condition"], r["seed"]) for r in model_records}
        if actual != expected or len(model_records) != len(expected):
            raise ConsistencyError("all treatment fits and frozen controls must finish before graph fit diagnostics")
        phase = time.monotonic()
        try:
            for record in model_records:
                enforce()
                key = record["bank_id"], record["condition"], record["seed"]
                bank, condition, seed = key
                model, support = models[key], self.original_supports[bank]
                before = shared.model_boundary(record, model, output, "before")
                rng_before = torch.get_rng_state().clone()
                predictions = np.empty((len(support), 4), np.float32)
                try:
                    with torch.no_grad():
                        for start in range(0, len(support), 1024):
                            enforce()
                            indices = support[start:start + 1024]
                            predictions[start:start + len(indices)] = model.online(torch.from_numpy(data["observations"][indices])).numpy()
                            self.fit_inference_rows += len(indices)
                            self.fit_forward_batches += 1
                            self.fit_peak_rss_bytes = max(self.fit_peak_rss_bytes, shared.peak_rss_bytes())
                    metrics = logged_graph_metrics(predictions, self.graphs[bank][support], self.tables[bank]["observed"][support])
                finally:
                    after = shared.model_boundary(record, model, output, "after")
                    unchanged = all(before[k + "_before"] == after[k + "_after"] == record[k + "_before"] for k in ("online", "target", "file", "source")) and torch.equal(rng_before, torch.get_rng_state())
                    if not unchanged:
                        raise ConsistencyError("graph fit inference altered model/source/RNG")
                self.fit_predictions[f"bank{bank}_state_rows"] = support
                prediction_key = f"bank{bank}_{condition}_seed{seed}"
                self.fit_predictions[prediction_key] = predictions
                self.fit_records.append({"bank_id": bank, "condition": condition, "seed": seed, "checkpoint": record["checkpoint"],
                    "prediction_key": prediction_key, "snapshot": record["saved"], "snapshot_sha256": record["file_before"],
                    "predictions": array_metadata({"predictions": predictions})["predictions"], **metrics,
                    "inference_state_rows": len(support), "online_output_values": 4 * len(support), "target_network_queries": 0,
                    "before": before, "after": after, "parameters_sources_rng_unchanged": unchanged})
                np.savez_compressed(output / "fit_predictions.npz", **self.fit_predictions)
                self.fit_peak_rss_bytes = max(self.fit_peak_rss_bytes, shared.peak_rss_bytes())
                self.save_fit()
                enforce()
        finally:
            self.fit_seconds += time.monotonic() - phase
            completed = {(r["bank_id"], r["condition"], r["seed"]) for r in self.fit_records}
            self.fit_integrity = {"expected_models": len(expected), "actual_models": len(self.fit_records),
                "complete": completed == expected and len(self.fit_records) == len(expected) and all(r["parameters_sources_rng_unchanged"] for r in self.fit_records)}
            self.save_fit()

    def fit_result(self):
        pooled = []
        for condition in self.conditions:
            rows = [r for r in self.fit_records if r["condition"] == condition]
            if not rows:
                continue
            states, edges = sum(r["states"] for r in rows), sum(r["observed_edges"] for r in rows)
            row = {"condition": condition, "models": len(rows), "states": states, "observed_edges": edges}
            for metric in ("state_mean_abs_error", "state_mean_squared_error", "restricted_action_agreement", "mean_graph_regret", "unrestricted_argmax_outside_logged_fraction"):
                row[metric] = sum(r[metric] * r["states"] for r in rows) / states
            row["edge_mean_abs_error"] = sum(r["edge_mean_abs_error"] * r["observed_edges"] for r in rows) / edges
            for metric in ("edge_max_abs_error", "max_graph_regret"):
                row[metric] = max(r[metric] for r in rows)
            pooled.append(row)
        return {"classification": "final supported-state fit against exact logged graph; descriptive, not a gate or full-world fit",
            "per_seed": self.fit_records, "pooled": pooled, "pooling": "state-presentation weighted state metrics; edge-presentation weighted edge MAE; maxima across models",
            "inference_state_rows": self.fit_inference_rows, "online_forward_batches": self.fit_forward_batches,
            "completed_prediction_state_rows": sum(r["states"] for r in self.fit_records), "observed_edge_comparisons": sum(r["observed_edges"] for r in self.fit_records),
            "online_output_values": 4 * self.fit_inference_rows, "neural_successor_queries": 0,
            "wall_seconds": self.fit_seconds, "sampled_process_peak_rss_bytes": self.fit_peak_rss_bytes, "integrity": self.fit_integrity}

    def save_fit(self):
        write_json(self.output / "fit_diagnostics.json", self.fit_result())

    def finalize_exposure(self, data, transitions, supports, counts, samples, exposure, output):
        integrity = super().finalize_exposure(data, transitions, supports, counts, samples, exposure, output)
        queries = {f"bank{b}_{CONDITIONS[0]}_seed{s}": a for (b, s), a in self.control_queries.items()}
        queries.update({f"bank{b}_{CONDITIONS[1]}_seed{s}": a for (b, s), a in self.query_counts.items()})
        np.savez_compressed(output / "recorded_query_counts.npz", **queries)
        for row in self.action_exposure["per_seed"]:
            bank, seed = row["bank_id"], row["seed"]
            if row["condition"] == CONDITIONS[1]:
                row["query_provenance"] = "zero optimizer neural successor queries; fixed logged-graph labels prepared before fitting"
                summary = next(r for r in exposure["summaries"] if (r["bank_id"], r["condition"], r["seed"]) == (bank, CONDITIONS[1], seed))
                summary["successor_queries"]["source"] = row["query_provenance"]
                continue
            old = next(r for r in self.control_action_rows if (r["bank_id"], r["condition"], r["seed"]) == (bank, CONDITIONS[0], seed))
            table = self.tables[bank]
            source, action = np.nonzero(table["observed"] & ~table["ends"])
            expected = np.zeros(len(data["observations"]), np.uint64)
            np.add.at(expected, table["successor_indices"][source, action], counts[(bank, CONDITIONS[0], seed)][source].astype(np.uint64))
            new = next((r for r in self.action_exposure["per_seed"] if (r["bank_id"], r["condition"], r["seed"]) == (bank, CONDITIONS[1], seed)), None)
            target_fields = ("state_presentations", "action_target_presentations", "terminal_action_targets", "nonterminal_action_targets")
            self.paired_target_checks.append({"bank_id": bank, "seed": seed, "same_budget": self.updates == 30000,
                "control_target_and_query_counts_match_archive": all(row[k] == old[k] for k in (*target_fields, "nonterminal_target_queries")),
                "control_query_vector_matches_archive": np.array_equal(expected, self.control_queries[(bank, seed)]),
                "same_budget_target_counts_identical": bool(new) and all(row[k] == new[k] for k in target_fields) if self.updates == 30000 else None,
                "treatment_neural_queries_zero": new is None or new["nonterminal_target_queries"] == 0})
        for bank, target in self.graphs.items():
            after = array_metadata({"targets_float64": target, "targets_float32": self.graph_tensors[bank].numpy()})
            self.graph_integrity.append({"bank_id": bank, "before": self.graph_before[bank], "after": after,
                "unchanged": after == self.graph_before[bank], "read_only_reference": not target.flags.writeable,
                "read_only_float32_cast": not self.graph_casts[bank].flags.writeable,
                "tensor_matches_saved_cast": np.array_equal(self.graph_tensors[bank].numpy(), self.graph_casts[bank], equal_nan=True)})
        self.action_exposure["classification"] = "matched observed-action target presentations; fixed labels require no optimizer successor inference"
        self.save_fit()
        write_json(output / "action_exposure.json", self.action_exposure)
        complete = (integrity["complete"] and len(self.graphs) == len(self.bank_ids) and self.fit_integrity["complete"]
            and all(r["unchanged"] and r["read_only_reference"] and r["read_only_float32_cast"] and r["tensor_matches_saved_cast"] for r in self.graph_integrity)
            and len(self.paired_target_checks) == len(self.bank_ids) * len(self.seeds)
            and all(r["control_target_and_query_counts_match_archive"] and r["control_query_vector_matches_archive"] and r["treatment_neural_queries_zero"]
                and (r["same_budget_target_counts_identical"] if r["same_budget"] else True) for r in self.paired_target_checks))
        return {"complete": complete}

    def provenance(self):
        return {**super().provenance(), "control_recorded_tables": self.control_table_checks, "paired_target_count_integrity": self.paired_target_checks,
            "logged_graph_integrity": self.graph_integrity, "fit_diagnostic_integrity": self.fit_integrity,
            "optimization_access": "logged_graph_update(agent,observations,observed_mask,targets_float32,indices); no reward/end/successor/fulltransition/oracle arguments"}

    def configure_result(self, result):
        super().configure_result(result)
        result["run"]["interpretation"] = "logged_graph_descriptive_only" if result["run"]["interpretation"] == "recorded_actions_descriptive_only" else result["run"]["interpretation"]
        result["run"]["limitations"] = ["Exact targets describe the closed logged-action graph, not full-world optimal control.",
            "The constrained backup operator, current-state replay and logged action loss stay fixed; evolving lagged neural targets become fixed graph values.",
            "Same action-target count does not imply same compute: exact treatment has zero optimizer successor neural queries; graph preparation and final fit inference are separate.",
            "Fresh policy selection remains unrestricted across four outputs; logged-mask ranking diagnostics concern training support only.",
            "Three banks share training maps, learner initializations and evaluation panels; no new memory, collection, competence gate or significance claim."]
        result["robustness"]["classification"] = "descriptive exact logged-graph target comparison; no new gates or significance claims"
        result["logged_graph"] = {"classification": "exact logged-action graph, not full-world optimal values", "per_bank": self.graph_records}
        result["fit_diagnostics"] = self.fit_result()
        result["run"].update(graph_preparation_edges=sum(r["observed_edges"] for r in self.graph_records),
            graph_preparation_successor_backups=sum(r["graph_preparation_successor_backups"] for r in self.graph_records),
            fit_diagnostic_state_rows=result["fit_diagnostics"]["inference_state_rows"], fit_diagnostic_observed_edge_comparisons=result["fit_diagnostics"]["observed_edge_comparisons"],
            fit_diagnostic_wall_seconds=self.fit_seconds, fit_diagnostic_sampled_peak_rss_bytes=self.fit_peak_rss_bytes,
            checkpoint_probes=0, probe_query_count=0)
        for name in ("logged_graph", "fit_diagnostics"):
            write_json(self.output / f"{name}.json", result[name])
            result["artifacts"][name] = artifact_path(self.output / f"{name}.json")
        result["artifacts"]["logged_graph_targets"] = artifact_path(self.output / "logged_graph_targets.npz")
        result["artifacts"]["fit_predictions"] = artifact_path(self.output / "fit_predictions.npz")


def run_study(output, protocol_file, **kwargs):
    kwargs.setdefault("panel_seed_start", 1120000)
    comparison = LoggedGraphComparison()
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
    parser.add_argument("--panel-seed-start", type=int, default=1120000)
    parser.add_argument("--panel-stride", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--checkpoints")
    parser.add_argument("--smoke", action="store_true")
    for name in shared.ARCHIVES + LoggedGraphComparison.extra_archives:
        parser.add_argument("--" + name.replace("_", "-") + "-dir", type=Path)
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.updates, args.panel_count, args.maps_per_panel, args.panel_seed_start = "0", 24, 2, 2, 1200000
    archives = {name: getattr(args, name + "_dir") for name in shared.ARCHIVES + LoggedGraphComparison.extra_archives if getattr(args, name + "_dir") is not None}
    result = run_study(args.output, args.protocol_file, bank_ids=[int(v) for v in args.bank_ids.split(",")], seeds=[int(v) for v in args.seeds.split(",")], updates=args.updates,
        panel_count=args.panel_count, maps_per_panel=args.maps_per_panel, panel_seed_start=args.panel_seed_start, panel_stride=args.panel_stride,
        max_seconds=args.max_seconds, max_rss_bytes=args.max_rss_bytes, dashboard=args.dashboard, archives=archives, smoke=args.smoke,
        checkpoints=[int(v) for v in args.checkpoints.split(",")] if args.checkpoints else None)
    print(json.dumps(result["run"], indent=2))
    if result["run"]["status"] != "complete":
        raise SystemExit(124)


if __name__ == "__main__":
    main()
