"""
dashboard/scan.py — Build the static index that the dashboard reads.

Scans `training_runs/*/experiment.json`, gathers each run's metadata + the
list of available replays + a downsampled training-curve series, and writes
`dashboard/data/index.json`.

The dashboard is fully static (no backend). To browse it locally:

    python dashboard/scan.py
    python -m http.server          # from repo root
    # open http://localhost:8000/dashboard/index.html

Rebuild the index any time a new run completes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "training_runs"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _downsample(values: list[float], max_points: int = 500) -> list[float]:
    if len(values) <= max_points:
        return values
    step = len(values) / max_points
    return [values[int(i * step)] for i in range(max_points)]


#: Optional CSV columns -> output series key. `required=False` series are
#: only included in the summary if the column is actually present in this
#: particular CSV (checked once against the DictReader's fieldnames, not
#: per-row) — this is what lets the frontend show/hide whole chart blocks
#: per algorithm instead of rendering a flat-zero placeholder chart for a
#: metric that doesn't apply (e.g. there is no "epsilon" for a PPO run, no
#: "mean_gate"/"cher_dep_idx" outside the GOP+CHER architecture, no
#: "mean_hard_tier_score"/"anchor_weight" outside the v7-ablations
#: experiments). Column names are krishna-centric throughout, matching the
#: existing reward/pellets convention, even though Hunter has some of these
#: too — keeps the chart count from ballooning and stays consistent with
#: what was already established.
_OPTIONAL_SERIES: list[tuple[str, tuple[str, ...]]] = [
    ("epsilons", ("epsilon", "krishna_eps")),          # Phase 1 / 2 / 3
    ("losses", ("loss", "krishna_loss")),                # all phases
    ("avg100s", ("avg100",)),                            # Phase 3+
    ("gates", ("mean_gate",)),                           # GOP (Phase 3)
    ("cher_deps", ("cher_dep_idx",)),                    # CHER (Phase 3)
    ("entropies", ("krishna_entropy",)),                 # PPO (v8)
    ("approx_kls", ("krishna_approx_kl",)),               # PPO (v8)
    ("clipfracs", ("krishna_clipfrac",)),                 # PPO (v8)
    ("mean_hard_tier_scores", ("mean_hard_tier_score",)),  # v7-ablations (rectified sampling)
    ("anchor_weights", ("anchor_weight",)),               # v7-ablations (frozen anchor)
]


def _read_log_summary(log_path: Path) -> dict[str, Any]:
    """Read episode_stats.csv and return downsampled series for charting.

    Different training scripts write different columns (Phase 1's
    single-agent naming vs Phase 2/3's krishna_-prefixed self-play naming vs
    v8's PPO-specific columns) — required series (episode/reward/pellets/
    win) fall back across the known naming variants; everything else is
    genuinely optional (see _OPTIONAL_SERIES) and only included if present.
    """
    if not log_path.exists():
        return {}
    import math

    episodes: list[int] = []
    rewards: list[float] = []
    pellets: list[float] = []
    wins: list[int] = []
    extra: dict[str, list[float]] = {key: [] for key, _cols in _OPTIONAL_SERIES}

    with open(log_path) as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        # Resolve each optional series to the one column name (if any) that's
        # actually present in THIS csv, once, before scanning rows.
        resolved: dict[str, str | None] = {}
        for key, candidates in _OPTIONAL_SERIES:
            resolved[key] = next((c for c in candidates if c in fieldnames), None)

        for row in reader:
            episodes.append(int(row["episode"]))
            # Phase 1 CSV: "reward"; self-play CSVs: "krishna_reward"
            reward_val = row.get("reward") or row.get("krishna_reward") or "0"
            rewards.append(float(reward_val))
            pellets.append(float(row["pellets"]))
            # Phase 1 CSV: "won"; self-play CSVs: "winner"
            won_val = row.get("won") or row.get("winner") or ""
            wins.append(1 if won_val in ("True", "true", "1", "krishna") else 0)
            for key, _candidates in _OPTIONAL_SERIES:
                col = resolved[key]
                val = row.get(col) if col else None
                extra[key].append(float(val) if val not in (None, "", "None") else float("nan"))

    out: dict[str, Any] = {
        "n_episodes": len(episodes),
        "episodes": _downsample(episodes),
        "rewards": _downsample(rewards),
        "pellets": _downsample(pellets),
        "wins": _downsample(wins),
    }
    for key, values in extra.items():
        if any(not math.isnan(v) for v in values):
            out[key] = _downsample(values)
    return out


def _safe_rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def _list_replays(run_dir: Path, base: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Enumerate replays/*.jsonl and read their header+footer for quick summary."""
    replays_dir = run_dir / "replays"
    if not replays_dir.exists():
        return []
    out = []
    for p in sorted(replays_dir.glob("episode_*.jsonl")):
        header = None
        footer = None
        try:
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("type") == "header":
                        header = obj
                    elif obj.get("type") == "footer":
                        footer = obj
        except Exception as e:
            out.append({"file": p.name, "error": str(e)})
            continue
        out.append({
            "file": p.name,
            "path": _safe_rel(p, base),
            "episode_id": header.get("episode_id") if header else None,
            "phase": header.get("phase") if header else None,
            "difficulty": header.get("difficulty") if header else None,
            "seed": header.get("seed") if header else None,
            "outcome": footer.get("outcome") if footer else "incomplete",
            "total_steps": footer.get("total_steps") if footer else None,
            "pellets_collected": footer.get("pellets_collected") if footer else None,
            "total_reward": footer.get("total_reward") if footer else None,
        })
    return out


def scan_one(run_dir: Path, base: Path = REPO_ROOT) -> dict[str, Any] | None:
    exp_path = run_dir / "experiment.json"
    if not exp_path.exists():
        # legacy runs without experiment.json — list minimally so they show up
        return {
            "id": run_dir.name,
            "legacy": True,
            "name": run_dir.name,
            "path": _safe_rel(run_dir, base),
            "phase_label": "legacy",
            "replays": _list_replays(run_dir, base),
            "summary": _read_log_summary(run_dir / "logs" / "episode_stats.csv"),
        }
    try:
        exp = json.loads(exp_path.read_text())
    except Exception as e:
        return {"id": run_dir.name, "error": f"bad experiment.json: {e}"}

    log_path = run_dir / "logs" / "episode_stats.csv"
    return {
        "id": exp.get("id", run_dir.name),
        "legacy": False,
        "name": exp.get("name", run_dir.name),
        "phase_label": exp.get("phase_label", ""),
        "path": _safe_rel(run_dir, base),
        "experiment_json": _safe_rel(exp_path, base),
        "algo": exp.get("algo", {}),
        "hyperparams": exp.get("hyperparams", {}),
        "training": exp.get("training", {}),
        "results": exp.get("results", {}),
        "notes": exp.get("notes", ""),
        "summary": _read_log_summary(log_path),
        "replays": _list_replays(run_dir, base),
    }


def build_index(existing_path: Path | None = None) -> dict[str, Any]:
    """Build the run index, MERGING with whatever's already at `existing_path`
    rather than replacing it wholesale.

    `training_runs/` is gitignored, so no local checkout has every run that
    was ever indexed -- only whoever happened to run it, and only until they
    clean up disk space. `dashboard/data/index.json` IS committed, precisely
    so the dashboard shows the full historical gallery for anyone who clones
    the repo. A naive rebuild-from-local-training_runs/ would silently
    delete every historical entry whose directory isn't present locally
    right now -- this nearly happened while testing v8/v7-ablations support
    (see the commit that introduced this merge behavior for the full story).

    Runs found locally always win over a stale indexed copy of the same id
    (so re-scanning a run you still have gets you fresh data); runs indexed
    previously but no longer present locally are kept as-is.
    """
    runs_by_id: dict[str, Any] = {}

    if existing_path and existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
            for r in existing.get("runs", []):
                if "id" in r:
                    runs_by_id[r["id"]] = r
        except (json.JSONDecodeError, OSError):
            pass

    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            entry = scan_one(d)
            if entry is not None:
                runs_by_id[entry["id"]] = entry

    runs = sorted(runs_by_id.values(), key=lambda r: r.get("id", ""), reverse=True)
    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_DIR / "index.json"))
    parser.add_argument("--no-merge", action="store_true",
                         help="Rebuild from local training_runs/ only, discarding any "
                              "historical entries not present locally. Rarely what you "
                              "want -- see build_index()'s docstring.")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    idx = build_index(existing_path=None if args.no_merge else out)
    # json.dumps with allow_nan=False would raise on -inf/inf; instead
    # sanitize the structure first so JS JSON.parse never sees bare Infinity.
    def _sanitize(obj):
        if isinstance(obj, float):
            import math
            return None if (math.isinf(obj) or math.isnan(obj)) else obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj
    out.write_text(json.dumps(_sanitize(idx), indent=2))
    print(f"Wrote {out} with {len(idx['runs'])} runs")


if __name__ == "__main__":
    main()
