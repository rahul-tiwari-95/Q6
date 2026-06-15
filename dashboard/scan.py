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


def _read_log_summary(log_path: Path) -> dict[str, Any]:
    """Read episode_stats.csv and return downsampled series for charting."""
    if not log_path.exists():
        return {}
    episodes: list[int] = []
    rewards: list[float] = []
    pellets: list[float] = []
    wins: list[int] = []
    epsilons: list[float] = []
    losses: list[float] = []
    with open(log_path) as fh:
        for row in csv.DictReader(fh):
            episodes.append(int(row["episode"]))
            # Phase 1 CSV: "reward"; Phase 2 CSV: "krishna_reward"
            reward_val = row.get("reward") or row.get("krishna_reward") or "0"
            rewards.append(float(reward_val))
            pellets.append(float(row["pellets"]))
            # Phase 1 CSV: "won"; Phase 2 CSV: "winner"
            won_val = row.get("won") or row.get("winner") or ""
            wins.append(1 if won_val in ("True", "true", "1", "krishna") else 0)
            # Phase 1 CSV: "epsilon"; Phase 2 CSV: "krishna_eps"
            eps_val = row.get("epsilon") or row.get("krishna_eps") or "0"
            epsilons.append(float(eps_val))
            # Phase 1 CSV: "loss"; Phase 2 CSV: "krishna_loss"
            loss_val = row.get("loss") or row.get("krishna_loss") or ""
            losses.append(float(loss_val) if loss_val not in (None, "", "None") else float("nan"))
    return {
        "n_episodes": len(episodes),
        "episodes": _downsample(episodes),
        "rewards": _downsample(rewards),
        "pellets": _downsample(pellets),
        "wins": _downsample(wins),
        "epsilons": _downsample(epsilons),
        "losses": _downsample(losses),
    }


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


def build_index() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            entry = scan_one(d)
            if entry is not None:
                runs.append(entry)
    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_DIR / "index.json"))
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    idx = build_index()
    out = Path(args.out)
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
