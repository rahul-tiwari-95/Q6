"""Verify shipped pilot hashes and adaptation aggregates without retraining.

This checks internal integrity, not scientific validity or independent replication.
Run from any directory: python /path/to/Q6/scripts/verify_pilot_artifacts.py
"""

from collections import defaultdict
import csv
import hashlib
import json
import math
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


def main():
    for version in ("v1", "v2"):
        verify_adaptation(ROOT / "experiments/adaptation" / f"pilot_{version}")
    folder = ROOT / "experiments/provenance/release-pilot-v1"
    manifest = read_json(folder / "manifest.json")
    if manifest["algorithm"] != "sha256":
        raise ValueError("Unsupported manifest algorithm")
    for name, digest in manifest["files"].items():
        check_hash(folder / name, digest)
    print(f"Provenance: all {len(manifest['files'])} manifest files verified")


if __name__ == "__main__":
    main()
