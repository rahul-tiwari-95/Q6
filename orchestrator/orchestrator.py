"""
A small local process orchestrator for running several long training jobs on
one machine, with crash detection, checkpoint-aware auto-resume, and a live
status view.

Why this exists
----------------
Q6 has (at least) two experiments — v8's PPO port and the v7-ablations DQN
run — that people want to run *at the same time*, for many hours, on a
laptop, without babysitting a terminal. That means three ordinary but
easy-to-get-wrong things need to be true:

  1. If a job dies (crash, laptop sleep interrupting it badly, whatever),
     something notices and restarts it — resuming from its last checkpoint,
     not from scratch.
  2. Resuming a job means replaying its FULL original command line plus
     `--resume <run_dir>`, not just `--resume` on its own — hyperparameters
     are not restored from the checkpoint (see train_v8.py / train_phase3.py;
     this was found the hard way while testing resume for this project).
  3. Running N jobs on one machine shouldn't require babysitting N terminal
     tabs — one process should say what's going on.

Nothing here is Q6-specific. A job is just {id, cwd, command}. The only
Q6-shaped assumption is optional: if the job's own stdout prints a line
matching `run_dir=<path>` (as both train_v8.py and train_phase3.py already
do), the orchestrator will find it automatically and use it for resume.
Anything that doesn't print such a line just won't get auto-resume — it
still gets supervised, logged, and restarted (from scratch) on crash.

Config hot-reload on restart: the config file (`--config`) is re-read for a
job's entry right before each RESTART attempt (not the first launch, and not
on every poll cycle -- see Orchestrator._reload_job_spec). This means a fix
to a job's `cwd`/`command`/etc on disk takes effect on that job's next
restart even though the orchestrator process itself has been running the
whole time and never re-parsed the file at startup. Changes are printed
loudly when they're picked up, and a missing/malformed config file (or a job
whose id disappeared from it) falls back to the in-memory spec rather than
crashing or dropping the job -- also printed loudly, never silently. This
exists because of a real incident: `v8_ippo`'s `cwd` in `q6_jobs.json` was
wrong, and by the time it was fixed on disk the orchestrator was already
running with the broken spec loaded into memory -- every restart kept
replaying the same broken cwd, burning through all `max_restarts` in about
5 minutes, then sitting silently "failed" for the remaining ~45 hours of the
run while a second job in the same process ran fine. Nobody was watching
closely enough at minute 5 to catch it, and short of killing and restarting
the whole orchestrator (losing the other job's live progress too), there was
no way to get the fix into the running process. This closes that gap.

macOS note on "load balancing": there is no supported way to pin a Python
subprocess to specific CPU cores on macOS (psutil.Process.cpu_affinity()
raises NotImplementedError there; it's a Linux/Windows-only API). So this
does NOT do CPU-affinity pinning. What it does instead: a configurable cap
on concurrently-running jobs, and an optional `max_load_average` throttle
(os.getloadavg(), which IS available on macOS) that delays starting new
queued jobs while the system is already busy. That's the honest version of
"load-aware scheduling" achievable here.

Usage
-----
    python orchestrator/orchestrator.py --config orchestrator/q6_jobs.json

Ctrl-C sends SIGTERM to every running child (which, for both Q6 training
scripts, triggers their own graceful checkpoint-and-exit handler) and waits
briefly for them to exit before the orchestrator itself exits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_RUN_DIR_RE = re.compile(r"run_dir=(\S+)")
_EPISODE_LINE_RE = re.compile(r"^\s*ep\s+\d+")


# ---------------------------------------------------------------------------
# Job spec / runtime state
# ---------------------------------------------------------------------------

@dataclass
class JobSpec:
    id: str
    cwd: str
    command: List[str]
    max_restarts: int = 10
    restart_backoff_seconds: float = 10.0


VALID_STATUSES = {"pending", "running", "completed", "crashed", "stopped", "failed"}


@dataclass
class JobState:
    spec: JobSpec
    status: str = "pending"
    run_dir: Optional[str] = None
    restart_count: int = 0
    last_episode_line: str = ""
    started_at: Optional[float] = None
    next_restart_at: float = 0.0
    log_path: Optional[Path] = None
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _log_fh: Optional[Any] = field(default=None, repr=False)

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "id": self.spec.id,
            "status": self.status,
            "run_dir": self.run_dir,
            "restart_count": self.restart_count,
            "pid": self.proc.pid if (self.proc and self.status == "running") else None,
            "last_progress": self.last_episode_line,
            "log": str(self.log_path) if self.log_path else None,
            "command": self.spec.command,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(
        self,
        jobs: List[JobSpec],
        log_dir: Path,
        status_path: Path,
        poll_interval: float = 10.0,
        status_print_interval: float = 30.0,
        max_concurrent_jobs: Optional[int] = None,
        max_load_average: Optional[float] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.jobs = [JobState(spec=j) for j in jobs]
        self.log_dir = log_dir
        self.status_path = status_path
        self.poll_interval = poll_interval
        self.status_print_interval = status_print_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_load_average = max_load_average
        # Remembered so that _reload_job_spec() can re-read this job's entry
        # from disk before each restart (see below) -- without this, the
        # ONLY way to get a config fix into an already-running orchestrator
        # process is to kill and restart the whole thing, losing every other
        # job's live progress too. May be None (e.g. tests constructing an
        # Orchestrator directly from in-memory JobSpecs); hot-reload is
        # simply disabled in that case, same as before this feature existed.
        self.config_path = config_path
        self._stopping = False
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Resume across orchestrator restarts, not just mid-session crashes:
        # if a status file from a previous run exists, recover each job's
        # last-known run_dir so a job that was mid-flight when the whole
        # orchestrator (or the laptop) went down still resumes correctly,
        # rather than starting over.
        self._recover_from_prior_status()

    # ------------------------------------------------------------------
    # Startup recovery
    # ------------------------------------------------------------------

    def _recover_from_prior_status(self) -> None:
        if not self.status_path.exists():
            return
        try:
            prior = json.loads(self.status_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        prior_by_id = {j["id"]: j for j in prior.get("jobs", [])}
        for job in self.jobs:
            prev = prior_by_id.get(job.spec.id)
            if not prev:
                continue
            if prev.get("status") == "completed":
                job.status = "completed"
            elif prev.get("run_dir"):
                job.run_dir = prev["run_dir"]
                job.restart_count = int(prev.get("restart_count", 0))
                print(f"[orchestrator] recovered prior state for {job.spec.id}: "
                      f"run_dir={job.run_dir}  (will resume from checkpoint)")

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _build_command(self, job: JobState) -> List[str]:
        cmd = list(job.spec.command)
        if job.run_dir:
            cmd += ["--resume", job.run_dir]
        return cmd

    # ------------------------------------------------------------------
    # Config hot-reload (restarts only)
    # ------------------------------------------------------------------

    def _reload_job_spec(self, job: JobState) -> None:
        """Re-read this job's entry from the on-disk config file and, if it
        differs, swap it in as `job.spec` before a restart.

        Why this exists: the orchestrator only parses the config file once,
        at startup (see build_orchestrator()/__init__). If someone fixes a
        bug in the config (wrong cwd, wrong flag, ...) while the orchestrator
        is already running, an already-running process previously had no way
        to notice -- every restart kept replaying the stale JobSpec captured
        at construction time. That's exactly what happened to `v8_ippo`: its
        `cwd` was fixed on disk minutes after a broken orchestrator run
        started, but the running process never re-read it, burned through
        all `max_restarts` in ~5 minutes, and then sat silently "failed" for
        the remaining ~45 hours of the run.

        Only called for RESTARTS (see _start_job) -- a job's very first
        launch already uses a fresh-off-disk spec from __init__, so
        re-parsing again there would be redundant.

        Deliberately NOT called on every poll cycle: re-reading the config
        file that often is wasteful, and risks reading it mid-edit. Reading
        it once, right before we're about to act on it, is enough to close
        the "already fixed but a running process won't notice" gap.

        Fails open: if the config file is missing, malformed, or no longer
        has an entry for this job's id, we fall back to the in-memory spec
        (loud about it, not silent) rather than crashing the orchestrator or
        dropping the job.
        """
        if self.config_path is None:
            return
        try:
            fresh_config = json.loads(self.config_path.read_text())
            entries = {entry["id"]: entry for entry in fresh_config["jobs"]}
            new_spec = JobSpec(**entries[job.spec.id])
        except FileNotFoundError:
            print(f"[orchestrator] {job.spec.id}: config file {self.config_path} is "
                  f"missing -- falling back to the in-memory spec for this restart.")
            return
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[orchestrator] {job.spec.id}: could not reload config from "
                  f"{self.config_path} ({exc!r}) -- falling back to the in-memory "
                  f"spec for this restart.")
            return

        changes = []
        if new_spec.cwd != job.spec.cwd:
            changes.append(f"cwd was {job.spec.cwd!r}, now {new_spec.cwd!r}")
        if new_spec.command != job.spec.command:
            changes.append(f"command was {job.spec.command!r}, now {new_spec.command!r}")
        if new_spec.max_restarts != job.spec.max_restarts:
            changes.append(f"max_restarts was {job.spec.max_restarts!r}, "
                            f"now {new_spec.max_restarts!r}")
        if new_spec.restart_backoff_seconds != job.spec.restart_backoff_seconds:
            changes.append(f"restart_backoff_seconds was {job.spec.restart_backoff_seconds!r}, "
                            f"now {new_spec.restart_backoff_seconds!r}")
        if changes:
            print(f"[orchestrator] {job.spec.id}: config changed on reload -- "
                  + "; ".join(changes))
        job.spec = new_spec

    # ------------------------------------------------------------------
    # Starting / polling jobs
    # ------------------------------------------------------------------

    def _start_job(self, job: JobState) -> None:
        # A restart (as opposed to this job's very first launch) -- re-read
        # its config entry from disk first, in case it was fixed/changed
        # while this orchestrator process has been running. run_dir being
        # set is the common case (crash happened after the job logged
        # run_dir=...); restart_count > 0 also catches a crash that happened
        # before the job ever got that far.
        if job.run_dir is not None or job.restart_count > 0:
            self._reload_job_spec(job)

        job.log_path = self.log_dir / f"{job.spec.id}.log"
        job._log_fh = open(job.log_path, "a", buffering=1)
        cmd = self._build_command(job)
        mode = "resume" if job.run_dir else "fresh start"
        job._log_fh.write(
            f"\n===== orchestrator launching ({mode}) at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
            f"cwd={job.spec.cwd}\ncommand={cmd}\n\n"
        )
        job._log_fh.flush()
        print(f"[orchestrator] starting {job.spec.id} ({mode}): {' '.join(cmd)}")
        try:
            job.proc = subprocess.Popen(
                cmd, cwd=job.spec.cwd, stdout=job._log_fh, stderr=subprocess.STDOUT,
            )
        except (OSError, FileNotFoundError) as exc:
            print(f"[orchestrator] FAILED to launch {job.spec.id}: {exc}")
            job.status = "failed"
            return
        job.status = "running"
        job.started_at = time.time()
        print(f"[orchestrator] {job.spec.id} pid={job.proc.pid}")

    def _extract_run_dir(self, job: JobState) -> None:
        if job.run_dir or not job.log_path or not job.log_path.exists():
            return
        try:
            text = job.log_path.read_text(errors="ignore")
        except OSError:
            return
        m = _RUN_DIR_RE.search(text)
        if m:
            job.run_dir = m.group(1)

    def _extract_progress(self, job: JobState) -> None:
        if not job.log_path or not job.log_path.exists():
            return
        try:
            # Tail-read: only the last ~8KB, not the whole (possibly large)
            # log file, every poll cycle.
            with job.log_path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                tail = f.read().decode(errors="ignore")
        except OSError:
            return
        for line in reversed(tail.splitlines()):
            if _EPISODE_LINE_RE.match(line):
                job.last_episode_line = line.strip()
                break

    def _poll_job(self, job: JobState) -> None:
        if job.status != "running":
            return
        self._extract_run_dir(job)
        self._extract_progress(job)
        rc = job.proc.poll()
        if rc is None:
            return
        if job._log_fh:
            job._log_fh.close()
            job._log_fh = None
        if rc == 0:
            job.status = "completed"
            print(f"[orchestrator] {job.spec.id} completed successfully.")
            return
        if self._stopping:
            # Don't schedule a restart while shutting down -- just record what
            # actually happened. Without this guard, a job that crashes (or
            # finishes exiting from its own SIGTERM handler) in the narrow
            # window during shutdown would otherwise get silently scheduled
            # for a restart that never runs, or worse, left showing "running"
            # with a stale PID in the final status (see _handle_shutdown).
            job.status = "stopped"
            print(f"[orchestrator] {job.spec.id} exited with code {rc} during shutdown.")
            return
        job.restart_count += 1
        if job.restart_count > job.spec.max_restarts:
            job.status = "failed"
            print(f"[orchestrator] {job.spec.id} exited with code {rc} and exceeded "
                  f"max_restarts={job.spec.max_restarts} — giving up. See {job.log_path}")
            return
        print(f"[orchestrator] {job.spec.id} exited with code {rc} "
              f"(restart {job.restart_count}/{job.spec.max_restarts}); "
              f"will restart from its checkpoint in {job.spec.restart_backoff_seconds:.0f}s.")
        job.status = "pending"
        job.next_restart_at = time.time() + job.spec.restart_backoff_seconds

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _load_average_ok(self) -> bool:
        if self.max_load_average is None:
            return True
        try:
            load1, _load5, _load15 = os.getloadavg()
        except (OSError, AttributeError):
            return True  # not available on this platform; don't block on it
        return load1 <= self.max_load_average

    def _schedule(self) -> None:
        now = time.time()
        running_count = sum(1 for j in self.jobs if j.status == "running")
        pending = [j for j in self.jobs if j.status == "pending" and now >= j.next_restart_at]
        if not pending:
            return
        if not self._load_average_ok():
            return
        for job in pending:
            if self.max_concurrent_jobs is not None and running_count >= self.max_concurrent_jobs:
                break
            self._start_job(job)
            if job.status == "running":
                running_count += 1

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _write_status(self) -> None:
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "jobs": [j.to_status_dict() for j in self.jobs],
        }
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.status_path)  # atomic-ish on POSIX

    def _print_status_table(self) -> None:
        print(f"\n[orchestrator] status @ {time.strftime('%H:%M:%S')}")
        for job in self.jobs:
            progress = job.last_episode_line or "—"
            print(f"  {job.spec.id:<20} {job.status:<10} restarts={job.restart_count:<3} "
                  f"{progress}")

    def _all_terminal(self) -> bool:
        return all(j.status in ("completed", "failed", "stopped") for j in self.jobs)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _handle_shutdown(self, signum, frame) -> None:
        if self._stopping:
            return  # already shutting down; a second Ctrl-C won't re-enter this
        self._stopping = True
        print("\n[orchestrator] shutdown requested — sending SIGTERM to running jobs "
              "so they can checkpoint...")
        # Snapshot of what looked "running" at the instant the signal arrived.
        # A fast-crash-looping job can exit on its own between this snapshot
        # and the reconciliation pass below -- that's expected and handled,
        # not a bug (see the reconciliation loop's comment).
        running = [j for j in self.jobs if j.status == "running" and j.proc and j.proc.poll() is None]
        for job in running:
            job.proc.send_signal(signal.SIGTERM)
        deadline = time.time() + 60
        for job in running:
            remaining = max(0.0, deadline - time.time())
            try:
                job.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[orchestrator] {job.spec.id} did not exit in time — killing.")
                job.proc.kill()
                job.proc.wait()

        # Reconciliation pass over EVERY job still marked "running" (not just
        # the snapshot above): _poll_job now sees self._stopping=True, so it
        # records the real exit code as "stopped" instead of scheduling a
        # restart. Without this, a job that crashed/exited in the narrow
        # window between the snapshot and now would be left showing "running"
        # with a stale, already-dead PID in the final status.
        for job in self.jobs:
            if job.status == "running":
                self._poll_job(job)
            elif job.status == "pending":
                job.status = "stopped"
            if job._log_fh:
                job._log_fh.close()
                job._log_fh = None
        self._write_status()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        print(f"[orchestrator] pid={os.getpid()}  managing {len(self.jobs)} job(s): "
              f"{', '.join(j.spec.id for j in self.jobs)}")
        print(f"[orchestrator] logs: {self.log_dir}   status: {self.status_path}")
        print("[orchestrator] each job's own pid is also printed as it starts, and "
              "always readable from the status file above (jobs[].pid) while running.")

        last_status_print = 0.0
        while not self._stopping and not self._all_terminal():
            self._schedule()
            for job in self.jobs:
                self._poll_job(job)
            self._write_status()
            if time.time() - last_status_print >= self.status_print_interval:
                self._print_status_table()
                last_status_print = time.time()
            time.sleep(self.poll_interval)

        self._write_status()
        self._print_status_table()
        if self._all_terminal() and not self._stopping:
            print("\n[orchestrator] all jobs reached a terminal state. Exiting.")


# ---------------------------------------------------------------------------
# Config loading / CLI
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def build_orchestrator(
    config: Dict[str, Any],
    log_dir: Path,
    status_path: Path,
    config_path: Optional[Path] = None,
) -> Orchestrator:
    job_specs = [JobSpec(**j) for j in config["jobs"]]
    return Orchestrator(
        jobs=job_specs,
        log_dir=log_dir,
        status_path=status_path,
        poll_interval=config.get("poll_interval_seconds", 10.0),
        status_print_interval=config.get("status_print_interval_seconds", 30.0),
        max_concurrent_jobs=config.get("max_concurrent_jobs"),
        max_load_average=config.get("max_load_average"),
        config_path=config_path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                    help="Path to a JSON job config (see orchestrator/q6_jobs.json for an example).")
    p.add_argument("--log-dir", type=Path, default=None,
                    help="Directory for per-job log files (default: <config dir>/orchestrator_logs)")
    p.add_argument("--status-file", type=Path, default=None,
                    help="Path to the live status JSON file (default: <config dir>/orchestrator_status.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    log_dir = args.log_dir or (args.config.parent / "orchestrator_logs")
    status_path = args.status_file or (args.config.parent / "orchestrator_status.json")
    orch = build_orchestrator(config, log_dir, status_path, config_path=args.config)
    orch.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
