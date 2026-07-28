"""
Tests for orchestrator/orchestrator.py.

Two levels: pure unit tests for the small stateless pieces (command
construction, log parsing), and integration tests that spawn real short-lived
subprocesses through the orchestrator's own scheduling/polling methods (not
the full blocking run() loop with signal handlers, so tests stay fast and
don't fight pytest's own signal handling) to verify crash detection and
resume-flag construction actually work end to end against a real process.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from orchestrator.orchestrator import JobSpec, JobState, Orchestrator


# ---------------------------------------------------------------------------
# Unit tests — pure logic, no subprocesses
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def test_fresh_job_has_no_resume_flag(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"))
        job = JobState(spec=JobSpec(id="a", cwd=".", command=["python3", "s.py", "--x", "1"]))
        assert orch._build_command(job) == ["python3", "s.py", "--x", "1"]

    def test_resuming_job_appends_resume_dir_after_original_flags(self):
        # This is the exact footgun found while testing v8/v7-ablations resume:
        # hyperparameters are NOT restored from the checkpoint, so the ORIGINAL
        # flags must be replayed, with --resume appended, not substituted.
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"))
        job = JobState(
            spec=JobSpec(id="a", cwd=".", command=["python3", "s.py", "--rollout-len", "2048"]),
            run_dir="training_runs/20260101_000000_a",
        )
        cmd = orch._build_command(job)
        assert cmd == [
            "python3", "s.py", "--rollout-len", "2048",
            "--resume", "training_runs/20260101_000000_a",
        ]


class TestExtractRunDir:
    def test_finds_run_dir_line(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text("[start] 100 episodes  device=cpu  run_dir=/foo/bar/baz\nmore text\n")
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=log)
        orch._extract_run_dir(job)
        assert job.run_dir == "/foo/bar/baz"

    def test_does_not_overwrite_existing_run_dir(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text("run_dir=/should/not/be/used\n")
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=log, run_dir="/already/set")
        orch._extract_run_dir(job)
        assert job.run_dir == "/already/set"

    def test_missing_log_file_is_a_noop(self, tmp_path):
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=tmp_path / "nope.log")
        orch._extract_run_dir(job)  # must not raise
        assert job.run_dir is None


class TestExtractProgress:
    def test_finds_last_episode_line(self, tmp_path):
        log = tmp_path / "job.log"
        log.write_text(
            "[start] ...\n"
            "  ep    10/100  mode=joint  steps=1000  avg100=1.23\n"
            "  ep    20/100  mode=fsp    steps=1000  avg100=4.56\n"
        )
        orch = Orchestrator(jobs=[], log_dir=tmp_path, status_path=tmp_path / "s.json")
        job = JobState(spec=JobSpec(id="a", cwd=".", command=[]), log_path=log)
        orch._extract_progress(job)
        assert "ep    20/100" in job.last_episode_line


class TestLoadAverageThrottle:
    def test_none_means_always_ok(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"),
                             max_load_average=None)
        assert orch._load_average_ok() is True

    def test_very_high_threshold_is_ok(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"),
                             max_load_average=10_000.0)
        assert orch._load_average_ok() is True

    def test_zero_threshold_blocks(self):
        orch = Orchestrator(jobs=[], log_dir=Path("/tmp/x"), status_path=Path("/tmp/x/s.json"),
                             max_load_average=-1.0)
        assert orch._load_average_ok() is False


# ---------------------------------------------------------------------------
# Integration tests — real short-lived subprocesses
# ---------------------------------------------------------------------------

_FAKE_SUCCESS = """
import sys
print("[start] fake job  run_dir={}".format(sys.argv[1]), flush=True)
print("  ep 1/1  fake progress line", flush=True)
sys.exit(0)
"""

_FAKE_CRASH_THEN_RESUME_SUCCEEDS = """
import sys
run_dir = sys.argv[1]
print("[start] fake job  run_dir={}".format(run_dir), flush=True)
if "--resume" in sys.argv:
    print("  ep 2/2  resumed and finished", flush=True)
    sys.exit(0)
else:
    print("  ep 1/2  about to crash", flush=True)
    sys.exit(1)
"""


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def _poll_until(orch: Orchestrator, job: JobState, predicate, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        orch._schedule()
        orch._poll_job(job)
        if predicate(job):
            return
        time.sleep(0.05)
    pytest.fail(f"condition not met within {timeout}s; last status={job.status}")


class TestIntegrationRealSubprocess:
    def test_successful_job_reaches_completed(self, tmp_path):
        script = _write_script(tmp_path, "fake_success.py", _FAKE_SUCCESS)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run")
        spec = JobSpec(id="succeeds", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        _poll_until(orch, job, lambda j: j.status == "completed")

        assert job.run_dir == run_dir_arg
        assert "fake progress line" in job.last_episode_line
        assert job.restart_count == 0

    def test_crash_triggers_restart_with_resume_flag_and_then_succeeds(self, tmp_path):
        script = _write_script(tmp_path, "fake_crash.py", _FAKE_CRASH_THEN_RESUME_SUCCEEDS)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run2")
        spec = JobSpec(id="crashes_once", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg],
                        restart_backoff_seconds=0.1, max_restarts=3)
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        # First launch: crashes (exit 1). Orchestrator should mark it pending
        # for restart, having already captured run_dir from its stdout.
        _poll_until(orch, job, lambda j: j.status == "pending" and j.restart_count == 1)
        assert job.run_dir == run_dir_arg

        # Backoff elapses, orchestrator restarts it — this time WITH --resume,
        # which the fake script uses to decide to succeed instead of crash.
        _poll_until(orch, job, lambda j: j.status == "completed", timeout=10.0)
        assert job.restart_count == 1  # only needed one restart

    def test_exceeding_max_restarts_marks_failed(self, tmp_path):
        # A script that always crashes, regardless of --resume.
        always_crash = "import sys\nprint('run_dir=%s' % sys.argv[1], flush=True)\nsys.exit(1)\n"
        script = _write_script(tmp_path, "fake_always_crash.py", always_crash)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run3")
        spec = JobSpec(id="always_crashes", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg],
                        restart_backoff_seconds=0.05, max_restarts=2)
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        _poll_until(orch, job, lambda j: j.status == "failed", timeout=10.0)
        assert job.restart_count == 3  # initial attempt + 2 retries, then gives up

    def test_status_file_is_written_and_json_valid(self, tmp_path):
        script = _write_script(tmp_path, "fake_success2.py", _FAKE_SUCCESS)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run4")
        spec = JobSpec(id="succeeds2", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg])
        status_path = tmp_path / "status.json"
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=status_path)
        job = orch.jobs[0]

        _poll_until(orch, job, lambda j: j.status == "completed")
        orch._write_status()

        payload = json.loads(status_path.read_text())
        assert payload["jobs"][0]["id"] == "succeeds2"
        assert payload["jobs"][0]["status"] == "completed"

    def test_recovers_run_dir_from_prior_status_file(self, tmp_path):
        # Simulate a previous orchestrator session having written a status
        # file with a job mid-flight, then a fresh Orchestrator instance
        # (as if the whole process/machine restarted) picking it back up.
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({
            "updated_at": "x",
            "jobs": [{"id": "recovering", "status": "running", "run_dir": "/prior/run/dir",
                      "restart_count": 2, "pid": None, "last_progress": "", "log": None,
                      "command": []}],
        }))
        spec = JobSpec(id="recovering", cwd=".", command=["python3", "whatever.py"])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=status_path)

        assert orch.jobs[0].run_dir == "/prior/run/dir"
        assert orch.jobs[0].restart_count == 2

    def test_does_not_recover_a_completed_job_as_needing_resume(self, tmp_path):
        status_path = tmp_path / "status.json"
        status_path.write_text(json.dumps({
            "updated_at": "x",
            "jobs": [{"id": "done_already", "status": "completed", "run_dir": "/prior/run/dir",
                      "restart_count": 0, "pid": None, "last_progress": "", "log": None,
                      "command": []}],
        }))
        spec = JobSpec(id="done_already", cwd=".", command=["python3", "whatever.py"])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=status_path)

        assert orch.jobs[0].status == "completed"

    def test_shutdown_reconciles_a_job_that_crashed_in_the_signal_race_window(self, tmp_path):
        # Regression test for a real bug found via a live end-to-end run:
        # a job that crashes near-instantly (fast crash loop) can exit on its
        # own in the narrow window between _handle_shutdown's initial
        # "running" snapshot and the reconciliation pass -- without the fix,
        # it's left reported as "running" with an already-dead PID forever.
        instant_crash = "import sys\nprint('run_dir=%s' % sys.argv[1], flush=True)\nsys.exit(1)\n"
        script = _write_script(tmp_path, "fake_instant_crash.py", instant_crash)
        run_dir_arg = str(tmp_path / "training_runs" / "fake_run5")
        spec = JobSpec(id="crashes_instantly", cwd=str(tmp_path),
                        command=[sys.executable, str(script), run_dir_arg])
        orch = Orchestrator(jobs=[spec], log_dir=tmp_path / "logs", status_path=tmp_path / "status.json")
        job = orch.jobs[0]

        orch._start_job(job)
        assert job.status == "running"
        job.proc.wait(timeout=5)  # let the real process actually finish exiting on its own
        assert job.proc.poll() is not None  # confirm it's genuinely dead already

        # _handle_shutdown's own "running" snapshot will exclude this job
        # (poll() is no longer None), so its ONLY path to a correct final
        # status is the reconciliation pass.
        orch._handle_shutdown(signum=15, frame=None)

        assert job.status == "stopped"
        assert job.status != "running"
