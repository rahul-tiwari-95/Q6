# Local training-run orchestrator

A small, dependency-free (stdlib only) Python daemon for running several
long-lived processes on one machine, with crash detection, checkpoint-aware
auto-resume, and a live status view — built for running Q6's `v8` and
`v7-ablations` experiments in parallel on a laptop without babysitting two
terminal tabs for days, but nothing about it is Q6-specific. A job is just
`{id, cwd, command}`.

## Why

Three ordinary things need to be true for an unattended multi-day run to be
safe to walk away from:

1. If a job dies, something notices and restarts it **from its last
   checkpoint**, not from scratch.
2. Resuming correctly means replaying the job's **full original command
   line** plus `--resume <run_dir>` — not just `--resume` on its own.
   Hyperparameters (rollout length, snapshot cadence, etc.) are not restored
   from the checkpoint file; this was found the hard way while testing
   resume for this project (see `Q6.md` section 4 for the broader pattern —
   this project takes "verify before you trust it" seriously).
3. Running N jobs on one machine shouldn't require N terminal tabs.

## Usage

```bash
python3 orchestrator/orchestrator.py --config orchestrator/q6_jobs.json
```

Runs in the foreground. Ctrl-C sends `SIGTERM` to every running child (which,
for both Q6 training scripts, triggers their own graceful
checkpoint-and-exit handler) and waits up to 60s for clean exits before the
orchestrator itself exits.

Per-job output streams to `orchestrator_logs_<config stem>/<job_id>.log`. A
live `orchestrator_status_<config stem>.json` is rewritten every poll cycle
with each job's status, restart count, run directory, and last-seen episode
progress line — useful if you want to build a dashboard tile on top of it
later, or just `cat` it from another terminal. Both paths default to being
keyed off the config file's own stem (e.g. `q6_jobs.json` →
`orchestrator_status_q6_jobs.json`) rather than just its directory — see
"Running multiple orchestrators at once" below for why — and can be
overridden with `--log-dir`/`--status-file`.

## Job config format

```json
{
  "poll_interval_seconds": 15,
  "status_print_interval_seconds": 60,
  "max_concurrent_jobs": null,
  "max_load_average": null,
  "jobs": [
    {
      "id": "some_job",
      "cwd": "/path/to/run/it/from",
      "command": ["python3", "script.py", "--flag", "value"],
      "max_restarts": 20,
      "restart_backoff_seconds": 15.0
    }
  ]
}
```

- `max_concurrent_jobs`: cap on how many jobs run at once; `null` = no cap
  (start everything immediately). Extra jobs queue and start as slots free
  up.
- `max_load_average`: if set, new (queued or restarting) jobs won't be
  started while `os.getloadavg()`'s 1-minute figure exceeds this. `null` =
  no throttle.
- `max_restarts` / `restart_backoff_seconds`: after a job crashes this many
  times, it's marked `failed` and left alone rather than restart-looping
  forever on something that isn't going to self-heal (e.g. a real bug).

Auto-resume is automatic and requires no config: if a job's stdout contains
a line matching `run_dir=<path>` (both `train_v8.py` and `train_phase3.py`
already print this on startup), the orchestrator remembers it and appends
`--resume <that path>` when relaunching after a crash. A job that doesn't
print such a line still gets supervised/logged/restarted — just always from
scratch, not resumed.

**The orchestrator itself is resumable too.** If `orchestrator_status.json`
exists from a previous session when you start it again, each job's
last-known `run_dir` is recovered from it — so if the whole machine went
down (not just one job), starting the orchestrator again picks every
still-in-progress job back up from its checkpoint, not from zero.

## Config hot-reload on restart

The config file is only fully parsed once, at startup — but a job's own
entry (`cwd`, `command`, `max_restarts`, `restart_backoff_seconds`) is
**re-read from disk right before each restart attempt** for that job (not on
every poll cycle, and not for a job's very first launch — see
`Orchestrator._reload_job_spec` in `orchestrator.py`). If it changed, the
job switches to the fresh values for that restart and prints something like:

```
[orchestrator] v8_ippo: config changed on reload -- cwd was '/Users/rahul/Q6', now '/Users/rahul/Q6-v8'
```

This exists because of a real incident: `v8_ippo`'s `cwd` in `q6_jobs.json`
was wrong (pointed at the main repo instead of its dedicated worktree). By
the time it was fixed on disk, the orchestrator process was already running
with the broken spec loaded into memory — every restart kept replaying the
same broken cwd, burned through all `max_restarts` in about 5 minutes, and
then sat silently `failed` for the rest of a ~45 hour run, while a second
job in the same process ran fine to completion. Nobody was watching closely
enough at minute 5 to catch it, and short of killing and restarting the
whole orchestrator (interrupting every other job's live progress too),
there was previously no way to get a config fix into an already-running
process.

If the config file is missing, malformed, or no longer has an entry for a
given job's id at restart time, the orchestrator falls back to that job's
in-memory spec rather than crashing or dropping the job — and says so
explicitly in its output, so a broken config never fails silently either.

## Ctrl-C'd jobs are marked `stopped`, not `completed`

Both `train_v8.py` and `train_phase3.py` exit with code `0` when they receive
`SIGTERM`/`SIGINT`, after checkpointing — that's what makes a clean shutdown
possible. But it means exit code `0` alone can't distinguish "finished all
its episodes" from "was interrupted mid-run and checkpointed on the way
out." The orchestrator checks whether *it* is the one that requested the
shutdown before trusting `rc == 0` as "completed" — if the process exited `0`
while the orchestrator was already tearing everything down, the job is
marked `stopped` instead, with a note to check its last logged episode.
Without this, a job Ctrl-C'd partway through (e.g. at episode 2000 of a
6000-episode run) would be permanently and silently mislabeled
`completed successfully`, and nothing would prompt you to resume it. To
actually finish such a job, just start the orchestrator again with the same
config — auto-resume (above) picks it back up from its last checkpoint.

## Running multiple orchestrators at once

If two config files live in the same directory (e.g. `q6_jobs.json` and
`q6_jobs_v2.json`, both under `orchestrator/`), each orchestrator instance
gets its own default log directory and status file, keyed off the config's
stem (`orchestrator_status_q6_jobs.json` vs.
`orchestrator_status_q6_jobs_v2.json`). This exists because of a second real
incident, found immediately after the one above: with both configs
defaulting to the exact same `orchestrator_status.json`, running a second
orchestrator instance while a first was already live silently overwrote the
first's status entries — a genuinely-still-running job's entry vanished from
the file entirely, clobbered by the second process's next write, even
though the job itself kept running fine underneath. See `default_paths()`
in `orchestrator.py`.

## macOS note on "load balancing"

There is no supported way to pin a subprocess to specific CPU cores on
macOS — `psutil.Process.cpu_affinity()` raises `NotImplementedError` there;
it's a Linux/Windows-only API, and this tool doesn't attempt to work around
that. What it does instead is honest about what's actually available:
a concurrency cap (`max_concurrent_jobs`) and an optional load-average
throttle (`max_load_average`, via `os.getloadavg()`, which **is** available
on macOS). If you need real CPU isolation, that's an OS-level (or
containers/cgroups-on-Linux) problem this tool doesn't solve.

## Enabling the frozen-anchor ablation for `v7_ablations`

The shipped `q6_jobs.json` runs `v7_ablations` with ablation 1 (rectified
opponent sampling) on by default and ablation 2 (frozen-collect-head anchor)
off, because ablation 2 needs a Phase-1 checkpoint path that isn't stable
across machines/regenerations. To enable it, add to that job's `command`:

```json
"--anchor-checkpoint", "/path/to/training_runs/<run>/checkpoints/agent_final.pth",
"--anchor-weight", "0.5",
"--anchor-decay-eps", "3000"
```

See `Q6.md` section 6.1 for what this checkpoint needs to represent and why
a properly-converged one (not a quick smoke-scale one) matters for a real
result.

## Extending beyond Q6

Nothing in `orchestrator.py` imports anything from this repo. Point
`cwd`/`command` at any long-running script that (a) exits 0 on success,
nonzero on failure, and (b) optionally prints `run_dir=<path>` and accepts
`--resume <path>`, and it's supervised the same way.
