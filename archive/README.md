# Archive

Old training scripts and one-off analysis tools from previous versions.
Preserved here for reference but not imported anywhere in the current codebase.

| File | Version | Purpose |
|------|---------|---------|
| `hunter_version_1.py` | v0.1 | Original training script (all enemies from start, agent learned nothing) |
| `train_curriculum.py` | v0.3 | First curriculum learning script (10K episodes) |
| `quick_train.py` | v0.2 | Quick training for rapid prototyping |
| `analyze_results.py` | v0.2 | Basic post-training analysis |
| `analyze_v04_detailed.py` | v0.4 | Detailed v0.4 results analysis |
| `analyze_v04_simple.py` | v0.4 | Quick v0.4 results summary |
| `debug_episodes.py` | v0.2 | Episode-by-episode debug viewer |
| `verify_fix.py` | v0.2 | Verified bugfixes applied |

Current codebase uses: `main.py` (v0.5 training), `utils/logger.py`, `utils/success_scoring.py`
