# Q6 Repo — Audit Fix Plan
# Created: 2026-05-29

- [x] 1. Fix A* wall-walking bug in hunter_progressive.py (changed != 1 → != 0 in both `get_valid_moves` and `a_star_search`)
- [x] 2. Fix double epsilon decay in main.py (removed redundant inline decay at line ~890)
- [x] 3. Fix .gitignore (added training_runs/, plots/, node_modules/) — removed 932 cached files from git
- [x] 4. Archive dead code into archive/ folder (8 old scripts, 2145 lines removed from root)
- [x] 5. Remove Node.js artifacts (package.json, package-lock.json, node_modules/)
- [x] 6. Fix README inconsistencies (step penalty documented as -0.01, actual is -0.001 — corrected)
- [x] 7. Add config.py (centralized constants — DQN hparams, rewards, grid values, curriculum)
- [x] 8. Add pytest test suite (tests/test_config.py, test_environment.py, test_hunter.py — 60 tests, all passing)
- [x] 9. Fix n_episodes/phase_episodes mismatch (was 15000 vs 18000 — fixed to 18000)
- [x] 10. Update TODO and verify everything works
