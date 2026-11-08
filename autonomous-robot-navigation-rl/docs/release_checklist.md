# Core release acceptance

Historical milestone-6 acceptance record (6 November 2026). Test counts and
observations below belong to that frozen release. PPO, later UI changes and
current setup checks are tracked separately in [plan closure](plan_closure_status.md).

- [x] Versioned fixed map, deterministic scenario manifests and shared environment.
- [x] Tested endings/rewards, tabular updates, timeout bootstrap, metrics and model loading.
- [x] Random/BFS/Q-learning/SARSA/DQN and five-seed equal-budget Part A comparison.
- [x] Bounded shared search, fresh matched confirmations, controlled reward ablation.
- [x] Frozen vote with component comparison and memory/training/latency costs.
- [x] Predefined fixed-map robustness, all seeds and negative findings retained.
- [x] Animated viewer, synchronized diagnostics and matched/early comparisons.
- [x] Live success/failure/terminal-holding check; 64 automated tests.
- [x] Isolated preparation/training reproduction: table and validation metrics identical.
- [x] Final source/models/analysis frozen before holdout; all 42 records evaluated.
- [x] 38,976 episodes and 3,456,940 actions audited; BFS all optimal.
- [x] Final selection unchanged; audited aggregate results visible in the dashboard.
- [x] Four final figures inspected; technical report, run matrix and Windows instructions.
- [x] Exact frozen restore verifies 184 files; archive protected against newline conversion.
- [x] 6,763 pre-milestone-6 root/data/artifact files unchanged.
- [x] No public deployment, share link, push or automatic commit.

Known limits: familiar map only, poor SARSA/DQN and residual refined-policy
failures, one ensemble, small seed counts, no clean network dependency installation
test, no cross-hardware bitwise guarantee, no exact training resume. Deferred scope
remains deferred except for the later owner-authorized PPO validation extension.
New final-result-driven model work needs a new holdout or explicit
exploratory labeling. The current dashboard reports four trained RL families;
its historical final-evaluation panel and this release record identify completion
of milestone 6.
