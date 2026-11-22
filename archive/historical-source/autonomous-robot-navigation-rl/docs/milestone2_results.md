# Milestone 2 — first learned navigation

Implemented Q-learning and SARSA through one training loop, independent read-only
evaluation, algorithm-aware checkpoint loading, and saved replay/custom frozen-policy
demonstrations for either learner. No DQN, tuning, voting or deferred extensions
were added. The original benchmark and milestone-1 artifacts remain unchanged.

## Measured development runs

The [protocol](milestone2_protocol.md) and configuration were saved before the runs.
Each learner received 200,000 interactions per seed, alpha .3, gamma .99 and the
same epsilon schedule (1 → .1 over 80% of budget). No best-seed or checkpoint
selection took place. Monitoring used the existing 30 validation scenarios every
10,000 interactions; final-budget evaluation used all 928 validation scenarios.

| Algorithm | Seed | Success | Collision | Timeout | SPL |
|---|---:|---:|---:|---:|---:|
| Q-learning | 7 | 616 | 2 | 310 | 0.632962 |
| Q-learning | 19 | 621 | 0 | 307 | 0.637938 |
| Q-learning | 43 | 613 | 1 | 314 | 0.624182 |
| SARSA | 7 | 255 | 15 | 658 | 0.259340 |
| SARSA | 19 | 222 | 25 | 681 | 0.224857 |
| SARSA | 43 | 238 | 18 | 672 | 0.239962 |

Each row contains 928 completed validation episodes. Across the three independent
seeds, Q-learning success averaged **66.45%**, sample SD **0.44 percentage points**;
SARSA averaged **25.68%**, sample SD **1.78 percentage points**. Mean SPL was
0.631694 (SD .006965) and .241387 (SD .017286), respectively. Repeated seed-7
runs are excluded from these averages and uncertainty summaries.

These settings favor Q-learning in this preliminary tabular development matrix;
this is not the full three-agent Part A study or evidence of universal superiority.
The earlier Q-learning 8.30% result used only 30,000 interactions, so its improvement
cannot be attributed to an algorithm change under an equal budget. The task and
rewards were not changed to improve the results.

Timeouts remain common: mean timeout rates are 33.44% for Q-learning and 72.23% for
SARSA. On the 320 long-distance validation tasks, Q-learning succeeds on 77/81/87
and SARSA on 5/1/3 across seeds 7/19/43. All runs discover all 79 training goals,
but discovering a goal does not mean learning efficient routes from every start.
Nearly all validation initial states were encountered as states during training
(99.78%–100%); results concern held-out initial conditions on a familiar map.

The old 30k pilot was inspected before running the matrix. Its 503 timeouts each
visited only 2–11 distinct positions over 200 actions, confirming repeated-state
loops. Seven goals had no training discovery in that pilot. The longer budget
tests additional learning on the same task without masking or planner fallback.
The diagnostic artifact lists the actual goal IDs and Q-value range.

The eight runs total **1,600,000 training interactions** (six independent runs plus
two repeats), 575,491 monitoring interactions and 803,520 full-validation actions.
The suite, including exports and trajectory audits, took 110.11 seconds wall time
on this machine. Training-loop times varied from 6.03 to 11.43 seconds per run;
they exclude environment construction, monitoring/checkpoints, final evaluation,
exports and rendering. Other local work was active, so wall time is descriptive.

## Reproducibility and correctness

- **23 tests pass**, including hand-calculated SARSA versus max-Q targets, terminal
  targets, timeout bootstrapping, next-action carry/reset boundaries, fixture
  learning, monitoring isolation, checkpoint dispatch and compressed export.
- Gymnasium's environment checker passes on the unchanged task.
- Seed-7 repeats for both algorithms produce identical Q-tables, training CSV,
  per-episode validation metrics and full replay records after excluding policy ID.
  Repeat runs are reproducibility evidence, not additional independent seeds.
- Every saved validation episode was checked against the environment and frozen
  policy: **7,424 episodes / 803,520 actions**. Saved Q values and chosen actions
  match the loaded checkpoint; no validation update changes its table.
- Completed episode logs plus any separately recorded budget-cutoff episode
  reconcile to exactly 200,000 visits per run. Budget cutoffs do not invent timeout
  outcomes. Goal-sampling counts include only starts that receive an action.
- **160 periodic inference checkpoints** load with valid checksums and signatures.
  Full configuration, dependency/hardware metadata and source snapshots are saved.
  Each source snapshot matches its run manifest. Checkpoints are inference-only;
  exact training resume is unsupported. Failed/interrupted status is recorded by
  the runner rather than claiming a complete-budget result.
- The evaluation CLI independently reloaded SARSA seed 7 and reproduced 255
  successes, 15 collisions, 658 timeouts and SPL .259340 on validation.
- Final-test policy executions: **zero**. The loader/UI/CLI continue to exclude the
  final test. Benchmark files and the complete milestone-1 bundle are unchanged.
  SHA-256 preservation checks covered 45 existing files with zero changes.

## Viewer and remaining work

The viewer runs locally at `http://127.0.0.1:8517`. It loads both algorithms,
displays actual saved successes/failures, and offers frozen-policy custom demos.
AppTest checks include the new SARSA replay and SARSA custom-policy selection,
as well as playback controls and synchronized counters. Training never runs in UI.

The previous browser-access blocker was resolved by launching a separate local
Chrome profile inside the project's ignored cache. The existing signed-in browser
was not used for content inspection. The previously running Streamlit process had
cached the old agents module; restarting the owned server loaded the new API.
Browser playback visibly advanced both BFS and saved Q-learning seed 7 from
`[0,0]` to `[0,3]`, with three actions, return 9.97, step reward 9.99 and success.
Dark-mode metric contrast and narrow-window counter clipping found during this
inspection were corrected. Browser details are recorded in `milestone2_ui_check.json`.
Backward seeking showed action 1 at `[0,1]` with return -.01; Next showed action 2
at `[0,2]` with return -.02. Reset returned to the initial frame. A real Q-learning
collision replay ended after one action with return -5.01: attempted cell `[2,3]`,
actual robot cell `[2,4]`. No quantitative browser responsiveness claim is made.
SARSA's replay path was checked through AppTest rather than manual Chrome playback.

The learned policies still need better long-route reliability. The next planned
milestone adds DQN and the full equal-budget Part A comparison; it does not add PPO,
new maps, moving obstacles, continuous control, robotics hardware, tuning, ensembles
or the full dashboard. No final model or main-study shortlist is selected here.

Evidence: [suite summary](../artifacts/studies/milestone2-tabular-v1/summary.json),
[trajectory/repeat audits](../artifacts/studies/milestone2-tabular-v1/verification.json),
[run matrix and timing](../artifacts/studies/milestone2-tabular-v1/manifest.json),
[earlier-pilot diagnostics](../artifacts/studies/milestone2-tabular-v1/original_smoke_diagnostics.json).

The original runner is archived with the study and its hash matches the recorded
execution. Later changes add output-prefix CLI options for safe reproduction.
Post-run application changes affect styling/layout only; training, environment,
policy updates and benchmark inputs match the execution snapshots.
