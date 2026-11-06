# Autonomous Robot Navigation Using Reinforcement Learning

Private technical report · completed core release · 6 November 2026

## Problem and solution

Build and study a robot controller that reaches varied goals on one fixed static
grid using reinforcement learning. The implemented solution combines a versioned
Gymnasium task, reproducible Q-learning/SARSA/DQN experiments, bounded refinement,
frozen policy voting, a local animated Streamlit dashboard and audited holdout
evaluation. This is a simulated discrete navigation study, not a physical robot.

The predeclared final system remains **m4-tuned-q-learning-seed101**, despite any
final ranking. It reached **840/928 goals (90.52%)**,
with 7 collisions, 81 timeouts
and SPL 0.867870. The matched seed101 original-setting control reached
623/928 (67.13%), SPL 0.636640.

## Environment and state design

One immutable 10×10 connected bottleneck map, four cardinal actions and coordinates
of robot and goal. Learned policies receive no obstacle channel, invalid-action
mask, planner direction or fallback. Tabular methods index the four coordinates;
DQN uses those same values scaled by dimension minus one. BFS explicitly knows
the map and is a minimum-step reference, not a learned competitor.

An attempted action costs -.01; reaching the goal adds +10; a wall/boundary attempt
adds -5 and ends in collision. Goal/collision terminate; 200 actions truncate unless
a terminal event occurs on that last action. Learners bootstrap through truncation.
These rules and metric calculations have hand-solvable tests.

The simulator generated experience; no external energy dataset was used. Saved
start–goal pairs are stratified by goal/distance into approximately 70/15/15 train/
validation/test. Rounding occurs within strata; exact counts/hashes are recorded.
The final set contains 928 pairs: 258 short (1–5), 350 medium (6–10), 320 long (>=11).
All held-out goals occur in training. Trajectories may visit held-out initial states;
this is held-out initial conditions on a familiar map, not unseen-state/map transfer.

## Relevant methods and fair evaluation

Q-learning uses a max next-action target; SARSA uses its carried next behavior
action. Exploration and random training tie-breaking are confined to training;
evaluation uses greedy fixed-priority ties. Vanilla Stable-Baselines3 DQN uses a
replay buffer and target network; no Double/Dueling/PPO claim is made.

Each main learned run received 200,000 interactions, with a fixed 30-pair validation
monitor every 10k. Checkpoints maximize monitor success, then SPL, then earliest
age. Configuration/family choices use complete validation, never the final set.
All five seeds, including failures, are retained. Returns are compared only under
the same reward; SPL averages success×shortest/max(shortest,actions), failures zero.
Successful route length is conditional on success and always has a success count.

## Part A and Part B development evidence

Part A compared five seeds each of Q-learning, SARSA and DQN at equal budgets.
Validation success means were 65.24%, 21.94% and 5.32%; the declared rule shortlisted
Q-learning and SARSA. Poor DQN outcomes were retained rather than changing the task.

Part B sampled eight shared settings per shortlisted family with two development
seeds. Fresh five-seed confirmations were paired with original-setting controls.
Both families selected learning rate .9, discount .99, epsilon .7→.03 with decay
fraction .3. Validation success improved from 64.74% to 89.96% for Q-learning and
24.55% to 33.00% for SARSA. Q-learning met declared mean targets; SARSA missed all.

A controlled reward-only variant removed the step cost without retuning. Success
fell to 32.95% and 19.70% respectively, so that variant was not promoted. No tabular
index-renaming claim was made as feature engineering. Neural representation changes
were not needed after DQN was not shortlisted.

The frozen Q-learning vote uses independent seeds [101,211,307], majority action and
first-member priority for three-way ties. On validation it lost five goals versus
predeclared single seed101 while improving SPL. Its three Q arrays require 960,000
bytes versus 320,000, excluding Python overhead. Earlier shared latency measurement
was 7.51 versus 1.66 microseconds per decision; local timing is not hardware-isolated.
The 600k member training actions are already included in Part B, not counted twice.

## Frozen final evaluation

Freeze digest: `f536d8ab40dc461f7dd2f94cb31cda2ce9a612f13f19c4bedd7c431bf91f8c28`. Exact source, runner/analysis, task/split bytes, model and
metadata/configuration hashes and installed versions were sealed **before** parsing
test scenarios. The runner requires the digest and verifies every frozen file
before/after execution. No source/model/analysis changes occurred during evaluation.
The selected single seed101 and matched baseline were fixed before test. All
35 learned checkpoints, one vote, five random streams and BFS were evaluated.

| Frozen system | Replicates | Success ± SD | Collision | Timeout | SPL |
|---|---:|---:|---:|---:|---:|
| Part A Q-learning | 5 | 65.43% ± 2.17 pp | 0.45% | 34.12% | 0.617727 |
| Part A SARSA | 5 | 20.37% ± 4.95 pp | 4.35% | 75.28% | 0.192424 |
| Part A DQN | 5 | 5.78% ± 0.95 pp | 30.43% | 63.79% | 0.057484 |
| Q-learning Baseline | 5 | 65.52% ± 2.66 pp | 0.22% | 34.27% | 0.620085 |
| SARSA Baseline | 5 | 23.23% ± 2.48 pp | 2.09% | 74.68% | 0.219343 |
| Q-learning Refined | 5 | 90.43% ± 0.71 pp | 0.54% | 9.03% | 0.863781 |
| SARSA Refined | 5 | 32.44% ± 2.64 pp | 2.11% | 65.45% | 0.303690 |
| Q-learning Voting | 1 | 91.38% | 0.54% | 8.08% | 0.893175 |
| Random reference | 5 | 2.56% ± 0.66 pp | 97.44% | 0.00% | 0.019191 |
| BFS reference | 1 | 100.00% | 0.00% | 0.00% | 1.000000 |

SD is sample SD across five training seeds, or across random RNG streams as labeled;
voting and BFS have one record and no SD. Part A seeds differ from Part B controls;
those groups are not paired or pooled. Repeated scenarios/records are not independent
trained agents. No episode-level confidence interval or significance claim is made.

Q-learning's seed-matched success change is +24.91
percentage points (paired SD 3.03 pp),
with SPL +0.243696. Its collision rate **increased**
0.32 pp, from 0.22% to 0.54%.
Four of five matched seeds have more collisions. Higher success is not a claim of
uniformly safer movement. SARSA gains 9.20 pp
but remains weak and misses all three unchanged mean targets (80% success, <=2%
collision, SPL >=.75). Tuned Q-learning meets all three.

On final test, voting reached 848 goals versus the selected single's 840: eight more,
with SPL .893175 versus .867870. The selected system is **not changed** using that
holdout advantage. One fixed ensemble provides no ensemble-seed uncertainty estimate.
BFS solved every one of the 928 final cases optimally; RL is not expected to beat
an exact shortest-path planner here.

## Core robustness and failure interpretation

The milestone-5 label functions were frozen unchanged before final execution.
Distance, canonical BFS corridor, start-wall proximity, start/goal quadrant and
quadrant crossing each partition the same pairs; dimensions overlap. Corridor
depends on a canonical shortest route and its tie order, not every learned route.

| System | Short / 258 | Medium / 350 | Long / 320 |
|---|---:|---:|---:|
| Q-learning Baseline | 97.60% | 79.71% | 24.12% |
| Q-learning Refined | 97.52% | 97.94% | 76.50% |
| SARSA Refined | 73.18% | 26.74% | 5.81% |
| Q-learning Voting | 97.67% | 95.71% | 81.56% |

Refined Q-learning succeeds on 89.09% of reference-corridor cases (772 pairs),
versus 58.86% for controls; no-reference-corridor success declines from 98.46% to
97.05% (156 pairs). Different-quadrant success is 88.56% versus same-quadrant 96.51%.
These associations are confounded with distance/geometry and are not causal effects.
Every group/seed metric and support is saved in the final summary. Long-route
failures remain material. SARSA/DQN's shorter successful paths do not offset their
many failed episodes; read conditional route plots with the full outcome partition.

## Dashboard and observable behavior

Six local views cover overview, simulator, saved learning curves, comparisons,
optimization/robustness and methodology. The simulator replays individual actions
with play/pause, step/reset, seeking and speed controls; start/goal/trail/attempted
collision, action/reward/status counters stay synchronized. Prefix visits, optional
BFS reference, recorded Q scores/votes and an event log support inspection.
Matched-agent and early/selected-checkpoint views synchronize by action index and
hold finished agents at terminal frames. They never train or contribute benchmark
metrics. Frozen aggregate final results appear only after the final audit passes;
interactive scenarios remain validation/custom, and generic split loaders stay sealed.

The live browser check passed BFS success, an actual learned-policy collision and
baseline/refined shared indices 0/1/2/3. A stale server import required restarting.
Live controls that did not advance inside the timer-refreshed display were moved
outside it; both live behavior and tests then passed. Details remain in milestone-5
UI evidence. No quantitative frame-rate benchmark is claimed.

## Verification, costs and reproducibility

Before freezing: **64 tests passed in 34.47 seconds**, the Gymnasium checker and
`pip check` passed, and a synthetic final pipeline exercised sealing, audit, output
publication and overwrite refusal without real final scenarios. After publication,
five dashboard page tests passed with actual final data. All four final figures were
visually inspected. No final-result-prompted changes to scientific code/models occurred.

Final evaluation: 42 records, **38,976 episodes /
3,456,940 actions**, 109.68 seconds.
The complete action/score/state/reward/flag replay audit verified the same episodes
and actions, taking 87.87 seconds including analysis.
Audit replay is a repeatability pass, not another independent test sample.
All 184 frozen inputs remain unchanged. The study used about 53.1 MiB before release
documentation, below the 500-MiB evaluation cap; the evaluation cap was 15 minutes.

Historical records contain 87 positive-budget development runs and
17,040,000 training actions, including pilots/repeats. Main Part A accounts
for 3m and Part B for 12.4m; search/refinement effort is not hidden in the final
policy's 200k budget. A separate isolated reproduction added 200k actions and took
8.76 seconds: canonical preparation, selected Q arrays and all validation episode
metrics matched exactly. Total recorded training including that reproduction is
17,240,000; no training used final outcomes. Historical wrapper costs
are zero training and member costs are not double-counted. Per-run training time,
monitoring and evaluation costs are preserved in manifests/run matrices.

Setup was verified using installed locked dependencies on Windows/Python 3.14.10,
not a clean network dependency install or cross-hardware test. The exact versions
are in the freeze. Model snapshots are inference-only; exact training resume is
unsupported. Another machine/library version can differ bitwise. README commands
cover setup, preparation, training, validation, freezing/evaluation and local UI.

Git normally normalizes text line endings, which can change raw input hashes after
checkout. The new final archive has scoped `-text` attributes to preserve its exact
bytes. A post-evaluation packaging helper restored all 184 archived files into an
isolated workspace and verified them with the original freeze; this preparation
did not execute policies. Git-filter byte checks passed on representative metadata,
test-manifest and runner files. Optional unchanged replay uses the archived code
and matching versions, never new model selection. This helper is a packaging
addition after evaluation; the frozen scientific runner/analysis/models are unchanged.
The final table was also inspected in the live native Chrome overview.

Preserved **6,763 pre-milestone-6 root/data/artifact files**, including local
ignored models/traces. No energy project or sources directory was present here.
All work stays local; no commit, push, public share link or deployment was performed.

## Development plan, limits and future work

Milestones 1–2 established the task and tabular learning; 3 compared Part A; 4 tuned
and ablated/voted; 5 integrated the dashboard and core robustness; 6 froze, evaluated,
audited and documented the core release. The individual development log records
implementation and Codex assistance without fictional team allocations.

Residual timeouts/collisions, weak SARSA/DQN, five-seed limits and one-ensemble
uncertainty remain. The final set is now spent: subsequent performance changes
motivated by it require a new holdout or explicit exploratory labeling. PPO,
dynamics noise, unseen maps, moving obstacles, partial observations, continuous
control and physical hardware remain deferred; simulator collision rates are not
physical safety certification. No further tuning or winner promotion followed test.

## Evidence and method attribution

- [Environment specification](environment_specification.md), [Part A](milestone3_results.md),
  [Part B](milestone4_results.md), [integration/robustness](milestone5_results.md).
- [Final protocol](milestone6_protocol.md), [freeze](../artifacts/studies/final-v1/freeze.json),
  [summary](../artifacts/studies/final-v1/summary.json), [audit](../artifacts/studies/final-v1/evaluation/verification.json),
  [per-seed CSV](../artifacts/studies/final-v1/final_seed_results.csv),
  [run matrix](../artifacts/studies/final-v1/development_run_matrix.csv),
  [figures](../artifacts/studies/final-v1/evaluation/figures/), [reproduction](milestone6_reproduction.json).
- External method/library attribution: [Gymnasium environment API](https://gymnasium.farama.org/introduction/create_custom_env/),
  [termination and truncation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API),
  [SB3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html),
  [SB3 evaluation](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
  Installed SB3 replay/target code and project source were inspected and tested;
  no third-party training dataset or planner-derived learned supervision was used.
