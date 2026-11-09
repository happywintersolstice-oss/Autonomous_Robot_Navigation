# Autonomous Robot Navigation Using Reinforcement Learning

Consolidated technical report · updated 9 November 2026

## Current status and main findings

The six core milestones and the subsequent PPO, representation, neural tuning, learning-replay and dashboard extensions are complete. Four trained RL algorithms are available. The latest complete regression run passed 158 tests, and native Chrome checks covered desktop and narrow layouts.

The original selected controller is m4-tuned-q-learning-seed101. Its frozen final evaluation reached 840 of 928 goals, success 90.52%, with 7 collisions, 81 timeouts and SPL .867870. Later development studies retain this historical selection.

The five-seed neural follow-up found success means of 6.88% baseline versus 7.37% selected settings for DQN and 19.44% versus 23.69% for PPO. The two added seeds averaged declines for both algorithms. DQN had more collisions and a slightly lower mean return after tuning. PPO had fewer collisions, more timeouts and substantial seed variability. Both neural families miss the declared targets.

The owner chose to defer independent four-family evaluation to the upcoming unseen-map study. The current fixed map has 79 free cells and 6,162 possible distinct ordered start–goal pairs. Its 4,306 training, 928 validation and 928 test pairs cover all of them. The original test outcomes were inspected during the core release. A new split or new training seed on this map would reuse that task space. The current extensions therefore retain their exploratory development-validation scope.

## Problem, environment and information contract

The project studies goal-directed navigation on a fixed, connected 10×10 bottleneck grid. Q-learning, SARSA, vanilla Stable-Baselines3 DQN and categorical PPO receive robot and goal coordinates. BFS explicitly knows the map and supplies an optimal shortest-path reference. Random actions provide a lower reference.

Actions are Up, Down, Left and Right. Each attempted action costs -.01. Reaching the goal adds +10. A wall or boundary attempt adds -5 and ends the episode while retaining the actual robot position. Goal and collision terminate. A 200-action limit truncates unless a terminal event occurs on the last action. Training bootstraps through truncation.

Tabular policies index the four coordinates. The original DQN/PPO policies use those coordinates scaled by dimension minus one. Their networks use [64,64] hidden layers. Learned controllers have no obstacle channel, action masking, planner instructions or planner fallback. The environment and metric rules have hand-solvable tests.

Experience is generated in the simulator. Goal/distance-stratified manifests fix the training and held-out initial conditions on the same familiar map. Training trajectories can visit states that also occur in held-out tasks. The evidence concerns familiar-map behavior.

## Methods and shared experiment rules

Q-learning uses a maximum next-action target. SARSA uses its carried next behavior action. Exploration and random training tie-breaking occur during learning. Evaluation uses deterministic actions and fixed-priority ties. DQN uses replay and a target network. PPO uses a categorical policy and value function.

Every main comparison run trains for 200,000 interactions. A separate read-only environment evaluates a fixed 30-route validation monitor every 10,000 interactions. Checkpoints rank by monitor success, then SPL, then earliest interaction. Full selected-checkpoint validation uses all 928 validation pairs.

Report per-seed success, collision, timeout, SPL, return and successful route length. SPL assigns zero to failures and otherwise uses shortest/max(shortest,actions). Successful route length is conditional on success and must be read with success counts. Aggregate within each training seed before calculating equal-weight means and sample seed SD. Random-reference variability spans RNG streams. A fixed BFS reference and one voting ensemble have one record each.

Preserve poor seeds, incomplete attempts, configurations, selected/final models, monitoring records, training ledgers and source identities. Validation execution, audits and display demonstrations do not add learner training interactions.

## Evidence cohorts and comparability

| Cohort | Training seeds | Budget per main run | Evaluation scope |
| --- | --- | --- | --- |
| Original baseline comparison plus later PPO | 7, 19, 43, 67, 91 | 200,000 | Development validation, four families |
| Tabular baseline/refined confirmations | 101, 211, 307, 401, 503 | 200,000 | Paired development validation |
| DQN absolute/relative input study | 101, 211, 307, 401, 503 | 200,000 | Paired development validation |
| Neural baseline/selected confirmations | 7307, 7411, 7523, 7607, 7717 | 200,000 | Staged development validation |
| Original frozen final release | Predeclared core matrix | Existing trained checkpoints | Original 928 test pairs |

These cohorts have separate questions, training seeds and selection histories. The baseline four-family comparison uses the original settings. Tabular tuning screened eight shared candidates with two development seeds. Neural tuning screened four candidates per family using one development seed at 100,000 steps. Search opportunities differ, so the refined tables support their within-family comparisons.

## Four-algorithm baseline validation

| Algorithm | Training seeds | Success mean ± seed SD | Collision mean | Timeout mean | SPL mean |
| --- | --- | --- | --- | --- | --- |
| Q-learning | 5 | 65.24% ± 1.64 pp | 0.17% | 34.59% | 0.619171 |
| SARSA | 5 | 21.94% ± 5.37 pp | 3.94% | 74.12% | 0.206985 |
| DQN | 5 | 5.32% ± 1.04 pp | 31.77% | 62.91% | 0.052956 |
| PPO | 5 | 15.22% ± 13.46 pp | 59.91% | 24.87% | 0.151961 |

PPO was added after the original three-family release. Its baseline study used the existing map, information, training budget and validation scenarios. The original Part A shortlist selected Q-learning and SARSA from validation. Later PPO inclusion expands the development comparison while preserving the original shortlist history.

## Tabular refinement, reward and voting

| Family and settings | Success mean ± seed SD | Collision mean | SPL mean |
| --- | --- | --- | --- |
| q-learning control | 64.74% ± 2.59 pp | 0.28% | 0.611012 |
| q-learning tuned | 89.96% ± 0.79 pp | 0.19% | 0.858850 |
| sarsa control | 24.55% ± 2.14 pp | 1.96% | 0.231198 |
| sarsa tuned | 33.00% ± 1.64 pp | 2.03% | 0.309894 |

Both tabular families selected learning rate .9, discount .99 and epsilon .7→.03 with decay fraction .3. Q-learning met the declared validation mean targets. SARSA missed them. A controlled reward-only experiment removed the step cost without retuning. Success fell to 32.95% for Q-learning and 19.70% for SARSA, so this variant was not promoted.

The Q-learning vote uses ordered independent members 101, 211 and 307, majority action and first-member priority for three-way ties. On validation it lost five goals against the predeclared single seed101 while improving SPL. Its three Q arrays require 960,000 bytes versus 320,000 for one policy, excluding Python overhead. The earlier measured decision latency was 7.51 versus 1.66 microseconds. Member training is counted once.

## Controlled DQN input comparison

M8 compared four normalized absolute coordinates against those four plus two deterministic goal-minus-robot offsets. All other DQN settings remained fixed. Five paired seeds completed 200,000 interactions per condition. The six-input model has 128 additional online first-layer parameters with the fixed [64,64] architecture.

Mean validation success was 5.97% absolute versus 6.51% absolute-plus-relative. The paired change was +.54 percentage points with sample paired-seed SD 2.88 pp. Two seeds improved, two worsened and one tied. Collisions declined while timeouts increased. The tested input design did not establish consistent improvement. The study retained actual chronological training traces for its new runs.

## Neural tuning and five-seed extension

M9 screened four configurations each for DQN/PPO on development training seed 7201. Both selected c01 before the original three confirmation seeds. DQN c01 uses learning rate .001, target updates every 250 steps and an exploration decay fraction of 1.0. PPO c01 uses learning rate .001, rollouts of 500 steps and entropy coefficient .01. Other original hyperparameters and the four-coordinate input remain fixed. Bundled candidates do not isolate each factor.

The initial confirmation used seeds 7307, 7411 and 7523. The owner then requested expansion after those outcomes were known. Added seeds 7607 and 7717 were declared before their training. The five-seed total reuses the original three runs per condition, and the two added seeds are also reported separately.

| Algorithm and settings | Success mean ± seed SD | Collision mean | Timeout mean | SPL mean | Mean return |
| --- | --- | --- | --- | --- | --- |
| DQN baseline | 6.88% ± 2.25 pp | 20.80% | 72.33% | 0.066929 | -1.8088 |
| DQN selected c01 | 7.37% ± 0.55 pp | 25.04% | 67.59% | 0.072190 | -1.8770 |
| PPO baseline | 19.44% ± 9.91 pp | 65.71% | 14.85% | 0.193967 | -1.6667 |
| PPO selected c01 | 23.69% ± 14.62 pp | 56.38% | 19.94% | 0.235088 | -0.8778 |

| Algorithm | Five-seed paired success change ± SD | Added-two-seed change ± SD | Five-seed wins / losses / ties |
| --- | --- | --- | --- |
| DQN | +0.50 ± 1.95 pp | -1.02 ± 2.51 pp | 4 / 1 / 0 |
| PPO | +4.25 ± 9.06 pp | -0.43 ± 12.04 pp | 4 / 1 / 0 |

The initial three seeds had paired success changes of +1.51 pp for DQN and +7.36 pp for PPO. Both added-seed averages declined, reducing the five-seed gains. DQN selected settings increase collisions and slightly worsen mean return relative to the new matched baseline. PPO improves mean return and reduces collisions, with more timeouts and greater seed variation. Both miss success≥80%, collision≤2% and SPL≥.75. The frozen choices were retained through these regressions.

## Historical frozen final evaluation

The original final release sealed its scientific source, analysis, configurations, splits, checkpoints and installed versions before test access. Its freeze digest is `f536d8ab40dc461f7dd2f94cb31cda2ce9a612f13f19c4bedd7c431bf91f8c28`. The following table belongs to that original three-family release and excludes later PPO/neural extensions.

| Frozen system | Replicates | Success mean ± sample SD | Collision mean | Timeout mean | SPL mean |
| --- | --- | --- | --- | --- | --- |
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

The predeclared single seed101 remained selected. Voting reached 848 goals versus that single policy’s 840, with SPL .893175 versus .867870. One fixed ensemble does not estimate variation across independently constructed ensembles. The final ranking caused no promotion. BFS solved all 928 final cases optimally.

Across five matched tabular seeds, Q-learning final success rose by 24.91 pp with paired SD 3.03 pp and SPL +.243696. Its collision rate rose by .32 pp, with four of five seeds having more collisions. SARSA gained 9.20 pp and still missed its targets. Successful-route lengths must be interpreted with these outcome rates.

The historical final matrix had 42 records, 38,976 episodes and 3,456,940 actions. Evaluation took 109.68 seconds. Audit and analysis took 87.87 seconds. Audit replay checks the same cases. All 184 sealed inputs still match their archive and seal.

## Fixed-map robustness and failure interpretation

The six predeclared grouping dimensions are distance, canonical BFS corridor, start-wall proximity, start region, goal region and quadrant crossing. Dimensions overlap and their associations are descriptive. Corridor labels depend on the canonical shortest route and tie order.

| Historical final group | Short / 258 | Medium / 350 | Long / 320 |
| --- | --- | --- | --- |
| Q-learning baseline | 97.60% | 79.71% | 24.12% |
| Q-learning refined | 97.52% | 97.94% | 76.50% |
| SARSA refined | 73.18% | 26.74% | 5.81% |
| Q-learning voting | 97.67% | 95.71% | 81.56% |

Refined Q-learning final success was 89.09% on 772 reference-corridor cases versus 58.86% for controls. On the 156 remaining cases it changed from 98.46% to 97.05%. Different-quadrant success was 88.56% versus 96.51% for same-quadrant cases. Distance and geometry are confounded in these descriptive partitions.

The later four-family validation analysis reconciled 18,560 episodes and 340 group/seed rows using 104 fixed inputs. It adds PPO group/route/cost analysis to the dashboard. The neural follow-up retains every failed route and poor seed. Short routes among rare successes do not establish robust navigation.

## Dashboard and actual learning replay

The local Streamlit dashboard has simulator, overview, learning-analysis, comparison, optimization/robustness, neural-tuning and methodology pages. The simulator separates Trained replay, Learning replay, Compare agents and Try a route. The sidebar contains navigation. Model filters and detailed diagnostics remain near their related controls.

Playback advances one discrete action at a time with Play, Pause, Next step, Reset, speed and timeline controls. Start, goal, robot, trail and collision attempt stay synchronized with actions, reward and status. The future path is hidden during replay. Matched comparisons hold finished models at terminal frames. DQN/tabular views display Q scores, while PPO displays probabilities and a state value. Display actions do not train models or enter benchmark aggregates.

Prospective learning recordings retain actual exploratory actions. The four-family demonstration bundle has 40 staged recordings. The original M9 suite has 220, and the combined five-seed confirmations have 200, including 80 added recordings. The combined count reuses 120 original confirmation records. Historical actions absent from the original logs remain unavailable. A training-budget prefix has an explicit incomplete status.

Native Chrome checks passed at 1536×912 and 854×722. The five-seed/original study switch, scope note, tables, figures, added-seed detail and costs were checked. A squeezed second-model/route row now stacks at narrow widths. Wide tables retain internal scrolling. An illustrative matched PPO route reached its four-action success while holding the baseline’s one-action collision. Seeking and Reset showed the expected recorded positions and returns. Manual Pause is also covered by automated tests. Phone/touch, screen-reader usability and measured frame-rate/input-latency benchmarks remain untested.

## Failure recovery, cost and reproducibility

A Windows OSError 22 interrupted PPO progress-file saving during the first five-seed extension attempt. Four completed DQN additions were retained verbatim. The paired PPO attempts stopped at or after 50k baseline and 40k selected steps. Recovery retrained the remaining PPO conditions from zero with the frozen settings and seeds under an attempt2 cohort suffix. Seed 7717 first trained in that cohort.

JSON output now writes a complete same-directory temporary file and atomically replaces its destination. Readers and replacement operations have bounded transient-access retries. Tests verify byte-format compatibility, old-data preservation on permanent failure and concurrent readers. The exact Windows locking source is unconfirmed. All other learning-engine core syntax and engine file hashes match the original versions. Saved parameter/optimizer state and counters match across failed/restarted PPO prefixes at 50k and 40k.

The completed additions total 1.6m training steps. Incomplete PPO attempts add at least 90k steps, with the killed worker’s exact final count unavailable. Recovery and the combined audit took 474.10 seconds within the declared 1,800-second cap. The 20 combined confirmations represent 4m training steps and reuse 2.4m original confirmation steps. Per-study costs exclude double counting reused runs.

Neural per-run training seconds, monitoring/validation seconds, 100-query decision latency and checkpoint bytes are available for every seed. Two one-thread learners ran concurrently during these phases. Timing combines runs executed under different machine loads. File size measures saved models. Source snapshots and raw checksum metadata retain provenance. Exact training resume and cross-machine bitwise equality remain unsupported.

The historical core development matrix contained 87 positive-budget runs and 17.04m training actions. An isolated reproduction added 200k actions and reproduced the selected Q table and validation metrics. Later extension costs are recorded separately in their study reports and include screening, pilots, failed attempts and reused components explicitly.

A later isolated environment installed all 58 pinned dependencies, passed dependency/Gymnasium checks and restored the 184-file frozen release. Cached wheels were reused on this Windows machine. Cold-cache downloads and other hardware were not tested. The report/commit closeout reran the full suite with 158 tests passing in 76.39 seconds. The five-seed audit replayed 1,656,863 validation actions and 2,743 actual training actions, checking 400 periodic checkpoint checksums across 20 complete runs.

Preservation checks passed for 11,563 pre-extension and 12,142 pre-recovery artifact/data files. An older cross-session root README checksum differs because of an earlier committed README update. The root README matches the starting revision for the extension. Existing files outside the navigation project and archived final evidence are preserved.

## Independent evaluation boundary and next stage

On 9 November 2026 the owner chose to defer independent four-algorithm evaluation to the upcoming unseen-map study. All 6,162 possible distinct ordered start–goal pairs are already represented in the existing manifests. The 928 test outcomes were inspected before PPO and later neural tuning. Another evaluation on those pairs would be retrospective.

The next stage must establish a separate versioned benchmark, disjoint training/validation/final layout sets, a common policy-information contract and predeclared budgets, training seeds, metrics and selection rules. Seal the final layouts and freeze checkpoints/analysis before their final evaluation. Existing fixed-map checkpoint compatibility checks need an explicit new design for the new task. Detailed dataset/model decisions remain for that stage.

The current stage consolidates existing evidence and commits the verified implementation. It records zero new canonical benchmark training or final-test executions. Regression tests can train isolated fixtures. Unseen-map performance and a four-family confirmatory result remain unmeasured. Dynamics noise, moving obstacles, continuous movement, sensors, ROS and hardware remain deferred beyond their separately authorized stages.

## Reproduction and evidence index

Windows setup, preparation, training, validation and local UI commands are in the project README. Historical frozen execution uses its archived scientific runner and 184 exact inputs. Saved trajectories and audits support record verification. Periodic checkpoints and full compressed trajectories remain local where documented by the ignore rules.

Commit packaging preserves the recorded bytes of Windows-generated legacy JSON/CSV/log files through explicit checkout line endings. New neural confirmation archives use recursive raw-byte rules. An isolated checkout verified all 7,104 indexed artifact/data files against the working records, including the LF map definition and historical CRLF Part B report. The current working-tree preservation check also covers 12,603 artifact/data files, including local ignored inputs.

- [Project setup and commands](../README.md), [environment contract](environment_specification.md), [current status](plan_closure_status.md) and [development log](development_log.md).
- [Baseline comparison](../artifacts/studies/four-rl-v1/summary.json), [four-family analysis](four_way_analysis_results.md), [tabular refinement](milestone4_results.md) and [representation results](milestone8_results.md).
- [Neural five-seed outcomes](neural_confirmation_results.md), [verification](neural_confirmation_verification.json), [full summary](../artifacts/studies/neural-confirmation-five-v2/summary.json) and [audit](../artifacts/studies/neural-confirmation-five-v2/verification.json).
- [Historical final protocol](milestone6_protocol.md), [freeze](../artifacts/studies/final-v1/freeze.json), [summary](../artifacts/studies/final-v1/summary.json) and [final audit](../artifacts/studies/final-v1/evaluation/verification.json).
- [Independent evaluation handoff](evaluation_boundary_and_handoff.md).
- [Report and commit verification](report_closeout_verification.json).

Method attribution remains documented in the project plan and development log, including Gymnasium and the installed Stable-Baselines3 DQN/PPO implementations. The simulator supplies the experience. Codex assisted implementation, execution, debugging and reporting.
