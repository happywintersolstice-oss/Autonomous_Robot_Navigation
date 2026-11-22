# Milestone 9 controlled DQN/PPO tuning

Authorized on 9 November 2026 following owner acceptance of the dashboard visuals.
This exploratory post-release study preserves the original release, previous
studies, task, splits and selected Q-learning controller. The six core milestones
are complete. M9 addresses the remaining untuned neural baselines.

## Fixed question and scope

Can a bounded search improve either original four-coordinate neural baseline on
the familiar map? Use the existing DQN and categorical PPO implementations,
normalized absolute robot/goal coordinates, four discrete actions and the
original rewards, 200-action limit and reset distribution. Keep [64,64] networks,
one CPU thread per learner, existing batch/update semantics and read-only
evaluation. There is no change to inputs, network capacity or the benchmark.
All unlisted settings inherit the original `m3-dqn-seed7` or `m7-ppo-seed7`
configuration, replacing run identity, seed, budget and bounded trace settings.

## Predeclared matrix

`configs/neural_tuning.yaml` defines four candidates for each algorithm. c00
retains its original hyperparameters. DQN alternatives change learning rate,
target-update interval, exploration schedule and/or discount. PPO alternatives
change learning rate, rollout length, entropy coefficient, clipping and/or
discount. Bundled candidates test configurations. Effects of individual factors
cannot be separated by this small search. Candidate values are experimental
hypotheses chosen before training.

| Phase | Training seeds | Steps per run | Runs |
| --- | --- | --- | --- |
| Mechanics/runtime pilots | 7101 | 10,000 | 2 |
| Four-candidate screen per algorithm | 7201 | 100,000 | 8 |
| Baseline versus selected settings | 7307, 7411, 7523 | 200,000 | At most 12 |

The main plan allows three seeds for a clearly labeled preliminary study under
compute constraints. This milestone uses that allowance and a maximum of
3,220,000 training interactions including pilots. The one-seed screen has high
selection uncertainty. Fresh confirmation seeds estimate variability after
selection. Matching seed numbers does not imply identical trajectories.

Pilots exercise unchanged baseline settings and estimate costs. Exclude pilots
and search runs from confirmation averages. Do not change settings from pilot
performance. Freeze the candidate plan, source hash, protocol hash, runner hash,
seed matrix and measured runtime estimate before the main screen.

## Selection and evaluation

Every run completes its exact budget. Use the saved fixed 30-pair monitoring
subset every 10,000 steps with a separate environment and no learning. Preserve
the existing DQN intermediate callback timing and PPO post-update rollout-boundary
timing. Choose a checkpoint by monitor success, then SPL, then earliest step.
Evaluate the selected checkpoint on all 928 development-validation pairs.

Rank the four candidates for each algorithm by full validation success, then
SPL, then lower collision rate, then ascending candidate ID. Freeze this choice
before any confirmation run. Do not reselect using confirmation outcomes. If
c00 wins, retain the original settings and train only three fresh control runs.
Both summary conditions then explicitly reference those same runs, with zero
paired difference. They do not become six independent replicates.

The original final-test manifest will have zero new policy executions. The
existing selected controller remains unchanged. This study shares development
validation with earlier work, so fresh training seeds do not supply a fresh
holdout or evidence for unseen-map transfer.

## Reporting and recorded replay

The primary comparison is selected-minus-control success percentage points for
each matched confirmation seed. Report all seeds, mean, sample seed SD,
wins/losses/ties, collision, timeout, SPL, return and successful route length.
Conditional route-length means exclude seeds without successes and report the
contributor count. Show monitoring curves with sample seed SD. There is no
episode-level significance or confidence-interval claim.

Report training, monitoring and validation seconds, saved-checkpoint bytes and
decision microseconds. Two separate training processes may run concurrently,
each with one learner thread. Timing reflects concurrent machine load.
Training excludes construction/monitoring/export. Validation includes recording
and export. Record total suite cost including pilots and screening.

The aspirational selected-settings means are success at least 80%, collisions at
most 2% and SPL at least .75. A milestone completes with verified evidence even
when a target fails. Report failures and regressions directly.

Save the first two actual training episodes starting at or after each boundary
at 0%, 25%, 50%, 75% and 95% of each budget. Capture raw actions/transitions,
actual step bounds and complete/cutoff status. Use the existing passive recorder
without extra policy or RNG calls. These ten bounded records per run support
Learning replay. Final trained replay uses selected checkpoints separately.

## Limits and integrity

Cap new work at 22 runs, 3,220,000 training steps, two processes, 900 wall seconds
per run, 4,500 wall seconds for the suite including verification, and 1 GiB new
artifacts. Pilot projection uses measured training cost scaled by 1.3, monitoring
scaled by budget and twice measured validation cost, across the worst-case
matrix divided by two workers, plus 180 seconds. Stop before the main screen if
the projection exceeds the suite cap. Outcomes can change cost, so the actual
wall cap remains enforced. Preserve any incomplete evidence and report failures.

Hash every existing artifact/data file before starting and check after each phase
and publication. Keep every run source snapshot, configuration, ledger, visitation,
monitor records, selected/final model and bounded recordings. Preserve periodic
models and full validation trajectories locally for a deep audit. Those large
reproducible files are excluded from version control following existing project
conventions. Exact training resume remains unsupported.

Before publication, replay every validation transition through the environment,
reconcile aggregates and scenario coverage, compare actions and recorded neural
diagnostics with the selected checkpoint, check inference fingerprints, all
periodic checkpoint checksums, budgets, chronological training recordings and
source identities. Publish only after all checks pass. Keep the M9 dashboard
page separate from previous studies. Failed or weak models remain inspectable.

Unseen maps, dynamics noise, moving obstacles, sensors, continuous control, ROS
and physical hardware remain deferred.
