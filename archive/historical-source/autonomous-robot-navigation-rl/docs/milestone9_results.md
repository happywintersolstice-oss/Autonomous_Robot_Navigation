# Milestone 9 measured neural tuning results

Completed on 9 November 2026. This preliminary study screened four DQN and four PPO configurations on one development training seed, then compared baseline and selected settings on three fresh matched confirmation seeds. Both families selected c01 before confirmation began.

Each confirmation run completed 200,000 training steps and evaluated its monitoring-selected checkpoint on all 928 development-validation routes. All six core milestones and previous studies remain historical evidence. The original final controller is Q-learning.

## Confirmation results

| Algorithm and settings | Success mean ± seed SD | Collision mean | Timeout mean | SPL mean ± seed SD | Mean return |
| --- | --- | --- | --- | --- | --- |
| DQN baseline | 5.60% ± 0.96 pp | 24.28% | 70.11% | 0.054753 ± 0.008537 | -2.0662 |
| DQN selected c01 | 7.11% ± 0.60 pp | 22.84% | 70.04% | 0.069696 ± 0.005008 | -1.8416 |
| PPO baseline | 16.99% ± 10.29 pp | 65.27% | 17.74% | 0.169184 ± 0.103853 | -1.9460 |
| PPO selected c01 | 24.35% ± 12.04 pp | 62.18% | 13.47% | 0.243534 ± 0.120386 | -0.9730 |

| Algorithm | Mean paired success change | Paired seed SD | Mean paired SPL change | Wins / losses / ties |
| --- | --- | --- | --- | --- |
| DQN | +1.51 pp | 0.75 pp | +0.014943 | 3 / 0 / 0 |
| PPO | +7.36 pp | 7.44 pp | +0.074350 | 3 / 0 / 0 |

DQN improved on each seed, with a small mean success gain. Approximately 70% of its routes still time out. PPO improved on each seed with larger variation in the size of the gain. Its selected-settings collision mean remains above 62%. Both families miss all three aspirational targets of 80% success, at most 2% collisions and SPL at least .75.

The three seeds are the independent comparison units. Sample SD describes training-seed variability. The 928 routes share one familiar map and are reused development validation. No episode-level significance, fresh-holdout conclusion or unseen-map generalization is claimed. The original final selection is retained.

## Every confirmation seed

| Algorithm | Seed | Baseline goals / 928 | Selected goals / 928 | Success change | Baseline / selected checkpoint step |
| --- | --- | --- | --- | --- | --- |
| DQN | 7307 | 42 | 61 | +2.05 pp | 50,000 / 10,000 |
| DQN | 7411 | 59 | 65 | +0.65 pp | 90,000 / 10,000 |
| DQN | 7523 | 55 | 72 | +1.83 pp | 30,000 / 10,000 |
| PPO | 7307 | 71 | 97 | +2.80 pp | 20,000 / 80,000 |
| PPO | 7411 | 260 | 291 | +3.34 pp | 150,000 / 120,000 |
| PPO | 7523 | 142 | 290 | +15.95 pp | 180,000 / 130,000 |

Every run completed its full budget even when an earlier checkpoint was selected. A larger successful-route mean can reflect success on different routes. Conditional route length alone does not isolate efficiency.

| Algorithm and settings | Successful route length mean ± seed SD | Contributing seeds | Return seed SD | Collision seed SD | Timeout seed SD |
| --- | --- | --- | --- | --- | --- |
| DQN control | 4.084 ± 0.279 actions | 3 | 0.4695 | 12.55 pp | 11.88 pp |
| DQN tuned | 4.900 ± 0.597 actions | 3 | 0.0680 | 4.46 pp | 5.05 pp |
| PPO control | 4.094 ± 0.168 actions | 3 | 1.0562 | 5.89 pp | 16.08 pp |
| PPO tuned | 4.193 ± 0.685 actions | 3 | 1.0952 | 11.29 pp | 23.33 pp |

## Screen and selected settings

All candidates retained the original four absolute normalized coordinates, [64,64] networks, fixed map/rewards and existing training backend. Screening used seed 7201 and 100,000 steps per candidate. These screen scores are excluded from confirmation means.

| Algorithm | Candidate | Success | SPL | Collisions |
| --- | --- | --- | --- | --- |
| DQN | c00 | 3.66% | 0.036279 | 24.57% |
| DQN | c01 | 6.03% | 0.059280 | 30.93% |
| DQN | c02 | 4.09% | 0.040948 | 41.59% |
| DQN | c03 | 5.06% | 0.050377 | 28.34% |
| PPO | c00 | 28.56% | 0.285560 | 71.44% |
| PPO | c01 | 31.25% | 0.312500 | 68.75% |
| PPO | c02 | 5.82% | 0.058190 | 63.58% |
| PPO | c03 | 4.74% | 0.047414 | 62.50% |

DQN c01 uses learning rate .001, target updates every 250 steps and an exploration decay fraction of 1.0. PPO c01 uses learning rate .001, rollouts of 500 steps and entropy coefficient .01. All other original hyperparameters remain fixed. Bundled candidates cannot separate the effect of each changed factor. Candidate ranking used success, SPL, lower collisions and ascending candidate ID, with the choice frozen before confirmation.

## Recorded cost

| Algorithm and settings | Training seconds mean ± SD | Monitoring seconds mean | Validation seconds mean | Decision microseconds mean ± SD | Checkpoint bytes mean |
| --- | --- | --- | --- | --- | --- |
| DQN control | 88.78 ± 0.55 | 7.84 | 9.29 | 54.78 ± 5.07 | 102101 |
| DQN tuned | 87.25 ± 0.03 | 8.51 | 9.32 | 57.73 ± 7.33 | 102086 |
| PPO control | 104.28 ± 4.92 | 5.68 | 5.65 | 129.09 ± 12.44 | 143254 |
| PPO tuned | 106.99 ± 6.34 | 4.65 | 4.54 | 140.61 ± 9.66 | 143247 |

Two separate one-thread learners ran concurrently. Timings are descriptive for this machine and its concurrent load. Training excludes construction, monitoring and export. Validation includes recording/export. Decision latency is a 100-query mean excluding rendering. File size does not measure memory use. Full per-seed metrics and costs are retained in the summary.

The verified suite completed 22 runs and 3,220,000 training steps in 1073.28 seconds (17.89 minutes), including its audit, within the 4,500-second cap. The initial publication used 97.64 MiB within the 1 GiB cap. Total recorded training, monitoring and validation costs across pilots, screening and confirmation were 1580.87, 107.71 and 152.83 seconds respectively. Concurrent process costs are additive and differ from elapsed time.

An initial attempt was stopped while investigating delayed startup output. Its two seed-7101 pilots each reached 10,000 training steps, then were interrupted during validation/export. That additional 20,000-step setup cost is preserved under neural-tuning-v1 and excluded from comparisons. Total training across both attempts was 3,240,000 steps. The fresh suite used pilot seed 7103. Screening/confirmation seeds and candidate settings were unchanged before outcomes from those phases.

## Verification and dashboard

All 22 run audits passed. The verifier replayed 1,751,597 validation actions and 3,464 actual training actions, reconciled 20,416 validation episodes, all 322 periodic checkpoint checksums, selected checkpoint rules, source snapshots, ledgers and visitation budgets. Selected-model actions and Q scores/PPO probabilities matched saved records. Inference state stayed fixed. All 9,371 existing artifact/data files retained their pre-study hashes.

An independent preservation check also verified all 8,816 original protected files, the 184 archived frozen inputs and their seal. The root plan and existing project files outside this project were preserved. No energy project or sources/ directory was present in this checkout. New final-test policy executions were zero.

All 148 tests passed in 40.45 seconds after the final figure adjustment, including both matched-model buttons and the complete existing suite. Both scientific figures were visually inspected. The learning-curve axis labels were refined after training and the published figure checksums refreshed. The source snapshots retain the exact training source. This presentation adjustment changed no learned model or measured result.

The owner confirmed the preceding dashboard visuals before M9. The new Neural tuning research page and replay buttons were checked through the Streamlit application harness. A fresh native-browser visual inspection of this new research page was not performed.

Choose Neural tuning in Dashboard view for results, monitoring curves, costs and screening details. Compare DQN models and Compare PPO models open the first predeclared confirmation seed with synchronized replay. Every fresh run includes ten actual training episodes sampled across five declared training stages in Learning replay. Trained replay uses the saved selected model separately.

## Evidence and remaining work

- [Registered protocol](milestone9_protocol.md) and [configuration](../configs/neural_tuning.yaml).
- [Verified summary](../artifacts/studies/neural-tuning-v2/summary.json), [frozen choices](../artifacts/studies/neural-tuning-v2/selected_configurations.json) and [full audit](../artifacts/studies/neural-tuning-v2/verification.json).
- [Paired confirmation figure](../artifacts/studies/neural-tuning-v2/paired_neural_tuning.png) and [monitoring curves](../artifacts/studies/neural-tuning-v2/neural_tuning_learning.png).

Five-seed confirmation and an independently designed four-family final evaluation remain future work. Weak success, residual collisions/timeouts, one-seed screening uncertainty and familiar-map development reuse limit these results. Unseen maps, noise, moving obstacles, continuous control, sensors, ROS and hardware remain deferred.
