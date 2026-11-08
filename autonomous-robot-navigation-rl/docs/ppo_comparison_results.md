# Four RL algorithms: measured development validation

Completed 8 November 2026. Q-learning, SARSA and DQN reuse the original Part A
baseline runs. PPO adds five newly trained runs. All families use seeds 7, 19,
43, 67, 91, 200,000 training interactions per seed, the same task and 928
validation scenarios per seed. Checkpoints use the registered 30-pair validation
monitor: highest success, then SPL, then earliest checkpoint.

| Algorithm | Success mean ± sample seed SD | Collision mean | Timeout mean | SPL mean |
| --- | ---: | ---: | ---: | ---: |
| Q-learning | 65.24% ± 1.64 pp | 0.17% | 34.59% | 0.6192 |
| SARSA | 21.94% ± 5.37 pp | 3.94% | 74.12% | 0.2070 |
| DQN | 5.32% ± 1.04 pp | 31.77% | 62.91% | 0.0530 |
| PPO | 15.22% ± 13.46 pp | 59.91% | 24.87% | 0.1520 |

SD is across independent training seeds, not a confidence interval. These are
baseline runs; the separately tuned Q-learning/SARSA runs are excluded.
Neither DQN nor PPO has a hyperparameter-tuned variant in this comparison.

## PPO runs

| Run | Selected training interaction | Successes / 928 | Collisions | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| `m7-ppo-seed7` | 180,000 | 260 | 668 | 0 |
| `m7-ppo-seed19` | 190,000 | 292 | 636 | 0 |
| `m7-ppo-seed43` | 150,000 | 76 | 500 | 352 |
| `m7-ppo-seed67` | 10,000 | 25 | 441 | 462 |
| `m7-ppo-seed91` | 20,000 | 53 | 535 | 340 |

Every run completed its full 200,000-interaction budget, including those whose
selected checkpoint was early. Each saved 20 monitoring checkpoints. There was
one fixed PPO configuration, no search, no post-pilot configuration change and
no retry to discard a poor seed. The independent 10,000-interaction seed-101
pilot is excluded from these statistics.

The PPO matrix took 654.21 seconds wall time, including monitoring and export;
572.26 seconds were recorded as training time excluding construction, monitoring,
checkpoint I/O and validation/export. Runtime comparisons with older algorithms
are descriptive because the historical runs were executed separately. Artifacts
used approximately 29.19 MB at matrix completion, including pilot and study files.

## Evidence and limitations

The [verification record](../artifacts/studies/four-rl-v1/verification.json)
passed all 20 run audits. It reconciles training budgets, validation coverage,
saved aggregates and checkpoint selection. For PPO it also verifies source
snapshots, all 100 monitoring checkpoints and 241,569 validation actions against
the loaded policy's action probabilities and state values, without mutating the
policy. All 128 registered historical comparison files retained their hashes.

Inspect the [summary](../artifacts/studies/four-rl-v1/summary.json),
[per-run CSV](../artifacts/studies/four-rl-v1/comparison.csv),
[comparison plot](../artifacts/studies/four-rl-v1/comparison.png) and
[monitoring curves](../artifacts/studies/four-rl-v1/learning.png).
The dashboard exposes these under Agent comparison and Learning analysis;
the simulator's Saved run selector exposes each PPO run and actual failures.
PPO diagnostics show categorical action probabilities and state value.

PPO remains unreliable under this budget and representation: three of its five
seeds achieved below 10% success, and collisions account for most pooled failures.
Its higher mean than DQN does not establish statistical superiority or general
navigation competence. No additional tuning has been performed.

This is an exploratory post-release comparison on development validation, on a
single familiar map. It is not a new blinded final evaluation, unseen-map
transfer or a hardware result. No PPO final-test policy execution was added.
The original frozen final archive, selections and reported final results remain
unchanged. Other deferred additions remain deferred.

All 90 project tests passed after publication. Automated dashboard checks cover
replay, custom demonstrations, matched playback and the verified table. A separate
preservation check examined 7,249 protected files with no unauthorized changes and
verified all 184 archived frozen inputs against their seal.
Those initial visual attempts were blocked by unreliable Chrome URL detection.
A later owner-opened local tab enabled successful native Chrome verification;
the current observations and playback correction are recorded in
[plan closure](plan_closure_status.md). The 90-test count above records the
original PPO extension; the current suite has 96 passing tests.
