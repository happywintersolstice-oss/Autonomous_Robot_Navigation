# Milestone 6 — final evaluation results

The core release is complete. All settings, checkpoints, vote rules and scientific
analysis were frozen before final-test execution. Selected system remains
`m4-tuned-q-learning-seed101`; no holdout-based promotion or further tuning.

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

The selected single policy reached 840/928 goals (90.52%), with 7 collisions,
81 timeouts and SPL .867870. Its matched original-setting control reached 623/928
(67.13%), 1 collision, 304 timeouts and SPL .636640. Voting reached 848/928 and
outperformed that single on the holdout; it was not retrospectively selected.
Q-learning's paired five-seed gain is +24.91 pp (SD 3.03 pp), with a .32-pp mean
collision increase. SARSA improves but misses every declared target.

Evaluation completed 42 records / 38,976 episodes / 3,456,940 actions in 109.68 seconds.
All saved transitions/scores were replay-audited; audit plus analysis took 87.87
seconds. BFS is optimal on every final case. All 184 frozen inputs remain unchanged.
64 tests pass; five actual final-data dashboard page checks pass separately.
No new evaluation training occurred; the isolated pre-freeze reproduction added
200k interactions and reproduced the selected table/validation metrics exactly.

Final-test access now occurs only in the dedicated frozen runner. Generic trainers,
CLI evaluation and the simulator retain validation/custom restrictions. The UI
shows audited aggregate final results. Final data may no longer support new model
selection; use a new holdout or explicit exploratory labels for later changes.

See [technical report](technical_report.md), [protocol](milestone6_protocol.md),
[verification](milestone6_verification.json), [preservation](milestone6_preservation.json)
and [frozen evidence](../artifacts/studies/final-v1/summary.json). Known limits include
familiar-map evaluation, residual failures, weak SARSA/DQN, one ensemble, no clean
network dependency-install verification and inference-only checkpoints.
