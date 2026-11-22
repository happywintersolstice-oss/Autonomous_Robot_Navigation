# Learning replays for all four algorithms

Fresh recorded-training demonstrations are available for Q-learning, SARSA, DQN and PPO. These recordings are separate from the original runs and frozen results.

| Algorithm | Training steps | Episodes recorded | Recorded actions |
| --- | ---: | ---: | ---: |
| Q-learning | 200,000 | 10 | 50 |
| SARSA | 200,000 | 10 | 260 |
| DQN | 200,000 | 10 | 148 |
| PPO | 200,000 | 10 | 854 |

Total: **800,000 new training steps**, **40 recordings** and **1,312 recorded actions**. The training suite took 384.86 seconds including monitoring, export and the initial audit. Artifacts occupy approximately 22.19 MiB.

## Open a learning replay

1. Open the local dashboard and choose **Learning replay**.
2. In **Training run**, choose a **Recorded training demo** for the desired algorithm.
3. Choose an episode at the **0%, 25%, 50%, 75% or 95% training stage**.
4. Use Play, Pause, Next step or the replay-position slider.

Each fresh run uses seed 1109, the existing baseline settings and 200,000 training steps. Ten actual episodes were recorded per run: the first two starting at or after each declared boundary. They were not selected by outcome. DQN happened to collide in all ten sampled episodes; late training does not guarantee success. Samples use different scenarios, so they do not establish an improvement curve or an algorithm ranking.

Original historical trajectories were not reconstructed. The older M8 DQN first-three-episode recordings remain available. These new demonstrations do not replace a saved model, promote a controller, modify the fixed-map task, enter the original benchmark tables or open a new final-test evaluation.

## Verification

All **136 tests passed in 53.59 seconds**, including recording-enabled/disabled learning equivalence for all four families, schedule validation, checksum/tamper checks, all-family late-stage replay and the complete existing suite.
The artifact audit verified each recording against its declared chronological window, complete training CSV, visitation totals, training scenario, seed, checksum and raw environment transitions. All source snapshots retain the hash captured during recording.
The initial aggregate audit rejected one SARSA return because Python compensated summation produced -6.92 while the sequential CSV accumulator produced -6.920000000000002. Only this aggregate comparison now allows absolute roundoff up to 1e-12; raw actions and per-step rewards remain checked exactly. Re-auditing required no retraining or raw-data edits. A regression test accepts tiny accumulation noise and rejects a 0.01 reward change.
All **8,816 pre-existing protected files** and **184 archived frozen inputs**, including the original freeze seal, verified unchanged. No new final-test policy executions occurred.

Live browser verification of the newly added algorithm choices is pending; all four choices and late-stage playback have passed the Streamlit application harness.

Artifacts: `artifacts/studies/learning-replays-v1/{plan,manifest,run_results,verification}.json`. Protocol: `docs/learning_replay_protocol.md`. Re-run integrity checks without training with `python scripts/record_learning_replays.py --verify-existing`.
