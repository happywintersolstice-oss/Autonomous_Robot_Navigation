# Five-seed neural confirmation and native dashboard verification

Completed on 9 November 2026 following the owner request for both five-seed expansion and a visual dashboard check. The original three-seed outcomes were already known. This is a staged exploratory development-validation extension. The original three-seed bundle remains available separately.

## Five-seed results

| Algorithm and settings | Success mean ± seed SD | Collision mean | Timeout mean | SPL mean ± seed SD | Mean return |
| --- | --- | --- | --- | --- | --- |
| DQN baseline | 6.88% ± 2.25 pp | 20.80% | 72.33% | 0.066929 ± 0.021067 | -1.8088 |
| DQN selected c01 | 7.37% ± 0.55 pp | 25.04% | 67.59% | 0.072190 ± 0.004943 | -1.8770 |
| PPO baseline | 19.44% ± 9.91 pp | 65.71% | 14.85% | 0.193967 ± 0.099710 | -1.6667 |
| PPO selected c01 | 23.69% ± 14.62 pp | 56.38% | 19.94% | 0.235088 ± 0.144389 | -0.8778 |

| Algorithm | Mean paired success change | Paired seed SD | Wins / losses / ties | Mean paired SPL change |
| --- | --- | --- | --- | --- |
| DQN | +0.50 pp | 1.95 pp | 4 / 1 / 0 | +0.005261 |
| PPO | +4.25 pp | 9.06 pp | 4 / 1 / 0 | +0.041121 |

The added seeds weaken the earlier gains. DQN has a small five-seed success increase, with more collisions and a slightly lower mean return than its matched baseline. PPO has a larger average success gain and fewer collisions, with more timeouts and substantial seed variability. Both selected configurations miss all three declared target means of 80% success, at most 2% collisions and SPL at least .75.

Five training seeds are the comparison units. The 928 routes share a familiar map and development-validation history. Sample SD describes training-seed variability. This study provides no fresh-holdout result or episode-level significance claim. The original final controller remains Q-learning.

## Two added seeds separately

| Algorithm | Added-seed baseline success | Added-seed selected success | Mean paired change | Paired seed SD |
| --- | --- | --- | --- | --- |
| DQN | 8.78% | 7.76% | -1.02 pp | 2.51 pp |
| PPO | 23.11% | 22.68% | -0.43 pp | 12.04 pp |

The original three seeds averaged success changes of +1.51 pp for DQN and +7.36 pp for PPO. The two added seeds averaged declines for both algorithms. Each family had one added-seed loss and one added-seed gain. The frozen c01 choices were kept throughout.

## Every paired seed

| Algorithm | Training seed | Baseline goals / 928 | Selected goals / 928 | Success change | Baseline / selected checkpoint step |
| --- | --- | --- | --- | --- | --- |
| DQN | 7307 | 42 | 61 | +2.05 pp | 50,000 / 10,000 |
| DQN | 7411 | 59 | 65 | +0.65 pp | 90,000 / 10,000 |
| DQN | 7523 | 55 | 72 | +1.83 pp | 30,000 / 10,000 |
| DQN | 7607 | 98 | 72 | -2.80 pp | 160,000 / 20,000 |
| DQN | 7717 | 65 | 72 | +0.75 pp | 40,000 / 10,000 |
| PPO | 7307 | 71 | 97 | +2.80 pp | 20,000 / 80,000 |
| PPO | 7411 | 260 | 291 | +3.34 pp | 150,000 / 120,000 |
| PPO | 7523 | 142 | 290 | +15.95 pp | 180,000 / 130,000 |
| PPO | 7607 | 138 | 55 | -8.94 pp | 100,000 / 200,000 |
| PPO | 7717 | 291 | 366 | +8.08 pp | 190,000 / 190,000 |

Each of the 20 combined confirmation runs completed 200,000 training steps. Reused original seeds are 7307, 7411 and 7523. Added seeds are 7607 and 7717. Selected checkpoint ages may precede the completed training budget. All runs retain the original four-coordinate observation, network architecture, map, rewards, training split, fixed 30-route monitor and 928-route validation.

| Algorithm and settings | Return seed SD | Successful route length mean ± seed SD | Route contributing seeds | Collision seed SD | Timeout seed SD |
| --- | --- | --- | --- | --- | --- |
| DQN control | 0.5688 | 4.767 ± 1.326 actions | 5 | 11.21 pp | 9.65 pp |
| DQN tuned | 0.1224 | 4.915 ± 0.659 actions | 5 | 5.50 pp | 5.94 pp |
| PPO control | 1.0599 | 4.142 ± 0.261 actions | 5 | 4.50 pp | 14.15 pp |
| PPO tuned | 1.5103 | 4.500 ± 0.829 actions | 5 | 11.97 pp | 24.58 pp |

Route length is conditional on success. Different successful route sets prevent an isolated efficiency conclusion from these means.

## Failure recovery and costs

Four added DQN runs completed in neural-confirmation-five-v1. PPO baseline seed 7607 then failed at the 50,000-step monitoring boundary with Windows OSError 22 while rewriting manifest.json. The paired worker was stopped after its 40,000-step checkpoint. Both interrupted attempts are preserved.

The JSON writer now creates a complete temporary file beside its target and atomically replaces the target, with bounded retries for transient Windows access errors. Readers also retry transient access errors. Regression tests verify byte-format compatibility, preservation of old data on permanent failure, retry behavior and concurrent readers. The precise Windows locking source remains unconfirmed.

neural-confirmation-five-v2 reuses the four completed DQN additions verbatim and runs the four remaining PPO conditions from step zero. Their new run names have an attempt2 suffix identifying the recovery cohort. Seed 7717 had no earlier PPO training attempt. All configurations differ from their original condition only in seed and run identity. No performance-based rerun or configuration reselection occurred.

All learning-engine files match the original source hashes except JSON input/output functions and their supporting imports. A syntax comparison verifies all other core definitions unchanged. At the shared saved prefixes, the failed/restarted PPO baseline at 50k and selected settings at 40k have identical learned parameters, optimizer state and update/timestep counters. These checks support the I/O-only nature of the repair.

| Algorithm and settings | Training seconds mean ± seed SD | Monitoring seconds mean | Validation seconds mean | Decision microseconds mean ± seed SD | Checkpoint bytes mean |
| --- | --- | --- | --- | --- | --- |
| DQN control | 115.15 ± 36.12 | 8.79 | 10.80 | 65.59 ± 32.04 | 102103 |
| DQN tuned | 113.64 ± 36.18 | 9.46 | 10.47 | 55.13 ± 8.84 | 102088 |
| PPO control | 131.72 ± 38.54 | 5.73 | 5.71 | 162.42 ± 71.37 | 143256 |
| PPO tuned | 132.16 ± 35.52 | 6.78 | 9.19 | 166.63 ± 36.47 | 143251 |

Timings combine separately executed original and added runs on this machine. Two one-thread learners ran concurrently within each execution phase. Training excludes monitoring/export, validation includes recording/export, and decision latency is a 100-query mean excluding rendering. File size does not measure memory use. These are descriptive costs under changing machine load.

The eight successful additions total 1,600,000 training steps. The interrupted PPO attempts add at least 90,000 steps, with the killed worker's exact final count unavailable. The first execution phase lasted 437.67 seconds before failure. Recovery and the combined audit took 474.10 seconds, within the recovery's 1,800-second cap. These phase timings exclude intervening investigation, repair and visual/regression checks. All extension bundles and new run directories occupy 49.69 MiB locally.

## Audit and preservation

All 20 combined run audits passed, covering 18,560 validation episodes, 1,656,863 replayed validation actions, 2,743 recorded training actions and 400 periodic checkpoint checksums. Ledgers, visitation budgets, monitoring selection, recorded neural diagnostics and source identities reconcile. All 200 combined training recordings remain available, including 80 from the added seeds.

Independent checks verified all 11,563 pre-extension artifact/data files and all 12,142 files protected before recovery. All 184 sealed final-release inputs match their archive and seal. An older cross-session protected-file list matches 8,815 of 8,816 files. Its root README entry differs because that file was updated in an earlier committed change. The root README matches the starting commit cda55b4 and has no changes from this task. No energy or sources/ files were modified. New final-test policy executions are zero.

## Native dashboard and regression verification

Native Chrome was inspected at 1536 × 912 and 854 × 722 pixels. The five-seed table, staged-expansion note, original/five-seed selector, scientific figures, added-seed section and per-seed cost table were checked. Switching studies restored the original three-seed values and returned correctly to five seeds. Narrow tables retain their normal internal horizontal scrolling.

The first narrow check found that the second-model selector was squeezed beside the route selector when the sidebar was open. Those fields now stack at narrow widths, and the full-width selector was rechecked in the native browser. Playback controls use two button columns with a full-width speed slider.

An illustrative saved validation route, r8c1-r5c0, was chosen to exercise successful playback and failure holding. PPO selected settings seed 7307 reached [5,0] in four actions, return 9.96 and SPL 1.000. Its matched baseline held at its one-action collision at [8,1], return -5.01. Seeking to frame 2 showed the selected model at [7,0] with return -0.02 while the baseline remained fixed. Reset restored both to action 0 and zero return. A separate single step on r0c0-r0c3 moved both PPO models to [1,0] for -0.01. These are display checks outside benchmark aggregation. Manual Pause also passes the automated comparison tests.

All 158 tests passed in 68.21 seconds after correcting the late-learning-replay test to reference the completed recovery run. Both scientific figures were visually inspected. The local dashboard remains available at http://127.0.0.1:8518.

## Evidence and remaining limits

- [Declared protocol and recovery addendum](neural_confirmation_protocol.md) and [extension configuration](../configs/neural_confirmation.yaml).
- [Verified five-seed summary](../artifacts/studies/neural-confirmation-five-v2/summary.json), [all paired results](../artifacts/studies/neural-confirmation-five-v2/run_results.json) and [deep audit](../artifacts/studies/neural-confirmation-five-v2/verification.json).
- [Native and regression verification](neural_confirmation_verification.json) and [original three-seed results](milestone9_results.md).

The staged expansion improves the variability estimate but uses previously inspected development validation. Both models remain weak. Independent four-family final evaluation remains outstanding. Unseen maps, dynamics noise, moving obstacles, continuous control, sensors, ROS and hardware remain deferred.
