# History-aware SARSA recovery: measured development results

Completed and verified 17 November 2026 under the [registered protocol](loop_intervention_protocol.md).
This is development validation on four layouts, with zero new training and zero final-layout evaluations. The historical controller and M13 deterministic diagnosis are unchanged.

| Measure | Base SARSA | History-aware recovery |
|---|---:|---:|
| Mean success | 63.37% | 67.77% |
| Mean SPL | 0.623572 | 0.652713 |
| Mean collision | 0.69% | 1.04% |

Success improved by 4.40 percentage points, missing the registered +5-point promotion requirement. SPL improved and mean collision stayed below 2%. No promotion was made.
The nine rows represent three pretrained policies with three intervention RNG streams each. Each deterministic baseline is evaluated once and reused for its three matched streams. They are not nine independent training seeds.

| Training seed | Recovery RNG | Base goals /192 | Recovery goals /192 | Collisions | Timeouts | Interventions |
|---|---|---:|---:|---:|---:|---:|
| 12101 | 17011 | 125 | 137 | 4 | 51 | 87 |
| 12101 | 17021 | 125 | 136 | 4 | 52 | 90 |
| 12101 | 17033 | 125 | 136 | 4 | 52 | 87 |
| 12203 | 17011 | 117 | 126 | 0 | 66 | 54 |
| 12203 | 17021 | 117 | 126 | 0 | 66 | 54 |
| 12203 | 17033 | 117 | 126 | 0 | 66 | 54 |
| 12301 | 17011 | 123 | 128 | 2 | 62 | 93 |
| 12301 | 17021 | 123 | 128 | 2 | 62 | 93 |
| 12301 | 17033 | 123 | 128 | 2 | 62 | 92 |

Across the paired streams, 76 baseline timeouts became successes and six became collisions. All baseline successes remained successes. There were 539 remaining timeouts, all with persistent periodic suffixes under the new trace-only definition. A repeated state alone is insufficient to classify a history-aware controller.

Execution took 21.50 seconds against the 180-second cap. The initial verification measured 4,595,068 bytes against a 256 MiB cap and checked 246,867 recorded actions, including repeated checks of the reused deterministic baselines.
Per-run base/wrapper latency probes (100 calls each) are retained in paired.csv and summary.json. These local short probes vary with load and do not establish a general runtime improvement.

![Paired success and SPL](../artifacts/studies/loop-recovery-dev-v1/paired.png)

Figure metadata: grid-v1 dynamics; full-occupancy-coordinates-v1 input, egocentric-map-goal-v1 features; development validation (four layouts, 192 routes per policy); M11 monitor-selected checkpoints (success, SPL, earliest); training seeds 12101/12203/12301, intervention RNG 17011/17021/17033. Lines are paired policy/RNG observations, with the same deterministic baseline reused across RNG streams.

Files: [paired rows](../artifacts/studies/loop-recovery-dev-v1/paired.csv), [summary](../artifacts/studies/loop-recovery-dev-v1/summary.json), [verification](../artifacts/studies/loop-recovery-dev-v1/verification.json), [frozen inputs](../artifacts/studies/loop-recovery-dev-v1/freeze.json). All full recorded trajectories and bounded illustrative examples are retained.

Open Results > Fixed layout > Optimization and robustness > Matched recovery recording for synchronized saved replay. Controller history, training seed, recovery RNG, development layout and shared action index are shown.

The margin and recovery cap were chosen before scoring this registered run and were not retuned after its result. This narrow intervention tests one hypothesis; it does not establish training causes or physical safety. A confirmatory generalization claim requires a fresh versioned holdout after freeze.
