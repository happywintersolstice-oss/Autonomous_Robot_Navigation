# Structured maps and reward design: measured results

Completed 17 November 2026 under the [registered protocol](navigation_lab_protocol.md). New benchmark `structured-navigation-v2`; study `map-reward-study-v1`. All prior map data, model identities and frozen results are preserved.

## Map improvements

The original bottleneck map has **zero degree-one dead ends**, 21 obstacles and graph diameter 19. The new maps are connected but have purposefully constrained routes.

| Map | Obstacles | Dead ends | Deepest branch | Junctions | Cycle rank | Diameter | Train / validation / reserved pairs |
|---|---:|---:|---:|---:|---:|---:|---|
| dead-end-maze | 51 | 5 | 10 | 3 | 0 | 36 | 1622 / 365 / 365 |
| looped-corridors | 48 | 3 | 19 | 9 | 4 | 34 | 1836 / 408 / 408 |
| rooms-and-doorways | 42 | 3 | 4 | 26 | 16 | 22 | 2304 / 501 / 501 |
| deep-cul-de-sacs | 51 | 5 | 28 | 3 | 0 | 40 | 1618 / 367 / 367 |

![New map gallery](../data/navigation-lab/structured-navigation-v2/gallery.png)

Orange markers identify actual degree-one dead ends. Topology was chosen before training, not selected by model outcomes. Loop rank counts independent cycles in the free-cell graph. Held-out starting pairs are on familiar maps; this is not another unseen-layout evaluation.

## Does changing reward remove loops?

Three seeds per map and condition completed 100,000 actions each: 36 runs and **3,600,000 training actions**. Checkpoints use the same 18-route monitor and success/SPL/earliest rule. The complete validation audit checked **14,769 episodes and 639,148 actions**. All reloaded Q values/actions, reward components, environment transitions, selected checkpoint ages and aggregates matched. Model arrays stayed unchanged in validation. Reserved pairs had zero evaluation episodes.

| Reward | Success mean | Success seed SD | Collision mean | Timeout mean | SPL mean | Looping timeouts |
|---|---:|---:|---:|---:|---:|---:|
| original | 77.30% | 1.14 pp | 2.15% | 20.54% | 0.764681 | 978 |
| time-cost | 78.90% | 0.92 pp | 2.16% | 18.94% | 0.779388 | 916 |
| potential | 77.29% | 2.54 pp | 7.09% | 15.62% | 0.756179 | 737 |

Means weight maps equally within each training seed, then seeds equally. Sample SD describes three training seeds after map averaging. Looping-timeout counts pool all actual validation cases (4,923 per condition); the maps have unequal route counts, so counts and equal-map means have different weighting.

Increasing the step cost from -.01 to -.05 improved mean success by **1.60 percentage points**, raised SPL from .764681 to .779388, and reduced looping timeouts from 978 to 916. Mean collision was nearly unchanged (2.15% to 2.16%). This is a modest development improvement, not evidence that loops are solved.

Potential shaping on top of the stronger time cost reduced looping timeouts to 737, but mean success was 77.29%, collision rose to **7.09%**, and SPL fell to .756179. It is a **failed safety/quality trade-off**, retained in the comparison rather than recommended. Finite learning and monitor-selected checkpoints do not inherit an ideal convergence guarantee from the shaping theorem.

The UI defaults to the time-cost condition as an illustrative new-map pilot, with the training seed chosen in advance (18101). It does not replace the historical selected controller or claim deployment readiness. All three conditions and all seeds remain available.

## Why repeated states occur

A deterministic stateless policy picks the same action for the same robot/goal observation. A mistaken pair of decisions can therefore repeat until the action cap. Sparse credit assignment, incomplete table coverage and earlier shared-feature approximation can each contribute. The previous -.01/action loop approaches a discounted continuing cost of -1 at gamma .99, whereas an immediate collision costs -5.01. This can make looping a locally attractive error, although successful routes on these reachable grids are preferable for an optimal policy.

The potential experiment uses `0.2 * (0.99 * Phi(next) - Phi(current))`, where Phi is negative Manhattan distance and true terminal potential is zero. It uses no BFS route, action mask or visited-cell memory. The [potential-shaping construction](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) preserves ideal optimal policies for the corresponding base task under its assumptions; it does not promise a finite-budget improvement. Stronger step costs change the base objective and must be assessed separately.

A blanket revisit penalty was deliberately excluded: exploration or a history-aware controller can need to backtrack out of a dead end. Such a penalty would also make reward depend on episode history unless that history becomes part of the state. No cycle termination or playback escape rule hides the measured failures.

## Map and seed detail

| Map | Reward | Seed | Selected step | Success | Collision | Timeout | SPL |
|---|---|---|---:|---:|---:|---:|---:|
| dead-end-maze | original | 18101 | 100000 | 76.99% | 1.92% | 21.10% | 0.7699 |
| dead-end-maze | time-cost | 18101 | 100000 | 76.99% | 2.19% | 20.82% | 0.7699 |
| dead-end-maze | potential | 18101 | 100000 | 78.36% | 9.32% | 12.33% | 0.7836 |
| dead-end-maze | original | 18203 | 100000 | 78.08% | 2.47% | 19.45% | 0.7808 |
| dead-end-maze | time-cost | 18203 | 100000 | 77.26% | 1.92% | 20.82% | 0.7726 |
| dead-end-maze | potential | 18203 | 100000 | 78.36% | 9.32% | 12.33% | 0.7836 |
| dead-end-maze | original | 18301 | 100000 | 76.44% | 1.92% | 21.64% | 0.7644 |
| dead-end-maze | time-cost | 18301 | 100000 | 77.26% | 2.74% | 20.00% | 0.7726 |
| dead-end-maze | potential | 18301 | 90000 | 74.79% | 8.49% | 16.71% | 0.7479 |
| looped-corridors | original | 18101 | 100000 | 81.62% | 2.45% | 15.93% | 0.8121 |
| looped-corridors | time-cost | 18101 | 100000 | 80.88% | 1.23% | 17.89% | 0.8033 |
| looped-corridors | potential | 18101 | 90000 | 76.96% | 6.86% | 16.18% | 0.7478 |
| looped-corridors | original | 18203 | 90000 | 72.30% | 1.47% | 26.23% | 0.7193 |
| looped-corridors | time-cost | 18203 | 100000 | 83.33% | 1.23% | 15.44% | 0.8267 |
| looped-corridors | potential | 18203 | 90000 | 77.94% | 6.13% | 15.93% | 0.7527 |
| looped-corridors | original | 18301 | 100000 | 82.11% | 1.72% | 16.18% | 0.8138 |
| looped-corridors | time-cost | 18301 | 100000 | 82.11% | 2.21% | 15.69% | 0.8116 |
| looped-corridors | potential | 18301 | 100000 | 82.84% | 6.86% | 10.29% | 0.7984 |
| rooms-and-doorways | original | 18101 | 100000 | 82.63% | 2.79% | 14.57% | 0.7961 |
| rooms-and-doorways | time-cost | 18101 | 100000 | 83.03% | 1.80% | 15.17% | 0.7963 |
| rooms-and-doorways | potential | 18101 | 100000 | 90.42% | 5.79% | 3.79% | 0.8521 |
| rooms-and-doorways | original | 18203 | 100000 | 88.02% | 1.60% | 10.38% | 0.8574 |
| rooms-and-doorways | time-cost | 18203 | 90000 | 79.44% | 2.79% | 17.76% | 0.7718 |
| rooms-and-doorways | potential | 18203 | 70000 | 85.63% | 7.78% | 6.59% | 0.8201 |
| rooms-and-doorways | original | 18301 | 100000 | 84.83% | 2.40% | 12.77% | 0.8160 |
| rooms-and-doorways | time-cost | 18301 | 100000 | 84.43% | 1.40% | 14.17% | 0.8072 |
| rooms-and-doorways | potential | 18301 | 50000 | 77.25% | 8.78% | 13.97% | 0.7388 |
| deep-cul-de-sacs | original | 18101 | 90000 | 65.67% | 2.72% | 31.61% | 0.6567 |
| deep-cul-de-sacs | time-cost | 18101 | 90000 | 70.57% | 3.54% | 25.89% | 0.7057 |
| deep-cul-de-sacs | potential | 18101 | 90000 | 70.57% | 5.18% | 24.25% | 0.7057 |
| deep-cul-de-sacs | original | 18203 | 90000 | 67.85% | 2.45% | 29.70% | 0.6785 |
| deep-cul-de-sacs | time-cost | 18203 | 100000 | 76.84% | 2.72% | 20.44% | 0.7684 |
| deep-cul-de-sacs | potential | 18203 | 70000 | 55.59% | 7.36% | 37.06% | 0.5559 |
| deep-cul-de-sacs | original | 18301 | 90000 | 71.12% | 1.91% | 26.98% | 0.7112 |
| deep-cul-de-sacs | time-cost | 18301 | 100000 | 74.66% | 2.18% | 23.16% | 0.7466 |
| deep-cul-de-sacs | potential | 18301 | 100000 | 78.75% | 3.27% | 17.98% | 0.7875 |

## Dead-end and detour route subsets

These descriptive rows pool actual route cases across maps and policy seeds; they are not independent seed estimates or causal difficulty effects. Categories overlap.

| Reward | Route category | Cases | Success | Collision | Timeout | SPL |
|---|---|---:|---:|---:|---:|---:|
| original | starts_in_dead_end | 441 | 38.32% | 13.83% | 47.85% | 0.3789 |
| original | goal_in_dead_end | 360 | 76.94% | 0.56% | 22.50% | 0.7639 |
| original | requires_detour | 2871 | 68.62% | 0.73% | 30.65% | 0.6814 |
| time-cost | starts_in_dead_end | 441 | 37.87% | 12.93% | 49.21% | 0.3744 |
| time-cost | goal_in_dead_end | 360 | 76.11% | 0.83% | 23.06% | 0.7573 |
| time-cost | requires_detour | 2871 | 71.09% | 0.87% | 28.04% | 0.7054 |
| potential | starts_in_dead_end | 441 | 35.37% | 38.32% | 26.30% | 0.3466 |
| potential | goal_in_dead_end | 360 | 72.22% | 7.22% | 20.56% | 0.7076 |
| potential | requires_detour | 2871 | 69.00% | 6.37% | 24.63% | 0.6730 |

Execution took 118.16 seconds against the 1,200-second limit. All runs completed with no interrupted attempts. This is one Q-learning development experiment on four known structured layouts. It does not establish a reward-only explanation for old unseen-map failures, nor fix generalization of the earlier SARSA/DQN/PPO models.

Open **Demo > Navigation map**. Choose a map, reward condition, training seed and route; the route is retained when switching reward designs. Play/step/seek uses saved actions. The UI shows each last-step base reward and potential component separately.

Artifacts: [summary](../artifacts/studies/map-reward-study-v1/summary.json), [audit](../artifacts/studies/map-reward-study-v1/verification.json), [frozen inputs](../artifacts/studies/map-reward-study-v1/freeze.json). All selected models, monitoring records and full validation trajectories are retained.

## Software checks and saved demonstration

The full suite passed **229 tests in 97.30 seconds**. After adding explicitly illustrative success/failure shortcuts, the final targeted map/environment/UI suite passed **22 tests in 7.63 seconds**. The original 184-file and unseen 114-file release seals and M13 source/output hashes still verify. [Verification record](evidence/navigation-lab-verification-v1.json).

Initial successful examples are explicitly illustrative and do not determine the benchmark means. Timeout and collision shortcuts remain visible, and changing rewards keeps the same selected route. The current rehearsal inventory is [demo-v3](evidence/demo-v3.json); the old demo-v1 inventory remains preserved as historical remediation evidence.
