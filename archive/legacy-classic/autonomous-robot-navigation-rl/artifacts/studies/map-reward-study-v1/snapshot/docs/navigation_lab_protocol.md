# Structured maps and reward comparison: registered protocol

17 November 2026. Benchmark `structured-navigation-v2`; study `map-reward-study-v1`. These are new, familiar-layout development experiments. The old bottleneck map, unseen-layout benchmark, checkpoints and frozen results retain their original identities.

## Map contract

Generate four deterministic 10 x 10 layouts: a dead-end maze, looped corridors, rooms with doorways, and deep cul-de-sacs. Geometry selection precedes learning. Each layout must be connected, have at least three degree-one dead ends, three junctions, eight degree-two corridor cells, a dead-end branch at least three moves deep, and graph diameter at least 20. The deep-cul-de-sac family requires depth at least eight. Count topology from the free-cell adjacency graph, rather than naming a map difficult from appearance alone. Rotation/reflection-equivalent maps within the set are rejected.

Per-map ordered start/goal pairs are split deterministically by goal/distance stratum into training, validation and reserved pairs. Monitoring uses up to 18 fixed validation pairs spread across distance bands. Every route is reachable. Record whether a route needs a detour and starts/ends at a dead end. The reserved pairs are prepared but never scored in this development study. Held-out initial pairs on a familiar map are not unseen layouts or wholly unseen state observations.

## Reward hypothesis and controlled conditions

The original task charges -.01/action, adds +10 at the goal or -5 on collision. An undiscounted 200-step timeout costs -2 versus -5.01 for one collision. Under gamma .99, the discounted continuation cost of an endless -.01/action loop approaches -1. A poorly estimated policy can prefer a safe loop to a collision risk. Globally optimal navigation on these reachable maps still favors successful routes; repeated states alone do not prove a reward-only cause. Coverage, sparse credit assignment and approximation are alternative contributors.

Train tabular Q-learning separately for every map, reward condition and training seed. This deliberately removes the earlier cross-layout linear approximation from the reward comparison; it does not establish that the same reward fixes SARSA, DQN or PPO generalization.

| Condition | Base move cost | Goal/collision additions | Potential scale |
|---|---:|---|---:|
| original | -.01 | +10 / -5 | 0 |
| time-cost | -.05 | +10 / -5 | 0 |
| potential | -.05 | +10 / -5 | .2 |

Potential is negative Manhattan distance, computed only from the observed robot/goal coordinates. Added reward is `scale * (gamma * Phi(next) - Phi(current))`, using the same gamma .99 as learning. Potential is zero for true absorbing goal/collision termination. An external time-limit truncation retains next-state potential and bootstrap. This follows the potential-based shaping construction in [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). The ideal policy-invariance result concerns the corresponding base reward, not an assurance about finite training, monitoring selection, this whole three-condition comparison or learned accuracy.

Do not award only positive progress rewards, substitute BFS distances into training rewards, mask unsafe actions, add a hidden planner, penalize all revisits, or terminate on cycles. Dead-end escape can require backtracking or moving away from the goal. Stronger time costs can also encourage premature collisions in poorly learned policies, so collision is a guardrail and all failures must be reported.

## Budget, selection and analysis

Four maps x three conditions x three predeclared seeds (18101, 18203, 18301), **100,000 actions each**, totaling **3.6 million training actions**. Learning rate .9, gamma .99, epsilon .7 to .03 over the first 30% of each budget. Monitor every 10,000 actions; select highest monitor success, then SPL, then earliest step. Complete the full budget regardless of the selected age. The experiment is a bounded development study, not another 200k main study.

Compare conditions within the same map and training seed, then average maps equally within each seed before averaging seeds. Matching seeds does not imply identical experience after policies diverge. Report success, collision, timeout, SPL and movement revisits/looping timeouts, with all per-map/per-seed rows. Do not compare raw returns across reward scales as a quality ranking. Include conditional dead-end/detour subsets and their supports. Keep the original controller choice unchanged; demonstration choices are explicitly illustrative.

Limits: 1,200 seconds and 512 MiB. Freeze configuration, protocol, layout/scenario and source bytes before scoring. Retain partial attempts without publishing a complete headline. Save all selected checkpoints, monitoring records and validation trajectories. Independently reload models, reexecute saved transitions/rewards and reconcile metrics and selection. Save a geometry gallery before training.

Reproduction from the application directory:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.navigation_lab prepare
.\.venv\Scripts\python.exe -m robot_navigation.navigation_lab run
.\.venv\Scripts\python.exe -m robot_navigation.navigation_lab verify
```

Completed studies are verified and reused, not overwritten. Any changed configuration or map needs a new version. The UI replays recorded actions and exposes base/potential reward components. It does not retrain or swap old models onto changed maps.
