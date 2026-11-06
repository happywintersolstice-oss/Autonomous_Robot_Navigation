# Environment v1

One fixed 10 × 10 map, `bottleneck-v1`, static obstacles, deterministic point robot.
Coordinates are zero-based `(row, column)`. Map JSON rows use `.` for free and `#`
for blocked. The open north, offset vertical walls, central wall and eastern
passages were chosen before training to provide open travel, detours and bottlenecks.
`benchmark_report.json` records actual connectivity, distance histograms and counts.

Observation: Gymnasium `MultiDiscrete([10,10,10,10])`, robot row/column then goal
row/column. The Q-table indexes these four coordinates directly, with four action
values per state. Up=0, down=1, left=2, right=3. No wait/diagonal action, masking,
planner fallback or early stopping on repeated positions. The map is shown to
humans; the learned agent sees coordinates only. BFS explicitly knows the map.

Each attempted action costs -0.01. Goal adds +10; obstacle/boundary collision adds
-5, keeps the actual position unchanged, and ends the episode. These events are
mutually exclusive. The 200th action truncates only when it has no terminal event.
Timeout has no extra penalty. Goal/collision set `terminated`; timeout alone sets
`truncated`. The tested built-in external action counter gives terminal events
priority over timeout. `reset(seed=...)` seeds Gymnasium's reset-scenario sampling.

Q-learning uses `r + gamma*max(Q(next))` except for `terminated=True`, when the
target is just `r`. Truncation bootstraps from the actual final observation before
reset. Greedy evaluation breaks ties in up/down/left/right order; training breaks
max-value ties randomly and uses a linear epsilon schedule. Random draws from all
four actions uniformly. BFS follows descending distance with the same fixed tie
order. All policies act through the same environment/rollout interface.

Milestone 2 adds SARSA with the same coordinate table and exploration schedule.
Its nonterminal target is `r + gamma*Q(next_observation,next_behavior_action)`;
sample the behavior action before updating and execute that same action next.
On truncation, bootstrap using the old episode's final observation/action, then
reset and draw a fresh behavior action. Goal/collision use `r` only. A training
budget cutoff is logged separately and does not change environment outcomes.

Milestone 3's DQN wrapper scales the same four coordinates by dimension minus
one into a float32 Box(0,1,shape=(4,)); it changes no actions, rewards, flags or
scenario sampling. There is no fitted normalization or map channel. The inference
adapter accepts the raw coordinates and applies that same fixed scaling.
Checkpoints require `normalized-coordinates-v1` plus the original task signature.
SB3's vector wrapper stores the terminal observation on autoreset; tests verify
that replay samples mask timeout dones to preserve bootstrapping, while true
terminal events keep done=1. The saved network is vanilla SB3 DQN.

Scenario preparation enumerates all reachable, distinct ordered pairs; a fixed
Python RNG shuffles each goal/distance stratum, allocating approximately 70/15/15.
Holdout sizes round to nearest integer within each stratum; strata under four pairs
remain in training. Exact counts/fractions are saved, rather than asserted to be
exactly 70/15/15. Bins: short 1–5, medium 6–10, long >=11. Every held-out goal has
many training starts. Monitoring is a fixed distance-balanced validation subset
of 30, at most 50. Splits are initial conditions on a familiar map, not unseen
states or unseen layouts. Training may visit held-out initial states; the pilot
records that overlap. Preparation and tests inspect holdout structure; no policy
executes the final test, and training/evaluation/UI loaders reject that split.

The benchmark remains provisional for main-study freezing until the learning
pilot is reviewed. No rewards/layouts are altered based on this pilot. Map, task
configuration and observation/action/reward versions form an artifact signature.
Replay loading re-executes each saved action to verify states/rewards/flags/metrics.
Checkpoint loading also verifies SHA-256 and Q-table shape/finite values.

Metric definitions: success, collision and timeout counts partition completed
episodes; SPL averages success × shortest/max(shortest,actions), so failures
contribute zero. Successful route length is conditional on success, with its count;
return sums every action reward. Prefix playback shows actions/reward so far and
does not call a partial path's efficiency a completed SPL.

Method references consulted during implementation:
[Gymnasium custom environments](https://gymnasium.farama.org/introduction/create_custom_env/),
[Farama termination/truncation](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).
