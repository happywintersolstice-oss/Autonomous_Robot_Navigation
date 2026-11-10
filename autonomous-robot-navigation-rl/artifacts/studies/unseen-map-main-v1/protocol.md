# Milestone 11. Shared features and multi-seed unseen-map development

Registered 10 November 2026 before feature-pilot or main training. The owner
selected shared features for all four algorithms, including feature-based value
functions for Q-learning and SARSA. The experiment uses the existing immutable
unseen-maps-v1 development split and its original dynamics and rewards.

## Representation and value functions

Every agent receives the same deterministic 1,234 features derived from the full
104-value raw observation. The vector contains a bias, two scaled goal offsets,
four map-derived cardinal clearance distances, a 19×19 relative-goal one-hot
vector, a nine-goal-sector/four-neighbor-obstacle-pattern one-hot vector, and two
19×19 robot-centered channels for occupied cells and map boundaries. The occupied
channel is divided by sqrt(21), and the boundary channel by sqrt(261). Each of
these blocks has unit squared norm on the registered maps. The full map and
goal remain represented. Clearance is calculated from the supplied map.

Q-learning and SARSA use one shared linear action-value function per action.
The update is w[a] += alpha * TD_error * x / (x·x). Q-learning bootstraps from
the maximum next action value. SARSA samples and carries its next behavior
action before the update. Terminations have zero bootstrap, and timeouts
bootstrap from their final state. Evaluation uses argmax with Up first on ties.
DQN and PPO use the identical features through a fixed SB3 extractor, followed
by their registered 64×64 networks. The linear and neural model capacities
differ. Model size and decision costs will be reported.

All four actions remain selectable. No shortest-path/planner outputs enter the
features. BFS remains the optimal known-map reference. Recorded replays store
raw coordinate transitions and the map identity, plus the feature version.

## Registered matrix and limits

The configuration is configs/unseen_main.yaml. Four engineering pilots use seed
12007 and 25,000 steps each. The main matrix uses fresh seeds 12101, 12203 and
12301, with exactly 200,000 steps per algorithm/seed. Planned training totals
2.5 million transitions across 16 runs. Pilots check implementation and runtime.
They retain their outcomes and do not choose hyperparameters or drop a family.

All runs monitor the same 24 validation cases every 12,500 steps, after the
preceding update. Selection maximizes success, then SPL, then prefers the earlier
checkpoint. Complete validation has 192 routes on four unseen layouts. A separate
72-case familiar-layout probe measures fit to training routes. Evaluation
fingerprints learned state before and after inference.

There are at most two concurrent CPU workers with one Torch thread each. The
limits are 900 seconds per run, 9,000 seconds per suite, 1 GiB of study files,
20 attempts total and 3.3 million training transitions including technical
failures/retries. Each logical run gets at most two attempts. Poor complete
runs remain in the matrix. Incomplete attempts are retained with observed step
counts or explicit lower bounds. Checkpoints are inference snapshots. Successful
complete runs can be reused, and exact training resume is unsupported.

Workers execute the archived source using the frozen configuration. The
coordinator owns progress writes. Per-run folders, atomic metadata writes and
an exclusive study lock control concurrent execution. Existing fixed-map and
milestone 10 evidence are inputs to preservation checks.

## Analysis and acceptance

Hard gates require NumPy/Torch feature parity, checked TD updates, exact budgets,
declared split membership, reloadable models, valid actual training traces,
selected-checkpoint action/diagnostic parity and unchanged inference state.
The original BFS/Random validation reference records are reused with their
hashes, scenario order and actions verified. Final-layout policy access remains
reserved for a later frozen independent evaluation.

Report per-seed and per-map outcomes, equal-weight training-seed means and
sample SD, and counts with their actual support. Conditional successful route
lengths retain zero-success counts and exclude undefined seed means. Routes
within a layout/policy are dependent observations. These three training seeds
provide limited uncertainty information, and the four validation layouts are
development data. Targets remain 80% success, at most 2% collision and SPL .75.
Missing targets is an outcome to report.

Milestone 10 and milestone 11 differ in representation, linear value-function
implementation, training budget and seeds. Cross-milestone changes are
descriptive. This experiment compares four methods under the shared new design
and does not isolate a causal representation effect.

The neural implementation follows the official
[SB3 custom feature-extractor interface](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).
The linear updates and their normalization are specified explicitly above and
covered by hand-calculated regression checks.
