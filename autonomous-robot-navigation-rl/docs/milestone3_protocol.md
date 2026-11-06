# Milestone 3 — Part A protocol

Written before the main comparison. Keep the fixed map, task signature, rewards,
200-action cap and saved splits unchanged. No test policy execution or UI test
browser. No PPO, unseen maps, tuning, action masking, dense planner supervision,
ensembles or full dashboard. The purpose is an honest three-algorithm comparison
and two-approach shortlist, including failed runs.

First run a 10,000-interaction DQN smoke pilot (seed 7) and measure actual training,
monitoring, full-validation and artifact costs. Verify Gymnasium and SB3 checkers,
normalization parity, terminal/truncation replay behavior, inference immutability,
save/load and fixture learning. Use installed CPU PyTorch and one intra-op thread.
The pilot checks correctness/throughput; do not search settings against its score.
Freeze the main run matrix only after reporting pilot measurements. A failed
learning score is retained, without changing task rules.

Planned main matrix: Q-learning, SARSA and vanilla Stable-Baselines3 DQN; five
training seeds 7, 19, 43, 67, 91; 200,000 training interactions each. Use the
unchanged 30-pair validation monitoring manifest every 10,000 interactions in a
separate raw environment. Monitoring actions are extra and logged. Greedy actions
break ties by first maximum in up/down/left/right order. No validation updates,
replay-buffer insertion or fitted normalization. Preserve all seeds and checkpoints.

Tabular alpha .3, gamma .99, epsilon 1 → .1 over 80% of the budget. DQN uses a
4-input normalized coordinate vector (robot row/column, goal row/column, scaled
by dimension minus one), the same information as the table, and an MLP [64,64].
Fixed DQN settings: Adam learning rate .0003, gamma .99, replay capacity 50,000,
warm-up 1,000, batch 64, train frequency 4, one gradient step per rollout,
hard target synchronization every 1,000 actions, epsilon 1 → .1 over 80%, CPU.
This is vanilla DQN, not Double/Dueling/prioritized replay. The warm-up's uniformly
random actions and SB3's greedy tie breaking differ from tabular random training
ties and are disclosed; task information and interaction budget remain equal.

Checkpoint selection: maximize monitoring success, then SPL, then prefer the
earliest checkpoint. Evaluate each seed's selected checkpoint on the complete
validation manifest exactly once; retain final-budget checkpoint separately and
do not call selected-checkpoint training interactions the total run budget. This
is development validation selection, not independent final performance. Compare
all three on the same 928 validation pairs, retain full compressed trajectories,
and label subset versus full-manifest results. At the 200k boundary, DQN monitoring
uses weights after the final scheduled update; earlier DQN callback checkpoints
are pre-gradient at that boundary, recorded explicitly.

Shortlist algorithm families by mean full-validation success across five seeds,
then mean SPL, then lower seed success SD, then lower total training wall time,
then algorithm name for exact ties. No seed cherry-picking or test use. Report
every seed and sample SD over training seeds, not episode-level confidence.

Produce success/SPL learning curves with seed variability, common-scenario outcome
tables, failures by distance, collision/timeout heatmaps, successful route-length
distributions, and verified loop/detour trajectory examples. Q-value displays are
decision evidence, not causal explanations. BFS knows the map; learned policies
receive coordinates only. BFS/random references are evaluated through the same
environment on validation and remain outside the learned-family shortlist.
Common trajectory panels use seed 7 and the first stable validation scenario ID;
failure examples are the first occurrence per taxonomy in seed/scenario order.

Pilot gate outcome: the real 10k pilot discovered 36/79 training goals and reached
20/928 validation goals (358 collisions, 550 timeouts). The fixture learns, both
checkers and timeout/normalization/save-load tests pass, and parameters/replay/RNG
stay unchanged in evaluation. Sparse discoveries and repeated timeouts remain
learning limitations, not evidence of an encoding/target defect. Retain the planned
200k budget and unchanged baseline settings; no task or score-driven adjustment.

Checkpoint archives are local trusted artifacts with checksums/task/encoding
metadata. They are inference-only; optimizer serialization does not imply exact
resume because replay, RNG and environment state are not bundled. Interrupted
runs retain status and latest checkpoint; they are not complete-budget results.
