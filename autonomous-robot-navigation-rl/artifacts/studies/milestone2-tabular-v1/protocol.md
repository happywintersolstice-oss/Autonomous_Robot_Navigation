# Milestone 2 protocol — written before running the pilots

Implement SARSA and one shared tabular training loop, retain Q-learning and the
original smoke bundle, and demonstrate saved learned-agent replay. No DQN, tuning,
ensembles or environment extensions. Existing map, reward, observations, splits
and sealed final test remain unchanged. Main-study benchmark freeze stays pending.

Run Q-learning and SARSA with seeds 7, 19, 43, each for 200,000 training interactions.
Use alpha .3, gamma .99, linear epsilon 1 → .1 over 80% of the budget for both.
This extends the diagnostic budget from the earlier 30k run without changing the
task. It is not an isolated algorithm improvement or a completed Part A comparison.
No performance-based selection, reward adjustment, action masking or planner
fallback is allowed. Keep the final-budget checkpoints, including poor runs.

Monitor every 10k on the unchanged saved 30-scenario validation subset in a separate
environment. Evaluate all 928 validation scenarios once at the end with greedy
fixed tie-breaking and no learning. Preserve every validation trajectory, per-episode
metrics, goal discovery/visitation data, source/configuration/dependencies, and all
periodic inference checkpoints. Stream gzip records from the evaluation itself;
do not re-run episodes just to export them. Keep the final test sealed.

Repeat seed 7 for each algorithm with identical settings and verify equal tables,
episode logs, deterministic validation metrics and replay actions; timing fields
and run identity may differ. Repeats are reproducibility checks, not extra independent
seeds. Summaries average within each seed, then report three seed values and sample
standard deviation. No formal statistical superiority claim.

SARSA samples its next behavior action before updating, carries that same action
to the next step, and bootstraps from the actual final observation on truncation.
After a reset, draw a fresh action on the reset observation. Tests distinguish this
from max-Q updates, verify terminal/truncation handling, and exercise reset action
handling in the real loop. A budget cutoff does not invent an episode timeout.

Examine the 30k pilot's timeout loops alongside undiscovered goals and Q values
before main-study decisions. Earlier throughput suggests these eight runs are
small locally; cap this milestone at this matrix with no automatic search.
Checkpoints remain inference-only; record interrupted/failed status honestly.
Browser-visible animation remains an inherited open verification gate unless
direct verification succeeds in this milestone.
