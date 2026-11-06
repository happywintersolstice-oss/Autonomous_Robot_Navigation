# Milestone 4 — Part B refinement protocol

Written before the search. Milestones 1–3 are complete; Part A shortlisted
Q-learning and SARSA. Keep its original map, scenario files, rewards and artifacts
unchanged. The final test remains sealed. No DQN retuning, PPO, new maps, moving
obstacles, continuous control, hardware, planner fallback, masking or full dashboard.

## Frozen run matrix

All training runs use 200,000 interactions, existing training pairs, the fixed
30-pair validation monitor every 10k, and complete validation of the selected
checkpoint. Select checkpoints by monitor success, SPL, then earliest interaction.
Random search samples eight distinct settings uniformly without replacement from
the finite grid in `configs/part_b.yaml`, using RNG seed 20261106. Use the same
eight settings for both algorithms, each on development seeds 7 and 19: 32 runs.
Save the actual candidate list and configuration/source hashes before training.

Choose one setting per algorithm by mean complete-validation success over both
development seeds, then mean SPL, lower success SD and candidate ID. Do not select
individual favorable seeds. Confirm both selected settings on fresh training seeds
101, 211, 307, 401, 503. Run matched original-Part-A settings on those same five
seeds as controls: 20 more runs. Report paired per-seed changes, means and sample
SD, including every poor run. The original Part A evidence remains a separate
historical reference, not a substitute for these matched controls.

Predeclared provisional targets for each tuned family: validation success >=80%,
collision <=2%, SPL >=.75 at this budget. Misses remain results; do not redefine
targets after seeing outcomes. Preserve the previous practical playback target of
2 actions/second; this milestone tests artifact integration, not a new quantitative
browser responsiveness benchmark.

## Controlled reward experiment

Freeze each chosen hyperparameter setting, then repeat its five confirmation
seeds with only ordinary step reward changed from -.01 to 0. Goal +10, collision
-5, all transitions/actions/observations, task limits, reset pairs and budget stay
the same. Use explicit reward version `terminal-only-v1` and a derived task hash.
Reuse the original coordinate pairs through a validated parent-task relationship;
never rewrite or relabel the canonical benchmark manifests. These 10 runs are a
paired ablation, not an additional hyperparameter search. Compare success,
collisions, timeouts and SPL; returns under the two reward versions are not a
cross-version quality ranking. Do not automatically promote the reward variant.

## Voting and system selection

Choose the tuned algorithm family with the better five-seed mean validation
success on the original step-cost task, then SPL, lower success SD, training cost
and name. This choice precedes voting evaluation. The selected single policy is
its seed-101 checkpoint, not the best seed. The voting members are exactly its
seeds 101, 211 and 307 in that fixed priority order. Each proposes an action;
majority wins, and a three-way tie uses the first member's action. No Q-value
averaging, masking, fallback or voting-specific training. Evaluate the vote and
every component on the same 928 validation pairs; retain proposals/ties in replays.
Report vote-versus-each-member and versus the predeclared single policy, parameter
memory, checkpoint storage, summed member training cost and inference latency.
Do not require voting to improve; a negative finding completes the experiment.
System choice after this comparison remains development-only; final freeze and
test evaluation belong to milestone 6. No final-test-based changes are allowed.

## Limits and evidence

Exact training matrix: 32 search + 20 paired control/confirmation + 10 reward
ablation = 62 runs, 12.4m training interactions. No automatic follow-on trials.
Cap the run phase at 45 wall minutes and 750 MiB of new run artifacts; if reached,
stop before the next run, preserve partial results and report reduced coverage.
Estimate training, monitoring, final validation and storage separately from local
Part A measurements before executing. All checkpoints are inference-only; retain
failed/interrupted status and do not claim exact training resume.

Audit saved trajectories/scores, rule-based checkpoint/configuration choices,
training counts, signatures and source/model hashes. For new runs, retain selected
checkpoint visitation separately from full-budget visitation so overlap is labeled
at the correct training prefix. Produce inspectable summaries, paired differences,
search results, learning curves, reward-ablation plots and voting diagnostics.
The minimal viewer must load matching reward variants and frozen voting artifacts
with clear labels, without training or exposing final-test scenarios.
