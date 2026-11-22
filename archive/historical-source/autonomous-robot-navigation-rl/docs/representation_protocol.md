# Milestone 8: controlled DQN representation study

Owner chose the controlled DQN input comparison on 8 November 2026. All six
core milestones, the additional PPO baseline and four-family dashboard integration
are complete. This is a new exploratory post-release development-validation
study; no old model, study, canonical task, scenario split or frozen final result
will be overwritten. No final-test policy execution or new environment is allowed.

## Question and fixed treatments

Compare DQN with four normalized absolute coordinates `[r/H,c/W,gr/H,gc/W]`
against those same four features plus `[(gr-r)/H,(gc-c)/W]`, where H/W are
dimension minus one (minimum divisor one for fixtures). Relative features are
redundant deterministic transforms of the same observation, not extra information.
Retain absolute position to avoid obstacle-context aliasing. There is no map
channel, fitted normalization, action mask, planner advice, fallback or reward change.

The second input has six features and 128 extra online first-layer parameters
with the fixed [64,64] architecture; report that capacity difference and total
parameter/file sizes. This tests an input design, not information-equivalent
networks with identical parameter counts. All other DQN hyperparameters are
the original Part A values, fixed in `configs/representation_study.yaml` before
pilot/main training. This is not a DQN hyperparameter-tuning search.

## Matrix and selection

First run both encodings independently with pilot seed 1701 and 10,000 training
interactions. Pilots verify mechanics and estimate runtime/storage only; exclude
them from main statistics and make no settings change based on pilot performance.
Then train both treatments on each of five fresh paired seeds
`[101,211,307,401,503]`, exactly 200,000 interactions per run (10 runs / 2 million
main interactions). Run control then relative within each seed, in that fixed order.
Identical seeds do not imply identical trajectories because network dimensions
and subsequent behavior differ; analyze paired training-seed differences.

Use the original task, 200-action cap, training reset distribution, 30-pair fixed
monitor every 10k, separate read-only environments, and all 928 validation pairs.
Retain the existing DQN monitor timing: intermediate callback before boundary
transition storage/update, final checkpoint after the final scheduled update.
Select each run by monitor success, then SPL, then earliest interaction, and
complete the full budget. Report selected checkpoint age separately. Preserve all
20 periodic models, final/selected checkpoints, source snapshots, ledgers,
visitation, validation trajectories and poor seeds. No post-results reruns.

## Training traces

Capture the first three chronological training episodes in each new run, regardless
of outcome. Record actual sampled actions and raw before/after observations,
rewards, flags, task/scenario IDs, training seed and interaction bounds. Do not
query a policy again or invent frozen-policy scores for changing exploratory
behavior. Replay completed records through the shared environment. Mark any
budget/interruption prefix incomplete, without fabricating terminal metrics.
Recording is bounded, adds no environment interactions, and does not claim to
recover historical trajectories. Test that enabling capture leaves seeded learned
weights, visitation and episode outcomes unchanged on hand-solvable fixtures.

## Reporting, limits and integrity

Report all seeds, mean/sample SD for success, collisions, timeouts, SPL, return,
conditional successful route length and costs. Primary comparison: relative minus
control success percentage points; secondary SPL change. Show five paired deltas
and their mean/sample SD, not episode-level confidence intervals or significance.
Also report the six unchanged geometry dimensions descriptively, and monitoring
curves against interactions. Do not promote a new selected system using this study.

Cap each run at 300 wall seconds and the main matrix at 2,400 wall seconds and
1 GiB new artifacts. Check the measured pilot estimate before launching the main
matrix; if it exceeds the cap, stop and report it rather than changing budgets
after results. Preserve failed/interrupted runs; exact resume remains unsupported.
Record actual total cost, including pilots, monitoring and validation separately.

Hash all existing artifact/data files before training and verify them afterward;
archive the plan, protocol, runner, analyzer and source hash before main execution.
Final selection and original blinded evaluation remain immutable. All new outcomes
are exploratory validation on familiar-map initial conditions. Neural tuning,
unseen maps, dynamics noise and hardware remain outside this milestone.
