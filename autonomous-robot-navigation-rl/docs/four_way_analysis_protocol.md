# Four-family saved-evidence integration protocol

Post-release extension authorized by the owner on 8 November 2026. Reuse the
verified `four-rl-v1` five-seed baseline matrix. No training, policy execution,
checkpoint selection, tuning or final-test access is part of this analysis.

Write a new `four-rl-analysis-v1` study, refusing overwrite. Preserve every
previous run, study, task file and the original frozen release. Reconcile each
run's saved episode metrics against both its validation summary and the verified
four-way summary. Check selected checkpoint metadata, checksum and size.

Apply the six existing milestone-5 geometry dimensions unchanged: distance,
canonical-reference corridor, start-wall proximity, start quadrant, goal quadrant
and quadrant crossing. Verify regenerated labels match the saved historical labels
and persist them before reading run episode outcomes. Each dimension partitions
the same 928 validation scenarios; dimensions overlap and are not independent tests.

Report success, collision, timeout, SPL, return and conditional successful route
length by family, group and seed. Seed means are unweighted; sample SD uses ddof=1.
Exclude undefined route-length means for zero-success seeds, retain their outcomes,
and show the contributing seed count and successful episode count. Never replace
undefined route length with zero. Reconcile group support, outcomes and actions.

Report recorded per-seed training time, validation time, decision latency and
selected checkpoint size, plus means and sample SD. Training time excludes
construction, monitoring, checkpoint I/O and validation/export. Evaluation time
includes reset/recording/export and excludes rendering. Decision latency is the
historical 100-query mean, not a new shared hardware-isolated benchmark. Checkpoint
bytes exclude metadata and are not an estimate of model RAM. Cost comparisons are
descriptive: runs were executed separately, with equal interaction budgets rather
than equal update counts or compute. Publish seed details and CSV exports.

Retain all poor seeds and negative findings. The table compares original Part A
baseline families; tuned tabular results remain separate. PPO remains untuned and
excluded from the historical frozen final evaluation. Five seeds limit precision;
these groups do not establish causation or unseen-map generalization.
