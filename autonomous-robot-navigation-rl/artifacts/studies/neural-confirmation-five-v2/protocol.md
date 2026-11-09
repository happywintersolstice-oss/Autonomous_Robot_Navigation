# Five-seed staged neural confirmation

The owner authorized both five-seed expansion and native dashboard verification
on 9 November 2026. The original three-seed M9 results were already available.
This extension is declared afterward and remains exploratory development
validation. Preserve all original study files and the frozen final release.

## Fixed matrix

Retain original confirmation seeds 7307, 7411 and 7523. Add seeds 7607 and 7717
for DQN/PPO, each with a baseline and its frozen c01 settings. Run exactly eight
new 200,000-step jobs, totaling 1.6 million added training steps. Do not repeat
screening or change settings from any new outcome. Both new seeds were checked
against existing run manifests before execution. Retain failed or poor seeds.

Copy each condition's original training configuration, changing only its seed
and unique run ID. Verify the existing learning/evaluation engine against the
original source snapshots. Keep four absolute coordinates, [64,64] networks,
the original map, rewards, split, fixed 30-route monitor every 10k, 200-step task
limit and full 928-route validation. Use the same checkpoint selection rule.
Record ten actual training episodes across the five existing stage boundaries.

## Reporting and limits

Report every paired seed, five-seed means and sample SD, paired success/SPL
changes, all outcome rates, returns, conditional successful route lengths and
recorded costs. Show the two added seeds separately from the original three.
Five-seed averages reuse the original runs. They are a staged expansion, with
no fresh holdout, episode-level significance claim or new final-test execution.
The selected release controller remains Q-learning.

Use at most two separate one-thread learners. Cap added work at eight runs,
1.6 million training steps, 900 seconds per run, 2,400 suite seconds including
audits, and 1 GiB of added artifacts. Estimate runtime before starting from
the prior matched confirmation costs, adding 35% training allowance, twice
monitoring/validation cost and 180 audit seconds. Stop if the estimate exceeds
the cap. Exact resume and overwriting existing results are unsupported.

Freeze settings, all job configurations, seed matrix, runner/protocol hashes,
engine hashes and current recording-source hash before training. Hash all old
artifact/data files, verify preservation after training and publication, and
deep-audit all 20 combined confirmation runs against each run's frozen source.
Reconcile ledgers, visitation, monitoring selection, checkpoint checksums,
validation metrics, selected-model diagnostics and actual training records.

Publish a separate five-seed bundle only after the audit passes. Keep access to
the original three-seed dashboard evidence. Verify the resulting research page
and matched playback visually in the native browser, including a narrower
layout. Record observed geometry, interactions, results and remaining issues.
Unseen maps, dynamics noise, continuous control, sensors and hardware remain
deferred.

## Recovery declared after a file-write failure

The first extension attempt completed all four added DQN runs. PPO baseline
seed 7607 then failed while overwriting manifest.json at 50,000 steps with
Windows OSError 22. The paired selected-settings worker was stopped after its
40,000-step checkpoint. Both incomplete attempts and all four completed DQN
results are retained in neural-confirmation-five-v1. Exact killed-worker work
beyond its last persisted checkpoint is unavailable.

The writer previously truncated a destination file in place. The repair writes
a complete same-directory temporary file, then atomically replaces its target
with bounded retries for transient Windows sharing/mapping errors. Serialized
JSON bytes retain the existing format. The exact Windows locking source is
unconfirmed.

The recovery bundle is neural-confirmation-five-v2. Reuse the four completed
DQN runs verbatim. Restart only the unfinished PPO work from step zero, using
the same selected settings and seeds 7607/7717, with an attempt2 suffix on each
new PPO run ID. Execute four new 200k jobs, totaling 800,000 recovery training
steps, with a new 1,800-second suite cap and the same 900-second run/1 GiB caps.
The interrupted attempts add at least 90,000 training steps beyond the 1.6m
successful addition budget. Report this failure cost separately.

The only learning-engine source difference allowed is JSON input/output handling and
its supporting imports. Readers also retry transient Windows access errors
during replacement. Compare all other core syntax and all other engine
file hashes against the original snapshots before recovery. Keep the original
and added-seed results separate. Do not retry based on model performance.
The observed narrow-screen selector layout is also corrected before these
recovery jobs and verified after publication.
