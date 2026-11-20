# Partial/dynamic benchmark: implementation validation protocol

Proposed study: **Comparative Evaluation of Reinforcement Learning Strategies
for Autonomous Navigation in Unknown and Dynamic Indoor Grid Environments**.

This protocol authorizes implementation tests and bounded functional smoke
runs only. It does not authorize a substantive comparative training study or
fresh final evaluation. Date: 20 November 2026.

## Development dataset

Configuration: `configs/partial_dynamic.yaml`; generator
`partial-connected-d4-development-v1`; generation seed 2026112001. Generate
24 connected training layouts and six development layouts, each with 21 walls,
diameter at least 14 and six route pairs (two each with base length 1–6, 7–12,
13+). Each pair has static and dynamic versions: 288 training scenarios and
72 development scenarios. The learned input excludes all preparation statistics.

Reject exact and rotated/reflected (D4) duplicates across new splits and against
the existing fixed/unseen/structured map JSON sources. The old final map file
is already inspected evidence; reading its geometry hashes only prevents reuse
and does not execute historical evaluation. Record source hashes and exclusions.
Report realized diameter/dead-end/junction and single-block detour statistics;
these defaults are not a scientifically optimized difficulty distribution.

Each dynamic route has three distinct blockage cells. One lies on its base
shortest path where possible. Starts are sampled from 0–12 plus six times the
event index; durations are uniformly 3–8 steps. Event seeds are independent of
policy/training actions. Some nominal events may not affect a particular route
trajectory, may occur after it ends or may be safely suppressed. Pair nominal
scenarios across policies and disclose realized differences.

No new final layouts are generated, read or evaluated here. Ten fresh final
layouts remain a proposal. Before substantive selection/tuning, the future
study must reserve and seal independent final geometry and event manifests
with a separately reviewed access policy. The development loader has no final
accessor and refuses any requested test/final split.

## Gates before training

Run the existing regression baseline first. Then validate sensor, memory,
encoding, observational-equivalence leakage, environment/time/events, splits,
same-information reference, metrics and replay. Validate the four adapter
update/checkpoint/autoreset paths using tiny unit fixtures. The `validate`
command writes a source/config/test-hashed gate record only after all new
tests pass. `smoke` refuses absent, failed or stale gates. Changes require
validation again, not a silent gate bypass.

## Bounded functional smoke matrix

All four methods train from scratch at seed **22001**, exactly **256 actions
each** (1,024 total), on the first two outcome-independently ordered training
scenarios (paired static/dynamic cases of one training layout). This is one
functional seed, not an estimate of comparative performance or uncertainty.

Shared input/reward/sensor/event rules are in the environment specification.
All methods use the same 412 permitted affine features. Q-learning/SARSA use
learning rate .1, discount .99, epsilon 1 to .1 over 80% of the budget. DQN uses
two 32-unit hidden layers, learning rate .0003, replay capacity 512, learning
starts 32, batch 32, one update per four steps and target update interval 64.
PPO uses 32-unit actor/value hidden layers, learning rate .0003, 32-step
rollouts, batch 32 and two epochs. The remaining algorithm defaults are the
installed SB3 2.9.2 defaults, retained in saved model parameters. One Torch
thread per model is used; training is sequential on CPU. Timings describe this
smoke run, not isolated hardware benchmarks. No tuning/checkpoint selection.

Require finite changed parameters, exact interaction counts, five legal action
outputs, correct terminated-versus-truncated targets, checkpoint reload equality,
unchanged learned state during inference, and complete saved-action audits.
Record actual training actions through an environment wrapper before vector
autoreset; do not reconstruct them with a final policy. Budget-cutoff prefixes
are not invented completed episodes.

Reload each model and record one fixed case in each of familiar-static,
familiar-dynamic, unseen-static and unseen-dynamic. “Familiar” here means the
actual smoke training layout; unseen means development geometry. All four
methods receive the same case IDs, event seeds and evaluation seed zero.
These 16 assessment episodes test the pipeline. Keep failures and raw metrics;
do not rank the methods or infer generalization from these samples.

## Recording and reproducibility

The validation bundle stores config, protocol, source and new development data
snapshots and raw-byte hashes, plus dependency versions. Each model has strict
task/action/observation/feature metadata and a checkpoint hash. Episode records
include layout/scenario hashes, nominal event seed, training seed, policy ID,
checkpoint hash, code hashes, observations, sensor/memory snapshots, applied and
suppressed event logs, actions/rewards and separate termination/truncation flags.
Replay audits reproduce all dynamics, observations, memory, events and metrics.
Metrics and recovery denominators are defined in the environment specification.

The old 104-value input, full-map 1,234-feature extractors, four-action matrices
and old checkpoints are incompatible and are never reused. Existing app and
asset inventories remain untouched. New raw-byte artifacts live under
`artifacts/partial-dynamic-navigation-v1/`; testing scratch lives in `.runtime/`.

## Future study and team responsibilities (not executed)

Member 1 owns Q-learning and observation validation; member 2 SARSA and
sensor/memory tests; member 3 DQN and dynamics; member 4 PPO and replay/metrics.
These are proposed responsibilities, not claims about individual authorship.
The online reference and interpretation are shared group work.

Phase 5 requires agreement on interactions/search budgets, at least three
independent training seeds per family, screening/confirmation separation,
monitoring/selection rules, event/horizon/representation sensitivity and an
independent final reserve. Capacity and compute differences must be reported.
Phase 6 requires explicit protocol/model/final-freeze authorization. UI work is
deferred to that phase: never route debug ground truth into agent inputs.
Waiting incentives, the cap, blockage durations, baseline heuristics and
representation learnability remain unresolved empirical choices.
