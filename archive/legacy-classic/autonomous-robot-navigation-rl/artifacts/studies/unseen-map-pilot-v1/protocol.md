# Milestone 10. Unseen-map foundation and four-algorithm pilot

Registered 10 November 2026 before benchmark generation and training. The owner
authorized map-aware Q-learning, SARSA, DQN and PPO. This milestone establishes
the new benchmark and measures a bounded pilot. A subsequent milestone will
freeze the main multi-seed budget and conduct independent final evaluation.

## Question and shared information

Can controllers trained across layouts navigate new layouts? Every learned
family receives the same 104 values, comprising robot/goal row and column
scaled by 9, followed by the full row-major 10×10 binary obstacle map. This is a
fully observed coordinate task. Actions, collision termination, the 200-action
cap and rewards remain the core task's existing dynamics. Policies select from
all four actions. Evaluation uses deterministic argmax, with Up first on ties.

The tabular implementation reuses the existing Q-learning/SARSA updates with one
table per observed occupancy layout. On an unseen layout all entries have their
initial zero value and inference leaves them unchanged. This deliberate tabular
baseline exposes the transfer limitation of exact-state lookup. DQN and PPO use
shared functions across layouts. No planner-derived features or action masks
are supplied to these four policies. BFS is the known-map planning reference.

## Separate benchmark

The registered configuration is `configs/unseen_maps.yaml`. It declares 12
training, 4 validation and 10 sealed final layouts. Each 10×10 layout has 21
blocked cells, connected free space and shortest-path diameter at least 14.
Generation rejects duplicates under all rotations and reflections, including
the original fixed benchmark. A seeded rejection sequence determines all
layouts before training. Each layout has 16 short, 16 medium and 16 long routes,
using the core distance bands. Layout identity is included in scenario IDs.

Training samples uniformly from the declared training routes. The fixed monitor
uses two routes per distance band per validation layout, giving 24 cases.
Complete pilot validation uses 192 cases. Final has 480 cases on 10 different
layouts. The preparation audit checks geometry, hashes and split integrity.
Training, ordinary loaders, replays and the dashboard accept train/validation
only. Final scenario records are locally stored behind this software access
boundary. Final outcomes require a later dedicated frozen runner.

A post-training familiar-layout diagnostic uses the same two-routes-per-band
rule on the 12 training layouts, giving 72 cases. It reports training-task fit
separately from the 192 new-layout validation cases.

## Pilot execution and acceptance

Each of the four algorithms gets exactly 25,000 training transitions with seed
11003. DQN and PPO use 64×64 MLPs on CPU with one Torch thread. All parameters
are registered in the YAML. Two monitoring opportunities occur at 12,500 and
25,000 transitions, after the preceding update and before further collection.
Selection maximizes monitor success, then SPL, then prefers the earlier step.
Validation follows selection and cannot update policies. Failed/incomplete
attempts remain recorded. Checkpoints support inference, with exact training
resume outside this milestone.

Limits are 600 wall seconds per run, 1,800 per suite and 512 MiB of new study
artifacts. Record construction, training, monitoring, validation, audit and total
costs separately. Capture an actual training episode after the 0%, 50% and 95%
windows, including a budget cutoff when applicable. Save complete validation
trajectories and replay examples with map/scenario/checkpoint provenance.

Hard gates require split integrity, optimal BFS on all validation routes, exact
budgets, valid saved transitions, matching reloaded actions, outcome
reconciliation and unchanged inference fingerprints. Poor learning remains a
reported result. The core study's aspirational main-study targets remain 80%
success, at most 2% collisions and SPL at least .75. Pilot results do not provide
a multi-seed algorithm ranking or evidence from the final holdout.

Report equal-weight map means, per-map support, success/collision/timeout/SPL,
return and conditional successful route lengths, with costs. All pilot routes
within an algorithm share one trained policy. Routes are dependent observations
within layouts, and layouts share that policy. Main-study uncertainty must
account for both training seeds and layouts.

## Implementation and sources

Reuse the core environment, BFS, metric functions, tabular updates, checkpoint
ranking, inference fingerprints and recorded-replay validation. The fixed-map
benchmark, prior checkpoints, research artifacts and UI entry points remain
separate from this versioned study. The new dashboard page uses saved validation
and training episodes through the centered replay presentation.

SB3's callback lifecycle and supported spaces were checked against the official
[DQN documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html),
[PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
and [callback documentation](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html).
The installed package implementation determines execution behavior.
