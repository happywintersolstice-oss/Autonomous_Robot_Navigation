# Milestone 8: measured DQN representation results

Completed exploratory development-validation input comparison. All ten main runs
completed exactly 200,000 interactions each, across five fresh paired seeds.
The original four-algorithm comparison, selected controller and frozen final
evaluation remain unchanged. No hyperparameter search or final-test execution
was added. See [the registered protocol](representation_protocol.md).

| Inputs | Success mean ± seed SD | Collision mean | Timeout mean | SPL mean |
| --- | ---: | ---: | ---: | ---: |
| control | 5.97% ± 1.32 pp | 36.38% | 57.65% | 0.058648 |
| relative | 6.51% ± 2.30 pp | 24.40% | 69.09% | 0.063980 |

Relative minus control success: **+0.54 percentage points**, sample paired-seed SD 2.88 pp.
SPL change: **+0.005331**, sample paired-seed SD 0.027887.
SD is across five independent paired training seeds, not an episode confidence
interval or significance test. The two pilot runs are excluded.

The result is mixed: two seeds improved, two worsened and one tied. Most of the
mean gain comes from seed 101. Collisions declined while timeouts increased, and
both mean success rates remain below 7%. The tested input design did not produce
a consistent improvement; no new controller is promoted.

| Seed | Control goals / 928 | Relative goals / 928 | Success change (pp) | Selected control / relative interaction |
| --- | ---: | ---: | ---: | ---: |
| 101 | 46 | 97 | +5.50 | 20,000 / 160,000 |
| 211 | 73 | 56 | -1.83 | 40,000 / 20,000 |
| 307 | 60 | 50 | -1.08 | 40,000 / 50,000 |
| 401 | 56 | 57 | +0.11 | 60,000 / 50,000 |
| 503 | 42 | 42 | +0.00 | 30,000 / 40,000 |

Both treatments completed the full budget even when an early checkpoint was
selected. Monitoring uses the fixed 30-pair subset; reported endpoint outcomes
use all 928 validation pairs. All poor seeds are retained.

## Representation and costs

Control inputs are four normalized absolute coordinates. Relative inputs retain
those four and add two goal-minus-robot offsets. They contain the same information;
absolute position remains available for obstacle context. The six-input online
network has 4,868 parameters versus 4,740: 128 additional first-layer parameters.
The result cannot separate feature design from that associated capacity increase.

| Inputs | Mean training seconds / seed | Mean successful route actions | Mean decision microseconds | Mean selected checkpoint bytes |
| --- | ---: | ---: | ---: | ---: |
| control | 147.748 | 4.482 | 84.156 | 102100.0 |
| relative | 156.985 | 4.494 | 115.267 | 104237.0 |

Main matrix wall time: **1938.53 seconds**; predeclared cap 2,400 seconds.
Pilot estimate: 1831.51 seconds. New artifact size at matrix completion: 53.36 MiB (cap 1 GiB).
Main training costs 2,000,000 interactions; pilots add 20,000, for 2,020,000
new study training interactions. Unit-test fixture learning is separate.
Training times exclude monitoring/checkpoint I/O and validation/export.
Decision latency is a recorded 100-query mean; checkpoint bytes exclude metadata
and do not measure RAM. Preflight audits and UI checks overlapped part of the
matrix, so these are descriptive local timings, not isolated hardware rankings.
Route means are conditional on success, with undefined zero-success group means
omitted and contributing seeds/counts retained in the full summary.

## Fixed-map distance groups

| Inputs | Short success | Medium success | Long success |
| --- | ---: | ---: | ---: |
| control | 14.81% | 4.29% | 0.69% |
| relative | 15.81% | 4.29% | 1.44% |

All six unchanged geometry dimensions are retained per treatment and seed in
the full summary. Each dimension partitions the same validation pairs; dimensions
overlap. Geometry/distance confound group comparisons. This is familiar-map
held-out initial conditions, not unseen-map transfer or physical robot readiness.

## Training traces and verification

Each pilot and main run retains its first three chronological training episodes,
irrespective of outcome: 36 records across 12 runs. They contain actual exploratory
actions, raw states, rewards, flags and interaction bounds; they are not frozen-
policy trajectories or a reconstruction of historical training. They mainly
describe the initial random warm-up, not late learning. Incomplete-prefix support
is tested and never fabricates terminal metrics.
Trace capture integrity-replays completed records in a separate environment;
these auxiliary replay actions add no learner training or benchmark episodes.
The publication audit independently replays them again.

The audit verified **1,187,423 validation actions**, **200 periodic checkpoints**,
and **78 captured main training actions** against their chronological ledgers.
All **7,746** registered historical artifact/data files retained their hashes.
The audit reconciles budgets, selection rules, source snapshots, encodings,
scores, states, flags, metrics and inference immutability.

- [Summary and seed/group records](../artifacts/studies/representation-v1/summary.json)
- [Paired differences](../artifacts/studies/representation-v1/paired_differences.csv)
- [Paired endpoint plot](../artifacts/studies/representation-v1/paired_representation.png)
- [Monitoring curves](../artifacts/studies/representation-v1/representation_learning.png)
- [Verification](../artifacts/studies/representation-v1/verification.json)

Dashboard results are gated on a passed audit and output checksums. The original
four-family baseline table remains separate; this study does not label DQN tuned
or promote a replacement selected system. Current test/live-check status is in
[the milestone status](representation_status.md).
