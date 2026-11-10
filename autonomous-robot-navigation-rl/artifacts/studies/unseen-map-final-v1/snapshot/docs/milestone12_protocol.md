# Milestone 12. Frozen independent unseen-map evaluation

Registered 10 November 2026 before final policy execution. The owner authorized
evaluation of the twelve selected M11 policies on the ten reserved layouts.
The benchmark generation and layout split were registered in M10. M11 development
validation selected the checkpoints. This stage freezes that complete roster,
the task, input features, models, code, dependencies and analysis before parsing
the reserved scenarios in the dedicated final runner.

## Frozen support and behavior

Use every M11 main policy, Q-learning, SARSA, DQN and PPO, with training seeds
12101, 12203 and 12301. Copy the exact selected checkpoint and metadata from
each completed verified run. Retain the selected interaction and development
summary as provenance. Freeze source, configuration, protocol, benchmark files
and policy inputs under a SHA-256 file manifest and seal digest. Copying opaque
benchmark bytes for the snapshot precedes parsing the final layout records.

Each learned policy evaluates all 480 routes on ten final layouts, 48 routes per
layout. Total learned support is 5,760 policy–route episodes. BFS evaluates the
same 480 routes once. Random evaluates the same cases with seeds 13109, 13217
and 13309, preserving one RNG stream across the declared case order per seed.
Total planned support is 7,680 episodes. Learned evaluation uses environment
seed zero and deterministic argmax actions, including Up-first Q ties. The shared
1,234-feature input and original movement/reward/200-action rules apply.

The final runner performs frozen inference. A fresh load and learned-state
fingerprint precede each run, and an identical fingerprint must follow evaluation
and the 100-query warm latency probe. Ordinary training, development evaluation
and benchmark loaders retain their final-access guards. Saved final replays are
accessible through the sealed release and a verified completed result registry.

## Analysis registered before access

Report success, collision, timeout, SPL, return, conditional successful route
length, outcome counts and support for each policy and each layout. Four-family
headline values are equal-weight means over three training seeds. Report sample
SD over those seeds. Random has three evaluation RNG seeds, and BFS has one
deterministic run. Undefined successful-route means are excluded with their
defined-seed count retained. Dependent routes within a layout/policy do not count
as independent replications.

For success, collision, timeout and SPL, report 95% percentile cluster-bootstrap
intervals over the ten layouts. Use 5,000 draws from seed 13411 and the same
resampled layout indices for all methods. Average the fixed seed roster within
each layout, then resample layout means. These intervals describe layout
variation conditional on the trained policies. Training-seed SD is reported
separately. Ten layouts and three training seeds give limited precision. Report
development-to-final changes descriptively, without selecting new settings or
policies from final outcomes. Targets remain success .80, collision at most .02
and SPL .75. Preserve missed targets and every completed result.

Report evaluation wall time, actual environment actions, checkpoint bytes,
warm decision latency, original training costs and release storage. Execution
uses one CPU evaluation process with one Torch thread. A 3,600-second evaluation
wall limit and 512 MiB release limit apply. Audit time is separate and measured.

## Execution, recovery and acceptance

Create the snapshot and seal before constructing the final benchmark accessor.
Run the archived implementation against those exact snapshot inputs. Use one
exclusive release lock and atomic metadata writes. Publish each complete result
by renaming a separate pending directory. Interrupted pending files are retained
as incomplete attempts and never pooled with complete results. Compatible
completed results are reused after checksums and input identity checks. Any
technical replay of an interrupted deterministic run is disclosed. Successful
results remain the single canonical result for each registered policy/seed.

Save every trajectory compressed, per-route metrics and bounded examples of each
observed outcome on each layout. Independently replay every saved transition
through the core environment, reload frozen policies and compare their actions
and Q scores or PPO diagnostics. Check BFS optimality and exact Random streams.
Recompute summaries, intervals and support, and verify the seal before and after
execution. Regression fixtures exercise access guards, tampering, recovery,
uncertainty calculations and read-only dashboard playback. Preserve earlier
studies, original project documents, the energy project and sources.

The dashboard presents final evaluation separately from development studies,
including aggregate, per-seed/per-layout, reference, cost and saved-replay views.
The milestone is complete when all sixteen registered evaluations, the audit,
regression checks, dashboard checks and measured report are complete. Final
access makes these layouts inspected evaluation evidence for future work.
