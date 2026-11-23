# Phase 5B: registered A-only relative-goal scaling diagnosis

Protocol version `partial-a-learning-diagnosis-v1`, study `phase5b-a-only-v1`.
The root work order `CODEX_PHASE5B_BASIC_NAVIGATION_DIAGNOSIS.md` authorizes this
bounded study only. Start from reviewed GitHub commit
`d3822423ffd6fa3c83f9a26fc15808fa9103e36c`, on a new branch
`feature/phase5b-basic-navigation-diagnosis-v1`. No main merge, final/test access,
obstacle curriculum, full Phase 5, UI changes or post-hoc parameter selection.

## Gates, immutable evidence and registration

G0 verifies every historical Phase 5A delivery pin and the pre-existing tracked
file inventory. Historical source/config/checkpoints/results remain unchanged.
The original dirty main workspace remains separate. New source modules extend
the old provenance glob: audit the recorded historical source list, not a newly
invented Phase 5A signature; do not rewrite its original strict registration.

G1 is a read-only audit of old checkpoints and traces, with no newly executed
learned policy or training interactions. Reconstruct linear TD arithmetic and
behavior RNG against archived 2k/6k/10k weights. Inspect all four A reset decisions
at every checkpoint; save training-only feature distributions, Q contributions,
TD errors, update norms, values, optimizer moments, exploration/reset schedules
and permitted-information examples. Old inference checkpoints do not retain
live gradients; state that limitation rather than fabricate measurements.
Mechanical rotation, Bellman, termination/truncation, SARSA continuation,
feature parity/leakage, versioning, continuity and failure-injection tests must
pass. A demonstrated baseline correctness/optimization defect halts training
and receives a reproducible defect report, separate from any scaling comparison.

G2 requires passed baseline/full/focused regression XMLs, current validation
hashes, a passed zero-learning audit, deterministic fixture audit and frozen
machine-readable manifest/seal before any primary training action. Inputs include
source, tests, config, protocol, brief, historical pins and case definitions.
Registration and execution are exclusive. Each family/arm has exactly one attempt;
failed actions consume reservations. Never delete an attempt to obtain a retry.

G3 reports all fixed checkpoints and the unchanged screen below. G4 independently
replays environment transitions and saved checkpoint decisions, checks source and
output hashes, parameter invariance, summaries, prefixes and budget reservations.
Full regressions and historical preservation must pass after the experiment.

## One factor and fixed task

B0 uses the original `partial-visible-affine-v1` numpy encoder, linear class and
SB3 builder unchanged. B1 uses `partial-visible-relative-scale4-v1`: exactly the
same 412 values, except the final two relative-goal coordinates are multiplied
by four. B1 numpy and vectorized Torch extractors must match. Raw observation
remains 409 float32 values in [-1,1]; transformed relative coordinates may lie
in [-4,4]. A strict feature signature rejects cross-arm checkpoint loading.

Both arms use the unchanged 10x10 environment, exact coordinates, two-cell
occluding cardinal rays, legal map memory, five existing actions, -0.01 per
action, +10 goal and -5 collision, no extra timeout penalty and external 200-step
truncation. Linear methods are normalized semi-gradient approximations, not
tabular. No planner fallback, action mask, imitation label, forced movement or
analytic witness enters a learner. The capacity witness is reset-observation
arithmetic only: no learning and no executed witness policy. The separately
labelled goal-direction script is a mechanics reference, not an RL competitor.

## Cases, order and randomness

Use exactly the old eight A training cases, in their original list order, sampled
uniformly by the original environment RNG. Each arm starts its independent
training environment at seed 23021. Episode-index case streams have identical
prefixes within a family when their lengths overlap; unequal episode lengths
mean different action trajectories and episode totals. Log all reset case IDs,
seeds and sampling order. No repeated trainer resets at checkpoint boundaries.

Original A evaluation is exactly the old four development cases. Register 16
additional cases before training: generator seed 23022, four cases per direction,
eight one-step/eight two-step goals, four starts per quadrant, distinct starts,
no pair overlap with training or original diagnostics. Sample candidates in
lexicographic order on the same open grid. Edge/interior counts are descriptive,
not required balanced. Freeze every case and its hash. These are development
pairs on familiar geometry, not unseen-layout generalization.

At each checkpoint assess both fixed panels once with reset seed 0. Run the
unchanged observation-only planner and separately labelled goal-direction script
once per case. References total 40 episodes; learned assessments total
4 families x 2 arms x 3 checkpoints x 20 cases = 480 episodes. Untrained policies
are retained per arm. Paired tables compare identical case IDs, never pooled
model-family samples. There is no tuning evaluation or confirmation seed.

## Optimization and finite ceilings

Primary seed 23021. Per-arm cap 6,000 actual training actions, checkpoints at
0/2,000/6,000; total cap 48,000. Global wall cap 1,800 seconds, per-arm cap 300
seconds, monotonic guards before actions/checkpoint operations. Evaluation and
control reservations cap at 100,000 policy-executed actions. Replay verification
re-executes saved actions only and is separately labelled, not a policy assessment
or an additional learning run. The primary clock includes controls, training,
checkpoint/evaluation and in-run export checks; preflight and post-run independent
audits/regressions are outside it and reported separately.

Use the exact model block of `partial_feasibility_v1.yaml`: alpha .1, gamma .99,
neural LR .0003, 32x32 nets, DQN buffer 10k/starts100/batch50/frequency4/target250,
PPO rollout100/batch50/epochs5 and the unchanged SB3 defaults. Linear epsilon keeps
the original 10,000-action horizon (1 to .1 over 8,000), so at 6k it is .325;
do not silently compress it to the new budget. DQN public learn calls are 2k
then 4k with reset_num_timesteps=False, exactly as the Phase 5A first two blocks.
Its cumulative-endpoint-relative epsilon schedule may rise at the boundary;
this is logged and identical in rule within both arms. PPO entropy coefficient
remains zero. This is one within-family feature factor, not matched optimization
between different families. CPU runs are sequential with one Torch thread.

Continue the same live learner/optimizer/buffer/RNG/environment between blocks.
Save trusted compressed continuation state (including RNG and buffer/optimizer
state), linear continuation actions and immutable compressed action prefixes.
Evaluate reloaded copies under restored Python/NumPy/Torch RNG; fingerprint live
parameters, optimizer, buffers and local RNG before/after assessment. Observe
actual clipped gradient/update norms without replacing optimizer behavior.
Save training seconds per block separately from evaluation/serialization wall.

New artifact ceiling 512 MiB. Deterministic compression is used for traces and
continuation states. Check prospective write sizes plus buffered-write headroom,
before actions and large exports. Reserve 200,000 bytes before an action and
32 MiB before starting a complete assessment; keep 1 MiB for closing buffered
streams and failure metadata. Count actual compressed bytes at the write boundary.
A ceiling, interruption, numerical exception
or integrity failure stops the trial, preserves the actual partial prefix/error
and spent reservations, and forbids retry/progression. No evidence is dropped to
make a budget fit. An unfinished arm yields INCOMPLETE_BUDGET_OR_EXECUTION, never
a completed comparison. A correctness/integrity failure yields IMPLEMENTATION_BLOCKED.

## Frozen screening rule and interpretation

For each family's final B1 checkpoint, require all of:

- Original A at least 3/4 goals and at least +1 over both its own untrained B1
  and the matched B0 final checkpoint.
- Expanded A at least 12/16 goals and at least +4 over matched B0 final.
- Every engineering/integrity check passed, no pair leakage or policy bypass.

No earlier checkpoint can replace 6k. Report exact within-case paired outcomes,
success/failure changes, direction/distance/quadrant/band strata, action/value
margins, Wait/loops, training episodes/prefixes and measured resources. No p-values,
winner claims, unseen-layout conclusions or automatic onward permission.

Terminal verdict: IMPLEMENTATION_BLOCKED for correctness/integrity failure;
INCOMPLETE_BUDGET_OR_EXECUTION for unfinished execution; otherwise
SCALING_HYPOTHESIS_SUPPORTED_IN_ONE_SEED if at least one family passes the complete
screen, else SCALING_HYPOTHESIS_NOT_SUPPORTED. Give the exact count of families.
Recommend only one narrow follow-on, requiring new authorization.

## Commands and report

From the app root with its installed interpreter and PYTHONPATH=src:
`python -m robot_navigation.partial_phase5b diagnose`, then `partial_phase5b validate`,
`partial_commands validate`, `partial_phase5b register`, `partial_phase5b run`,
`partial_phase5b validate-postrun`, `partial_phase5b audit`, `partial_phase5b report`. Registration snapshots code,
tests and the exact executed protocol. Run refuses an existing execution lock.
Commit this implementation and actual evidence to the new feature branch;
do not claim a push unless performed. Preserve old reports as read-only links.
