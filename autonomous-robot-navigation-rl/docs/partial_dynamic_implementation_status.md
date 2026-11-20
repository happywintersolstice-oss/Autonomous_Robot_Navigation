# Partial/dynamic benchmark: implementation and branch status

Date: 20 November 2026. Namespace: `partial-dynamic-navigation-v1`.
Specification: repository-root `CODEX_ROBOT_NAVIGATION_IMPLEMENTATION_SPEC.md`.

## Publication scope

Branch `feature/partial-dynamic-navigation-v1` extends committed base
`cf7f0f11d3a94d8533c6a1ff7b4834f1db78f141`. It contains only the new benchmark,
its development data, tests, documentation and isolated validation evidence.
Existing committed experiments, reports, tests, launchers and UI are unchanged.

The original workspace also contains uncommitted warehouse, dark-theme, 3D and
organization changes. Those changes are deliberately excluded from this branch
and remain untouched in the original workspace. The earlier local 250/292-test
results included that broader workspace; they are not the branch test count.
The earlier local smoke bundle also captured that broader source tree, so it
is not included here. The branch has its own fresh source/config/data snapshot
and validation bundle, with no dependency on those uncommitted app changes.

## Completed phases

| Phase | Implementation and evidence |
| --- | --- |
| 0: audit and baseline | Recorded base revision; preserved existing tracked files; validated in a clean isolated worktree |
| 1: sensing/memory/input | Four two-cell occupancy rays; occlusion/boundaries; stale memory, visits and uninterrupted blocked streaks; exact 409-value float32 vector |
| 2: environment | New Gymnasium Discrete(5), Wait, seeded event schedules, occupied-cell suppression, conservative rewards, correct terminal/truncation handling |
| 3: development benchmark/reference/evaluation | 24 training and 6 development layouts; 360 paired static/dynamic scenarios; D4/historical exclusion; observation-only online planner; metrics and replay |
| 4: four RL adapters | Fresh five-action linear Q-learning/SARSA and SB3 DQN/PPO, shared permitted features, strict checkpoint metadata, bounded functional smoke validation |
| 5: comparative development study | Not run; requires budget/seed/search/selection agreement |
| 6: final evaluation and UI | Not run; requires protocol/model/final-freeze authorization |

Code is in fourteen `src/robot_navigation/partial_*.py` modules. Configuration is
`configs/partial_dynamic.yaml`. Nine `developer/tests/test_partial_*.py` files
exercise the new behavior. The [environment specification](partial_dynamic_environment_spec.md)
defines the data-flow boundary, version IDs, sensor/time/reward semantics and
metric denominators. The [protocol](partial_dynamic_experiment_protocol.md)
declares the dataset and bounded smoke settings; all large-study choices remain
provisional and gated.

## Branch validation

All **43 new validation gates** pass. A fresh-checkout check caught one issue:
`validate` created `.runtime` after invoking pytest, so the first invocation
could not create its nested temporary directory. The branch creates the parent
first and includes a regression test for that case. No old assertion was relaxed.

The isolated full suite passed **278 tests in 144.58 seconds**, including all 43
new gates, with no failures or skips. The result is recorded in the [verification report](../artifacts/partial-dynamic-navigation-v1/branch-validation-v1/verification.json)
and its JUnit evidence. The branch smoke bundle is independently replay-audited.

Tests cover hidden-map/future-event equivalence, sensor occlusion, interrupted
memory visibility, true out-of-range activation/clearance, unchanged chosen
Wait actions, collision/boundary/goal priority, deterministic schedules and safe
suppression, disjoint geometry, reference wait-vs-detour behavior, exact metric
counts, SARSA pre-update continuation/reset handling, SB3 terminal-observation
bootstrapping, and model save/load compatibility.

## Dataset and actual smoke findings

The [development manifest](../data/partial/partial-dynamic-navigation-v1/manifest.json)
has 288 training and 72 development cases on 30 connected new geometries. It
excludes 31 previously inspected historical geometry identities under D4
rotations/reflections. There is **no generated or accessible fresh final set**.
Route lengths are balanced across three bands; structural/event-impact quotas
and the provisional 3-8-step durations still need development review.

The [manual trace](../artifacts/partial-dynamic-navigation-v1/branch-validation-v1/manual-discovery-trace.json)
waits three times, observes a sole-corridor obstacle reopen, and completes seven
moves. All ten actions and their sensor/memory/event states reproduce exactly.

The [smoke summary](../artifacts/partial-dynamic-navigation-v1/branch-validation-v1/smoke/summary.json)
uses seed 22001 and **256 training actions per algorithm, 1,024 total**. All four
models have finite changed parameters, matching checkpoint reloads and unchanged
learned state during inference. The bundle contains real training actions and
four paired assessment conditions per model, rather than reconstructed learning.

All four models choose Wait throughout their four assessment cases and time out.
No training goal was discovered in these tiny runs. These are functional tests,
not successful learning or comparative performance evidence. A 200-Wait route
returns -2, versus -5.01 for an immediate collision; stalling incentives, sparse
successful exploration, limited budget and representation capacity require
registered development analysis. No failure was removed or action overridden,
and no extra training/reward change was used to manufacture success.

## Commands

Use the documented locked environment from the application directory:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands validate
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands prepare
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands trace --output .runtime/partial-manual-trace.json
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands smoke --output .runtime/partial-functional-smoke-v1
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands audit --output .runtime/partial-functional-smoke-v1
```

Use fresh trace/smoke output paths; existing bundles are not overwritten. Smoke
refuses missing/failed/stale source/config/test gates and accepts only the
registered four-by-256-action matrix, with a 120-second wall limit. There is no
large-training or final-evaluation command. The development reader refuses test
or final access. Checkpoints reject old four-action/full-map formats.

## Reproducibility and unresolved work

The [smoke input inventory](../artifacts/partial-dynamic-navigation-v1/branch-validation-v1/smoke/inputs.json)
pins its clean-branch source/config/protocol/data snapshot; the
[output inventory](../artifacts/partial-dynamic-navigation-v1/branch-validation-v1/smoke/outputs.json)
pins weights, actual action ledgers and assessed replays. Validation/test-source
snapshots and historical seal checks are retained in the branch verification
bundle. No dependencies or existing app files were changed.

Before phase 5, agree interaction/tuning budgets and at least three independent
training seeds per family, review structural/dynamic difficulty composition,
reserve an independent final cohort before substantive selection, and register
representation/duration/horizon/reward analyses. Occupancy alone cannot always
distinguish permanent from temporary blockage. The online reference is a
transparent heuristic, not an optimal dynamic planner. Recovery/loop/post-hoc
wait diagnostics have the precise limitations in the environment specification.

No large comparative training, final evaluation or phase-6 UI/browser review
has been run. The existing committed UI remains intact. Known-versus-true-map
visualization and any learned recovery intervention remain planned and gated.
