# Partial/dynamic benchmark implementation status

Implementation date: 20 November 2026. Specification: repository-root
`CODEX_ROBOT_NAVIGATION_IMPLEMENTATION_SPEC.md`.

## Scope and gates

This is an additive implementation of `partial-dynamic-navigation-v1`. Existing
warehouse, fixed-grid and unseen-static experiments and the app remain intact.
Phases 0–4 are the implementation scope; phase 5 needs agreement on study budgets,
and phase 6 needs protocol/model/final-freeze authorization. No substantive model
comparison, final evaluation or new UI is authorized by this implementation run.

| Phase | Status |
| --- | --- |
| 0: inspect and baseline | Tested: 250 existing tests passed; 9,490 pre-existing files inventoried |
| 1: sensing, memory, observation | Implemented/tested: nine initial unit tests passed |
| 2: dynamic Gymnasium environment | Implemented/tested: twenty combined core/leakage tests passed |
| 3: development manifests, reference, metrics | Implemented/tested: 30 layouts, 360 scenarios; reference and replay gates passed |
| 4: four adapters and bounded smoke validation | Implemented/tested: all 42 new gates passed; 4 x 256 training actions and 16 assessment episodes audited |
| 5: developmental training study | Not run; budgets and selection protocol pending agreement |
| 6: final evaluation and UI | Not run; authorization and fresh final freeze pending |

## Baseline and compatibility

Git HEAD: `cf7f0f11d3a94d8533c6a1ff7b4834f1db78f141`. The tree already contains
uncommitted warehouse, dark-theme, 3D, organization and reporting changes, plus
the user specification. These are preserved, not reset or folded into new task
semantics. Local Python/dependency versions and pre-existing file hashes are
captured in the validation evidence. Baseline command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime/partial-baseline
```

The brief's `Demo` route and `robot_navigation.demo_assets` command are stale:
the current interface is **Simulator / Results / Guide**, and the current asset
verifier is `robot_navigation.project_assets`. Keep those working entry points.
No historical source, protocol, checkpoint, report or scientific artifact is
to be edited. New code uses separate `partial_*` modules and versioned paths.

The initial full baseline passed **250 tests in 178.41 seconds**. After adding
the extension, a byte-for-byte check of all **9,490** pre-existing files found
**zero changed files and zero missing files**. The current project asset check,
historical Gymnasium checker and installed dependency consistency also passed.
The final combined regression result is recorded in the validation report
linked below; the app itself, old tests and dependency files were not modified.

## Implemented files and phase evidence

| Phase | New files and evidence | Tests/results |
| --- | --- | --- |
| 0 | This status file; baseline Git status/revision and raw-byte inventory in validation bundle | Existing regression baseline: 250 passed |
| 1 | `partial_sensor.py`, `partial_memory.py`, `partial_observation.py`, `partial_contract.py` | Range/occlusion/boundaries; occupancy-only equivalence; stale age and interrupted streaks; exact 409 slices: nine passed |
| 2 | `partial_dynamic_env.py`, `partial_fixtures.py` | Gymnasium API, five actions, time/reward/terminal rules, suppression, deterministic replay, hidden-map/future-event equivalence: twenty combined tests passed |
| 3 | `partial_benchmark.py`, `partial_baselines.py`, `partial_evaluation.py`, new `data/partial/` namespace | D4/historical exclusion, disjoint layouts, seeded events, Wait/detour/permanent-wall cases, metrics and tamper rejection; extended environment/reference/split group: seventeen passed |
| 4 | `partial_features.py`, `partial_agents.py`, `partial_neural.py`, `partial_training.py`, `partial_commands.py`, `configs/partial_dynamic.yaml` | Eleven adapter tests initially passed; complete new validation command: 42 passed in 5.85 seconds |

New regression files are `developer/tests/test_partial_{sensor,memory,observation,
dynamic_env,leakage,benchmark,evaluation,agents,commands}.py`. Existing tests
were retained without assertion changes. One implementation defect was found
and fixed during adapter validation: SB3 mutates the returned `info` dictionary
when adding vector/autoreset metadata. The trusted recorder now deep-copies it
before returning, keeping raw recorded transitions reproducible. Both failing
neural recording audits then passed; no expected dynamics were weakened.

The final combined command passed **292 tests in 186.01 seconds**:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime/partial-final-regressions --junitxml=.runtime/partial-final-regressions.xml
```

That is all 250 pre-existing tests plus 42 new tests, with zero failures or
skips. The new benchmark's smoke bundle separately passed its checksum,
checkpoint, training-action and assessment-replay audit command.
Final test review added explicit out-of-bounds collision and SARSA reset-action
assertions to the new tests; all 42 new gates passed again in 5.56 seconds.
No production code or pre-existing test changed after the full regression run.

The [environment specification](partial_dynamic_environment_spec.md) defines
the data-flow boundary, exact time convention, versions, metric denominators and
limitations. The [validation protocol](partial_dynamic_experiment_protocol.md)
declares the short matrix, settings, data split and future authorization gates.

## Actual data and smoke findings

The development manifest contains **24 training and 6 development layouts**,
with **288 training and 72 development scenarios**. Thirty new geometries are
D4-disjoint and exclude **31** already inspected historical geometries. Base
diameters range from 17 to 26 and dead-end counts from 1 to 9. Of 540 scheduled
single-cell blockage impacts across dynamic scenarios, 16 disconnect the base
route while active. These are preparation statistics, not policy performance.
Route lengths are balanced by bands; dead-end/bottleneck/impact quotas are not
scientifically optimized and should be reviewed before a substantive study.

The inspectable [manual discovery trace](../artifacts/partial-dynamic-navigation-v1/implementation-validation-v1/manual-discovery-trace.json)
waits three times, observes the sole-corridor obstacle reopen, then reaches the
goal in seven moves. Its ten actions, observations, memory updates and events
replay exactly. No learner or hidden planner supplies these manual actions.
Tests also cover a short detour beating waiting, a long detour for which waiting
is faster, unchanged permanent walls, remote activation/clearance with stale
memory, and trajectory-dependent occupied-cell suppression.

The [functional smoke bundle](../artifacts/partial-dynamic-navigation-v1/implementation-validation-v1/smoke/summary.json)
contains all four fresh models at seed 22001, **256 actions each**, with real
training recordings. All parameters changed and remained finite; checkpoints
reloaded identically; inference preserved learned state; all **1,024 training
actions and 16 assessment episodes** passed independent saved-action audits.
The sequential smoke matrix took 16.17 seconds including recording/export/audits.

| Method | Training actions | Completed training episodes (all collisions) | Four smoke assessment cases |
| --- | --- | --- | --- |
| Q-learning | 256 | 18 | Four timeouts; 200 Wait actions in each |
| SARSA | 256 | 19 | Four timeouts; 200 Wait actions in each |
| DQN | 256 | 35 | Four timeouts; 200 Wait actions in each |
| PPO | 256 | 28 | Four timeouts; 200 Wait actions in each |

**Implementation validation passed; successful learning has not been shown.**
No model discovered a goal during these tiny training runs. Their deterministic
smoke decisions selected Wait throughout assessment. A 200-Wait episode returns
-2, whereas an immediate collision returns -5.01: the conservative baseline
can favor stalling before a successful route is learned. This observation does
not isolate reward design as the cause or compare learning families. The small
budget and broad shared feature approximation are also unresolved factors.
No waits were overridden, failures removed, rewards changed, or extra training
launched to make the output look better.

## Commands and validation evidence

Run from `autonomous-robot-navigation-rl/` using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands validate
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands prepare
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands trace --output .runtime/partial-manual-trace.json
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands smoke --output .runtime/partial-functional-smoke-v1
.\.venv\Scripts\python.exe -m robot_navigation.partial_commands audit --output .runtime/partial-functional-smoke-v1
```

Choose a fresh output name for each trace/smoke run; existing bundles are not
overwritten. The smoke command refuses stale source/config/test gates and has
a hard matrix limit of four algorithms by 256 actions, plus a 120-second wall
limit. There is no large-training or final-evaluation command. The development
reader also refuses final/test access. Exact snapshot inputs, output checksums,
raw recordings, per-case metrics and metadata remain in the new namespace.

Current evidence:

- [Implementation verification report](../artifacts/partial-dynamic-navigation-v1/implementation-validation-v1/verification.json)
- [Baseline revision/status and file hashes](../artifacts/partial-dynamic-navigation-v1/implementation-validation-v1/preexisting-inventory.json)
- [Development manifest](../data/partial/partial-dynamic-navigation-v1/manifest.json)
- [Smoke source/config/data input hashes](../artifacts/partial-dynamic-navigation-v1/implementation-validation-v1/smoke/inputs.json)
- [Smoke output hashes](../artifacts/partial-dynamic-navigation-v1/implementation-validation-v1/smoke/outputs.json)

## Not implemented/run and remaining decisions

No phase-5 comparative training, tuning, confidence intervals, fresh final
dataset, final evaluation or phase-6 UI was run. No new dynamic performance
headline is claimed. The existing warehouse Simulator / Results / Guide and
root launcher retain their original behavior. Old checkpoint formats remain
incompatible with this task and are explicitly rejected.

Before phase 5: agree budgets and at least three independent seeds per family;
review difficulty/event-impact composition; seal an independent final reserve
before model selection; register screening/confirmation rules; test representation,
duration and horizon sensitivity; diagnose waiting without silently adding a
recovery controller. Reward changes require a new task identity and a registered
matched comparison. The online reference is a transparent heuristic that can
also fail, not an optimal dynamic planner. Recovery/loop/wait diagnostics are
operational definitions with the limitations stated in the environment spec.

After protocol/model/final-freeze authorization, phase 6 may add the distinct
known-versus-true-map view, sensor/event overlays and side-by-side policies.
Those visual features are planned, not implemented or visually validated here.

## Provisional choices and unresolved research questions

The 409-value input, two-cell cardinal sensor, 3–8-step event durations,
200-action cap and conservative rewards are engineering defaults, not validated
learning choices. A blocked reading cannot identify permanent versus temporary
occupancy. Linear versus neural capacity and compute differ. Successful learning,
generalization and superiority over planning remain unmeasured questions.

Only training and development layouts will be generated during implementation.
Ten fresh final layouts remain a proposal, not a generated or exposed test set.
They must be independently sealed before substantive model selection.
