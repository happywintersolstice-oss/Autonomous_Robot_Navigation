# Codex Implementation Brief — Phase 5C: Directional Learning and State–Action Coverage Diagnosis

**Project:** Comparative Evaluation of Reinforcement Learning Strategies for Autonomous Navigation in Unknown and Dynamic Indoor Grid Environments  
**Repository:** `happywintersolstice-oss/Autonomous_Robot_Navigation`  
**Authoritative starting branch:** `feature/phase5b-basic-navigation-diagnosis-v1`  
**Reviewed Phase 5B commit:** `ae3ae73bf7ba5cbe36dcf879125f15fd14c43f4e` (verify current HEAD before making changes)  
**Working project directory:** `autonomous-robot-navigation-rl/`  
**Proposed new work branch:** `feature/phase5c-directional-coverage-v1`  
**Proposed study identifier:** `phase5c-directional-coverage-v1`  
**Status:** Instruction/implementation brief; no results are claimed here.

## 0. Mission and authority

Implement a **small, independently auditable Phase 5C** to determine why the four existing reinforcement-learning families fail to produce consistently goal-directed movement on **open, static 10×10 Fixture A**. The first deliverable is a **zero-training causal-hypothesis audit** of existing Phase 5A/5B evidence. A bounded, **one-factor sampling intervention** is permitted **only if** a preregistered analysis justifies it and engineering tests pass. Otherwise stop at diagnosis and report why no learning experiment was run.

Do not treat this brief as permission to start full Phase 5, train on obstacle/dynamic fixtures B–E, change rewards, design a new 644-feature encoder, add a planner fallback, train on final/unseen layouts, modify the historical benchmark, rewrite the UI, or merge to `main`.

The primary research question for Phase 5C is:

> Is poor directional performance on simple open-grid navigation associated with uneven **goal-direction / state–action coverage**, and does changing only the **training-case sampling schedule** improve deterministic goal-reaching behaviour at a fixed interaction budget?

A secondary question is whether some models are encountering a **goal-conditioning/value-learning problem even when the relevant directions were adequately represented**. A negative diagnosis is a legitimate outcome. Do not force a training run merely because a budget was allocated.

## 1. Read these source files before touching code

Treat committed source, not a prior assistant summary, as authoritative. Inspect at least:

- `docs/partial_phase5b_results.md` and `docs/partial_phase5b_diagnosis.md`
- `docs/partial_phase5b_protocol.md` and `configs/partial_phase5b_v1.yaml`
- `docs/partial_feasibility_results.md` and `docs/partial_feasibility_diagnosis_v1.md`
- `src/robot_navigation/partial_phase5b.py`
- `src/robot_navigation/partial_phase5b_diagnostics.py`
- `src/robot_navigation/partial_phase5b_fixtures.py`
- `src/robot_navigation/partial_feasibility_fixtures.py`
- `src/robot_navigation/partial_features_scaled.py`
- `src/robot_navigation/partial_training.py`, `partial_agents.py`, `partial_neural.py`
- `src/robot_navigation/partial_dynamic_env.py`, `partial_observation.py`, `partial_sensor.py`, `partial_memory.py`
- `artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/learning-curves.csv`
- `artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/paired-results.csv`
- `artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/final-verdict.json`
- Phase 5B `training-case-stream.json`, `training-actions.jsonl.gz`, optimizer logs, case-level evidence and checkpoint summaries for **all eight previous arms**.

Verify expected branch, parent commits, source hashes, input dimensions, action IDs, fixture hashes, saved records and baseline test counts; if inconsistent, **stop and report** rather than silently adapt. Link relevant original evidence in the new report. Preserve all old artifacts **byte-for-byte**.

### Verified facts motivating Phase 5C (do not recast as new outcomes)

The completed **single-seed** Phase 5B trial used eight arms (Q-learning, SARSA, DQN, PPO × B0 original/B1 relative-goal ×4), **6,000 actions per arm**, for 48,000 training actions. Every family failed the predeclared **complete** learning screen. On the 16-case expanded A diagnostic panel at 6,000 actions:

| Family | B0 original features | B1 relative-goal ×4 | B1 − B0 |
|---|---:|---:|---:|
| Q-learning | 6/16 | 7/16 | +1 |
| SARSA | 5/16 | 7/16 | +2 |
| DQN | 2/16 | **10/16** | **+8** |
| PPO | 0/16 | 5/16 | +5 |

DQN B1's 16-case direction breakdown was **Up 3/4, Down 4/4, Left 3/4, Right 0/4**. Its six expanded-case failures were timeouts with **zero Wait actions**, so an all-Wait explanation alone is insufficient. Its successes were not uniformly efficient; the report documents substantial route excess and oscillations. The goal-directed information is representable by the existing features on adjacent-goal cases, so insufficient representational capacity has **not** been demonstrated to explain Fixture A failure.

**Critical design caution:** The existing eight A training pairs already include **two examples per direction** (one one-step and one two-step), so the list of training cases is direction-balanced. A mere round-robin direction scheduler may not materially change effective **state–action exposure**. Audit actual episode selections, transition counts, on-goal action choices, and rightward goal-bearing states first. Do **not** presuppose the problem is direction imbalance. Repeated testing on already-inspected expanded development pairs is not an independent confirmation.

## 2. Non-negotiable environment and policy contract

For every diagnosis and (if justified) intervention:

- Grid: fixed 10×10 **open static A**; no extra obstacles/events during this stage.
- Agent knows current position and goal, not a hidden obstacle map. Observation remains the **409-value float32 vector** created from current local sensing and legal memory.
- Perfect cardinal sensors: North/South/East/West, two-cell rays, first-obstacle occlusion; memory updates only observed cells.
- Actions: `0=Up`, `1=Down`, `2=Left`, `3=Right`, `4=Wait`. Sensor ray order differs from action order; test explicitly.
- Rewards unchanged: `-0.01` per action including Wait; `+10` goal bonus; `-5` collision penalty; **no** extra timeout penalty or hidden shaping; max 200 actions is an external truncation.
- Q-learning/SARSA remain linear semi-gradient approaches; DQN/PPO remain the existing neural implementations.
- **Hold features fixed at existing B1 `partial-visible-relative-scale4-v1` for both new Phase 5C arms** if a learning experiment is warranted. That is a development choice informed by the inspected Phase 5B evidence, **not** independent model selection. No other scale factor may be tuned here.
- Keep the existing algorithm-specific optimizers, architectures, exploration parameters, replay settings, learning rates, discount, reset handling and termination/truncation semantics fixed. Record the exact historical settings instead of claiming all methods have identical optimization capacity.
- No action masks, forced movement, anti-Wait logic, hand-coded goal-directed action labels, imitation learning, planner fallback, oracle action access, future event data, environment state injection, obstacle-type labels or hidden optimal-distance shaping.
- Diagnostics may use privileged truth **only after or separate from actions**, labelled explicitly as evaluation-only. The policy must receive only the permitted observation.

## 3. Phase 5C stage C0 — ZERO-LEARNING evidence audit (mandatory; no new model actions)

Implement a read-only analysis over existing Phase 5B evidence. Audit and reconcile with the raw saved training records, not just the final summary.

### 3.1 Coverage definitions and denominators

For each **algorithm × feature arm × available checkpoint**, produce exact counts, with denominators, for:

1. **Reset case exposure**: count of episode starts for every one of the eight A training start–goal pairs; counts by goal direction and by one-/two-step distance. Do not confuse episode starts with total environment interactions.
2. **Decision-step exposure**: actual number of action decisions while goal lies north/south/east/west (including any agent overshoot where the relative goal changes direction). At diagonal offsets report multi-axis cases separately or use an explicitly fixed dominant-axis rule; never count the same transition as two disjoint samples without declaring that convention.
3. **State–action coverage**: selected Up/Down/Left/Right/Wait by current goal direction, robot grid quadrant, boundary/interior band, distance band, and visited/history features. Show both action counts and the proportion that **reduced current Manhattan distance**, preserved it or increased it. For moving actions, distinguish attempted collisions and actual successful moves.
4. **Near-goal exposure**: at distance exactly 1 or 2, frequency of correct immediate target action, wrong-direction movements, Wait, and collision; distinguish exploratory action (`epsilon`/stochastic PPO) from greedy/policy-preferred action at that same pre-action observation.
5. **Learning signal**: successful goal transitions by action direction, position/goal pair, 1-/2-step cases; positive goal rewards stored in DQN replay and represented in PPO rollouts where recorded. Record optimizer/update counts and actual exploration schedules by checkpoint. Do not invent absent stored information.
6. **Decision quality**: first action on all original and expanded development cases, pre-action Q values or PPO probabilities, correct-direction margin, direction-by-direction outcome, Wait fraction, loop/oscillation count, total actions, and failure category.
7. **Paired historical condition**: compare B0 and B1 **within each family** using the same inspected cases, but never equate feature-arm trajectories or pretend that their stepwise state distributions were identical.

Use explicit categories for **zero-probability / unobserved states**; a direction with few samples is not proof of causal undertraining. Keep all denominator definitions in a data dictionary. Distinguish episodes, transitions, goals and distinct states in every table.

### 3.2 Counterfactual score probes (zero training)

For saved policies (including untrained, 2k and 6k), evaluate policy action scores/probabilities on **predeclared synthetic permitted observations** with:

- A fixed robot position and four goals at a one-cell offset, maintaining other legal observation/memory fields as consistently as possible.
- Identical relative displacement at different absolute coordinates (translation probe).
- One- versus two-cell goal distances in each direction.
- Mirrored/rotated **legal** observations with correctly rotated occupancy, rays, visit/age/streak history, absolute coordinates and target action mapping.
- The existing recorded right-goal failures (including scores of Right versus Up/Down/Left/Wait) versus right-goal successes where available.

Do not rotate just the goal while leaving incompatible map channels unchanged. Probes must use legal observations constructed by the new environment or carefully validated equivalent transformation. Identify whether observed policy decisions are direction-biased, location-biased or state-history-sensitive; **do not assert causality from probes**.

### 3.3 Mandatory zero-learning verdict

Write `diagnosis-only.md` and machine-readable `coverage-audit.json` with:

- Hashes/paths for every source evidence file read; exact case/transition totals and discrepancies.
- Side-by-side direction exposure, goal transition exposure and failure margins per family.
- A concrete **case-sampling hypothesis** and **falsifiable expected change**, OR an explicit finding that sampling is not a defensible first intervention.
- Why the existing two-per-direction A training case list does or does not support a scheduling intervention.
- A predeclared `coverage_intervention_justified: true|false` accompanied by an auditable reason and numerical evidence (not a vague judgement).

**Gate:** If the source audit fails, stop `IMPLEMENTATION_BLOCKED`. If the case-sampling hypothesis is unsupported (for example episode exposure is balanced but rightward failures arise despite substantial right-goal and Right-action coverage), stop `DIAGNOSIS_ONLY_SAMPLING_NOT_JUSTIFIED`. **Do not train simply to fill eight arms or produce a positive headline.** You may recommend a later optimizer/credit-assignment experiment, but do not execute it under this brief.

## 4. Phase 5C stage C1 — Single-variable sampling experiment (conditional)

Only if Gate 3.3 supports it, register and then execute **one** preregistered paired intervention. It may be further narrowed or cancelled based on diagnostic feasibility **before** registration, not rewritten after outcomes.

### 4.1 The only variable allowed to change

Hold the same **eight existing A training scenarios**, the B1 feature transform, reward, action space, algorithm hyperparameters, network sizes, exploration rules, model seed within paired arm, training interaction budget and assessment cases fixed. Compare:

- **C0 baseline:** the existing seeded uniform-random selection of the eight A scenarios at episode resets, as used in Phase 5B.
- **C1 intervention:** episode-reset **stratified/cyclic selection** using exactly those eight scenarios with seeded, shuffled balanced eight-episode blocks (two starts per direction, with the available one-/two-step cases). All actions remain chosen by the RL model itself. No goal-direction or case identity is added to observation. No change to transition reward or environment dynamics.

**Important:** C1 guarantees equal episode-start exposure over complete eight-episode blocks, **not** equal action counts, equal successful goal rewards, equal visitation or identical trajectories. Record the remainder and actual action-level exposure. Before running, demonstrate through a no-learning seeded schedule simulation that C1 meaningfully differs from C0 in the exposure characteristic identified in Stage C0; if it does not, stop without training rather than running a vacuous ablation.

The above sampling arms must be defined and hash-frozen before execution. Do not invent a third post-hoc prioritization arm, oversample only the failed Right cases, modify the case set, or use failure outcomes to adapt sampling during the trial. If the evidence instead supports a different intervention, **propose it separately and seek a new specification**; do not silently substitute one.

### 4.2 Fixed development evaluation panels

Evaluate both arms on the *same* case IDs in exactly this order:

1. **Original A:** four existing Phase 5A development pairs (previously inspected).
2. **Expanded A:** 16 Phase 5B development pairs (previously inspected).
3. **New Phase 5C development-only panel:** 16 newly generated predeclared open-grid cases not overlapping training, original A or expanded A pairs. Require four goals per direction, eight one-step/eight two-step, balanced quadrants, and a recorded boundary-band distribution; choose seed `23042`, audit D4 transformations/overlap and case hashes before training. Geometry remains familiar open grid; call them *new development pairs*, **not unseen maps** and **not independent confirmation of a model selected using the same panels**.

Never inspect fresh sealed final maps or use the old held-out static layouts as new independent final data. Do not create a final map reserve under this task.

Use same-information online replanner and a clearly separated scripted mechanic check on all evaluation cases. The script is not a learned comparator. Fixed case set, fixed seed, deterministic evaluation; no checkpoint selection on evaluation outcomes.

### 4.3 Strict finite budget (maximum, not a target)

If training gate passes, apply exactly:

| Item | Maximum / rule |
|---|---|
| Algorithms | Q-learning, SARSA, DQN, PPO |
| Arms | C0 uniform and C1 stratified |
| Primary seed | `23041` for every paired arm; independent model/RNG instances |
| Actions per arm | 4,000 actual environment actions |
| Checkpoints | 0, 2,000, 4,000 actions; all retained |
| Total training ceiling | 32,000 actions (4 × 2 × 4,000) |
| Evaluation action ceiling | 200,000 total, measured actual actions |
| Primary suite wall ceiling | 1,800 seconds |
| Per-arm wall ceiling | 300 seconds |
| New artifact byte ceiling | 512 MiB; fail safely if approaching |
| Training seeds | One primary seed only; no seed hunting or confirmation inference |
| Retry policy | No unregistered retries, post-result parameter changes or hidden additional attempts |

Record resource use including training seconds, assessment seconds, serialization overhead, memory and CPU scope. No time estimate is promised; enforce hard measured ceilings. If a limit is reached, preserve the exact incomplete record and report `INCOMPLETE_BUDGET_OR_EXECUTION` instead of pretending the study completed. If the archive ceiling is impractical, **stop before training** and propose a hash-addressed raw-evidence strategy; never drop or truncate evidence to fit a success story.

### 4.4 Paired fairness and implementation details

- Initialise each family/arm independently with the same **within-family seed** and initial model parameters (verify parameter fingerprint equality at time 0); C0 and C1 will diverge as trajectories differ.
- Use the same feature version `partial-visible-relative-scale4-v1`, with strict checkpoint signatures; no cross-version loads.
- Match training **interaction counts**, not episodes or actual visited states; report that PPO/TD/neural optimizer work is not identical.
- Continue the same live policy, optimizer, DQN replay, RNG and environment across 2k/4k boundaries; no checkpoint boundary reseeding. Recreate independent fresh model streams between arms.
- Preserve true action traces before any SB3 autoreset, exact terminal versus truncation semantics, SARSA pre-update next-action handling, DQN exploration schedule (including any SB3 continuation change), and PPO stochastic training vs deterministic evaluation.
- Capture the actual selected case-ID sequence and exactly when resets occur. Sampling code is trusted training orchestration, **not** a policy input; enforce no additional hidden input at inference.
- Distinguish initial goal-direction exposure from **current** goal direction after wandering; schedule control applies to episode starts only.
- Report action probabilities / Q values, goal-direction margins, training goal rewards and actual train-vs-evaluation behaviour.
- Maintain source, model, manifest, data and saved-policy hashes. Run existing tests and new tests before registration; run all audits after execution.

## 5. Tests required *before* any learning run

Implement focused tests and document commands/results:

- Existing 409-value observation and 412-value B1 feature parity (NumPy/Torch), no hidden state injection; immutable five-action and reward contract.
- Sensor North/South/East/West order distinct from action order, rotation/collision test.
- All saved Phase 5B evidence remains hash-identical; audit contains no training actions and causes no checkpoint mutation.
- Every reset counts one correctly categorised training case, including incomplete episodes and early budget cutoffs.
- C0's seeded selection matches actual existing uniform selection; C1's full eight-episode blocks contain the declared case distribution, with reproducible shuffle and a recorded partial last block.
- Neither sampler inspects prior success/failure, learned scores, hidden topology, test outcomes or future blockage schedule to pick its next training case.
- Initial C0/C1 policies and RNG setup are paired within family; checkpoint continuation does not silently restart exploration/optimizer/replay.
- Artificial directional probe rotation permutes actions correctly; no impossible occupancy/visit/timestamp combinations.
- Saved action/score replay matches actual policy checkpoints; checkpoint metadata rejects wrong feature or action signatures.
- Outcome metrics separate success/collision/timeout, episodes/actions, natural termination/external truncation, learned versus scripted references.
- Injected limit/exception failures are recorded accurately and do not generate a false passing verdict.
- New cases have no duplicate training or prior development start–goal pairs; no final data is touched.
- Full pre-existing regression suite remains passing. A successful unit test suite **is not** evidence that the models learned.

## 6. Metrics, analysis, interpretation and predeclared screening

Provide tables for each family and each arm at **all three** registered checkpoints, never only a best checkpoint. Report:

- Success/collision/timeout on original 4, expanded 16, new development 16, **each denominator separately**.
- Success by goal direction (Up/Down/Left/Right), by distance 1 versus 2, quadrant and boundary band.
- First-action goal-direction correctness, goal-direction score/probability margin; no assertion that greedy action is globally optimal in non-A tasks.
- Median and mean episode actions; excess actions on success relative to **open-grid** Manhattan distance, never charge successful-only metrics to failures.
- Wait fractions, longest wait streaks, revisits, A-B-A-B oscillation windows and longest no-progress interval.
- Actual episode-start distribution, action-step distribution, goal-reaching transition count per direction, action chosen when goal is one step away, numeric optimizer/update diagnostics.
- Direct **per-case paired** success-to-failure, failure-to-success, retain-success, retain-failure totals for C1 vs C0, plus change in average steps and direction exposure.
- Action budgets and runtime; differences in model capacity and family-specific updates; no cross-family superiority claim from this pilot.

**Predeclared directional signal screen** (interpret strictly as a *single-seed diagnostic*, never as proven improvement): at the **4,000-action checkpoint**, for a given family C1 must satisfy all of:

1. On the **new 16-case development panel**, at least **12/16** successes and at least **+3 successes** relative to matched C0.
2. In those same cases, at least **2/4 Right goals**, and no goal direction below **2/4**.
3. On the existing 16-case expanded A panel, at least **10/16** successes and at least **+2 successes** over matched C0; on original 4, at least **2/4**.
4. Verified training-case exposure changed as hypothesised; no integrity, leakage, or reward/action-rule failure.

The above are **new Phase 5C thresholds** and **not** Phase 5B's thresholds. Freeze them before training and never relax them afterward. If C0 itself performs well, C1 can legitimately fail the **improvement** criterion despite adequate absolute success.

With one seed and correlated cases, do not calculate naive IID significance tests or claim that a family is definitively better. Report effect-size direction and uncertainty limitations. No automatic permission for obstacle/dynamic training, full Phase 5, final access or merging—even if some family passes.

## 7. Deliverables and file organization

Create a *new* versioned namespace and keep previous files untouched. Suggested files (adjust names only after documenting reasons):

```text
# From autonomous-robot-navigation-rl/
configs/partial_phase5c_v1.yaml
src/robot_navigation/partial_phase5c.py
src/robot_navigation/partial_phase5c_coverage.py
src/robot_navigation/partial_phase5c_fixtures.py
src/robot_navigation/partial_phase5c_report.py
developer/tests/test_partial_phase5c.py
docs/partial_phase5c_protocol.md
docs/partial_phase5c_diagnosis.md
docs/partial_phase5c_results.md
artifacts/partial-dynamic-navigation-v1/phase5c-directional-coverage-v1/
    source-and-evidence-manifest.json
    coverage-audit.json
    diagnosis-only.md
    preregistered-protocol.json  # only if training justified
    train-action-budget-ledger.jsonl  # only if training executed
    controls/                    # labelled conventional / scripted
    training-and-evaluation/     # arms, checkpoints, raw replay evidence
    learning-curves.csv
    paired-results.csv
    replay-audit.json
    gate-results.json
    final-verdict.json
    delivery-verification.json
```

Avoid copying the entire repository or thousands of nested source snapshots into this new artifact directory. The Phase 5B branch already had very long Windows paths and large archives. Instead, create a **minimal** source snapshot of newly created/changed source, a complete manifest with Git commit/tree SHA and per-file SHA-256 for all dependencies, and hashes of immutable historical evidence. Preserve replay completeness and auditability; do not discard necessary action records. On Windows, document `git -c core.longpaths=true` if needed; do not rename frozen historical files.

Do not overwrite frozen Phase 5A/5B artifact directories or re-run training into their namespaces. Write coherent docs with commands and root-relative paths. Report actual test counts after running rather than claiming expected counts. A validated code pathway and a successful learning result are separate gates.

### Outcome labels

Use one terminal verdict, with a machine-readable reason:

- `IMPLEMENTATION_BLOCKED`: source, contract, reproducibility or validation issue.
- `DIAGNOSIS_ONLY_SAMPLING_NOT_JUSTIFIED`: audit passed, but sampling is not a credible intervention; **0 new training actions**.
- `INCOMPLETE_BUDGET_OR_EXECUTION`: preregistered pilot interrupted or incomplete; never fill missing rows.
- `SAMPLING_HYPOTHESIS_SUPPORTED_IN_ONE_SEED`: one or more families meet **every** Phase 5C screening criterion.
- `SAMPLING_HYPOTHESIS_NOT_SUPPORTED`: completed pilot, no family meets the complete screen.

If only a subset of families meets the screening, give the exact family count and their names, and retain all failures. Never turn any terminal verdict into automatic approval for full training.

## 8. Workflow and final report to the user

1. Verify branch head and inspect all listed historical sources. Create a new Phase 5C branch **without altering the user's existing dirty workspace**. If branch/worktree state is unsafe, stop and explain.
2. Build **read-only** historical coverage and directional probe analysis. Audit calculations from raw Phase 5B records; write a reproducible diagnosis with explicit denominators.
3. Decide the sampling gate based on predeclared criteria and evidence. If unsupported, report diagnosis-only results and stop without training.
4. If justified, implement both samplers and strictly matched model builders, generate and hash-freeze new development cases, implement failure-injection/unit/leakage tests, run all regression gates.
5. Register the frozen protocol (source/data/model/seed/case/threshold/limit hashes) **before the first training action**. Then run at most the allowed eight arms; preserve every success/failure and partial prefix.
6. Perform replay audits, budget reconciliation, case-pair reconciliation, checkpoint reload parity, historical hash preservation and post-run regression suite.
7. Create result report with exact counts, directional failure traces, tables, controls, limitations, predeclared screen, and **one** narrowly argued next proposal. No automatic further experimentation.
8. Present to the user what code was created, what was actually run, commit branch/name, verified results and blockers. **Commit or push only if explicitly requested**; never merge into `main` on this brief.

### Suggested Codex invocation

> Read `CODEX_PHASE5C_DIRECTIONAL_COVERAGE_DIAGNOSIS.md` in full. Start from `feature/phase5b-basic-navigation-diagnosis-v1` and verify the reviewed commit. Run the zero-training coverage and goal-conditioning audit first. Do not assume a direction-balanced sampler can fix the observed Right failures because the eight training tasks are already balanced by direction. If the preregistered diagnosis supports a sampling intervention, implement and execute only the bounded A-only C0-versus-C1 experiment under the frozen contract, then audit and report every outcome. If the hypothesis is not supported, stop with a diagnosis-only report and zero new training actions. Preserve all historical artifacts, never access final evaluation data, and do not start full Phase 5 or merge into main.

---

**Scope principle:** Phase 5C is designed to **explain and test** a specific failure, not to engineer a successful-looking result. Negative findings, preserved evidence and correct stop decisions are acceptable outcomes.
