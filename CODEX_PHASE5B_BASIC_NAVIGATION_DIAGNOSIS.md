# Codex work order — Phase 5B: Basic Navigation Learning Diagnosis

**Status:** Authorized scope: one bounded, registered **A-only** learning-diagnosis experiment. **Not** full Phase 5, not final evaluation, not a claim of successful RL.  
**Repository:** `https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation`  
**Working branch:** `feature/partial-dynamic-navigation-v1` (latest reviewed Phase 5A commit: `d3822423ffd6fa3c83f9a26fc15808fa9103e36c`). Create a clearly named **new feature branch** from the actual reviewed head to avoid altering archived results; do not silently reset or merge anything.  
**Project directory:** `autonomous-robot-navigation-rl/`  
**Date of brief:** 20 November 2026  
**Owner's aim:** Four-member comparison of Q-learning, SARSA, DQN and PPO for indoor robot navigation on unknown maps, with local sensing, map memory, temporary obstacles and a Wait action. This work order deliberately addresses only the earliest navigation-learning prerequisite.

---

## 0. Read the actual repository before editing

Inspect these exact existing sources (adjust only if the paths have genuinely changed):

- `docs/partial_feasibility_results.md`
- `docs/partial_feasibility_diagnosis_v1.md`
- `docs/partial_feasibility_protocol.md`
- `configs/partial_feasibility_v1.yaml` and `configs/partial_dynamic.yaml`
- `src/robot_navigation/partial_feasibility.py`
- `src/robot_navigation/partial_feasibility_fixtures.py`
- `src/robot_navigation/partial_agents.py`, `partial_training.py`, `partial_neural.py`, `partial_features.py`, `partial_observation.py`
- `src/robot_navigation/partial_dynamic_env.py`, `partial_sensor.py`, `partial_memory.py`, `partial_contract.py`
- `src/robot_navigation/partial_pilot_diagnostics.py`, `partial_pilot_ledger.py`, and existing `developer/tests/test_partial_*.py`
- `artifacts/partial-dynamic-navigation-v1/feasibility-pilot-v1/` measured records and its provenance/hash manifests.

Read the repository's `AGENTS.md` or equivalent contributor instructions **if present**. Do not trust a report without checking the underlying source and immutable evidence. Verify branch, clean/dirty status, commit identity and available compute/dependencies. Preserve pre-existing uncommitted changes. If the precise branch or evidence is unavailable, stop with an explicit blocker; **do not recreate or manufacture historical results**.

### Ground truth motivating this task — Phase 5A (do not rewrite)

Phase 5A used one seed (`23001`), 10,000 training actions per family, and **four A diagnostic episodes** (adjacent goals on familiar open-grid geometry). Final A outcomes:

| Algorithm | Untrained successes / 4 | 10k-action successes / 4 | 10k-action Wait fraction | Completed training goals |
| --- | ---: | ---: | ---: | ---: |
| Q-learning | 1 | 1 | 26.79% | 126 |
| SARSA | 1 | 0 | 0.00% | 186 |
| DQN | 1 | 1 | 8.40% | 390 |
| PPO | 2 | 1 | 31.61% | 188 |

The previously registered gate (**at least 2/4 and at least +1 improvement over the same algorithm's untrained checkpoint**) failed for all four. Trained action traces showed waits, collisions and oscillations; this was **not** a universal all-Wait collapse. Phase 5A's engineering/method gates passed as reported, with 305 final full regression tests and audits of 40,000 training actions; those claims must not be silently reinterpreted as proof of successful navigation learning.

A **zero-learning analytical witness** showed the existing 412 affine features can express correct *initial directional choices* for the four A goals using relative-goal coordinates. It did not prove the TD learners can acquire those values or optimize the full navigation problem. The XOR capacity probe is also not proof that fixture A is unlearnable. Do not jump straight to 644 features as though that conclusion were established.

**Research question for Phase 5B:** Why do goal-bearing training episodes fail to produce a reliably goal-directed deterministic policy on simple unfamiliar start-goal pairs of the *same familiar open geometry*? Does a controlled enhancement of the existing relative-goal signal improve this outcome?

---

## 1. Scope and non-negotiable invariants

Keep **all** of the following unchanged within this Phase 5B experiment:

1. Grid: 10×10; exact robot and goal coordinates; environment starts with partial knowledge.
2. Four cardinal occupancy rays of maximum distance two, with first-wall occlusion.
3. Agent-owned occupancy/age/visit/blocked-streak memory; no hidden true-map or future-obstacle information in the learner's input.
4. Raw observation `partial-memory-409-v2`: exactly 409 `float32` values, with existing feature-index semantics.
5. Five actions with existing action IDs: Up, Down, Left, Right, Wait.
6. Rewards and timing: step −0.01, success +10, collision −5, no extra timeout penalty, 200-step external truncation.
7. The Phase 5A **A** training fixtures (8) and original A evaluation fixtures (4) remain immutable. No events or changing obstacles in A.
8. Q-learning and SARSA remain explicitly named *linear semi-gradient approximations*, not tabular algorithms. DQN/PPO remain the separately configured neural implementations.
9. No forced-movement rule, planner fallback, action masking based on hidden occupancy, learning from true shortest paths, privileged exploration labels, demonstration/imitation data, or hand-coded goal-following inserted into learned policies.
10. Keep complete train-vs-evaluation traces and exactly distinguish real completed episodes from incomplete budget prefixes. Preserve all unsuccessful checkpoints.

Do **not** run B/C/D/E obstacle/dynamic training; change obstacle generation; generate/read any new final/test set; edit the existing Phase 5A study or evidence; merge into `main`; rewrite the Streamlit UI; report this as a generalization result; or claim a winner among algorithms based on these diagnostic runs.

If a correctness defect is found in existing code, **do not bury the fix inside a feature ablation**. Write a minimal reproducing test, version the correction, explain its impact, and stop before the training comparison pending a clean registered baseline. Correctness and hyperparameter changes cannot be conflated.

---

## 2. Phase 5B-0 — zero-learning diagnosis and preflight (must come first)

Implement a **read-only** diagnostic command for existing Phase 5A checkpoints and A training/evaluation traces. Do not retrain, mutate evidence, or select a better past checkpoint. Collect at least:

- Action-ID consistency from `partial_contract.MOVES` through `env.step`, labels, action-value ordering, SARSA continuation action and SB3 inference; test all 4 cardinal rotations.
- Goal-relative vectors and actual first-action value/probability for **every** A evaluation start-goal case at 0/2k/6k/10k, with deterministic tie policy disclosed.
- One-step and two-step worked Bellman targets for nonterminal, goal, collision, and external truncation transitions; compare analytical targets to real update functions with synthetic test vectors.
- Q-learning and SARSA feature norms, weighted Q components (bias, position, goal, occupancy, age, visits, streak, elapsed time, relative goal), TD errors, update magnitudes, action-value margins, and how often argmax is Wait or an invalid directional attempt.
- DQN/PPO input scales, gradient/update finiteness, actual exploration/entropy schedules, reward-bearing replay or rollout transitions, initial/final logits or Q-margins, and any repeated reset/seed effects; do not infer causal mechanisms from aggregate action counts alone.
- Distribution of the 412 features over **training observations**, including magnitude, nonzero frequency, clipping/saturation, and the ratio of relative-goal contribution to other channels; never fit statistics on diagnostic/evaluation cases.
- At least one successful and one unsuccessful trajectory explanation per family where such examples exist, with **pre-action permitted information** and clearly labelled evaluator-only diagnostics separated.

Use consistent math and avoid confusion between the sensor-ray order (N/S/E/W) and action order (Up/Down/Left/Right/Wait). The dynamic sensor can signal walls but fixture A is open; do not blame dynamic-event composition for an A-only failure.

### Analytic baseline — not a learned agent

Reproduce the existing zero-learning relative-goal witness on A reset observations and test that its directional argmax agrees with the correct first step. Explicitly label it **capacity witness, no learning, no executed policy**. Do **not** initialize or replace RL weights with this handcrafted witness for the experimental runs.

Write a short preflight report describing which hypotheses remain plausible. If an **interface or optimization bug** is demonstrated by a failing deterministic test, halt training, issue a reproducible defect report, and request a versioned fix. Otherwise proceed to the registered A-only scaling comparison.

---

## 3. Phase 5B-1 — pre-register ONE controlled feature-scaling comparison

### Single experimental factor

Compare **two** treatments, with all other task/model settings held fixed **within each algorithm**:

| Arm | Feature transform | Dimension | Task and reward |
| --- | --- | ---: | --- |
| `B0-original` | Existing `partial-visible-affine-v1`: bias + raw 409 + 2 relative-goal coordinates | 412 | Original unchanged |
| `B1-relative-x4` | Identical 412 vector except **multiply only the last two relative-goal features by 4.0** | 412 | Original unchanged |

This is **a priori** a learning-signal/scaling hypothesis, not a proved improvement. The intervention is the *feature transformation*, not a change to the raw 409 sensor/memory observation. No new information is introduced. Preserve the original extractor implementation and checkpoint identity; implement B1 under a **new feature version/signature** (e.g. `partial-visible-relative-scale4-v1`). The neural vectorized extractor and linear numpy encoder must produce numerically matched values for the same 409 input. Add strict version-aware checkpoint loading, test feature bounds appropriately (feature magnitudes may exceed 1 even though the raw observation stays in [-1,1]), and reject incompatible checkpoints. Do not combine this with learning-rate, optimizer, reward, action, network-size or episode-horizon changes.

The choice of multiplier 4 is explicitly **predeclared**; do not browse several multipliers and report only the best. If zero-learning diagnostics already establish that this intervention would be invalid (e.g. nonfinite states), stop rather than silently substitute another transform. Do not modify stored Phase 5A outputs or treat them as a matched treatment run.

### Dataset and controls

- **Training:** Exactly the eight existing A training cases, same sampling distribution, case order rules and reset seeds across B0 and B1 within a model. Record the actual case sampling stream; if algorithms inherently consume randomness differently, preserve comparable declared distributions rather than claiming identical trajectories.
- **Original A diagnostics:** Keep the exact original four A evaluation cases for continuity with Phase 5A; these are already inspected **development cases**, not an untouched holdout.
- **Expanded A diagnostics:** Before the first training action, register **16 additional** A-only evaluation cases on the same 10×10 open geometry, with balanced action directions and 1–2-step goals, diverse positions, valid goals, no duplicate `(start,goal)` training/evaluation pairs, and no access to any final unseen maps. Save a deterministic fixture generator/manifest, seed, hashes and an audit. These are **additional development cases**, not generalization to new layouts.
- **References:** An observation-only planner, untrained checkpoints and a documented goal-direction script (script is privileged/analytic mechanics demonstration, not a learned model). Report all separately.

Do not alter or discard fixtures after inspecting performance. Reserve a fixed evaluation seed and pre-register exact checkpoint criteria.

### Training/action/compute ceilings

Use a **separately registered** Phase 5B study ID and artifact namespace, e.g. `partial-a-learning-diagnosis-v1`; *never* write into `feasibility-pilot-v1`.

- Four families: Q-learning, SARSA, DQN, PPO.
- Two arms per family: B0 and B1.
- One new primary training seed **23021**, applied independently to each arm. Not a cross-seed performance conclusion.
- **6,000 training interactions per arm maximum**; checkpoints at 0, 2,000 and 6,000. Total ceiling: **48,000** actual training actions across eight runs.
- Time ceiling: **1,800 seconds for the complete Phase 5B trial**, **300 seconds per arm**. If platform limits differ, predeclare a stricter feasible limit **before** any training; do not silently raise it.
- Evaluation ceiling: **100,000** actual assessment/control actions total; no repeated evaluation for opportunistic tuning.
- Artifact ceiling: **512 MiB** of newly generated Phase 5B records/checkpoints, with a clear failure policy before execution. If the existing trace format makes this insufficient, report a preflight blocker and propose deterministic compression instead of dropping evidence.
- Each arm: same declared algorithm hyperparameters as Phase 5A (`configs/partial_feasibility_v1.yaml`) except for the changed **feature transform**. Keep recorded neural exploration schedules; if continuation or reset semantics differ between B0/B1, reject causal interpretation and correct the protocol before training.
- Save RNG/optimizer/replay state as needed to continue a multi-block training run correctly; measure true rather than inferred interactions, and prevent evaluation from advancing the trainer RNG.
- Pair arms by family and seed. **Do not pool action samples across model families** as if they were IID replicas, and do not equate a PPO epoch with a DQN gradient update.

If any run fails, preserve the partial prefix, error and spent budget; **no hidden retry**. Training stops when the budget is spent regardless of results. No downstream dynamic experiment may start automatically.

---

## 4. Required metrics and falsifiable decision rules

For every model × arm × checkpoint (including untrained), save:

1. Original A: successes / 4, collisions, timeouts, mean/median actions, Wait fraction, longest Wait streak, revisit and oscillation counts.
2. Expanded A: successes / 16 with exact numerator and denominator; disaggregate by direction, goal distance (one/two steps), and start-position quadrant/band if balanced.
3. Training: actual interactions, complete episodes, incomplete prefixes, goal/collision/timeout counts, reward distribution, per-action counts, exploration schedule and update counts; do not pool these with deterministic evaluation.
4. Values: goal-direction action margin, chosen action/score or probabilities, first-step optimal-direction rate, action-value/entropy distribution; retain observation examples of successes and failures.
5. Numerical/compute: feature norm distribution, parameter/update norms, finite status, checkpoint hashes, training wall seconds and measured process resources.
6. Deterministic replay: verify saved rewards, observations, actions, terminal vs truncation semantics, snapshots, signatures and that inference did not modify model weights.
7. B1−B0 **within-family paired differences** on the same fixed diagnostic cases; report changes in both success and failure modes. Provide exact case-pair tables, not significance claims from one seed.

**Predeclared improvement check (development screening only):** For an individual model at the final 6,000-action checkpoint, B1 is *promising* only if:

- At least **3/4** original A successes **and** at least **one more** success than both its B1 untrained checkpoint and its matched B0 6k checkpoint; **and**
- At least **12/16** expanded A successes **and** at least **four more** successes than the matched B0 6k checkpoint; **and**
- All engineering/integrity checks pass; the improvement is not caused by bypassing observations, forcing actions, changing reward, changing dataset, or evaluating on training start-goal pairs.

These are **strict feasibility screening thresholds**, not statistical inference, accuracy guarantees, or permission to declare an algorithm winner. Do not weaken or revise them after viewing the results. A negative result is acceptable and must be reported.

**Gate names:**
- `G0_SOURCE_AND_EVIDENCE`: existing baseline/audit preserved; correct branch and stable checksums.
- `G1_ZERO_LEARNING_DIAGNOSIS`: action orders, targets, gradients, memory boundaries and analytic witness validated, no correctness blockers.
- `G2_FAIR_REGISTERED_COMPARISON`: manifest/seed/arm budgets fixed and verified before training.
- `G3_MEASURED_LEARNING`: report actual B0/B1 outcomes and screening thresholds for each family.
- `G4_AUDIT_AND_REPRODUCIBILITY`: full unit/regression and replay checks plus complete budget reconciliation.

Use terminal verdict exactly one of:
- `IMPLEMENTATION_BLOCKED`: correctness or integrity checks fail; no learning conclusion.
- `SCALING_HYPOTHESIS_SUPPORTED_IN_ONE_SEED`: at least one model clears the predeclared B1 screen (still not a full Phase 5 approval).
- `SCALING_HYPOTHESIS_NOT_SUPPORTED`: none clears the screen, even if selected metrics improve.
- `INCOMPLETE_BUDGET_OR_EXECUTION`: an attempt did not finish and cannot be treated as a completed comparison.

The verdict must state explicitly how many families passed, if any, and what remains unresolved. Never turn a failed gate into a positive conclusion by cherry-picking an earlier checkpoint.

---

## 5. Tests Codex must add before training

- Action ID/direction consistency tests (north/south/east/west and Wait), including rotation and coordinate normalization.
- Bellman-target hand-calculation tests for linear Q-learning and SARSA; goal/collision termination; truncated continuation; sampled SARSA next action; no bootstrapping after true termination.
- Observation-equivalence test: two hidden maps/events with identical sensor histories result in identical 409 observations, and identical B1 feature vectors.
- No leakage test: feature extractor and learner cannot receive true map, scenario ID, future obstacle schedule, obstacle type, privileged planner path or diagnostic snapshot.
- Parity test: B0 numpy vs neural extraction; B1 numpy vs neural extraction on zero/random/edge/corner/goal-near cases, including correct sign and feature order.
- B0 backward-compatibility check against existing 412 vector; Phase 5A checkpoint loading must remain compatible with the **old** signature, and new B1 checkpoints cannot be loaded under B0.
- Deterministic registration and duplicate-case prevention; expanded A cases cover all four first-move directions and both one- and two-step goals.
- Test paired evaluation on identical case IDs and no checkpoint/optimizer/RNG contamination from evaluation.
- Full repository regression suite and the pre-existing partial benchmark test suite. Any regression must be fixed or reported before learning runs.
- Failure injection: interrupted run, corrupt checkpoint/manifest, mismatched feature version, stale hash, incomplete episode prefix, wrong interaction count, and no double spending of the budget.

---

## 6. Outputs to commit to the new feature branch

Prefer these new paths, adapting to repository conventions without overwriting existing materials:

```text
CODEX_PHASE5B_BASIC_NAVIGATION_DIAGNOSIS.md

autonomous-robot-navigation-rl/
├── configs/partial_phase5b_v1.yaml
├── docs/partial_phase5b_protocol.md
├── docs/partial_phase5b_results.md
├── docs/partial_phase5b_diagnosis.md
├── src/robot_navigation/partial_phase5b.py
├── src/robot_navigation/partial_phase5b_diagnostics.py
├── src/robot_navigation/partial_features_scaled.py   # Or equivalent versioned module
├── developer/tests/test_partial_phase5b.py
└── artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/
    ├── manifest.json
    ├── gate-results.json
    ├── training-budget-ledger.jsonl
    ├── expanded-A-fixtures.json
    ├── learning-curves.csv
    ├── paired-results.csv
    ├── training-and-evaluation-evidence/
    ├── checkpoint-hashes.json
    ├── replay-audit.json
    └── final-verdict.json
```

Save **actual** artifacts, not placeholders. A partially executed study should produce an explicit partial/failure report instead of fabricated tables. Link original Phase 5A reports as read-only evidence. Include commit SHA, branch, source/config/data hashes, dependency versions, exact commands, environment and resource details, and any deviations/preflight blockers. Use relative Markdown links that resolve inside the repository.

### Required report outline

1. Executive verdict and scope limits.
2. Original Phase 5A facts versus new Phase 5B measurements (clearly separated).
3. Zero-learning audit: action alignment, TD targets, feature magnitude/goal conditioning.
4. Registered B0 and B1 design, versions, budgets and case manifests.
5. Per-family and per-checkpoint original A and expanded A results (all failures retained).
6. Wait/collision/loop examples using permitted observations.
7. Paired per-family treatment differences and non-causal alternatives.
8. Runtime/resource accounting and independent audit outcomes.
9. Gate decision and **one** narrowly justified recommendation: remain with A for another isolated hypothesis, or propose independent seed replication **only after new authorization**. No silent progression to B/C/D/E.

---

## 7. Work style, safety, and stopping rules

**Implement deliberately, not optimistically.** Before launching any training, complete code review, mechanical tests, the zero-learning audit, and a machine-readable frozen protocol. Preserve existing files and do not erase previous results or unsuccessful attempts. Do not mark a green gate simply because tests execute without exceptions.

At the end, report exactly:

- Branch and commit SHA; whether changes were pushed or remain local. **Do not claim a push happened unless it did.**
- Files added/changed and confirmation that existing Phase 5A/historical files were not modified.
- Test counts, failures/skips and whether source-boundary / information-leakage tests passed.
- Actual number of training actions and evaluations, by family and arm; every aborted/unfinished prefix.
- Original A and expanded A successes at each checkpoint, including untrained.
- Whether B1 beats B0 under the **predeclared thresholds**, without claiming generalization or causal proof across seeds.
- Evidence locations for each raw trace, checkpoint and replay audit.
- One concise proposed next step requiring review **before** any further training.

**Do not:** rewrite the original report, force all models to pass, tune the feature multiplier after seeing outcomes, change reward as a hidden confounder, replace RL decisions with the analytic goal-direction witness, use held-out final layouts, merge into `main`, or start full Phase 5/Phase 6. If the user has not explicitly authorized a GitHub push in this Codex session, leave the new branch local and give the exact safe commit/push instructions.

---

## Suggested instruction to Codex

> Read `CODEX_PHASE5B_BASIC_NAVIGATION_DIAGNOSIS.md`, inspect the actual Phase 5A branch evidence, and implement its zero-learning audit and one-factor A-only feature-scaling experiment. Register and validate everything before training. Preserve historical evidence. Run only the bounded Phase 5B actions permitted here, keep failed checkpoints, and report actual results with replay-audited files. Do not merge into `main` or start obstacle/dynamic/full Phase 5 work.
