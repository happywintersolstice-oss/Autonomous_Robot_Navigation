# Codex Engineering Brief — Phase 5D: Goal Conditioning and Credit Assignment Diagnosis

**Project:** Comparative Evaluation of Reinforcement Learning Strategies for Autonomous Navigation in Unknown and Dynamic Indoor Grid Environments  
**Repository:** `happywintersolstice-oss/Autonomous_Robot_Navigation`  
**Input branch:** `feature/phase5c-directional-coverage-v1`  
**Verified Phase 5C commit:** `c1505cc1dab0623b9c7eb86b7bfae9ac3216fa4d`  
**New working branch (suggested):** `feature/phase5d-goal-credit-diagnosis-v1`  
**Study ID:** `phase5d-goal-credit-diagnosis-v1`  
**Mode:** **READ-ONLY HISTORICAL ANALYSIS. ZERO NEW TRAINING. ZERO NEW POLICY-EXECUTED ENVIRONMENT EPISODES.**

## 0. Auftrag / instruction to Codex

Read this entire brief, inspect the **current repository at the specified Phase 5C commit**, and implement one rigorous, reproducible **Phase 5D analysis-only** study. You must **not** assume that this brief describes the current code exhaustively: inspect existing paths and data schemas before authoring the analysis. Record any discrepancy, and prefer stopping with a precise blocker rather than guessing.

The task is to explain, as far as existing evidence permits, why Q-learning, SARSA, DQN and PPO obtain rewards during simple open-grid training but fail to make reliable **goal-conditioned greedy decisions** on different open-grid start/goal states. The especially informative case is **DQN B1**, where training contains Right-goal signals but final evaluation shows **0/4 Right-goal successes**.

The deliverable is **code + frozen analysis protocol + verifiable derived artifacts + an honest Markdown report**, committed to a *new feature branch if the user has expressly authorized Codex to commit*. Do not merge into `main`; do not modify historical results or silently initiate full Phase 5.

## 1. Verified scientific starting point (do not rewrite or reframe)

All statements below describe **existing historical studies**, not new Phase 5D measurements.

- **Phase 5A (`feasibility-pilot-v1`, seed 23001):** Four learners each received 10,000 training actions and all failed the preregistered four-case Fixture A improvement gate. Training nevertheless included successful goals. Evidence: `docs/partial_feasibility_results.md` and `docs/partial_feasibility_diagnosis_v1.md`.
- **Phase 5B (`phase5b-a-only-v1`, seed 23021):** Original **B0** 412 features versus **B1** with *only* the final two relative-goal features scaled by 4; otherwise comparable within each family. Exactly 6,000 training actions per arm, eight arms, 48,000 total. The final expanded development successes out of 16 were: Q-learning **6 B0 / 7 B1**, SARSA **5 / 7**, DQN **2 / 10**, PPO **0 / 5**. No family passed the *complete* preregistered screen; `SCALING_HYPOTHESIS_NOT_SUPPORTED` does **not** mean zero measured improvement. See `docs/partial_phase5b_results.md`.
- **Phase 5C (`phase5c-directional-coverage-v1`):** Read-only audit of 48,000 *unique historical* training actions, **24 overlapping checkpoint prefixes** (0/2k/6k × eight arms), 41,705 recorded historical assessment actions in 480 episodes, and 960 score-only probes. **Zero new training actions and zero new policy/environment assessment actions.** The proposed balancing-sampler gate failed: `DIAGNOSIS_ONLY_SAMPLING_NOT_JUSTIFIED`. See `docs/partial_phase5c_results.md` and `docs/partial_phase5c_diagnosis.md`.
- For **DQN B1 at 6k**, final expanded development outcomes: **Up 3/4, Down 4/4, Left 3/4, Right 0/4**. Historical evidence: 104 Right-goal starts among 506 resets, 210 training decision states with goal exactly one cell Right, 88 Right actions from those states reaching goal, plus 88 Right-approach goal arrivals aggregated across reset cases (these counts have **different denominators**). The two Right-start training cases produced 87 completed goals, also a **different denominator**. All four final expanded Right-goal evaluation episodes failed, and their initial Right-action margins were negative.
- **PPO B1** had 118 one-cell-Right decision states and 20 Right goal arrivals during training, but zero contemporaneous greedy Right preferences in that near-goal subset. Exploratory/stochastic goal attainment must never be conflated with learned deterministic policy success.
- Phase 5C's final full suite reported **399 passed**, focused suite **33 passed**; these are historical reported receipts, not new Phase 5D test results. Phase 5C preserved historical source/evidence; do not modify it.

**Do not turn these data into a claim that credit assignment, representation, or optimization is already proven to be the cause.** The purpose of Phase 5D is to test carefully defined competing hypotheses using inspectable evidence.

## 2. Scope and hard prohibitions

### Allowed

- Read Phase 5A/5B/5C **training traces**, checkpoint metadata, frozen model checkpoints, existing evaluation outcomes, optimizer records, feature transforms, current model implementation and historical synthetic probes.
- Pure forward inference on **already saved checkpoints**, only on a deterministic, **preregistered** set of **legal 409-dimensional observations**; no `env.step()` or policy-run episode. Reusing exact Phase 5C probe observations is preferred.
- Analytical, side-effect-free calculations: value decomposition for linear agents; frozen-snapshot Q/TD residual diagnostics when mathematically justified; PPO policy probabilities/entropy and recorded value/advantage diagnostics where available; matched state/goal probes; direction-stratified descriptive statistics; versioned reproducibility tests.
- New analysis-only code, minimal new tests and **new** artifacts under a new study namespace. Lightweight synthetic unit fixtures are allowed if explicitly labelled *mechanics tests*, not scientific training or policy evaluation.

### Forbidden without a new user-approved brief

- Any new RL training actions, gradient updates, replay optimization, new reward/learning rate/entropy/discount/architecture/feature change, imitation learning, dataset resampling for retraining, policy initialization by a handcrafted rule or action override.
- Any new policy-driven `env.step`, rollout, simulated episode, start/goal assessment or full evaluation. Inference on a sealed list of observations **does not** count as an executed navigation episode and may not be reported as success rate.
- New final/test set generation or access; inspecting/fitting against historical **held-out final** cases; touching `main`; retroactive changes to Phase 5A/5B/5C configs, checkpoints, evidence, registered protocols or published results.
- Cherry-picking the best checkpoint, reclassifying exploratory actions as greedy, calling goal-direction labels optimal dynamic actions, or claiming statistical significance from one historical seed.
- Blind duplication of huge archived snapshots into the new commit. Preserve by Git commit/tree IDs and checksums; write only the smallest necessary source snapshot.

## 3. Verify repository state and inventory before analysis

1. Verify new branch forked from `c1505cc1...` (or record a stop if upstream has advanced; do not silently mix revisions). Maintain original dirty worktrees/index unmodified. New work should use a separate checkout/worktree where practical.
2. Inspect the actual contracts, observation schema and saved evidence schema in:
   - `src/robot_navigation/partial_contract.py`, `partial_observation.py`, `partial_features.py`, `partial_features_scaled.py`, `partial_agents.py`, `partial_neural.py`, `partial_training.py`.
   - `src/robot_navigation/partial_phase5b.py`, `partial_phase5b_diagnostics.py`, `partial_phase5c.py`, `partial_phase5c_coverage.py`, `partial_phase5c_fixtures.py`, `partial_pilot_diagnostics.py`.
   - `configs/partial_phase5b_v1.yaml`, `configs/partial_phase5c_v1.yaml`; all three prior protocol/result documents.
   - `artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/` and `.../phase5c-directional-coverage-v1/`, particularly `training-and-evaluation-evidence`, `counterfactual-scores.csv`, `historical-decision-quality.csv`, `right-goal-cases.csv`, `coverage-audit.json`, source/evidence manifests and replay audits.
3. Audit actual presence, record fields, timestamp semantics and completeness for the linear update logs, DQN optimizer updates/replay/checkpoints, PPO probabilities, recorded training decisions and observation transitions. **Do not assume that historic per-transition TD targets, log-probabilities or advantages were recorded.** Create an evidence availability matrix with `available / not recorded / reconstructable from verified contemporaneous state / snapshot-only` statuses.
4. Inventory and hash **every input actually read**; record branch SHA, checked-out tree SHA, Python and pinned NumPy/PyTorch/SB3/Gymnasium versions, feature signature and original files' hashes. Do not create an enormous nested copy of the repository.
5. Establish baseline regression and Phase 5C read-only verification before any analytical findings. If critical preservation/information-boundary checks fail, stop with `IMPLEMENTATION_BLOCKED`.

## 4. Research questions and competing hypotheses

**RQ1 — Is the goal-bearing signal reflected in the saved greedy policy?** Compare success-bearing historical states and fixed development reset states with the **same goal-direction definition**. For each checkpoint, measure how often a goal-reducing action is top-ranked; whether Right is disfavored at Right-goal start states; and how preferences depend on position and memory-history.

**RQ2 — Is there a mismatch between successful training and deterministic evaluation due to different states?** Compare *observable-state distributions*, not just directional counts. Control descriptively for (position, goal, Manhattan distance, elapsed step, known occupancy, visit count, observation age and memory history). Report exact matches when available, and near matches with explicit distance criteria and support sizes. Do not claim causal state-distribution shift from non-identical trajectories.

**RQ3 — How do goal rewards relate to learned action values or probabilities?** For each family, evaluate what can be measured from saved evidence. Separate observed historical updates from *counterfactual snapshot calculations*. In particular, a DQN TD target computed using a 6k checkpoint on a 1k historical transition is **not** the target actually used at that historical time.

**RQ4 — Do memory/absolute-position components overwhelm relative-goal sensitivity?** Compare legal matched observations that hold agent-visible history constant while varying only *legally feasible* start-goal relationships. Analyze linear feature contribution exactly and neural frozen-policy sensitivity descriptively. Avoid treating synthetic intervention effects as training outcomes.

**RQ5 — What is the narrowest defensible next experimental hypothesis?** If there is evidence for a particular deficit, articulate **one** separately registered future training experiment (proposal only). Otherwise report unresolved confounding and ask for improved instrumentation rather than proposing arbitrary bigger budgets.

## 5. Pre-register analysis and cohort construction BEFORE generating outcomes

Create `docs/partial_phase5d_protocol.md` and a hash-locked `analysis-plan.json`. Explicitly register:

- Study identifiers, allowed input files and SHA256 hashes, deterministic sort order, grouping/bins, algorithms/arms/checkpoints, analysis formulas and missingness policy.
- **All eight Phase 5B arms** (`q-learning`, `sarsa`, `dqn`, `ppo` × `B0-original`, `B1-relative-x4`) at checkpoints **0, 2000, 6000**. Do not select best checkpoint after examining the results.
- Full historical training action stream once per arm at 6k; earlier 2k prefixes are **nested and not independent**. Report checkpoint trends without summing prefixes as independent data.
- The same **4 original + 16 expanded Phase 5B** diagnostic pairs (already inspected; development only), and every case rather than selected failures alone.
- The existing Phase 5C 40 legal probe observations (and their 960 checkpoint/arm combinations); prefer zero novel probes. Any new probe family must be enumerated and hashed before score generation, explain why existing probes are insufficient and ensure legal sensor/memory histories.
- Matched train/evaluation state groups: precise equality on full float32 observation (hash), then optional coarse bins clearly marked *approximate matching*. Distinguish episode-start direction, current relative-goal direction, action attempted, reward-bearing arrival and chosen greedy action.
- Primary descriptive outcomes: correct goal-reducing argmax rate, Right-vs-non-Right score margin (Q-value or probability within family only), direction-stratified failure counts, observed goal-transition count, state-overlap frequency and unsupported-bin fraction.
- Secondary: Q target/residual estimates where supported, feature contributions, PPO value/probability changes, D4 symmetry/translation probes, delta-to-goal versus memory sensitivity, prior checkpoint changes. Treat multiple comparisons as exploratory.
- All denominators, inclusive/exclusive data windows, ties (argmax action ID), binary labels, per-case IDs, missing/null logic, and a rule to avoid any retrospective screening threshold alteration.

No new training branch/gate or conditional experiment is authorized by this read-only brief. Report all measured outcomes, even if they contradict the hypothesis.

## 6. Correct definitions and analytical treatment

- Action IDs: **0 Up, 1 Down, 2 Left, 3 Right, 4 Wait**. Sensor ray order **North, South, East, West** is **not** the action-ID order; test mapping independently.
- 10×10 partial-observation grid; **409-float32 original observation** includes normalized current/goal coordinates, four rays, discovered occupancy, observation age, visits, blocked streaks and time. Feature arms: 412-value B0, 412-value B1 with **only final two relative-goal features ×4**. No true obstacle interior, future blockage schedule, object type or hidden lifetime.
- For one-axis open-grid goal cases, an action **reduces Manhattan distance** if moving one legal cardinal cell decreases `abs(gr-r)+abs(gc-c)`. For obstacle scenarios this is **not** a global optimality label; Phase 5D is restricted to **Fixture A only** for goal-direction interpretations.
- Greedy selection uses the policy's original deterministic action semantics; ties are counted and retained, including deterministic first-index tie resolution. Do **not** use exploratory training action as an estimate of argmax.
- **Q-learning / SARSA:** Record `Q(s,a)` and action margins. For linear policies, decompose `w_a · phi(s)` by existing groups and reconstruct the exact sum to tolerance. If logged contemporaneous TD errors/targets are available, verify against the actual update formula; otherwise mark target/residual as **checkpoint-based hypothetical** rather than actual historic update. For SARSA, the behavior next-action used in a recorded transition must be used when reconstructable; never substitute `max Q` and call it SARSA.
- **DQN:** Score frozen online network at 0/2k/6k and, where saved/available, frozen target network. A one-step snapshot diagnostic may use `r + gamma * (1-terminated) * max_a Q_target(s',a)` under the registered DQN algorithm; verify actual implementation's treatment of truncation and replay preprocessing rather than inferring conventions. **Snapshot-derived targets are not historical per-step optimizer targets** without verified matching historical network state. Report terminal goal action's observed `r=9.99` separately from whether `Q(s,Right)` ranks highest at the final checkpoint.
- **PPO:** No Q(s,a) exists in this architecture. Analyze **policy action probabilities**, log-odds/margins, entropy and **state-value estimate** at frozen snapshots. Reconstruct historical advantages/returns only if rollout content and contemporaneous value/bootstrap/log-prob information are sufficient; if only last rollout is saved, say so. Do not fabricate historical per-action advantages or compare raw PPO probabilities against Q magnitudes.
- Collisions, timeouts and incomplete prefixes are distinct. Do not relabel incomplete training prefixes as timeouts or include them in completed-episode success rates. Attribute a positive terminal reward to the exact goal transition, not to any bootstrapped positive value.
- State hashes are based on exact float32 409-value observations and are not equivalent to position-goal hashes, case IDs or full hidden-world states. Episode reset counts are not independent state-action decisions.
- A goal-conditioned policy score difference on matched probes is descriptive **sensitivity**, not proof of causal credit assignment or a policy's dynamic optimality.

## 7. Implement the analysis in bounded submodules

Prefer new modules, integrating existing reading/score utilities without editing frozen code:

- `src/robot_navigation/partial_phase5d.py`: `prepare`, `analyze`, `verify`, `report` CLI workflow, read-only source checks, exclusive new namespace, deterministic analysis plan seal, no retries or overwrites. No execution path invokes `policy.model.learn`, linear `.update`, `.train`, optimizer `.step`, or model-driven `env.step`.
- `src/robot_navigation/partial_phase5d_credit.py`: algorithm-specific value/credit calculations; contemporaneous vs snapshot provenance; missingness / uncertainty; unit-tested formulas. Protect model weights/RNG/optimizer replay buffers against mutation during inference.
- `src/robot_navigation/partial_phase5d_states.py`: legal matched-state sampling and alignment to Phase 5C probes; no fabricated raw sensor/memory state. Strict match vs approximate match labelled separately.
- `src/robot_navigation/partial_phase5d_report.py`: derive all tables and plots from evidence, not copy/pasted literal results.
- `configs/partial_phase5d_v1.yaml`, `docs/partial_phase5d_protocol.md`, `docs/partial_phase5d_results.md`, `docs/partial_phase5d_diagnosis.md`, `developer/tests/test_partial_phase5d.py`.

These file names are suggested; inspect project conventions, and explain deviations. Keep all new evidence within `artifacts/partial-dynamic-navigation-v1/phase5d-goal-credit-diagnosis-v1/`. Existing historical inputs are read only. The brief itself can be saved at repository root under its exact filename.

## 8. Required measurements and outputs

### A. Historical goal/choice alignment

For every arm × checkpoint and each of four goal directions:

- Episode starts, distinct exact observations and position-goal pairs; decisions with current axis-aligned goal one cell away; attempts at reducing/away/Wait; goal arrivals; deterministic top-ranked action count (only when pre-action decision scores were logged), tie rates and supported denominators.
- Separate **training exploration behavior** from saved-checkpoint deterministic assessment. For DQN B1, reconcile Phase 5C's Right-start 104, near-Right 210, right-arrival 88, final expanded Right 0/4 as sanity checks, not mutually interchangeable proportions.
- At reset of all 20 Phase 5B development cases, report the saved checkpoint's five-score vector, goal action, argmax, margin, per-case historical outcome, its first action and result. Do not execute a new episode.

### B. Credit diagnostics (algorithm-specific)

- For linear methods: trace what actual positive goal transitions and update logs reveal; report per-group contributions for Right vs other actions on **all predeclared legal probes**; verify contributions add to score for every case. Show the actual alpha/normalization/discount mechanics, checking concrete trace consistency where fields permit it.
- For DQN: compare saved online/target scores on selected preregistered historical goal-transition observations and paired unseen-to-training state probes. Describe checkpoint-only estimated TD residuals and the limitations of using a later network on an earlier transition. Record rank changes between 0 → 2k → 6k and any preference inversions. Do not infer that a higher snapshot Q on a reward transition proves correct reward propagation throughout the training history.
- For PPO: report action probability and entropy on every legal probe and training-observation cohort; if actual historical advantage/value records exist, audit their provenance and computations. Otherwise explicitly report that full time-varying GAE/reward-credit path **cannot be reconstructed**.
- Stratify meaningful subset analyses by Manhattan distance (1/2), goal direction, translated position/quadrant, visits, known cells and memory age, indicating sample size. If matched support is insufficient for a direction, say **unresolved due to support**, not "no effect".

### C. Legal counterfactual comparison

Use fixed Phase 5C observations (or additionally preregistered legal observations) to inspect:

- Same position and same sensory/memory history with *different visible goal coordinates*, provided goal locations are legal and the history remains possible under the new goal. Explicitly validate these constraints.
- D4 rotated/mirrored legal histories with corresponding action permutation; score equivariance/descriptive disagreement, with tie-awareness.
- Memory-history variations only when each is obtainable from a legal action/sensor sequence; no arbitrary edits to occupancy/age grids that produce impossible robot knowledge.

These are **zero-action forward score probes**. They are **not** agent navigation results or unseen independent samples.

### D. Case-level explanation

Every claimed example must show: model/arm/checkpoint, case ID, observation before action, goal relation, ray order, known-map count, chosen greedy action, all five scores (Q or probabilities), goal-action margin, actual historical outcome and trace ref. Report **all four** DQN B1 failed Right cases plus at least the other 12 expanded cases via complete tables, not just anecdotes. Other families must be covered systematically, even if DQN is the focal example.

### E. Evidence availability and methodology

Create: `evidence-availability.json`, `credit-metrics.csv`, `matched-states.csv`, `case-decisions.csv`, `direction-coverage.csv`, `diagnostic-summary.json`, `validation-gates.json`, `output-hashes.json`, and a source/evidence manifest. Include explicit `historical`, `snapshot`, `synthetic-score-only`, `not-observed` provenance fields. If a document is missing, do **not** quietly replace it with a derived fiction.

## 9. Validation gates (fail closed)

- **G0 provenance/preservation:** Head/tree/input hashes documented; all Phase 5A/B/C artifacts byte-identical; no final/test reads; no `main` changes; no giant duplicate snapshot.
- **G1 source semantics:** Same true action ordering, reward/terminal/truncation and feature signatures; no hidden map/future event; exact model loading and checked observation dimension; no unexplained checkpoint mismatch.
- **G2 accounting:** Match Phase 5B counts per unique arm (6,000 actions), sum 48,000 unique actions; overlapping 2k prefixes not double-counted; reconcile Phase 5C 41,705 assessment actions and 960 score calls as historical figures, not fresh actions. Missing log records must be surfaced, not fabricated.
- **G3 numerical:** All values finite; linear feature attribution sums to Q; valid PPO probability normalization/entropy; expected TD calculations match independent worked fixtures; terminal goals and external truncation handled under the actual code; SARSA next action exact when reconstructed; saved checkpoint weights, optimizer and RNG unchanged by inference.
- **G4 scientific interpretation:** Cohorts and probes frozen before scoring; no output-selective sample changes; correct denominators; missingness/provenance; no claim of causal mechanism from association; no between-family raw score comparison.
- **G5 final:** Run focused and full existing regressions; independently verify data and hashes; regenerate all tables from raw derived records. If any gate fails, use `IMPLEMENTATION_BLOCKED`; do not claim complete analysis.

Include tests that deliberately catch wrong **sensor ray vs action-ID** mapping, mislabelling a `6k` snapshot target as historic `1k` target, PPO Q-value fabrication, missing PPO rollout history, duplicate-prefix summation, corrupted checkpoint hash, goal swap with invalid history, model mutation during inference and accessing final/test files. Keep test-only synthetic updates separate from the zero-training scientific analysis and label their counts explicitly.

## 10. Hard resource ceilings and operations

- New Phase 5D scientific **training actions: exactly 0**; **new policy-executed environment assessment actions: exactly 0**. Never reserve a training budget or initialize a live trainer.
- New forward-only policy score calls: **maximum 5,000** across all checkpoints/arms, counting all duplicate calls, including legal-probe and historical-observation scoring. Prefer reuse of existing Phase 5C scores to keep compute minimal. The 960 pre-existing calls are **historical**, not counted as newly executed; independently verified reuse is labelled.
- Analysis wall clock cap: **1,200 seconds** excluding separate baseline/postrun test processes, which must be separately timed. Source/input manifest and output bytes cap: **128 MiB** for newly produced artifacts. A proposed extension needs separate authorization; do not truncate required evidence silently.
- Lock before processing and write protocol/source hashes before computing results. Refuse existing namespace, crash-then-retry without authorization or overwriting files. Predeclared deterministic sampling order; all random operations seeded and declared. If the evidence cannot be safely processed under ceilings, stop and explain.
- Implement a static or runtime guard ensuring no model `learn`, trainer `train`, policy `update`, optimizer `step`, `env.step`, final split read or network request appears in the scientific analysis path. Unit tests may instantiate synthetic objects or run bounded mechanical checks **without** constituting new learning evidence.
- Compressed archives may be read in streaming mode. Limit stored per-decision data to what is necessary; hash and reference the existing trace rather than creating another 10,000-file artifact tree.

## 11. Decision rule and allowed final verdicts

This is **a diagnostic study, not a pass/fail performance contest**. Use explicit gates:

1. `IMPLEMENTATION_BLOCKED`: provenance, correctness, input access or replay/integrity failures. Stop and document precisely.
2. `DIAGNOSTIC_COMPLETED_INSUFFICIENT_HISTORICAL_SIGNAL`: gates pass but historical data cannot distinguish the hypotheses (for example, absent contemporaneous TD targets or PPO GAE, or inadequate matched-state support). Document missing fields and propose precise future instrumentation; no training.
3. `DIAGNOSTIC_COMPLETED_NARROW_HYPOTHESIS_IDENTIFIED`: gates pass and a clearly evidenced **descriptive** pattern suggests one specific causal test (e.g., goal sensitivity depends systematically on absolute position/memory despite similar visible goal relation). Define a **proposal only** with an isolated factor, matched control, budget and independent seed plan; do not execute it.

Do **not** label the third outcome as established causality, learning success or permission to advance to B/C/D/E. Be candid when findings are mixed. A study may report a plausible hypothesis yet still be insufficient to select one next intervention.

## 12. Final deliverable and acceptance checklist

Provide the following in the final Codex response and in `docs/partial_phase5d_results.md`:

- Exact Git commit and new branch; existing historical commit hashes; preservation statement.
- Protocol and evidence-source manifest links; all outputs and tests; machine-readable verdict; explanation of where the data are and how to verify them.
- A clearly labelled **Phase 5D read-only findings** table with denominators by family and arm, 0/2k/6k checkpoints, exact support for Right-goal issues, score/goal alignment, credit audit status and confidence limitations.
- DQN B1 full Right-case analysis and comparison with Down/Up/Left plus Q-learning, SARSA, PPO; not a DQN-only brief.
- Distinguish empirically recorded updates, checkpoint score calculations and analytically constructed legal probes. Explain which credit assignment claims were **not** supported or are not reconstructable.
- Test result counts for new runs (not historical counts), clean source-boundary results, audit verification and exact resource usage; zero training/env actions explicitly verified.
- One narrow recommended follow-up proposal, **or** a recommendation to instrument missing quantities before any further learning. Neither implies automatic authorization.

**Success means producing a trustworthy, reproducible diagnosis without manufacturing a performance improvement.** Keep the Phase 5B/5C evidence intact, do not merge to `main`, and stop before any new training, obstacles, dynamic benchmark redesign or UI work.

---

## Copy/paste instruction for Codex

> Read `CODEX_PHASE5D_GOAL_CONDITIONING_CREDIT_ASSIGNMENT_DIAGNOSIS.md` in full. Start from `feature/phase5c-directional-coverage-v1` at `c1505cc1dab0623b9c7eb86b7bfae9ac3216fa4d`, create an isolated Phase 5D feature branch, inspect and hash all Phase 5A/B/C input evidence, and perform the **read-only** goal-conditioning/credit-assignment diagnosis. Implement the analysis protocol, diagnostics, exact tests and verifiable reports. **Do not train, run new evaluation episodes, touch final/test sets, alter rewards/features/optimizers, overwrite historic artifacts or merge to main.** Report what is demonstrated and what remains unknown, then stop.
