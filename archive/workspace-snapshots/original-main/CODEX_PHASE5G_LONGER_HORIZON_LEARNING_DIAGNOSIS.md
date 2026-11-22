# CODEX — Phase 5G: Longer-Horizon Instrumented Learning and Goal-Directedness Diagnosis

**Project:** Autonomous Robot Navigation with Reinforcement Learning  
**Repository:** `happywintersolstice-oss/Autonomous_Robot_Navigation`  
**Required parent branch:** `feature/phase5f-instrumented-learning-pilot-v1`  
**Required parent HEAD:** `6fb6aac124067b00016f5514ebbc66ef777dea33`  
**Create branch:** `feature/phase5g-longer-horizon-diagnosis-v1`  
**Artifact namespace:** `artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/`  
**Authorization:** One bounded, instrumented Fixture A learning study for **DQN and PPO only**, including a *fixed, previously defined development evaluation panel*. No parameter search, final/unseen testing, dynamic-map training, full Phase 5, or Phase 6.

## 1. Background and the one research question

- Phase 5A: goals occurred in training but reliable deterministic basic navigation was not established.
- Phase 5B: B1 relative-goal ×4 improved DQN to 10/16 on an expanded fixed open-grid development panel, but its Right-goal outcomes were 0/4; no model met the full registered learning screen.
- Phase 5C: Right exposure and goal arrivals existed; balanced episode-start sampling did not pass its preregistered justification gate.
- Phase 5D: the historical DQN batch/target path and PPO advantage/minibatch histories were not fully recorded.
- Phase 5E: passive live instrumentation passed eight synthetic on/off equivalence comparisons and independent arithmetic checks, within the pinned single-CPU/single-environment library contract.
- **Phase 5F (committed parent):** exactly 500 actual training actions for each B1 model, 100 DQN optimizer steps and 50 PPO optimizer steps; 8 DQN and 6 PPO goal-reaching training events; 73 DQN goal-reward sample draws (7 distinct goal experiences) and 30 PPO goal-row minibatch draws (6 distinct experiences). All mandatory learning records and math checks passed. **No navigation evaluation was performed.**

**Phase 5G question:** During a longer, fixed and instrumented learning run, do reward-bearing transitions continue to reach the *actual optimizer* and do the policies' goal-directed first actions and fixed development navigation outcomes improve, deteriorate or remain inconsistent across checkpoints?

This is a diagnostic, not a claim that either algorithm is better. A few training goals or replay draws are not an out-of-sample navigation success rate. Do not diagnose “bad credit assignment” from an association alone.

## 2. Mandatory preflight and source integrity

1. Start from the exact parent commit and its tree. Stop on mismatch rather than silently rebasing or changing the source. Create the new feature branch without touching `main`, other feature branches, older studies, or the user's pre-existing dirty worktree.
2. Read the committed `docs/partial_phase5f_results.md`, `docs/partial_phase5f_diagnosis.md`, `docs/partial_phase5f_protocol.md`, `docs/partial_phase5e_results.md`, `configs/partial_phase5e_library_v1.json`, and the actual observer/trace/verification code. Confirm parent Phase 5F read-only verification; do **not** rerun its one-shot entry point or old validation experiments.
3. Inspect the exact Phase 5B Fixture A **training cases** and pre-existing **original four + expanded sixteen development cases** and their hashes. Keep those three sets distinct. Never load a reserved, held-out, dynamic final, “unseen”, or secret-evaluator set.
4. Confirm the pinned installed source and versions, CPU single environment, 409 observations, 412 B1 features and five action IDs. If an observer/library mismatch exists, fail closed. Do not change library versions, code paths, rewards, sensors, memory, model class, NN architecture, or task definition.
5. Before any training or development evaluation, create a source-and-data manifest with exact parent SHA, source hashes, feature/config signature, training IDs, **previously existing** development IDs, library hashes, RNG protocol, evaluation mode, caps and report schemas. Record original file inventories and verify all old artifacts after completion.
6. Record if the working tree already contains unrelated modifications. Do not delete, overwrite, stage or commit them.

## 3. Predeclared scope and hard budgets

| Property | Binding instruction |
|---|---|
| Models | DQN and PPO, one independent run each; **no Q-learning/SARSA training** |
| Training environment | Existing open-grid Fixture A **training** scenarios only, original uniform case sampler |
| Feature arm | `B1-relative-x4` / `partial-visible-relative-scale4-v1` |
| Training seed | **23121** independently initialized per model |
| Training horizon | **2,000 actual environment actions per model**, 4,000 total hard ceiling |
| Checkpoints for measurement | **0, 500, 1,000, 2,000 actions**; require clean exact-step snapshots, no extra interactions |
| Hyperparameters | Exactly the historical Phase 5B B1 settings: gamma=.99, learning rate=.0003, `[32,32]` network; DQN/PPO details confirmed from checked code, no guessed overrides |
| DQN updates | Existing learning_starts / batch / train_freq / target sync / optimizer semantics; actual count must be measured |
| PPO updates | Existing rollout n_steps / epochs / batch configuration; ensure exact checkpoint boundaries are compatible with rollouts |
| Reward, collisions, Wait, timeout | Unmodified historical environment rules |
| Instrumentation | Existing Phase 5E observer enabled for both research runs; default remains disabled elsewhere |
| Development evaluation | Existing 4+16 fixed Fixture A panels only; deterministic inference at 0/500/1000/2000. **Not** used for training or choosing hyperparameters |
| Development evaluation episode cap | 200 steps per case; maximum **32,000** evaluation `env.step` calls across both models and all four checkpoints (2×4×20×200) |
| Research training action ceiling | **4,000 cumulative**, separate from assessment and legacy regressions |
| New synthetic validation | Default **0 environment actions / 0 optimizer operations**; audit existing traces offline. If engineering tests truly require synthetic operations, stop and request separate authorization rather than hiding them under the learning budget |
| Final/unseen/held-out | **0 accesses and 0 evaluations** |
| New learning modifications | No feature change, reward shaping, sampler rebalancing, alternate optimizer, curriculum, fallback planner, action masking or hyperparameter tuning |
| Runtime hard stop | **1,800 seconds per model** and **3,600 seconds combined training**; separate accounting for assessments and unchanged legacy regressions |
| Artifact cap | **512 MiB** for this new study namespace; reserve conservatively before emitting required tensor logs |

**Important schedule caveat:** DQN's exploration schedule may depend on the supplied `total_timesteps` (Phase 5F used 500, this phase uses 2,000). Therefore the 500-action checkpoint in a new 2,000-action run is **not necessarily an exact continuation or controlled equivalent** of the old Phase 5F 500-action run. PPO optimizer states also depend on its run history. Register the schedule actually used; do not combine F and G results into one trajectory or claim a causal 500-to-2,000 improvement based on cross-study snapshots.

**No replay of Phase 5F as a warm-start:** Start fresh under the declared 2,000-step horizon. Do not load the F checkpoint unless a separately approved, fully reproducible RNG/optimizer/buffer continuation protocol exists. The current authorized design uses fresh initialization at the registered seed.

If any checkpoint cannot be captured exactly without additional environment actions, flag the issue at registration and stop; never exceed caps just to complete a rollout. If the instrumentation cannot finish inside the artifact cap, stop before starting rather than drop required live fields.

## 4. Seal the protocol *before* training or evaluation

Write `docs/partial_phase5g_protocol.md`, `configs/partial_phase5g_v1.yaml` and a hashed sealed manifest. Freeze:

- Exact seed, training cases and sampling, 2,000-step per-model training horizon, per-model training order (DQN then PPO), four measurement checkpoints and fixed development case IDs/order.
- At checkpoint zero, evaluate the initialized policy under its actual deterministic action rule. All checkpoint evaluations must be **pure inference**, without updates, exploration, extra random draws in training or altered training RNG state; use checkpoint copies/subprocesses and verify training-stream isolation.
- Evaluator **action, observation, initial-state and success/termination semantics**, max 200 steps, no privileged map details in policy inputs, no `diagnostic_snapshot()` supplied to policy. The evaluator may use public ground truth only for metrics after actions.
- The **primary diagnostic metric**: first-action goal alignment on the fixed panels, reported overall and U/D/L/R by checkpoint with exact numerator/denominator. Define goal-reducing actions via legal Manhattan-distance reduction on open-grid goals. Report invalid moves/ties/Wait separately; an initially reducing action is not a guarantee of success.
- **Secondary descriptive metrics**: success counts on original 4, expanded 16 and each direction's four cases; path length or steps to goal, collision/timeout, Wait rate, revisit/oscillation, mean goal-directed score margin (DQN Q values; PPO probabilities/log odds), exact step trajectories and per-checkpoint training goal-event counts.
- **Learning evidence metrics**: number of actual goal experiences, distinct goal transitions, replay/rollout draw counts, DQN online/target actual TD targets/masks/optimizer updates, PPO actual GAE/returns/batch advantages/optimizer updates, and directional goal-reward-to-update traces. Report denominators distinctly.
- **Interpretation criteria** (descriptive, NOT automatic permission for bigger training): whether both instruments meet all trace gates, and for each model whether expanded success improves from its own checkpoint-0 value by ≥4/16, achieves ≥12/16, and obtains ≥2/4 successes in **each** direction at 2,000 actions. These are exploratory readiness thresholds, not statistical significance or permission to select the best checkpoint after seeing outcomes. Always retain/report all four checkpoints, even if earlier is better.
- Hard time/storage limits, no-retry/partial receipt protocol, exact source fingerprints, loss tolerances and immutable output schema.

If a registered threshold is not met, report it as failed. Do not edit it post hoc. Do not opportunistically add another evaluation panel or an extra seed to improve reported performance.

## 5. Training implementation and passive evidence capture

Reuse the unchanged training implementations and Phase 5E observer. No observer may cause additional forwards, replay draws, RNG calls or optimizer operations.

- Record every accepted `env.step` with global action index, episode/reset/case identity, action, observation hashes, raw reward, terminated/truncated, and reason for end or incomplete prefix.
- **DQN:** identity-stable replay slots/generations, sampled indices (including duplicates), actual consumed tensors, target masks and target-network snapshots, actual TD targets/Huber losses, optimizer before/after fingerprints, target syncs, epsilon schedule and actual per-update action counters. Recompute actual targets/losses independently from captured tensors.
- **PPO:** sampled actions and probabilities, raw and collector-adjusted rewards, boundary terminal values, actual rollout rows/GAE/returns/advantages, shuffled minibatch membership, advantage normalization, ratio and clipping, policy/value/entropy losses, optimizer before/after fingerprints and actual update counts. PPO Q-values remain `not_applicable`.
- Do not reconstruct missing historical samples with later checkpoints; label observations `live_actual`, derived arithmetic `snapshot_reconstructed`, and missing items `not_observed` as appropriate.
- Save model and optimizer/replay/rollout provenance and checkpoint hashes without changing the training trajectory. Maintain a chained append-only action/update ledger and exclusive trace writer; pre-reserve budget for each action/update before it occurs.
- Explicitly separate *goal rewards encountered*, *distinct goal experiences sampled*, *total sample draws of goal experiences*, *actual updates involving them*, and *successful deterministic assessment episodes*. Never treat these as one performance number.

## 6. Fixed development evaluation without leakage

Only after exact checkpoint snapshots are recorded, evaluate each checkpoint on the **same previously defined original 4 and expanded 16 development cases**, including the Right-goal failures. An evaluator should restore a checkpoint copy and use a detached evaluation RNG; verification must prove no training RNG or optimizer state changed by assessment. No checkpoint can update weights.

Report for each model/checkpoint:

| Column | Definition |
|---|---|
| Model / training actions | DQN or PPO; 0, 500, 1,000 or 2,000 |
| Original success | Number completed out of 4, not combined with expanded |
| Expanded success | Number completed out of 16 |
| Up / Down / Left / Right | Number successful out of 4 in each direction |
| First-action goal alignment | Exactly defined numerator/20 on combined panels, plus panel-specific rates |
| First-step Right margin | Q(Right) minus highest non-Right Q for DQN; PPO action probability difference/log odds, explicitly different scales |
| Wait / loops / collisions | Counts and rates using actual evaluation actions |
| Exact shortest-route inefficiency | Only if using permitted evaluator-side reference; never give true shortest-route or full map to an RL agent |

Preserve all case-level paths and first-action scores, not just averages. Separate **static open-grid known task** results from any real-world navigation performance claims. Do not use the case results to choose a new training regime. The 20-case panels were already consulted in prior phases, so they are **development**, not untouched validation or final generalization evidence.

## 7. Verification gates and fail-closed rules

Use read-only verification after the run. No training, environment stepping, inference or policy scoring during `verify`.

1. **G0 provenance:** exact parent SHA and source/library pins; inventory and historical checksum preservation.
2. **G1 budget accounting:** at most 2,000 actual training actions per model and at most 4,000 total; exact recorded optimizer operations; no overflow, double count, hidden retry or unauthorized train/eval actions. Evaluation has its own ledger and 32,000 action ceiling.
3. **G2 live trace completeness:** action-to-reset/episode identity; replay sample identity/order and target fingerprints; PPO boundary/rollout/minibatch identities; closed shards, hashes, no missing required field.
4. **G3 independent arithmetic:** DQN actual targets/Huber losses; PPO GAE/returns/advantage normalization/losses and optimizer lineage. Use inherited preregistered numeric tolerance (atol=2e-5, rtol=2e-6) on float32 arithmetic; never weaken bitwise/provenance constraints.
5. **G4 evaluation correctness:** identical frozen panels across checkpoints, deterministic action semantics, no optimizer/model mutation, zero additional training RNG consumed by evaluation, correct success/collision/timeout/Wait counts, all 4 original and 16 expanded case rows per model/checkpoint.
6. **G5 integrity and tests:** complete unchanged parent regression suite separately ledgered; focused offline tests only; failed receipts retained; model/checkpoint/source hashes and `main`/dirty workspace unchanged; zero final/unseen accesses.

If a required signal is absent, a library path differs, the trace closes improperly, a quota would be breached, or checkpoint evaluation cannot be isolated, report `PHASE5G_INCOMPLETE_OR_BLOCKED`; do not fabricate data, silently weaken limits, rerun a failed arm, or continue to a different feature branch. If all scientific/engineering gates pass, report `PHASE5G_MEASUREMENT_VALIDATED` even if **navigation still fails**. Separately classify learning readiness as `BASIC_NAVIGATION_SCREEN_MET` or `BASIC_NAVIGATION_SCREEN_NOT_MET` for each model, according to the frozen checkpoint-2,000 screen.

## 8. Regression and accounting rule

Run existing full regressions **unchanged**, in a separately named legacy ledger. Do not modify inherited tests or charge their ordinary fixtures to Phase 5G research/synthetic action caps. Record completed legacy transitions/optimizer operations/wall time; retain honest `not_observed` values if a test process is interrupted. New focused tests must be offline readers or fake-ledger checks only (zero real environment transitions/optimizer operations). Preserve unexpected failures and their corrections. Do not rerun a scientific study because report generation failed; a report-only delivery fix must be separately hashed and must not modify scientific records.

## 9. Deliverables and publication

Create:

- `configs/partial_phase5g_v1.yaml` and `docs/partial_phase5g_protocol.md` with the frozen procedure, checkpoints, case IDs/hashes, thresholds and hard caps.
- `docs/partial_phase5g_results.md` with a checkpoint table for both models, primary/secondary metrics, all 20 cases, learning/optimizer trace evidence, runtime overhead, limitations, unexpected failures and resource accounting.
- `docs/partial_phase5g_diagnosis.md` with what has and has **not** been established; if learning remains poor, propose **one** specific future intervention but do not implement it.
- Minimal versioned new code for `register`, `run`, `verify`, `report` or equivalent, using existing library observers instead of copying/rewriting them. New offline-focused tests and retained JUnit/log receipts.
- Under `phase5g-longer-horizon-v1/`: immutable sealed manifest, new action/optimizer/evaluation ledgers, compressed actual live learning traces, exact checkpoint hashes/copies, fixed case-level deterministic development evaluations, arithmetic verification, failure receipts, output hashes, storage/time accounting and original workspace preservation audit. Do **not** copy a nested repo or historical study directories.

Commit and push **only** the Phase 5G feature branch once complete, with a clear commit subject and full report. If push fails, distinguish local from remote status. Do not merge `main` or modify historical branches. Then **STOP**. No model tuning, longer runs, extra seeds, dynamic training, final evaluation or Phase 6 without explicit new authorization.

## Codex execution instruction

> Read and implement `CODEX_PHASE5G_LONGER_HORIZON_LEARNING_DIAGNOSIS.md` on a fresh branch based precisely on `feature/phase5f-instrumented-learning-pilot-v1` at `6fb6aac124067b00016f5514ebbc66ef777dea33`. Run exactly one fresh B1 DQN and one fresh B1 PPO experiment, each capped at 2,000 actual Fixture A training actions, with registered live Phase 5E instrumentation and seed 23121. Capture checkpoints at 0/500/1,000/2,000; evaluate only on the existing original four and expanded sixteen fixed development cases, without mutating training state. Verify all trace math, budgets and preservation; commit/push your new branch. Report all negative results and stop before any extra training, final/unseen evaluation, full Phase 5 or Phase 6.
