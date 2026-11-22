# CODEX — Phase 5F: Instrumented Learning-Signal Pilot

**Project:** Autonomous Robot Navigation with Reinforcement Learning  
**Repository:** `happywintersolstice-oss/Autonomous_Robot_Navigation`  
**Reviewed parent branch:** `feature/phase5e-training-instrumentation-v1`  
**Reviewed parent commit:** `192ba8dca9957ac6d6b38965956b06f05b694548`  
**Create a new branch:** `feature/phase5f-instrumented-learning-pilot-v1`  
**New study namespace:** `phase5f-instrumented-learning-pilot-v1`  
**Authorization:** Exactly two tiny instrumented, real Fixture A training runs; **not** full Phase 5 and **not** a navigation performance study.

## 1. Objective and reason for this study

Phase 5A showed that agents can receive goal rewards during exploration yet navigate unreliably in deterministic evaluation. Phase 5B observed a DQN B1 improvement to 10/16 expanded Fixture A successes, but Right-goal performance remained 0/4 and no family passed the complete preregistered gate. Phase 5C found Right-related training exposure and no sufficient justification for the proposed episode-balanced sampler. Phase 5D could not reconstruct the historical DQN optimizer sample/target sequence or full PPO GAE and minibatch history. Phase 5E added passive instrumentation, proved bitwise on/off equivalence on the pinned mechanical CPU path, and recorded a verdict of `INSTRUMENTATION_VALIDATED_NO_RESEARCH_TRAINING`.

**Phase 5F asks one question only:** Can the approved Phase 5E instrumentation capture and independently verify actual DQN and PPO learning signals during short, unchanged, real Fixture A training episodes?

Do not infer that instrumentation improves navigation, fixes Right-goal selection, or proves credit assignment causality. The pilot is a trace-completeness and measurement-validity exercise.

## 2. Mandatory preflight and branch hygiene

1. Confirm the exact parent branch, parent HEAD, and repository/application root. Work on a **new** Phase 5F feature branch based on the parent commit above; if HEAD differs, halt and document the discrepancy before proceeding.
2. Read the committed `docs/partial_phase5e_protocol.md`, `docs/partial_phase5e_results.md`, `docs/partial_phase5e_diagnosis.md`, `configs/partial_phase5e_library_v1.json`, and current `partial_training_instrumentation.py`, `partial_instrumentation_trace.py`, `partial_phase5e.py` code. Verify parent evidence using its read-only verification entry point; do **not** rerun the sealed Phase 5E validation.
3. Read Phase 5B fixture definitions and baseline training configuration, specifically the legal open-grid Fixture A **training split**. Reuse existing training cases without changing geometry, rewards, sensors, action IDs, observation vectors, policies, hyperparameters, or environment termination rules.
4. Save an inventory/digests of the parent source, historical artifacts, relevant configurations and any original dirty workspace. Do not modify historical code or evidence, including old Phase 5D delivery issues. Do not merge to `main`.
5. Confirm the CPU/single-env installed library/source compatibility contract **before** training. If mismatched, fail closed; do not silently patch old observer hooks or pin a different version. If the observer's current supported path cannot capture required fields, report `INSTRUMENTATION_BLOCKED` and stop.
6. No final, held-out, unseen, reserved layouts or events may be loaded or accessed, even for a quick check.

## 3. Strict and frozen experiment contract

| Property | Required value |
|---|---|
| Environment | Original, unchanged open-grid Fixture A **training** cases only |
| Algorithms | **DQN and PPO only** |
| Feature version | `B1-relative-x4` / `partial-visible-relative-scale4-v1` |
| Primary seed | **23121** for each independent model run |
| Training actions | **500 per model** — cumulative hard maximum **1,000** |
| Training runs | Exactly **one** DQN and **one** PPO, no trial seeds or retries |
| Model configuration | Existing Phase 5B B1 hyperparameters and architecture, unchanged |
| Training-case stream | Previously defined Fixture A training sampling rule, unchanged; log actual case IDs |
| Observation/action contract | 409-value observation; 412 affine features; 5 actions (Up, Down, Left, Right, Wait) |
| Reward/termination | Unchanged partial-navigation contract |
| Instrumentation | Phase 5E `observe(..., sink=...)` enabled; existing pinned path |
| Novel feature/reward/optimizer/sampler changes | **Forbidden** |
| Research evaluation episodes | **0** |
| Final / unseen evaluation | **0** |
| Full Phase 5 or Phase 6 | **Forbidden** |

Any learning step attributable to this pilot must be debited **before** execution. Training interaction count means actual `env.step` calls, not resets, optimizer steps, samples or synthetic tests. PPO may finish a rollout only if the 500-action ceiling can be respected exactly; do not exceed the cap to complete a rollout. Ensure DQN warmup, update frequency, PPO `n_steps` and batch sizes use the actual existing code rather than guessed values. Make both algorithms' scientific budgets comparable by counted environment actions; do **not** claim equal optimizer computation or equivalent sample efficiency.

If any prerequisite cannot be honored, **stop** with an honest incomplete status. Do not substitute another algorithm, seed, case generator, feature set, or training budget.

## 4. Register and seal BEFORE the first research action

Create a frozen Phase 5F protocol, run manifest, immutable action-budget ledger and unique artifact namespace. Register:

- Parent commit/tree, dependency/runtime/library and source hashes, hardware mode, branch, configuration hash, supported observer call path, project/task version signatures.
- Frozen seed `23121`, algorithms, B1 representation, source of Fixture A training cases, case IDs and split audit, 500-action per-model ceiling and 1,000 total hard ceiling.
- Deterministic predeclared training order (DQN first, PPO second), stop conditions, exact accounting rules, checksums, artifact-size caps and replay/verification procedure.
- Required trace schema, quality/coverage gates below, and objective **trace correctness**, not policy success.
- No ability to append a different training arm, seed, checkpoint-selection procedure, or evaluation panel after inspecting pilot results.

This is one attempt only. Failed/partial runs and artifacts remain preserved. A crash, exhausted budget, corrupt trace, incomplete sink closure or unsupported path must not trigger a silent retry, modified namespace, deleted history, or extra training actions.

## 5. Capture real learning, not synthetic or reconstructed substitutes

Use Phase 5E's versioned `partial-learning-live-v1` trace schema and existing `TraceWriter`, supported observer, identity validation, hash/provenance checks and quota handling. The observer must remain passive and call each actual learner/optimizer operation only once.

**Common record requirements**

- Actual training action ID, ordered reset/episode/case ID, seed, observation and next-observation digests, reward, terminated/truncated flags, outcome and action-selection/exploration context when actually observable.
- Environment and source information boundaries: no hidden true grid, future dynamic schedule, privileged shortest path, final/test records or secret evaluator state supplied to models or logs.
- Explicit source classification: `live_actual`, `historical_logged`, `snapshot_reconstructed`, `not_observed`, `not_applicable`; do not present inferred calculations as observed optimizer tensors.
- Every trace record includes event order, model/optimizer/target versions when relevant, tensor dtype/shape/checksum, finite-number checks and chain/inventory hashes.
- Preserve raw terminated versus external timeout/truncation semantics and actual replay masking or collector timeout bootstrap values. Do not silently reinterpret a positive timeout reward as a goal reward.

**DQN-specific requirements**

- Actual replay insertion transition IDs and ring slot/generation; actual sampled indices in order, including duplicates; consumed batch tensors and done/timeout masks.
- Pre-update online selected-action Q, target-network next-state maximum Q, **actual TD target**, discount, actual Huber loss and available gradient/optimizer quantities.
- Online/target model and optimizer fingerprints both before and after each optimizer update, real target-network sync events, exploration schedule, `learning_starts`, training-step counters and replay-buffer integrity.
- Link sampled goal transitions to their later actual optimizer-use events where supported. Report observed sample-selection frequencies without implying every goal experience contributed to a given update.

**PPO-specific requirements**

- Actual collected action/probability/log-probability/value, raw environment reward and **collector-adjusted** reward, original episode/rollout boundary and timeout bootstrap value.
- Exact rollout row IDs, pre-update last values/dones, actual GAE and returns, shuffled minibatch order and actual normalized advantages (with the library's real sample-std behavior).
- Ratio, clipped/unclipped objectives, policy/value/entropy losses, actual total loss, KL, clipping, early-stop events, optimizer-step fingerprints and whether a batch performed an actual update.
- **No PPO Q-values**: report policy probabilities, log odds and V where appropriate. Never fabricate off-policy DQN-style TD targets for PPO.

For any quantity not available on the pinned path, record `not_observed` and identify the exact missing field. If it is required for the pilot's primary trace-completeness gate, stop and report blocked rather than guessing.

## 6. Independent verification and measurement gates

After the two budgeted runs, perform **read-only** verification of committed logs without stepping the research environment, calling policy predictions or training. Independently check:

1. **Action accounting**: exactly 500 accepted training transitions per completed arm; maximum 1,000 aggregate; complete reset/step/terminal linkage, no duplicated or omitted transition IDs, correct handling of incomplete episode prefixes.
2. **Trace completeness**: every actual optimizer operation has exactly one corresponding update record; sampled replay identities resolve to actual prior insertions; PPO minibatches resolve to the correct rollout/row/epoch; target/optimizer version history is coherent.
3. **DQN arithmetic**: recompute TD targets from actual consumed reward/mask/target network values and independent loss arithmetic using registered float32 tolerances; verify terminal vs timeout bootstrapping.
4. **PPO arithmetic**: independently recompute backwards GAE from actual saved pre-update boundary values, returns, normalization and clipped/total loss from consumed tensors; report unsupported/absent fields rather than filling them in.
5. **RNG and model preservation**: verify observer did not invoke an extra sampler/forward/update. The Phase 5E bitwise on/off *synthetic* equivalence is inherited evidence; **do not claim live bitwise on/off equivalence for Phase 5F**, since an uninstrumented matched research run is not authorized.
6. **Split integrity**: all recorded cases belong to declared Fixture A training IDs; zero research evaluation/final/unseen accesses.
7. **Budgets/provenance**: source/library/config/checkpoint hashes, finite tensors, ordered trace shards, no missing completion marker, frozen registration, exact counts and no unapproved writes.

When the model sees a reward-bearing transition, identify its true environment outcome; separately determine if/when the update actually consumed it. Compute observational summaries of the relevant DQN target vs chosen-action Q or PPO advantage/probability **without claiming causality or navigation improvement**. Give at least one traceable reward-to-update example per model **if it occurred**, otherwise report `not_observed` with counts and explain why the short pilot may not contain the desired event.

### Hard failure conditions

Return `INSTRUMENTATION_BLOCKED` or `PILOT_INCOMPLETE` if actual source/library mismatch, trace corruption, unresolvable sample IDs, missing required PPO boundary signal, extra actions beyond a cap, invalid checksums, nonfinite updates, unsupported device/vectorization, or unexplained disagreement in independent numerical checks. Do not weaken a tolerance after seeing the result.

### Successful engineering outcome

`INSTRUMENTED_LEARNING_TRACES_VALIDATED` is allowed **only** when both 500-action real Fixture A arms are complete, all mandatory observed fields are present and independently verified, every hard constraint is satisfied, and baseline/targeted regressions pass. Success means valid training evidence, **not successful navigation or an algorithm ranking**.

## 7. Test strategy and limits

- Run the unchanged full parent regression suite as a **separate ledger**. Do not omit inherited tests or debit their existing learner/environment fixtures against the Phase 5F scientific budget.
- Add narrowly scoped tests for Phase 5F registration, exact 500-action stopping, active observer trace integration, DQN replay goal-transition links, PPO rollout boundaries, read-only verification, attempt failure preservation and fail-closed unsupported modes.
- Any new **synthetic-only** test transitions/optimizer operations must have a separate declared hard cap, ledger and provenance. Set these limits in the preregistration based on a preflight dry inspection of the test plan, **not after observing outcomes**. They are separate from the 1,000-action scientific ceiling, and cannot be relabelled as scientific goal evidence.
- Run a source/information-boundary test and verify that the instrumentation is still disabled by default outside the two explicit pilot runs.
- Retain test receipts for genuine failures and corrections, with no stealth retries of research training. Repairing report formatting or test code after the study must use an explicit separately hashed delivery-only appendix and cannot alter frozen scientific inputs/records.

## 8. Artifact and runtime safeguards

Create a fresh `artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1/` namespace, with compressed streaming trace shards and **no nested repository/source-snapshot copies**. Record aggregate file sizes before writing new shards; cap newly generated study artifacts at **512 MiB** and the two pilot training runs together at **1,800 seconds** elapsed (excluding separately accounted legacy regression tests). Give each arm a **900-second** ceiling; exhaustion stops the run without a training retry. If expected logs exceed the byte cap, stop before the first run and seek separate authorization rather than silently dropping required fields. Do not raise these caps after registration.

Do not run evaluation episodes for checkpoint selection. Checkpoint inference/continuation state only when necessary for provenance; do not re-use these checkpoints to claim performance or choose a training winner. Save an incomplete prefix and evidence when stopped early.

## 9. Required deliverables

New files under the Phase 5F branch and namespace:

- `docs/partial_phase5f_protocol.md` — immutable registration and prespecified gates.
- `configs/partial_phase5f_v1.yaml` — frozen scope, seed, models, cases, budgets, artifact limits.
- `docs/partial_phase5f_results.md` — actual numbers, trace fidelity, observed learning signals, negative findings and limits.
- `docs/partial_phase5f_diagnosis.md` — what can and cannot be inferred; one justified next proposal only.
- A minimal new implementation/tests/CLI for `register`, `run`, `verify`, `report` or equivalent, respecting one-shot semantics; do **not** modify old trial CLIs.
- The evidence namespace: sealed plan, parent/source/library hashes, training-action/optimizer ledgers, compressed DQN/PPO actual live traces, summaries, independent math checks, failure history, test receipts, output hash inventory, verification verdict and original repository preservation receipt.

The report must include the full per-algorithm actual action counts, optimizer counts, number of sampled goal-reward transitions, DQN target/residual summaries, PPO observed advantage/return summaries, time/space overhead and any missing signals. Explicitly distinguish **recorded research training** from Phase 5E synthetic verification and earlier Phase 5B training. Provide clear links between the trace records, checkpoint versions, and summary rows. A short useful result with honest missingness is preferable to a fabricated perfect pass.

## 10. Publication and explicit stop

After independent validation, commit and push this **new** feature branch, including its complete report and immutable receipts. If the committed branch cannot be pushed, report local commit and remote status separately; do not claim publication. Preserve Phase 5A/B/C/D/E and `main` byte-for-byte; never merge into `main` or edit the original user's dirty workspace.

**STOP after Phase 5F.** Do not launch additional seeds, DQN/PPO training, Q-learning/SARSA training, new evaluation tasks, obstacle/dynamic curricula, full Phase 5, final held-out evaluation or Phase 6. Even if all gates pass, a later learning-improvement experiment requires separate user authorization and registration.

---

## Exact message to execute

> Implement the attached `CODEX_PHASE5F_INSTRUMENTED_LEARNING_SIGNAL_PILOT.md` as a single bounded Phase 5F study on a fresh branch based on `feature/phase5e-training-instrumentation-v1` at `192ba8dca9957ac6d6b38965956b06f05b694548`. Train exactly one B1 DQN and one B1 PPO for at most 500 genuine Fixture A training actions each, with live Phase 5E instrumentation, frozen seed 23121 and unchanged model/reward/environment settings. Verify actual learning traces and their independent numerical consistency. Report honestly, preserve previous artifacts, commit/push the Phase 5F branch, and stop. Do not evaluate navigation performance or proceed to full Phase 5/Phase 6.
