# CODEX — Phase 5E: Training Instrumentation and Learning-Signal Verification

**Repository:** `happywintersolstice-oss/Autonomous_Robot_Navigation`  
**Base branch:** `feature/phase5d-goal-credit-diagnosis-v1`  
**Base commit:** `fe200034e816b10352453471f9c680b7c77d6ca4`  
**New branch:** `feature/phase5e-training-instrumentation-v1`  
**Study ID:** `phase5e-training-instrumentation-v1`  
**Authorization:** Instrumentation and bounded synthetic mechanical tests **only**. No new scientific training or evaluation.

## 1. Context and objective

Read the actual committed code, protocol and reports before implementing:

- `autonomous-robot-navigation-rl/docs/partial_phase5d_results.md`
- `autonomous-robot-navigation-rl/docs/partial_phase5d_diagnosis.md`
- `autonomous-robot-navigation-rl/docs/partial_phase5d_delivery_completion.md`
- `autonomous-robot-navigation-rl/docs/partial_phase5c_results.md`
- `autonomous-robot-navigation-rl/docs/partial_phase5b_results.md`
- `autonomous-robot-navigation-rl/src/robot_navigation/partial_phase5b.py`
- `autonomous-robot-navigation-rl/src/robot_navigation/partial_phase5b_diagnostics.py`
- `autonomous-robot-navigation-rl/src/robot_navigation/partial_training.py`
- `autonomous-robot-navigation-rl/src/robot_navigation/partial_neural.py`
- `autonomous-robot-navigation-rl/configs/partial_feasibility_v1.yaml`

The previous Phase 5D verdict was `DIAGNOSTIC_COMPLETED_INSUFFICIENT_HISTORICAL_SIGNAL`. In Phase 5B, scaled DQN reached 10/16 expanded simple-navigation goals but 0/4 Right cases. Phase 5C showed training contained Right-related experience: 104 Right-goal starts, 210 one-cell-Right decisions, and 88 Right-action goal arrivals (different denominators). The missing evidence is **the actual historical DQN replay sample/target/network sequence and complete PPO rollout/GAE/minibatch history**. Phase 5D could audit 24,000 linear updates but could not uniquely assign the neural-learning cause.

**Goal:** Build disabled-by-default logging of *actual* DQN/PPO learning computations and independently validate that logging does not change learning. This is an engineering instrumentation task, not a learning experiment or cure for Right-goal failures.

## 2. Hard restrictions

1. **Zero new scientific RL training actions, zero research assessment/navigation episodes and zero final/unseen evaluation access.** Small deterministic *synthetic mechanical tests* are allowed but counted separately; their goal outcomes are not research results.
2. Preserve the 10×10 environment, 409 raw observations, 412 B0 features, B1 relative-goal ×4, two-cell cardinal sensing, stale memory, action mapping `0 Up / 1 Down / 2 Left / 3 Right / 4 Wait`, rewards, timeout handling and learning hyperparameters.
3. Never change loss functions, architectures, action policies, replay/rollout sampling, initialization, target synchronization, optimizers, reward, epsilon, training-case sampler, action masks, planner fallback or training budgets.
4. No additional sampler call, random draw, permutation, optimizer step, environment step or model update merely for observation. Logger disabled/enabled must be operationally identical.
5. Keep Phase 5A–5D evidence byte-identical. Preserve original dirty workspace, Git index, historical branches and `main`. Use a new feature branch/worktree. Do not modify or retry old one-shot studies.
6. Classify every measurement as `live_actual`, `historical_logged`, `snapshot_reconstructed`, `synthetic_mechanical`, `not_observed` or `not_applicable`. Never promote a reconstructed post-update target to an actual training target. PPO action probabilities are not Q-values.
7. No auto-progression into learning experiments, obstacle training, full Phase 5, final testing or Phase 6.

## 3. Installed library audit and protocol registration

Inspect installed Stable-Baselines3, Torch, NumPy and Gymnasium source and record versions and SHA256 pins. Trace actual call paths for DQN replay sample indices, buffer overwrites, TD target calculation, optimizer step, target sync; and PPO `collect_rollouts`, last bootstrap values, `compute_returns_and_advantage`, minibatch index shuffling, normalized advantages and optimizer calls. Do not assume another SB3 release works identically.

Before mechanical validation, freeze a protocol: exact hooks, schema, RNG and parameter invariance checks, test fixtures and seeds, pass/fail thresholds, source hashes, resource ceilings, and failure-injection procedure. Implement additive optional observers; if hooking the library is unavoidable, observe the real method **once** and restore hooks even on exceptions. Reject unsupported library versions rather than approximating their behavior.

Suggested paths:

- `docs/partial_phase5e_protocol.md`
- `configs/partial_phase5e_v1.yaml`
- `src/robot_navigation/partial_training_instrumentation.py` (split DQN/PPO helpers if needed)
- `src/robot_navigation/partial_phase5e.py` (`validate`, `audit`, `verify` only)
- `developer/tests/test_partial_phase5e.py`

## 4. Shared trace contract

Version all log schemas and include: study/run/algorithm/arm/seed/source/config/dependency hashes; episode/reset/case and monotonic environment interaction IDs; replay transition ID, **slot and slot generation** to distinguish overwrites; environment index; rollout ID; optimizer update/epoch/minibatch IDs; online and target network version fingerprints; actual raw/adjusted reward, terminated/truncated and consumed done masks; source-kind provenance, tensor shape/dtype, nonfinite checks, content hashes and explicit absence reasons. Preserve sample order and duplicates.

Log compact binary/numeric arrays where appropriate rather than huge repeated 409-value vectors. Use bounded compressed shards, metadata/data validation, fail-closed quotas, deterministic record-order checks and small representative full-tensor synthetic samples. Avoid copying whole historical source trees into the new archive.

## 5. DQN: observe real replay and TD credit

On every *actual* DQN optimizer step, capture:

1. Actual replay minibatch indices **in original sample order**, including repeated indices, ring slot, slot generation, transition ID, env index and exact buffer sampling mode. Observe true consumed observations, actions, rewards, next observations and masks after any library normalization.
2. Actual **pre-update** online `Q(s,a)`, target-network `max_a Q_target(s',a)`, actual discount, continuation mask, **real TD target tensor**, individual TD residual, actual loss and loss reduction.
3. Actual optimizer update ID, learning rate, clipped gradient norms, parameter update magnitude, online/target fingerprints before and after, target sync event ID and epsilon/training-action counters.
4. Exact distinction between a real terminal goal/collision and external time-limit truncation, including the installed Stable-Baselines3 timeout masking and any `terminal_observation` behavior.

**Critical library issue:** The replay sample object might not expose sampled indices. Capture them passively *at the existing index generation/consumption site* and use a transition-ID sidecar keyed by replay slot plus generation. Never call `.sample()` again or infer indices from identical observation values; the replay may contain duplicates. Test ring-buffer wraparound and repeated samples. Record the true tensor used for the loss; recomputing with a saved later network is not the same thing.

If a required actual value cannot be captured without altering behavior, stop with `UNSUPPORTED_INSTRUMENTATION_PATH`. Do not fabricate values or call the attempt fully validated.

## 6. PPO: observe real rollout, GAE and optimizer records

For *actual* PPO collection and training:

1. Record rollout/environment/step IDs, observations and executed actions, pre-update action probabilities/log-probs, `V(s)`, raw environment rewards **separately** from any collector-adjusted rewards, episode-start and terminal/truncated flags, terminal observations and time-limit bootstrapping.
2. Record exact **pre-update** final-rollout `last_values` and last dones, and the real GAE inputs/parameters/masks. Do not substitute post-update critic values for missing bootstrap values.
3. After `compute_returns_and_advantage`, capture **actual** rollout-buffer values, advantages and returns and their row-to-transition mapping. Verify recurrence using the installed library's handling of episode boundaries and timeout adjustments.
4. Observe actual sampled/shuffled rollout indices for each epoch and minibatch **without another permutation**. Record raw/normalized advantages, normalization statistics, old/new log-probs, probability ratio, clipped objective, value/entropy/total losses, approximate KL, clipping/early-stop events and true optimizer step IDs.
5. Record model/optimizer fingerprints and count actual optimizer steps. Separate stochastic training behavior from deterministic policy evaluation. PPO does **not** output DQN-style `Q(s,a)`.

The old saved PPO last 100 transitions at 2k/6k cannot stand in for a full historical GAE trajectory. This logger is for **future authorized runs only**. Missing boundary/batch values should be reported as `not_observed`, never inferred.

## 7. Q-learning and SARSA integration

Reuse Phase 5B's tested normalized semi-gradient diagnostics; add thin passive shared-schema adapters only where necessary. Record actual feature signature/transform, sampled behavior action, prior Q, target, TD error, true next sampled action for SARSA, normalized update and post-update weights. Do not replace or modernize the linear agents as part of instrumentation. DQN/PPO-only fields are `not_applicable`.

## 8. Required mechanical tests (not research experiments)

### 8A. On/off equivalence, all four models

Using fixed tiny synthetic streams in isolated matched processes/states, run each learner once with logger disabled and once with logger enabled. Assert identical:

- Environment actions, chosen behavior actions/probabilities, observations, reward, termination and reset sequence.
- RNG streams, replay sampling indices/order and buffer overwrite contents (DQN).
- PPO rollout collection, GAE/returns, actual minibatch permutation, optimizer updates and early stops.
- Linear targets and weights; neural losses, clipped gradients and optimizer counters; model/target/optimizer parameters and buffer states at matched boundaries.

Prefer bitwise CPU equality. If backend determinism requires a tolerance, preregister the smallest defensible tolerance and reject different sampled operations. The logger must not make an extra model/random/env/optimizer call with behavioral effects.

### 8B. DQN numerical truth

- Tiny analytic nonterminal, goal, collision and external truncation targets; test the actual library-consumed done mask and the formula `r + gamma * mask * max Q_target(next)`.
- Identical observations with distinct transition IDs, sampling with replacement, ring overwrite/slot generation and target sync at known updates.
- Verify actual pre-update target tensor and loss against independently calculated values using **those same captured tensors**, not post-hoc network snapshots.
- Failure injection: malformed index, duplicated/invalid transition identity, mismatched network version, nonfinite tensors, checksum corruption and quota exhaustion.

### 8C. PPO numerical truth

- Terminal, truncation, ongoing rollout and time-limit bootstrap cases; compare raw vs adjusted reward and last-value capture.
- Independent GAE/return calculations from the actual saved rollout-buffer inputs and exact pre-update bootstrap.
- Original shuffled minibatch IDs, normalization and PPO loss/ratio/clipping calculations; verify any early stopping matches library behavior.
- Check that observing stochastic policy actions leaves actions/RNG unchanged. Reject missing bootstrap or mismatched rollout IDs; retain partial failure receipts.

### 8D. Information and preservation

Test 409-value input and 5-action order, B0/B1 signatures and checkpoint loading; forbidden true-map, future-event and sealed-final access; previous experiment SHA256 pins, `main` status and original dirty workspace. Execute the existing full regression suite and focused Phase 5E suite; report real counts only.

## 9. Fixed resource ceilings and failure handling

Record the following ceilings in the sealed implementation protocol **before running synthetic tests**:

| Category | Hard limit |
|---|---:|
| New scientific training actions | **0** |
| New research navigation-assessment actions | **0** |
| New sealed-final/unseen evaluations | **0** |
| Synthetic mechanical environment transitions | **1,000 total** |
| Synthetic mechanical optimizer steps, all models combined | **500 total** |
| Synthetic validation wall time | **1,200 seconds** |
| New validation artifact bytes, excluding code/tests | **128 MiB** |

Keep a ledger separating synthetic tests, old recorded data reads, and scientific actions; never count a unit-test update as a research training result. On a resource or correctness failure, stop, preserve the genuine partial receipt and provide `INCOMPLETE_VALIDATION` or `INSTRUMENTATION_BLOCKED`. Development-phase source/test fixes may be made transparently *before* final protocol sealing, retaining failure history; do not erase failed attempts or rerun a frozen one-shot validation to manufacture a pass.

## 10. Deliverables

Create new, versioned content in the existing project layout:

1. `docs/partial_phase5e_protocol.md` — frozen source/library call paths, version compatibility, logger hooks, data schema, synthetic fixtures, numerical tolerances and budget.
2. `docs/partial_phase5e_results.md` — actual logger on/off comparison, DQN/PPO numerical checks, full/focused test results, overhead, failed tests/fixes and final verdict.
3. `docs/partial_phase5e_diagnosis.md` — exactly what the new logs can reveal about a *future* learning run and which causal questions remain unresolved.
4. `configs/partial_phase5e_v1.yaml`, optional disabled-by-default instrumentation modules, and read-only `validate`, `audit`, `verify` commands. Do not add an experiment-training entry point in Phase 5E.
5. `developer/tests/test_partial_phase5e.py` or equivalent focused test files covering all models, reproducibility, real DQN targets/sample IDs, real PPO GAE/minibatches, timeout semantics, injected failures and zero forbidden calls.
6. `artifacts/partial-dynamic-navigation-v1/phase5e-training-instrumentation-v1/`: compact source/dependency/parent hashes, schema examples, operation ledger, paired equivalence receipts, synthetic numerical evidence, previous-artifact preservation, failure history, verifier output and final delivery verification.

Reference existing Phase 5A–5D evidence by repository commit and hash rather than copying multi-GB archives or full source trees. Report commands and reproduction requirements honestly: verify must be read-only and must not replay scientific training. Never silently repair or rewrite the old Phase 5D renderer; if new delivery needs a fix, make its code and receipt separately auditable.

## 11. Mandatory acceptance gates

| Gate | Acceptance condition |
|---|---|
| **G0 — Source and preservation** | Correct parent, original artifacts/Git state preserved, zero scientific learning/evaluation/final-data actions |
| **G1 — Actual capture** | DQN real minibatch indices/TD targets/network versions and PPO real bootstrap/GAE/minibatch/loss values with verified transition identity |
| **G2 — No behavioral interference** | Disabled/enabled runs match actions, sampler order, RNG, buffers, optimizers, model/target parameters and learning operations |
| **G3 — Numerical truth** | Independently check actual-value DQN Bellman targets, PPO GAE/returns and linear TD with correct truncation semantics |
| **G4 — Budgets and failure handling** | All caps enforced; malformed inputs, interrupted logs and integrity failures fail closed |
| **G5 — Delivery and reproducibility** | Regression and focused tests pass; hashes, manifests, CLI verification, reported failures and previous workspace preservation are valid |

Return **exactly one** of:

- `INSTRUMENTATION_VALIDATED_NO_RESEARCH_TRAINING` — every gate passed, with evidence.
- `INSTRUMENTATION_BLOCKED` — missing genuine signals, noninterference or numerical correctness blocked.
- `INCOMPLETE_VALIDATION` — interruption or limit prevented complete testing.

A successful verdict permits **only** a recommendation for a *separately authorized* small, registered Fixture A learning pilot using the new instrumentation. It does not prove better navigation, generalization, correct learned credit or model superiority.

## 12. Branch and final response

Implement on `feature/phase5e-training-instrumentation-v1`, based on the verified committed Phase 5D branch. After a successful audit, commit and push only new Phase 5E source/tests/compact synthetic evidence; never merge `main` or change frozen history. If publishing is not permitted or fails, report the actual state rather than claiming a push. Preserve any other uncommitted work. On Windows, use a scoped Git long-path option if needed rather than renaming archived files.

Codex's final response must list: branch and commit/push status; passed/failed test counts; exact synthetic and scientific action totals; DQN and PPO fields captured versus unsupported; bitwise/tolerance on-off equivalence evidence; preservation hashes; report and evidence paths; and **one** proposed next small study requiring user approval. Do not start it.

**Definition of done:** Future learning can be observed with actual replay/rollout/optimizer data and controlled validation demonstrates the observers do not alter the learners. This phase must end **before** any new research training.
