# Phase 5E training instrumentation results

**INSTRUMENTATION_VALIDATED_NO_RESEARCH_TRAINING**


Branch `feature/phase5e-training-instrumentation-v1`, reviewed parent `fe200034e816b10352453471f9c680b7c77d6ca4`, tree `b23e1f69573fcc92b24721ea9a9970cf43e8b8c6`. This report is generated before the Phase 5E publication commit; the final response/Git branch identifies that commit. Instrumentation and synthetic mechanics only. No scientific learning, research navigation-assessment episode, sealed-final/unseen evaluation or Phase 6. No model superiority, improved navigation or resolved Right-goal cause is claimed.


## Scope and source provenance


[Protocol](partial_phase5e_protocol.md), [diagnosis](partial_phase5e_diagnosis.md), [evidence summary](../artifacts/partial-dynamic-navigation-v1/phase5e-training-instrumentation-v1/summary.json). The hash-sealed attempt plan pins the new source, tests, configuration, minimal flat source snapshot and actual installed library implementations. The library audit contains exact call-path source and dependency SHA256 values; no assumption about another SB3 release is made.


| Package | Installed version |
| --- | --- |
| stable-baselines3 | 2.9.2 |
| torch | 2.15.0 |
| numpy | 2.5.4 |
| gymnasium | 1.4.0 |


The optional observer is disabled by default. A missing sink installs no hooks. Enabled observers capture actual live computations on the pinned CPU/single-environment path. Original learner methods are called once, with no replacement optimizer, second sampler, extra permutation, prediction or environment action for logging. The independent profile witness is present in both off/on members and verifies the optional line/optimizer/tensor observers introduce no behavioral difference. All hooks and reduction wrappers are restored, including under injected exceptions.


## Paired on/off equivalence and measured overhead


| Model | Arm | Transitions per member | Updates per member | Equality | Off seconds | On seconds | On/off ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dqn | B0-original | 4 | 1 | bitwise / tolerance 0 | 0.333481 | 0.525038 | 1.57442 |
| dqn | B1-relative-x4 | 4 | 1 | bitwise / tolerance 0 | 0.240387 | 0.554299 | 2.30586 |
| ppo | B0-original | 100 | 10 | bitwise / tolerance 0 | 3.3158 | 9.32264 | 2.81158 |
| ppo | B1-relative-x4 | 100 | 10 | bitwise / tolerance 0 | 4.50381 | 11.139 | 2.47324 |
| q-learning | B0-original | 8 | 8 | bitwise / tolerance 0 | 0.465583 | 0.696764 | 1.49654 |
| q-learning | B1-relative-x4 | 8 | 8 | bitwise / tolerance 0 | 0.44496 | 0.696369 | 1.56501 |
| sarsa | B0-original | 8 | 8 | bitwise / tolerance 0 | 0.387274 | 0.742147 | 1.91634 |
| sarsa | B1-relative-x4 | 8 | 8 | bitwise / tolerance 0 | 0.452629 | 0.729227 | 1.61109 |


Each member runs in a fresh matched subprocess using seed 23101 and original model hyperparameters. CPU comparison tolerance is zero. Compared records include all executed actions and probabilities, resets, raw observations/rewards/flags, global/env/action RNG, actual replay samples and PPO minibatches, losses, clipped gradients, optimizer state, model/target weights, complete buffer fingerprints and forward counts. Timing is mechanical-fixture overhead including capture and serialization, not a representative production benchmark or statistical estimate.


The fixed synthetic stream deliberately repeats one legal 409-value observation while emitting controlled continuing/goal/collision/timeout flags and rewards. Actions cannot change that artificial stream. Its positive labels are not research navigation successes. DQN collects four transitions below normal warmup and then executes one explicitly isolated real train-method unit update. The target-sync callback is exercised at its actual 250th callback invocation without more environment steps. PPO collects its original 100-step rollout and normal five-epoch/two-minibatch training, then both disposable test instances receive the declared target_kl=1e-12 early-stop fault fixture on that same existing rollout. No historical model, config or production hyperparameter is changed.


## Actual signals captured and independent numerical checks


| Learner | Captured actual fields | Independent check / limitation |
| --- | --- | --- |
| Q-learning / SARSA | Actual target/error, phi, prior/live weights, chosen action, actual next sampled SARSA action, alpha/gamma, normalized increment and post-update weights | Existing unchanged normalized semi-gradient rule; no max substitution for SARSA; prior Q is algebra on actual pre-update inputs |
| DQN | Real batch indices in order including duplicates; env index, slot/generation/transition IDs; consumed replay tensors and masked dones; actual pre-update chosen Q, target max, discounts, target tensor and Huber loss; clipped gradients; update IDs/LR/fingerprints; target sync and epsilon/counters | Actual r+gamma×(1-done)×target max; terminal targets equal raw reward; external timeout bootstrap retained; never reconstructed with a later network |
| PPO | Actual sampled actions/probabilities/log-probs/V; raw and collector-adjusted rewards; timeout terminal observations/values; exact final pre-update bootstrap/dones; rollout rows/GAE/returns; original shuffled batches; actual normalization mean/std; ratios/objectives/value/entropy/total losses/KL; clip/early-stop events and optimizer IDs | Backward GAE with actual bootstrap, returns identity, sample-std normalization and loss arithmetic; PPO Q fields explicitly not_applicable |


| Pair | DQN loss batches | PPO loss batches | GAE rollouts | Linear updates | Neural optimizer steps |
| --- | --- | --- | --- | --- | --- |
| dqn-B0-original | 1 | 0 | 0 | 0 | 1 |
| dqn-B1-relative-x4 | 1 | 0 | 0 | 0 | 1 |
| ppo-B0-original | 0 | 11 | 1 | 0 | 10 |
| ppo-B1-relative-x4 | 0 | 11 | 1 | 0 | 10 |
| q-learning-B0-original | 0 | 0 | 0 | 8 | 0 |
| q-learning-B1-relative-x4 | 0 | 0 | 0 | 8 | 0 |
| sarsa-B0-original | 0 | 0 | 0 | 8 | 0 |
| sarsa-B1-relative-x4 | 0 | 0 | 0 | 8 | 0 |


Float32 numerical reconstruction uses preregistered atol=2e-5, rtol=2e-6; linear algebra uses atol=1e-10. These numerical tolerances never relax bitwise on/off equality, identities, permutations or sampled operations. Actual raw terminal/truncation flags, replay timeouts, consumed masks, collector reward augmentation and final rollout continuation are kept separate. Versioned records carry provenance, shape/dtype, finite checks and hashes. Missing values remain not_observed; absent architecture fields are not_applicable.


## Supported path and unsupported conditions


The validated path is the current project CPU, one synchronous environment, 409 inputs/412 transformed features, Discrete(5), standard one-step replay and default production architecture. Unsupported modes fail closed: unmatched library versions/source, GPU, multiple environments, VecNormalize, SDE, memory-optimized/n-step replay, prefilled replay without identity sidecar, active debugger trace. This is explicit compatibility scope, not a claim that every SB3 mode or release has been tested. Case identifiers must be provided by the caller/public metadata; missing IDs are recorded as not_observed rather than reading hidden geometry. The instrumentation does not recover missing historical Phase 5B/DQN minibatches or old PPO GAE; it records future authorized execution only.


## Tests, injections and failure history


| Test process | Tests | Failures | Errors | Skipped |
| --- | --- | --- | --- | --- |
| New final Phase 5E focused suite | 32 | 0 | 0 | 0 |
| Complete unchanged parent regression suite | 432 | 0 | 0 | 0 |


Negative fixtures require rejection of malformed replay indices, duplicate transition IDs, wrong generations/rollout IDs, wrong network versions, nonfinite tensors, corrupt payloads/checkpoints, missing bootstrap, interrupted logs, provenance spoofing, exhausted quotas and forbidden final/unseen paths. These expected rejections are passing tests, not discarded failed experiments. Genuine unexpected development failures, if any, are listed below and their receipts retained.


| Attempt | Status | Actual test counts | Handling |
| --- | --- | --- | --- |
| 0001 | failed development attempt | {'tests': 31, 'failures': 0, 'errors': 19, 'skipped': 0} | Preserved genuine receipt; any source fix occurred before final seal; operations remain cumulatively charged |
| 0002 | failed development attempt | {'tests': 2, 'failures': 0, 'errors': 2, 'skipped': 0} | Preserved genuine receipt; any source fix occurred before final seal; operations remain cumulatively charged |
| 0003 | passed development attempt | {'tests': 2, 'failures': 0, 'errors': 0, 'skipped': 0} | Preserved; counted in cumulative mechanical ledger |
| 0004 | failed development attempt | {'tests': 2, 'failures': 1, 'errors': 0, 'skipped': 0} | Preserved genuine receipt; any source fix occurred before final seal; operations remain cumulatively charged |
| 0005 | passed development attempt | {'tests': 2, 'failures': 0, 'errors': 0, 'skipped': 0} | Preserved; counted in cumulative mechanical ledger |


## Resource accounting


| Category | Actual | Hard cap |
| --- | --- | --- |
| New Phase 5E transitions, cumulative across development and final | 812 | 1000 |
| New Phase 5E optimizer steps, cumulative | 158 | 500 |
| New active synthetic validation wall seconds, cumulative | 131.107 | 1200 |
| Cumulative artifact bytes before final report metadata (includes retained local development) | 101914041 | 134217728 |
| New scientific training actions | 0 | 0 |
| New research assessment actions | 0 | 0 |
| Sealed-final/unseen evaluations | 0 | 0 |


The user explicitly authorized unchanged legacy regressions to be separately ledgered. They were not excluded, modified or charged to the new Phase 5E fixture cap. Their actual operations/time are reported below, not hidden inside a scientific count. Failed-call attempts are distinguished from completed environment transitions.


| Legacy quantity | Actual |
| --- | --- |
| completed_environment_transitions | 967872 |
| completed_neural_optimizer_steps | 1908 |
| completed_linear_updates | 175 |
| failed_environment_step_calls | 11 |
| wall_seconds | 398.389 |


The new validation namespace is checked against 128 MiB; final delivered file sizes can be summed from the files named in output-hashes.json. Source/tests are small new additive files; old archives are referenced by hashes and commit rather than copied. The cumulative ledger includes failed development operations and provides no free retry.


## Preservation and verification


| Preservation field | Value |
| --- | --- |
| status | passed |
| parent_commit | fe200034e816b10352453471f9c680b7c77d6ca4 |
| parent_tree | b23e1f69573fcc92b24721ea9a9970cf43e8b8c6 |
| parent_files_checked | 10097 |
| original_files_checked | 9647 |
| main_head | cf7f0f11d3a94d8533c6a1ff7b4834f1db78f141 |
| historical_refs | {'main': 'cf7f0f11d3a94d8533c6a1ff7b4834f1db78f141', 'feature/phase5b-basic-navigation-diagnosis-v1': 'ae3ae73bf7ba5cbe36dcf879125f15fd14c43f4e', 'feature/phase5c-directional-coverage-v1': 'c1505cc1dab0623b9c7eb86b7bfae9ac3216fa4d', 'feature/phase5d-goal-credit-diagnosis-v1': 'fe200034e816b10352453471f9c680b7c77d6ca4'} |
| base_inventory_sha256 | bff9671ff36f9eddc7ac131a6a33812b535b0c5a100b70e5d2e65b0020bc5ab9 |
| original_inventory_sha256 | 63b1da65038d3915d845be09256598c4ddc69f29ffe220791c453520b9b10af2 |
| historical_data_parsing | none; byte checks only; no old study rerun |


All previous tracked files and experiment evidence are byte-identical to the parent inventory. Main HEAD, the original dirty file contents, expanded untracked status, staged index and historical branch refs match the pre-work snapshot. The old Phase 5D renderer and its separately documented delivery limitations are untouched.


Run from the application root using the installed interpreter and `PYTHONPATH=src`: `python -m robot_navigation.partial_phase5e verify` reads hashes/captured arrays only and performs no training, environment stepping or predictions. `validate` is exclusive and must not be rerun on an existing final namespace; all mechanical operations are cumulatively charged. Evidence includes attempt-plan/seal, source/library hashes, schema-bearing trace shards, independent witnesses, paired receipts, numeric audit, operation ledger, both JUnit/log processes, development failure history, preservation and output hashes.


## Exactly one proposed next study — not authorized or started


Separately register a Fixture A B1 DQN/PPO instrumentation pilot: seed 23121, 500 actions each, no evaluation or algorithm change. Requires new user authorization; not started. Its objective would be checking actual reward-to-replay/rollout-to-update evidence, not claiming that instrumentation cures navigation or proves generalization. No learning pilot or Phase 6 is started by this validation.



## Development corrections retained before final sealing


| Attempt | Actual result | Cause | Correction | Threshold changed |
| --- | --- | --- | --- | --- |
| 0001 | 12 passed, 19 setup errors | An unused linear action-space RNG was initialized from OS entropy during witness final-state collection. Actions and weights already matched. | Seed the synthetic action space with the registered fixture seed before learner construction. | False |
| 0002 | 2 setup errors | Witness retained the mutable environment info dictionary; SB3 later attached a terminal_observation ndarray, causing witness JSON serialization failure. | Copy info at the actual environment return before downstream VecEnv augmentation. | False |
| 0004 | 1 passed, 1 failed | A unit assertion demanded zero DQN scalar arithmetic error; independently promoted float64 calculation differed from actual float32 by 7.450581818169155e-11. | Apply the pre-existing preregistered numerical atol 2e-5; keep bitwise on/off equality at zero tolerance. | False |


## Validation runner continuation

The runner connection closed after the final 32-test Phase 5E suite passed, while the separate legacy regression process was active. No legacy JUnit or operation receipt survived that interrupted process, so its partial counts/time are explicitly not_observed; they are not included in the exact completed legacy-run counts above. The complete unchanged parent suite was subsequently run to completion using the separately registered continuation helper. The passed Phase 5E fixtures were not rerun, and the sealed source, captured tensors and cumulative new-mechanics ledger stayed unchanged. The exact total legacy activity including the interrupted process is unavailable. See runner-interruption.json, continuation-plan.json, continuation-verification.json and publication-delivery-plan.json in the evidence namespace. Both delivery helpers are separately source-hashed and make no scientific training, final evaluation or new Phase 5E mechanical calls.
