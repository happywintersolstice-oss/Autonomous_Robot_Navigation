# Phase 5E instrumentation protocol

Parent fe200034e816b10352453471f9c680b7c77d6ca4; branch
feature/phase5e-training-instrumentation-v1. Engineering only. Scientific actions,
research assessment/navigation episodes, sealed-final/unseen evaluations: zero.
The supplied root Phase 5E brief is authoritative. No history or Phase 5D
renderer repair. Production task, architectures, initialization, features,
rewards, hyperparameters, optimizers and sampling remain unchanged.

## Registered execution and limits

`python -m robot_navigation.partial_phase5e validate --development` may run a
hash-sealed development attempt, preserving failed receipts. A final `validate`
seals its exact source before running the synthetic suite once and refuses a
retry. `audit` processes existing captured numbers and produces the report;
`verify` is read-only. Neither audit nor verify loads a learner or steps an env.
Every development/final attempt shares an append-only operation ledger. The
limits are 1,000 new Phase 5E synthetic transitions, 500 new mechanical optimizer
steps, 1,200 cumulative seconds of active Phase 5E synthetic validation, and
128 MiB of new validation evidence. Reservations occur before each operation.
No reset, deleted failure or fresh namespace grants a free retry. Failed final
validation yields INCOMPLETE_VALIDATION or INSTRUMENTATION_BLOCKED.

The user explicitly clarified that the complete **unchanged legacy regression
suite** runs separately: count its transitions, optimizer operations, test
results and time in a legacy ledger. Those operations do not consume the new
Phase 5E mechanical caps. Do not omit or modify an inherited test to meet caps.
The final validation runs all test files recorded by the parent Git tree and
the new focused Phase 5E suite as separate processes, avoiding an unintended
second execution of Phase 5E fixtures. Historical SHA checks read bytes only;
no held-out observations are selected, parsed, scored or evaluated.

## Installed implementation and compatibility

Pin exact installed versions and SHA256 source for the methods listed in
partial_phase5e_library_v1.json. Audit full installed call-path snippets, source
files, Torch/NumPy compiled-module origins and Gymnasium entry points. The
supported path is CPU SB3 2.9.2, Torch 2.15.0, NumPy 2.5.4, Gymnasium 1.4.0,
409 float32 raw observations, 412 B0/B1 features, Discrete(5), one synchronous
environment, original standard one-step replay. Reject unsupported versions,
source changes, prefilled/resumed replay without identities, VecNormalize,
memory-optimized replay, SDE and an existing debugger trace. The production
models already use this path. Explicit unsupported conditions are not silently
approximated or claimed as validated.

## Passive hooks and actual-value schema

`observe(policy, sink=None)` is a genuine no-op. An explicit sink installs
per-thread Python line/return observation on pinned real functions and wraps the
existing optimizer method exactly once. ExitStack restores every hook and tensor
reduction wrapper on success or exception. No observer invokes a model forward,
sampler, random generator, environment transition or replacement learner update.

DQN observes ReplayBuffer.add at its real slot, then the real _get_samples
return frame containing the original batch_inds and env_indices. A sidecar
keys transition ID by slot/environment/generation. Preserve duplicates/order;
never deduplicate or infer identity from repeated observations. At the real
train zero_grad line capture replay_data, discounts, current chosen Q,
next-target max Q, **actual target tensor** and actual Huber mean loss, all
before update. Capture real post-library-clipping gradients, before/after
network/optimizer fingerprints, update magnitude and learning rates at the one
original optimizer step. Observe target synchronization at its actual scheduled
_on_step call and record epsilon/counters. Raw terminated/truncated and stored
timeouts are distinct from consumed dones*(1-timeouts).

PPO observes actual ActorCriticPolicy.forward return probabilities, actions,
V and log probability without another prediction. Record raw env rewards before
collector adjustment and RolloutBuffer.add rewards afterwards. Capture each
actual timeout terminal_value immediately before reward augmentation. Observe
the compute_returns_and_advantage call's exact last_values/dones **before**
any optimizer update, then its actual values/advantages/returns. Preserve row IDs.
Observe the actual permutation after its one generation, and actual sampled
indices before each minibatch consumption. Capture the real normalized advantage
and actual normalization mean/std reduction outputs; the original Tensor methods
are called once and restored. Before the real KL check capture ratios, unclipped
and clipped objectives, value predictions, entropy/value/policy/total loss, KL,
clip fraction and the actual early-stop decision. PPO has no Q values. Default
production target_kl is None; a missing event is not an observed early stop.

Linear adapters observe the existing normalized semi-gradient methods, without
replacing their update rule. Record real target/error, phi, alpha/gamma,
chosen action, sampled next SARSA action and before/after weights. Prior Q is
exact algebra on the captured live pre-update weights/features, explicitly
distinct from a separate model prediction. No max substitution for SARSA.

Schema partial-learning-live-v1 carries study/run/algorithm/arm/seed/source,
config/dependency hashes, environment/reset/case/transition IDs, replay
slot/generation, rollout/epoch/minibatch/update IDs and network fingerprints.
Each tensor carries shape, dtype, byte hash and finite checks. Record order and
record checksums are mandatory. Source kinds are live_actual, historical_logged,
snapshot_reconstructed, synthetic_mechanical, not_observed, not_applicable.
Actual consumed computations are live_actual, with synthetic_mechanical as their
validation classification. Absent case IDs require an explicit absence reason.
Hidden maps and future event schedules are not read or supplied to a model.
Use exclusive compressed shards, conservative byte reservations, and reject
missing completion records, corrupt checksums, nonfinite or reordered payloads.

## Fixed mechanical fixtures and equivalence

Seed 23101, CPU one thread, unchanged model config from Phase 5A. Four families
and both feature arms; each off/on member is a fresh matched subprocess. A fixed
synthetic stream repeats a legal 409-value reset observation deliberately, with
cyclic continuing/goal/collision/timeout/continuing reward/flag records. Actions
are recorded but cannot change this artificial stream. This is not a navigable
research task or evidence of better goal performance.

Linear members execute 8 mechanical transitions each. DQN members collect four
transitions below the original 100-step warmup, then explicitly invoke one real
DQN.train update as a unit fixture (50 sampled rows, replacement/duplicates).
Call the unchanged target-sync callback up to its genuine 250th call, with no
additional environment or optimizer operation. PPO members collect the original
100-step rollout and execute its original five epochs/two batches. Total paired
fixture cost: 480 transitions, 108 updates. Production hyperparameters remain
unchanged; only the finite test harness invokes the isolated DQN unit update.

An independent always-on profiling witness runs in both members, separate from
the optional logger's line hooks. It captures executed actions/probabilities,
reset/reward/flag sequence, actual sample order and tensors, optimizer losses,
clipped gradients, forward call counts, RNG and complete model/target/optimizer/
buffer fingerprints. Require exact CPU equality (tolerance zero) of the witness
records/final states. Logger on must not add a forward, sample or random draw.
Checkpoint save/load is tested without environment execution and with RNG saved.
After the normal configured PPO run only, both mechanical members set their
disposable test instance's target_kl to 1e-12 and call the existing train method
on its already collected rollout. This declared fault/branch fixture tests an
actual early stop, with zero added environment transitions and no expected
optimizer update. It is not a production hyperparameter change or experiment;
no config, historical checkpoint or future training setting is edited. Preserve
its actual loss/KL and independently witnessed stop even when no step occurs.

For independent numerical validation use only the actual captured tensors:
DQN r+gamma*(1-consumed_done)*target_max and Huber loss; PPO backwards GAE with
actual pre-update bootstrap, returns=advantages+V, sample normalization (sample
std, ddof=1), ratios, clipping, entropy/value/policy/total losses and KL. Numeric
kernel agreement tolerance is atol=2e-5, rtol=2e-6 (float32 accumulation), linear
algebra atol=1e-10. This tolerance never applies to on/off equality or sampled
operations. No threshold changes after outcomes.

Additional zero-environment fixtures test replay ring overwrite/generations,
replacement order, terminal/truncation algebra, malformed/duplicate identities,
network-version mismatch, missing GAE/bootstrap/rollout IDs, nonfinite tensors,
checksum corruption, interrupted shards, quota exhaustion, hook restoration,
disabled no-op behavior, sensor/action ordering, feature/checkpoint contracts
and forbidden final/unseen paths. These injected failures must be rejected,
with genuine partial traces retained. They grant no scientific training budget.

## Gates and terminal interpretation

G0 parent/Git/historical preservation and no scientific/final activity; G1 actual
signals and identities; G2 bitwise noninterference; G3 independent arithmetic;
G4 cumulative budgets and failure handling; G5 unchanged complete legacy suite,
focused tests, source/manifest pins, honest failure history and read-only verify.
Only all gates allow INSTRUMENTATION_VALIDATED_NO_RESEARCH_TRAINING. Otherwise
INSTRUMENTATION_BLOCKED or INCOMPLETE_VALIDATION. No partial pass is called
validated. No new Phase 5D renderer mutation, research pilot, final evaluation or
Phase 6. A later small registered Fixture A pilot is a proposal requiring new
authorization, never an automatic next action.
