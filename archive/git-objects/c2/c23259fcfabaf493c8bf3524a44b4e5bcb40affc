# Evaluation boundary and unseen-map handoff

## Owner decision on 9 November 2026

After reviewing the fixed-map coverage, the owner chose to defer independent
four-algorithm evaluation to the upcoming unseen-map study. The current stage
consolidates the existing evidence and commits the verified implementation.
The independent evaluation remains a next-stage deliverable.

## Confirmed fixed-map coverage

The connected benchmark has 79 free cells and 79×78 = 6,162 possible distinct
ordered start–goal pairs. Saved manifests contain 4,306 training, 928 validation
and 928 test pairs. Their union covers all 6,162 pairs, with disjoint split IDs.

The original frozen release already evaluated its test pairs. PPO and later
neural tuning followed that release. The four-family comparison, representation
study and staged five-seed neural confirmations use development validation.
Changing training seeds or repartitioning the same fixed-map pairs cannot
establish a previously untouched task-space holdout.

The original final record remains historical evidence. The current stage records
zero new canonical benchmark training or final-test policy executions. The
selected controller remains m4-tuned-q-learning-seed101. Regression tests can
exercise training and release logic on isolated fixtures.

## Required decisions before the unseen-map implementation

1. Define the question and policy information. Distinguish zero-shot application
   of current fixed-map policies from training controllers across layouts.
   Record a common information contract for Q-learning, SARSA, DQN and PPO.
   Their current coordinate-only inputs and fixed-map compatibility checks
   require an explicit new-task design for layout transfer.
2. Establish a separate versioned benchmark. Declare layout generation,
   reachability/difficulty checks, duplicate detection and train/validation/final
   layout separation before model selection. Preserve the existing benchmark.
3. Declare budgets and seed handling. Validate compute/storage on a pilot, then
   freeze shared training budgets, training seeds, monitor cases, candidate
   opportunities and the checkpoint/configuration selection rule. Retain poor
   seeds and incomplete attempts.
4. Protect final layout access. Keep test layouts and their outcomes outside
   development selection. Freeze models, task/observation/action/reward versions,
   analysis, references and input hashes before the dedicated final runner.
5. Define analysis and acceptance. Include Random/BFS references, per-map and
   per-training-seed support, outcome rates, SPL, return, conditional route
   lengths and recorded costs. Treat repeated routes as dependent within a
   trained policy/layout. Declare the aggregation and uncertainty units in
   advance. Report failures and missed targets.
6. Audit execution and presentation. Verify inference fingerprints, every saved
   transition, scenario/model identity, split integrity, reference optimality and
   budget accounting. Add compatible unseen-map replay with visible task/model
   provenance and test it through the normal UI entry points.

These are handoff requirements. Layout counts, seeds, representation and runtime
limits remain to be declared for the next study. This document creates no maps,
models, new split or claimed unseen-map outcome. Dynamics noise, moving obstacles,
continuous control, sensors, ROS and hardware require their own scoped stages.

## Existing evidence to carry forward

- [Consolidated technical report](technical_report.md).
- [Five-seed results and recovery costs](neural_confirmation_results.md) and
  [verification](neural_confirmation_verification.json).
- [Current scope and completion record](plan_closure_status.md).
- [Original frozen final summary](../artifacts/studies/final-v1/summary.json).
