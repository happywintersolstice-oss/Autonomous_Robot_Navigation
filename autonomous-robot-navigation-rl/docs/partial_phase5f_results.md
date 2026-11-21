# Phase 5F actual instrumented learning signals

Verdict: **INSTRUMENTED_LEARNING_TRACES_VALIDATED**.

This is a measurement-validity pilot on real Fixture A training, not a navigation-performance comparison. No evaluation episodes, checkpoint selection, additional seeds, full Phase 5 or Phase 6 were run.

Parent: `192ba8dca9957ac6d6b38965956b06f05b694548`. Branch: `feature/phase5f-instrumented-learning-pilot-v1`. Frozen seed 23121; B1-relative-x4; original Phase 5B configuration and eight training cases.

## Actual runs

| Algorithm | Training actions | Actual optimizer steps | Recorded goal experiences | Goal sample draws | Distinct sampled goals | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 500 | 100 | 8 | 73 | 7 | 26.666 |
| PPO | 500 | 50 | 6 | 30 | 6 | 36.137 |

## DQN actual signal summary

5000 sample draws represent 465 distinct experiences. Repeated sample use is optimizer exposure, not additional environment actions.

Completed episodes: 22; resets: 23; trailing incomplete-episode actions: 3; empty final auto-reset: False. Raw outcomes: `{"collision": 14, "running": 478, "success": 8}`.

Observed quantities:

```json
{
  "actual_target": {
    "n": 5000,
    "min": -5.010000228881836,
    "max": 9.989999771118164,
    "mean": 0.047980596826970576,
    "std": 1.5214523694971622
  },
  "residual_snapshot_reconstructed": {
    "n": 5000,
    "min": -5.158275604248047,
    "max": 10.067606925964355,
    "mean": 0.0020788606509566307,
    "std": 1.5088294696750126
  },
  "goal_draw_frequencies": {
    "24": 14,
    "113": 21,
    "178": 17,
    "157": 11,
    "215": 8,
    "448": 1,
    "471": 1
  },
  "profile_counts": {
    "selected_actions": 500,
    "replay_sample_calls": 100,
    "replay_consumptions": 100,
    "target_forwards": 100,
    "online_forwards": 348,
    "native_optimizer_steps": 100
  }
}
```

Earliest observed goal-reward-to-update link (IDs and sequence point directly into the closed trace):

```json
{
  "transition_id": 24,
  "update_id": 1,
  "sample_row": 38,
  "trace_sequence": 215,
  "reward": 9.99,
  "actual_target": 9.989999771118164,
  "chosen_Q": 0.05085461586713791,
  "residual_snapshot": 9.9391450881958,
  "case_id": "phase5a-handwritten-fixtures-v1-A-train-07",
  "slot": 23,
  "generation": 1
}
```

Evidence: [dqn trace shards](../artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1/dqn/trace/), [run receipt](../artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1/dqn/run-receipt.json). Final provenance checkpoint SHA256: `65a882e94755126f94cf7967831fa0d7e90206698ce5c8dcc28b5e2e71c45fa5`. Trace-chain tail: `73f7d3878d73ca27415d4ee05ded60d0512a3944e061cf93fd4a6145b6ad8ffd`.

Independent checks: `{"DQN": 100, "PPO": 0, "GAE": 0, "linear": 0}`; every actual optimizer operation has one ledger debit, completion and observer record. Required missing fields: `[]`.

## PPO actual signal summary

2500 sample draws represent 500 distinct experiences. Repeated sample use is optimizer exposure, not additional environment actions.

Completed episodes: 17; resets: 18; trailing incomplete-episode actions: 47; empty final auto-reset: False. Raw outcomes: `{"collision": 11, "running": 483, "success": 6}`.

Observed quantities:

```json
{
  "advantages": {
    "n": 500,
    "min": -6.727189064025879,
    "max": 9.91561222076416,
    "mean": -0.7329875491522253,
    "std": 3.305452266741845
  },
  "returns": {
    "n": 500,
    "min": -5.010000228881836,
    "max": 9.989999771118164,
    "mean": 0.15480442389845847,
    "std": 3.215131235929222
  },
  "timeout_bootstraps": 0,
  "profile_counts": {
    "actor_forwards": 500,
    "critic_forwards": 5,
    "rollout_consumptions": 50,
    "rollout_permutations": 25,
    "policy_training_evaluations": 50,
    "native_optimizer_steps": 50
  },
  "Q_values": "not_applicable"
}
```

Earliest observed goal-reward-to-update link (IDs and sequence point directly into the closed trace):

```json
{
  "transition_id": 8,
  "rollout_id": 1,
  "row": 7,
  "epoch": 0,
  "minibatch_id": 1,
  "update_id": 1,
  "trace_sequence": 313,
  "case_id": "phase5a-handwritten-fixtures-v1-A-train-06",
  "raw_reward": 9.99,
  "raw_advantage": 9.91561222076416,
  "normalized_advantage": 1.618405818939209,
  "return": 9.989999771118164,
  "collection_action_probability": 0.19981640577316284,
  "actual_update_action_probability": 0.19981642067432404
}
```

Evidence: [ppo trace shards](../artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1/ppo/trace/), [run receipt](../artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1/ppo/run-receipt.json). Final provenance checkpoint SHA256: `999690da810a35c6e18410ff9e4b6b090b5ff0737770f337cbf447ea59ddfb0c`. Trace-chain tail: `e17e922f6739ac2c895a435787928a87a3bad2017c849453d36862e04825a7bf`.

Independent checks: `{"DQN": 0, "PPO": 50, "GAE": 5, "linear": 0}`; every actual optimizer operation has one ledger debit, completion and observer record. Required missing fields: `[]`.

## Validation and limits

Unchanged parent regression tests: `{"tests": 464, "failures": 0, "errors": 0, "skipped": 0}`; elapsed 492.810 seconds. Parent process: `{"completed_environment_transitions": 967872, "completed_neural_optimizer_steps": 1908, "completed_linear_updates": 175, "failed_environment_step_calls": 11}`. Child E regression mechanics: `{"transitions": 480, "optimizer_steps": 108, "wall_seconds": 0}`. Combined legacy environment transitions: 968352; combined optimizer and partial-linear steps: 2191. These are legacy fixtures, not F scientific learning or new F synthetic validation.

Targeted post-pilot tests: `{"tests": 3, "failures": 0, "errors": 0, "skipped": 0}`. Every test attempt, stdout/stderr and JUnit receipt is retained under tests/. New F synthetic physical transitions/optimizer steps: **0 / 0**, against preregistered 0 / 0 caps. Pretests use fake journal reservations; posttests read actual pilot traces.

The original E read-only verifier initially rejected the newly added untracked F instruction file in the original workspace. Its scoped pass used a process-local Git exclude for that one file, without changing old code, settings or evidence. The initial failure, exact diff and scoped pass are retained in parent-preflight.json. The separate F preservation inventory includes the full current original workspace, including that file.

Current evidence bytes before this report/output inventory: 21137948; hard cap 536870912. Conservative preregistered uncompressed upper estimate: 386556800. Actual pilot runtime: 63.023511399995186 seconds; hard caps 900 per arm /1800 combined. No uninstrumented matched research run exists, so additional live instrumentation overhead and live bitwise on/off equivalence are **not_observed**. E's synthetic on/off equivalence remains inherited evidence only.

DQN targets and losses are actual consumed optimizer tensors; reported residuals are snapshot_reconstructed algebra on those same pre-update tensors, not later checkpoint inference. PPO reports probabilities, log-probabilities, V, GAE and returns, with Q-values not_applicable. Raw environment goal rewards are separated from timeout bootstrap additions. Observed reward use does not prove causal credit assignment or navigation improvement. A 500-action arm may contain no goal event; missing examples are reported as not_observed, never synthesized.

All historical source/evidence and original dirty-main files are byte-checked by preservation.json. No historical trial was rewritten. Source/config/library provenance, closed trace shard hashes, checkpoints and tests are indexed in output-hashes.json. Run `python -m robot_navigation.partial_phase5f verify` from this application with PYTHONPATH=src for read-only verification. The one-shot run command cannot be retried after run.lock exists.

Stopped after Phase 5F. No further learning/evaluation is authorized.
