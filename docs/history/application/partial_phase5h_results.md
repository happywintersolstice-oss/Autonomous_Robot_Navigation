# Phase 5H controlled goal-conditioning result

Verdict: **INCONCLUSIVE_SMALL_PILOT**.

Fresh DQN control H0 versus DQN H1 with lambda0.05 auxiliary direction cross-entropy, seed23141, same original B1 architecture and eight training cases. These are repeatedly used static development panels, not untouched validation/generalization evidence.

| Arm | Training actions | Checkpoint | Original success /4 | Expanded success /16 | U/D/L/R /4 | Expanded first alignment /16 | Wait / actions | Timeouts / collisions |
|---|---:|---:|---:|---:|---|---:|---|---|
| H0 | 0 | 0 | 1 | 4 | 0/4/0/0 | 4 | 0/95 | 0 / 15 |
| H0 | 2000 | 2000 | 1 | 9 | 2/3/2/2 | 6 | 626/2047 | 10 / 0 |
| H1 | 0 | 0 | 1 | 4 | 0/4/0/0 | 4 | 0/95 | 0 / 15 |
| H1 | 2000 | 2000 | 0 | 7 | 0/0/4/3 | 7 | 169/2612 | 13 / 0 |

Difference-in-differences of expanded success gains: **-2** cases; descriptive one seed only.

Fixed promising gates:

```json
{
  "at_least_12": false,
  "each_direction_2": false,
  "success_advantage_3": false,
  "alignment_advantage_3": false,
  "wait_within_10_points": true,
  "no_new_complete_direction_collapse": true
}
```

## What changed

H1 adds a Linear32-to-4 head from the second shared ReLU. Both shared Linear layers receive TD plus auxiliary gradient; the original five-action Q output receives TD only, and the extra projection receives auxiliary only. The original target Q network and its hard-copy schedule are unchanged. Original Adam has the head appended to its existing parameter group; the original single backward/update and policy-wide norm10 clipping handle the joint objective. The native DQN.train body is not rewritten. The auxiliary head never chooses or masks actions.

Direction labels come exclusively from the current public goal-position delta, largest-magnitude component, vertical-first ties; exact-goal samples are excluded. A correct coarse label is not an obstacle-aware optimal-action oracle. The task is potentially trivial; replay classification accuracy alone is not evidence of navigation improvement. This is not a parameter-count-controlled comparison.

## Actual learning evidence

| Arm | Goal encounters | Distinct goals sampled | Goal draws | Updates using goals | Optimizer steps | Training replay auxiliary accuracy |
|---|---:|---:|---:|---:|---:|---|
| H0 | 60 | 55 | 686 | 371 | 475 | not_applicable |
| H1 | 52 | 41 | 362 | 254 | 475 | 14884/23750 |

Replay accuracy counts repeated optimizer draws, not distinct environments or development labels. All475 updates per arm record compact actual chosen Q/target-max/reward/mask/TD-target vectors, sample IDs/hashes, component losses/logits, gradient norms and optimizer/network fingerprints. Independent TD/Huber/cross-entropy/total arithmetic and replay joins were checked once; no later checkpoint inference substitutes for missing tensors. [Optimizer summaries](../artifacts/partial-dynamic-navigation-v1/phase5h-fast-goal-conditioning-v1/optimizer-summary.json), [all checkpoint metrics](../artifacts/partial-dynamic-navigation-v1/phase5h-fast-goal-conditioning-v1/checkpoint-metrics.csv), [all 80 case rows and first Q vectors](../artifacts/partial-dynamic-navigation-v1/phase5h-fast-goal-conditioning-v1/case-results.csv). Full action paths remain in cases-H0/H1.json and compact assessment ledgers.

## Pairing, resources and verification scope

```json
{
  "training_actions": 4000,
  "optimizer_steps": 950,
  "assessment_actions": 4849,
  "assessment_episodes": 80,
  "training_seconds": {
    "H0": 20.279585700001917,
    "H1": 18.179892200001632
  },
  "assessment_seconds": 13.64602630000445,
  "total_run_seconds": 52.40009139999165,
  "assessment_parent_isolation": {
    "before": {
      "rng": {
        "python": "b47d48e15000575dd404c09102033c5c3af6b0fcb7dc6f072f06f9934f80298c",
        "numpy": "c570f2d00b9b9f7a909ad1c1e7b9797c074eeb3efcd8c74858f063114de540b6",
        "torch": "0f86ac75e4276b805957385dc4a5557326b77775036695ec29d8ef55410c54c1"
      },
      "training_hashes": {
        "train-H0.jsonl.gz": "77fe85fba7fe38eea54c6d60a8b58d7d3edc6a8fb5dd48300cde19806ad03c21",
        "train-H1.jsonl.gz": "ec4d9754eced613b72e3ed8e360814d24289303f42296b66fe5d1e828bd938d2"
      }
    },
    "after": {
      "rng": {
        "python": "b47d48e15000575dd404c09102033c5c3af6b0fcb7dc6f072f06f9934f80298c",
        "numpy": "c570f2d00b9b9f7a909ad1c1e7b9797c074eeb3efcd8c74858f063114de540b6",
        "torch": "0f86ac75e4276b805957385dc4a5557326b77775036695ec29d8ef55410c54c1"
      },
      "training_hashes": {
        "train-H0.jsonl.gz": "77fe85fba7fe38eea54c6d60a8b58d7d3edc6a8fb5dd48300cde19806ad03c21",
        "train-H1.jsonl.gz": "ec4d9754eced613b72e3ed8e360814d24289303f42296b66fe5d1e828bd938d2"
      }
    }
  },
  "mechanical": {
    "transitions": 20,
    "optimizer_steps": 6,
    "scope": "one H1-only 20-step smoke plus two fixed-batch lambda0 updates; no research",
    "lambda0_Q_update_match": true,
    "initial_common_Q_equal": true,
    "same_fixed_batch_ids": true,
    "aux_gradient_to_shared_and_head_only": true,
    "Q_only_action_selection": true,
    "parameter_counts": {
      "H0": {
        "online_q": 14437,
        "target_q": 14437,
        "auxiliary": 0,
        "optimizer_parameters": 14437
      },
      "H1": {
        "online_q": 14437,
        "target_q": 14437,
        "auxiliary": 132,
        "optimizer_parameters": 14569
      }
    }
  },
  "full_regression_suite": "DEFERRED_TO_MAJOR_MILESTONE"
}
```

Initial online/target Q equality and initial global/case/action RNG equality:

```json
{
  "common_initial_equal": true,
  "initial_rng_equal": true
}
```

Measured parameter counts and native optimizer/replay counters:

```json
{
  "H0": {
    "actions": 2000,
    "optimizer_steps": 475,
    "native_calls": {
      "native_optimizer": 475,
      "native_replay_consumption": 475
    },
    "wall_seconds": 20.279585700001917,
    "parameter_counts": {
      "online_q": 14437,
      "target_q": 14437,
      "auxiliary": 0,
      "optimizer_parameters": 14437
    }
  },
  "H1": {
    "actions": 2000,
    "optimizer_steps": 475,
    "native_calls": {
      "native_optimizer": 475,
      "native_replay_consumption": 475
    },
    "wall_seconds": 18.179892200001632,
    "parameter_counts": {
      "online_q": 14437,
      "target_q": 14437,
      "auxiliary": 132,
      "optimizer_parameters": 14569
    }
  }
}
```

Full legacy regression suite: **DEFERRED_TO_MAJOR_MILESTONE**. No claim that479 inherited tests passed in this phase. Focused tests and one isolated20-transition/six-optimizer-operation mechanical validation were completed before research, including the lambda0 fixed-minibatch comparison. No additional synthetic run or post-study test repeat was performed. Specific direct source/config hashes and inherited G verification/preservation receipts replaced a new 10k-file byte audit. The working-tree/commit scope is additions under H paths only; original dirty main HEAD/status/index are checked.

Not claimed here: G-level full gradient tensors, every five-Q training action vector, comprehensive forward/RNG-call census, live bitwise instrumentation equivalence, causal mechanism, statistical significance, generalization or a four-model ranking. Same-seed pairing fixes common initialization and initial RNG streams; trajectories need not stay identical after actions diverge. Parent training RNG/evidence and child weights/optimizer/RNG stayed unchanged by detached inference.

No extra seed, tuning, warm start, behavior cloning, reward shaping, action masking, planner fallback, supervised pretraining, reserved/final/unseen/dynamic task, full four-model training or Phase6. Stop after H.

Measured interpretation: H1 finished at 7/16 expanded successes versus H0 9/16, a difference of -2 cases and difference-in-differences -2. Expanded first alignment was 7/16 versus 6/16, below the required +3 advantage. H1 direction successes were {'Up': 0, 'Down': 0, 'Left': 4, 'Right': 3} versus H0 {'Up': 2, 'Down': 3, 'Left': 2, 'Right': 2}; Up and Down were lost despite the training-only auxiliary accuracy of 62.67% (14884/23750 repeated replay draws). This is not evidence of robust improved navigation.

Wait fraction fell from 30.58% to 6.47%, while timeouts rose from 10/20 to 13/20 and ABAB loop windows from 859 to 2111. Low Wait use therefore did not remove failure: moving revisits were 1289 versus 2404. The auxiliary arm is not promising under the frozen criteria and should not replace the control based on this result. All eight focused tests passed; the full legacy suite remains deferred.
