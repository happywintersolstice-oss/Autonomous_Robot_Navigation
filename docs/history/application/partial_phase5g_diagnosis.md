# Phase 5G diagnosis

Engineering verdict: **PHASE5G_MEASUREMENT_VALIDATED**.

Measured readiness:

```json
{
  "dqn": {
    "verdict": "BASIC_NAVIGATION_SCREEN_NOT_MET",
    "checks": {
      "expanded_gain_at_least_4": true,
      "expanded_at_least_12": false,
      "each_direction_at_least_2": false
    },
    "expanded_gain": 5
  },
  "ppo": {
    "verdict": "BASIC_NAVIGATION_SCREEN_NOT_MET",
    "checks": {
      "expanded_gain_at_least_4": false,
      "expanded_at_least_12": false,
      "each_direction_at_least_2": false
    },
    "expanded_gain": 1
  }
}
```

Reward encounters, distinct sampled experiences, repeated goal draws, updates consuming them and deterministic development successes have separate denominators. Complete traces establish that measured signals reached actual optimization; they do not establish that a particular signal caused goal-directed behavior. All checkpoint outcomes, including reversals and directional failures, remain in the results.

The unchanged representation, reward, uniform sampler and learner settings were held fixed. There was no hyperparameter search, action masking, planner fallback, auxiliary objective, checkpoint selection, dynamic training or generalization evaluation. A fresh 2000-action DQN schedule is not a continuation/control-equivalent extension of F.

One specific future intervention, **not implemented or authorized**: preregister a bounded comparison adding an auxiliary four-way goal-direction prediction loss from the already public relative-goal vector, while fixing reward, sampler and training budget. Its purpose would be to test whether explicitly representing goal direction changes first-action alignment after optimizer reward exposure; these associations do not establish that weak goal conditioning is the cause or that the intervention will work. Require separate authorization and a sealed control/intervention design before execution.

Stopped after Phase5G. No full Phase5, final/unseen evaluation or Phase6 was started.
