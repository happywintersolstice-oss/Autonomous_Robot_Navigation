Each Part B refinement vs its Part A starting point (test maps, nominal). p < 0.05 = difference unlikely to be chance.

Note: runs trained with `env_overrides` (e.g. ppo_drop_* with 20% LiDAR dropout) are also TESTED with those settings, so compare them with each other, or see the robustness table for the baseline under the same condition.

| refined | baseline | success_baseline | success_refined | delta_success | delta_collision | p_value |
|---|---|---|---|---|---|---|
| td3_long | td3 | 0.64 | 0.913 | 0.273 | -0.273 | 0.0 |
| td3_tuned | td3 | 0.64 | 0.833 | 0.193 | -0.247 | 0.0 |
| sac_long | sac | 0.8 | 0.953 | 0.153 | -0.14 | 0.0 |
| q_learning_tuned | q_learning | 0.153 | 0.267 | 0.113 | 0.013 | 0.021 |
| ppo_tuned | ppo | 0.54 | 0.54 | 0.0 | 0.02 | 1.0 |
| dueling_dqn_tuned | dueling_dqn | 0.633 | 0.573 | -0.06 | 0.02 | 0.343 |
| dueling_dqn_curriculum | dueling_dqn | 0.633 | 0.573 | -0.06 | 0.06 | 0.338 |
| ppo_drop_maxrange | ppo | 0.54 | 0.473 | -0.067 | 0.093 | 0.293 |
| ppo_drop_lastvalid | ppo | 0.54 | 0.467 | -0.073 | 0.08 | 0.258 |
| q_learning_noroute | q_learning | 0.153 | 0.0 | -0.153 | -0.153 | 0.0 |
| ppo_drop_none | ppo | 0.54 | 0.307 | -0.233 | 0.26 | 0.0 |
| ppo_drop_stack3 | ppo | 0.54 | 0.187 | -0.353 | 0.367 | 0.0 |
| ppo_euclidean | ppo | 0.54 | 0.1 | -0.44 | -0.053 | 0.0 |
| dueling_dqn_noroute | dueling_dqn | 0.633 | 0.027 | -0.607 | -0.08 | 0.0 |
| sac_noroute | sac | 0.8 | 0.013 | -0.787 | -0.173 | 0.0 |
