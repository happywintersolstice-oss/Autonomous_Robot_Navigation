Each Part B refinement vs its Part A starting point (test maps, nominal). p < 0.05 = difference unlikely to be chance.

Note: runs trained with `env_overrides` (e.g. ppo_drop_* with 20% LiDAR dropout) are also TESTED with those settings, so compare them with each other, or see the robustness table for the baseline under the same condition.

| refined | baseline | success_baseline | success_refined | delta_success | delta_collision | p_value |
|---|---|---|---|---|---|---|
| dueling_dqn_hybrid | dueling_dqn | 0.547 | 0.687 | 0.14 | -0.027 | 0.018 |
| sarsa_hybrid | sarsa | 0.387 | 0.52 | 0.133 | 0.12 | 0.027 |
| ppo_hybrid | ppo | 0.487 | 0.607 | 0.12 | -0.033 | 0.047 |
| q_learning_hybrid | q_learning | 0.353 | 0.46 | 0.107 | 0.073 | 0.078 |
| dueling_dqn_tuned | dueling_dqn | 0.547 | 0.633 | 0.087 | -0.1 | 0.149 |
| sac_hybrid | sac | 0.733 | 0.82 | 0.087 | 0.06 | 0.099 |
| sac_long | sac | 0.733 | 0.813 | 0.08 | -0.033 | 0.13 |
| dueling_dqn_curriculum | dueling_dqn | 0.547 | 0.607 | 0.06 | -0.073 | 0.349 |
| ppo_tuned | ppo | 0.487 | 0.527 | 0.04 | -0.08 | 0.56 |
| td3_tuned | td3 | 0.667 | 0.673 | 0.007 | 0.047 | 1.0 |
| q_learning_tuned | q_learning | 0.353 | 0.36 | 0.007 | 0.047 | 1.0 |
| ppo_drop_lastvalid | ppo | 0.487 | 0.487 | 0.0 | -0.02 | 1.0 |
| ppo_drop_maxrange | ppo | 0.487 | 0.487 | 0.0 | 0.047 | 1.0 |
| td3_hybrid | td3 | 0.667 | 0.66 | -0.007 | -0.053 | 1.0 |
| ppo_euclidean | ppo | 0.487 | 0.447 | -0.04 | -0.067 | 0.577 |
| ppo_drop_stack3 | ppo | 0.487 | 0.4 | -0.087 | 0.167 | 0.167 |
| ppo_drop_none | ppo | 0.487 | 0.393 | -0.093 | 0.18 | 0.134 |
