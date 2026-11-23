# Fact sheet: Member 3

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## ppo_drop_lastvalid

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [1122.9, 673.0, 1079.5]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'dropout_prob': 0.2}` / `{'impute': 'last_valid'}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 26 |
| Success rate | 0.487 (95% CI 0.393–0.573) |
| Collision rate | 0.393 (95% CI 0.313–0.480) |
| Collision with a person | 0.093 |
| Timeout rate | 0.120 |
| Path efficiency vs A* (successes only) | 0.954 (95% CI 0.914–0.991) |
| Time to goal (s, successes only) | 15.0 |
| Share of steps inside a person's comfort zone | 0.147 |
| Return: mean / IQM | 2.40 / 3.10 |
| Success std across seeds (stability) | 0.047 |
| Decision time (ms) | 0.75 |

- How episodes end: {'success': 0.487, 'collision_wall': 0.3, 'too_slow': 0.093, 'collision_human': 0.093, 'stuck_oscillating': 0.027}

- Success under stress conditions: nominal 0.487, bigger_map 0.527, careless_people 0.427, crowded 0.433, fast_people 0.473, hardest_level 0.147, lidar_dropout_20 0.487, lidar_dropout_40 0.52, people_start_late 0.56

- Sample efficiency (area under validation curve, 0–1) per seed: [0.52, 0.466, 0.43]

- Your figures: `importance_ppo_drop_lastvalid.png`, `saliency_ppo_drop_lastvalid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_drop_maxrange

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [1128.4, 1077.1, 964.1]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'dropout_prob': 0.2}` / `{'impute': 'max_range'}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 25 |
| Success rate | 0.487 (95% CI 0.407–0.567) |
| Collision rate | 0.460 (95% CI 0.380–0.540) |
| Collision with a person | 0.140 |
| Timeout rate | 0.053 |
| Path efficiency vs A* (successes only) | 0.942 (95% CI 0.887–0.982) |
| Time to goal (s, successes only) | 14.8 |
| Share of steps inside a person's comfort zone | 0.129 |
| Return: mean / IQM | 2.50 / 2.80 |
| Success std across seeds (stability) | 0.019 |
| Decision time (ms) | 0.77 |

- How episodes end: {'success': 0.487, 'collision_wall': 0.32, 'collision_human': 0.14, 'too_slow': 0.047, 'stuck_oscillating': 0.007}

- Success under stress conditions: nominal 0.487, bigger_map 0.593, careless_people 0.453, crowded 0.373, fast_people 0.473, hardest_level 0.113, lidar_dropout_20 0.487, lidar_dropout_40 0.447, people_start_late 0.547

- Sample efficiency (area under validation curve, 0–1) per seed: [0.548, 0.525, 0.464]

- Your figures: `importance_ppo_drop_maxrange.png`, `saliency_ppo_drop_maxrange.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_drop_none

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [904.6, 1067.4, 978.6]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'dropout_prob': 0.2}` / `{'impute': 'none'}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 31 |
| Success rate | 0.393 (95% CI 0.313–0.480) |
| Collision rate | 0.593 (95% CI 0.500–0.680) |
| Collision with a person | 0.120 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.926 (95% CI 0.863–0.976) |
| Time to goal (s, successes only) | 15.1 |
| Share of steps inside a person's comfort zone | 0.134 |
| Return: mean / IQM | 0.21 / -1.51 |
| Success std across seeds (stability) | 0.025 |
| Decision time (ms) | 0.76 |

- How episodes end: {'collision_wall': 0.473, 'success': 0.393, 'collision_human': 0.12, 'too_slow': 0.013}

- Success under stress conditions: nominal 0.393, bigger_map 0.467, careless_people 0.36, crowded 0.247, fast_people 0.353, hardest_level 0.093, lidar_dropout_20 0.393, lidar_dropout_40 0.36, people_start_late 0.413

- Sample efficiency (area under validation curve, 0–1) per seed: [0.38, 0.429, 0.363]

- Your figures: `importance_ppo_drop_none.png`, `saliency_ppo_drop_none.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_drop_stack3

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [1100.0, 1155.9, 1024.1]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'dropout_prob': 0.2}` / `{'impute': 'none', 'frame_stack': 3}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 30 |
| Success rate | 0.400 (95% CI 0.307–0.493) |
| Collision rate | 0.580 (95% CI 0.473–0.680) |
| Collision with a person | 0.120 |
| Timeout rate | 0.020 |
| Path efficiency vs A* (successes only) | 0.926 (95% CI 0.877–0.971) |
| Time to goal (s, successes only) | 15.3 |
| Share of steps inside a person's comfort zone | 0.163 |
| Return: mean / IQM | -0.01 / -1.66 |
| Success std across seeds (stability) | 0.043 |
| Decision time (ms) | 0.78 |

- How episodes end: {'collision_wall': 0.46, 'success': 0.4, 'collision_human': 0.12, 'too_slow': 0.02}

- Success under stress conditions: nominal 0.4, bigger_map 0.427, careless_people 0.38, crowded 0.207, fast_people 0.367, hardest_level 0.107, lidar_dropout_20 0.4, lidar_dropout_40 0.353, people_start_late 0.387

- Sample efficiency (area under validation curve, 0–1) per seed: [0.379, 0.368, 0.3]

- Your figures: `importance_ppo_drop_stack3.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_euclidean

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [964.9, 740.0, 1027.4]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'shaping': 'euclidean'}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 28 |
| Success rate | 0.447 (95% CI 0.367–0.533) |
| Collision rate | 0.347 (95% CI 0.260–0.433) |
| Collision with a person | 0.073 |
| Timeout rate | 0.207 |
| Path efficiency vs A* (successes only) | 0.954 (95% CI 0.919–0.985) |
| Time to goal (s, successes only) | 13.6 |
| Share of steps inside a person's comfort zone | 0.137 |
| Return: mean / IQM | 1.77 / 1.40 |
| Success std across seeds (stability) | 0.009 |
| Decision time (ms) | 0.69 |

- How episodes end: {'success': 0.447, 'collision_wall': 0.273, 'too_slow': 0.127, 'stuck_oscillating': 0.08, 'collision_human': 0.073}

- Success under stress conditions: nominal 0.447, bigger_map 0.527, careless_people 0.407, crowded 0.4, fast_people 0.48, hardest_level 0.173, lidar_dropout_20 0.473, lidar_dropout_40 0.4, people_start_late 0.5

- Sample efficiency (area under validation curve, 0–1) per seed: [0.509, 0.525, 0.411]

- Your figures: `importance_ppo_euclidean.png`, `saliency_ppo_euclidean.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_hybrid

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [1000.2, 1052.6, 1083.9]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 17 |
| Success rate | 0.607 (95% CI 0.527–0.687) |
| Collision rate | 0.380 (95% CI 0.293–0.460) |
| Collision with a person | 0.127 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.957 (95% CI 0.932–0.978) |
| Time to goal (s, successes only) | 11.7 |
| Share of steps inside a person's comfort zone | 0.115 |
| Return: mean / IQM | 5.55 / 8.03 |
| Success std across seeds (stability) | 0.019 |
| Decision time (ms) | 0.76 |

- How episodes end: {'success': 0.607, 'collision_wall': 0.253, 'collision_human': 0.127, 'stuck_oscillating': 0.007, 'too_slow': 0.007}

- Success under stress conditions: nominal 0.607, bigger_map 0.753, careless_people 0.553, crowded 0.473, fast_people 0.593, hardest_level 0.253, lidar_dropout_20 0.587, lidar_dropout_40 0.507, people_start_late 0.687

- Sample efficiency (area under validation curve, 0–1) per seed: [0.538, 0.534, 0.541]

- Your figures: `importance_ppo_hybrid.png`, `saliency_ppo_hybrid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_tuned

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [863.9, 1057.3, 1053.5]
- Hyper-parameters: `{'learning_rate': 0.000259, 'gamma': 0.985, 'ent_coef': 0.0017, 'n_steps': 1024, 'clip_range': 0.1875, 'batch_size': 256, 'n_epochs': 10, 'gae_lambda': 0.95}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 20 |
| Success rate | 0.527 (95% CI 0.447–0.607) |
| Collision rate | 0.333 (95% CI 0.253–0.420) |
| Collision with a person | 0.087 |
| Timeout rate | 0.140 |
| Path efficiency vs A* (successes only) | 0.961 (95% CI 0.930–0.983) |
| Time to goal (s, successes only) | 14.6 |
| Share of steps inside a person's comfort zone | 0.154 |
| Return: mean / IQM | 3.08 / 4.43 |
| Success std across seeds (stability) | 0.009 |
| Decision time (ms) | 0.77 |

- How episodes end: {'success': 0.527, 'collision_wall': 0.247, 'collision_human': 0.087, 'too_slow': 0.087, 'stuck_oscillating': 0.053}

- Success under stress conditions: nominal 0.527, bigger_map 0.587, careless_people 0.46, crowded 0.44, fast_people 0.553, hardest_level 0.2, lidar_dropout_20 0.493, lidar_dropout_40 0.427, people_start_late 0.567

- Sample efficiency (area under validation curve, 0–1) per seed: [0.484, 0.501, 0.503]

- Your figures: `importance_ppo_tuned.png`, `saliency_ppo_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
