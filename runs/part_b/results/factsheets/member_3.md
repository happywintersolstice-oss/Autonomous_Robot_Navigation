# Fact sheet: Member 3

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## ppo_drop_lastvalid

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [559.9, 570.1, 567.6]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True, 'dropout_prob': 0.2}` / `{'impute': 'last_valid'}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 22 |
| Success rate | 0.467 (95% CI 0.373–0.560) |
| Collision rate | 0.513 (95% CI 0.407–0.627) |
| Collision with a person | 0.360 |
| Timeout rate | 0.020 |
| Path efficiency vs A* (successes only) | 0.988 (95% CI 0.970–0.998) |
| Time to goal (s, successes only) | 29.3 |
| Share of steps inside a person's comfort zone | 0.167 |
| Return: mean / IQM | 8.13 / 7.23 |
| Success std across seeds (stability) | 0.038 |
| Decision time (ms) | 1.20 |

- How episodes end: {'success': 0.467, 'collision_human': 0.36, 'collision_wall': 0.153, 'stuck_oscillating': 0.02}

- Success under stress conditions: nominal 0.467, bigger_hospital 0.5, careless_people 0.413, crowded 0.4, fast_people 0.4, hardest_level 0.2, lidar_dropout_20 0.467, lidar_dropout_40 0.467, people_start_late 0.607

- Sample efficiency (area under validation curve, 0–1) per seed: [0.586, 0.561, 0.589]

- Your figures: `importance_ppo_drop_lastvalid.png`, `saliency_ppo_drop_lastvalid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_drop_maxrange

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [553.9, 559.1, 559.3]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True, 'dropout_prob': 0.2}` / `{'impute': 'max_range'}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 21 |
| Success rate | 0.473 (95% CI 0.393–0.547) |
| Collision rate | 0.527 (95% CI 0.453–0.607) |
| Collision with a person | 0.387 |
| Timeout rate | 0.000 |
| Path efficiency vs A* (successes only) | 0.979 (95% CI 0.962–0.993) |
| Time to goal (s, successes only) | 29.0 |
| Share of steps inside a person's comfort zone | 0.167 |
| Return: mean / IQM | 8.27 / 7.45 |
| Success std across seeds (stability) | 0.009 |
| Decision time (ms) | 1.17 |

- How episodes end: {'success': 0.473, 'collision_human': 0.387, 'collision_wall': 0.14}

- Success under stress conditions: nominal 0.473, bigger_hospital 0.507, careless_people 0.393, crowded 0.387, fast_people 0.447, hardest_level 0.207, lidar_dropout_20 0.473, lidar_dropout_40 0.367, people_start_late 0.56

- Sample efficiency (area under validation curve, 0–1) per seed: [0.559, 0.539, 0.534]

- Your figures: `importance_ppo_drop_maxrange.png`, `saliency_ppo_drop_maxrange.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_drop_none

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [575.3, 576.5, 566.5]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True, 'dropout_prob': 0.2}` / `{'impute': 'none'}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 24 |
| Success rate | 0.307 (95% CI 0.227–0.393) |
| Collision rate | 0.693 (95% CI 0.607–0.773) |
| Collision with a person | 0.407 |
| Timeout rate | 0.000 |
| Path efficiency vs A* (successes only) | 0.979 (95% CI 0.969–0.989) |
| Time to goal (s, successes only) | 37.2 |
| Share of steps inside a person's comfort zone | 0.188 |
| Return: mean / IQM | 2.82 / -1.39 |
| Success std across seeds (stability) | 0.038 |
| Decision time (ms) | 1.18 |

- How episodes end: {'collision_human': 0.407, 'success': 0.307, 'collision_wall': 0.287}

- Success under stress conditions: nominal 0.307, bigger_hospital 0.313, careless_people 0.247, crowded 0.273, fast_people 0.287, hardest_level 0.127, lidar_dropout_20 0.307, lidar_dropout_40 0.193, people_start_late 0.387

- Sample efficiency (area under validation curve, 0–1) per seed: [0.251, 0.32, 0.327]

- Your figures: `importance_ppo_drop_none.png`, `saliency_ppo_drop_none.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_drop_stack3

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [576.3, 601.6, 605.3]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True, 'dropout_prob': 0.2}` / `{'impute': 'none', 'frame_stack': 3}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 26 |
| Success rate | 0.187 (95% CI 0.120–0.253) |
| Collision rate | 0.800 (95% CI 0.733–0.860) |
| Collision with a person | 0.380 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.923 (95% CI 0.823–0.979) |
| Time to goal (s, successes only) | 47.1 |
| Share of steps inside a person's comfort zone | 0.166 |
| Return: mean / IQM | -1.52 / -6.42 |
| Success std across seeds (stability) | 0.025 |
| Decision time (ms) | 1.28 |

- How episodes end: {'collision_wall': 0.42, 'collision_human': 0.38, 'success': 0.187, 'too_slow': 0.013}

- Success under stress conditions: nominal 0.187, bigger_hospital 0.227, careless_people 0.167, crowded 0.147, fast_people 0.247, hardest_level 0.08, lidar_dropout_20 0.187, lidar_dropout_40 0.127, people_start_late 0.28

- Sample efficiency (area under validation curve, 0–1) per seed: [0.165, 0.196, 0.273]

- Your figures: `importance_ppo_drop_stack3.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_euclidean

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network n/a (table)
- Training time per seed (s): [645.1, 614.9, 693.0]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True, 'shaping': 'euclidean'}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 29 |
| Success rate | 0.100 (95% CI 0.033–0.187) |
| Collision rate | 0.380 (95% CI 0.193–0.580) |
| Collision with a person | 0.073 |
| Timeout rate | 0.520 |
| Path efficiency vs A* (successes only) | 0.821 (95% CI 0.702–0.975) |
| Time to goal (s, successes only) | 63.5 |
| Share of steps inside a person's comfort zone | 0.055 |
| Return: mean / IQM | -4.90 / -6.01 |
| Success std across seeds (stability) | 0.059 |
| Decision time (ms) | 1.16 |

- How episodes end: {'collision_wall': 0.307, 'too_slow': 0.3, 'stuck_oscillating': 0.22, 'success': 0.1, 'collision_human': 0.073}

- Success under stress conditions: nominal 0.1, bigger_hospital 0.073, careless_people 0.073, crowded 0.067, fast_people 0.093, hardest_level 0.02, lidar_dropout_20 0.127, lidar_dropout_40 0.06, people_start_late 0.12

- Sample efficiency (area under validation curve, 0–1) per seed: [0.068, 0.1, 0.102]

- Your figures: `importance_ppo_euclidean.png`, `saliency_ppo_euclidean.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo_tuned

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [256, 256]
- Training time per seed (s): [699.8, 695.8, 700.5]
- Hyper-parameters: `{'learning_rate': 0.000165, 'n_steps': 256, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.9603, 'gae_lambda': 0.95, 'clip_range': 0.145, 'ent_coef': 4.88e-05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 16 |
| Success rate | 0.540 (95% CI 0.460–0.627) |
| Collision rate | 0.453 (95% CI 0.367–0.533) |
| Collision with a person | 0.333 |
| Timeout rate | 0.007 |
| Path efficiency vs A* (successes only) | 0.975 (95% CI 0.960–0.989) |
| Time to goal (s, successes only) | 28.8 |
| Share of steps inside a person's comfort zone | 0.169 |
| Return: mean / IQM | 9.79 / 10.19 |
| Success std across seeds (stability) | 0.016 |
| Decision time (ms) | 1.21 |

- How episodes end: {'success': 0.54, 'collision_human': 0.333, 'collision_wall': 0.12, 'stuck_oscillating': 0.007}

- Success under stress conditions: nominal 0.54, bigger_hospital 0.593, careless_people 0.453, crowded 0.46, fast_people 0.46, hardest_level 0.273, lidar_dropout_20 0.473, lidar_dropout_40 0.353, people_start_late 0.587

- Sample efficiency (area under validation curve, 0–1) per seed: [0.513, 0.593, 0.521]

- Your figures: `importance_ppo_tuned.png`, `saliency_ppo_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
