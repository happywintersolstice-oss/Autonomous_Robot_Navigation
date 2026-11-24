# Fact sheet: Member 4

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## sac_long

- Algorithm: `sac`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1359.1, 1363.5, 1357.2]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 1 |
| Success rate | 0.953 (95% CI 0.893–0.993) |
| Collision rate | 0.047 (95% CI 0.007–0.107) |
| Collision with a person | 0.013 |
| Timeout rate | 0.000 |
| Path efficiency vs A* (successes only) | 0.974 (95% CI 0.962–0.983) |
| Time to goal (s, successes only) | 32.6 |
| Share of steps inside a person's comfort zone | 0.163 |
| Return: mean / IQM | 22.01 / 22.10 |
| Success std across seeds (stability) | 0.038 |
| Decision time (ms) | 1.21 |

- How episodes end: {'success': 0.953, 'collision_wall': 0.033, 'collision_human': 0.013}

- Success under stress conditions: nominal 0.953, bigger_hospital 0.96, careless_people 0.527, crowded 0.913, fast_people 0.887, hardest_level 0.813, lidar_dropout_20 0.627, lidar_dropout_40 0.3, people_start_late 0.96

- Sample efficiency (area under validation curve, 0–1) per seed: [0.636, 0.679, 0.736]

- Your figures: `importance_sac_long.png`, `saliency_sac_long.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## sac_noroute

- Algorithm: `sac`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1133.0, 1134.9, 1115.5]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': False}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 31 |
| Success rate | 0.013 (95% CI 0.000–0.047) |
| Collision rate | 0.013 (95% CI 0.000–0.047) |
| Collision with a person | 0.007 |
| Timeout rate | 0.973 |
| Path efficiency vs A* (successes only) | 0.703 (95% CI 0.471–0.936) |
| Time to goal (s, successes only) | 55.1 |
| Share of steps inside a person's comfort zone | 0.047 |
| Return: mean / IQM | -8.12 / -8.03 |
| Success std across seeds (stability) | 0.019 |
| Decision time (ms) | 1.12 |

- How episodes end: {'too_slow': 0.973, 'success': 0.013, 'collision_human': 0.007, 'collision_wall': 0.007}

- Success under stress conditions: nominal 0.013, bigger_hospital 0.0, careless_people 0.013, crowded 0.013, fast_people 0.007, hardest_level 0.0, lidar_dropout_20 0.013, lidar_dropout_40 0.027, people_start_late 0.013

- Sample efficiency (area under validation curve, 0–1) per seed: [0.015, 0.0, 0.0]

- Your figures: `importance_sac_noroute.png`, `saliency_sac_noroute.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## td3_long

- Algorithm: `td3`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1038.8, 1022.6, 1029.0]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2, 'action_noise_std': 0.2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 2 |
| Success rate | 0.913 (95% CI 0.860–0.953) |
| Collision rate | 0.073 (95% CI 0.027–0.127) |
| Collision with a person | 0.033 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.986 (95% CI 0.978–0.992) |
| Time to goal (s, successes only) | 30.6 |
| Share of steps inside a person's comfort zone | 0.172 |
| Return: mean / IQM | 20.90 / 22.07 |
| Success std across seeds (stability) | 0.009 |
| Decision time (ms) | 1.09 |

- How episodes end: {'success': 0.913, 'collision_wall': 0.04, 'collision_human': 0.033, 'stuck_oscillating': 0.013}

- Success under stress conditions: nominal 0.913, bigger_hospital 0.933, careless_people 0.56, crowded 0.927, fast_people 0.893, hardest_level 0.84, lidar_dropout_20 0.653, lidar_dropout_40 0.333, people_start_late 0.92

- Sample efficiency (area under validation curve, 0–1) per seed: [0.375, 0.479, 0.525]

- Your figures: `importance_td3_long.png`, `saliency_td3_long.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## td3_tuned

- Algorithm: `td3`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [773.6, 774.9, 755.8]
- Hyper-parameters: `{'learning_rate': 0.000632, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 512, 'gamma': 0.9728, 'tau': 0.0055, 'train_freq': 4, 'gradient_steps': 2, 'action_noise_std': 0.2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 3 |
| Success rate | 0.833 (95% CI 0.760–0.893) |
| Collision rate | 0.100 (95% CI 0.047–0.160) |
| Collision with a person | 0.040 |
| Timeout rate | 0.067 |
| Path efficiency vs A* (successes only) | 0.969 (95% CI 0.924–0.994) |
| Time to goal (s, successes only) | 34.9 |
| Share of steps inside a person's comfort zone | 0.179 |
| Return: mean / IQM | 17.42 / 20.77 |
| Success std across seeds (stability) | 0.025 |
| Decision time (ms) | 1.14 |

- How episodes end: {'success': 0.833, 'collision_wall': 0.06, 'collision_human': 0.04, 'stuck_oscillating': 0.04, 'too_slow': 0.027}

- Success under stress conditions: nominal 0.833, bigger_hospital 0.747, careless_people 0.493, crowded 0.773, fast_people 0.793, hardest_level 0.68, lidar_dropout_20 0.46, lidar_dropout_40 0.16, people_start_late 0.847

- Sample efficiency (area under validation curve, 0–1) per seed: [0.504, 0.448, 0.677]

- Your figures: `importance_td3_tuned.png`, `saliency_td3_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
