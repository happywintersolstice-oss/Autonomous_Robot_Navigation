# Fact sheet: Member 2

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## double_dqn

- Algorithm: `double_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [1391.3, 1425.8, 1566.0]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 5 |
| Success rate | 0.513 (95% CI 0.407–0.620) |
| Collision rate | 0.453 (95% CI 0.340–0.580) |
| Collision with a person | 0.213 |
| Timeout rate | 0.033 |
| Path efficiency vs A* (successes only) | 0.989 (95% CI 0.982–0.995) |
| Time to goal (s, successes only) | 32.4 |
| Share of steps inside a person's comfort zone | 0.190 |
| Return: mean / IQM | 8.96 / 8.96 |
| Success std across seeds (stability) | 0.066 |
| Decision time (ms) | 2.64 |

- How episodes end: {'success': 0.513, 'collision_wall': 0.24, 'collision_human': 0.213, 'stuck_oscillating': 0.033}

- Success under stress conditions: nominal 0.513, bigger_hospital 0.54, careless_people 0.367, cluttered 0.513, crowded 0.44, fast_people 0.473, hardest_level 0.32, lidar_dropout_20 0.433, lidar_dropout_40 0.253, noisy_lidar 0.48, people_start_late 0.547

- Sample efficiency (area under validation curve, 0–1) per seed: [0.384, 0.436, 0.454]

- Your figures: `importance_double_dqn.png`, `saliency_double_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dqn

- Algorithm: `dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [1473.6, 1576.5, 1452.9]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 6 |
| Success rate | 0.480 (95% CI 0.347–0.600) |
| Collision rate | 0.480 (95% CI 0.360–0.620) |
| Collision with a person | 0.233 |
| Timeout rate | 0.040 |
| Path efficiency vs A* (successes only) | 0.988 (95% CI 0.977–0.997) |
| Time to goal (s, successes only) | 29.9 |
| Share of steps inside a person's comfort zone | 0.174 |
| Return: mean / IQM | 8.41 / 7.92 |
| Success std across seeds (stability) | 0.099 |
| Decision time (ms) | 2.37 |

- How episodes end: {'success': 0.48, 'collision_wall': 0.247, 'collision_human': 0.233, 'stuck_oscillating': 0.04}

- Success under stress conditions: nominal 0.48, bigger_hospital 0.52, careless_people 0.3, cluttered 0.507, crowded 0.393, fast_people 0.413, hardest_level 0.267, lidar_dropout_20 0.42, lidar_dropout_40 0.267, noisy_lidar 0.473, people_start_late 0.52

- Sample efficiency (area under validation curve, 0–1) per seed: [0.364, 0.448, 0.412]

- Your figures: `importance_double_dqn.png`, `importance_dqn.png`, `importance_dueling_dqn.png`, `saliency_double_dqn.png`, `saliency_dqn.png`, `saliency_dueling_dqn.png`, `tuning_dueling_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dueling_dqn

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [1574.4, 1570.9, 1616.7]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 3 |
| Success rate | 0.633 (95% CI 0.520–0.733) |
| Collision rate | 0.353 (95% CI 0.253–0.460) |
| Collision with a person | 0.153 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.982 (95% CI 0.971–0.990) |
| Time to goal (s, successes only) | 32.5 |
| Share of steps inside a person's comfort zone | 0.178 |
| Return: mean / IQM | 12.88 / 15.49 |
| Success std across seeds (stability) | 0.066 |
| Decision time (ms) | 3.04 |

- How episodes end: {'success': 0.633, 'collision_wall': 0.2, 'collision_human': 0.153, 'stuck_oscillating': 0.013}

- Success under stress conditions: nominal 0.633, bigger_hospital 0.78, careless_people 0.373, cluttered 0.66, crowded 0.58, fast_people 0.587, hardest_level 0.447, lidar_dropout_20 0.387, lidar_dropout_40 0.213, noisy_lidar 0.607, people_start_late 0.747

- Sample efficiency (area under validation curve, 0–1) per seed: [0.441, 0.377, 0.543]

- Your figures: `importance_dueling_dqn.png`, `saliency_dueling_dqn.png`, `tuning_dueling_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
