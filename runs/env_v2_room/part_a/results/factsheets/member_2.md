# Fact sheet: Member 2

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## double_dqn

- Algorithm: `double_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [680.7, 553.7, 614.9]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 5 |
| Success rate | 0.520 (95% CI 0.413–0.620) |
| Collision rate | 0.400 (95% CI 0.273–0.553) |
| Collision with a person | 0.120 |
| Timeout rate | 0.080 |
| Path efficiency vs A* (successes only) | 0.935 (95% CI 0.889–0.964) |
| Time to goal (s, successes only) | 13.6 |
| Share of steps inside a person's comfort zone | 0.147 |
| Return: mean / IQM | 2.91 / 4.15 |
| Success std across seeds (stability) | 0.057 |
| Decision time (ms) | 0.70 |

- How episodes end: {'success': 0.52, 'collision_wall': 0.28, 'collision_human': 0.12, 'stuck_oscillating': 0.053, 'too_slow': 0.027}

- Success under stress conditions: nominal 0.52, bigger_map 0.613, careless_people 0.48, cluttered 0.467, crowded 0.447, fast_people 0.493, hardest_level 0.267, lidar_dropout_20 0.533, lidar_dropout_40 0.36, noisy_lidar 0.493, people_start_late 0.587

- Sample efficiency (area under validation curve, 0–1) per seed: [0.511, 0.504, 0.505]

- Your figures: `importance_double_dqn.png`, `saliency_double_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dqn

- Algorithm: `dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [692.5, 722.7, 733.2]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 6 |
| Success rate | 0.520 (95% CI 0.413–0.633) |
| Collision rate | 0.413 (95% CI 0.313–0.507) |
| Collision with a person | 0.147 |
| Timeout rate | 0.067 |
| Path efficiency vs A* (successes only) | 0.981 (95% CI 0.943–0.998) |
| Time to goal (s, successes only) | 11.9 |
| Share of steps inside a person's comfort zone | 0.132 |
| Return: mean / IQM | 3.17 / 4.27 |
| Success std across seeds (stability) | 0.071 |
| Decision time (ms) | 0.63 |

- How episodes end: {'success': 0.52, 'collision_wall': 0.267, 'collision_human': 0.147, 'stuck_oscillating': 0.047, 'too_slow': 0.02}

- Success under stress conditions: nominal 0.52, bigger_map 0.647, careless_people 0.473, cluttered 0.527, crowded 0.453, fast_people 0.527, hardest_level 0.253, lidar_dropout_20 0.467, lidar_dropout_40 0.373, noisy_lidar 0.513, people_start_late 0.647

- Sample efficiency (area under validation curve, 0–1) per seed: [0.504, 0.539, 0.498]

- Your figures: `importance_double_dqn.png`, `importance_dqn.png`, `importance_dueling_dqn.png`, `saliency_double_dqn.png`, `saliency_dqn.png`, `saliency_dueling_dqn.png`, `tuning_dueling_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dueling_dqn

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [748.5, 766.3, 754.3]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 4 |
| Success rate | 0.547 (95% CI 0.453–0.647) |
| Collision rate | 0.340 (95% CI 0.260–0.420) |
| Collision with a person | 0.127 |
| Timeout rate | 0.113 |
| Path efficiency vs A* (successes only) | 0.922 (95% CI 0.883–0.958) |
| Time to goal (s, successes only) | 13.5 |
| Share of steps inside a person's comfort zone | 0.148 |
| Return: mean / IQM | 3.30 / 5.05 |
| Success std across seeds (stability) | 0.038 |
| Decision time (ms) | 0.73 |

- How episodes end: {'success': 0.547, 'collision_wall': 0.213, 'collision_human': 0.127, 'stuck_oscillating': 0.1, 'too_slow': 0.013}

- Success under stress conditions: nominal 0.547, bigger_map 0.62, careless_people 0.467, cluttered 0.52, crowded 0.467, fast_people 0.607, hardest_level 0.353, lidar_dropout_20 0.48, lidar_dropout_40 0.387, noisy_lidar 0.587, people_start_late 0.64

- Sample efficiency (area under validation curve, 0–1) per seed: [0.523, 0.461, 0.518]

- Your figures: `importance_dueling_dqn.png`, `saliency_dueling_dqn.png`, `tuning_dueling_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
