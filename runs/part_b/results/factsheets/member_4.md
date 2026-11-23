# Fact sheet: Member 4

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## sac_hybrid

- Algorithm: `sac`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1288.7, 1310.0, 1168.7]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 1 |
| Success rate | 0.820 (95% CI 0.740–0.893) |
| Collision rate | 0.140 (95% CI 0.060–0.227) |
| Collision with a person | 0.007 |
| Timeout rate | 0.040 |
| Path efficiency vs A* (successes only) | 0.903 (95% CI 0.872–0.931) |
| Time to goal (s, successes only) | 17.5 |
| Share of steps inside a person's comfort zone | 0.189 |
| Return: mean / IQM | 9.37 / 13.47 |
| Success std across seeds (stability) | 0.043 |
| Decision time (ms) | 0.83 |

- How episodes end: {'success': 0.82, 'collision_wall': 0.133, 'too_slow': 0.033, 'stuck_oscillating': 0.007, 'collision_human': 0.007}

- Success under stress conditions: nominal 0.82, bigger_map 0.907, careless_people 0.613, crowded 0.733, fast_people 0.807, hardest_level 0.687, lidar_dropout_20 0.513, lidar_dropout_40 0.373, people_start_late 0.807

- Sample efficiency (area under validation curve, 0–1) per seed: [0.496, 0.525, 0.606]

- Your figures: `importance_sac_hybrid.png`, `saliency_sac_hybrid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## sac_long

- Algorithm: `sac`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1573.2, 1549.8, 1513.1]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 2 |
| Success rate | 0.813 (95% CI 0.740–0.887) |
| Collision rate | 0.047 (95% CI 0.013–0.100) |
| Collision with a person | 0.007 |
| Timeout rate | 0.140 |
| Path efficiency vs A* (successes only) | 0.916 (95% CI 0.890–0.941) |
| Time to goal (s, successes only) | 15.5 |
| Share of steps inside a person's comfort zone | 0.167 |
| Return: mean / IQM | 9.41 / 13.53 |
| Success std across seeds (stability) | 0.034 |
| Decision time (ms) | 0.85 |

- How episodes end: {'success': 0.813, 'too_slow': 0.127, 'collision_wall': 0.04, 'stuck_oscillating': 0.013, 'collision_human': 0.007}

- Success under stress conditions: nominal 0.813, bigger_map 0.713, careless_people 0.633, crowded 0.767, fast_people 0.813, hardest_level 0.6, lidar_dropout_20 0.64, lidar_dropout_40 0.467, people_start_late 0.827

- Sample efficiency (area under validation curve, 0–1) per seed: [0.618, 0.686, 0.646]

- Your figures: `importance_sac_long.png`, `saliency_sac_long.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## td3_hybrid

- Algorithm: `td3`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1018.0, 956.0, 1018.7]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2, 'action_noise_std': 0.2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 12 |
| Success rate | 0.660 (95% CI 0.553–0.767) |
| Collision rate | 0.060 (95% CI 0.000–0.133) |
| Collision with a person | 0.000 |
| Timeout rate | 0.280 |
| Path efficiency vs A* (successes only) | 0.955 (95% CI 0.914–0.982) |
| Time to goal (s, successes only) | 16.9 |
| Share of steps inside a person's comfort zone | 0.194 |
| Return: mean / IQM | 5.55 / 9.52 |
| Success std across seeds (stability) | 0.071 |
| Decision time (ms) | 0.79 |

- How episodes end: {'success': 0.66, 'stuck_oscillating': 0.193, 'too_slow': 0.087, 'collision_wall': 0.06}

- Success under stress conditions: nominal 0.66, bigger_map 0.68, careless_people 0.52, crowded 0.6, fast_people 0.653, hardest_level 0.407, lidar_dropout_20 0.633, lidar_dropout_40 0.453, people_start_late 0.667

- Sample efficiency (area under validation curve, 0–1) per seed: [0.342, 0.371, 0.556]

- Your figures: `importance_td3_hybrid.png`, `saliency_td3_hybrid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## td3_tuned

- Algorithm: `td3`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [492.9, 835.5, 981.9]
- Hyper-parameters: `{'learning_rate': 0.000439, 'gamma': 0.9746, 'tau': 0.0175, 'batch_size': 128, 'buffer_size': 200000, 'learning_starts': 5000, 'train_freq': 4, 'gradient_steps': 2, 'action_noise_std': 0.2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 9 |
| Success rate | 0.673 (95% CI 0.580–0.780) |
| Collision rate | 0.160 (95% CI 0.053–0.293) |
| Collision with a person | 0.067 |
| Timeout rate | 0.167 |
| Path efficiency vs A* (successes only) | 0.918 (95% CI 0.881–0.948) |
| Time to goal (s, successes only) | 15.1 |
| Share of steps inside a person's comfort zone | 0.158 |
| Return: mean / IQM | 6.14 / 10.07 |
| Success std across seeds (stability) | 0.062 |
| Decision time (ms) | 0.74 |

- How episodes end: {'success': 0.673, 'too_slow': 0.12, 'collision_wall': 0.093, 'collision_human': 0.067, 'stuck_oscillating': 0.047}

- Success under stress conditions: nominal 0.673, bigger_map 0.7, careless_people 0.533, crowded 0.593, fast_people 0.673, hardest_level 0.42, lidar_dropout_20 0.533, lidar_dropout_40 0.407, people_start_late 0.72

- Sample efficiency (area under validation curve, 0–1) per seed: [0.563, 0.667, 0.575]

- Your figures: `importance_td3_tuned.png`, `saliency_td3_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
