# Fact sheet: Member 4

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## sac

- Algorithm: `sac`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1892.8, 1823.5, 1863.6]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 1 |
| Success rate | 0.800 (95% CI 0.700–0.893) |
| Collision rate | 0.187 (95% CI 0.107–0.273) |
| Collision with a person | 0.040 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.954 (95% CI 0.922–0.979) |
| Time to goal (s, successes only) | 40.3 |
| Share of steps inside a person's comfort zone | 0.201 |
| Return: mean / IQM | 15.86 / 19.76 |
| Success std across seeds (stability) | 0.065 |
| Decision time (ms) | 2.71 |

- How episodes end: {'success': 0.8, 'collision_wall': 0.147, 'collision_human': 0.04, 'too_slow': 0.013}

- Success under stress conditions: nominal 0.8, bigger_hospital 0.813, careless_people 0.447, cluttered 0.747, crowded 0.767, fast_people 0.84, hardest_level 0.613, lidar_dropout_20 0.393, lidar_dropout_40 0.147, noisy_lidar 0.813, people_start_late 0.82

- Sample efficiency (area under validation curve, 0–1) per seed: [0.333, 0.433, 0.494]

- Your figures: `importance_sac.png`, `saliency_sac.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## td3

- Algorithm: `td3`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1358.3, 1644.9, 1501.1]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2, 'action_noise_std': 0.2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 2 |
| Success rate | 0.640 (95% CI 0.507–0.767) |
| Collision rate | 0.347 (95% CI 0.213–0.473) |
| Collision with a person | 0.133 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.957 (95% CI 0.945–0.973) |
| Time to goal (s, successes only) | 38.9 |
| Share of steps inside a person's comfort zone | 0.210 |
| Return: mean / IQM | 11.01 / 13.74 |
| Success std across seeds (stability) | 0.098 |
| Decision time (ms) | 2.92 |

- How episodes end: {'success': 0.64, 'collision_wall': 0.213, 'collision_human': 0.133, 'stuck_oscillating': 0.013}

- Success under stress conditions: nominal 0.64, bigger_hospital 0.733, careless_people 0.3, cluttered 0.693, crowded 0.7, fast_people 0.7, hardest_level 0.527, lidar_dropout_20 0.38, lidar_dropout_40 0.147, noisy_lidar 0.707, people_start_late 0.74

- Sample efficiency (area under validation curve, 0–1) per seed: [0.198, 0.298, 0.312]

- Your figures: `importance_td3.png`, `saliency_td3.png`, `tuning_td3.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
