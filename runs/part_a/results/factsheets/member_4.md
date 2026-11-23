# Fact sheet: Member 4

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## sac

- Algorithm: `sac`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [932.7, 930.0, 980.6]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 1 |
| Success rate | 0.733 (95% CI 0.660–0.807) |
| Collision rate | 0.080 (95% CI 0.020–0.147) |
| Collision with a person | 0.020 |
| Timeout rate | 0.187 |
| Path efficiency vs A* (successes only) | 0.853 (95% CI 0.811–0.895) |
| Time to goal (s, successes only) | 18.4 |
| Share of steps inside a person's comfort zone | 0.169 |
| Return: mean / IQM | 7.44 / 11.51 |
| Success std across seeds (stability) | 0.025 |
| Decision time (ms) | 0.86 |

- How episodes end: {'success': 0.733, 'too_slow': 0.173, 'collision_wall': 0.06, 'collision_human': 0.02, 'stuck_oscillating': 0.013}

- Success under stress conditions: nominal 0.733, bigger_map 0.753, careless_people 0.567, cluttered 0.833, crowded 0.667, fast_people 0.773, hardest_level 0.48, lidar_dropout_20 0.587, lidar_dropout_40 0.38, noisy_lidar 0.713, people_start_late 0.72

- Sample efficiency (area under validation curve, 0–1) per seed: [0.483, 0.513, 0.481]

- Your figures: `importance_sac.png`, `saliency_sac.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## td3

- Algorithm: `td3`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [877.4, 822.8, 862.4]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2, 'action_noise_std': 0.2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 3 |
| Success rate | 0.667 (95% CI 0.580–0.760) |
| Collision rate | 0.113 (95% CI 0.033–0.200) |
| Collision with a person | 0.020 |
| Timeout rate | 0.220 |
| Path efficiency vs A* (successes only) | 0.952 (95% CI 0.920–0.982) |
| Time to goal (s, successes only) | 14.8 |
| Share of steps inside a person's comfort zone | 0.198 |
| Return: mean / IQM | 6.03 / 9.94 |
| Success std across seeds (stability) | 0.038 |
| Decision time (ms) | 0.71 |

- How episodes end: {'success': 0.667, 'too_slow': 0.127, 'stuck_oscillating': 0.093, 'collision_wall': 0.093, 'collision_human': 0.02}

- Success under stress conditions: nominal 0.667, bigger_map 0.627, careless_people 0.52, cluttered 0.707, crowded 0.613, fast_people 0.713, hardest_level 0.427, lidar_dropout_20 0.507, lidar_dropout_40 0.38, noisy_lidar 0.66, people_start_late 0.613

- Sample efficiency (area under validation curve, 0–1) per seed: [0.458, 0.402, 0.477]

- Your figures: `importance_td3.png`, `saliency_td3.png`, `tuning_td3.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
