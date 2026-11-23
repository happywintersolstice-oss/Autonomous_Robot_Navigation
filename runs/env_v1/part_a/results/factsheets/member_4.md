# Fact sheet: Member 4

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## sac

- Algorithm: `sac`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [487.2, 419.3, 200.8]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 10 |
| Success rate | 0.160 (95% CI 0.107–0.227) |
| Collision rate | 0.467 (95% CI 0.373–0.560) |
| Collision with a person | 0.433 |
| Timeout rate | 0.373 |
| Path efficiency vs A* (successes only) | 0.802 (95% CI 0.665–0.899) |
| Time to goal (s, successes only) | 25.6 |
| Share of steps inside a person's comfort zone | 0.159 |
| Return: mean / IQM | -4.87 / -7.13 |
| Success std across seeds (stability) | 0.016 |
| Decision time (ms) | 0.76 |

- How episodes end: {'collision_human': 0.433, 'too_slow': 0.367, 'success': 0.16, 'collision_wall': 0.033, 'stuck_oscillating': 0.007}

- Success under stress conditions: nominal 0.16, bigger_map 0.213, cluttered 0.167, crowded 0.053, fast_people 0.14, hardest_level 0.06, lidar_dropout_20 0.173, lidar_dropout_40 0.207, noisy_lidar 0.16, people_start_late 0.2

- Sample efficiency (area under validation curve, 0–1) per seed: [0.177, 0.183, 0.071]

- Your figures: `importance_sac.png`, `saliency_sac.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## td3

- Algorithm: `td3`; action space: continuous; code: `src/medinav/agents/continuous_control.py`
- Seeds trained: [0, 1, 2]
- Budget: 150000 environment steps per seed; network [256, 256]
- Training time per seed (s): [337.5, 327.0, 262.1]
- Hyper-parameters: `{'learning_rate': 0.0003, 'buffer_size': 200000, 'learning_starts': 5000, 'batch_size': 256, 'gamma': 0.99, 'tau': 0.005, 'train_freq': 4, 'gradient_steps': 2, 'action_noise_std': 0.2}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 7 |
| Success rate | 0.413 (95% CI 0.280–0.547) |
| Collision rate | 0.453 (95% CI 0.353–0.560) |
| Collision with a person | 0.393 |
| Timeout rate | 0.133 |
| Path efficiency vs A* (successes only) | 0.888 (95% CI 0.810–0.945) |
| Time to goal (s, successes only) | 18.9 |
| Share of steps inside a person's comfort zone | 0.141 |
| Return: mean / IQM | -0.16 / -0.94 |
| Success std across seeds (stability) | 0.106 |
| Decision time (ms) | 0.49 |

- How episodes end: {'success': 0.413, 'collision_human': 0.393, 'stuck_oscillating': 0.093, 'collision_wall': 0.06, 'too_slow': 0.04}

- Success under stress conditions: nominal 0.413, bigger_map 0.453, cluttered 0.307, crowded 0.213, fast_people 0.333, hardest_level 0.127, lidar_dropout_20 0.493, lidar_dropout_40 0.413, noisy_lidar 0.433, people_start_late 0.387

- Sample efficiency (area under validation curve, 0–1) per seed: [0.215, 0.392, 0.181]

- Your figures: `importance_td3.png`, `saliency_td3.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
