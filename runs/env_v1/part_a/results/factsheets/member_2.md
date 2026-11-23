# Fact sheet: Member 2

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## double_dqn

- Algorithm: `double_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [294.9, 302.9, 266.0]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 4 |
| Success rate | 0.460 (95% CI 0.353–0.573) |
| Collision rate | 0.440 (95% CI 0.360–0.527) |
| Collision with a person | 0.273 |
| Timeout rate | 0.100 |
| Path efficiency vs A* (successes only) | 0.941 (95% CI 0.909–0.965) |
| Time to goal (s, successes only) | 14.2 |
| Share of steps inside a person's comfort zone | 0.119 |
| Return: mean / IQM | 1.84 / 1.64 |
| Success std across seeds (stability) | 0.075 |
| Decision time (ms) | 0.63 |

- How episodes end: {'success': 0.46, 'collision_human': 0.273, 'collision_wall': 0.167, 'stuck_oscillating': 0.087, 'too_slow': 0.013}

- Success under stress conditions: nominal 0.46, bigger_map 0.48, cluttered 0.433, crowded 0.28, fast_people 0.327, hardest_level 0.113, lidar_dropout_20 0.44, lidar_dropout_40 0.367, noisy_lidar 0.44, people_start_late 0.5

- Sample efficiency (area under validation curve, 0–1) per seed: [0.464, 0.386, 0.443]

- Your figures: `importance_double_dqn.png`, `saliency_double_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dqn

- Algorithm: `dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [289.0, 302.7, 265.4]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 3 |
| Success rate | 0.487 (95% CI 0.380–0.593) |
| Collision rate | 0.447 (95% CI 0.347–0.540) |
| Collision with a person | 0.253 |
| Timeout rate | 0.067 |
| Path efficiency vs A* (successes only) | 0.903 (95% CI 0.866–0.938) |
| Time to goal (s, successes only) | 13.4 |
| Share of steps inside a person's comfort zone | 0.135 |
| Return: mean / IQM | 2.32 / 2.74 |
| Success std across seeds (stability) | 0.066 |
| Decision time (ms) | 0.61 |

- How episodes end: {'success': 0.487, 'collision_human': 0.253, 'collision_wall': 0.193, 'stuck_oscillating': 0.047, 'too_slow': 0.02}

- Success under stress conditions: nominal 0.487, bigger_map 0.553, cluttered 0.427, crowded 0.28, fast_people 0.407, hardest_level 0.107, lidar_dropout_20 0.5, lidar_dropout_40 0.38, noisy_lidar 0.52, people_start_late 0.513

- Sample efficiency (area under validation curve, 0–1) per seed: [0.416, 0.409, 0.47]

- Your figures: `importance_double_dqn.png`, `importance_dqn.png`, `importance_dueling_dqn.png`, `saliency_double_dqn.png`, `saliency_dqn.png`, `saliency_dueling_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dueling_dqn

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [328.2, 332.7, 278.1]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 2 |
| Success rate | 0.487 (95% CI 0.393–0.593) |
| Collision rate | 0.400 (95% CI 0.320–0.480) |
| Collision with a person | 0.260 |
| Timeout rate | 0.113 |
| Path efficiency vs A* (successes only) | 0.927 (95% CI 0.856–0.975) |
| Time to goal (s, successes only) | 13.2 |
| Share of steps inside a person's comfort zone | 0.131 |
| Return: mean / IQM | 2.76 / 3.17 |
| Success std across seeds (stability) | 0.052 |
| Decision time (ms) | 0.75 |

- How episodes end: {'success': 0.487, 'collision_human': 0.26, 'collision_wall': 0.14, 'stuck_oscillating': 0.08, 'too_slow': 0.033}

- Success under stress conditions: nominal 0.487, bigger_map 0.533, cluttered 0.473, crowded 0.28, fast_people 0.407, hardest_level 0.1, lidar_dropout_20 0.447, lidar_dropout_40 0.367, noisy_lidar 0.48, people_start_late 0.54

- Sample efficiency (area under validation curve, 0–1) per seed: [0.359, 0.327, 0.429]

- Your figures: `importance_dueling_dqn.png`, `saliency_dueling_dqn.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
