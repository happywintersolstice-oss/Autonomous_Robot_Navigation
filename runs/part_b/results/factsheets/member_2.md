# Fact sheet: Member 2

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## dueling_dqn_curriculum

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [527.4, 511.0, 503.3]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 14 |
| Success rate | 0.573 (95% CI 0.460–0.693) |
| Collision rate | 0.413 (95% CI 0.306–0.520) |
| Collision with a person | 0.240 |
| Timeout rate | 0.013 |
| Path efficiency vs A* (successes only) | 0.999 (95% CI 0.998–1.000) |
| Time to goal (s, successes only) | 32.7 |
| Share of steps inside a person's comfort zone | 0.176 |
| Return: mean / IQM | 10.96 / 12.33 |
| Success std across seeds (stability) | 0.075 |
| Decision time (ms) | 1.16 |

- How episodes end: {'success': 0.573, 'collision_human': 0.24, 'collision_wall': 0.173, 'too_slow': 0.007, 'stuck_oscillating': 0.007}

- Success under stress conditions: nominal 0.573, bigger_hospital 0.653, careless_people 0.373, crowded 0.467, fast_people 0.42, hardest_level 0.273, lidar_dropout_20 0.287, lidar_dropout_40 0.113, people_start_late 0.667

- Sample efficiency (area under validation curve, 0–1) per seed: [0.368, 0.362, 0.384]

- Your figures: `importance_dueling_dqn_curriculum.png`, `saliency_dueling_dqn_curriculum.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dueling_dqn_noroute

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [709.1, 716.9, 694.6]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': False}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 30 |
| Success rate | 0.027 (95% CI 0.000–0.080) |
| Collision rate | 0.273 (95% CI 0.073–0.520) |
| Collision with a person | 0.053 |
| Timeout rate | 0.700 |
| Path efficiency vs A* (successes only) | 0.578 (95% CI 0.276–0.786) |
| Time to goal (s, successes only) | 45.3 |
| Share of steps inside a person's comfort zone | 0.048 |
| Return: mean / IQM | -9.41 / -8.91 |
| Success std across seeds (stability) | 0.038 |
| Decision time (ms) | 1.10 |

- How episodes end: {'stuck_oscillating': 0.533, 'collision_wall': 0.22, 'too_slow': 0.167, 'collision_human': 0.053, 'success': 0.027}

- Success under stress conditions: nominal 0.027, bigger_hospital 0.013, careless_people 0.027, crowded 0.027, fast_people 0.027, hardest_level 0.013, lidar_dropout_20 0.027, lidar_dropout_40 0.007, people_start_late 0.033

- Sample efficiency (area under validation curve, 0–1) per seed: [0.039, 0.014, 0.005]

- Your figures: `importance_dueling_dqn_noroute.png`, `saliency_dueling_dqn_noroute.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dueling_dqn_tuned

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [589.3, 591.4, 592.1]
- Hyper-parameters: `{'learning_rate': 0.00026, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.9915, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.452, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 15 |
| Success rate | 0.573 (95% CI 0.480–0.660) |
| Collision rate | 0.373 (95% CI 0.293–0.453) |
| Collision with a person | 0.267 |
| Timeout rate | 0.053 |
| Path efficiency vs A* (successes only) | 0.992 (95% CI 0.980–1.000) |
| Time to goal (s, successes only) | 33.5 |
| Share of steps inside a person's comfort zone | 0.189 |
| Return: mean / IQM | 10.17 / 11.09 |
| Success std across seeds (stability) | 0.038 |
| Decision time (ms) | 1.21 |

- How episodes end: {'success': 0.573, 'collision_human': 0.267, 'collision_wall': 0.107, 'stuck_oscillating': 0.053}

- Success under stress conditions: nominal 0.573, bigger_hospital 0.567, careless_people 0.387, crowded 0.493, fast_people 0.52, hardest_level 0.34, lidar_dropout_20 0.367, lidar_dropout_40 0.173, people_start_late 0.633

- Sample efficiency (area under validation curve, 0–1) per seed: [0.402, 0.402, 0.53]

- Your figures: `importance_dueling_dqn_tuned.png`, `saliency_dueling_dqn_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
