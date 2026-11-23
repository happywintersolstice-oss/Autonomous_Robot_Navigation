# Fact sheet: Member 2

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## dueling_dqn_curriculum

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [1051.9, 966.0, 1165.4]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 16 |
| Success rate | 0.607 (95% CI 0.527–0.687) |
| Collision rate | 0.267 (95% CI 0.200–0.340) |
| Collision with a person | 0.107 |
| Timeout rate | 0.127 |
| Path efficiency vs A* (successes only) | 0.953 (95% CI 0.934–0.972) |
| Time to goal (s, successes only) | 12.8 |
| Share of steps inside a person's comfort zone | 0.137 |
| Return: mean / IQM | 5.22 / 7.81 |
| Success std across seeds (stability) | 0.019 |
| Decision time (ms) | 0.81 |

- How episodes end: {'success': 0.607, 'collision_wall': 0.16, 'collision_human': 0.107, 'stuck_oscillating': 0.093, 'too_slow': 0.033}

- Success under stress conditions: nominal 0.607, bigger_map 0.6, careless_people 0.52, crowded 0.467, fast_people 0.567, hardest_level 0.307, lidar_dropout_20 0.527, lidar_dropout_40 0.407, people_start_late 0.64

- Sample efficiency (area under validation curve, 0–1) per seed: [0.418, 0.452, 0.514]

- Your figures: `importance_dueling_dqn_curriculum.png`, `saliency_dueling_dqn_curriculum.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dueling_dqn_hybrid

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [1023.5, 1055.4, 1062.7]
- Hyper-parameters: `{'learning_rate': 0.0005, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'gamma': 0.99, 'train_freq': 4, 'target_update_interval': 2000, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 7 |
| Success rate | 0.687 (95% CI 0.540–0.827) |
| Collision rate | 0.313 (95% CI 0.173–0.460) |
| Collision with a person | 0.120 |
| Timeout rate | 0.000 |
| Path efficiency vs A* (successes only) | 0.990 (95% CI 0.978–0.999) |
| Time to goal (s, successes only) | 11.4 |
| Share of steps inside a person's comfort zone | 0.160 |
| Return: mean / IQM | 7.17 / 11.19 |
| Success std across seeds (stability) | 0.106 |
| Decision time (ms) | 0.79 |

- How episodes end: {'success': 0.687, 'collision_wall': 0.193, 'collision_human': 0.12}

- Success under stress conditions: nominal 0.687, bigger_map 0.747, careless_people 0.64, crowded 0.527, fast_people 0.627, hardest_level 0.373, lidar_dropout_20 0.533, lidar_dropout_40 0.473, people_start_late 0.753

- Sample efficiency (area under validation curve, 0–1) per seed: [0.512, 0.529, 0.557]

- Your figures: `importance_dueling_dqn_hybrid.png`, `saliency_dueling_dqn_hybrid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## dueling_dqn_tuned

- Algorithm: `dueling_dqn`; action space: discrete; code: `src/medinav/agents/dqn_family.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [256, 256]
- Training time per seed (s): [1113.2, 1090.1, 1049.1]
- Hyper-parameters: `{'learning_rate': 0.000162, 'gamma': 0.9703, 'target_update_interval': 1000, 'exploration_fraction': 0.309, 'buffer_size': 100000, 'learning_starts': 5000, 'batch_size': 128, 'train_freq': 4, 'exploration_final_eps': 0.05}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 15 |
| Success rate | 0.633 (95% CI 0.547–0.720) |
| Collision rate | 0.240 (95% CI 0.160–0.327) |
| Collision with a person | 0.093 |
| Timeout rate | 0.127 |
| Path efficiency vs A* (successes only) | 0.950 (95% CI 0.919–0.978) |
| Time to goal (s, successes only) | 14.4 |
| Share of steps inside a person's comfort zone | 0.162 |
| Return: mean / IQM | 5.99 / 8.99 |
| Success std across seeds (stability) | 0.034 |
| Decision time (ms) | 0.79 |

- How episodes end: {'success': 0.633, 'collision_wall': 0.147, 'collision_human': 0.093, 'too_slow': 0.08, 'stuck_oscillating': 0.047}

- Success under stress conditions: nominal 0.633, bigger_map 0.667, careless_people 0.527, crowded 0.493, fast_people 0.667, hardest_level 0.34, lidar_dropout_20 0.56, lidar_dropout_40 0.427, people_start_late 0.687

- Sample efficiency (area under validation curve, 0–1) per seed: [0.504, 0.555, 0.543]

- Your figures: `importance_dueling_dqn_tuned.png`, `saliency_dueling_dqn_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
