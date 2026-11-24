# Fact sheet: Member 1

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## q_learning_noroute

- Algorithm: `q_learning`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [1219.6, 1229.8, 1161.8]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': False}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 32 |
| Success rate | 0.000 (95% CI 0.000–0.000) |
| Collision rate | 0.027 (95% CI 0.007–0.060) |
| Collision with a person | 0.007 |
| Timeout rate | 0.973 |
| Path efficiency vs A* (successes only) | n/a |
| Time to goal (s, successes only) | n/a |
| Share of steps inside a person's comfort zone | 0.066 |
| Return: mean / IQM | -9.45 / -7.85 |
| Success std across seeds (stability) | 0.000 |
| Decision time (ms) | 0.80 |

- How episodes end: {'stuck_oscillating': 0.873, 'too_slow': 0.1, 'collision_wall': 0.02, 'collision_human': 0.007}

- Success under stress conditions: nominal 0.0, bigger_hospital 0.0, careless_people 0.0, crowded 0.0, fast_people 0.0, hardest_level 0.0, lidar_dropout_20 0.007, lidar_dropout_40 0.007, people_start_late 0.0

- Sample efficiency (area under validation curve, 0–1) per seed: [0.0, 0.0, 0.0]

- Your figures: `importance_q_learning_noroute.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## q_learning_tuned

- Algorithm: `q_learning`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [916.1, 888.6, 886.3]
- Hyper-parameters: `{'alpha': 0.144, 'gamma': 0.9356, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.562}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 33 agents incl. baselines) | 25 |
| Success rate | 0.267 (95% CI 0.147–0.400) |
| Collision rate | 0.193 (95% CI 0.107–0.280) |
| Collision with a person | 0.027 |
| Timeout rate | 0.540 |
| Path efficiency vs A* (successes only) | 0.914 (95% CI 0.890–0.936) |
| Time to goal (s, successes only) | 62.1 |
| Share of steps inside a person's comfort zone | 0.190 |
| Return: mean / IQM | -1.60 / -4.23 |
| Success std across seeds (stability) | 0.100 |
| Decision time (ms) | 1.05 |

- How episodes end: {'stuck_oscillating': 0.427, 'success': 0.267, 'collision_wall': 0.167, 'too_slow': 0.113, 'collision_human': 0.027}

- Success under stress conditions: nominal 0.267, bigger_hospital 0.307, careless_people 0.1, crowded 0.26, fast_people 0.273, hardest_level 0.207, lidar_dropout_20 0.487, lidar_dropout_40 0.36, people_start_late 0.187

- Sample efficiency (area under validation curve, 0–1) per seed: [0.181, 0.218, 0.15]

- Your figures: `importance_q_learning_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
