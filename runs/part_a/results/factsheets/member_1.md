# Fact sheet: Member 1

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## q_learning

- Algorithm: `q_learning`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [1609.6, 1626.1, 1618.4]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 9 |
| Success rate | 0.153 (95% CI 0.000–0.287) |
| Collision rate | 0.180 (95% CI 0.100–0.273) |
| Collision with a person | 0.067 |
| Timeout rate | 0.667 |
| Path efficiency vs A* (successes only) | 0.882 (95% CI 0.828–0.925) |
| Time to goal (s, successes only) | 62.1 |
| Share of steps inside a person's comfort zone | 0.140 |
| Return: mean / IQM | -3.96 / -5.92 |
| Success std across seeds (stability) | 0.111 |
| Decision time (ms) | 0.90 |

- How episodes end: {'stuck_oscillating': 0.593, 'success': 0.153, 'collision_wall': 0.113, 'too_slow': 0.073, 'collision_human': 0.067}

- Success under stress conditions: nominal 0.153, bigger_hospital 0.187, careless_people 0.067, cluttered 0.14, crowded 0.167, fast_people 0.173, hardest_level 0.14, lidar_dropout_20 0.427, lidar_dropout_40 0.48, noisy_lidar 0.267, people_start_late 0.153

- Sample efficiency (area under validation curve, 0–1) per seed: [0.109, 0.031, 0.1]

- Your figures: `importance_q_learning.png`, `tuning_q_learning.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## sarsa

- Algorithm: `sarsa`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [1994.6, 1961.2, 1947.6]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 10 |
| Success rate | 0.100 (95% CI 0.020–0.213) |
| Collision rate | 0.167 (95% CI 0.053–0.340) |
| Collision with a person | 0.073 |
| Timeout rate | 0.733 |
| Path efficiency vs A* (successes only) | 0.799 (95% CI 0.700–0.961) |
| Time to goal (s, successes only) | 69.3 |
| Share of steps inside a person's comfort zone | 0.166 |
| Return: mean / IQM | -5.57 / -6.85 |
| Success std across seeds (stability) | 0.085 |
| Decision time (ms) | 1.16 |

- How episodes end: {'stuck_oscillating': 0.553, 'too_slow': 0.18, 'success': 0.1, 'collision_wall': 0.093, 'collision_human': 0.073}

- Success under stress conditions: nominal 0.1, bigger_hospital 0.1, careless_people 0.047, cluttered 0.153, crowded 0.16, fast_people 0.133, hardest_level 0.093, lidar_dropout_20 0.227, lidar_dropout_40 0.307, noisy_lidar 0.167, people_start_late 0.107

- Sample efficiency (area under validation curve, 0–1) per seed: [0.09, 0.069, 0.045]

- Your figures: `importance_sarsa.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
