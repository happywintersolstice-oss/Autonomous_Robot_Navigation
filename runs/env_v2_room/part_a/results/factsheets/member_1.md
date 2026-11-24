# Fact sheet: Member 1

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## q_learning

- Algorithm: `q_learning`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [397.7, 411.1, 406.1]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 10 |
| Success rate | 0.353 (95% CI 0.253–0.453) |
| Collision rate | 0.147 (95% CI 0.087–0.213) |
| Collision with a person | 0.040 |
| Timeout rate | 0.500 |
| Path efficiency vs A* (successes only) | 0.926 (95% CI 0.885–0.969) |
| Time to goal (s, successes only) | 19.2 |
| Share of steps inside a person's comfort zone | 0.169 |
| Return: mean / IQM | -0.67 / -1.63 |
| Success std across seeds (stability) | 0.062 |
| Decision time (ms) | 0.46 |

- How episodes end: {'stuck_oscillating': 0.393, 'success': 0.353, 'too_slow': 0.107, 'collision_wall': 0.107, 'collision_human': 0.04}

- Success under stress conditions: nominal 0.353, bigger_map 0.353, careless_people 0.273, cluttered 0.307, crowded 0.293, fast_people 0.407, hardest_level 0.113, lidar_dropout_20 0.42, lidar_dropout_40 0.46, noisy_lidar 0.36, people_start_late 0.36

- Sample efficiency (area under validation curve, 0–1) per seed: [0.297, 0.241, 0.321]

- Your figures: `importance_q_learning.png`, `tuning_q_learning.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## sarsa

- Algorithm: `sarsa`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [426.9, 430.5, 449.9]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 9 |
| Success rate | 0.387 (95% CI 0.293–0.493) |
| Collision rate | 0.120 (95% CI 0.040–0.200) |
| Collision with a person | 0.027 |
| Timeout rate | 0.493 |
| Path efficiency vs A* (successes only) | 0.947 (95% CI 0.910–0.974) |
| Time to goal (s, successes only) | 19.2 |
| Share of steps inside a person's comfort zone | 0.174 |
| Return: mean / IQM | 0.34 / -0.51 |
| Success std across seeds (stability) | 0.052 |
| Decision time (ms) | 0.44 |

- How episodes end: {'success': 0.387, 'stuck_oscillating': 0.38, 'too_slow': 0.113, 'collision_wall': 0.093, 'collision_human': 0.027}

- Success under stress conditions: nominal 0.387, bigger_map 0.427, careless_people 0.3, cluttered 0.287, crowded 0.3, fast_people 0.367, hardest_level 0.1, lidar_dropout_20 0.44, lidar_dropout_40 0.467, noisy_lidar 0.4, people_start_late 0.34

- Sample efficiency (area under validation curve, 0–1) per seed: [0.344, 0.301, 0.368]

- Your figures: `importance_sarsa.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
