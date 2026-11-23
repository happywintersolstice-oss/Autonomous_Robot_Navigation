# Fact sheet: Member 1

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## q_learning

- Algorithm: `q_learning`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [197.0, 194.4, 206.2]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 8 |
| Success rate | 0.287 (95% CI 0.213–0.367) |
| Collision rate | 0.487 (95% CI 0.393–0.587) |
| Collision with a person | 0.293 |
| Timeout rate | 0.227 |
| Path efficiency vs A* (successes only) | 0.979 (95% CI 0.954–0.993) |
| Time to goal (s, successes only) | 16.2 |
| Share of steps inside a person's comfort zone | 0.134 |
| Return: mean / IQM | -2.04 / -4.99 |
| Success std across seeds (stability) | 0.025 |
| Decision time (ms) | 0.38 |

- How episodes end: {'collision_human': 0.293, 'success': 0.287, 'stuck_oscillating': 0.207, 'collision_wall': 0.193, 'too_slow': 0.02}

- Success under stress conditions: nominal 0.287, bigger_map 0.287, cluttered 0.187, crowded 0.127, fast_people 0.233, hardest_level 0.0, lidar_dropout_20 0.353, lidar_dropout_40 0.353, noisy_lidar 0.267, people_start_late 0.287

- Sample efficiency (area under validation curve, 0–1) per seed: [0.198, 0.221, 0.183]

- Your figures: `importance_q_learning.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## sarsa

- Algorithm: `sarsa`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [206.6, 219.9, 214.4]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 9 |
| Success rate | 0.273 (95% CI 0.180–0.380) |
| Collision rate | 0.540 (95% CI 0.460–0.627) |
| Collision with a person | 0.273 |
| Timeout rate | 0.187 |
| Path efficiency vs A* (successes only) | 0.955 (95% CI 0.919–0.989) |
| Time to goal (s, successes only) | 14.4 |
| Share of steps inside a person's comfort zone | 0.130 |
| Return: mean / IQM | -2.41 / -5.95 |
| Success std across seeds (stability) | 0.062 |
| Decision time (ms) | 0.30 |

- How episodes end: {'collision_human': 0.273, 'success': 0.273, 'collision_wall': 0.267, 'stuck_oscillating': 0.173, 'too_slow': 0.013}

- Success under stress conditions: nominal 0.273, bigger_map 0.287, cluttered 0.12, crowded 0.153, fast_people 0.233, hardest_level 0.04, lidar_dropout_20 0.347, lidar_dropout_40 0.34, noisy_lidar 0.28, people_start_late 0.293

- Sample efficiency (area under validation curve, 0–1) per seed: [0.221, 0.233, 0.189]

- Your figures: `importance_sarsa.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
