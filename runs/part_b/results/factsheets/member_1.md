# Fact sheet: Member 1

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_b/results` (test maps, nominal condition unless stated).

## q_learning_hybrid

- Algorithm: `q_learning`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [430.1, 426.8, 430.2]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 27 |
| Success rate | 0.460 (95% CI 0.373–0.553) |
| Collision rate | 0.220 (95% CI 0.147–0.293) |
| Collision with a person | 0.040 |
| Timeout rate | 0.320 |
| Path efficiency vs A* (successes only) | 0.952 (95% CI 0.925–0.974) |
| Time to goal (s, successes only) | 21.6 |
| Share of steps inside a person's comfort zone | 0.210 |
| Return: mean / IQM | 1.13 / 1.60 |
| Success std across seeds (stability) | 0.043 |
| Decision time (ms) | 0.46 |

- How episodes end: {'success': 0.46, 'stuck_oscillating': 0.26, 'collision_wall': 0.18, 'too_slow': 0.06, 'collision_human': 0.04}

- Success under stress conditions: nominal 0.46, bigger_map 0.513, careless_people 0.333, crowded 0.347, fast_people 0.433, hardest_level 0.147, lidar_dropout_20 0.573, lidar_dropout_40 0.593, people_start_late 0.44

- Sample efficiency (area under validation curve, 0–1) per seed: [0.354, 0.387, 0.395]

- Your figures: `importance_q_learning_hybrid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## q_learning_tuned

- Algorithm: `q_learning`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [465.5, 459.7, 464.4]
- Hyper-parameters: `{'alpha': 0.0986, 'gamma': 0.9655, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.4842}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 33 |
| Success rate | 0.360 (95% CI 0.280–0.453) |
| Collision rate | 0.193 (95% CI 0.107–0.300) |
| Collision with a person | 0.040 |
| Timeout rate | 0.447 |
| Path efficiency vs A* (successes only) | 0.927 (95% CI 0.891–0.962) |
| Time to goal (s, successes only) | 17.7 |
| Share of steps inside a person's comfort zone | 0.174 |
| Return: mean / IQM | -0.53 / -0.89 |
| Success std across seeds (stability) | 0.033 |
| Decision time (ms) | 0.46 |

- How episodes end: {'stuck_oscillating': 0.38, 'success': 0.36, 'collision_wall': 0.153, 'too_slow': 0.067, 'collision_human': 0.04}

- Success under stress conditions: nominal 0.36, bigger_map 0.453, careless_people 0.293, crowded 0.293, fast_people 0.367, hardest_level 0.147, lidar_dropout_20 0.453, lidar_dropout_40 0.447, people_start_late 0.34

- Sample efficiency (area under validation curve, 0–1) per seed: [0.337, 0.339, 0.361]

- Your figures: `importance_q_learning_tuned.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## sarsa_hybrid

- Algorithm: `sarsa`; action space: discrete; code: `src/medinav/agents/tabular.py`
- Seeds trained: [0, 1, 2]
- Budget: 4000 episodes per seed; network n/a (table)
- Training time per seed (s): [416.5, 401.5, 422.4]
- Hyper-parameters: `{'alpha': 0.1, 'gamma': 0.99, 'eps_start': 1.0, 'eps_end': 0.05, 'eps_decay_frac': 0.8}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 35 agents incl. baselines) | 21 |
| Success rate | 0.520 (95% CI 0.433–0.607) |
| Collision rate | 0.240 (95% CI 0.147–0.333) |
| Collision with a person | 0.073 |
| Timeout rate | 0.240 |
| Path efficiency vs A* (successes only) | 0.941 (95% CI 0.913–0.969) |
| Time to goal (s, successes only) | 21.6 |
| Share of steps inside a person's comfort zone | 0.223 |
| Return: mean / IQM | 1.95 / 3.37 |
| Success std across seeds (stability) | 0.033 |
| Decision time (ms) | 0.53 |

- How episodes end: {'success': 0.52, 'stuck_oscillating': 0.18, 'collision_wall': 0.167, 'collision_human': 0.073, 'too_slow': 0.06}

- Success under stress conditions: nominal 0.52, bigger_map 0.667, careless_people 0.4, crowded 0.38, fast_people 0.54, hardest_level 0.28, lidar_dropout_20 0.687, lidar_dropout_40 0.687, people_start_late 0.54

- Sample efficiency (area under validation curve, 0–1) per seed: [0.409, 0.414, 0.489]

- Your figures: `importance_sarsa_hybrid.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
