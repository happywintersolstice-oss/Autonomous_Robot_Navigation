# Fact sheet: Member 3

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## a2c

- Algorithm: `a2c`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [1167.3, 1703.3, 1636.4]
- Hyper-parameters: `{'learning_rate': 0.0007, 'n_steps': 16, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 7 |
| Success rate | 0.480 (95% CI 0.360–0.593) |
| Collision rate | 0.520 (95% CI 0.407–0.640) |
| Collision with a person | 0.320 |
| Timeout rate | 0.000 |
| Path efficiency vs A* (successes only) | 0.979 (95% CI 0.958–0.995) |
| Time to goal (s, successes only) | 26.8 |
| Share of steps inside a person's comfort zone | 0.158 |
| Return: mean / IQM | 8.78 / 8.05 |
| Success std across seeds (stability) | 0.082 |
| Decision time (ms) | 1.44 |

- How episodes end: {'success': 0.48, 'collision_human': 0.32, 'collision_wall': 0.2}

- Success under stress conditions: nominal 0.48, bigger_hospital 0.447, careless_people 0.427, cluttered 0.467, crowded 0.36, fast_people 0.367, hardest_level 0.187, lidar_dropout_20 0.447, lidar_dropout_40 0.293, noisy_lidar 0.5, people_start_late 0.52

- Sample efficiency (area under validation curve, 0–1) per seed: [0.311, 0.175, 0.368]

- Your figures: `importance_a2c.png`, `saliency_a2c.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [1675.0, 1440.2, 1439.2]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 600, 'guidance': True}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 4 |
| Success rate | 0.540 (95% CI 0.400–0.667) |
| Collision rate | 0.433 (95% CI 0.320–0.560) |
| Collision with a person | 0.300 |
| Timeout rate | 0.027 |
| Path efficiency vs A* (successes only) | 0.986 (95% CI 0.971–0.996) |
| Time to goal (s, successes only) | 29.0 |
| Share of steps inside a person's comfort zone | 0.156 |
| Return: mean / IQM | 10.30 / 10.65 |
| Success std across seeds (stability) | 0.099 |
| Decision time (ms) | 2.51 |

- How episodes end: {'success': 0.54, 'collision_human': 0.3, 'collision_wall': 0.133, 'stuck_oscillating': 0.027}

- Success under stress conditions: nominal 0.54, bigger_hospital 0.493, careless_people 0.46, cluttered 0.587, crowded 0.453, fast_people 0.5, hardest_level 0.233, lidar_dropout_20 0.44, lidar_dropout_40 0.34, noisy_lidar 0.5, people_start_late 0.627

- Sample efficiency (area under validation curve, 0–1) per seed: [0.586, 0.562, 0.603]

- Your figures: `importance_ppo.png`, `saliency_ppo.png`, `tuning_ppo.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
