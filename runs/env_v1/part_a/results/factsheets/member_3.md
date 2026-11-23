# Fact sheet: Member 3

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## a2c

- Algorithm: `a2c`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [278.6, 284.4, 270.2]
- Hyper-parameters: `{'learning_rate': 0.0007, 'n_steps': 16, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 5 |
| Success rate | 0.460 (95% CI 0.373–0.553) |
| Collision rate | 0.513 (95% CI 0.427–0.593) |
| Collision with a person | 0.187 |
| Timeout rate | 0.027 |
| Path efficiency vs A* (successes only) | 0.956 (95% CI 0.921–0.980) |
| Time to goal (s, successes only) | 10.7 |
| Share of steps inside a person's comfort zone | 0.119 |
| Return: mean / IQM | 2.15 / 1.86 |
| Success std across seeds (stability) | 0.033 |
| Decision time (ms) | 0.85 |

- How episodes end: {'success': 0.46, 'collision_wall': 0.327, 'collision_human': 0.187, 'stuck_oscillating': 0.02, 'too_slow': 0.007}

- Success under stress conditions: nominal 0.46, bigger_map 0.5, cluttered 0.373, crowded 0.293, fast_people 0.373, hardest_level 0.093, lidar_dropout_20 0.42, lidar_dropout_40 0.4, noisy_lidar 0.473, people_start_late 0.547

- Sample efficiency (area under validation curve, 0–1) per seed: [0.25, 0.329, 0.362]

- Your figures: `importance_a2c.png`, `saliency_a2c.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [326.8, 327.4, 278.9]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 6 |
| Success rate | 0.453 (95% CI 0.373–0.540) |
| Collision rate | 0.527 (95% CI 0.447–0.607) |
| Collision with a person | 0.207 |
| Timeout rate | 0.020 |
| Path efficiency vs A* (successes only) | 0.955 (95% CI 0.912–0.992) |
| Time to goal (s, successes only) | 12.3 |
| Share of steps inside a person's comfort zone | 0.116 |
| Return: mean / IQM | 1.64 / 1.33 |
| Success std across seeds (stability) | 0.025 |
| Decision time (ms) | 0.73 |

- How episodes end: {'success': 0.453, 'collision_wall': 0.32, 'collision_human': 0.207, 'stuck_oscillating': 0.013, 'too_slow': 0.007}

- Success under stress conditions: nominal 0.453, bigger_map 0.48, cluttered 0.427, crowded 0.3, fast_people 0.4, hardest_level 0.067, lidar_dropout_20 0.447, lidar_dropout_40 0.393, noisy_lidar 0.46, people_start_late 0.5

- Sample efficiency (area under validation curve, 0–1) per seed: [0.413, 0.431, 0.436]

- Your figures: `importance_ppo.png`, `saliency_ppo.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
