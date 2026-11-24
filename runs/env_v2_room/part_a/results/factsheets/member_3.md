# Fact sheet: Member 3

Measured facts for your algorithms. Interpretation and justification are yours to write.
Source: `runs/part_a/results` (test maps, nominal condition unless stated).

## a2c

- Algorithm: `a2c`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [546.3, 665.3, 614.2]
- Hyper-parameters: `{'learning_rate': 0.0007, 'n_steps': 16, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 8 |
| Success rate | 0.433 (95% CI 0.353–0.513) |
| Collision rate | 0.467 (95% CI 0.367–0.560) |
| Collision with a person | 0.120 |
| Timeout rate | 0.100 |
| Path efficiency vs A* (successes only) | 0.867 (95% CI 0.804–0.922) |
| Time to goal (s, successes only) | 15.9 |
| Share of steps inside a person's comfort zone | 0.128 |
| Return: mean / IQM | 1.39 / 0.59 |
| Success std across seeds (stability) | 0.019 |
| Decision time (ms) | 0.78 |

- How episodes end: {'success': 0.433, 'collision_wall': 0.347, 'collision_human': 0.12, 'too_slow': 0.1}

- Success under stress conditions: nominal 0.433, bigger_map 0.5, careless_people 0.393, cluttered 0.427, crowded 0.327, fast_people 0.427, hardest_level 0.16, lidar_dropout_20 0.433, lidar_dropout_40 0.34, noisy_lidar 0.413, people_start_late 0.493

- Sample efficiency (area under validation curve, 0–1) per seed: [0.427, 0.309, 0.379]

- Your figures: `importance_a2c.png`, `saliency_a2c.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## ppo

- Algorithm: `ppo`; action space: continuous; code: `src/medinav/agents/policy_gradient.py`
- Seeds trained: [0, 1, 2]
- Budget: 300000 environment steps per seed; network [128, 128]
- Training time per seed (s): [728.8, 738.4, 608.1]
- Hyper-parameters: `{'learning_rate': 0.0003, 'n_steps': 512, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 'ent_coef': 0.005}`
- Env overrides / wrappers: `{'difficulty': 2, 'max_steps': 400}` / `{}`

| Metric | Value |
|---|---|
| Rank (by success, of 11 agents incl. baselines) | 7 |
| Success rate | 0.487 (95% CI 0.400–0.580) |
| Collision rate | 0.413 (95% CI 0.307–0.527) |
| Collision with a person | 0.080 |
| Timeout rate | 0.100 |
| Path efficiency vs A* (successes only) | 0.962 (95% CI 0.937–0.985) |
| Time to goal (s, successes only) | 14.4 |
| Share of steps inside a person's comfort zone | 0.143 |
| Return: mean / IQM | 2.18 / 2.89 |
| Success std across seeds (stability) | 0.034 |
| Decision time (ms) | 0.76 |

- How episodes end: {'success': 0.487, 'collision_wall': 0.333, 'collision_human': 0.08, 'too_slow': 0.067, 'stuck_oscillating': 0.033}

- Success under stress conditions: nominal 0.487, bigger_map 0.56, careless_people 0.453, cluttered 0.5, crowded 0.36, fast_people 0.513, hardest_level 0.173, lidar_dropout_20 0.453, lidar_dropout_40 0.433, noisy_lidar 0.48, people_start_late 0.533

- Sample efficiency (area under validation curve, 0–1) per seed: [0.461, 0.477, 0.504]

- Your figures: `importance_ppo.png`, `saliency_ppo.png`, `tuning_ppo.png`
- Shared figures: `comparison.png`, `failure_breakdown.png`, `learning_curves.png`, `robustness_success.png`, `trajectories.png`

## Questions to prepare for the Q&A (answer them yourself)

- Why does this algorithm suit (or not suit) LiDAR navigation among moving people?
- Which hyper-parameter would you change first, and what do you expect to happen?
- What do the failure types say about what the agent has not learned?
- Is the difference to the next-best agent statistically meaningful (see `significance.md`)?
