# Fourth RL algorithm: PPO comparison protocol

Owner requested four trained RL algorithms and explicitly selected PPO on
8 November 2026. This expands the earlier three-algorithm scope. The original
milestone-6 archive and all previous runs remain immutable.

Use vanilla Stable-Baselines3 PPO with a categorical four-action policy and
separate two-layer 64-unit actor/critic networks. Input is the same four
coordinates, divided by dimension minus one, as DQN. No map input, action mask,
planner fallback, reward shaping or environment change. Deterministic evaluation
chooses the greatest action probability; displayed probabilities are not Q-values.

Freeze the single configuration in configs/ppo_comparison.yaml before training.
It is a configured baseline, not a hyperparameter search or a tuned PPO claim.
Use one independent software pilot (seed 101, 10,000 interactions), then main
seeds 7, 19, 43, 67, 91 with exactly 200,000 environment interactions each.
The 1,000-step rollout length divides the budget exactly; batch size 100 divides
each rollout. One CPU environment and one PyTorch thread. Fixed learning rate,
ten optimization epochs per rollout, normalized advantages, no value clipping,
no target-KL stopping and the explicitly recorded SB3 defaults.

Monitor the existing saved 30-pair validation subset every 10,000 interactions,
after the preceding PPO update and before collecting the next rollout. Include
the final post-update checkpoint. Select by monitor success, SPL, then earliest
interaction, matching the historical selection rule. Complete the whole budget.
Validate the selected checkpoint on all 928 development-validation pairs.
Save every checkpoint, training ledger, visitation, probabilities, trajectories,
replays and run/source manifests. Exact training resume is not supported.

Compare all five PPO seeds with the saved Part A Q-learning, SARSA and DQN seeds
7, 19, 43, 67, 91. Verify matching task, scenario IDs, budgets and historical
artifact hashes; do not retrain or overwrite them. Report seed means, sample SD,
individual seeds and outcome counts, including all failures. The main table uses
the original baseline configurations, not the separately tuned tabular runs.
Equal interaction budgets do not imply equal update counts or compute cost.

This is a post-release, exploratory extension: validation and the old final
results have already been inspected. Do not reopen the final-test manifest,
execute its policies, alter the selected system, rewrite its freeze or present
these results as a newly blinded final evaluation. A new independent evaluation
would be needed for a new confirmatory performance claim. New holdout creation,
PPO tuning, moving obstacles, unseen maps and hardware remain outside this task.

Bound each run to 1,200 wall seconds and the main suite to 2,700 seconds / 1 GiB
of new artifacts. Record actual pilot throughput before running the main matrix.
If limits are exceeded, stop and report partial actual results without changing
the budget or settings to obtain a better score.

Implementation reference: [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
