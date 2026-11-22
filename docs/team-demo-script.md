# Four-person product demonstration (3-5 minutes)

Use the existing app at http://127.0.0.1:8517. The saved 24-layout product-development panel was used to choose checkpoints. Its scores are not unseen scientific accuracy. Each member explains and operates one genuine RL family; the planner is a separate non-RL reference.

| Member | Responsibility | Selected model actions | Panel success | Checked simulator example |
|---|---|---:|---:|---|
| 1 | Q-learning | 30,000 | 23/24 | Open loading floor: success in 14 steps |
| 2 | SARSA | 10,000 | 22/24 | Storage rack detour: success in 17 steps |
| 3 | DQN | 10,000 | 18/24 | Storage rack detour: success in 21 steps |
| 4 | PPO | 30,000 | 21/24 | Blocked aisle, Temporary: success in 13 steps, including 6 Waits |

1. **Member 1 (45-60 seconds):** open Robot Simulator, select the Best on development Q-learning checkpoint and Open loading floor. Run journey, Play, Pause and Step. Show Robot memory versus scene truth. Q-learning learns a TD action value; the selected variant uses frozen targets to control the observed unstable baseline. All preferences are learned; there is no masked or scripted route.
2. **Member 2 (45-60 seconds):** choose Best on development SARSA and Storage rack detour. Run, Seek and End. Explain SARSA's target uses its sampled next behavior action, whereas Q-learning uses a greedy target. These are independent learned models, even though both use the same public feature encoding. Its selected model uses disclosed train-only progress feedback; the displayed reward is always official.
3. **Member 3 (45-60 seconds):** select Best on development DQN and Storage rack detour. Show its longer 21-step real route. Explain replay, neural Q values and target updates. Be candid: DQN timed out on the Open loading floor preset in the post-selection demo check, and on 6/24 development layouts. Keep its own failed journey visible; never switch silently to a planner.
4. **Member 4 (60-90 seconds):** choose Best on development PPO and Blocked aisle, Temporary. Run and step through six Wait actions, then clearing and goal arrival. The robot receives occupancy only: permanent walls and temporary pallets are indistinguishable while blocked. It does not know when clearing will happen. PPO learns action probabilities and a value model from rollouts. Open Compare & Export, load the saved panel, inspect starter/new counts and static/dynamic subsets, then download summary CSV and replay ZIP.

The four original starters succeeded on 0/24, 1/24, 3/24 and 1/24 on this same panel; the selected models reached 23/24, 22/24, 18/24 and 21/24. The observed-map planner reached 24/24 using only remembered occupancy, with no policy fallback role. Show it only when explicitly labeled non-RL.

Playback reads stored frames; Reset does not retrain or rerun the environment. Quick training creates a starter and is optional for this short demonstration; use the delivered selected checkpoints rather than waiting for a new job. Training starts only after the explicit button or CLI command. Standard is bounded at 50,000 actions/model and a one-hour worker limit; cancellation preserves a partial run without publishing its models.

Actual post-selection preset checks (16 episodes, kept separate from selection) are in `runs/product-phase2/20261122T132126Z-41f7725b/manual-demo/demo-checks.json`, with real replays. Three families succeeded on all four presets; DQN failed the open-floor case. That is a demonstration limitation, not a hidden success.
