# Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO

This protocol is registered before running the new twelve-run comparison. The existing single-seed warehouse model is prior exploratory work and is excluded from this matrix. The study concerns the familiar `warehouse-delivery-v1` floor, not unseen warehouses, physical robots or an independent final assessment.

## Common comparison

All four algorithms receive the same 715 training start–goal pairs, the same 188 development assessment pairs, the same 13×17 floor, seven destination stations, four movement actions, terminal collisions and a 200-action cap. Rewards are -0.05 per action, +10 at the goal and -5 on collision. Discount is 0.99. Tabular algorithms use raw robot/goal coordinates; neural algorithms scale the same four coordinates by grid dimension minus one. No learner receives the obstacle map, an action mask, a route planner, a loop escape rule or extra features. BFS is a separate known-map reference.

Seeds are 19411, 19421 and 19431 for each family. Every run trains from scratch for exactly 600,000 environment interactions: 7.2 million across twelve runs. Final-budget checkpoints are retained. Monitoring is descriptive and never selects a checkpoint, algorithm or seed. The first registered seed is the default playback seed, regardless of results. Every failure and timeout remains accessible.

All settings are in `configs/warehouse_comparison.yaml`. Q-learning and SARSA use learning rate 0.3 and epsilon decreasing from 1 to 0.03 over 80% of the budget. DQN is the existing vanilla SB3 implementation, with a 64×64 network, 50,000-entry replay buffer, batch 64, one gradient update every four actions and target updates every 1,000 actions. PPO uses the existing categorical actor/value implementation, 64×64 actor/value networks, complete 1,000-step rollouts, batch 100 and ten epochs. Neural learning rates are 0.0003. Optimizer updates and wall times differ by algorithm; equal interaction budgets do not imply equal compute.

## Monitoring, recording and assessment

A fixed 35-route monitoring set includes all seven named jobs and deterministic additional development routes. Monitoring occurs every 100,000 actions, without modifying policies. DQN intermediate checks occur at its callback boundary before that step's replay/gradient storage; PPO checks occur after the preceding complete rollout update. The final check for each algorithm occurs after its final scheduled update. Actual monitoring phase labels are saved.

Six chronological training episodes are recorded per run: the first two starting at or after each boundary 0, 200,000 and 400,000. Sampling never depends on outcome. Budget-cutoff prefixes remain incomplete recordings, not invented completed episodes. These actions include exploration and changing policies; frozen-model inference must not be substituted for training history.

Each reloaded final checkpoint is evaluated deterministically on all 188 development routes, in the same order with the same evaluation seed. Raw trajectories, actions, scores/probabilities and outcomes are saved. Their dynamics and metrics are audited by replaying their recorded actions through the environment. Table or neural-state fingerprints must remain unchanged during evaluation. Source, map, route, configuration and output hashes are retained. Existing final cohorts are never executed or overwritten.

Report completion, collision, timeout and SPL as equal-weight training-seed means with sample SD across three seeds. Report conditional successful-route length with its contributing seed count. Also report training interactions, completed training episodes, loop/revisit counts, checkpoint bytes and descriptive timings. Routes are repeated assessment cases, not independent training replicates. The 188 development routes and named jobs were already inspected in prior single-seed work; do not relabel them as a fresh final holdout.

## Execution and limits

At most two CPU worker processes run concurrently; each neural worker uses one torch thread. Record this when interpreting wall times. Each run has a 1,800-second training wall limit. Preserve any failed or interrupted run with its reason, and do not overwrite or silently replace it. No configuration search, post-result reward change or promotion of a favourable training seed is part of this comparison. Extensions require a separately versioned protocol and output namespace.

The primary interface, report and current documentation adopt the warehouse project title. Older abstract-map and multi-layout studies remain labelled research archives and retain their original identifiers and checksums.
