# Four-member demonstration: partially observed dynamic robot navigation

Allow 3-5 minutes. Start at the repository root with `py -3 run_app.py`, or double-click **Start Robot Navigation.cmd**, then open http://127.0.0.1:8517. If using an existing environment, supply `--python <path-to-python.exe>`. Leave the launcher terminal open. Do not start training or rerun comparisons during this presentation.

Use **Robot Simulator**, **Temporary** obstacle mode, default start/goal and evaluation seed **0** for all four demonstrations. Explicitly select each **Phase 3 selected** checkpoint below. None uses a planner fallback. Presets are independently constructed demo/development scenes, not copied from the frozen held-out panel. The checked trajectories were recorded before the final product test; selection was already locked.

| Member | Family | Preset | Verified outcome | Actions | Waits | Checkpoint |
|---|---|---|---|---:|---:|---|
| 1 | q-learning | Temporary bridge - Wait | success | 13 | 6 | `20261122T165606Z-ea2dd627/q-learning` |
| 2 | sarsa | New static maze | success | 34 | 0 | `20261122T165803Z-b0736b3d/sarsa` |
| 3 | dqn | Temporary gate - detour | success | 9 | 0 | `20261122T165937Z-f7774052/dqn` |
| 4 | ppo | Permanent dead end with detour | success | 28 | 0 | `20261122T165456Z-9875f272/ppo` |

1. **Member 1, Q-learning (45-60 seconds):** choose *Temporary bridge - Wait*, run once, Play, then Pause/Step across the six Waits. Explain that linear Q-learning learns action values through TD updates and bootstraps a greedy next-action value with a lagged target. The occupied cell looks like a wall; the policy has no clearing schedule. Show the blue robot memory and the separate human-only truth.
2. **Member 2, SARSA (45-60 seconds):** choose *New static maze*. Explain that SARSA uses the actual next action sampled by its exploratory behavior policy, rather than Q-learning's greedy bootstrap. Seek through a turn or revisit. This long winding route was genuinely completed in 34 actions.
3. **Member 3, DQN (45-60 seconds):** choose *Temporary gate - detour*. Show the nine-action, zero-Wait route around the blockage. DQN learns neural action values with replay, exploration decay and target-network updates; the model independently chose this route. Do not suggest every DQN route is efficient: on *Permanent dead end with detour*, this checkpoint required 199 actions including 113 Waits.
4. **Member 4, PPO (45-60 seconds):** choose *Permanent dead end with detour*. Show the 28-action route with zero Waits. Explain the actor/critic, rollout updates, clipped policy objective and entropy, rather than value-maximization at training time. Then open **Compare & Export**, select **Phase 3 product held-out**, show same-panel Phase2/Phase3 rows, download summary/cases/JSON/ZIP, and open an honestly failed saved journey.

For the failure segment, use Q-learning on *New static maze*: **timeout at 200 actions**, including **19 Waits**. It also times out on the permanent-detour preset. Playback/seek only reads saved frames. In Compare & Export, choose **Failures**, select a labeled RL route, then **Open recorded journey in Simulator**. Do not hide a failed RL episode by swapping in the separately labeled Observed-Map Replanner.

All models see the original 409 public values. Phase3 builds a versioned 431-value internal state using eight past decision positions, the last action and a Wait counter, then 520 action features. All five actions, including collision-causing moves, remain possible. Occupancy history does not identify permanent versus temporary obstacles, and memory can be stale. Checkpoints are inference artifacts, not exact optimizer-resume snapshots.

The results page is read-only for the frozen product test. Development scores were used for selection; the new 64-case held-out results are a separate, single final assessment. See [PRODUCT_PHASE3_RESULTS.md](../PRODUCT_PHASE3_RESULTS.md) for measured rates and limitations. Never describe the older Phase2 24-case selection panel as unseen accuracy.

In **Train Models**, inspect feature versions/action counts, choose a checkpoint integrity check and verify it. A real training job requires an explicit Start; Refresh displays worker progress and Cancel requests termination. No live training is necessary for the demo. **Archives** remains read-only historical reference; Phase5A-H evidence and all Phase2 checkpoints are preserved.

Checked replay evidence (relative to application directory):

- q-learning: [Temporary bridge - Wait](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/demo-checks/episodes/20261122T170947Z-206da965.json.gz).
- sarsa: [New static maze](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/demo-checks/episodes/20261122T170948Z-e6392d99.json.gz).
- dqn: [Temporary gate - detour](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/demo-checks/episodes/20261122T170951Z-99316f63.json.gz).
- ppo: [Permanent dead end with detour](../autonomous-robot-navigation-rl/runs/product-phase3/20261122T163932Z-f85cb312/demo-checks/episodes/20261122T170951Z-bc76d75e.json.gz).

The previous Phase2 script is preserved byte-for-byte in the Phase3 study folder as `phase2-team-demo-script.md`, and remains in the Phase2 Git history.
