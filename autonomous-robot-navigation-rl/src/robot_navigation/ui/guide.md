### What this application does

Compare four independently trained warehouse-navigation robots on the same routes. A robot must reach G from R, avoid occupied cells, and finish within 200 actions. Temporary blockers may disappear, but the robot is never told their type or clearing time. Scene truth is a human-only view; Robot memory shows what the controller can actually know.

### Find your way around

- **Robot Simulator:** choose a map, Static/Temporary obstacles and a controller. Phase 3 selected checkpoints are the current product choices; Phase 2 baseline checkpoints remain available for comparison. Run journey records an episode once. Play, Pause, Step, Back, Seek, Reset and End only display its saved frames. A timeout or collision remains the actual recorded result.
- **Train Models:** inspect saved checkpoints and verify their task/version/checksum. Training starts only with Start training. Quick is a small starter, not a promise of good navigation. Refresh reads progress; Cancel requests a safe stop. Closing a browser does not stop its worker.
- **Compare & Export:** load saved Phase 2/Phase 3 development evidence or the once-published Phase 3 product held-out assessment. The held-out view is read-only. Summary/case CSV, JSON and replay ZIP keep the evidence, including failed routes. Choose Failures and Open recorded journey in Simulator to inspect a failure.
- **Guide:** read these instructions and the checked four-person demonstration.
- **Archives:** read earlier research and ZIP export instructions. Historical experiments are reference material and are not selectable old application modes.

### What the robot sees and learns

The unchanged public observation has 409 values: position and goal, four range-two sensor rays, remembered occupancy, observation age, visits, blocked-observation counts and elapsed steps. Occupancy alone cannot identify a permanent wall versus a temporary blocker. Phase 3 adds only eight past public decision positions, the last action and consecutive Waits: 431 internal state values and 520 action features. Phase 2 keeps its original 480-feature format.

- **Q-learning:** learns action values through temporal-difference updates, choosing a greedy next action and evaluating it using lagged target weights.
- **SARSA:** updates using the actual next action sampled by its current exploratory behavior.
- **DQN:** learns neural action values with experience replay and a target network.
- **PPO:** learns a neural actor and critic from rollouts using a clipped policy objective and entropy.

All five actions stay possible: Up, Down, Left, Right and Wait. There is no hidden planner fallback or collision mask. The separately labeled Observed-Map Replanner is a non-RL reference, not a substitute for a failed learner.

Every action costs -0.01; reaching the goal adds +10; a blocked move adds -5 and ends the episode. Timeout at 200 actions adds no extra penalty. Optional training feedback is disclosed separately and does not change official evaluation rewards. Successes, collisions, timeouts, Waits, movement revisits and successful-only steps must be read together: a low Wait percentage can hide movement loops.

### A 3-5 minute walkthrough
1. Choose a **Phase 3 selected** checkpoint explicitly in Simulator. Member 1: Q-learning, *Temporary bridge - Wait*; member 2: SARSA, *New static maze*; member 3: DQN, *Temporary gate - detour*; member 4: PPO, *Permanent dead end with detour*. Keep Temporary mode and seed 0. These trajectories were checked before held-out evaluation.
2. Run once, then Play/Pause/Step/Seek saved frames. Compare human-only truth with robot memory. Wait is correct on the bridge; all five actions remain available.
3. Show Q-learning's honestly timed-out *New static maze*. Do not switch to the replanner and call it RL success.
4. In Compare & Export, view the saved **Phase 3 product held-out** assessment, compare Phase2/Phase3 on the same panel, download CSV/JSON/ZIP and open a failure replay. Held-out viewing never reruns evaluation.
5. Train Models shows real feature versions, actions and integrity checks. Start training only when explicitly intended; no live training is needed for this presentation. See the four-person script below for checked actions/Waits and exact checkpoint IDs.

### History is reference material
The old coordinate-only four-action application has been retired. Its data, source, failed trials and Phase 5A–5H history remain accessible through Archives and the repository archive guide. Older models can be incompatible with this task. Archived scores are never presented as results of new app training. There is no old application mode or model-transfer adapter.
