# Using the new robot application

1. Start **Start Robot Navigation.cmd** and visit http://127.0.0.1:8517.
2. In Simulator choose *Blocked aisle — wait for clearing*, Temporary and **Observed-Map Replanner (non-RL reference)**. Run once. It must observe that the nearby blocker cleared; it is not told the clearing time. Play/Pause/Step/Seek/Reset read saved frames only.
3. Compare viewer-only scene truth with robot memory. Grey occupied memory does not distinguish a wall from a pallet; pale cells are unknown. Sensor order N/S/E/W differs from action order Up/Down/Left/Right/Wait. A blocked attempted move is terminal collision.
4. Try *Storage rack detour* and *Temporary pallet detour*. Custom start/goal must be distinct reachable free cells away from scheduled blockers.
5. In Train Models explicitly start real Q-learning/SARSA/DQN/PPO Quick, then Refresh. Tiny models can fail: no fallback fixes their actions.
6. Compare four checkpoints on identical manual cases. Inspect static/dynamic denominators and download CSV plus replays. Blank recovery rates mean no opportunity, not zero performance.

Policies see409 public values, transformed to412 B1 features. Actions0..4 are Up,Down,Left,Right,Wait. Step reward−.01, goal bonus+10, collision penalty−5, maximum200 steps and no timeout bonus. Truth and schedules are only for trusted display/metrics.

History is read-only: [archive guide](../archive/README.md). Archived four-action models are not converted, old scores are not new comparison results, and no old UI mode is available. New outputs live in `autonomous-robot-navigation-rl/runs/`.
