# Warehouse robot navigation

A comparative application for **Q-learning, SARSA, DQN and PPO** using partial observations, agent memory and temporary blockages. The optional four-model voting ensemble and the observed-map replanner are separately labeled.

## Launch

Double-click **Start Robot Navigation.cmd**, or run from this repository root:

```powershell
py -3 run_app.py
```

Open **http://127.0.0.1:8517**. For installation, see [Windows setup](docs/windows-install.md). `py -3 run_app.py --check` checks imports without training or evaluation.

## Work on the project

Open **Robot Navigation.code-workspace**. The Desktop repository is the actual current `main` source tree; launching and editing no longer depend on a nested worktree or forwarding script.

| Location | Purpose |
| --- | --- |
| `autonomous-robot-navigation-rl/` | Current app, policies, data and saved models |
| `docs/` | Current user/developer guides, results and submission support |
| `docs/history/` | Previous implementation briefs, reports and validation records |
| `archive/` | Preserved historical data/source, catalogs and recovery snapshots |
| `scripts/` | Historical ZIP export and verification tools |

Start with [UI instructions](docs/user-guide.md), [how the code works](docs/code-walkthrough.md), [measured results](docs/results.md), or [the assignment status](docs/rubric-completion-status.md).

The app has Simulator, Train Models, Compare & Export, Guide and Archives. Training and development comparisons require an explicit action. Published held-out results and replay playback are read-only. Models are inference checkpoints, not exact optimizer-resume snapshots.

Historical evidence is preserved separately. Automated test files were removed at the owner's request; see [maintenance notes](docs/maintenance.md). No old application mode or legacy warehouse implementation is installed.
