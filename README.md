# Partial & Dynamic Robot Navigation

One application for a 10x10 robot with range-two occupancy sensing, memory and **Up / Down / Left / Right / Wait**. Four real RL families—Q-learning, SARSA, DQN and PPO—can be trained, simulated and compared. The Observed-Map Replanner and Random are separate non-RL references.

## Start on Windows

Follow [Windows installation](docs/windows-install.md) once, then double-click **Start Robot Navigation.cmd** or run `py -3 run_app.py`. Open **http://127.0.0.1:8517**. The launcher uses `autonomous-robot-navigation-rl/.venv`; `--python <path>` selects an existing compatible environment. Keep its terminal open.

- **Robot Simulator**: choose a preset, custom start/goal, Static/Temporary obstacles and controller. Run once, then Play/Pause/Back/Step/Seek/Reset saved frames. Scene truth is viewer-only; robot memory is separate. Collisions, loops and timeouts remain visible.
- **Train Models**: explicitly start a bounded quick or standard job, Refresh progress, Cancel safely and select new native checkpoints. No automatic training on rerun.
- **Compare & Export**: evaluate four checkpoints on identical manual development cases, with labeled references; download CSV, manifest and replay ZIP.
- **Guide**: sensing, stale memory, five actions, rewards, models and a 3–5-minute walkthrough.
- **Archives**: read-only historical reports and ZIP export instructions. No old simulator or model-transfer mode.

```powershell
cd autonomous-robot-navigation-rl
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset quick --models all
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset standard --models all
.\.venv\Scripts\python.exe -m robot_navigation.app_training --list-models
.\.venv\Scripts\python.exe -m robot_navigation.app_service --compare --include-random
```

Quick defaults to 256 actions/model; standard to 10,000 and must be user initiated. `--actions`, `--seed`, `--models` and `--output` are bounded. PPO requires a multiple of32 quick /100 standard actions; DQN a multiple of4. New immutable product runs are in `autonomous-robot-navigation-rl/runs/`. Saves are inference checkpoints, not exact-resume snapshots. Tiny starter models demonstrate the workflow, not strong navigation or scientific efficacy.

## All earlier history is preserved

The old four-action application is retired from the active package. **There is no Classic/Dynamic toggle.** [Archive guide](archive/README.md), [inventory](archive/INDEX.csv) and [integrity receipt](archive/manifests/integrity.json) cover classic results, Phase 5A–5H data, models, negative trials, logs, reports, source versions and unique local historical outputs. Sealed partial research remains at original paths; retired assets/source are archived. Original dirty-workspace files were copied read-only.

```powershell
python scripts/export_history_archives.py --list
python scripts/export_history_archives.py --output-dir history-exports
python scripts/export_history_archives.py --output-dir history-exports --verify
```

ZIPs stream to disk with inventories/checksums; duplicate giant ZIP blobs are not committed. Missing files/LFS payloads fail explicitly. Historical final/unseen evidence is reference-only and never mixed into new scores. Four-action models are not converted.

See [delivery report](USABLE_APP_DELIVERY.md), [user guide](docs/user-guide.md) and [developer guide](docs/developer-guide.md). Use focused integration tests, not archived research suites, to validate the app.
