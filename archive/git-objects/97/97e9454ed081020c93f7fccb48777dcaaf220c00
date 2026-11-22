# Autonomous Robot Navigation — milestone 2

A local Python research pilot: one versioned static grid, Gymnasium environment,
random/BFS references, Q-learning and SARSA, reproducible scenario splits, measured validation,
and an animated Streamlit replay viewer. Existing workspace files are separate.

**Current development pilots:** three independent seeds per algorithm, 200,000
interactions each. Mean validation success is 66.45% for Q-learning and 25.68% for
SARSA. Seed-7 repeats reproduce tables/logs/replays exactly. These are preliminary
tabular results; the three-agent Part A comparison is still later. Timeouts remain
common. The final test has never been executed. See [milestone 2 results](docs/milestone2_results.md).
The original [30,000-interaction pilot](docs/milestone_results.md) and artifacts
remain unchanged.

## Windows setup

From this project directory in PowerShell (tested with Python 3.14.10):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m robot_navigation.cli check
```

The lock contains the tested dependencies and build tools, without an editable
absolute path. Python 3.11+ is declared; only the installed 3.14.10 interpreter was
tested here. Later SB3/PyTorch requirements will need their own compatibility check.
No API key, GPU or cloud account is required.

## Open the saved demonstration, without training

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8517 --server.headless true --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8517**. Binding is local only. Stop with Ctrl+C. Change the
port if occupied. Select `bfs success`, press Play, and watch each discrete action.
The library also has random and actual Q-learning success/collision/timeout replays.
Play, Pause, Next step, Reset, speed and replay seeking all use the same stored
transitions; changing display controls preserves the scenario and current frame.
Seeking pauses playback. At the end, Reset then Play to watch again. The map shows
start/goal markers, robot, obstacles, coordinates, traveled trail, and collision
attempts. Position, action, step reward, cumulative reward and status stay together.
The future route stays hidden. Completed SPL is shown only at the terminal frame.

Custom demonstrations accept distinct, reachable free start/goal cells on the
unchanged map, then run BFS, random or the frozen saved Q-table once and replay it.
They never update the policy or benchmark. The viewer rejects incompatible map,
task, reward or observation/action versions and corrupted checkpoints/replays.
No map editor or final-test scenario browser is provided in this milestone.

The viewer supports both checkpoint algorithms, including old Q-learning files.
Saved runs named `m2-q-learning-*` and `m2-sarsa-*` contain the new learned episodes.
Playback never updates a table. Browser verification and any remaining limitations
are recorded in the milestone 2 results.

## Reproduce the task and run a new pilot

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.cli prepare
.\.venv\Scripts\python.exe -m robot_navigation.cli check
Copy-Item configs/smoke.yaml configs/my-smoke.yaml
```

Set a **new `run_id`** in `configs/my-smoke.yaml` (for example `my-sarsa-seed7`),
add `algorithm: sarsa` (or `q-learning`), and choose the budget. The epsilon schedule,
alpha, gamma and seed are also configurable there. Both algorithms share one loop.

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.cli train --config configs/my-smoke.yaml
```

`smoke` remains an alias for `train`, defaulting to `configs/smoke.yaml` and
Q-learning when `algorithm` is absent. Do not rerun the original run ID in place.

Read-only evaluation of a saved table, without training:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.cli evaluate --checkpoint artifacts/runs/m2-sarsa-seed7/checkpoint.npz --split validation
```

Only `validation` and `monitoring` are accepted; there is no final-test option.
To reproduce the full predeclared development matrix in a fresh output namespace:

```powershell
.\.venv\Scripts\python.exe scripts/run_tabular_milestone.py --suite-id milestone2-repeat --run-prefix m2-repeat
```

Use a new suite ID and prefix each time. The runner refuses to overwrite completed
runs; the original matrix, protocol, results, runner and audits are saved in
`artifacts/studies/milestone2-tabular-v1/`.

`prepare` deterministically regenerates the same versioned manifests, map image
and benchmark report. Do not change a benchmark file in place for experiments;
create a new version if the task changes. `train` refuses to overwrite a run,
trains only from training pairs, monitors the saved validation subset, saves its
final table, and evaluates all validation pairs with learning disabled. When enabled,
reference agents are evaluated on validation; it never executes the test holdout.
It also exports all validation trajectories in compressed JSONL directly from the
evaluation, representative replay files, trajectory
images, training CSV, visitation counts, configuration, dependency/hardware/source
metadata, checksums, goal discoveries, throughput and initial-state overlap. The
training log covers completed episodes; any budget-cutoff episode is recorded
separately. Failed/interrupted runs retain their status and periodic snapshots.
All model snapshots are inference-only; exact training resume is unsupported.

## Inspect saved evidence

- [Benchmark report](data/scenarios/benchmark_report.json): map identity, connectivity,
  distance bins/histogram, exact split sizes, per-goal frequencies and split hashes.
- [Environment specification](docs/environment_specification.md): transitions, spaces,
  seeding, timeout semantics, encoding and metrics.
- [Validation summary](artifacts/runs/smoke-seed7/validation.json) and
  [pilot diagnostics](artifacts/runs/smoke-seed7/pilot.json).
- [Training log](artifacts/runs/smoke-seed7/training.csv),
  [monitoring](artifacts/runs/smoke-seed7/monitoring.json),
  [run manifest](artifacts/runs/smoke-seed7/manifest.json),
  [BFS route image](artifacts/runs/smoke-seed7/bfs_trajectory.png).
- [Milestone 2 protocol](docs/milestone2_protocol.md),
  [three-seed summaries](artifacts/studies/milestone2-tabular-v1/summary.json), and
  [reproducibility/trajectory audits](artifacts/studies/milestone2-tabular-v1/verification.json).
- [Development log](docs/development_log.md), [milestone 1 checks](docs/verification.json),
  [milestone 2 checks](docs/milestone2_verification.json), and
  [milestone 2 results](docs/milestone2_results.md).

Small saved demo artifacts are included; caches, visitation arrays and bulky full
trajectory JSONL files remain local and are ignored by Git. No remote publication
or deployment was performed. The benchmark is provisional until pilot review;
all results concern held-out initial conditions on a familiar map. Most validation
initial states were visited during training, so this is not unseen-state transfer.

DQN, the full Part A comparison, refinement, voting and the full dashboard are later
milestones. PPO, unseen maps, moving obstacles, continuous movement and physical
robots remain deferred. No such extensions are implemented here.
