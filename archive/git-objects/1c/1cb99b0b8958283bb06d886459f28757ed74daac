# Autonomous Robot Navigation — milestone 3 / Part A

A local Python research pilot: one versioned static grid, Gymnasium environment,
random/BFS references, Q-learning, SARSA and vanilla SB3 DQN, reproducible scenario splits, measured validation,
and an animated Streamlit replay viewer. Existing workspace files are separate.

**Part A:** five independent seeds per learned algorithm, 200,000 interactions
each, common validation monitoring, and an explicit validation-only checkpoint
and algorithm-family selection rule. See [milestone 3 results](docs/milestone3_results.md)
for measured outcomes, failure analysis and the two-family shortlist. The final
test has never been executed. The original [30k pilot](docs/milestone_results.md)
and [tabular development runs](docs/milestone2_results.md) remain unchanged.

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
tested here. The complete lock now includes Stable-Baselines3 2.9.2 and CPU
PyTorch 2.15.0, tested on this interpreter. The optional `dqn` dependency group
in `pyproject.toml` declares the neural-agent requirements.
No API key, GPU or cloud account is required.

## Open the saved demonstration, without training

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8517 --server.headless true --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8517**. Binding is local only. Stop with Ctrl+C. Change the
port if occupied. Select `bfs success`, press Play, and watch each discrete action.
The library also has random and actual learned-agent success/collision/timeout replays.
Play, Pause, Next step, Reset, speed and replay seeking all use the same stored
transitions; changing display controls preserves the scenario and current frame.
Seeking pauses playback. At the end, Reset then Play to watch again. The map shows
start/goal markers, robot, obstacles, coordinates, traveled trail, and collision
attempts. Position, action, step reward, cumulative reward and status stay together.
The future route stays hidden. Completed SPL is shown only at the terminal frame.

Custom demonstrations accept distinct, reachable free start/goal cells on the
unchanged map, then run BFS, random or the frozen selected learned policy once and replay it.
They never update the policy or benchmark. The viewer rejects incompatible map,
task, reward or observation/action versions and corrupted checkpoints/replays.
No map editor or final-test scenario browser is provided in this milestone.

The viewer supports all three checkpoint algorithms, including old Q-learning
files. `m3-*` runs use `selected.npz` or `selected.zip` from validation monitoring;
`checkpoint.*` retains the final-budget model separately. Playback never trains
or changes learned state. All custom demos stay outside benchmark results.

## Reproduce Part A

The original `part-a-v1` study and `m3-*` runs are retained. Use fresh output names:

```powershell
.\.venv\Scripts\python.exe scripts/run_part_a.py --suite-id part-a-repeat --run-prefix m3-repeat
.\.venv\Scripts\python.exe scripts/analyze_part_a.py --suite-id part-a-repeat
```

The fixed matrix is in `configs/part_a.yaml`; the smoke pilot and pre-study freeze
are documented in `docs/milestone3_protocol.md` and the study's
`benchmark_freeze.json`. No automatic hyperparameter search occurs. The runner
refuses to overwrite prior runs and records failed/interrupted status. Analysis
audits the saved replay scores/actions/metrics and source/checkpoint checksums,
generates research figures, and applies the predeclared five-seed shortlist rule.
The complete compressed trajectories and periodic checkpoints stay local and
are ignored by Git; summaries, examples, selected/final models and demo replays
are retained. Analysis needs those local full records, or a reproduced study.

Run a separate DQN pilot by copying `configs/dqn_smoke.yaml`, changing `run_id`,
then invoking `python -m robot_navigation.cli train-part-a --config <your-yaml>`
with the virtual-environment interpreter. The normalized vector contains the same
four coordinates as the table; DQN gets no obstacle grid, action mask or BFS advice.

Read-only evaluation of the selected DQN model:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.cli evaluate --checkpoint artifacts/runs/m3-dqn-seed7/selected.zip --split validation
```

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
- [Part A summary](artifacts/studies/part-a-v1/summary.json),
  [shortlist](artifacts/studies/part-a-v1/shortlist.json),
  [failure analysis](artifacts/studies/part-a-v1/failure_analysis.json), and
  [research figures](artifacts/studies/part-a-v1/figures/).
- [Milestone 3 checks](docs/milestone3_verification.json),
  [browser replay check](docs/milestone3_ui_check.json), and
  [measured Part A results](docs/milestone3_results.md).
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
or deployment was performed. Part A has a separate saved benchmark-freeze record;
all results concern held-out initial conditions on a familiar map. Most validation
initial states were visited during training, so this is not unseen-state transfer.

Refinement, tuning, voting and the full dashboard are later
milestones. PPO, unseen maps, moving obstacles, continuous movement and physical
robots remain deferred. No such extensions are implemented here.
