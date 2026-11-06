# Autonomous Robot Navigation — milestone 4 / Part B

A local Python research pilot: one versioned static grid, Gymnasium environment,
random/BFS references, Q-learning, SARSA and vanilla SB3 DQN, reproducible scenario splits, measured validation,
and an animated Streamlit replay viewer. Existing workspace files are separate.

**Part A:** five independent seeds per learned algorithm, 200,000 interactions
each, common validation monitoring, and an explicit validation-only checkpoint
and algorithm-family selection rule. See [milestone 3 results](docs/milestone3_results.md)
for measured outcomes, failure analysis and the two-family shortlist. The final
test has never been executed. The original [30k pilot](docs/milestone_results.md)
and [tabular development runs](docs/milestone2_results.md) remain unchanged.

**Milestone 4:** the Q-learning/SARSA shortlist receives a bounded random search,
fresh-seed paired baseline/tuned confirmations, a reward-only ablation and frozen
policy voting. See [milestone 4 results](docs/milestone4_results.md) for achieved
metrics, missed targets and limitations. The original Part A evidence is preserved.

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

It also loads explicit terminal-only reward variants and frozen three-policy
voting descriptors. The reward version/cost and voting member seeds are shown.
A checkpoint for a different reward version is rejected unless that exact approved
variant is active. Canonical map/scenario files are never rewritten by experiments.

## Reproduce Part B

The study uses eight frozen random candidates per shortlisted family, two
development seeds, five fresh confirmation seeds, and matched original-setting
controls. A paired reward ablation changes only step reward from -.01 to 0.
Voting uses three predeclared seeds of the winning tuned original-reward family,
without averaging Q values or training the voting rule. See the
[predeclared protocol](docs/milestone4_protocol.md) and `configs/part_b.yaml`.

```powershell
.\.venv\Scripts\python.exe scripts/run_part_b.py --suite-id part-b-repeat --run-prefix m4-repeat
.\.venv\Scripts\python.exe scripts/analyze_part_b.py --suite-id part-b-repeat
```

Use fresh names each time; prior runs are not overwritten. The full matrix has
62 training runs / 12.4m interactions, capped at 45 run-phase minutes and 750 MiB
of run artifacts. Search alone costs 6.4m interactions; report it separately from
the chosen policy's cost. Fresh training seeds check seed stability on the same
validation tasks; they do not turn selected validation scores into final-test results.

Read-only evaluation of a terminal-only checkpoint requires the matching variant:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.cli evaluate --checkpoint artifacts/runs/m4-terminal-q-learning-seed101/selected.npz --reward-variant terminal-only --split validation
```

Voting is saved in `artifacts/runs/m4-voting-<family>/voting.json`; it references
the three member checkpoints and metadata by relative path with hashes. Retain
that directory layout when copying a bundle. The wrapper has zero new training;
its members' summed training cost, table memory, disk size and latency are recorded
in `artifacts/studies/part-b-v1/voting_comparison.json`. Saved voting transitions
contain actual member proposals and tie diagnostics, not invented Q scores.

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
- [Part B summary](artifacts/studies/part-b-v1/summary.json),
  [paired changes](artifacts/studies/part-b-v1/paired_improvements.json),
  [reward ablation](artifacts/studies/part-b-v1/reward_ablation.json),
  [voting comparison](artifacts/studies/part-b-v1/voting_comparison.json), and
  [target assessment](artifacts/studies/part-b-v1/target_assessment.json).
- [Milestone 4 verification](docs/milestone4_verification.json),
  [UI checks and limits](docs/milestone4_ui_check.json), and
  [preservation check](docs/milestone4_preservation.json).
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

The full dashboard, expanded core robustness, final freeze/test evaluation and
release documentation are later milestones. PPO, unseen maps, moving obstacles, continuous movement and physical
robots remain deferred. No such extensions are implemented here.
