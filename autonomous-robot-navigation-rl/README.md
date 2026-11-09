# Autonomous Robot Navigation — completed core release / milestone 6

A local Python research pilot: one versioned static grid, Gymnasium environment,
random/BFS references, Q-learning, SARSA, vanilla SB3 DQN and PPO, reproducible scenario splits, measured validation,
and an animated Streamlit replay viewer. Existing workspace files are separate.

For experiment labels such as `m4-tuned-q-learning-seed101`, seeds and saved file
types, see the [file and experiment naming guide](docs/file_and_experiment_guide.md).

**Fourth-algorithm extension:** the owner requested PPO after the core release.
Its implementation and separate validation workflow are described in
[PPO extension status](docs/ppo_extension_status.md) and the
[registered protocol](docs/ppo_comparison_protocol.md). All five PPO seeds completed
200,000 interactions each and passed artifact verification. The dashboard's
**Agent comparison** view includes all four RL algorithms using development
validation. See [measured results and limitations](docs/ppo_comparison_results.md).
Those five PPO runs are the original additional baselines. M9 adds a separate
preliminary neural tuning study below.
The [current plan-closure status](docs/plan_closure_status.md) tracks remaining
gaps. Additional route/cost tables and all six four-family robustness dimensions
are available in the dashboard, backed by the separate `four-rl-analysis-v1`
saved-evidence study. See [analysis details](docs/four_way_analysis_results.md).

**Milestone 8: controlled DQN inputs.** The owner selected a paired comparison
of four absolute coordinates versus those coordinates plus two relative goal
offsets. This is an input ablation on the unchanged task, with fresh paired seeds
and bounded chronological training traces. All ten 200k-interaction main runs
completed and passed audit. Success averaged 5.97% control versus 6.51% relative;
the +0.54 pp paired change was inconsistent across seeds. See
[all measured results](docs/milestone8_results.md), the
[registered protocol](docs/representation_protocol.md) and
[execution status](docs/representation_status.md) for actual progress.

**Milestone 9 controlled DQN/PPO tuning.** Four configurations per algorithm were
screened before three fresh matched confirmation seeds at 200,000 steps each.
Success averaged 5.60% baseline versus 7.11% selected settings for DQN, and 16.99%
versus 24.35% for PPO. Both remain weak and miss the declared targets. Open
**Neural tuning** in the dashboard for results, curves, costs and matched replay.
See [the measured results](docs/milestone9_results.md) and
[the registered protocol](docs/milestone9_protocol.md). The original frozen final
controller remains Q-learning. This is preliminary development validation.

**Part A:** five independent seeds per learned algorithm, 200,000 interactions
each, common validation monitoring, and an explicit validation-only checkpoint
and algorithm-family selection rule. See [milestone 3 results](docs/milestone3_results.md)
for measured outcomes, failure analysis and the two-family shortlist. Part A/B
selection used validation only; frozen final evaluation is now complete.
The original [30k pilot](docs/milestone_results.md)
and [tabular development runs](docs/milestone2_results.md) remain unchanged.

**Milestone 4:** the Q-learning/SARSA shortlist receives a bounded random search,
fresh-seed paired baseline/tuned confirmations, a reward-only ablation and frozen
policy voting. See [milestone 4 results](docs/milestone4_results.md) for achieved
metrics, missed targets and limitations. The original Part A evidence is preserved.

**Milestone 5:** six local dashboard views, prefix visit/BFS/decision diagnostics,
event seeking, synchronized agent and early-versus-selected checkpoint replay,
and predefined fixed-map robustness groups. See [measured results and limits](docs/milestone5_results.md).
The analysis reuses saved validation metrics; it adds no training or final-test runs.

**Milestone 6:** predeclared models and analysis were frozen, evaluated on all 928
final pairs and fully replay-audited. Refined Q-learning averaged **90.43%** success
versus **65.52%** for matched controls. The selected single remains
`m4-tuned-q-learning-seed101`: 840/928 goals, 7 collisions, 81 timeouts, SPL .867870.
Voting reached 848 goals but was not promoted using final data. See the
[technical report](docs/technical_report.md), [final results](docs/milestone6_results.md)
and [verification](docs/milestone6_verification.json). Final data is now spent for
selection; later performance changes need a new holdout or exploratory labeling.

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
port if occupied. The control room opens the predeclared refined Q-learning run
(`m4-tuned-q-learning-seed101`) and its saved success replay. Press Play to watch
each discrete action; older runs and their failures remain in Mission setup.
Restart an already-running server after pulling new Python modules; reloading the
browser alone may retain imports from an earlier milestone.
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
Overview and Agent comparison also display audited aggregate final results. The
simulator continues using validation/custom scenarios and never opens final pairs.

Choose **Dashboard view** for Overview, Learning analysis, Agent comparison,
Optimization and robustness, or Methodology. The simulator remains the default.
Optional trail/visit/BFS overlays use the displayed prefix and a labeled planner
reference. Decision tables show actual recorded Q scores or voting proposals for
the last displayed move; initial-frame scores are unavailable. Open the displayed
event log and select a frame to seek backward. No future action log is exposed.

For side-by-side playback select **Episode source → Matched comparison**, choose
the same validation scenario and a second run, then Play or seek the shared timeline.
Finished agents remain at their terminal frames while the other continues. Reward
versions and all task identities must match. Comparison replay is generated once
when its selection changes, and is excluded from measured benchmark summaries.

The dark simulator interface groups mission setup and map layers in the sidebar,
with shared playback controls above the map. Matched comparison initially pairs
the selected refined run with its seed-101 baseline. Telemetry distinguishes
training seeds in run labels from the replay seed in task details. A repeating
action-pattern warning uses the displayed prefix and never changes the policy.
See the [dashboard rework notes](docs/dashboard_rework.md) for verification and
the distinction between this UI revision and the original frozen release.

For **Early versus selected checkpoint**, choose an archived exemplar in Saved run:
`m3-q-learning-seed7`, `m3-sarsa-seed7`, `m3-dqn-seed7`,
`m4-tuned-q-learning-seed101`, or `m4-tuned-sarsa-seed101`. The early snapshot is
10,000 interactions; the later one is monitoring-selected, with its actual age in
the run manifest. Unavailable early models produce a clear message. The five
archived early snapshots are included independently of the ignored bulk checkpoints.

The viewer supports all four checkpoint algorithms, including old Q-learning
files. `m3-*` runs use `selected.npz` or `selected.zip` from validation monitoring;
`checkpoint.*` retains the final-budget model separately. Playback never trains
or changes learned state. All custom demos stay outside benchmark results.

It also loads explicit terminal-only reward variants and frozen three-policy
voting descriptors. The reward version/cost and voting member seeds are shown.
A checkpoint for a different reward version is rejected unless that exact approved
variant is active. Canonical map/scenario files are never rewritten by experiments.

## Reproduce the core robustness analysis

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
.\.venv\Scripts\python.exe scripts/analyze_core_robustness.py --suite-id milestone5-repeat
```

Use a fresh suite ID. This reanalysis validates 26 saved system/seed records on
928 validation pairs each, reconciles aggregates, verifies models, and reports six
predefined scenario dimensions. It does not retrain. Periodic models, if present,
are copied into the new study for early-checkpoint demonstrations; when absent,
that comparison is marked unavailable. The canonical dashboard reads the included
`milestone5-v1` study. DQN dependencies are needed to verify its saved checkpoints.

## Reproduce the selected policy and frozen release

In a fresh `.cache` workspace this prepares the identical benchmark, trains the
selected Q-learning configuration for 200k interactions, reloads it and checks its
selected array and all validation metrics against the saved model:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
.\.venv\Scripts\python.exe scripts/verify_reproduction.py --workspace .cache/reproduction-repeat
```

The original check passed using installed locked dependencies on Windows/Python
3.14.10. A subsequent fresh virtual-environment installation also passed all 96
tests, imports, dependency consistency and frozen-release preparation using the
unchanged lock file. Pip reused cached wheels; cold-cache downloads and other
hardware were not tested. See [current verification](docs/plan_closure_status.md).
The reproduced model is under that workspace's `artifacts/runs/`; original runs
are never overwritten. No final cases enter training.

The exact original freeze is [final-v1](artifacts/studies/final-v1/freeze.json), with
digest in `seal.json`. Its runner refuses changed input bytes and an existing
evaluation output. Its snapshot is protected from Git line-ending normalization.
Restore and verify that exact source/model/data archive after checkout with:

```powershell
.\.venv\Scripts\python.exe scripts/replay_frozen_release.py --workspace .cache/frozen-check
```

This prepares and checks hashes/dependencies only. To repeat the unchanged frozen
evaluation, add `--evaluate` and use another fresh workspace name. It runs the
archived code via PYTHONPATH, retains the original model choice and cannot overwrite
the recorded final results. Repeated cases provide repeatability evidence, not an
independent test set or another selection opportunity. The original two-stage
freeze/evaluate procedure and matrix/limits are in [the final protocol](docs/milestone6_protocol.md).

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

For the separate DQN representation study, copy
`configs/representation_study.yaml`, choose a fresh `suite_id` and `run_prefix`,
then run `scripts/run_representation_study.py pilot --config <your-yaml>` followed
by the same command with `run`. The runner checks the pilot estimate before the
main matrix, refuses overwrite and retains failures. The canonical analyzer is
`scripts/analyze_representation_study.py` for `representation-v1`.
Relative checkpoints declare `absolute-relative-coordinates-v1`; model loading
checks the saved encoding and six-input architecture. Old DQN checkpoints retain
their four-feature inference. New runs can set `training_trace_episodes` from 0
to 10 (default 0); this study captures the first three episodes and explicitly
labels incomplete budget/interruption prefixes. Training trace downloads live in
the simulator's saved training-progress expander, separate from validation replays.

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
Final evaluation uses the separate digest-gated frozen runner.
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
- [Milestone 5 results](docs/milestone5_results.md),
  [robustness summary](artifacts/studies/milestone5-v1/summary.json),
  [per-seed groups](artifacts/studies/milestone5-v1/groups.csv),
  [paired group differences](artifacts/studies/milestone5-v1/paired_groups.json),
  [verification](docs/milestone5_verification.json), and
  [automated and live browser checks](docs/milestone5_ui_check.json).
- [Technical report](docs/technical_report.md), [final results](docs/milestone6_results.md),
  [final summary](artifacts/studies/final-v1/summary.json),
  [per-seed final metrics](artifacts/studies/final-v1/final_seed_results.csv),
  [development run matrix](artifacts/studies/final-v1/development_run_matrix.csv),
  [release checklist](docs/release_checklist.md), and
  [preservation](docs/milestone6_preservation.json).
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

All six core milestones are complete. Source and evidence remain local.
PPO was subsequently added as the owner-authorized exploratory extension above.
Unseen maps, moving obstacles, continuous movement and physical robots remain deferred.
