# Autonomous Warehouse Robot Navigation

Open **Robot Navigation.code-workspace** for the clean editor layout, or double-click **Start Robot Navigation.cmd** to launch this checkout at **http://127.0.0.1:8517**. See [the workspace guide](docs/clean-workspace.md). The earlier navy/cyan warehouse presentation is restored with all five Phase 3 pages.

One application for a 10x10 robot with range-two occupancy sensing, memory and **Up / Down / Left / Right / Wait**. Four real RL families—Q-learning, SARSA, DQN and PPO—can be trained, simulated and compared. The Observed-Map Replanner and Random are separate non-RL references.

## Start on Windows

Follow [Windows installation](docs/windows-install.md) once, then double-click **Start Robot Navigation.cmd** or run `py -3 run_app.py`. Open **http://127.0.0.1:8517**. The launcher uses `autonomous-robot-navigation-rl/.venv`; `--python <path>` selects an existing compatible environment. Keep its terminal open.

- **Robot Simulator**: choose a preset, custom start/goal, Static/Temporary obstacles and controller. Run once, then Play/Pause/Back/Step/Seek/Reset saved frames. Scene truth is viewer-only; robot memory is separate. Collisions, loops and timeouts remain visible.
- **Train Models**: explicitly start a bounded quick or standard job, Refresh progress, Cancel safely and select new native checkpoints. No automatic training on rerun.
- **Compare & Export**: view the single saved Phase3 product held-out assessment or separate Phase3/Phase2 development evidence, compare baseline/selected checkpoints and non-RL references, download CSV/JSON/replay ZIP, and open failure replays. Held-out viewing never reruns evaluation.
- **Guide**: sensing, stale memory, five actions, rewards, models and a 3–5-minute walkthrough.
- **Archives**: read-only historical reports and ZIP export instructions. No old simulator or model-transfer mode.

```powershell
cd autonomous-robot-navigation-rl
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset quick --models all --features product-v3
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset standard --models all --features product-v3
.\.venv\Scripts\python.exe -m robot_navigation.app_training --list-models
.\.venv\Scripts\python.exe -m robot_navigation.app_service --compare --include-random
```

Quick defaults to 256 actions/model; standard to 10,000 and must be user initiated. `--actions`, `--seed`, `--models` and `--output` are bounded. PPO requires a multiple of32 quick /200 standard actions; DQN a multiple of4. New immutable product runs are in `autonomous-robot-navigation-rl/runs/`. Saves are inference checkpoints, not exact-resume snapshots. Tiny starter models demonstrate the workflow, not strong navigation or scientific efficacy.

## All earlier history is preserved

The old four-action application is retired from the active package. **There is no Classic/Dynamic toggle.** [Archive guide](archive/README.md), [inventory](archive/INDEX.csv) and [integrity receipt](archive/manifests/integrity.json) cover classic results, Phase 5A–5H data, models, negative trials, logs, reports, source versions and unique local historical outputs. Sealed partial research remains at original paths; retired assets/source are archived. Original dirty-workspace files were copied read-only.

```powershell
python scripts/export_history_archives.py --list
python scripts/export_history_archives.py --output-dir history-exports
python scripts/export_history_archives.py --output-dir history-exports --verify
```

ZIPs stream to disk with inventories/checksums; duplicate giant ZIP blobs are not committed. Missing files/LFS payloads fail explicitly. Historical final/unseen evidence is reference-only and never mixed into new scores. Four-action models are not converted.

See [delivery report](USABLE_APP_DELIVERY.md), [user guide](docs/user-guide.md) and [developer guide](docs/developer-guide.md). Use focused integration tests, not archived research suites, to validate the app.

## Product Phase 2: real trained checkpoints

The app now labels **Best on development**, **Trained** and **Starter** checkpoints, and Compare opens the saved 24-layout comparison without automatic evaluation. Selected Q-learning/SARSA/DQN/PPO achieved **23/24, 22/24, 18/24 and 21/24** on this development-selected panel, compared with original starters **0/24, 1/24, 3/24 and 1/24**. These are development scores, not unseen accuracy.

See [actual results and failure history](PRODUCT_PHASE2_RESULTS.md) and the [four-person demo](docs/team-demo-script.md). Training uses a seeded train-only curriculum and versioned public-observation features; official evaluation rewards and partial sensing are unchanged. Optional progress feedback is disclosed and off by default. The archive and old starters remain unchanged. Launch remains `py -3 run_app.py`, port **8517**.

## Phase3 product delivery

Phase3 adds a diverse new curriculum, independent 64-case product holdout, versioned public decision history, bounded genuine RL training and development-only checkpoint selection. The four new demo presets include a winding maze, permanent dead end, forced Wait bridge and optional temporary detour. The five pages and existing visual style are retained. Select **Phase 3 selected** or **Phase 2 baseline** in Simulator; old native checkpoint signatures remain supported.

See [measured Phase3 results](PRODUCT_PHASE3_RESULTS.md) and the [four-person checked demonstration](docs/team-demo-script.md). Results include failures and successful-only Wait denominators. This is a single product panel/seed, not a scientific final benchmark or proof of general superiority. Training feedback is versioned and train-only; official evaluation reward is unchanged. Phase2 results/selection/checkpoints and archived history remain unchanged.

Automated test files and test-only fixtures were removed at the owner's request. A verified source snapshot is retained in [archive/source-snapshots/removed-active-tests-20261123T101901Z](archive/source-snapshots/removed-active-tests-20261123T101901Z/manifest.json); previous validation results remain historical evidence. For a basic launch check, use `py -3 run_app.py --check`. This does not retrain models or rerun the frozen product test.

## Readable integrated code and assignment check

The current UI cleanup and Phase3 backend are integrated on `feature/phase3-cleanup-assignment-audit-v1`. There is one page router and one implementation of each page. Saved-model import paths are retained; active Python uses standard formatting and the algorithms include explanatory docstrings. The cleanup does not retrain models or rerun the frozen product test.

- [Start with the code walkthrough](docs/code-walkthrough.md): entry points, data flow, state slices, actual updates and safe editing boundaries.
- [Assignment and rubric audit](docs/assignment-compliance-audit.md): technical evidence and actual gaps. Full assignment compliance is not established; the final report/package, individual contribution/proposal evidence and AI-use compliance remain unresolved. Current model combination is now implemented and measured on development.
- [Cleanup validation](docs/cleanup-validation.md): compatibility checks, UI integration, packaging and preservation evidence.

Open `Robot Navigation.code-workspace` in this feature checkout for the integrated editor view. The separate Documents workspace and the original dirty Desktop/main checkout are preserved; they were not overwritten or merged into main.

## Current rubric support

The optional four-model voting ensemble is now available in Simulator. Its registered **development-only** comparison is saved under Compare & Export > Ensemble development: **36/36** goals, matching SARSA/DQN/PPO and exceeding Q-learning's34/36 on these previously inspected cases. The ensemble is slower and takes longer routes than the strongest single model; it does not replace the four selected policies. No new training or reuse of the published product holdout occurred.

See [actual completion status and measured evidence](docs/rubric-completion-status.md), [submission support](docs/submission/README.md), and [the live missing-evidence checklist](docs/submission/readiness-result.json). The technical appendix/outline are support materials, not a finished student-authored report. Human contribution, approved scope, permitted AI use, final report review, rehearsal and submission remain unverified.
