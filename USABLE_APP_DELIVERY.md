# New partial/dynamic application delivery

**Ready to launch.** The only active application is the 10×10, range-two, partially observed dynamic-navigation system. It includes five-action Q-learning, SARSA, DQN and PPO; permanent obstacles and temporary blockages; saved-episode animation; training; fair development comparison; CSV/JSON/replay exports. The previous application is archived and has no selectable runtime mode.

## Launch and use

On a normal checkout, follow [Windows installation](docs/windows-install.md), then double-click **Start Robot Navigation.cmd** or run `py -3 run_app.py`. Open **http://127.0.0.1:8517**. `py -3 run_app.py --check` checks imports without starting a server. Keep the launcher terminal open.

This delivery is isolated from the user's dirty original workspace in:

```text
C:\Users\User\Desktop\Autonomous_Robot_Navigation\autonomous-robot-navigation-rl\.runtime\newapp
```

An existing compatible Python environment was used without changing its editable installation. From the delivery root, the verified local command is:

```powershell
& 'C:\Users\User\Desktop\Autonomous_Robot_Navigation\autonomous-robot-navigation-rl\.venv\Scripts\python.exe' run_app.py --python 'C:\Users\User\Desktop\Autonomous_Robot_Navigation\autonomous-robot-navigation-rl\.venv\Scripts\python.exe' --server.headless true
```

For double-click launch in this isolated checkout, create its own environment using the Windows guide, or set `ROBOT_NAV_PYTHON` to the existing interpreter. The launcher explicitly sets the new checkout's source path. The server was smoke-tested, then stopped so it will not occupy the port.

1. **Robot Simulator:** choose one of four routes, Static/Temporary obstacles, legal start/goal and a controller. Run journey once. Play/Pause/Back/Step/Seek/Reset/End only read recorded frames. Scene truth is human-only; the robot memory view and N/S/E/W sensors show the permitted information. Download the genuine replay.
2. **Train Models:** explicitly choose Quick or Standard, models, actions and seed. Start a hidden CPU worker, Refresh progress or Cancel safely. Closing the browser does not stop a worker. New native checkpoints appear on refresh. CLI training is also supported; nothing trains automatically on rerun.
3. **Compare & Export:** select one checkpoint for each family, optionally add Random, then run the same four manual development cases. Inspect success/collision/timeout denominators, Wait, loops, recovery and static/dynamic subsets. Download case CSV, JSON or results plus all replays ZIP.
4. **Guide / Archives:** read task semantics and the short teammate walkthrough, inspect historical reports and find archive export commands. Historical scientific scores never populate new product comparisons.

## Available models and measured demo

Reviewed native starters are committed under `autonomous-robot-navigation-rl/runs/training/20261122T112025Z-a7301ff2/`.

- Exactly **128 CPU actions per family**, 512 total; seed 74001; eight fresh train-only cases; unchanged reward and observation contracts. No standard/full scientific training.
- All four saved/loaded genuine 409-observation, 412-feature B1, five-action checkpoints. Both linear weights changed. DQN's saved update counter is 24; PPO's is 8. These are inference checkpoints, not exact-resume snapshots.
- The four fixed manual development cases are separate start/goal pairs, sharing simple layouts with training. They are not unseen/generalization claims.

| Controller | Success | Collision | Timeout |
|---|---:|---:|---:|
| Q-learning | 0/4 | 0/4 | 4/4 |
| SARSA | 1/4 | 2/4 | 1/4 |
| DQN | 0/4 | 2/4 | 2/4 |
| PPO | 0/4 | 4/4 | 0/4 |
| Observed-Map Replanner (non-RL) | 4/4 | 0/4 | 0/4 |
| Random (non-RL) | 0/4 | 4/4 | 0/4 |

The actual comparison has 24 episodes and 1,632 actions in `runs/comparisons/20261122T112905Z-1ec216d7/` (paths relative to the Python application). The replanner waits six steps in the blocked aisle and takes observed detours. Poor RL behavior remains visible; there is no hidden planner fallback. Tiny budgets demonstrate the workflow, not strong navigation or deployment readiness.

From an installed checkout's Python application directory:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset quick --models all --actions 128 --seed 74001
.\.venv\Scripts\python.exe -m robot_navigation.app_training --list-models
.\.venv\Scripts\python.exe -m robot_navigation.app_service --compare --include-random
```

Quick defaults to 256 actions/model; Standard to 10,000 (explicit only). Maximum 50,000/model, one-hour worker cap. PPO requires exact rollout multiples 32 quick / 100 standard; DQN multiples4. Each command creates fresh immutable run IDs. New local outputs are ignored by Git by default; reviewed starters/results were explicitly included.

## Complete historical preservation

The archive was inventoried before old runtime retirement, followed by a full post-move SHA-256 gate on 14,527 versions. A final extension-independent audit added and verified two untouched historical scope-filter files (110 bytes total) plus 35 already-preserved path aliases:

- **14,529 unique file versions; 1,577,969,531 bytes; zero missing files; zero missing LFS payloads.** This is deduplicated version content, not a count of original path aliases.
- Classic assets, models, datasets, replays, logs, guides and retired source/tests are under `archive/legacy-classic/` and `archive/historical-source/`.
- Sealed partial Phase 5A–5H evidence remains at original versioned locations, with full path/hash references. Earlier/deleted Git versions and unique local dirty/runtime outputs are materialized in `archive/git-objects/` and `archive/workspace-snapshots/`.
- `archive/manifests/pre-migration.json`, `original-path-map.csv`, `source-versions.json`, `integrity.json` and `archive/INDEX.csv` document actual bytes, original paths, commits/blobs, aliases and backing paths. Git attributes protect archived/retained evidence bytes from line-ending conversion.
- The original dirty main worktree and prior branches were read, not rewritten. Historical reports were not regenerated or edited. Old relative links may require the original directory context/path map.

From repository root:

```powershell
python scripts/export_history_archives.py --list
python scripts/export_history_archives.py --output-dir history-exports
python scripts/export_history_archives.py --output-dir history-exports --verify
# Optional independent full recheck of all local sources:
python scripts/export_history_archives.py --verify-sources
```

The fresh folder receives **legacy-classic-history.zip**, **partial-navigation-research-history.zip** and **archive-inventory.json**. Copying and verification stream in chunks; fixed ZIP timestamps make repeat exports reproducible for identical sources. Missing data fails explicitly. Export never deletes sources. See [archive guide](archive/README.md).

Real coverage enumeration passed: classic 16,751 ZIP members / 11,927 unique versions / 1,414,480,765 uncompressed bytes; partial 6,589 members / 3,343 unique versions / 407,069,598 bytes. Shared source and original-path aliases explain overlap. A small two-collection ZIP fixture was actually generated and verified, including repeat-byte equality and tamper rejection. **The multi-gigabyte real collections were enumerated, not generated or ZIP-verified during delivery.** Duplicate full ZIP blobs are not committed; the above command produces them locally on demand.

## Validation and limits

[Evidence files](autonomous-robot-navigation-rl/docs/product-evidence/) include JUnit, archive coverage, native model signatures/update counters, measured comparison summary and cancellation evidence.

- Complete focused suite: **10 passed in 16.10s**. Actual 64-action/family learner fixture (256 actions) checked nonzero neural updates, save/load/inference, bad checkpoint errors, immutable contracts, observed-only Wait/detour, genuine timeout, passive replay, identical comparison panels, case CSV and replay counts, retired imports, new UI pages and streamed archive fixture.
- Expanded playback/Archives AppTest: **1 passed, 9 deselected in 2.45s**; no learner fixture. Play/Pause/Step/Back/End/Seek/Reset and read-only Archives passed.
- Background worker started hidden; an immediate cancellation request produced `cancelled`, zero learner actions, and a preserved partial run. No model was registered for that cancelled run.
- Root launcher import check passed; actual server started at 127.0.0.1:8517; `/_stcore/health` returned HTTP 200 with `ok`. UI verification used Streamlit AppTest, not a manual browser inspection.
- Restricted-runner attempts were interrupted: one redirected attempt yielded no result; a subsequent attempt passed eight backend tests (including its 256-action fixture) then stalled in AppTest. The complete successful suite and HTTP smoke used an unrestricted runner after sandbox socket access failed. These interrupted attempts are not reported as passes.
- No full legacy regression suite, historical experiment regeneration, final/unseen scientific evaluation or Phase 6 was run. No changes were made to authoritative partial environment/learner source bytes. The product uses simple manual development layouts and small starter models; larger explicitly initiated training is available but unmeasured here.

## Version control

Branch: **feature/new-dynamic-app-archive-history-v1**, based on Phase 5H `0c0ba4b0583d7f03fabba901397a0f41f0aac452`.

The delivery commit is the commit containing this report; resolve its exact SHA with `git log -1 --format=%H` after checkout. [Feature branch commit log](https://github.com/happywintersolstice-oss/Autonomous_Robot_Navigation/commits/feature/new-dynamic-app-archive-history-v1). The final delivery response supplies the pushed SHA. Main and earlier feature branches are not merged or force-pushed.
