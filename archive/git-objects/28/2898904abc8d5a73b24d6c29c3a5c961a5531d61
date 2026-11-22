# Developer guide

The application lives in `autonomous-robot-navigation-rl/`. Run all commands below from that directory. Windows with Python 3.14.10 is the verified local environment; the package declares Python 3.11 or later, but other installations need their own verification.

## Install

```powershell
cd autonomous-robot-navigation-rl
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
.\.venv\Scripts\python.exe -m pip check
```

From the repository root, double-click **Start Robot Navigation.cmd**, or run `python run_app.py`. A terminal-only launch without opening a browser is `python run_app.py --server.headless true`. The server listens locally on port 8517.

## Verify

```powershell
New-Item -ItemType Directory -Path .runtime -Force | Out-Null
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime/regression-tests
.\.venv\Scripts\python.exe -m robot_navigation.cli check
.\.venv\Scripts\python.exe -m robot_navigation.demo_assets verify
```

The regression suite lives in `developer/tests/`; pytest discovers it through `pyproject.toml`. `.runtime/` is ignored scratch space. Use a new temporary directory if another test process is still running. Streamlit AppTest exercises navigation, model selection, actual recorded actions, comparisons, route generation and malformed-data handling. It does not substitute for a visual browser review.

After intentional application or demo dependency changes, rebuild the current demo inventory, then verify it:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.demo_assets build
.\.venv\Scripts\python.exe -m robot_navigation.demo_assets verify
```

Rebuilding this current inventory does not revise historical release seals. Preserve the original final releases, study manifests, saved trajectories and byte-pinned protocols.

## Developer tools

Analysis, study and reproduction commands are in `developer/commands/`. For example:

```powershell
.\.venv\Scripts\python.exe developer/commands/replay_frozen_release.py --workspace .runtime/frozen-check
```

That command prepares and verifies the archived release in a fresh scratch workspace. It does not evaluate unless explicitly requested with `--evaluate`. Its internal `snapshot/scripts/` paths belong to the historical archive and are intentional.

Training and analysis commands can be expensive or write new results. Use fresh run/study IDs, follow the corresponding research protocol, and never overwrite published artifacts. Do not rerun studies just to refresh the interface.

See the [research command reference](../autonomous-robot-navigation-rl/docs/developer-reference.md), [evidence index](../autonomous-robot-navigation-rl/docs/evidence_index.md) and [project organization](project-structure.md). Archived reports describe their own historical UI and paths; this guide and the [user guide](user-guide.md) describe the current app.
