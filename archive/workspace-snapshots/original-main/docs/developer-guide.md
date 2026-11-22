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

From the repository root, double-click **Start Warehouse Navigation.cmd**, or run `python run_app.py`. A terminal-only launch without opening a browser is `python run_app.py --server.headless true`. The server listens locally on port 8517.

## Verify

```powershell
New-Item -ItemType Directory -Path .runtime -Force | Out-Null
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime/regression-tests
.\.venv\Scripts\python.exe -m robot_navigation.cli check
.\.venv\Scripts\python.exe -m robot_navigation.project_assets verify
```

The regression suite lives in `developer/tests/`; pytest discovers it through `pyproject.toml`. `.runtime/` is ignored scratch space. Use a new temporary directory if another test process is still running. Streamlit AppTest exercises navigation, model selection, actual recorded actions, comparisons, route generation and malformed-data handling. It does not substitute for a visual browser review.

After intentional application or project dependency changes, rebuild the current project inventory, then verify it:

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.project_assets build
.\.venv\Scripts\python.exe -m robot_navigation.project_assets verify
```

Rebuilding this current inventory does not revise historical release seals. Preserve the original final releases, study manifests, saved trajectories and byte-pinned protocols.

## Warehouse comparison reproduction

The registered experiment is `configs/warehouse_comparison.yaml`; its protocol and exact training engine are sealed before execution. The published directory contains all twelve runs. Opening the app never retrains them.

```powershell
.\.venv\Scripts\python.exe -m robot_navigation.warehouse_study freeze
.\.venv\Scripts\python.exe -m robot_navigation.warehouse_study run
```

The runner checks its registered inputs, uses at most two CPU workers, and publishes only when every run passes checkpoint-reload and recorded-action audits. A completed run is verified and skipped on re-entry. An incomplete/failed existing run is not silently overwritten. The current published study is read-only evidence: reproduce into a separate checkout/workspace with the registered source and an initially absent study output directory. A changed map, algorithm setting or protocol requires a new versioned study, rather than rewriting these results.

```powershell
.\.venv\Scripts\python.exe -c "from robot_navigation.warehouse_study import saved,checked_run,FAMILIES; s=saved(); [checked_run(a,k) for a in FAMILIES for k in s['systems'][a]['seeds']]; print(s['status'],s['total_training_interactions'])"
```

This checks input seals, publication hashes and every run's saved outputs. It does not evaluate policies again or modify historical evidence.

Regenerate the current report and figure from verified saved outputs (without training):

```powershell
.\.venv\Scripts\python.exe developer/commands/report_warehouse_comparison.py
```

## Developer tools

Analysis, study and reproduction commands are in `developer/commands/`. For example:

```powershell
.\.venv\Scripts\python.exe developer/commands/replay_frozen_release.py --workspace .runtime/frozen-check
```

That command prepares and verifies the archived release in a fresh scratch workspace. It does not evaluate unless explicitly requested with `--evaluate`. Its internal `snapshot/scripts/` paths belong to the historical archive and are intentional.

Training and analysis commands can be expensive or write new results. Use fresh run/study IDs, follow the corresponding research protocol, and never overwrite published artifacts. Do not rerun studies just to refresh the interface.

See the [research command reference](../autonomous-robot-navigation-rl/docs/developer-reference.md), [evidence index](../autonomous-robot-navigation-rl/docs/evidence_index.md) and [project organization](project-structure.md). Archived reports describe their own historical UI and paths; this guide and the [user guide](user-guide.md) describe the current app.

## Temporary 3D preview

`preview_3d.py` uses the already locked pydeck dependency and Streamlit's bundled renderer. The original control-room styling and 2D view are the defaults. The optional preview projects the same grid, wall cells and displayed trajectory prefix into a Cartesian scene. It runs no learning or benchmark evaluation.

The browser needs WebGL. No geographical basemap or map-provider API key is used. The camera is an interactive [OrbitView](https://deck.gl/docs/api-reference/core/orbit-view), with [extruded polygons](https://deck.gl/docs/api-reference/layers/polygon-layer) for walls. Renderer settings such as `cartesian` and `pixels` must use `pydeck.types.String` so pydeck serializes them as literal values rather than accessor expressions. Regression checks cover the serialized properties, visible-prefix geometry, view switching and comparison panels.
