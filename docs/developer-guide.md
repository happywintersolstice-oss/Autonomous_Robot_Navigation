# Active application development

Install using [Windows guide](windows-install.md). The only UI route is `app/streamlit_app.py` → `dynamic_demo_ui.render`. The root launcher fixes localhost:8517 and sets the new source path. There is no installed classic CLI.

`app_scenarios` declares fresh train-only and manual cases. `app_training` performs explicit bounded native training with immutable run IDs/progress/cancellation. `app_registry` reads new run manifests only and validates five-action compatibility. `app_service` records real episodes once, reads passive frames and compares identical cases without updates. Display snapshots never enter policies.

Unchanged `partial_*` environment/learners retain their contract. Neutral GridMap/atomic JSON live in `shared.py`; `core.py` re-exports them and `release.py` contains only SHA256. Archived old source is outside installed package discovery.

```powershell
cd autonomous-robot-navigation-rl
.\.venv\Scripts\python.exe -m pytest developer/tests/test_dynamic_app.py -q
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset quick --models all --actions 128
.\.venv\Scripts\python.exe -m robot_navigation.app_service --compare --include-random
```

The focused suite trains 64 actions/family in temporary output and checks actual neural updates, strict loading, sensor/blocker contracts, observed-only replanning, passive playback, fair CSV, AppTest routes and a streamed ZIP fixture. Do not run archived suites/old experiments for this product smoke.

Archive manifests cover historical tracked/Git/local file versions and hashes. Classic assets/source moved; sealed partial evidence stayed. ZIP exports stream copies on demand. The original dirty workspace and older branches remain untouched. Use original directory context/commit to reproduce historical software; do not import it into the app.

The root `pytest.ini` also restricts discovery to the new product tests and excludes archived research/runtime folders. Running `pytest` from the repository root therefore does not activate archived suites.
