# Project organization

| Location | Purpose |
| --- | --- |
| `Start Robot Navigation.cmd` | Double-click Windows launcher |
| `run_app.py` | Launcher usable from a terminal |
| `docs/user-guide.md` | Current everyday guide and model explanation |
| `docs/developer-guide.md` | Setup, checks and maintenance |
| `docs/archive/` | Historical project plan, code review and audit plan |
| `autonomous-robot-navigation-rl/app/` | Streamlit entry point |
| `autonomous-robot-navigation-rl/src/robot_navigation/` | Active Python implementation |
| `autonomous-robot-navigation-rl/developer/tests/` | Preserved automated regression suite |
| `autonomous-robot-navigation-rl/developer/commands/` | Actual analysis, reproduction and evaluation commands |
| `autonomous-robot-navigation-rl/configs/` | Environment and study settings |
| `autonomous-robot-navigation-rl/data/` | Versioned maps and scenario splits |
| `autonomous-robot-navigation-rl/artifacts/` | Trained models, recordings and scientific results |
| `autonomous-robot-navigation-rl/docs/` | Research protocols, results and evidence manifests |
| `autonomous-robot-navigation-rl/.venv/` | Installed local Python dependencies |
| `autonomous-robot-navigation-rl/.runtime/` | Ignored test scratch space and current pytest cache |

Developer commands are working tools, not disposable tests. Run them from the application directory. Temporary trial workspaces, copied virtual environments, caches and scratch scripts were removed. Regression tests were kept, as requested.

Historical model IDs, protocol names and paths inside frozen `source/` and `snapshot/` bundles are deliberately stable: their checksums make published results verifiable. Renaming them as if they were new production models would break that evidence. The application directory name is also retained so the existing installed environment continues to work.

The current interface uses `app_theme.py`, `navigation_demo.py`, `structured_demo_ui.py`, `route_picker.py` and `user_guide_ui.py`. The obsolete theme and compatibility-only lab UI wrapper have been removed.
