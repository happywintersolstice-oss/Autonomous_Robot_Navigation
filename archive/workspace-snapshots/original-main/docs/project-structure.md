# Project organization

The current project is **Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO**.

| Location | Purpose |
| --- | --- |
| `Start Warehouse Navigation.cmd` | Double-click Windows launcher |
| `run_app.py` | Terminal launcher for the same local app |
| `docs/user-guide.md` | UI instructions, algorithm rationale and code walkthrough |
| `docs/developer-guide.md` | Installation, verification and reproduction |
| `docs/archive/` | Earlier project plans and audits |
| `autonomous-robot-navigation-rl/app/streamlit_app.py` | Main app entry point |
| `autonomous-robot-navigation-rl/src/robot_navigation/navigation_ui.py` | Shared simulator navigation |
| `autonomous-robot-navigation-rl/src/robot_navigation/warehouse_ui.py` | Four-algorithm replay, training history and custom routes |
| `autonomous-robot-navigation-rl/src/robot_navigation/warehouse_results_ui.py` | Source-backed warehouse results |
| `autonomous-robot-navigation-rl/src/robot_navigation/warehouse_study.py` | Registered training, recording audits and aggregation |
| `autonomous-robot-navigation-rl/src/robot_navigation/project_assets.py` | Current dependency inventory and checks |
| `autonomous-robot-navigation-rl/configs/warehouse_comparison.yaml` | Shared budget, seeds and method settings |
| `autonomous-robot-navigation-rl/data/warehouse/warehouse-delivery-v1/` | Floor, stations, jobs and fixed route split |
| `autonomous-robot-navigation-rl/artifacts/warehouse/warehouse-four-algorithm-v1/` | Twelve model runs, recordings, sealed inputs and measured comparison |
| `autonomous-robot-navigation-rl/docs/warehouse_comparison_protocol.md` | Registered experiment rules |
| `autonomous-robot-navigation-rl/docs/warehouse_comparison_results.md` | Measured comparison and limitations |
| `autonomous-robot-navigation-rl/docs/technical_report.md` | Current technical report entry point |
| `autonomous-robot-navigation-rl/developer/tests/` | Automated regression tests |
| `autonomous-robot-navigation-rl/developer/commands/` | Analysis and historical reproduction commands |
| `autonomous-robot-navigation-rl/.runtime/` | Ignored disposable test space and cache |

The original dark interface remains in `app_theme.py`. `warehouse_scene.py` supplies warehouse annotations; `preview_3d.py` supplies the optional temporary 3D view. `structured_ui.py` and the historical result modules retain earlier maps and methods.

The application folder and Python package names remain stable to preserve the installed environment and historical paths. Frozen source bundles, model IDs and protocols keep their original names and checksums. In particular, `configs/warehouse_demo.yaml` is a pinned environment dependency of registered studies; its historical filename does not define the current project. The earlier single-seed warehouse run remains an exploratory archive and is excluded from the new three-seed comparison.
