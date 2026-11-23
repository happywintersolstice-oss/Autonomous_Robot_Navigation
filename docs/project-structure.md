# Project structure

| Path | Responsibility |
| --- | --- |
| `run_app.py`, `Start Robot Navigation.cmd` | Launch this checkout on localhost:8517 |
| `Robot Navigation.code-workspace` | One editor view of the current application and guides |
| `autonomous-robot-navigation-rl/app/streamlit_app.py` | Streamlit entry point |
| `autonomous-robot-navigation-rl/src/robot_navigation/ui/` | Five pages, shared presentation, stylesheet and help |
| `src/robot_navigation/app_*.py` inside the application | Checkpoint registry, explicit jobs and episode services |
| `src/robot_navigation/partial_*.py` | Task contract, environment, sensors, memory, policy foundations and metrics |
| `src/robot_navigation/phase3_*.py`, `product_*.py` | Versioned learned representations, curricula and registered study logic |
| `src/robot_navigation/ensemble.py`, `rubric_study.py` | Explicit model combination and its registered development evidence |
| `autonomous-robot-navigation-rl/configs/`, `data/` | Input configuration and versioned datasets |
| `autonomous-robot-navigation-rl/runs/` | Immutable model runs, selections and recorded comparisons |
| `autonomous-robot-navigation-rl/artifacts/` | Preserved scientific cohorts at their registered paths |
| `autonomous-robot-navigation-rl/developer/commands/` | Actual figure, appendix and submission-readiness tools |
| `docs/` | Current guides, results, rubric status and submission support |
| `docs/history/` | Retired briefs, reports and validation notes |
| `archive/` | Historical data/source, origin catalogs, relocation map and recovery snapshots |
| `scripts/` | Read-only historical ZIP export and verification |
| `.runtime/`, `.venv/`, build/cache folders | Ignored local runtime and environment files |

There is one active package and no legacy warehouse app or automated test directory. Historical model versions retain their import paths because their saved weights rely on them.
