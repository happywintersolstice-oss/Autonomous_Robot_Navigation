# Project layout

| Path | Purpose |
|---|---|
| `run_app.py`, `Start Robot Navigation.cmd` | Only app, localhost:8517 |
| `autonomous-robot-navigation-rl/app/streamlit_app.py` | New UI entry point |
| `Robot Navigation.code-workspace` | Clean editor view of active application and project guide |
| `src/robot_navigation/dynamic_demo_ui.py` | Five-page entry point and routing |
| `src/robot_navigation/ui/` | Warehouse theme, shared components and focused page modules |
| `src/robot_navigation/app_*.py` | Scenarios, strict native registry, explicit training and replay services |
| `src/robot_navigation/partial_*.py` | Five-action environment/encoders/learners/metrics |
| `src/robot_navigation/shared.py` | Neutral utilities; no old environment |
| `autonomous-robot-navigation-rl/runs/` | New immutable product outputs |
| `.runtime/jobs/` | Hidden-process progress, cancellation and logs |
| `archive/` | History, original source and complete origin/hash index |
| `artifacts/partial-dynamic-navigation-v1/` | Retained sealed Phase 5A–5H evidence, reference-only |
| `scripts/export_history_archives.py` | Stream/verify real historical ZIPs |
| `developer/tests/test_dynamic_app.py` | Focused product integration |

Historical code/tests/guides preserve original paths within archive or the path map. They are not installed as a secondary mode. All four active RL families use the new 409-observation/five-action task.

For a guided execution path, read [How the code works](code-walkthrough.md). See the separate [assignment audit](assignment-compliance-audit.md) for evidence and remaining submission requirements.

`src/robot_navigation/ensemble.py` owns the optional voting controller; `rubric_study.py` owns its registered development measurement. `docs/submission/` contains technical evidence and explicitly unfinished team submission support. No learner or historical evidence is duplicated into another app mode.
