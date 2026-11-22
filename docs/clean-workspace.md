# Clean development workspace

Open **Robot Navigation.code-workspace** from the repository root. It shows two folders: the active application and the project guide. Research archives, saved runs, machine environments and generated files are excluded from the editor's main file tree and search; they remain on disk and in the existing Git history.

## Start

Double-click **Start Navigation.cmd** inside the application, or **Start Robot Navigation.cmd** at the repository root. The app runs at http://127.0.0.1:8517. The launcher prints the workspace it is using and always sets the source path to this checkout.

The local machine uses its existing compatible Python environment, recorded in the ignored `autonomous-robot-navigation-rl/.runtime/python-path.txt`. On another machine, follow `windows-install.md`, create the application's `.venv`, or pass `python run_app.py --python <interpreter>`.

## Edit

| Location in the application | Purpose |
| --- | --- |
| `app/streamlit_app.py` | Application entry point |
| `src/robot_navigation/dynamic_demo_ui.py` | Five-page routing |
| `src/robot_navigation/ui/styles.css` | Warehouse visual theme |
| `src/robot_navigation/ui/components.py` | Navigation, headers and shared cards |
| `src/robot_navigation/ui/simulator.py` | Route controls, maps and passive playback |
| `src/robot_navigation/ui/training.py` | Explicit training and checkpoint checks |
| `src/robot_navigation/ui/comparison.py` | Saved evidence, comparisons and downloads |
| `src/robot_navigation/ui/reference.py` | Guide and read-only archives |
| `src/robot_navigation/app_*.py` | Registry and application services |
| `src/robot_navigation/partial_*.py`, `phase3_*.py` | Environment, policies and assessment logic |
| `developer/tests` | Active product acceptance tests |

The earlier dark warehouse interface is the visual reference. The active app retains Phase 3 models, partial sensing, five actions and honest recorded outcomes. There is no classic-app runtime behind the restored styling.

Training remains an explicit action. Viewing the saved holdout and playing recordings cannot rerun frozen evaluations. Use `test_product_phase3.py` for acceptance without training. Some integration tests in `test_dynamic_app.py` have training fixtures; exclude those when training is prohibited.

The original dirty Desktop checkout and its worktrees remain untouched. The integrated cleanup is on `feature/phase3-cleanup-assignment-audit-v1`. Its base is the completed Phase3 delivery. UI changes from the separate Documents workspace were copied into this branch and reviewed; the source workspace remains unchanged. Main has not been merged or changed.

Read [the code walkthrough](code-walkthrough.md), [the assignment audit](assignment-compliance-audit.md) and [cleanup validation](cleanup-validation.md) for the integrated delivery.
