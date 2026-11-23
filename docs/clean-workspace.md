# One active workspace

Open **Robot Navigation.code-workspace** at the Desktop repository root. It shows the current application and its guides. The primary Desktop checkout is synchronized with current main; it no longer runs through `.runtime/newapp`.

Double-click **Start Robot Navigation.cmd**, or run `py -3 run_app.py`. The launcher uses `autonomous-robot-navigation-rl/.venv`, sets the source path to this checkout and serves http://127.0.0.1:8517.

Edit `autonomous-robot-navigation-rl/src/robot_navigation/ui/` for the existing presentation; use the [code walkthrough](code-walkthrough.md) for service/environment/policy responsibilities. `partial_` modules describe the partially observed task and remain required. Saved-model module/class paths are retained for compatibility.

Historical briefs and prior notes are under `docs/history/`; complete source/data history is under `archive/`. These folders, saved runs, environments and local runtime scratch are excluded from ordinary editor search. They remain on disk and in Git rather than being mixed into the active simulator.

Earlier dirty Desktop source was snapshotted before replacement. Its old data/artifact directories were preserved in ignored `.runtime/recovery/` storage. Historical linked checkouts are reference workspaces, not app dependencies. See [maintenance notes](maintenance.md) for the consolidation receipt.
