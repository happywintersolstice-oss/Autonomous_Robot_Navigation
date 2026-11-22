# Windows installation and launch

Open PowerShell in the repository root. Use Python 3.11+; the delivered CPU smoke used Python 3.14. No GPU is required. PyTorch downloads can be large.

```powershell
git config core.longpaths true
cd autonomous-robot-navigation-rl
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
cd ..
py -3 run_app.py --check
py -3 run_app.py
```

Open **http://127.0.0.1:8517**, or double-click **Start Robot Navigation.cmd** next time. Keep its terminal open; Ctrl+C closes it. An existing environment can be selected with `py -3 run_app.py --python C:\path\to\python.exe`. If 8517 is occupied, intentionally stop that server before starting another.

The app opens directly to the new partial/dynamic simulator. In Train Models explicitly start Quick and Refresh. Training runs in a separate hidden process; closing the browser does not stop it. Cancel stops at a safe action boundary and preserves partial output. Standard is larger and never automatic.

A short CPU starter:

```powershell
cd autonomous-robot-navigation-rl
.\.venv\Scripts\python.exe -m robot_navigation.app_training --preset quick --models all --actions 128 --seed 74001
.\.venv\Scripts\python.exe -m robot_navigation.app_training --list-models
```

New models appear on refresh. Keep `runs/` to retain them. Reset only changes the replay cursor. Missing/incompatible models produce a friendly error, never a planner substitution.

Historical ZIP export (repository root): `python scripts/export_history_archives.py --output-dir history-exports`; verify with the same command plus `--verify`. Choose a fresh folder and sufficient disk space. Export never executes old policies or regenerates experiments.
