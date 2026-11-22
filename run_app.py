"""Launch the warehouse interface using this checkout's Phase 3 source."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
APP = ROOT / "autonomous-robot-navigation-rl"


def resolve_interpreter(explicit=None):
    if explicit or os.environ.get("ROBOT_NAV_PYTHON"):
        return explicit or os.environ["ROBOT_NAV_PYTHON"]
    local = APP / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if local.is_file():
        return str(local)
    # Reuse a local environment without relying on its editable installation.
    saved = APP / ".runtime/python-path.txt"
    if saved.is_file():
        configured = saved.read_text(encoding="utf-8-sig").strip()
        if Path(configured).is_file():
            return configured
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="interpreter")
    parser.add_argument("--port", type=int, default=8517)
    parser.add_argument("--check", action="store_true")
    args, extra = parser.parse_known_args()
    interpreter = resolve_interpreter(args.interpreter)
    env = dict(os.environ, PYTHONPATH=str(APP / "src"))
    check = subprocess.run(
        [
            interpreter,
            "-c",
            "import streamlit,stable_baselines3,torch; "
            "from robot_navigation.dynamic_demo_ui import render; "
            'print("Warehouse Phase 3 application imports ready")',
        ],
        cwd=APP,
        env=env,
    )
    if check.returncode:
        print(
            "Follow docs/windows-install.md, or supply --python with a compatible environment."
        )
        return check.returncode
    if args.check:
        return 0
    print(
        f"Autonomous Warehouse Robot Navigation: http://127.0.0.1:{args.port}",
        flush=True,
    )
    print(f"Workspace: {ROOT}", flush=True)
    command = [
        interpreter,
        "-m",
        "streamlit",
        "run",
        str(APP / "app/streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
        *extra,
    ]
    try:
        return subprocess.call(command, cwd=APP, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
