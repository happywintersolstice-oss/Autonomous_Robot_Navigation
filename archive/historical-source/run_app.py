"""Start the saved robot navigation app with its local Python environment."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
APPLICATION = ROOT / 'autonomous-robot-navigation-rl'


def main():
    interpreter = APPLICATION / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not interpreter.is_file():
        print('The app needs its Python environment first. Follow docs/developer-guide.md to install it.')
        return 1
    command = [str(interpreter), '-m', 'streamlit', 'run', 'app/streamlit_app.py',
               '--server.address', '127.0.0.1', '--server.port', '8517',
               '--browser.gatherUsageStats', 'false', *sys.argv[1:]]
    print('Robot Navigation opens at http://127.0.0.1:8517. Keep this window open while using the app.',flush=True)
    try:
        return subprocess.call(command,cwd=APPLICATION)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
