"""Launch the only active partial/dynamic app on localhost:8517."""
import argparse,importlib.util,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;APP=ROOT/'autonomous-robot-navigation-rl'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--python',dest='interpreter');parser.add_argument('--check',action='store_true');args,extra=parser.parse_known_args()
    local=APP/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    interpreter=args.interpreter or os.environ.get('ROBOT_NAV_PYTHON') or (str(local) if local.is_file() else sys.executable)
    env=dict(os.environ,PYTHONPATH=str(APP/'src'))
    check=subprocess.run([interpreter,'-c','import streamlit,stable_baselines3,torch; from robot_navigation.dynamic_demo_ui import render; print("New partial/dynamic application imports ready")'],cwd=APP,env=env)
    if check.returncode:
        print('Install the environment using docs/windows-install.md, then try again.');return check.returncode
    if args.check:return 0
    print('Partial & Dynamic Robot Navigation: http://127.0.0.1:8517. Keep this window open.',flush=True)
    try:return subprocess.call([interpreter,'-m','streamlit','run','app/streamlit_app.py','--server.address','127.0.0.1','--server.port','8517','--browser.gatherUsageStats','false',*extra],cwd=APP,env=env)
    except KeyboardInterrupt:return 0
if __name__=='__main__':raise SystemExit(main())
