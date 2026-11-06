"""Restore an exact frozen archive despite checkout text normalization.

Default prepares/verifies only. --evaluate repeats the unchanged frozen test,
which is repeatability evidence and must not be used for new model selection.
"""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from robot_navigation.core import ROOT, read_json
from robot_navigation.release import verify_freeze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',default='.cache/frozen-release-replay-v1')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    source=ROOT/'artifacts/studies/final-v1'
    seal=read_json(source/'seal.json')['freeze_digest']
    freeze=read_json(source/'freeze.json')
    if platform.python_version()!=freeze['python']:
        raise ValueError(f"Frozen replay requires tested Python {freeze['python']}")
    for package,expected in freeze['dependencies'].items():
        if version(package)!=expected:
            raise ValueError(f'Frozen dependency mismatch: {package}; expected {expected}')
    workspace=(ROOT/args.workspace).resolve()
    if not workspace.is_relative_to(ROOT/'.cache'):
        raise ValueError('Use a fresh workspace directory under project .cache')
    shutil.copytree(source/'snapshot',workspace)
    study=workspace/'artifacts/studies/final-replay'
    study.mkdir(parents=True)
    for filename in ('freeze.json','seal.json'):
        shutil.copy2(source/filename,study/filename)
    shutil.copytree(source/'snapshot',study/'snapshot')
    verify_freeze(study,seal,root=workspace)
    print(f'Exact frozen workspace verified: {workspace}',flush=True)
    if args.evaluate:
        env={**os.environ,'PYTHONPATH':str(workspace/'src'),
             'MPLCONFIGDIR':str(workspace/'.cache/matplotlib')}
        subprocess.run([sys.executable,str(workspace/'scripts/run_final_evaluation.py'),'evaluate',
                        '--suite-id','final-replay','--freeze-digest',seal],cwd=workspace,env=env,check=True)
    else:
        print('Preparation only; no final policy execution. Use --evaluate with a new workspace for unchanged repeatability evaluation.')


if __name__=='__main__':
    main()
