"""Isolated benchmark preparation + exact selected Q-policy training reproduction."""
import argparse
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from robot_navigation.agents import load_policy, viewer_checkpoint
from robot_navigation.benchmark import prepare
from robot_navigation.core import ROOT, load_task, read_json, write_json
from robot_navigation.part_a import train_part_a


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',default='.cache/reproduction-v1')
    args=parser.parse_args()
    isolated=(ROOT/args.workspace).resolve()
    if not isolated.is_relative_to(ROOT/'.cache'):
        raise ValueError('Reproduction workspace must be a new directory under project .cache')
    isolated.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    shutil.copytree(ROOT/'configs',isolated/'configs')
    for folder in ('src','app'):
        shutil.copytree(ROOT/folder,isolated/folder,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    shutil.copytree(ROOT/'data/maps',isolated/'data/maps')
    prepare(isolated)
    prepared=read_json(isolated/'data/scenarios/benchmark_report.json')
    original=read_json(ROOT/'data/scenarios/benchmark_report.json')
    assert prepared==original
    source=ROOT/'artifacts/runs/m4-tuned-q-learning-seed101'
    config=read_json(source/'configuration.json')['training']
    directory,summary,pilot=train_part_a(config,root=isolated)
    expected=load_policy(viewer_checkpoint(source),load_task())
    reproduced=load_policy(viewer_checkpoint(directory),load_task(isolated))
    assert np.array_equal(expected.q,reproduced.q)
    assert read_json(source/'q_learning_episodes.json')==read_json(directory/'q_learning_episodes.json')
    result={'status':'passed','isolated_workspace':isolated.relative_to(ROOT).as_posix(),
        'benchmark_report_identical':True,'selected_q_arrays_identical':True,'all_validation_episode_metrics_identical':True,
        'selected_interactions':read_json(directory/'manifest.json')['selected_interactions'],
        'reproduction_training_interactions':pilot['training_interactions'],'final_test_policy_executions':0,
        'seconds':perf_counter()-started,'limitations':'Uses installed locked dependencies on this machine; not a clean network dependency install or a hardware portability guarantee'}
    write_json(isolated/'verification.json',result)
    print(result)


if __name__=='__main__':
    main()
