"""The registered comparison retains all runs, exact budgets and auditable outcomes."""
import gzip
import json

import numpy as np
import pytest

from robot_navigation.core import ROOT,read_json,write_json,digest
from robot_navigation.episodes import aggregate,validate_replay
from robot_navigation.release import sha
from robot_navigation.warehouse import load
from robot_navigation.warehouse_study import (
    FAMILIES,checked_freeze,checked_run,configuration,location,monitoring_routes,plan,saved,
)


def test_registered_matrix_has_equal_budgets_and_disjoint_initial_pairs():
    task,_,routes=load()
    config=plan()
    registered=checked_freeze()
    assert len(routes['train'])==715 and len(routes['validation'])==188
    assert not {r['id'] for r in routes['train']}&{r['id'] for r in routes['validation']}
    assert len({r['id'] for r in monitoring_routes(routes)})==35
    assert registered['monitoring']==monitoring_routes(routes)
    for algorithm in FAMILIES:
        for seed in config['seeds']:
            c=configuration(algorithm,seed,config)
            assert c['interactions']==600000 and c['discount']==.99
    assert registered['signature']==task.signature


def test_published_comparison_reproduces_all_recorded_metrics_and_seed_aggregation():
    task,_,routes=load()
    result=saved()
    assert result['total_training_interactions']==7200000
    assert set(result['systems'])==set(FAMILIES)
    for algorithm,system in result['systems'].items():
        assert system['seeds']==[19411,19421,19431]
        rates=[]
        for seed in system['seeds']:
            run,folder=checked_run(algorithm,seed)
            assert run['training']['training_interactions']==600000
            assert run['training']['total_state_action_visits']==600000
            assert len(run['training_traces']['records'])==6
            with gzip.open(folder/'assessment.jsonl.gz','rt',encoding='utf-8') as file:
                records=[json.loads(line) for line in file]
            assert [r['scenario'] for r in records]==routes['validation']
            for episode in records:
                validate_replay(task,episode)
            actual=aggregate([e['metrics'] for e in records])
            assert actual['counts']==run['assessment']['counts']
            for metric in ('success_rate','collision_rate','timeout_rate','spl','mean_return'):
                assert actual[metric]==pytest.approx(run['assessment'][metric])
            rates.append(actual['success_rate'])
        assert system['metrics']['success_rate']['mean']==pytest.approx(np.mean(rates))
        assert system['metrics']['success_rate']['sample_sd']==pytest.approx(np.std(rates,ddof=1))


def test_checked_run_rejects_a_changed_output(tmp_path):
    folder=location(tmp_path)/'runs/warehouse-dqn-seed19411'
    folder.mkdir(parents=True)
    write_json(folder/'status.json',{'status':'verified'})
    write_json(folder/'summary.json',{'algorithm':'dqn'})
    write_json(folder/'outputs.json',{'summary.json':sha(folder/'summary.json')})
    assert checked_run('dqn',19411,tmp_path)[0]['algorithm']=='dqn'
    write_json(folder/'summary.json',{'algorithm':'ppo'})
    with pytest.raises(ValueError,match='output changed'):
        checked_run('dqn',19411,tmp_path)


def test_registered_inputs_cannot_change_after_training(tmp_path):
    folder=location(tmp_path)
    (folder/'snapshot').mkdir(parents=True)
    (tmp_path/'input.txt').write_bytes(b'registered')
    (folder/'snapshot/input.txt').write_bytes(b'registered')
    record={'files':{'input.txt':sha(tmp_path/'input.txt')}}
    write_json(folder/'freeze.json',record)
    write_json(folder/'seal.json',{'freeze_digest':digest(record)})
    assert checked_freeze(tmp_path)==record
    (tmp_path/'input.txt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):
        checked_freeze(tmp_path)
