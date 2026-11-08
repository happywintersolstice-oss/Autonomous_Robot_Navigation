"""Outcome/support reconciliation, undefined routes and verified publication."""
from copy import deepcopy
import math

import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT, read_json, write_json
from robot_navigation.four_way_analysis import ANALYSIS_STUDY, grouped_metrics, stats, saved_analysis
from robot_navigation.release import sha


def test_groups_preserve_zero_success_and_partition_counts_and_actions():
    labels = {'a': {'distance': 'short', 'region': 'north'},
              'b': {'distance': 'long', 'region': 'south'}}
    rows = [{'scenario_id':'a','outcome':'success','steps':3,'return':9.97,'spl':1.},
            {'scenario_id':'b','outcome':'timeout','steps':200,'return':-2.,'spl':0.}]
    groups = grouped_metrics(rows, labels)
    for dimension in ('distance', 'region'):
        selected = [g['metrics'] for g in groups if g['dimension']==dimension]
        assert sum(g['episodes'] for g in selected)==2
        assert sum(g['interactions'] for g in selected)==203
        assert sum(g['counts']['success'] for g in selected)==1
    failed = next(g['metrics'] for g in groups if g['group']=='long')
    assert failed['mean_successful_route_length'] is None
    assert failed['counts']=={'success':0,'collision':0,'timeout':1}
    assert stats([None,2.,4.])=={'mean':3.,'sample_sd':math.sqrt(2.),'contributing_seeds':2}
    assert stats([None])=={'mean':None,'sample_sd':None,'contributing_seeds':0}
    with pytest.raises(ValueError,match='Nonfinite'):
        stats([float('nan')])
    with pytest.raises(ValueError,match='coverage'):
        grouped_metrics(rows[:-1],labels)
    with pytest.raises(ValueError,match='coverage'):
        grouped_metrics([*rows,deepcopy(rows[0])],labels)


def test_saved_analysis_rejects_modified_published_data(tmp_path):
    study=tmp_path/'artifacts/studies'/ANALYSIS_STUDY
    assert saved_analysis(tmp_path) is None
    write_json(study/'status.json',{'status':'running'})
    assert saved_analysis(tmp_path) is None
    write_json(study/'summary.json',{'status':'verified'})
    write_json(study/'verification.json',{'status':'passed','output_hashes':{'summary.json':sha(study/'summary.json')}})
    write_json(study/'status.json',{'status':'complete'})
    assert saved_analysis(tmp_path)['status']=='verified'
    write_json(study/'summary.json',{'status':'altered'})
    with pytest.raises(ValueError,match='verification'):
        saved_analysis(tmp_path)


def test_real_analysis_reconciles_all_seeds_and_geometry_partitions():
    summary=saved_analysis()
    original=read_json(ROOT/'artifacts/studies/four-rl-v1/summary.json')
    assert list(summary['systems'])==['q-learning','sarsa','dqn','ppo']
    for algorithm,system in summary['systems'].items():
        assert system['metrics']['success_rate']['mean']==original['systems'][algorithm]['metrics']['success_rate']['mean']
        assert len(system['groups'])==17 and len(system['seed_results'])==5
        for cost in system['costs'].values():
            assert cost['mean']>0 and cost['contributing_seeds']==5
        for dimension in {g['dimension'] for g in system['groups']}:
            groups=[g for g in system['groups'] if g['dimension']==dimension]
            assert sum(g['scenarios_per_seed'] for g in groups)==928
            for index,seed in enumerate(system['seed_results']):
                for outcome,count in seed['metrics']['counts'].items():
                    assert sum(g['seed_results'][index]['metrics']['counts'][outcome] for g in groups)==count


def test_dashboard_exposes_costs_and_all_four_robustness_families():
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    def box(label):
        return next(widget for widget in app.selectbox if widget.label==label)
    box('Dashboard view').set_value('Agent comparison').run()
    assert not app.exception
    assert 'Successful route actions' in app.dataframe[0].value.columns
    costs=next(df.value for df in app.dataframe if 'Checkpoint bytes' in df.value.columns)
    assert list(costs['Algorithm'])==['Q-learning','SARSA','DQN','PPO']
    assert all(costs['Training seconds']>0)
    box('Dashboard view').set_value('Optimization and robustness').run()
    for dimension in ('distance','corridor','wall proximity','start region','goal region','region crossing'):
        box('Four-algorithm scenario dimension').set_value(dimension).run()
        assert not app.exception
        table=app.dataframe[0].value
        assert set(table['Algorithm'])=={'Q-learning','SARSA','DQN','PPO'}
        assert table[table['Algorithm']=='PPO']['Pairs per seed'].sum()==928
