"""Synthetic schemas/statistics only; no PPO performance evidence is generated."""
from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT,load_task,read_json,write_json
from robot_navigation.four_way import summarize,saved_summary
from robot_navigation.interface import run_label


def fixture_rows():
    rows=read_json(ROOT/'artifacts/studies/part-a-v1/run_results.json')
    # Fixture PPO rows duplicate DQN numbers to test aggregation, not learning.
    fixtures=[{**deepcopy(row),'algorithm':'ppo','run_id':f"fixture-ppo-{row['seed']}"}
              for row in rows if row['algorithm']=='dqn']
    return rows+fixtures


def test_four_way_summary_keeps_all_seeds_and_reports_sample_sd():
    rows=fixture_rows();summary=summarize(rows,load_task().signature,range(928))
    assert list(summary['systems'])==['q-learning','sarsa','dqn','ppo']
    assert all(system['seed_count']==5 for system in summary['systems'].values())
    assert summary['systems']['ppo']['metrics']==summary['systems']['dqn']['metrics']
    assert summary['systems']['ppo']['training_interactions']==1000000
    assert summary['final_test_evaluated'] is False
    assert summary['original_final_selection_changed'] is False


@pytest.mark.parametrize('change',['missing','duplicate','budget','count'])
def test_incomplete_or_incomparable_rows_cannot_be_published(change):
    rows=fixture_rows()
    if change=='missing': rows.pop()
    elif change=='duplicate': rows[-1]['seed']=7
    elif change=='budget': rows[-1]['pilot']['training_interactions']=10000
    else: rows[-1]['validation']['counts']['success']+=1
    with pytest.raises(ValueError):
        summarize(rows,load_task().signature,range(928))


def test_four_way_dashboard_requires_passed_verification(tmp_path):
    study=tmp_path/'artifacts/studies/four-rl-v1'
    assert saved_summary(tmp_path) is None
    write_json(study/'summary.json',{'status':'verified'})
    assert saved_summary(tmp_path) is None
    write_json(study/'verification.json',{'status':'failed'})
    with pytest.raises(ValueError,match='verification'):
        saved_summary(tmp_path)


def test_ppo_diagnostics_are_probabilities_instead_of_q_values():
    episode={'transitions':[{'observation':[0,0,0,1],'scores':None,
        'decision':{'action':3,'action_probabilities':[.1,.2,.3,.4],'state_value':.5}}]}
    script='from robot_navigation.dashboard_ui import diagnostics\n'+f'diagnostics({episode!r},1,"fixture")\n'
    app=AppTest.from_string(script,default_timeout=15).run()
    assert not app.exception
    assert list(app.dataframe[0].value.columns)==['Action','Action probability']
    assert any('not Q-values' in caption.value for caption in app.caption)
    assert 'Additional baseline' in run_label(ROOT/'m7-ppo-seed7')
