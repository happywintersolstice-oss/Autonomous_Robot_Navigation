"""Staged extensions preserve old rows and expose their added-seed evidence."""
from copy import deepcopy

import numpy as np
import pytest
import yaml

from robot_navigation.core import ROOT, load_task, read_json
from robot_navigation.neural_confirmation import STUDY, combined_summary, prepare, saved_summary


@pytest.fixture
def plan():
    return yaml.safe_load((ROOT/'configs/neural_confirmation.yaml').read_text())


def test_extension_changes_only_training_seed_and_unique_identity(plan):
    original,jobs,aggregate_plan,engine=prepare(plan,ROOT)
    assert len(original)==12 and len(jobs)==8 and len(engine)==12
    assert aggregate_plan['confirmation_seeds']==[7307,7411,7523,7607,7717]
    assert sum(job['config']['interactions'] for job in jobs)==1600000
    for job in jobs:
        config=job['config']
        reference=next(row for row in original if row['algorithm']==config['algorithm'] and row['stage']==config['stage'])
        before=read_json(ROOT/'artifacts/runs'/reference['run_id']/'configuration.json')['training']
        assert {key for key in before if before[key]!=config[key]}=={'seed','run_id'}
        assert config['seed'] in plan['additional_seeds']


def test_extension_rejects_retuning_or_budget_change(plan):
    for key,value in [('additional_seeds',[7607,7607]),('selected_candidates',{'dqn':'c02','ppo':'c01'}),('interactions',100000)]:
        with pytest.raises(ValueError,match='declared'):
            prepare({**plan,key:value},ROOT)


def additions_fixture(original,plan):
    additions=[]
    for family in plan['algorithms']:
        for stage in ('control','tuned'):
            reference=next(row for row in original if row['algorithm']==family and row['stage']==stage)
            for seed in plan['additional_seeds']:
                row=deepcopy(reference)
                row.update(seed=seed,run_id=f'm9-{stage}-{family}-seed{seed}')
                additions.append(row)
    return additions


def test_five_seed_aggregation_and_added_seed_results_are_separate(plan):
    original,_,aggregate_plan,_=prepare(plan,ROOT)
    before=deepcopy(original)
    additions=additions_fixture(original,plan)
    result=combined_summary(original,additions,plan,aggregate_plan,load_task().signature)
    assert original==before
    assert result['expansion_declared_after_original_results']
    assert result['new_training_interactions']==1600000
    assert result['combined_confirmation_training_interactions']==4000000
    for family,system in result['systems'].items():
        assert system['seed_count']==5
        assert result['added_seed_systems'][family]['seed_count']==2
        values=[row['validation']['success_rate'] for row in system['conditions']['tuned']['seed_results']]
        assert system['conditions']['tuned']['metrics']['success_rate']['mean']==np.mean(values)
        assert system['conditions']['tuned']['metrics']['success_rate']['sample_sd']==np.std(values,ddof=1)
    with pytest.raises(ValueError,match='incomplete'):
        combined_summary(original,additions[:-1],plan,aggregate_plan,{})
    additions[0]['candidate']='c03'
    with pytest.raises(ValueError,match='candidate'):
        combined_summary(original,additions,plan,aggregate_plan,{})


def test_published_five_seed_evidence_reconciles_declared_work(plan):
    result=saved_summary(ROOT)
    assert result is not None
    study=ROOT/'artifacts/studies'/STUDY
    verification=read_json(study/'verification.json')
    assert result['new_training_runs']==8
    assert result['unique_confirmation_runs']==20
    assert result['new_training_interactions']==1600000
    assert result['combined_confirmation_training_interactions']==4000000
    assert len(verification['audits'])==20
    assert sum(a['checkpoints_verified'] for a in verification['audits'])==400
    assert verification['historical_inputs_unchanged']
    assert verification['new_final_test_policy_executions']==0
    for family,system in result['systems'].items():
        assert system['seed_count']==5
        assert system['seeds']==plan['original_seeds']+plan['additional_seeds']
        assert result['added_seed_systems'][family]['seeds']==plan['additional_seeds']


def test_dashboard_switches_between_original_and_five_seed_evidence():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=25).run()
    next(b for b in app.selectbox if b.label=='Dashboard view').set_value('Neural tuning').run()
    assert not app.exception
    assert set(app.dataframe[0].value['Training seeds'])=={5}
    assert any('declared after' in item.value for item in app.info)
    next(r for r in app.radio if r.label=='Study').set_value('Original three-seed study').run()
    assert not app.exception
    assert set(app.dataframe[0].value['Training seeds'])=={3}
    next(r for r in app.radio if r.label=='Study').set_value('Five-seed extension').run()
    assert not app.exception
    assert set(app.dataframe[0].value['Training seeds'])=={5}


def test_added_seed_has_late_actual_learning_replay():
    from streamlit.testing.v1 import AppTest
    from robot_navigation.dashboard import training_paths
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=25).run()
    app.session_state['replay_tab']='Learning replay'
    app.run()
    directory=ROOT/'artifacts/runs/m9-tuned-ppo-seed7607-attempt2'
    next(b for b in app.selectbox if b.label=='Training run').set_value(directory).run()
    next(b for b in app.selectbox if b.label=='Training episode').set_value(training_paths(directory)[-1]).run()
    assert not app.exception
    player=app.session_state['playback']
    assert player.episode['mode']=='training'
    assert player.episode['recording_window_index']==4
    assert player.episode['training_start_interaction']>=190000
    next(b for b in app.button if b.label=='Next step').click().run()
    assert app.session_state['playback'].index==1
