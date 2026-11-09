"""Predeclared search boundaries and paired confirmation accounting."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT, load_task, read_json
from robot_navigation.interface import experiment_group, run_label
from robot_navigation.neural_tuning import (METRICS, choose_candidate, run_config,
                                            stats, summarize, validate_plan, STUDY, saved_summary)


@pytest.fixture
def plan():
    return yaml.safe_load((ROOT/'configs/neural_tuning.yaml').read_text())


def test_plan_reconciles_preliminary_seed_matrix_and_limits(plan):
    validate_plan(plan)
    assert 2*10000+8*100000+12*200000==plan['limits']['max_training_interactions_including_pilots']
    assert 2+8+12==plan['limits']['max_runs_including_pilots']
    for change in ({'confirmation_seeds':[7307,7411,7307]}, {'workers':3}, {'search_interactions':200000}):
        with pytest.raises(ValueError):
            validate_plan({**plan,**change})
    altered=deepcopy(plan);altered['candidates']['dqn']['c01']['observation_encoding']='relative'
    with pytest.raises(ValueError,match='outside'):
        validate_plan(altered)


@pytest.mark.parametrize('family',['dqn','ppo'])
def test_candidate_configuration_keeps_inputs_capacity_and_budget(plan,family):
    control=run_config(plan,family,'control',7307,root=ROOT)
    selected=run_config(plan,family,'tuned',7307,'c01',ROOT)
    assert control['interactions']==selected['interactions']==200000
    assert control['net_arch']==selected['net_arch']==[64,64]
    assert selected.get('observation_encoding')==control.get('observation_encoding')
    assert selected['training_trace_starts']==[0,50000,100000,150000,190000]
    changed={k for k in control if control[k]!=selected[k]}
    assert changed=={'run_id','stage',*plan['candidates'][family]['c01']}
    assert run_config(plan,family,'search',7201,'c02',ROOT)['interactions']==100000
    with pytest.raises(ValueError):
        run_config(plan,family,'control',7201,root=ROOT)
    with pytest.raises(ValueError):
        run_config(plan,family,'control',7307,'c01',ROOT)
    with pytest.raises(ValueError):
        run_config(plan,family,'unknown',7307,root=ROOT)


def search_rows(plan):
    return [{'stage':'search','algorithm':'dqn','seed':7201,'candidate':candidate,
             'validation':{'success_rate':.2,'spl':.2,'collision_rate':.1}}
            for candidate in plan['candidates']['dqn']]


def test_search_selection_is_complete_deterministic_and_confirmation_blind(plan):
    rows=search_rows(plan)
    assert choose_candidate(rows,'dqn',plan)=='c00'
    rows[1]['validation']['collision_rate']=.05
    assert choose_candidate(rows,'dqn',plan)=='c01'
    rows[2]['validation']['spl']=.21
    assert choose_candidate(rows,'dqn',plan)=='c02'
    rows[3]['validation']['success_rate']=.21
    assert choose_candidate(rows,'dqn',plan)=='c03'
    rows.append({'stage':'tuned','algorithm':'dqn','validation':{'success_rate':1}})
    assert choose_candidate(rows,'dqn',plan)=='c03'
    with pytest.raises(ValueError,match='complete'):
        choose_candidate(rows[1:],'dqn',plan)


def confirmations(plan):
    rows=[]
    for family in plan['algorithms']:
        for stage in ('control','tuned'):
            for seed in plan['confirmation_seeds']:
                validation={m:.1 if stage=='control' else .2 for m in METRICS}
                validation['episodes']=928
                rows.append({'algorithm':family,'stage':stage,'seed':seed,'validation':validation,
                             'pilot':{'training_interactions':200000},'costs':{'training_seconds':10}})
    return rows


def test_confirmation_uses_paired_training_seeds_and_sample_sd(plan):
    summary=summarize(confirmations(plan),{'dqn':'c01','ppo':'c02'},plan,load_task().signature)
    system=summary['systems']['dqn']
    assert system['seed_count']==3
    assert system['paired_change']['success_rate']['mean']==pytest.approx(.1)
    assert system['wins_losses_ties']=={'wins':3,'losses':0,'ties':0}
    assert stats([1,2,3,None])=={'mean':2.,'sample_sd':1.,'contributing_seeds':3}
    assert stats([None,None])=={'mean':None,'sample_sd':None,'contributing_seeds':0}
    assert not summary['final_test_evaluated'] and not summary['original_final_selection_changed']
    with pytest.raises(ValueError,match='matrix'):
        summarize(confirmations(plan)[1:],{'dqn':'c01','ppo':'c02'},plan,{})


def test_baseline_retention_explicitly_reuses_control_records(plan):
    rows=[r for r in confirmations(plan) if r['stage']=='control']
    summary=summarize(rows,{'dqn':'c00','ppo':'c00'},plan,{})
    for system in summary['systems'].values():
        assert system['baseline_retained']
        assert system['conditions']['control']['seed_results'] is system['conditions']['tuned']['seed_results']
        assert system['paired_change']['success_rate']['mean']==0
    with pytest.raises(ValueError,match='duplicate'):
        summarize(confirmations(plan),{'dqn':'c00','ppo':'c00'},plan,{})


def test_new_run_names_show_neural_study_provenance():
    path=Path('m9-tuned-ppo-seed7307')
    assert run_label(path)=='PPO · Selected settings · seed 7307'
    assert experiment_group(path)=='Neural tuning'
    assert run_label(Path('m9-search-dqn-c02-seed7201'))=='DQN · Tuning search c02 · seed 7201'
    assert 'Additional baseline' in run_label(Path('m7-ppo-seed7'))


def test_neural_tuning_page_without_published_results(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/neural_tuning.yaml').write_bytes((ROOT/'configs/neural_tuning.yaml').read_bytes())
    script=tmp_path/'page.py'
    script.write_text('from pathlib import Path\nfrom robot_navigation.core import load_task\n'
        'from robot_navigation.dashboard_ui import render_page\n'
        f'render_page("Neural tuning",load_task(),Path({str(tmp_path)!r}))\n')
    app=AppTest.from_file(str(script)).run()
    assert not app.exception and app.json
    assert any('Verified results' in item.value for item in app.info)


def test_published_neural_study_reconciles_all_seed_and_cost_rows(plan):
    summary=saved_summary(ROOT)
    assert summary is not None
    study=ROOT/'artifacts/studies'/STUDY
    verification=read_json(study/'verification.json')
    rows=read_json(study/'run_results.json')
    choices=read_json(study/'selected_configurations.json')
    assert choices['frozen_before_confirmation']
    assert len(rows)==summary['unique_training_runs']<=22
    assert sum(r['pilot']['training_interactions'] for r in rows)==summary['actual_training_interactions']<=3220000
    assert len(verification['audits'])==len(rows)
    assert verification['historical_inputs_unchanged']
    assert verification['new_final_test_policy_executions']==0
    assert not verification['original_final_selection_changed']
    for family,system in summary['systems'].items():
        assert choose_candidate(rows,family,plan)==system['selected_candidate']==choices['selected'][family]
        for stage,condition in system['conditions'].items():
            assert [r['seed'] for r in condition['seed_results']]==plan['confirmation_seeds']
            for row in condition['seed_results']:
                assert row['pilot']['training_interactions']==200000
                assert row['validation']['episodes']==928
                assert row['validation']['counts']['success']+row['validation']['counts']['collision']+row['validation']['counts']['timeout']==928


@pytest.mark.parametrize('family',['dqn','ppo'])
def test_published_neural_page_opens_predeclared_matched_pair(family):
    summary=saved_summary(ROOT)
    assert summary is not None
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=25).run()
    next(b for b in app.selectbox if b.label=='Dashboard view').set_value('Neural tuning').run()
    assert not app.exception and len(app.dataframe[0].value)==4
    system=summary['systems'][family]
    if system['baseline_retained']:
        assert not any(b.label==f'Compare {family.upper()} models' for b in app.button)
        return
    next(b for b in app.button if b.label==f'Compare {family.upper()} models').click().run()
    assert not app.exception
    assert app.session_state['dashboard_page']=='Navigation simulator'
    assert app.session_state['replay_tab']=='Compare agents'
    player=app.session_state['comparison_playback']
    assert len(player.episodes)==2 and player.index==0 and not player.playing
    assert player.episodes[0]['scenario']==player.episodes[1]['scenario']
    for episode,row in zip(player.episodes,(system['conditions']['tuned']['seed_results'][0],
                                           system['conditions']['control']['seed_results'][0]),strict=True):
        assert row['run_id'] in episode['policy_id']
    next(b for b in app.button if b.label=='Next step').click().run()
    assert app.session_state['comparison_playback'].index==1
