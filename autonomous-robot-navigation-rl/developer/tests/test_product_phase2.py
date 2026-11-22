"""Focused Phase 2 acceptance; no archived data hashing or historical test execution."""
import json,csv,zipfile,shutil
from pathlib import Path
import numpy as np
import pytest
from robot_navigation.core import ROOT,digest,read_json
from robot_navigation.product_curriculum import cases,manifest,CurriculumEnv
from robot_navigation.product_features import rows,ROW,VERSION,LocalLinearPolicy,task_signature
from robot_navigation.partial_contract import configuration
from robot_navigation.app_registry import registry,load_model,preferred_by_algorithm
from robot_navigation.app_service import saved_comparison,frame

def test_seeded_reachable_layout_separation_and_curriculum():
    m=manifest();assert len(m['development_cases'])==24 and len(m['training_cases'])==192 and not m['layout_overlap']
    assert {c['layout_hash'] for c in cases('train')}.isdisjoint(c['layout_hash'] for c in cases('development'))
    assert len({c['layout_hash'] for c in cases('development')})==24
    assert all(c['split']=='validation' for c in cases('development'))
    assert {'short','longer'}=={c['distance_band'] for c in cases('development')}
    assert any(c['events'] for c in cases('development')) and any(not c['events'] for c in cases('development'))
    e=CurriculumEnv(10000,42,'none');o,_=e.reset(seed=42);assert int(np.rint(np.abs(o[:2]-o[2:4]).sum()*9))<=3
    assert m==manifest()

def test_observation_only_features_no_mask_or_hidden_clear_time():
    from robot_navigation.app_scenarios import scenario,NAMES
    from robot_navigation.partial_dynamic_env import PartialDynamicEnv
    permanent=scenario(NAMES[2]);temporary=scenario(NAMES[2]);temporary['events'][0]['clear']=19
    a=PartialDynamicEnv([permanent]);b=PartialDynamicEnv([temporary]);oa,_=a.reset();ob,_=b.reset();assert np.array_equal(oa,ob);assert np.array_equal(rows(oa),rows(ob))
    p=LocalLinearPolicy('q-learning',configuration(),1)
    for action in range(5):
        p.weights[:]=0;p.weights[13+action]=10
        assert p.act(oa)==action  # Even an observed collision remains possible: no action mask.
    assert rows(oa).shape==(5,ROW)
    with pytest.raises(ValueError):rows({'events':temporary['events']})

def test_training_shaping_is_public_and_evaluation_reward_unchanged():
    from robot_navigation.partial_dynamic_env import PartialDynamicEnv
    e=CurriculumEnv(100,2,'progress-v1');o,_=e.reset(seed=2);n,r,t,x,info=e.step(4)
    assert r==info['official_reward']==-.01 and info['training_shaping']==0
    assert set(info)=={'outcome','steps','official_reward','training_shaping'}
    from robot_navigation.app_scenarios import scenario,NAMES
    raw=PartialDynamicEnv([scenario(NAMES[0],'Static')]);raw.reset();assert raw.step(4)[1]==-.01

def test_existing_starters_remain_loadable_and_failed_runs_are_excluded(tmp_path):
    items=[i for i in registry() if i['id'].startswith('20261122T112025Z-a7301ff2/')];assert len(items)==4
    e=CurriculumEnv(100,2,'none');o,_=e.reset(seed=2)
    for item in items:assert load_model(item).act(o) in range(5)
    folder=tmp_path/'training'/'failed';shutil.copytree(Path(items[0]['path']).parent.parent,folder)
    m=read_json(folder/'manifest.json');m['status']='cancelled';(folder/'manifest.json').write_text(json.dumps(m))
    assert not registry(tmp_path)

def test_selected_models_have_real_learning_and_native_signatures():
    selected=preferred_by_algorithm();assert set(selected)=={'q-learning','sarsa','dqn','ppo'}
    for algorithm,item in selected.items():
        assert item['status']=='Best on development' and item['actions']>=10000
        p=load_model(item);assert p.signature==task_signature(configuration())
        evidence=item['learning_evidence'];assert evidence['parameter_l2_change']>0 and evidence['initial_parameter_sha256']!=evidence['final_parameter_sha256']
        assert evidence['td_updates']==item['actions'] if algorithm in ('q-learning','sarsa') else evidence['optimizer_steps']>0

def test_saved_comparison_full_panel_export_and_replay_integrity():
    r=saved_comparison();assert r and r['manifest']['panel_case_count']==24 and r['manifest']['case_count']==240
    assert not r['manifest']['layout_overlap'] and r['manifest']['inference_only']
    groups={row['policy'] for row in r['rows']};assert len(groups)==10
    for policy in groups:
        rows=[row for row in r['rows'] if row['policy']==policy]
        assert [x['scenario_hash'] for x in rows]==r['manifest']['case_hashes']
        assert all(int(x['steps'])<=200 for x in rows)
    for row in r['manifest']['summary']:
        assert row['Successes']+row['Collisions']+row['Timeouts']==row['Cases']
        if not row['Successes']:assert row['Mean successful steps'] is None
    folder=Path(r['folder'])
    with zipfile.ZipFile(folder/'results-and-replays.zip') as z:
        assert len([n for n in z.namelist() if n.endswith('.json.gz')])==240
        assert {'summary.csv','case-results.csv','manifest.json'}.issubset(z.namelist())
    from robot_navigation.app_service import read_replay
    record=read_replay(folder/r['rows'][0]['replay']);assert frame(record,0)==record['initial_snapshot']


def test_completed_study_cannot_repeat_training(monkeypatch):
    import robot_navigation.product_phase2 as study_module
    selected=read_json(ROOT/'runs/selection.json');study=ROOT/'runs'/selected['study']
    monkeypatch.setattr(study_module,'train',lambda *a,**k:pytest.fail('Completed study retrained'))
    with pytest.raises(ValueError,match='already completed'):study_module.run(study)

def test_effective_neural_settings_match_serialized_models():
    from robot_navigation.product_features import effective_settings
    selected=preferred_by_algorithm()
    for algorithm in ('dqn','ppo'):
        p=load_model(selected[algorithm]);settings=effective_settings(p)
        assert settings['batch_size']==p.model.batch_size
        if algorithm=='dqn':assert settings['buffer_size']==20000 and settings['batch_size']==64 and settings['target_update_interval']==500
        else:assert settings['n_epochs']==4 and settings['n_steps']==200 and settings['batch_size']==100
