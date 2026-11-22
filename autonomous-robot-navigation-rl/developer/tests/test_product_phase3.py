"""Focused Phase3 acceptance. No retraining or held-out policy execution here."""
import csv,gzip,json,shutil,zipfile
from pathlib import Path
import numpy as np
import pytest
from robot_navigation.core import ROOT,read_json,digest
from robot_navigation.app_registry import RUNS,registry,phase3_selected,load_model
from robot_navigation.phase3_data import verify_frozen,train_dev,canonical,structure,shortest,AUDIT_PATH
from robot_navigation.phase3_features import PublicHistory,HistoryLinearPolicy,rows,load_policy,VERSION
from robot_navigation.partial_contract import configuration
from robot_navigation.partial_dynamic_env import PartialDynamicEnv,make_scenario
from robot_navigation.release import sha

STUDY=RUNS/'product-phase3/20261122T163932Z-f85cb312'

def test_frozen_split_structure_and_reachability():
    suite=verify_frozen();data=train_dev();audit=read_json(AUDIT_PATH)
    from robot_navigation.product_curriculum import manifest
    old=manifest();other=old['training_cases']+old['development_cases']+data['training']+data['development']
    held=suite['cases'];assert len(held)==64 and len(data['training'])==288 and len(data['development'])==32
    assert not {canonical(c['rows']) for c in held}&{canonical(c['rows']) for c in other}
    assert not {structure(c['rows']) for c in held}&{structure(c['rows']) for c in other}
    assert not any(audit[k] for k in ('layout_hash_overlap','structural_hash_overlap','family_overlap','train_development_structural_overlap'))
    assert len({c['canonical_layout_hash'] for c in held})==64
    assert sum(bool(c['events']) for c in held)==36
    assert {c['distance_band'] for c in held}=={'short','long'}
    assert {'forced-wait','detour-available','none'}=={c['blockage_kind'] for c in held}
    for c in held:
        assert len(shortest(c['rows'],c['start'],c['goal']))-1==c['base_distance']<=40
        assert all(e['clear']<=60 for e in c['events'])

def test_indistinguishable_occupancy_history_and_all_five_actions():
    open_rows=['.'*10 for _ in range(10)];wall=list(open_rows);wall[5]='..#.......'
    a=PartialDynamicEnv([make_scenario(wall,[5,1],[5,8])])
    b=PartialDynamicEnv([make_scenario(open_rows,[5,1],[5,8],[{'id':'hidden','cell':[5,2],'start':0,'clear':19}])])
    oa,_=a.reset();ob,_=b.reset();ha=PublicHistory();hb=PublicHistory()
    for _ in range(4):
        assert np.array_equal(oa,ob) and np.array_equal(rows(ha.state(oa)),rows(hb.state(ob)))
        ha.remember(oa,4);hb.remember(ob,4);oa=a.step(4)[0];ob=b.step(4)[0]
    p=HistoryLinearPolicy('q-learning',configuration(),1)
    for action in range(5):
        p.reset();p.weights[:]=0;p.weights[13+action]=10;assert p.act(oa)==action
    assert rows(ha.state(oa)).shape==(5,104)
    with pytest.raises(ValueError):p.act({'events':[]})

def test_wrapper_and_inference_history_alignment_and_public_shaping():
    from robot_navigation.phase3_training import HistoryCurriculumEnv
    env=HistoryCurriculumEnv(100,2,'liveness-v1');state,_=env.reset(seed=2);h=PublicHistory()
    for _ in range(6):
        np.testing.assert_array_equal(state,h.state(state[:409]));old=state[:409].copy();h.remember(old,4)
        state,reward,term,trunc,info=env.step(4)
        assert reward==pytest.approx(info['official_reward']+info['training_shaping'])
        assert info['official_reward']==-.01 and set(info)=={'outcome','steps','official_reward','training_shaping'}
        assert not term and not trunc
    np.testing.assert_array_equal(state,h.state(state[:409]))

def test_four_models_real_updates_hashes_reload_and_old_compatibility(tmp_path):
    selected=phase3_selected();assert len(selected)==4
    case=train_dev()['development'][0];obs=PartialDynamicEnv([case]).reset(seed=0)[0]
    for item in selected.values():
        evidence=item['learning_evidence'];assert item['actions']==40000 and item['feature_version']==VERSION
        assert evidence['parameter_l2_change']>0 and evidence['initial_parameter_sha256']!=evidence['final_parameter_sha256']
        assert evidence['td_updates']==40000 if item['algorithm'] in ('q-learning','sarsa') else evidence['optimizer_steps']>0
        a=load_model(item);b=load_model(item);before=a.state_fingerprint()
        assert a.act(obs)==b.act(obs);a.reset();b.reset();assert a.act(obs)==b.act(obs)
        assert a.state_fingerprint()==before
    old=read_json(RUNS/'selection.json')['selected'];items={i['id']:i for i in registry()}
    for x in old.values():assert load_model(items[x['id']]).act(obs) in range(5)
    starters=[i for i in registry() if i['id'].startswith('20261122T112025Z-a7301ff2/')];assert len(starters)==4
    for item in starters:assert load_model(item).act(obs) in range(5)
    item=selected['q-learning'];path=tmp_path/'policy.npz';shutil.copyfile(item['path'],path)
    m=read_json(Path(item['path']).with_suffix('.json'));m['agent_state_values']=409;path.with_suffix('.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='mismatch'):load_policy(path,configuration())

def test_incomplete_runs_excluded_and_limits_reject_without_actions(tmp_path):
    from robot_navigation.phase3_training import train
    item=phase3_selected()['q-learning'];folder=tmp_path/'training'/'incomplete';shutil.copytree(Path(item['path']).parent.parent,folder)
    m=read_json(folder/'manifest.json')
    for status in ('running','failed','cancelled'):
        m['status']=status;(folder/'manifest.json').write_text(json.dumps(m));assert registry(tmp_path)==[]
    for args in [dict(actions=50001,models=['q-learning']),dict(actions=201,models=['ppo']),dict(actions=3,models=['dqn'])]:
        with pytest.raises(ValueError):train(output=tmp_path,**args)
    assert len(list((tmp_path/'training').iterdir()))==1

def test_cancellation_and_hour_cap_before_any_transition(tmp_path,monkeypatch):
    import robot_navigation.phase3_training as module
    monkeypatch.setattr(module,'build_policy',lambda *a,**k:pytest.fail('Learner built after cancellation/cap'))
    job=tmp_path/'job';job.mkdir();(job/'cancel.request').touch()
    with pytest.raises(InterruptedError):module.train(models=['q-learning'],actions=1,output=tmp_path/'cancel',job_dir=job)
    clock=iter([0.,3601.]);monkeypatch.setattr(module,'perf_counter',lambda:next(clock))
    with pytest.raises(TimeoutError):module.train(models=['q-learning'],actions=1,output=tmp_path/'cap')
    for root in ('cancel','cap'):
        m=read_json(next((tmp_path/root/'training').glob('*/manifest.json')));assert m['actual_actions']==0
        assert registry(tmp_path/root)==[]

def test_locked_selection_no_test_tuning_or_reuse(monkeypatch):
    import robot_navigation.phase3_study as module
    selection=read_json(STUDY/'selection.json');assert not selection['holdout_outcomes_used'] and selection['heldout_policy_evaluations_at_selection']==0
    assert sha(STUDY/'selection.json')==read_json(STUDY/'selection-lock.json')['sha256']
    assert read_json(STUDY/'heldout-evaluation-complete.json')['run_count']==1
    monkeypatch.setattr(module,'train',lambda *a,**k:pytest.fail('Retraining'))
    monkeypatch.setattr(module,'verify_frozen',lambda:pytest.fail('Repeated final suite access'))
    with pytest.raises(ValueError,match='already been used'):module.run(STUDY)
    with pytest.raises(FileExistsError,match='already started'):module.final(STUDY)

def test_demo_presets_wait_detour_and_real_failure_without_early_cutoff():
    from robot_navigation.phase3_demo import NAMES,scenario
    from robot_navigation.phase3_evaluation import record
    from robot_navigation.partial_baselines import OnlinePlanner,AlwaysWait
    for name in NAMES:
        case=scenario(name);assert shortest(case['rows'],case['start'],case['goal'])
    waiting=record(scenario(NAMES[2]),OnlinePlanner());assert waiting['metrics']['outcome']=='success' and waiting['metrics']['wait_count']==6
    failure=record(scenario(NAMES[0],'Static'),AlwaysWait());assert len(failure['transitions'])==200
    assert failure['transitions'][-1]['truncated'] and failure['metrics']['outcome']=='timeout'
    assert failure['metrics']['return']==pytest.approx(-2.)
    demos=read_json(STUDY/'demo-checks/manifest.json')['rows']
    assert any(r['outcome']=='timeout' for r in demos) and all(any(r['algorithm']==a and r['outcome']=='success' for r in demos) for a in ('q-learning','sarsa','dqn','ppo'))

def test_final_matched_panel_metrics_replays_and_export_integrity():
    from robot_navigation.phase3_ui import saved,SCOPES
    from robot_navigation.partial_evaluation import episode_metrics
    from robot_navigation.phase3_evaluation import behavior,summarize
    result=saved(SCOPES[0]);folder=Path(result['folder']);manifest=result['manifest']
    assert manifest['panel_case_count']==64 and manifest['case_count']==640 and manifest['unique_executed_episodes']==640
    assert manifest['final_evaluation_run_count']==1 and manifest['inference_only']
    typed=[];seen_failure=seen_wait=0
    for row in result['rows']:
        with gzip.open(folder/row['replay'],'rt',encoding='utf-8') as f:r=json.load(f)
        m=episode_metrics(r);assert m==r['metrics'] and behavior(r)==r['behavior']
        assert digest(r['scenario'])==row['case_hash'] and r['evaluation_seed']==0
        assert len(r['transitions'])==int(row['steps'])<=200 and int(row['wait_count'])==m['wait_count']
        assert float(row['reward'])==pytest.approx(sum(t['reward'] for t in r['transitions']))
        if row['outcome']=='timeout':assert len(r['transitions'])==200 and r['transitions'][-1]['truncated'];seen_failure+=1
        if row['outcome']=='success' and int(row['wait_count'])>0:seen_wait+=1
        if row['posthoc_efficiency']:assert 0<float(row['posthoc_efficiency'])<=1
    assert seen_failure and seen_wait
    for label in {r['policy'] for r in result['rows']}:
        part=[r for r in result['rows'] if r['policy']==label];assert len(part)==64
        assert [r['case_hash'] for r in part]==manifest['case_hashes']
    for row in manifest['summary']:
        assert row['Successes']+row['Collisions']+row['Timeouts']==row['Cases']
        if not row['Successes']:assert row['Mean successful steps'] is None and row['Successful Wait %'] is None
    for name,meta in read_json(folder/'export-inventory.json')['files'].items():assert sha(folder/name)==meta['sha256']
    with zipfile.ZipFile(folder/'results-and-replays.zip') as z:
        assert z.testzip() is None and len([n for n in z.namelist() if n.endswith('.json.gz')])==640
    for f in read_json(folder/'failure-traces.json')['failures']:assert len(f['last8_public_decisions'])==8

def test_saved_viewing_and_passive_playback_never_execute_policies(monkeypatch):
    from robot_navigation.phase3_ui import saved,SCOPES
    from robot_navigation.app_service import read_replay,frame
    monkeypatch.setattr(PartialDynamicEnv,'step',lambda *a,**k:pytest.fail('Playback/evidence view stepped environment'))
    for scope in SCOPES:assert saved(scope)
    r=saved(SCOPES[0]);record=read_replay(Path(r['folder'])/r['rows'][0]['replay'])
    for n in (0,1,7,len(record['transitions']),0):assert frame(record,n)['step']==n

def test_streamlit_main_pages_saved_holdout_downloads_and_failure_callback(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import robot_navigation.app_service as service
    monkeypatch.setattr(service,'start_job',lambda *a,**k:pytest.fail('UI autotraining'))
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run();assert not app.exception
    assert len(app.sidebar.radio[0].options)==5
    assert 'Temporary bridge - Wait' in app.selectbox[0].options
    app.sidebar.radio[0].set_value('Train Models').run();assert not app.exception
    assert any(s.label=='Learning representation' for s in app.selectbox)
    app.sidebar.radio[0].set_value('Guide').run();assert not app.exception
    app.sidebar.radio[0].set_value('Compare & Export').run();assert not app.exception
    assert next(s for s in app.selectbox if s.label=='Evaluation scope').value=='Phase 3 product held-out'
    assert not any(b.label=='Run fair comparison' for b in app.button)
    assert len(app.get('download_button'))>=5
    monkeypatch.setattr(PartialDynamicEnv,'step',lambda *a,**k:pytest.fail('Saved UI replay made decisions'))
    next(b for b in app.button if b.label=='Open recorded journey in Simulator').click().run();assert not app.exception
    assert app.session_state['workspace_page']=='Robot Simulator' and app.session_state['journey']['metrics']['outcome']!='success'
    next(b for b in app.button if b.label=='Step').click().run();assert not app.exception and app.session_state['frame_index']==1
