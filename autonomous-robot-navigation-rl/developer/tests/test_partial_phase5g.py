"""Phase 5G offline tests: hard cap ZERO new physical transitions/updates."""
import ast,copy
from pathlib import Path
import numpy as np
import pytest

from robot_navigation.core import ROOT
from robot_navigation.partial_phase5g_io import Budget,AssessmentBudget,Journal,read_journal,cases,direction,CHECKPOINTS
from robot_navigation.partial_phase5g import settings,storage_plan,verify
from robot_navigation.partial_phase5g_assess import first_metrics,reducing_actions
from robot_navigation.partial_training_instrumentation import check_compatibility,observe
from robot_navigation.partial_instrumentation_trace import read_trace
from robot_navigation.partial_instrumentation_audit import validate_dqn,validate_ppo,validate_gae

class FakeJournal:
    def __init__(self):self.rows=[]
    def append(self,event,**fields):self.rows.append({'event':event,**fields})


def test_exact_training_limits_and_no_retry():
    j=FakeJournal();b=Budget(j)
    with pytest.raises(RuntimeError):b.start('ppo')
    b.start('dqn')
    with pytest.raises(RuntimeError):b.start('ppo')
    for i in range(2000):assert b.action('dqn')==i+1
    with pytest.raises(RuntimeError):b.action('dqn')
    for i in range(475):assert b.optimizer('dqn')==i+1
    with pytest.raises(RuntimeError):b.optimizer('dqn')
    with pytest.raises(RuntimeError):b.start('dqn')
    b.start('ppo')
    for i in range(2000):b.action('ppo')
    for i in range(200):b.optimizer('ppo')
    before=len(j.rows)
    with pytest.raises(RuntimeError):b.action('ppo')
    with pytest.raises(RuntimeError):b.optimizer('ppo')
    assert len(j.rows)==before and sum(b.counts.values())==4000


@pytest.mark.parametrize('seconds,match',[(1800,'Arm'),(3600,'Combined')])
def test_training_wall_caps(seconds,match):
    now=[0.];b=Budget(FakeJournal(),clock=lambda:now[0]);b.start('dqn');now[0]=seconds
    with pytest.raises(TimeoutError,match=match):b.action('dqn')


def test_assessment_32000_and_200_limits_without_environment():
    b=AssessmentBudget(FakeJournal())
    for case in range(160):
        b.start_case()
        for i in range(200):b.reserve(case_id=str(case))
        with pytest.raises(RuntimeError):b.reserve()
    b.start_case()
    with pytest.raises(RuntimeError):b.reserve()
    assert b.count==32000


def test_hash_chain_and_exclusive_failure_prefix(tmp_path):
    p=tmp_path/'prefix';j=Journal(p,activity='offline_fake_control');j.append('failure',operations=0);j.close()
    original=p.read_bytes()
    with pytest.raises(FileExistsError):Journal(p)
    assert p.read_bytes()==original
    p.write_bytes(original.replace(b'"operations":0',b'"operations":1'))
    with pytest.raises(ValueError):read_journal(p)


def test_run_lock_blocks_before_any_model_import(tmp_path,monkeypatch):
    import robot_navigation.partial_phase5g as g
    monkeypatch.setattr(g,'branch_gate',lambda *a:None);monkeypatch.setattr(g,'checked',lambda *a:(tmp_path,{},{}))
    (tmp_path/'run.lock').write_text('failed attempt preserved')
    with pytest.raises(FileExistsError):g.run()
    assert (tmp_path/'run.lock').read_text()=='failed attempt preserved'


def test_frozen_scope_and_unsupported_library(tmp_path):
    c=settings();assert c['synthetic_transition_cap']==c['synthetic_optimizer_cap']==0
    (tmp_path/'configs').mkdir();p=tmp_path/'configs/partial_phase5g_v1.yaml'
    p.write_text((ROOT/'configs/partial_phase5g_v1.yaml').read_text().replace('23121','23122'))
    with pytest.raises(ValueError):settings(tmp_path)
    with pytest.raises(RuntimeError,match='UNSUPPORTED'):check_compatibility({'versions':{},'methods':{}})
    import sys
    before=sys.gettrace(),sys.getprofile()
    with observe(object()):assert (sys.gettrace(),sys.getprofile())==before


def test_exact_existing_panel_ids_and_direction_balance():
    p=cases();assert [len(p[k]) for k in ('training_cases','original_cases','expanded_cases')]==[8,4,16]
    assert all('-A-train-' in c['id'] for c in p['training_cases'])
    assert all('-A-evaluation-' in c['id'] for c in p['original_cases'])
    for d in ('Up','Down','Left','Right'):assert sum(direction(c)==d for c in p['expanded_cases'])==4
    assert len({c['id'] for v in p.values() for c in v})==28


@pytest.mark.parametrize('algorithm',['dqn','ppo'])
def test_primary_metric_margins_ties_wait_invalid(algorithm):
    case={'start':[0,1],'goal':[0,2]};key='action_values' if algorithm=='dqn' else 'action_probabilities'
    scores={key:[.1,.1,.1,.6,.1]};r=first_metrics(case,3,scores,algorithm)
    assert r['aligned'] and r['reducing_actions']==[3] and r['right_margin']==.5
    assert reducing_actions(case['start'],case['goal'])==[3]
    ties=first_metrics(case,0,{key:[.2]*5},algorithm)
    assert ties['tie'] and ties['invalid_move'] and not ties['aligned']
    wait=first_metrics(case,4,{key:[.1,.1,.1,.1,.6]},algorithm)
    assert wait['wait'] and not wait['aligned']


def test_storage_envelope_from_existing_traces_offline():
    plan=storage_plan();assert plan['aggregate_padded_estimate']<536870912
    assert all(a['G_uncompressed_padded_estimate']<536870912 for a in plan['arms'].values())


@pytest.fixture(scope='module')
def old_traces():
    base=ROOT/'artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1'
    return {a:read_trace(base/a/'trace') for a in ('dqn','ppo')}


def test_offline_actual_F_arithmetic_and_corruption_rejection(old_traces):
    d=next(r for r in old_traces['dqn'] if r['event']=='dqn_loss');p=next(r for r in old_traces['ppo'] if r['event']=='ppo_loss')
    gi=next(r for r in old_traces['ppo'] if r['event']=='gae_input');gr=next(r for r in old_traces['ppo'] if r['event']=='gae_result')
    validate_dqn(d);validate_ppo(p);validate_gae(gi,gr)
    bad=copy.deepcopy(d);bad['actual_target']['data'][0][0]+=1
    with pytest.raises(ValueError):validate_dqn(bad)


def test_source_boundaries_and_exact_update_checkpoint_hook():
    live=(ROOT/'src/robot_navigation/partial_phase5g_live.py').read_text()
    assert live.count('m.learn(total_timesteps=2000,reset_num_timesteps=False)')==1
    assert live.index('result=old_train(*args,**kwargs)')<live.index("if m.num_timesteps in CHECKPOINTS:snapshot")
    assert "exclude=['train']" in live and all(cp%100==0 and cp%4==0 for cp in CHECKPOINTS)
    for name in ('partial_phase5g_verify.py','partial_phase5g_report.py'):
        tree=ast.parse((ROOT/'src/robot_navigation'/name).read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                func=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
                assert func not in {'learn','predict','step','build_policy','PartialDynamicEnv','load_scaled_policy','decision_values'}
    assess=(ROOT/'src/robot_navigation/partial_phase5g_assess.py').read_text()
    assert '.snapshot(' not in assess and 'diagnostic_snapshot(' not in assess and 'deterministic' in assess


@pytest.fixture(scope='module')
def actual_G():return verify(require_post=False)


@pytest.mark.pilot
def test_actual_G_budget_live_math_checkpoint_isolation(actual_G):
    assert actual_G['scientific_training_actions']==4000 and actual_G['optimizer_steps']==675
    assert actual_G['assessment_actions']<=32000 and actual_G['assessment']['assessment_cases']==160
    for a in ('dqn','ppo'):
        t=actual_G['training'][a];assert t['training_actions']==2000 and set(t['checkpoints'])=={'0','500','1000','2000'}
        assert not t['signals']['required_missing_fields']


@pytest.mark.pilot
def test_actual_G_panels_primary_metrics_and_frozen_readiness(actual_G):
    for a in ('dqn','ppo'):
        arm=actual_G['assessment']['arms'][a]
        for cp,node in arm['checkpoints'].items():
            assert node['original']['cases']==4 and node['expanded']['cases']==16 and node['combined']['alignment_denominator']==20
            assert len(node['case_metrics'])==20 and all(s['cases']==4 for s in node['expanded_by_direction'].values())
        assert arm['screen']['verdict'] in ('BASIC_NAVIGATION_SCREEN_MET','BASIC_NAVIGATION_SCREEN_NOT_MET')


@pytest.mark.pilot
def test_actual_G_preservation_zero_synthetic_and_readonly(actual_G):
    assert actual_G['verdict']=='PHASE5G_MEASUREMENT_VALIDATED' and actual_G['preservation']['status']=='passed'
    assert actual_G['new_synthetic_transitions']==actual_G['new_synthetic_optimizer_steps']==0
    assert not actual_G['required_missing_fields']
