"""F tests: ZERO new synthetic physical transitions or optimizer operations."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from robot_navigation.core import ROOT,read_json
from robot_navigation.partial_phase5f_io import Budget,Journal,read_journal,training_cases,location,inputs
from robot_navigation.partial_phase5f import settings,checked,verify
from robot_navigation.partial_training_instrumentation import observe,check_compatibility
from robot_navigation.partial_instrumentation_trace import finite_tree,numeric,array
from robot_navigation.partial_phase5f_verify import require

pytestmark=pytest.mark.filterwarnings('ignore:Unknown pytest.mark.pilot')


class FakeJournal:
    def __init__(self):self.rows=[]
    def append(self,event,**fields):self.rows.append({'event':event,**fields})


@pytest.mark.parametrize('algorithm',['dqn','ppo'])
def test_exact_500_guard_without_environment(algorithm):
    j=FakeJournal();b=Budget(j)
    b.start('dqn')
    if algorithm=='ppo':
        for _ in range(500):b.action('dqn')
        b.start('ppo')
    for i in range(500):assert b.action(algorithm)==i+1
    before=len(j.rows)
    with pytest.raises(RuntimeError):b.action(algorithm)
    assert len(j.rows)==before and b.counts[algorithm]==500


def test_order_retry_and_optimizer_caps():
    b=Budget(FakeJournal())
    with pytest.raises(RuntimeError):b.start('ppo')
    b.start('dqn')
    for i in range(100):assert b.optimizer('dqn')==i+1
    with pytest.raises(RuntimeError):b.optimizer('dqn')
    with pytest.raises(RuntimeError):b.start('dqn')
    b.start('ppo')
    for i in range(50):b.optimizer('ppo')
    with pytest.raises(RuntimeError):b.optimizer('ppo')
    with pytest.raises(RuntimeError):b.start('q-learning')


@pytest.mark.parametrize('seconds,expected',[(900,'Per-arm'),(1800,'Combined')])
def test_wall_guards(seconds,expected):
    now=[0.];b=Budget(FakeJournal(),clock=lambda:now[0]);b.start('dqn');now[0]=seconds
    with pytest.raises(TimeoutError,match=expected):b.action('dqn')


def test_journal_chain_attempt_preservation_and_quota(tmp_path):
    p=tmp_path/'ledger';j=Journal(p,activity='synthetic_control_no_physical_operations');j.append('reservation',count=1);j.close()
    assert read_journal(p)[0]['count']==1
    with pytest.raises(FileExistsError):Journal(p)
    original=p.read_bytes();p.write_bytes(original.replace(b'"count":1',b'"count":2'))
    with pytest.raises(ValueError):read_journal(p)
    q=Journal(tmp_path/'quota',byte_cap=1)
    with pytest.raises(RuntimeError):q.append('before_operation')
    q.close()


def test_split_only_decode_training_array():
    cases=training_cases();assert len(cases)==8
    assert all(c['fixture_split']=='train' and c['fixture_category']=='A' and not c['events'] for c in cases)
    import inspect
    text=inspect.getsource(training_cases)
    assert 'raw_decode' in text and "['training_cases']" not in text
    assert 'cases(' not in text.replace('training_cases(', '')


def test_config_and_unsupported_contract_fail_closed(tmp_path):
    c=settings();assert c['new_synthetic_transition_cap']==c['new_synthetic_optimizer_cap']==0
    (tmp_path/'configs').mkdir();text=(ROOT/'configs/partial_phase5f_v1.yaml').read_text()
    (tmp_path/'configs/partial_phase5f_v1.yaml').write_text(text.replace('23121','23122'))
    with pytest.raises(ValueError):settings(tmp_path)
    with pytest.raises(RuntimeError,match='UNSUPPORTED'):check_compatibility({'versions':{},'methods':{}})


def test_disabled_by_default_no_hooks_and_no_model_calls():
    before=(sys.gettrace(),sys.getprofile())
    with observe(object()):assert (sys.gettrace(),sys.getprofile())==before
    assert (sys.gettrace(),sys.getprofile())==before


def test_unsupported_active_device_without_constructing_model():
    fake=SimpleNamespace(device=SimpleNamespace(type='cuda'))
    policy=SimpleNamespace(model=fake)
    with pytest.raises(RuntimeError,match='UNSUPPORTED'):
        with observe(policy,sink=SimpleNamespace(emit=lambda *a,**k:None)):pass
    assert sys.gettrace() is None


def test_finite_payload_and_information_boundary():
    with pytest.raises(ValueError):numeric([np.nan])
    with pytest.raises(ValueError):finite_tree({'hidden_map':[[1]]})
    n=numeric(np.array([1,2],dtype=np.float32));assert array(n).dtype==np.float32
    n['data'][0]=5
    with pytest.raises(ValueError):array(n)
    live=(ROOT/'src/robot_navigation/partial_phase5f_live.py').read_text()
    assert '.snapshot(' not in live and 'load_scaled_policy' not in live
    assert 'model.learn(total_timesteps=500,reset_num_timesteps=False)' in live
    assert 'case_id' in live and 'outcome' in live


def test_verifier_has_no_research_execution_calls():
    for name in ('partial_phase5f_verify.py','partial_phase5f_report.py'):
        tree=ast.parse((ROOT/'src/robot_navigation'/name).read_text())
        forbidden={'learn','predict','step','build_policy','PartialDynamicEnv','load_scaled_policy'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                assert name not in forbidden


@pytest.fixture(scope='module')
def pilot_result():return verify(require_post=False)


@pytest.mark.pilot
def test_active_integration_actual_500_and_optimizer_ledger(pilot_result):
    assert pilot_result['scientific_training_actions']==1000 and pilot_result['actual_optimizer_steps']==150
    assert [pilot_result['arms'][a]['training_actions'] for a in ('dqn','ppo')]==[500,500]


@pytest.mark.pilot
def test_dqn_reward_links_and_ppo_boundaries(pilot_result):
    d=pilot_result['arms']['dqn']['signals'];p=pilot_result['arms']['ppo']['signals']
    assert d['all_sample_draws']==5000 and p['all_sample_draws']==2500
    assert p['rollouts']==5 and p['Q_values']=='not_applicable'
    assert not d['required_missing_fields'] and not p['required_missing_fields']
    for s in (d,p):
        if s['goal_sample_draws']:assert s['reward_to_update_example']['transition_id']>0
        else:assert s['reward_to_update_example']['status']=='not_observed'


@pytest.mark.pilot
def test_readonly_verification_split_provenance_no_new_operations(pilot_result):
    assert pilot_result['research_evaluation_actions']==pilot_result['new_synthetic_transitions']==pilot_result['new_synthetic_optimizer_steps']==0
    assert pilot_result['preservation']['status']=='passed'
    assert pilot_result['verdict']=='INSTRUMENTED_LEARNING_TRACES_VALIDATED'
