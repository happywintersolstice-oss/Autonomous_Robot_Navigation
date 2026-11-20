"""Read-only audit arithmetic, legal transformations and sampling gate failures."""
from copy import deepcopy
import inspect

import numpy as np
import pytest
import torch

from robot_navigation.core import read_json,write_json
from robot_navigation.partial_contract import MOVES,configuration
from robot_navigation.partial_dynamic_env import PartialDynamicEnv,make_scenario
from robot_navigation.partial_observation import SPACE
from robot_navigation.partial_features_scaled import RelativeScale4Extractor,encode_scaled,load_scaled_policy
from robot_navigation.partial_phase5c_fixtures import (HISTORY_ACTIONS,history_observation,probe_manifest,
    transform_cell,transform_observation,action_permutation,uniform_schedule,balanced_schedule)
from robot_navigation.partial_phase5c_coverage import (coverage,direction,score_record,sampling_gate,DIRECTIONS)
from robot_navigation.partial_phase5c import Evidence,settings,location,checked_plan


@pytest.mark.parametrize('length',[8,22])
@pytest.mark.parametrize('mirror',[False,True])
@pytest.mark.parametrize('turns',range(4))
def test_full_legal_history_transforms_match_scripted_environment(length,mirror,turns):
    env=PartialDynamicEnv([make_scenario(['.'*10]*10,transform_cell([4,4],turns,mirror),transform_cell([9,9],turns,mirror))])
    obs,_=env.reset(seed=0); perm=action_permutation(turns,mirror)
    for action in HISTORY_ACTIONS[:length]: obs,_,terminated,truncated,_=env.step(perm[action])
    assert not terminated and not truncated
    expected=transform_observation(history_observation(length),turns,mirror)
    assert np.array_equal(obs,expected)
    # The capacity witness action mapping, not a demand for learned symmetry.
    def scores(o):
        dr,dc=o[2:4]-o[:2]; return np.array([-dr,dr,-dc,dc,0])
    np.testing.assert_allclose(scores(expected)[perm],scores(history_observation(length)),atol=1e-7)
    assert int(np.argmax(scores(expected)))==perm[int(np.argmax(scores(history_observation(length))))]


def test_probe_manifest_parity_fixed_counts_and_no_model_or_hidden_inputs(monkeypatch):
    def forbidden(*args,**kwargs): pytest.fail('Score probe construction must not execute an action')
    monkeypatch.setattr(PartialDynamicEnv,'step',forbidden)
    a=probe_manifest(); b=probe_manifest()
    assert a==b and len(a['probes'])==40 and a['new_policy_environment_actions']==0
    observations=np.array([p['observation'] for p in a['probes']],dtype=np.float32)
    actual=RelativeScale4Extractor(SPACE)(torch.tensor(observations)).detach().numpy()
    np.testing.assert_allclose(actual,np.stack([encode_scaled(o) for o in observations]),atol=1e-7)
    reset=[p for p in a['probes'] if p['kind'].startswith('reset')]
    assert len(reset)==24 and {p['distance'] for p in reset}=={1,2}
    assert all(np.isfinite(actual).ravel())


def make_records(actions=(4,3)):
    case=make_scenario(['.'*10]*10,[5,5],[5,6])
    env=PartialDynamicEnv([case]); obs,_=env.reset(seed=23021)
    records=[{'kind':'reset','episode':0,'seed':23021,'scenario':case}]
    for a in actions:
        nxt,r,term,trunc,info=env.step(a)
        records.append({'kind':'step','episode':0,'observation':obs.tolist(),'action':a,'next_observation':nxt.tolist(),
            'reward':r,'terminated':term,'truncated':trunc,'info':info,
            'behavior_context':{'decision':{'action_values':[0,0,0,1,0]},'epsilon_used':.5}})
        obs=nxt
    return records,[case]


def test_reset_action_episode_and_near_goal_denominators_and_empty_autoreset():
    records,cases=make_records(); cv=coverage(records,cases)
    assert cv['reset_count']==1 and cv['total']['decisions']==2 and cv['total']['distinct_states']==2
    assert cv['episode_outcomes']=={'success':1}
    b=cv['near_goal']['1']['Right']
    assert b['decisions']==2 and b['waits']==1 and b['selected_reducing']==1 and b['greedy_reducing']==2
    assert b['actions']['Right']==1 and b['selected_first_argmax']==1
    assert cv['goal_rewards_by_arrival_action']['Right']==1
    records.append({**records[0],'episode':1,'seed':None})
    cv=coverage(records,cases)
    assert cv['reset_count']==2 and cv['episode_outcomes']=={'success':1,'empty_reset':1}
    assert cv['marginals']['direction']['Up']['selected_reducing_fraction'] is None


def test_incomplete_prefix_is_not_timeout_and_diagonals_are_disjoint():
    records,cases=make_records(actions=(4,)); cv=coverage(records,cases)
    assert cv['episode_outcomes']=={'incomplete_prefix':1}
    assert direction([-1,1])=='Up-Right' and direction([0,1])=='Right'
    for groups in cv['marginals'].values(): assert sum(b['decisions'] for b in groups.values())==1
    assert sum(b['decisions'] for b in cv['joint_observed'].values())==1
    empty=coverage([],cases)
    assert empty['reset_count']==0 and empty['total']['decisions']==0
    assert all(x is None for x in empty['reset_direction_shares'].values())


def test_attempted_collision_is_preserved_distance_not_a_successful_move():
    case=make_scenario(['.'*10]*10,[0,0],[0,1]); env=PartialDynamicEnv([case]); obs,_=env.reset()
    nxt,reward,term,trunc,info=env.step(0)
    records=[{'kind':'reset','episode':0,'seed':0,'scenario':case},
        {'kind':'step','episode':0,'observation':obs.tolist(),'next_observation':nxt.tolist(),'reward':reward,
         'action':0,'terminated':term,'truncated':trunc,'info':info,'behavior_context':{'decision':{'action_values':[1,0,0,0,0]}}}]
    b=coverage(records,[case])['total']
    assert b['collisions']==1 and b['successful_moves']==0
    assert b['actual_distance_change']['preserved']==1 and b['attempted_distance_change']['increased']==1


@pytest.mark.parametrize('n',[0,1,8,13,64,203])
def test_pure_seeded_samplers_uniform_matches_environment_and_balanced_blocks(n):
    expected=[]; env=PartialDynamicEnv([make_scenario(['.'*10]*10,[2,2],[2,3],scenario_id=str(i)) for i in range(8)])
    for i in range(n):
        env.reset(seed=23021 if i==0 else None)
        expected.append(int(env.diagnostic_snapshot()['scenario_id']))
    assert uniform_schedule(n,23021)==expected
    schedule=balanced_schedule(n,23041)
    assert schedule==balanced_schedule(n,23041) and len(schedule)==n
    for start in range(0,n//8*8,8): assert sorted(schedule[start:start+8])==list(range(8))
    assert list(inspect.signature(balanced_schedule).parameters)==['count','seed']
    with pytest.raises(TypeError): balanced_schedule(n,23041,outcomes=['success'])


def test_sampling_gate_cannot_pass_from_failures_alone_but_can_pass_a_real_deficit(monkeypatch):
    import robot_navigation.partial_phase5c_coverage as module
    c=settings(); records,cases=make_records(); cv=coverage(records,cases)
    cv.update(reset_count=100,reset_cases={str(i):12 for i in range(8)},
        reset_directions={'Up':25,'Down':25,'Left':25,'Right':25},
        reset_direction_shares={d:.25 for d in DIRECTIONS[:4]})
    cv['goal_rewards_by_arrival_action']={d:50 for d in DIRECTIONS[:4]}
    final={'dqn':{'coverage':cv,'expanded_direction_goals':{d:0 for d in DIRECTIONS[:4]}}}
    assert not sampling_gate(final,c)['coverage_intervention_justified']
    cv['reset_directions']['Right']=4; cv['reset_direction_shares']['Right']=.04
    cv['reset_cases']['3']=2;cv['reset_cases']['7']=2; cv['goal_rewards_by_arrival_action']['Right']=1
    def simulated(n,seed):
        return {'uniform':{'directions':{d:4 for d in DIRECTIONS[:4]},'cases':{str(i):2 for i in range(8)}},
                'balanced':{'directions':{d:25 for d in DIRECTIONS[:4]},'cases':{str(i):12 for i in range(8)}}}
    monkeypatch.setattr(module,'schedule_comparison',simulated)
    assert sampling_gate(final,c)['coverage_intervention_justified']
    cv['goal_rewards_by_arrival_action']['Right']=100
    cv['near_goal']['1']['Right']['decisions']=100
    assert not sampling_gate(final,c)['coverage_intervention_justified']


def test_score_only_probe_matches_saved_action_without_executing_policy(monkeypatch):
    from robot_navigation.core import ROOT
    from robot_navigation.partial_pilot_diagnostics import decision_values,preserve_random_state
    path=ROOT/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/training-and-evaluation-evidence/dqn/B1-relative-x4/06000/policy.zip'
    with preserve_random_state():
        policy=load_scaled_policy(path,configuration(),'B1-relative-x4')
        before=policy.state_fingerprint()
        monkeypatch.setattr(policy,'act',lambda *args:pytest.fail('No model action execution permitted'))
        o=np.array(probe_manifest()['probes'][0]['observation'],dtype=np.float32)
        quality=score_record(o,decision_values(policy,o))
        assert len(quality['scores'])==5 and quality['score_kind']=='Q'
        assert before==policy.state_fingerprint()


def test_corrupt_plan_and_evidence_boundary_block_before_analysis(tmp_path):
    reader=Evidence(tmp_path)
    with pytest.raises(ValueError,match='escaped'): reader.data(tmp_path.parent/'outside')
    folder=location(tmp_path);folder.mkdir(parents=True)
    write_json(folder/'analysis-plan.json',{})
    write_json(folder/'analysis-seal.json',{'sha256':'corrupt'})
    with pytest.raises(ValueError,match='changed'): checked_plan(tmp_path)


def test_injected_analysis_failure_preserved_and_never_passing(tmp_path,monkeypatch):
    import robot_navigation.partial_phase5c as module
    folder=location(tmp_path);folder.mkdir(parents=True)
    write_json(folder/'analysis-plan.json',{'fixture':'registered plan exists before injected failure'})
    write_json(folder/'probe-manifest.json',{'fixture':'registered probes exist before injected failure'})
    monkeypatch.setattr(module,'checked_plan',lambda root:{'settings':settings(),'baseline_tree':'fixture','source_inputs':{},'source_snapshot':{}})
    def failure(root): raise ValueError('injected historical integrity failure')
    monkeypatch.setattr(module,'preserve_history',failure)
    result=module.analyze(tmp_path)
    assert result['verdict']=='IMPLEMENTATION_BLOCKED'
    assert read_json(folder/'final-verdict.json')['new_training_actions']==0
    with pytest.raises(FileExistsError): module.analyze(tmp_path)


def test_report_csv_preserves_probe_columns_with_different_shapes(tmp_path):
    import csv
    from robot_navigation.partial_phase5c_report import csv_write
    path=tmp_path/'scores.csv'; csv_write(path,[{'probe':'reset','score':1},{'probe':'rotated','score':2,'D4':.4}])
    with path.open() as f: rows=list(csv.DictReader(f))
    assert rows[0]['D4']=='' and rows[1]['D4']=='0.4'


@pytest.mark.parametrize('family',['dqn','ppo'])
def test_saved_replay_rollout_signals_match_raw_historical_prefix(family):
    import gzip,json
    from robot_navigation.core import ROOT
    from robot_navigation.partial_phase5c import old_location,learning_signal
    reader=Evidence(ROOT)
    arm=old_location()/'training-and-evaluation-evidence'/family/'B1-relative-x4'
    cp=reader.json(arm/'02000/summary.json')
    prefix=reader.data(arm/'training-actions.jsonl.gz')[:cp['training_prefix_bytes']]
    trace=[json.loads(line) for line in gzip.decompress(prefix).decode().splitlines()]
    signal=learning_signal(reader,arm,2000,trace,cp)
    if family=='dqn': assert signal['DQN_replay']['transitions']==2000
    else: assert signal['PPO_last_rollout']['transitions']==100 and signal['PPO_last_rollout']['returns_finite']
