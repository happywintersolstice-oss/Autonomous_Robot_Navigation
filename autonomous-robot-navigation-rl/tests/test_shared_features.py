"""Shared feature parity, normalized TD math and actual four-family training."""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest
import torch

from test_unseen import benchmark
from robot_navigation.shared_agents import LinearMapPolicy
from robot_navigation.shared_features import FEATURE_COUNT, FEATURE_VERSION, encode_features
from robot_navigation.shared_neural import SharedMapExtractor
from robot_navigation.unseen_agents import load_map_policy
from robot_navigation.unseen_benchmark import map_observation
from robot_navigation.unseen_main import run_settings, summarize
from robot_navigation.unseen_study import train_run


def test_numpy_and_torch_features_match_full_map_and_goal(benchmark):
    observations = []
    for layout in benchmark.layouts('train')+benchmark.layouts('validation'):
        grid = benchmark.task(layout).grid
        walls = np.array([cell=='#' for row in layout['rows'] for cell in row])
        for r in range(10):
            for c in range(10):
                observations.append(map_observation([r,c,9-r,9-c],grid))
                features = encode_features(observations[-1])
                assert features.shape==(FEATURE_COUNT,) and np.isfinite(features).all()
                assert np.isclose(features[512:873]@features[512:873],walls.sum()/21)
                assert np.isclose(features[873:]@features[873:],1)
                assert features[7:368].sum()==1 and features[368:512].sum()==1
    extractor = SharedMapExtractor(gym.spaces.Box(0,1,(104,),dtype=np.float32))
    actual = extractor(torch.tensor(np.array(observations))).detach().numpy()
    np.testing.assert_allclose(actual,np.array([encode_features(o) for o in observations]),atol=1e-7)
    grid = benchmark.task(benchmark.layouts('train')[0]).grid
    terminal = map_observation([0,0,0,0],grid)
    assert encode_features(terminal)[7+180]==1
    assert encode_features(terminal)[3]==0  # Up meets the outside boundary.
    assert encode_features(terminal)[5]==0  # Left meets the outside boundary.


@pytest.mark.parametrize('algorithm',['q-learning','sarsa'])
def test_normalized_td_transfer_and_roundtrip(benchmark,tmp_path,algorithm):
    layout,scenario = benchmark.cases('train')[0]
    observation = map_observation([*scenario['start'],*scenario['goal']],benchmark.task(layout).grid)
    x = encode_features(observation)
    policy = LinearMapPolicy(benchmark,algorithm,'fixture')
    policy.update(observation,1,2,observation,True,.5,.9)
    np.testing.assert_allclose(policy.weights[1],x/(x@x))
    assert np.isclose(policy.scores(observation)[1],1)
    unseen,case = benchmark.cases('validation')[0]
    other = map_observation([*case['start'],*case['goal']],benchmark.task(unseen).grid)
    assert policy.scores(other)[1]>0
    before = policy.weights.copy()
    expected_target = 2+.9*(1 if algorithm=='q-learning' else 0)
    policy.update(observation,0,2,observation,False,.5,.9,next_action=2)
    np.testing.assert_allclose(policy.weights[0],before[0]+.5*expected_target*x/(x@x))
    if algorithm=='sarsa':
        with pytest.raises(ValueError,match='next behavior action'):
            policy.update(observation,0,1,observation,False,.1,.9)
    path = tmp_path/'linear.npz'
    policy.save(path,{})
    restored = load_map_policy(path,benchmark)
    assert restored.feature_version==FEATURE_VERSION
    np.testing.assert_array_equal(restored.scores(other),policy.scores(other))


@pytest.mark.parametrize('algorithm',['q-learning','sarsa','dqn','ppo'])
def test_shared_training_has_exact_budgets_and_auditable_traces(benchmark,tmp_path,algorithm,monkeypatch):
    from robot_navigation.unseen_audit import audit_run
    from robot_navigation.core import read_json
    original = benchmark._read
    def development_only(split):
        assert split!='test'
        return original(split)
    monkeypatch.setattr(benchmark,'_read',development_only)
    settings = deepcopy(benchmark.config['pilot'])
    settings.update(representation=FEATURE_VERSION,run_id=f'shared-{algorithm}')
    study = tmp_path/'shared-study'
    run = train_run(benchmark,study,algorithm,'fixture',settings=settings)
    directory = study/'runs'/run['run_id']
    system = {'algorithm':algorithm,'run_id':run['run_id'],
              'validation':read_json(directory/'validation.json'),
              'familiar':read_json(directory/'familiar.json')}
    result = audit_run(benchmark,study,system,settings,'fixture')
    assert run['training_interactions']==16 and result['training_traces']==1
    assert run['optimized_parameter_count']>0
    restored = load_map_policy(directory/run['viewer_checkpoint'],benchmark)
    assert restored.feature_version==FEATURE_VERSION
    if algorithm in ('dqn','ppo'):
        assert restored.model.observation_space.shape==(104,)


def test_registered_settings_are_copied_and_retry_names_are_unique():
    config = {'pilot_interactions':25000,'interactions':200000,'tabular':{'learning_rate':.1}}
    pilot = run_settings(config,'pilot','sarsa',7)
    main = run_settings(config,'main','sarsa',9,2)
    assert pilot['trace_windows']==[0,12500,23750]
    assert main['trace_windows']==[0,100000,190000]
    assert main['run_id'].endswith('-attempt2')
    pilot['tabular']['learning_rate']=1
    assert config['tabular']['learning_rate']==.1


def test_summary_keeps_poor_seeds_and_undefined_successful_lengths(tmp_path):
    from robot_navigation.core import write_json
    from robot_navigation.unseen_main import summarize
    study = tmp_path/'studies/unseen-map-main-v1'
    record = {'settings':{'algorithms':['q-learning'],'pilot_seed':7,'seeds':[11,13]},
              'attempts':[],'signature':'fixture'}
    rates = iter([.25,0.,1.])
    for phase in ('pilot','main'):
        for seed in ([7] if phase=='pilot' else [11,13]):
            run_id = f'{phase}-{seed}'
            rate = next(rates)
            counts = {'success':int(rate*4),'collision':4-int(rate*4),'timeout':0}
            evaluation = {'episodes':4,'counts':counts,'success_rate':rate,
                'collision_rate':1-rate,'timeout_rate':0.,'spl':rate,'mean_return':rate,
                'mean_successful_route_length':2. if rate else None,
                'by_map':{'map':{'success_rate':rate,'collision_rate':1-rate,'timeout_rate':0.,'spl':rate}}}
            run = {'run_id':run_id,'algorithm':'q-learning','training_seed':seed,
                'training_interactions':16,'selected_interactions':8,'training_seconds':1.,
                'total_seconds':2.,'checkpoint_bytes':10,'optimized_parameter_count':4936,
                'mean_decision_microseconds':20.}
            record['attempts'].append({'phase':phase,'algorithm':'q-learning','seed':seed,
                                      'run_id':run_id,'status':'complete','budget':16})
            directory = study/'runs'/run_id
            write_json(directory/'manifest.json',run)
            write_json(directory/'validation.json',evaluation)
            write_json(directory/'familiar.json',evaluation)
    write_json(study.parent/'unseen-map-pilot-v1/summary.json',{'references':{}})
    summary = summarize(study,record)
    stats = summary['systems'][0]['validation']['statistics']
    assert stats['success_rate']['mean']==.5
    assert np.isclose(stats['success_rate']['sd'],np.sqrt(.5))
    assert stats['mean_successful_route_length']=={'mean':2.,'sd':None,'defined_seeds':1}
    assert summary['systems'][0]['validation']['episodes']==8
    assert summary['total_training_interactions']==48
