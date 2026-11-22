"""Map separation, shared information, tabular transfer and actual pilot wiring."""
from copy import deepcopy
import shutil

import numpy as np
import pytest
import yaml

from robot_navigation.core import ROOT, GridMap, NavigationEnv, read_json
from robot_navigation.unseen_agents import MapTablePolicy, load_map_policy
from robot_navigation.unseen_benchmark import (MapBenchmark, MapNavigationEnv, audit_benchmark,
                                              configuration, geometry_hash, map_observation, prepare)
from robot_navigation.unseen_study import evaluate, train_run


@pytest.fixture
def benchmark(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'data/maps').mkdir(parents=True)
    shutil.copy2(ROOT/'configs/environment.yaml',tmp_path/'configs/environment.yaml')
    shutil.copy2(ROOT/'data/maps/bottleneck-v1.json',tmp_path/'data/maps/bottleneck-v1.json')
    config = configuration()
    config['layout_counts'] = {'train':2,'validation':1,'test':1}
    config['scenarios_per_band'] = 2
    config['monitoring_per_band'] = 1
    config['pilot'].update(interactions=16, monitor_every=8, trace_windows=[0])
    config['pilot']['dqn'].update(learning_starts=0, buffer_size=32, batch_size=4, net_arch=[8])
    config['pilot']['ppo'].update(n_steps=4,batch_size=4,n_epochs=1,net_arch=[8])
    (tmp_path/'configs/unseen_maps.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    prepare(tmp_path)
    return MapBenchmark(tmp_path)


def test_layout_generation_is_disjoint_symmetric_and_reproducible(benchmark):
    result = audit_benchmark(benchmark.root)
    assert result['splits']['validation'] == {'layouts':1,'scenarios':6}
    assert result['splits']['test'] == {'layouts':1,'scenarios':6}
    assert prepare(benchmark.root) == result
    layout = benchmark.layouts('train')[0]
    rows = np.array([list(row) for row in layout['rows']])
    transformed = GridMap('another-id',tuple(''.join(r) for r in np.rot90(rows)))
    assert geometry_hash(transformed) == layout['geometry_hash']
    with pytest.raises(ValueError,match='sealed'):
        benchmark.layouts('test')


def test_map_environment_uses_the_core_transition_and_full_information(benchmark):
    from gymnasium.utils.env_checker import check_env
    cases = benchmark.cases('train')
    env = MapNavigationEnv(benchmark,cases)
    check_env(env,skip_render_check=True)
    observation, _ = env.reset(seed=79)
    layout, scenario = env.layout, env.env.scenario
    task = benchmark.task(layout)
    original = NavigationEnv(task,[scenario])
    raw, _ = original.reset(seed=79,options={'scenario':scenario})
    assert observation.shape == (104,) and env.observation_space.contains(observation)
    assert np.array_equal(observation,map_observation(raw,task.grid))
    actual = env.step(0)
    expected = original.step(0)
    assert np.array_equal(actual[0],map_observation(expected[0],task.grid))
    assert actual[1:] == expected[1:]
    again, _ = MapNavigationEnv(benchmark,cases).reset(seed=79)
    assert np.array_equal(again,observation)


@pytest.mark.parametrize('algorithm',['q-learning','sarsa'])
def test_map_tables_delegate_updates_and_leave_unseen_states_unchanged(benchmark,tmp_path,algorithm):
    policy = MapTablePolicy(benchmark,algorithm,f'{algorithm}/fixture')
    layout, scenario = benchmark.cases('train')[0]
    task = benchmark.task(layout)
    observation = map_observation([*scenario['start'],*scenario['goal']],task.grid)
    policy.update(observation,1,2,observation,True,.5,.9,next_action=None)
    assert policy.scores(observation)[1] == 1
    unseen_layout, unseen_scenario = benchmark.cases('validation')[0]
    unseen = map_observation([*unseen_scenario['start'],*unseen_scenario['goal']],benchmark.task(unseen_layout).grid)
    before = policy.state_fingerprint()
    assert policy.act(unseen) == 0 and np.array_equal(policy.scores(unseen),np.zeros(4))
    assert policy.state_fingerprint() == before
    path = tmp_path/'model.npz'
    policy.save(path,{'training_interactions':1})
    restored = load_map_policy(path,benchmark)
    assert restored.state_fingerprint() == before
    assert np.array_equal(restored.scores(observation),policy.scores(observation))
    wrong = deepcopy(benchmark)
    wrong.signature = {**benchmark.signature,'observation_version':'coordinates-v1'}
    with pytest.raises(ValueError,match='mismatch'):
        load_map_policy(path,wrong)


def test_reference_evaluation_is_optimal_and_final_execution_is_blocked(benchmark):
    cases = benchmark.cases('validation')
    summary, rows, records = evaluate(benchmark,cases,'bfs',7)
    assert summary['success_rate'] == 1 and summary['spl'] == 1
    assert summary['counts']['success'] == 6 and len(rows) == 6
    assert all(record['layout_split']=='validation' for record in records)
    with pytest.raises(ValueError,match='sealed'):
        evaluate(benchmark,cases,'bfs',7,split='test')


def test_training_and_evaluation_reject_wrong_split_or_substituted_layout(benchmark):
    with pytest.raises(ValueError,match='declared training'):
        MapNavigationEnv(benchmark,benchmark.cases('validation'))
    layout, scenario = deepcopy(benchmark.cases('validation')[0])
    layout['geometry_hash'] = 'substituted-map'
    with pytest.raises(ValueError,match='declared development'):
        evaluate(benchmark,[(layout,scenario)],'bfs',7)


@pytest.mark.parametrize('algorithm',['q-learning','sarsa','dqn','ppo'])
def test_short_map_aware_runs_have_exact_budgets_and_reloadable_replays(benchmark,tmp_path,algorithm,monkeypatch):
    from robot_navigation.unseen_audit import load_recording
    original = benchmark._read
    def development_only(split):
        assert split != 'test', 'Training must not read final layouts'
        return original(split)
    monkeypatch.setattr(benchmark,'_read',development_only)
    study = tmp_path/'study'
    result = train_run(benchmark,study,algorithm,'fixture-source')
    assert result['status'] == 'complete' and result['training_interactions'] == 16
    assert sum(result['training']['map_interactions'].values()) == 16
    directory = study/'runs'/result['run_id']
    policy = load_map_policy(directory/result['viewer_checkpoint'],benchmark)
    task, episode = load_recording(benchmark,directory,result['replays'][0])
    from robot_navigation.unseen_agents import map_adapter
    adapter = map_adapter(policy,task.grid)
    assert adapter.act(episode['initial_observation']) == episode['transitions'][0]['action']
    assert read_json(directory/'validation.json')['episodes'] == 6
    assert [m['training_interactions'] for m in read_json(directory/'monitoring.json')] == [8,16]
    for item in result['training']['training_traces']:
        load_recording(benchmark,directory,item)
    broken = {**result['replays'][0],'path':'../outside.json'}
    with pytest.raises(ValueError,match='registered'):
        load_recording(benchmark,directory,broken)


def test_changed_map_files_are_rejected(benchmark):
    path = benchmark.directory/'validation.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='checksum'):
        benchmark.layouts('validation')


def test_older_sb3_recording_metadata_is_checked_without_changing_saved_transitions(benchmark,tmp_path):
    from robot_navigation.unseen_audit import replay_task
    result = train_run(benchmark,tmp_path/'study','q-learning','fixture-source')
    directory = tmp_path/'study/runs'/result['run_id']
    record = read_json(directory/result['training']['training_traces'][0]['path'])
    for transition in record['transitions']:
        transition['info']['TimeLimit.truncated'] = bool(transition['truncated'] and not transition['terminated'])
    original = deepcopy(record)
    replay_task(benchmark,record)
    assert record == original
    record['transitions'][0]['info']['TimeLimit.truncated'] = not record['transitions'][0]['info']['TimeLimit.truncated']
    with pytest.raises(ValueError,match='timeout metadata'):
        replay_task(benchmark,record)
