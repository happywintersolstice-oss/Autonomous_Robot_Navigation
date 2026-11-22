from copy import deepcopy
import hashlib

import numpy as np
import pytest

from robot_navigation.core import ROOT
from robot_navigation.partial_agents import LinearPolicy
from robot_navigation.partial_contract import configuration
from robot_navigation.partial_dynamic_env import PartialDynamicEnv, validate_scenario
from robot_navigation.partial_evaluation import record_episode, validate_replay
from robot_navigation.partial_feasibility import basic_gate, rollout, settings, EpisodeAbort, audit_episode
from robot_navigation.partial_feasibility_fixtures import cases, suite, ScriptedMechanics, hindsight_actions
from robot_navigation.partial_pilot_ledger import Budget, PilotLimit, ActionStream, PilotRecorder, ContinuousLinear
from robot_navigation.partial_training import validate_training_trace
from robot_navigation.partial_commands import read_records
from robot_navigation.partial_baselines import AlwaysWait


def tiny_budget():
    c = deepcopy(settings()); c.update(per_algorithm_actions=8, total_training_actions=32,
                                      evaluation_action_ceiling=20, suite_wall_seconds=100, per_algorithm_wall_seconds=10)
    return c


def test_fixture_counts_pair_separation_connectivity_and_direction_balance():
    assert len(cases()) == 20 and len(cases(split='train')) == 24
    for category in 'ABCDE':
        train = cases(category, 'train'); evaluation = cases(category)
        assert not {(tuple(s['start']), tuple(s['goal']), tuple(s['rows'])) for s in train}&{(tuple(s['start']), tuple(s['goal']), tuple(s['rows'])) for s in evaluation}
    for case in cases()+cases(split='train'):
        grid = validate_scenario(case)
        assert len(grid.distances(case['start'])) == len(grid.free_cells)
    directions = {tuple(np.sign(np.array(c['goal'])-c['start'])) for c in cases('A')}
    assert directions == {(-1, 0), (1, 0), (0, -1), (0, 1)}
    with pytest.raises(ValueError): cases(split='final')


@pytest.mark.parametrize('category', list('ABCDE'))
def test_all_handwritten_mechanics_and_hindsight_are_real(category):
    for entry in suite():
        if entry['case']['fixture_category'] != category: continue
        record = record_episode(entry['case'], ScriptedMechanics(entry['script']))
        validate_replay(record)
        assert record['metrics']['outcome'] == 'success'
        assert hindsight_actions(entry['case']) <= len(entry['script'])
        if category == 'C':
            first = record['initial_observation']
            assert sum(float(v == .5) for v in first[4:8]) == 4
            assert record['metrics']['wait_count'] > 0 and record['metrics']['recoveries'] == 1


def test_budget_counts_all_stages_and_stops_before_overshoot_and_retries():
    c = tiny_budget(); b = Budget(c)
    b.start_algorithm('q-learning')
    with pytest.raises(PilotLimit, match='retry'): b.start_algorithm('q-learning')
    for _ in range(8): b.training_action('q-learning')
    with pytest.raises(PilotLimit): b.training_action('q-learning')
    assert b.training['q-learning'] == 8
    for _ in range(20): b.evaluation_action()
    with pytest.raises(PilotLimit): b.evaluation_action()
    assert b.evaluation_claims == 20
    for a in ('sarsa', 'dqn', 'ppo'):
        for _ in range(8): b.training_action(a)
    assert sum(b.training.values()) == 32


def test_wall_deadlines_are_monotonic_and_abort_before_an_action():
    now = [0.]; c = tiny_budget(); b = Budget(c, clock=lambda: now[0]); b.start_algorithm('sarsa')
    now[0] = 10.
    with pytest.raises(PilotLimit, match='per-algorithm'): b.training_action('sarsa')
    assert b.training['sarsa'] == 0
    now[0] = 101.
    with pytest.raises(PilotLimit, match='Global'): b.evaluation_action()


def test_bounded_rollout_matches_old_evaluator_without_action_replacement():
    case = cases('A')[0]; c = settings(); b = Budget(c)
    actual = rollout(case, AlwaysWait(), configuration(), b)
    expected = record_episode(case, AlwaysWait())
    validate_replay(actual)
    assert actual['metrics'] == expected['metrics']
    assert [v['action'] for v in actual['transitions']] == [4]*200
    assert b.evaluation_claims == 200


def test_aborted_evaluation_preserves_incomplete_real_prefix_without_fake_flags():
    c = tiny_budget(); c['evaluation_action_ceiling'] = 3
    with pytest.raises(EpisodeAbort) as failure:
        rollout(cases('A')[0], AlwaysWait(), configuration(), Budget(c))
    record = failure.value.record
    assert record['completion'] == 'incomplete' and record['metrics'] is None
    assert len(record['transitions']) == 3
    assert not record['transitions'][-1]['terminated'] and not record['transitions'][-1]['truncated']
    assert audit_episode(record) == 3


@pytest.mark.parametrize('algorithm', ['q-learning', 'sarsa'])
def test_continuous_linear_chunks_append_only_and_replay_exact(tmp_path, algorithm):
    c = tiny_budget(); b = Budget(c); b.start_algorithm(algorithm)
    stream = ActionStream(tmp_path/'actions.jsonl.gz')
    env = PilotRecorder(cases('A', 'train'), configuration(), b, algorithm, stream, 'A', 23001)
    policy = LinearPolicy(algorithm, configuration(), 23001)
    trainer = ContinuousLinear(policy, b, c); trainer.set_env(env)
    stream.begin(); trainer.train(4); stream.close()
    prefix = stream.path.read_bytes(); first_hash = hashlib.sha256(prefix).hexdigest()
    stream.begin(); trainer.train(4); stream.close()
    assert hashlib.sha256(stream.path.read_bytes()[:len(prefix)]).hexdigest() == first_hash
    assert validate_training_trace(read_records(stream.path), configuration()) == 8
    assert trainer.updates == b.training[algorithm] == 8


def test_basic_gate_requires_improvement_over_tie_favored_baseline():
    c = settings(); baseline = {'counts': {'success': 1}}
    assert not basic_gate({'counts': {'success': 1}}, baseline, c)
    assert basic_gate({'counts': {'success': 2}}, baseline, c)
    assert not basic_gate({'counts': {'success': 2}}, {'counts': {'success': 2}}, c)


def test_fixed_features_ties_and_exploration_cover_all_five_actions():
    policy = LinearPolicy('q-learning', configuration(), 23001)
    obs, _ = PartialDynamicEnv(cases('A', 'train')).reset(seed=23001)
    assert policy.act(obs) == 0
    for epsilon in (0., 1.):
        rng = np.random.default_rng(23023)
        sampled = [policy.explore(obs, epsilon, rng) for _ in range(500)]
        assert set(sampled) == set(range(5))
        assert all(50 < sampled.count(a) < 150 for a in range(5))


def test_reopened_fixture_scripts_really_lose_visibility_before_revision():
    for entry in suite():
        if entry['case']['fixture_category'] != 'E' or entry['case']['fixture_split'] != 'evaluation': continue
        record = record_episode(entry['case'], ScriptedMechanics(entry['script']))
        cell = tuple(entry['case']['events'][0]['cell'])
        snapshots = [record['initial_snapshot']]+[t['snapshot'] for t in record['transitions']]
        assert any(s['memory']['occupancy'][cell[0]][cell[1]] != int(s['occupancy'][cell[0]][cell[1]])
                   and not s['memory']['currently_seen'][cell[0]][cell[1]] for s in snapshots)
        assert any(s['scan_update']['occupancy_corrections'] > 0 for s in snapshots)


def test_hindsight_detour_numbers_are_privileged_diagnostics_only():
    assert sorted(hindsight_actions(case) for case in cases('D')) == [9, 9, 11, 11]
    entry = next(e for e in suite() if e['case']['fixture_category'] == 'D' and e['case']['fixture_split'] == 'evaluation')
    record = rollout(entry['case'], ScriptedMechanics(entry['script']), configuration(), Budget(settings()))
    assert record['hindsight']['success_action_excess'] == 0
    assert record['hindsight']['clairvoyant_minimum_actions'] == len(entry['script'])


def test_bad_rollout_granularity_and_oversized_envelope_rejected_before_training(tmp_path):
    import yaml
    import shutil
    (tmp_path/'configs').mkdir()
    shutil.copy2(ROOT/'configs/partial_dynamic.yaml', tmp_path/'configs/partial_dynamic.yaml')
    c = settings(); c['model']['n_steps'] = 128
    (tmp_path/'configs/partial_feasibility_v1.yaml').write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='divide'): settings(tmp_path)
    c = settings(); c['per_algorithm_actions'] = 10001
    (tmp_path/'configs/partial_feasibility_v1.yaml').write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='envelope changed'): settings(tmp_path)


def test_new_evaluator_does_not_expose_remote_future_events_or_fixture_metadata():
    a = deepcopy(cases('A')[0]); b = deepcopy(a)
    b['id'] = 'different-hidden-id'; b['events'] = [{'id': 'future', 'cell': [9, 9], 'start': 10, 'clear': 14}]
    c = tiny_budget(); c['evaluation_action_ceiling'] = 3
    records = []
    for case in (a, b):
        with pytest.raises(EpisodeAbort) as failure: rollout(case, AlwaysWait(), configuration(), Budget(c))
        records.append(failure.value.record)
    for first, second in zip(records[0]['transitions'], records[1]['transitions']):
        assert first['observation'] == second['observation'] and first['next_observation'] == second['next_observation']
        assert first['reward'] == second['reward'] and set(first['info']) == {'outcome', 'steps'}
