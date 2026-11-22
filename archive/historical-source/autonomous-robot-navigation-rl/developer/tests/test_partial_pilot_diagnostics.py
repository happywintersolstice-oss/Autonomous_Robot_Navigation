import random

import numpy as np
import pytest

from robot_navigation.partial_baselines import AlwaysWait
from robot_navigation.partial_contract import configuration
from robot_navigation.partial_evaluation import record_episode
from robot_navigation.partial_feasibility_fixtures import cases, suite, ScriptedMechanics
from robot_navigation.partial_pilot_diagnostics import (evaluation_summary, failure_examples,
    preserve_random_state, representation_probe, reward_sanity, training_summary)


def test_reward_math_is_not_a_causal_conclusion():
    values = reward_sanity(.99)
    assert values['wait_200']['undiscounted_return'] == pytest.approx(-2)
    assert values['wait_200']['discounted_return'] == pytest.approx(-.01*(1-.99**200)/(1-.99))
    assert values['collision_immediately']['undiscounted_return'] == -5.01
    assert values['success_in_1_actions']['discounted_return'] == 9.99
    assert values['success_in_7_actions']['undiscounted_return'] == pytest.approx(9.93)


def test_representation_retains_information_but_affine_interactions_have_limits():
    result = representation_probe()
    assert result['feature_count'] == 412
    assert result['immediate_and_far_right_obstacle_distinguishable']
    assert (result['right_ray_no_obstacle'], result['right_ray_distance_one'], result['right_ray_distance_two']) == (0., .5, 1.)
    assert result['independent_goal_occupancy_parallelogram_error'] < 1e-7
    assert 'synthetic' in result['interpretation']


def test_checkpoint_evaluation_restores_python_numpy_and_torch_rng():
    torch = pytest.importorskip('torch')
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    py, numpy, t = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with preserve_random_state():
        random.seed(99); np.random.seed(99); torch.manual_seed(99)
        random.random(); np.random.random(); torch.rand(5)
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), t)


@pytest.mark.parametrize('algorithm', ['dqn', 'ppo'])
def test_actual_saved_model_reload_does_not_reseed_ongoing_training(algorithm):
    import torch
    from robot_navigation.core import ROOT
    from robot_navigation.partial_agents import load_policy
    path = ROOT/f'artifacts/partial-dynamic-navigation-v1/branch-validation-v1/smoke/{algorithm}/final.zip'
    random.seed(83); np.random.seed(83); torch.manual_seed(83)
    py, np_state, t = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with preserve_random_state():
        policy = load_policy(path, configuration())
        obs, _ = __import__('robot_navigation.partial_dynamic_env', fromlist=['PartialDynamicEnv']).PartialDynamicEnv(cases('A')).reset(seed=0)
        policy.act(obs)
    assert py == random.getstate() and np.array_equal(np_state[1], np.random.get_state()[1])
    assert torch.equal(t, torch.get_rng_state())


def test_report_renderer_handles_failure_and_unreached_categories(tmp_path):
    from robot_navigation.core import write_json
    from robot_navigation.partial_feasibility import location
    from robot_navigation.partial_feasibility_report import report
    folder = location(tmp_path); folder.mkdir(parents=True); (tmp_path/'docs').mkdir()
    stats = evaluation_summary([record_episode(cases('A')[0], AlwaysWait())])
    write_json(folder/'phase5a_pilot_protocol.json', {'git': {'head': 'fixture', 'branch': 'fixture'}})
    write_json(folder/'summary.json', {'verdict': 'REQUIRES_REPRESENTATION_OR_TRAINING_DIAGNOSIS',
        'budget': {'total_training_reserved': 0, 'evaluation_reserved': 200, 'elapsed_seconds': 1.},
        'systems': {'q-learning': {'training_actions': 0, 'basic_gate': False, 'status': 'completed',
            'halt': 'fixture', 'checkpoints': [{'phase': 'untrained', 'training_actions': 0, 'evaluation': {'A': stats}}]}},
        'controls': {'always-wait': {'A': stats}}})
    result = report(tmp_path)
    assert result['rows'] == 1
    text = (tmp_path/'docs/partial_feasibility_results.md').read_text(encoding='utf-8')
    assert 'missing category' in text and '644 features' in text
    assert (folder/'A-learning-diagnostics.png').exists()


def test_wait_distribution_and_zero_recovery_denominator():
    record = record_episode(cases('A')[0], AlwaysWait())
    result = evaluation_summary([record])
    assert result['counts'] == {'success': 0, 'collision': 0, 'timeout': 1}
    assert result['action_counts'] == {'Up': 0, 'Down': 0, 'Left': 0, 'Right': 0, 'Wait': 200}
    assert result['wait_fraction'] == 1 and result['longest_wait_streak'] == 200
    assert result['recovery_rate'] is None
    examples = failure_examples(record)
    assert examples['first_sustained_wait']['action_index'] == 1
    assert len(examples['first_sustained_wait']['observation_before_action']) == 409


def test_actual_training_and_incomplete_prefix_denominators_are_distinct():
    entry = next(e for e in suite() if e['case']['fixture_category'] == 'A')
    record = record_episode(entry['case'], ScriptedMechanics(entry['script']))
    trace = [{'kind': 'reset', 'phase': 'A', 'scenario': record['scenario'], 'snapshot': record['initial_snapshot']}]
    trace += [{'kind': 'step', **v} for v in record['transitions']]
    result = training_summary(trace)
    assert result['overall']['completed_episodes'] == 1 and result['overall']['counts']['success'] == 1
    prefix = record_episode(cases('A')[0], AlwaysWait())
    trace = [{'kind': 'reset', 'phase': 'A', 'scenario': prefix['scenario'], 'snapshot': prefix['initial_snapshot']}]
    trace += [{'kind': 'step', **v} for v in prefix['transitions'][:3]]
    result = training_summary(trace)
    assert result['overall']['completed_episodes'] == 0 and result['overall']['incomplete_prefixes'] == 1
    assert result['overall']['rates']['success'] is None
    assert result['overall']['actions'] == 3


def test_composition_distinguishes_event_impacts_cases_and_actual_exposure(tmp_path, monkeypatch):
    import copy
    import robot_navigation.partial_feasibility as pilot
    from robot_navigation.core import ROOT
    from robot_navigation.partial_pilot_ledger import Budget
    case = copy.deepcopy(cases('C')[0]); case['split'] = 'train'
    case['difficulty']['single_event_detour_lengths'] = [None]
    class Reader:
        def scenarios(self, split, dynamic): return [case] if split == 'train' else []
    monkeypatch.setattr(pilot, 'DevelopmentBenchmark', lambda root: Reader())
    result = pilot.composition(tmp_path/'composition', ROOT, Budget(pilot.settings()))
    counts = result['totals']
    assert result['reference_cases'] == 1 and counts['nominal_events'] == 1
    assert counts['single_cell_disconnecting_impacts'] == 1
    assert counts['temporarily_disconnected_during_relevant_window'] == 1
    assert counts['detour_available_during_relevant_window'] == 0
    assert counts['observed_relevant_at_executed_decision'] == 1
    assert counts['recoveries'] == counts['recovery_opportunities'] == 1
    assert counts['suppressed_activation_ticks'] == 0
