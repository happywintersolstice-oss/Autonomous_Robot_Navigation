from copy import deepcopy

import pytest

from robot_navigation.partial_baselines import AlwaysWait, OnlinePlanner, RandomPolicy
from robot_navigation.partial_contract import configuration
from robot_navigation.partial_dynamic_env import make_scenario
from robot_navigation.partial_evaluation import aggregate, record_episode, validate_replay
from robot_navigation.partial_fixtures import corridor, detour, long_detour


def test_same_information_planner_waits_for_only_corridor_and_detours():
    case = corridor(clear=6)
    record = record_episode(case, OnlinePlanner())
    assert validate_replay(record) is record
    metrics = record['metrics']
    assert metrics['outcome'] == 'success' and metrics['wait_count'] > 0
    assert metrics['recovery_opportunities'] == 1 and metrics['recoveries'] == 1
    assert metrics['occupancy_corrections'] >= 1
    alternate = record_episode(detour(clear=20), OnlinePlanner())
    assert alternate['metrics']['outcome'] == 'success'
    assert alternate['metrics']['actions'] < 27  # Waiting until tick 20 + seven moves.
    assert alternate['metrics']['wait_count'] == 0
    permanent = detour(clear=20); permanent['events'] = []
    rows = list(permanent['rows']); rows[5] = rows[5][:3]+'#'+rows[5][4:]
    permanent = make_scenario(rows, permanent['start'], permanent['goal'])
    assert record_episode(permanent, OnlinePlanner())['metrics']['outcome'] == 'success'


def test_wait_timeout_denominators_empty_recovery_and_replay_tampering():
    case = corridor(); case['events'] = []
    record = record_episode(case, AlwaysWait(), {**configuration(), 'max_steps': 5})
    m = record['metrics']
    assert m['actions'] == m['wait_count'] == m['longest_wait_streak'] == 5
    assert m['return'] == pytest.approx(-.05) and m['moves'] == m['repeated_movement_visits'] == 0
    assert m['two_cell_oscillation_windows'] == 0 and m['nominal_event_count'] == 0
    result = aggregate([record])
    assert result['counts'] == {'success': 0, 'collision': 0, 'timeout': 1}
    assert result['recovery_rate'] is None and result['mean_observed_to_recovery_delay'] is None
    assert m['posthoc_waits_with_open_route'] == 5
    bad = deepcopy(record); bad['transitions'][0]['reward'] = 1
    with pytest.raises(ValueError, match='transition mismatch'): validate_replay(bad)
    bad = deepcopy(record); bad['metrics']['wait_count'] = 0
    with pytest.raises(ValueError, match='metrics mismatch'): validate_replay(bad)


def test_failed_routes_still_reconcile_and_final_is_rejected():
    case = corridor()
    records = [record_episode(case, p, {**configuration(), 'max_steps': 8}) for p in (AlwaysWait(), RandomPolicy(3), OnlinePlanner())]
    result = aggregate(records)
    assert result['episodes'] == sum(result['counts'].values()) == 3
    assert result['total_actions'] == sum(r['metrics']['actions'] for r in records)
    case['split'] = 'test'
    with pytest.raises(ValueError, match='Only development'): record_episode(case, AlwaysWait())


def test_movement_revisits_are_distinct_from_wait_and_useful_return():
    class Script:
        policy_id = algorithm = 'manual-script'
        def __init__(self): self.actions = iter([3, 2, 3, 2, 4, 3, 3, 3, 3, 3, 3, 3])
        def act(self, observation): return next(self.actions)
    case = corridor(); case['events'] = []
    record = record_episode(case, Script())
    assert record['metrics']['wait_count'] == 1
    assert record['metrics']['repeated_movement_visits'] == 4
    assert record['metrics']['two_cell_oscillation_windows'] == 2
    assert record['metrics']['moves'] == 11 and record['metrics']['actions'] == 12


def test_short_block_with_long_detour_has_real_wait_vs_detour_tradeoff():
    class Script:
        policy_id = algorithm = 'manual-wait-then-right'
        def __init__(self): self.actions = iter([4]*3+[3]*7)
        def act(self, observation): return next(self.actions)
    case = long_detour(clear=3)
    waited = record_episode(case, Script())
    rerouted = record_episode(case, OnlinePlanner())
    assert waited['metrics']['outcome'] == rerouted['metrics']['outcome'] == 'success'
    assert waited['metrics']['actions'] == 10 and waited['metrics']['actions'] < rerouted['metrics']['actions']
