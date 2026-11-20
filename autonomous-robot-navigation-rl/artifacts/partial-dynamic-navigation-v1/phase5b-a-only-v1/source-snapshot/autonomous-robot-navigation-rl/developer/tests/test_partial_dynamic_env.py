from copy import deepcopy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from robot_navigation.partial_contract import configuration
from robot_navigation.partial_dynamic_env import PartialDynamicEnv, make_scenario
from robot_navigation.partial_fixtures import corridor


def test_gym_wait_reopening_and_scan_order():
    case = corridor(clear=3)
    env = PartialDynamicEnv([case]); check_env(env, skip_render_check=True)
    first, info = env.reset(seed=7)
    assert env.action_space.n == 5 and set(info) == {'outcome', 'steps'}
    assert first[8+53] == 1
    for t in (1, 2, 3):
        obs, reward, terminal, cutoff, _ = env.step(4)
        assert reward == -.01 and not terminal and not cutoff
        assert env.agent_memory.visit_count.sum() == 1
        assert obs[408] == pytest.approx(t/200)
    assert obs[8+53] == 0 and env.agent_memory.scan_step == 3
    for _ in range(7):
        obs, reward, terminal, cutoff, info = env.step(3)
    assert terminal and not cutoff and info['outcome'] == 'success' and reward == 9.99


@pytest.mark.parametrize('action,goal,expected', [(3, [5, 2], 'success'), (0, [5, 8], 'collision'), (4, [5, 8], 'timeout')])
def test_terminal_priority_and_truncation(action, goal, expected):
    c = {**corridor(), 'goal': goal, 'events': []}
    env = PartialDynamicEnv([c], {**configuration(), 'max_steps': 1})
    env.reset(seed=3)
    obs, reward, terminal, cutoff, info = env.step(action)
    assert info['outcome'] == expected
    assert terminal == (expected != 'timeout') and cutoff == (expected == 'timeout')
    if expected == 'collision':
        np.testing.assert_allclose(obs[:2], [5/9, 1/9]); assert reward == -5.01
    with pytest.raises(RuntimeError):
        env.step(4)


def test_activation_on_robot_is_suppressed_each_tick_without_extending_window():
    case = corridor(); case['events'] = [{'id': 'safe', 'cell': [5, 2], 'start': 1, 'clear': 4}]
    env = PartialDynamicEnv([case]); env.reset()
    env.step(3)
    snap = env.diagnostic_snapshot()
    assert snap['event_log'][-1] == {'step': 1, 'nominal_active': ['safe'], 'applied': [], 'suppressed': ['safe']}
    assert not snap['occupancy'][5][2]
    env.step(3)
    assert env.diagnostic_snapshot()['occupancy'][5][2]
    env.step(4); env.step(4)
    assert env.diagnostic_snapshot()['event_log'][-1]['nominal_active'] == []
    assert not env.diagnostic_snapshot()['occupancy'][5][2]


def test_action_execution_uses_observed_time_and_truncation_gets_next_events():
    case = corridor(); case['events'] = [{'id': 'new', 'cell': [5, 3], 'start': 1, 'clear': 4}]
    env = PartialDynamicEnv([case], {**configuration(), 'max_steps': 1})
    before, _ = env.reset(); assert before[8+53] == 0
    after, _, terminal, cutoff, _ = env.step(3)
    assert not terminal and cutoff and after[8+53] == 1
    # Arriving on an activation cell is safe; changing events cannot pre-empt the move.
    case['events'][0]['cell'] = [5, 2]
    env = PartialDynamicEnv([case]); env.reset(); env.step(3)
    assert env.diagnostic_snapshot()['event_log'][-1]['suppressed'] == ['new']


def test_identical_actions_reproduce_events_and_memory_and_reset_clears():
    envs = [PartialDynamicEnv([corridor()]) for _ in range(2)]
    initial = [e.reset(seed=43)[0] for e in envs]
    assert np.array_equal(*initial)
    for a in (4, 4, 4, 4, 3, 3):
        results = [e.step(a) for e in envs]
        assert np.array_equal(results[0][0], results[1][0]) and results[0][1:] == results[1][1:]
        assert envs[0].diagnostic_snapshot() == envs[1].diagnostic_snapshot()
    assert envs[0].agent_memory.visit_count.sum() == 3
    envs[0].reset(seed=43)
    assert envs[0].agent_memory.visit_count.sum() == 1
    assert envs[0].agent_memory.scan_step == 0
    for action in (-1, 5, 1.5, '4', True):
        with pytest.raises(ValueError):
            envs[0].step(action)


def test_real_out_of_range_activation_and_clearance_leave_stale_memory_until_rescanned():
    case = make_scenario(['.'*10]*10, [5, 1], [9, 9],
        [{'id': 'hidden-change', 'cell': [5, 3], 'start': 3, 'clear': 6}])
    env = PartialDynamicEnv([case]); first, _ = env.reset()
    assert first[8+53] == 0
    for a in (0, 0, 4): env.step(a)
    assert env.diagnostic_snapshot()['occupancy'][5][3]
    assert env.agent_memory.occupancy[5, 3] == 0 and env.agent_memory.last_seen_step[5, 3] == 0
    env.step(1); env.step(1)
    assert env.agent_memory.occupancy[5, 3] == 1
    env.step(0)
    assert not env.diagnostic_snapshot()['occupancy'][5][3] and env.agent_memory.occupancy[5, 3] == 1
    env.step(1)
    assert env.agent_memory.occupancy[5, 3] == 0


def test_event_and_geometry_validation():
    bad = deepcopy(corridor()); bad['events'][0]['cell'] = [0, 0]
    with pytest.raises(ValueError): PartialDynamicEnv([bad])
    bad = deepcopy(corridor()); bad['events'][0]['cell'] = bad['goal']
    with pytest.raises(ValueError): PartialDynamicEnv([bad])
    bad = deepcopy(corridor()); bad['events'].append({**bad['events'][0], 'id': 'overlap'})
    with pytest.raises(ValueError, match='Overlapping'): PartialDynamicEnv([bad])
    # Bounds are known, but the action is still executed unchanged and collides.
    edge = make_scenario(['.'*10]*10, [0, 0], [9, 9])
    for action in (0, 2):
        env = PartialDynamicEnv([edge]); env.reset()
        obs, reward, terminal, cutoff, info = env.step(action)
        assert obs[:2].tolist() == [0., 0.] and reward == -5.01
        assert terminal and not cutoff and info['outcome'] == 'collision'
