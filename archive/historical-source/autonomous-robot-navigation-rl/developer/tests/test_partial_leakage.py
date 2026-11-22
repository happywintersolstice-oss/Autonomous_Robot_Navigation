import numpy as np
import pytest

from robot_navigation.partial_dynamic_env import PartialDynamicEnv, make_scenario


def compare_histories(a, b, actions):
    envs = [PartialDynamicEnv([s]) for s in (a, b)]
    observations = [e.reset(seed=19) for e in envs]
    assert np.array_equal(observations[0][0], observations[1][0])
    assert observations[0][1] == observations[1][1]
    for action in actions:
        values = [e.step(action) for e in envs]
        assert np.array_equal(values[0][0], values[1][0])
        assert values[0][1:] == values[1][1:]
        assert envs[0].agent_memory.snapshot() == envs[1].agent_memory.snapshot()
    return envs


def test_hidden_topology_future_schedule_and_out_of_range_events_are_indistinguishable():
    rows = ['.'*10 for _ in range(10)]
    a = make_scenario(rows, [5, 5], [1, 1])
    altered = rows.copy(); altered[9] = '.'*9+'#'
    b = make_scenario(altered, [5, 5], [1, 1], scenario_id='different hidden map')
    compare_histories(a, b, [4, 0, 0, 4])
    b = make_scenario(rows, [5, 5], [1, 1], [{'id': 'hidden', 'cell': [9, 9], 'start': 0, 'clear': 5}])
    compare_histories(a, b, [4, 0, 0, 4])
    b['events'][0].update(start=100, clear=108)
    compare_histories(a, b, [4, 0, 0, 4])


def test_temporary_vs_permanent_class_is_not_observed_and_only_reopening_distinguishes():
    rows = ['.'*10 for _ in range(10)]
    changed = rows.copy(); changed[5] = '......#...'
    a = make_scenario(changed, [5, 5], [1, 1])
    b = make_scenario(rows, [5, 5], [1, 1], [{'id': 'temporary', 'cell': [5, 6], 'start': 0, 'clear': 3}])
    envs = compare_histories(a, b, [4, 4])
    values = [e.step(4)[0] for e in envs]
    assert not np.array_equal(*values)
    assert values[0][8+56] == 1 and values[1][8+56] == 0


def test_old_full_map_encoders_are_not_used(monkeypatch):
    import robot_navigation.unseen_benchmark as old
    import robot_navigation.shared_features as features
    def forbidden(*args, **kwargs):
        pytest.fail('New observations must never invoke a historical full-map encoder')
    monkeypatch.setattr(old, 'map_observation', forbidden)
    monkeypatch.setattr(features, 'encode_features', forbidden)
    env = PartialDynamicEnv([make_scenario(['.'*10]*10, [5, 5], [1, 1])])
    assert env.reset()[0].shape == (409,)
    assert set(env.step(4)[-1]) == {'outcome', 'steps'}


def test_every_policy_facing_feature_and_reference_uses_only_identical_history():
    from robot_navigation.partial_features import encode_features
    from robot_navigation.partial_agents import LinearPolicy
    from robot_navigation.partial_baselines import OnlinePlanner
    from robot_navigation.partial_contract import configuration
    rows = ['.'*10]*10
    changed = list(rows); changed[9] = '.'*9+'#'
    envs = compare_histories(make_scenario(rows, [5, 5], [1, 1]),
        make_scenario(changed, [5, 5], [1, 1]), [4])
    observations = [env.step(4)[0] for env in envs]
    assert np.array_equal(encode_features(observations[0]), encode_features(observations[1]))
    for family in ('q-learning', 'sarsa'):
        policy = LinearPolicy(family, configuration())
        policy.weights[:] = np.random.default_rng(7).normal(size=policy.weights.shape)
        assert np.array_equal(policy.scores(observations[0]), policy.scores(observations[1]))
    planner = OnlinePlanner()
    assert vars(planner) == {} and planner.act(observations[0]) == planner.act(observations[1])
