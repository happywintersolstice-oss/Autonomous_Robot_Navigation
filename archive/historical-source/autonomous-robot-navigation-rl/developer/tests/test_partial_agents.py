from copy import deepcopy

import numpy as np
import pytest

from robot_navigation.core import read_json, write_json
from robot_navigation.partial_agents import LinearPolicy, load_policy
from robot_navigation.partial_contract import configuration
from robot_navigation.partial_dynamic_env import PartialDynamicEnv, make_scenario
from robot_navigation.partial_features import encode_features, FEATURE_COUNT
from robot_navigation.partial_training import TrainingRecorder, train_linear, validate_training_trace


def fixture(cap=200, adjacent=False):
    config = {**configuration(), 'max_steps': cap}
    case = make_scenario(['.'*10]*10, [5, 5], [5, 6] if adjacent else [9, 9])
    return config, case


@pytest.mark.parametrize('family,next_action,target', [('q-learning', None, 2.8), ('sarsa', 0, 1.)])
def test_hand_calculated_five_action_update_and_terminal_bootstrap(family, next_action, target):
    config, case = fixture(); obs, _ = PartialDynamicEnv([case], config).reset()
    phi = encode_features(obs)
    assert phi.shape == (FEATURE_COUNT,) == (412,)
    policy = LinearPolicy(family, config)
    policy.weights[4] = 2*phi/(phi@phi)
    policy.update(obs, 2, 1., obs, False, .5, .9, next_action)
    assert policy.scores(obs)[2] == pytest.approx(.5*target)
    policy.weights[2].fill(0)
    policy.update(obs, 2, 1., obs, True, .5, .9)
    assert policy.scores(obs)[2] == pytest.approx(.5)
    if family == 'sarsa':
        with pytest.raises(ValueError, match='next behavior'): policy.update(obs, 2, 1., obs, False, .5, .9)


def test_sarsa_carries_preupdate_behavior_action_and_bootstraps_external_cap():
    config, case = fixture(cap=3)
    env = TrainingRecorder(PartialDynamicEnv([case], config), [case])
    class Spy(LinearPolicy):
        def __init__(self):
            super().__init__('sarsa', config); self.sequence = iter([3, 4, 2, 1]); self.continuations = []
        def explore(self, *args): return next(self.sequence)
        def update(self, *args):
            self.continuations.append((args[4], args[-1])); return super().update(*args)
    policy = Spy(); train_linear(policy, env, 3, 7, config['smoke'])
    assert [r['action'] for r in env.records if r['kind'] == 'step'] == [3, 4, 2]
    assert policy.continuations == [(False, 4), (False, 2), (False, 1)]
    assert validate_training_trace(env.records, config) == 3
    assert env.records[-1]['truncated'] and not env.records[-1]['terminated']
    # A truncation bootstrap action is discarded before the next reset episode.
    config, case = fixture(cap=1)
    env = TrainingRecorder(PartialDynamicEnv([case], config), [case])
    policy = Spy(); train_linear(policy, env, 2, 7, config['smoke'])
    assert [r['action'] for r in env.records if r['kind'] == 'step'] == [3, 2]
    assert policy.continuations == [(False, 4), (False, 1)]


def test_linear_checkpoint_rejects_old_shapes_versions_and_checksum(tmp_path):
    config, case = fixture(); obs, _ = PartialDynamicEnv([case], config).reset()
    policy = LinearPolicy('q-learning', config, 9)
    train_linear(policy, TrainingRecorder(PartialDynamicEnv([case], config), [case]), 32, 9, config['smoke'])
    path = tmp_path/'policy.npz'; policy.save(path, {'training_interactions': 32})
    loaded = load_policy(path, config)
    assert loaded.weights.shape == (5, 412) and np.array_equal(policy.scores(obs), loaded.scores(obs))
    metadata = read_json(path.with_suffix('.json'))
    for key, value in [('feature_version', 'egocentric-map-goal-v1'), ('actions', 4), ('observation_values', 104)]:
        write_json(path.with_suffix('.json'), {**metadata, key: value})
        with pytest.raises(ValueError, match='mismatch'): load_policy(path, config)
    write_json(path.with_suffix('.json'), metadata)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checksum'): load_policy(path, config)
    from robot_navigation.core import ROOT
    with pytest.raises(ValueError, match='mismatch'):
        load_policy(ROOT/'artifacts/runs/m4-tuned-q-learning-seed101/selected.npz', config)


def test_shared_neural_features_match_linear_without_hidden_map(monkeypatch):
    torch = pytest.importorskip('torch')
    from robot_navigation.partial_neural import PartialVisibleExtractor
    config, case = fixture(); env = PartialDynamicEnv([case], config); obs, _ = env.reset()
    extractor = PartialVisibleExtractor(env.observation_space)
    actual = extractor(torch.tensor(np.stack([obs, obs]))).detach().numpy()
    np.testing.assert_allclose(actual[0], encode_features(obs))
    assert np.array_equal(actual[0], actual[1])
    import robot_navigation.shared_neural as old
    monkeypatch.setattr(old.SharedMapExtractor, 'forward', lambda *args: pytest.fail('Historical extractor invoked'))
    assert extractor(torch.tensor(obs[None])).shape == (1, 412)


@pytest.mark.parametrize('family', ['dqn', 'ppo'])
def test_neural_tiny_updates_five_actions_checkpoint_and_recording(tmp_path, family):
    pytest.importorskip('stable_baselines3')
    from robot_navigation.partial_neural import build_neural
    config, case = fixture()
    recorder = TrainingRecorder(PartialDynamicEnv([case], config), [case])
    policy = build_neural(recorder, family, config['smoke'], 7)
    before = policy.state_fingerprint(); policy.model.learn(total_timesteps=64)
    assert policy.model.num_timesteps == recorder.interactions == 64
    assert all(set(r['info']) == {'outcome', 'steps'} for r in recorder.records if r['kind'] == 'step')
    assert validate_training_trace(recorder.records, config) == 64
    assert policy.state_fingerprint() != before
    obs, _ = PartialDynamicEnv([case], config).reset(seed=3)
    assert type(policy.act(obs)) is int and policy.act(obs) in range(5)
    path = tmp_path/f'{family}.zip'; policy.save(path, {'training_interactions': 64})
    loaded = load_policy(path, config)
    assert loaded.state_fingerprint() == policy.state_fingerprint() and loaded.act(obs) == policy.act(obs)
    with pytest.raises(ValueError): loaded.act(np.zeros(104, dtype=np.float32))


@pytest.mark.parametrize('adjacent,done,timeout', [(False, 0., 1.), (True, 1., 0.)])
def test_sb3_dqn_stores_terminal_observation_and_bootstraps_truncation(adjacent, done, timeout):
    pytest.importorskip('stable_baselines3')
    from robot_navigation.partial_neural import build_neural
    config, case = fixture(cap=1, adjacent=adjacent)
    policy = build_neural(PartialDynamicEnv([case], config), 'dqn', {**config['smoke'], 'train_freq': 1}, 7)
    policy.model.action_space.sample = lambda: 3  # Public warmup sampling hook, fixed RIGHT.
    policy.model.learn(total_timesteps=1)
    buffer = policy.model.replay_buffer
    assert buffer.timeouts[0, 0] == timeout and buffer.sample(1).dones.item() == done
    assert buffer.next_observations[0, 0, 1] == pytest.approx(6/9)
    assert buffer.next_observations[0, 0, 408] == pytest.approx(1/200)


@pytest.mark.parametrize('adjacent,expected', [(False, 1.97), (True, 9.99)])
def test_ppo_bootstrap_uses_final_partial_observation_only_on_truncation(adjacent, expected):
    torch = pytest.importorskip('torch')
    from robot_navigation.partial_neural import build_neural
    config, case = fixture(cap=1, adjacent=adjacent)
    policy = build_neural(PartialDynamicEnv([case], config), 'ppo', config['smoke'], 7)
    policy.model.policy.forward = lambda obs, deterministic=False: (
        torch.full((len(obs),), 3, dtype=torch.long), torch.zeros((len(obs), 1)), torch.zeros(len(obs)))
    observed = []
    def values(obs):
        observed.extend(obs.detach().cpu().numpy())
        return torch.full((len(obs), 1), 2.)
    policy.model.policy.predict_values = values
    policy.model.learn(total_timesteps=32)
    assert np.allclose(policy.model.rollout_buffer.rewards, expected)
    if not adjacent:
        assert any(o[408] == pytest.approx(.005) and o[1] == pytest.approx(6/9) for o in observed)
