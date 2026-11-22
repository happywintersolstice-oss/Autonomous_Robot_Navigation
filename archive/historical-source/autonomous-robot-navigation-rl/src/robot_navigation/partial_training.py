"""Bounded functional training utilities, with true actions recorded before autoreset."""
from copy import deepcopy
from time import perf_counter

import gymnasium as gym
import numpy as np

from .partial_dynamic_env import PartialDynamicEnv


class TrainingRecorder(gym.Wrapper):
    """Trusted recording channel, separate from observation/info passed to SB3."""
    def __init__(self, env, scenarios):
        super().__init__(env)
        self.cases = {c['id']: deepcopy(c) for c in scenarios}
        self.records, self.interactions, self.episode = [], 0, -1

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous = observation.copy(); self.episode += 1
        snapshot = self.env.diagnostic_snapshot()
        self.records.append({'kind': 'reset', 'episode': self.episode, 'seed': kwargs.get('seed'),
            'scenario': self.cases[snapshot['scenario_id']], 'observation': observation.tolist(), 'snapshot': snapshot})
        return observation, info

    def step(self, action):
        observation, reward, terminal, cutoff, info = self.env.step(action)
        self.interactions += 1
        self.records.append({'kind': 'step', 'episode': self.episode, 'observation': self.previous.tolist(),
            'action': int(action), 'next_observation': observation.tolist(), 'reward': reward,
            'terminated': terminal, 'truncated': cutoff, 'info': deepcopy(info),
            'snapshot': self.env.diagnostic_snapshot()})
        self.previous = observation.copy()
        return observation, reward, terminal, cutoff, info


def validate_training_trace(records, config):
    env, observation, interactions, episode = None, None, 0, -1
    for record in records:
        if record['kind'] == 'reset':
            episode += 1
            env = PartialDynamicEnv([record['scenario']], config)
            observation, _ = env.reset(seed=record['seed'])
            if observation.tolist() != record['observation'] or env.diagnostic_snapshot() != record['snapshot']:
                raise ValueError('Training reset record mismatch')
        elif record['kind'] == 'step' and env is not None:
            if observation.tolist() != record['observation']: raise ValueError('Training before-state mismatch')
            observation, reward, terminal, cutoff, info = env.step(record['action'])
            if (observation.tolist() != record['next_observation'] or reward != record['reward']
                    or terminal != record['terminated'] or cutoff != record['truncated']
                    or info != record['info'] or env.diagnostic_snapshot() != record['snapshot']):
                raise ValueError('Training transition record mismatch')
            interactions += 1
        else: raise ValueError('Training record requires a preceding reset')
        if record['episode'] != episode: raise ValueError('Training episode order mismatch')
    return interactions


def train_linear(policy, env, interactions, seed, config):
    """SARSA carries the sampled pre-update continuation action unchanged."""
    rng = np.random.default_rng(seed)
    def epsilon(step):
        fraction = min(1., step/(interactions*.8))
        return config['epsilon_start']+(config['epsilon_end']-config['epsilon_start'])*fraction
    observation, _ = env.reset(seed=seed)
    action = policy.explore(observation, epsilon(0), rng)
    started = perf_counter()
    for step in range(interactions):
        nxt, reward, terminal, cutoff, _ = env.step(action)
        next_action = policy.explore(nxt, epsilon(step+1), rng) if policy.algorithm == 'sarsa' and not terminal else None
        policy.update(observation, action, reward, nxt, terminal,
                      config['linear_learning_rate'], config['discount'], next_action)
        if step+1 < interactions:
            if terminal or cutoff:
                observation, _ = env.reset()
                action = policy.explore(observation, epsilon(step+1), rng)
            else:
                observation = nxt
                action = next_action if policy.algorithm == 'sarsa' else policy.explore(nxt, epsilon(step+1), rng)
    return {'interactions': interactions, 'training_seconds': perf_counter()-started}
