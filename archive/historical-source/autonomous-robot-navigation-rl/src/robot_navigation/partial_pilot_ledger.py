"""Finite counters, streaming true-action records and a continuous linear learner."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np

from .partial_contract import ACTION_NAMES
from .partial_dynamic_env import PartialDynamicEnv


class PilotLimit(RuntimeError):
    pass


class Budget:
    def __init__(self, settings, folder=None, clock=perf_counter):
        self.settings, self.clock, self.started = settings, clock, clock()
        self.training = dict.fromkeys(settings['algorithms'], 0)
        self.evaluation_claims = 0
        self.algorithm_started = {}
        self.journal = (Path(folder)/'budget-reservations.jsonl').open('x', encoding='utf-8', buffering=1) if folder else None

    def start_algorithm(self, algorithm):
        if algorithm in self.algorithm_started: raise PilotLimit('An algorithm cannot receive a free retry')
        self.algorithm_started[algorithm] = self.clock()

    def check(self, algorithm=None):
        now = self.clock()
        if now-self.started >= self.settings['suite_wall_seconds']: raise PilotLimit('Global wall-clock ceiling')
        if (algorithm in self.algorithm_started
                and now-self.algorithm_started[algorithm] >= self.settings['per_algorithm_wall_seconds']):
            raise PilotLimit(f'{algorithm}: per-algorithm wall-clock ceiling')

    def _log(self, kind, algorithm, count):
        if self.journal:
            self.journal.write(json.dumps({'kind': kind, 'algorithm': algorithm, 'count': count,
                                          'elapsed_seconds': self.clock()-self.started})+'\n')

    def training_action(self, algorithm):
        self.check(algorithm)
        if (self.training[algorithm] >= self.settings['per_algorithm_actions']
                or sum(self.training.values()) >= self.settings['total_training_actions']):
            raise PilotLimit('Training interaction ceiling, including all stages/retries')
        self.training[algorithm] += 1
        self._log('training_reserved', algorithm, self.training[algorithm])

    def evaluation_action(self, algorithm=None):
        self.check(algorithm)
        if self.evaluation_claims >= self.settings['evaluation_action_ceiling']:
            raise PilotLimit('Evaluation/control/composition interaction ceiling')
        self.evaluation_claims += 1
        self._log('evaluation_reserved', algorithm, self.evaluation_claims)

    def summary(self):
        return {'training_reserved': dict(self.training), 'total_training_reserved': sum(self.training.values()),
                'evaluation_reserved': self.evaluation_claims, 'elapsed_seconds': self.clock()-self.started}

    def close(self):
        if self.journal: self.journal.close()


class ActionStream:
    """Append completed gzip members at checkpoints; old prefix bytes never change."""
    def __init__(self, path):
        self.path, self.episode, self.file = Path(path), -1, None

    def begin(self):
        if self.file is not None: raise RuntimeError('Training member is already open')
        self.file = gzip.open(self.path, 'at' if self.path.exists() else 'xt', encoding='utf-8')

    def write(self, record):
        if self.file is None: raise RuntimeError('Training member must be open')
        self.file.write(json.dumps(record, allow_nan=False)+'\n')

    def close(self):
        if self.file is not None: self.file.close(); self.file = None


class PilotRecorder(gym.Wrapper):
    """Trusted output only. The model receives original observations/public info."""
    def __init__(self, cases, config, budget, algorithm, stream, phase, reset_seed):
        super().__init__(PartialDynamicEnv(cases, config))
        self.cases = {c['id']: deepcopy(c) for c in cases}
        self.budget, self.algorithm, self.stream, self.phase = budget, algorithm, stream, phase
        self.reset_seed, self.first_reset = reset_seed, True
        self.context = lambda: {}

    def reset(self, **kwargs):
        if self.first_reset and kwargs.get('seed') is None: kwargs['seed'] = self.reset_seed
        self.first_reset = False
        observation, info = self.env.reset(**kwargs)
        self.previous = observation.copy(); self.stream.episode += 1
        snapshot = self.env.diagnostic_snapshot()
        self.stream.write({'kind': 'reset', 'episode': self.stream.episode, 'seed': kwargs.get('seed'),
            'phase': self.phase, 'scenario': self.cases[snapshot['scenario_id']],
            'observation': observation.tolist(), 'snapshot': snapshot,
            'reset_semantics': 'New episode; a preceding nonterminal prefix remains incomplete, never a fabricated timeout'})
        return observation, info

    def step(self, action):
        self.budget.training_action(self.algorithm)  # Reserve durably before execution.
        context = self.context()
        observation, reward, terminal, cutoff, info = self.env.step(action)
        self.stream.write({'kind': 'step', 'episode': self.stream.episode, 'phase': self.phase,
            'global_training_interaction': self.budget.training[self.algorithm], 'observation': self.previous.tolist(),
            'action': int(action), 'next_observation': observation.tolist(), 'reward': reward,
            'terminated': terminal, 'truncated': cutoff, 'info': deepcopy(info),
            'snapshot': self.env.diagnostic_snapshot(), 'behavior_context': context})
        self.previous = observation.copy()
        return observation, reward, terminal, cutoff, info


def epsilon_at(step, settings):
    model = settings['model']
    progress = min(1., step/(settings['per_algorithm_actions']*.8))
    return model['epsilon_start']+(model['epsilon_end']-model['epsilon_start'])*progress


class ContinuousLinear:
    def __init__(self, policy, budget, settings):
        self.policy, self.budget, self.settings = policy, budget, settings
        self.rng = np.random.default_rng(settings['primary_seed'])
        self.observation, self.action, self.updates, self.used_epsilon = None, None, 0, None

    def set_env(self, env):
        self.env, self.observation, self.action = env, None, None
        env.context = lambda: {'epsilon_used': self.used_epsilon,
                               'tie_rule': 'random among maxima during explore; first argmax during greedy evaluation'}

    def train(self, actions):
        model = self.settings['model']
        algorithm = self.policy.algorithm
        for _ in range(actions):
            if self.observation is None:
                self.observation, _ = self.env.reset()
                self.used_epsilon = epsilon_at(self.budget.training[algorithm], self.settings)
                self.action = self.policy.explore(self.observation, self.used_epsilon, self.rng)
            nxt, reward, terminal, cutoff, _ = self.env.step(self.action)
            next_epsilon = epsilon_at(self.budget.training[algorithm], self.settings)
            next_action = self.policy.explore(nxt, next_epsilon, self.rng) if algorithm == 'sarsa' and not terminal else None
            self.policy.update(self.observation, self.action, reward, nxt, terminal,
                               model['linear_learning_rate'], model['discount'], next_action)
            self.updates += 1
            if terminal or cutoff:
                self.observation, self.action = None, None
            else:
                self.observation = nxt; self.used_epsilon = next_epsilon
                self.action = next_action if algorithm == 'sarsa' else self.policy.explore(nxt, next_epsilon, self.rng)
