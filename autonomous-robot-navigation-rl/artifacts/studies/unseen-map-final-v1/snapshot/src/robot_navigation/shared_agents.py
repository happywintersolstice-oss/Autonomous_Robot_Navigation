"""Normalized semi-gradient Q-learning/SARSA and shared-feature SB3 models."""
import hashlib
from pathlib import Path

import numpy as np

from .core import write_json
from .shared_features import FEATURE_COUNT, FEATURE_VERSION, encode_features
from .unseen_benchmark import OBSERVATION


class LinearMapPolicy:
    feature_version = FEATURE_VERSION

    def __init__(self,benchmark,algorithm,policy_id):
        if algorithm not in ('q-learning','sarsa'):
            raise ValueError('A linear Q-learning or SARSA family is required')
        self.benchmark,self.algorithm,self.policy_id = benchmark,algorithm,policy_id
        self.weights = np.zeros((4,FEATURE_COUNT),dtype=np.float64)

    def scores(self,observation):
        return self.weights @ encode_features(observation)

    def act(self,observation):
        return int(np.argmax(self.scores(observation)))

    def explore(self,observation,epsilon,rng):
        if rng.random() < epsilon:
            return int(rng.integers(4))
        scores = self.scores(observation)
        return int(rng.choice(np.flatnonzero(scores==scores.max())))

    def update(self,observation,action,reward,next_observation,terminated,alpha,gamma,next_action=None):
        features = encode_features(observation)
        target = float(reward)
        if not terminated:
            values = self.scores(next_observation)
            if self.algorithm == 'sarsa':
                if next_action not in range(4):
                    raise ValueError('SARSA requires its next behavior action')
                target += gamma*values[next_action]
            else:
                target += gamma*values.max()
        error = target-self.weights[action] @ features
        self.weights[action] += alpha*error*features/float(features @ features)
        if not np.isfinite(self.weights[action]).all():
            raise FloatingPointError('Linear value function diverged')

    def state_fingerprint(self):
        return hashlib.sha256(self.weights.tobytes()).hexdigest()

    def save(self,path,metadata):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,weights=self.weights)
        write_json(path.with_suffix('.json'),{**metadata,'signature':self.benchmark.signature,
            'algorithm':self.algorithm,'policy_id':self.policy_id,'encoding':OBSERVATION,
            'feature_version':FEATURE_VERSION,'feature_count':FEATURE_COUNT,
            'value_function':'shared linear normalized semi-gradient',
            'kind':'inference-only; exact training resume unsupported',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


def build_shared_policy(benchmark,env,algorithm,config,seed,policy_id):
    if algorithm in ('q-learning','sarsa'):
        return LinearMapPolicy(benchmark,algorithm,policy_id)
    from .shared_neural import SharedMapExtractor
    from .unseen_agents import build_neural
    return build_neural(benchmark,env,algorithm,config,seed,policy_id,
                        features_extractor=SharedMapExtractor,feature_version=FEATURE_VERSION)
