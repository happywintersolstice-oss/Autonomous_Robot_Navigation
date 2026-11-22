"""Fresh five-action semi-gradient Q-learning/SARSA; strict checkpoint identities."""

import hashlib
from pathlib import Path

import numpy as np

from .core import read_json, write_json
from .release import sha
from .partial_contract import FEATURE, signature
from .partial_features import FEATURE_COUNT, encode_features

ALGORITHMS = ("q-learning", "sarsa", "dqn", "ppo")


def metadata(policy, path, extra):
    return {
        **extra,
        "signature": policy.signature,
        "algorithm": policy.algorithm,
        "policy_id": policy.policy_id,
        "training_seed": policy.training_seed,
        "feature_version": FEATURE,
        "feature_count": FEATURE_COUNT,
        "actions": 5,
        "observation_values": 409,
        "checkpoint_sha256": sha(path),
        "kind": "inference-only; optimizer/buffer/RNG resume not supported",
    }


class LinearPolicy:
    def __init__(self, algorithm, config, seed=0):
        if algorithm not in ("q-learning", "sarsa"):
            raise ValueError("A linear RL family is required")
        self.algorithm, self.signature, self.training_seed = (
            algorithm,
            signature(config),
            seed,
        )
        self.policy_id = f"{algorithm}/partial-dynamic-seed{seed}"
        self.weights = np.zeros((5, FEATURE_COUNT), dtype=np.float64)
        self.checkpoint_sha256 = None

    def scores(self, observation):
        return self.weights @ encode_features(observation)

    def act(self, observation):
        return int(np.argmax(self.scores(observation)))

    def explore(self, observation, epsilon, rng):
        if not 0 <= epsilon <= 1:
            raise ValueError("Exploration probability must be in [0,1]")
        if rng.random() < epsilon:
            return int(rng.integers(5))
        scores = self.scores(observation)
        return int(rng.choice(np.flatnonzero(scores == scores.max())))

    def update(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        alpha,
        gamma,
        next_action=None,
    ):
        if (
            type(action) is not int
            or action not in range(5)
            or not 0 < alpha <= 1
            or not 0 <= gamma <= 1
            or not np.isfinite(reward)
        ):
            raise ValueError("Invalid linear update")
        features = encode_features(observation)
        target = float(reward)
        if not terminated:
            values = self.scores(next_observation)
            if self.algorithm == "sarsa":
                if type(next_action) is not int or next_action not in range(5):
                    raise ValueError("SARSA needs its next behavior-policy action")
                target += gamma * values[next_action]
            else:
                target += gamma * values.max()
        error = target - self.weights[action] @ features
        self.weights[action] += alpha * error * features / float(features @ features)
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("Linear values diverged")

    def state_fingerprint(self):
        return hashlib.sha256(self.weights.tobytes()).hexdigest()

    def save(self, path, extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".npz":
            raise ValueError("Linear checkpoint must use .npz")
        np.savez_compressed(path, weights=self.weights)
        write_json(path.with_suffix(".json"), metadata(self, path, extra))


def load_policy(path, config):
    path = Path(path)
    record = read_json(path.with_suffix(".json"))
    if (
        record.get("signature") != signature(config)
        or record.get("feature_version") != FEATURE
        or record.get("feature_count") != FEATURE_COUNT
        or record.get("actions") != 5
        or record.get("observation_values") != 409
    ):
        raise ValueError("Partial checkpoint task/observation/action/feature mismatch")
    if sha(path) != record["checkpoint_sha256"]:
        raise ValueError("Partial checkpoint checksum mismatch")
    algorithm = record["algorithm"]
    if algorithm in ("q-learning", "sarsa"):
        policy = LinearPolicy(algorithm, config, record["training_seed"])
        with np.load(path, allow_pickle=False) as data:
            weights = data["weights"]
            if weights.shape != (5, FEATURE_COUNT) or not np.isfinite(weights).all():
                raise ValueError("Invalid five-action linear weights")
            policy.weights[:] = weights
    elif algorithm in ("dqn", "ppo"):
        from .partial_neural import load_neural

        policy = load_neural(path, config, record)
    else:
        raise ValueError("Unknown partial RL family")
    policy.policy_id = record["policy_id"]
    policy.checkpoint_sha256 = record["checkpoint_sha256"]
    return policy
