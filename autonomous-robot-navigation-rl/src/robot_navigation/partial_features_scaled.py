"""Versioned Phase 5B transforms. Policies receive only the permitted 409 values."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from .core import read_json, write_json
from .release import sha
from .partial_agents import LinearPolicy, load_policy
from .partial_contract import FEATURE, signature
from .partial_features import FEATURE_COUNT, encode_features
from .partial_neural import NeuralPolicy, PartialVisibleExtractor, build_neural

SCALED = "partial-visible-relative-scale4-v1"
ARMS = {"B0-original": FEATURE, "B1-relative-x4": SCALED}


def encode_scaled(observation, feature_version=SCALED):
    if feature_version not in ARMS.values():
        raise ValueError("Unknown feature version")
    result = encode_features(observation)
    if feature_version == SCALED:
        result[-2:] *= 4.0
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 4.000001:
        raise ValueError("Invalid transformed feature bounds")
    return result


class RelativeScale4Extractor(PartialVisibleExtractor):
    def forward(self, observations):
        return torch.cat(
            (
                torch.ones_like(observations[:, :1]),
                observations,
                4.0 * (observations[:, 2:4] - observations[:, :2]),
            ),
            dim=1,
        )


def feature_signature(config, feature_version):
    if feature_version not in ARMS.values():
        raise ValueError("Unknown feature version")
    return {**signature(config), "feature_version": feature_version}


class ScaledLinearPolicy(LinearPolicy):
    """Same normalized semi-gradient rule as the original; only phi changes."""

    def __init__(self, algorithm, config, seed=0):
        super().__init__(algorithm, config, seed)
        self.signature = feature_signature(config, SCALED)

    def scores(self, observation):
        return self.weights @ encode_scaled(observation)

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
        phi = encode_scaled(observation)
        target = float(reward)
        if not terminated:
            values = self.scores(next_observation)
            if self.algorithm == "sarsa":
                if type(next_action) is not int or next_action not in range(5):
                    raise ValueError("SARSA needs its next behavior-policy action")
                target += gamma * values[next_action]
            else:
                target += gamma * values.max()
        error = target - self.weights[action] @ phi
        self.weights[action] += alpha * error * phi / float(phi @ phi)
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("Linear values diverged")

    def save(self, path, extra):
        super().save(path, extra)
        path = Path(path)
        record = read_json(path.with_suffix(".json"))
        write_json(path.with_suffix(".json"), {**record, "feature_version": SCALED})


class ScaledNeuralPolicy(NeuralPolicy):
    def __init__(self, algorithm, model, config, seed):
        super().__init__(algorithm, model, config, seed)
        self.signature = feature_signature(config, SCALED)

    def save(self, path, extra):
        super().save(path, extra)
        path = Path(path)
        record = read_json(path.with_suffix(".json"))
        write_json(path.with_suffix(".json"), {**record, "feature_version": SCALED})


def build_policy(env, algorithm, model_config, seed, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    if algorithm in ("q-learning", "sarsa"):
        cls = LinearPolicy if arm == "B0-original" else ScaledLinearPolicy
        return cls(algorithm, env.unwrapped.config, seed)
    if arm == "B0-original":
        return build_neural(env, algorithm, model_config, seed)
    if env.observation_space.shape != (409,) or env.action_space.n != 5:
        raise ValueError("Invalid observation/action contract")
    torch.set_num_threads(1)
    common = dict(
        seed=seed,
        device="cpu",
        learning_rate=model_config["neural_learning_rate"],
        gamma=model_config["discount"],
        verbose=0,
        policy_kwargs={
            "net_arch": model_config["net_arch"],
            "features_extractor_class": RelativeScale4Extractor,
        },
    )
    if algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            **common,
            buffer_size=model_config["buffer_size"],
            learning_starts=model_config["learning_starts"],
            batch_size=model_config["batch_size"],
            train_freq=model_config["train_freq"],
            gradient_steps=1,
            target_update_interval=model_config["target_update_interval"],
            exploration_initial_eps=model_config["epsilon_start"],
            exploration_final_eps=model_config["epsilon_end"],
            exploration_fraction=0.8
        )
    elif algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            **common,
            n_steps=model_config["n_steps"],
            batch_size=model_config["batch_size"],
            n_epochs=model_config["n_epochs"]
        )
    else:
        raise ValueError("Unknown algorithm")
    return ScaledNeuralPolicy(algorithm, model, env.unwrapped.config, seed)


def load_scaled_policy(path, config, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    path = Path(path)
    record = read_json(path.with_suffix(".json"))
    if (
        record.get("feature_version") != ARMS[arm]
        or record.get("signature") != feature_signature(config, ARMS[arm])
        or record.get("feature_count") != FEATURE_COUNT
        or record.get("actions") != 5
        or record.get("observation_values") != 409
    ):
        raise ValueError("Checkpoint task/feature version mismatch")
    if sha(path) != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if arm == "B0-original":
        return load_policy(path, config)
    algorithm = record["algorithm"]
    if algorithm in ("q-learning", "sarsa"):
        policy = ScaledLinearPolicy(algorithm, config, record["training_seed"])
        with np.load(path, allow_pickle=False) as data:
            if (
                data["weights"].shape != (5, 412)
                or not np.isfinite(data["weights"]).all()
            ):
                raise ValueError("Invalid linear checkpoint")
            policy.weights[:] = data["weights"]
    elif algorithm in ("dqn", "ppo"):
        model = {"dqn": DQN, "ppo": PPO}[algorithm].load(path, device="cpu")
        if (
            model.policy_kwargs.get("features_extractor_class")
            is not RelativeScale4Extractor
            or model.observation_space.shape != (409,)
            or model.action_space.n != 5
            or any(
                not torch.isfinite(v).all() for v in model.policy.state_dict().values()
            )
        ):
            raise ValueError("Invalid scaled neural checkpoint")
        policy = ScaledNeuralPolicy(algorithm, model, config, record["training_seed"])
    else:
        raise ValueError("Unknown algorithm")
    policy.policy_id, policy.checkpoint_sha256 = (
        record["policy_id"],
        record["checkpoint_sha256"],
    )
    return policy


def parameter_norm(policy):
    if hasattr(policy, "weights"):
        return float(np.linalg.norm(policy.weights))
    return float(
        torch.sqrt(
            sum(torch.sum(p.detach() ** 2) for p in policy.model.policy.parameters())
        ).item()
    )
