"""Independent SB3 models: 409 permitted inputs, 412 affine features, five outputs."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .core import write_json
from .partial_agents import metadata
from .partial_contract import signature
from .partial_features import FEATURE_COUNT
from .partial_observation import validate


class PartialVisibleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        if observation_space.shape != (409,) or observation_space.dtype != np.float32:
            raise ValueError("Partial extractor requires the new 409-value input")
        super().__init__(observation_space, features_dim=FEATURE_COUNT)

    def forward(self, observations):
        return torch.cat(
            (
                torch.ones_like(observations[:, :1]),
                observations,
                observations[:, 2:4] - observations[:, :2],
            ),
            dim=1,
        )


class NeuralPolicy:
    def __init__(self, algorithm, model, config, seed):
        self.algorithm, self.model, self.signature = algorithm, model, signature(config)
        self.training_seed = seed
        self.policy_id = f"{algorithm}/partial-dynamic-seed{seed}"
        self.checkpoint_sha256 = None

    def act(self, observation):
        value = validate(observation)
        action, _ = self.model.predict(value, deterministic=True)
        result = int(np.asarray(action).item())
        if result not in range(5):
            raise ValueError("Neural action is outside the five-action contract")
        return result

    def state_fingerprint(self):
        hasher = hashlib.sha256()
        for name, value in sorted(self.model.policy.state_dict().items()):
            hasher.update(name.encode())
            hasher.update(value.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def save(self, path, extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".zip":
            raise ValueError("Neural checkpoint must use .zip")
        self.model.save(path)
        write_json(path.with_suffix(".json"), metadata(self, path, extra))


def build_neural(env, algorithm, config, seed):
    if env.observation_space.shape != (409,) or env.action_space.n != 5:
        raise ValueError("New neural models require the partial environment")
    torch.set_num_threads(1)
    common = dict(
        seed=seed,
        device="cpu",
        learning_rate=config["neural_learning_rate"],
        gamma=config["discount"],
        verbose=0,
    )
    kwargs = {
        "net_arch": config["net_arch"],
        "features_extractor_class": PartialVisibleExtractor,
    }
    if algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            **common,
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            train_freq=config["train_freq"],
            gradient_steps=1,
            target_update_interval=config["target_update_interval"],
            exploration_initial_eps=config["epsilon_start"],
            exploration_final_eps=config["epsilon_end"],
            exploration_fraction=0.8,
            policy_kwargs=kwargs,
        )
    elif algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            **common,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            policy_kwargs=kwargs,
        )
    else:
        raise ValueError("A DQN/PPO family is required")
    return NeuralPolicy(algorithm, model, env.unwrapped.config, seed)


def load_neural(path, config, record):
    torch.set_num_threads(1)
    model = {"dqn": DQN, "ppo": PPO}[record["algorithm"]].load(path, device="cpu")
    if (
        model.observation_space.shape != (409,)
        or model.action_space.n != 5
        or model.policy_kwargs.get("features_extractor_class")
        is not PartialVisibleExtractor
    ):
        raise ValueError("Partial neural checkpoint shape/extractor mismatch")
    if any(not torch.isfinite(v).all() for v in model.policy.state_dict().values()):
        raise ValueError("Nonfinite neural parameters")
    return NeuralPolicy(record["algorithm"], model, config, record["training_seed"])
