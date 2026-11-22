"""Vanilla SB3 DQN, deterministic inference and coordinate-only normalization."""
import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

from .core import read_json, write_json

ENCODING = "normalized-coordinates-v1"


def normalize(observation, shape):
    h, w = shape
    scale = np.array([max(1, h - 1), max(1, w - 1)] * 2, dtype=np.float32)
    return np.asarray(observation, dtype=np.float32) / scale


class CoordinateObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0., 1., shape=(4,), dtype=np.float32)

    def observation(self, observation):
        return normalize(observation, self.env.unwrapped.grid.shape)


def _hash_value(hasher, value):
    if isinstance(value, torch.Tensor):
        _hash_value(hasher, value.detach().cpu().numpy())
    elif isinstance(value, np.ndarray):
        hasher.update(str((value.shape, value.dtype)).encode())
        hasher.update(value.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            hasher.update(str(key).encode())
            _hash_value(hasher, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            _hash_value(hasher, item)
    else:
        hasher.update(repr(value).encode())


class DQNPolicy:
    algorithm = "dqn"

    def __init__(self, task, model, policy_id):
        self.task, self.model, self.policy_id = task, model, policy_id

    def scores(self, observation):
        normalized = normalize(observation, self.task.grid.shape)
        with torch.inference_mode():
            tensor = torch.as_tensor(normalized, device=self.model.device).unsqueeze(0)
            return self.model.q_net(tensor)[0].cpu().numpy().copy()

    def act(self, observation):
        return int(np.argmax(self.scores(observation)))

    def action_and_scores(self, observation):
        scores = self.scores(observation)
        return int(np.argmax(scores)), scores

    def state_fingerprint(self):
        """Weights, target, optimizer, replay arrays/position and exploration counters."""
        hasher = hashlib.sha256()
        _hash_value(hasher, self.model.get_parameters())
        _hash_value(hasher, (self.model.num_timesteps, self.model._n_updates, self.model.exploration_rate))
        buffer = self.model.replay_buffer
        if buffer is not None:
            _hash_value(hasher, (buffer.pos, buffer.full))
            for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
                _hash_value(hasher, getattr(buffer, name))
        return hasher.hexdigest()

    def save(self, path, metadata):
        path = Path(path)
        if path.suffix != ".zip":
            raise ValueError("DQN checkpoints must use .zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        write_json(path.with_suffix(".json"), {**metadata, "signature": self.task.signature,
            "algorithm": self.algorithm, "policy_id": self.policy_id, "encoding": ENCODING,
            "kind": "inference-only; replay/RNG/environment state not saved; exact resume unsupported",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    @classmethod
    def load(cls, path, task):
        path = Path(path)
        metadata = read_json(path.with_suffix(".json"))
        task.check(metadata["signature"])
        if metadata.get("algorithm") != "dqn" or metadata.get("encoding") != ENCODING:
            raise ValueError("DQN algorithm/encoding mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        torch.set_num_threads(1)
        model = DQN.load(path, device="cpu")
        if model.observation_space.shape != (4,) or model.action_space.n != 4:
            raise ValueError("DQN observation/action shape mismatch")
        if any(not torch.isfinite(value).all() for value in model.policy.state_dict().values()):
            raise ValueError("Nonfinite DQN parameters")
        return cls(task, model, metadata["policy_id"])


def build_model(env, config):
    torch.set_num_threads(config["torch_threads"])
    return DQN("MlpPolicy", CoordinateObservation(env), seed=config["seed"], device="cpu",
        learning_rate=config["learning_rate"], gamma=config["discount"], buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"], batch_size=config["batch_size"],
        train_freq=config["train_freq"], gradient_steps=config["gradient_steps"],
        target_update_interval=config["target_update_interval"],
        exploration_initial_eps=config["epsilon_start"], exploration_final_eps=config["epsilon_end"],
        exploration_fraction=config["epsilon_decay_fraction"], policy_kwargs={"net_arch": config["net_arch"]},
        verbose=0)
