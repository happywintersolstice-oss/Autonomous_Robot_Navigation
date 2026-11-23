"""Observation wrappers: handling missing sensor data (Part B "missing information").

Real LiDARs drop returns (glass doors, dark surfaces, sensor glitches). The env
marks a dropped ray with the sentinel value -1. We compare strategies, mirroring
the classic ways of handling missing values in tabular data:

    none        feed the -1 sentinel unchanged      ("leave it, let the model cope")
    max_range   replace with 1.0 (= "nothing there") ("fill with a constant")
    zero        replace with 0.0 (= "obstacle at the sensor", pessimistic constant)
    last_valid  carry forward the last good reading  ("fill using the pattern over time")
    neighbour   average of the two adjacent rays     ("fill using spatial pattern")

`frame_stack` > 1 concatenates the last k observations so the network can infer
motion of people and smooth over dropouts by itself (alternative: RecurrentPPO).
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

IMPUTE_STRATEGIES = ("none", "max_range", "zero", "last_valid", "neighbour")


class ImputeLidar(gym.ObservationWrapper):
    def __init__(self, env, strategy: str = "max_range"):
        super().__init__(env)
        if strategy not in IMPUTE_STRATEGIES:
            raise ValueError(f"strategy must be one of {IMPUTE_STRATEGIES}")
        self.strategy = strategy
        self.n_rays = env.unwrapped.cfg["n_rays"]
        self._last = None

    def reset(self, **kwargs):
        self._last = None
        return super().reset(**kwargs)

    def observation(self, obs):
        obs = obs.copy()
        scan = obs[: self.n_rays]
        missing = scan < 0
        if self.strategy == "none" or not missing.any():
            if not missing.any():
                self._last = scan.copy()
            return obs
        if self.strategy == "max_range":
            scan[missing] = 1.0
        elif self.strategy == "zero":
            scan[missing] = 0.0
        elif self.strategy == "last_valid":
            fill = self._last if self._last is not None else np.ones_like(scan)
            scan[missing] = fill[missing]
        elif self.strategy == "neighbour":
            left, right = np.roll(scan, 1), np.roll(scan, -1)
            neigh = np.where((left >= 0) & (right >= 0), (left + right) / 2,
                             np.where(left >= 0, left, np.where(right >= 0, right, 1.0)))
            scan[missing] = neigh[missing]
        self._last = scan.copy()
        obs[: self.n_rays] = scan
        return obs


class FlatFrameStack(gym.ObservationWrapper):
    """Concatenate the last k observations into one flat vector (newest last)."""

    def __init__(self, env, k: int):
        super().__init__(env)
        self.k = int(k)
        low = np.tile(env.observation_space.low, self.k)
        high = np.tile(env.observation_space.high, self.k)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.frames = deque(maxlen=self.k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self.observation(None), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(None), r, term, trunc, info

    def observation(self, _):
        return np.concatenate(list(self.frames)).astype(np.float32)


def apply_wrappers(env, wrappers: dict):
    """wrappers = {"impute": "last_valid", "frame_stack": 3}  (both optional)."""
    impute = wrappers.get("impute", "max_range")
    if impute:
        env = ImputeLidar(env, impute)
    k = int(wrappers.get("frame_stack", 1) or 1)
    if k > 1:
        env = FlatFrameStack(env, k)
    return env
