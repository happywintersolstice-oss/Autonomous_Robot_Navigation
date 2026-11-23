"""One interface for every agent, so evaluation code never cares what is inside.

    agent.reset()                  called at the start of each episode
    agent.act(obs, env) -> action  env is passed so privileged baselines (A*) can
                                   read the map; learning agents must ignore it.
"""

from __future__ import annotations

import numpy as np


class Agent:
    name = "agent"
    action_mode = "discrete"

    def reset(self):
        pass

    def act(self, obs, env):
        raise NotImplementedError


class SB3Agent(Agent):
    """Wraps any Stable-Baselines3 / sb3-contrib model (handles LSTM state for RecurrentPPO)."""

    def __init__(self, model, name, action_mode, recurrent=False, deterministic=True):
        self.model, self.name, self.action_mode = model, name, action_mode
        self.recurrent, self.deterministic = recurrent, deterministic
        self.reset()

    def reset(self):
        self._state = None
        self._start = np.ones((1,), dtype=bool)

    def act(self, obs, env):
        if self.recurrent:
            action, self._state = self.model.predict(
                obs, state=self._state, episode_start=self._start, deterministic=self.deterministic
            )
            self._start = np.zeros((1,), dtype=bool)
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action) if self.action_mode == "discrete" else np.asarray(action, dtype=np.float32)
