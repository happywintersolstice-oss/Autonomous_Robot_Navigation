"""Ensembles: combine several trained agents into one decision (Part B).

QAverageEnsemble     average the Q-values of several DQN-family models (e.g. 5 seeds
                     or DQN + Double + Dueling), then take argmax. Averaging cancels
                     out each network's individual over-/under-estimation errors.
MajorityVoteEnsemble each discrete agent votes for an action; ties -> first agent's
                     choice. Works with ANY discrete agents (tabular + deep).
ActionAverageEnsemble average continuous (v, w) commands from e.g. PPO + SAC + TD3.
                     Smooths jerky individual policies.
SafetyShieldEnsemble a learned policy proposes, a simple rule checks the front LiDAR
                     sector and overrides with a turn if a collision is imminent.
                     (Classic "shielded RL" idea - a cheap hybrid of rules + learning.)

The hybrid A*-guided RL agent lives in the environment instead (config
`guidance: true`): A* gives the global waypoint, RL does local avoidance.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .base import Agent


class QAverageEnsemble(Agent):
    action_mode = "discrete"

    def __init__(self, agents, name="q_average"):
        from .dqn_family import q_values  # torch only needed for this ensemble

        self.agents, self.name, self._q = agents, name, q_values

    def reset(self):
        for a in self.agents:
            a.reset()

    def act(self, obs, env):
        q = np.mean([self._q(a.model, obs) for a in self.agents], axis=0)
        return int(np.argmax(q))


class MajorityVoteEnsemble(Agent):
    action_mode = "discrete"

    def __init__(self, agents, name="majority_vote"):
        self.agents, self.name = agents, name

    def reset(self):
        for a in self.agents:
            a.reset()

    def act(self, obs, env):
        votes = [a.act(obs, env) for a in self.agents]
        counts = Counter(votes)
        best = max(counts.values())
        return next(v for v in votes if counts[v] == best)


class ActionAverageEnsemble(Agent):
    action_mode = "continuous"

    def __init__(self, agents, weights=None, name="action_average"):
        self.agents, self.name = agents, name
        self.weights = np.ones(len(agents)) / len(agents) if weights is None else np.asarray(weights) / np.sum(weights)

    def reset(self):
        for a in self.agents:
            a.reset()

    def act(self, obs, env):
        acts = np.array([a.act(obs, env) for a in self.agents], dtype=float)
        return (self.weights[:, None] * acts).sum(axis=0).astype(np.float32)


class SafetyShieldEnsemble(Agent):
    def __init__(self, agent, n_rays=24, max_range=3.0, stop_distance=0.35, name=None):
        self.agent, self.name = agent, name or f"{agent.name}+shield"
        self.action_mode = agent.action_mode
        self.n_rays, self.max_range, self.stop = n_rays, max_range, stop_distance
        ang = (np.degrees(np.linspace(0, 2 * np.pi, n_rays, endpoint=False)) + 180) % 360 - 180
        self.front = np.flatnonzero(np.abs(ang) <= 30)
        self.left = np.flatnonzero((ang > 30) & (ang < 120))
        self.right = np.flatnonzero((ang < -30) & (ang > -120))
        self.overrides = 0

    def reset(self):
        self.agent.reset()

    def act(self, obs, env):
        action = self.agent.act(obs, env)
        scan = np.where(obs[: self.n_rays] < 0, 1.0, obs[: self.n_rays]) * self.max_range
        if scan[self.front].min() >= self.stop:
            return action
        self.overrides += 1
        turn_left = scan[self.left].mean() >= scan[self.right].mean()
        if self.action_mode == "discrete":
            return 3 if turn_left else 4
        return np.array([-1.0, 1.0 if turn_left else -1.0], dtype=np.float32)  # stop and rotate
