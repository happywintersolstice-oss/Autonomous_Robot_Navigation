"""Reference agents every RL result is compared against.

RandomAgent      lower bound: what does "no intelligence" score?
AStarAgent       classical robotics: plan the shortest path on the KNOWN static map,
                 then track it with a simple pure-pursuit controller. It has
                 privileged information (the full map) but is blind to people, so
                 it shows exactly where a static planner fails in a busy hospital.
"""

from __future__ import annotations

import math

import numpy as np

from ..env.planning import astar, inflate
from .base import Agent


class RandomAgent(Agent):
    def __init__(self, action_mode="discrete", seed=0):
        self.name, self.action_mode = "random", action_mode
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        if self.action_mode == "discrete":
            return int(self.rng.integers(5))
        return self.rng.uniform(-1, 1, size=2).astype(np.float32)


class AStarAgent(Agent):
    def __init__(self, action_mode="discrete", lookahead_m=0.4, margin_cells=2):
        self.name, self.action_mode = "astar", action_mode
        self.lookahead_m, self.margin_cells = lookahead_m, margin_cells
        self.path = None

    def reset(self):
        self.path = None

    def _plan(self, env):
        lay = env.unwrapped.layout
        start, goal = lay.cell(env.unwrapped.pos), lay.cell(lay.goal)
        # try with an extra safety margin first, then relax it in narrow maps
        for m in (self.margin_cells, 1, 0):
            grid = inflate(lay.grid, m)
            grid[start] = grid[goal] = False
            cells = astar(grid, start, goal)
            if cells:
                return np.array([lay.centre(c) for c in cells] + [lay.goal])
        return np.array([lay.goal])

    def act(self, obs, env):
        u = env.unwrapped
        if self.path is None:
            self.path = self._plan(env)
        # pure pursuit: aim at the first path point at least `lookahead` away,
        # starting from the closest path point (never steering backwards).
        d = np.linalg.norm(self.path - u.pos, axis=1)
        i = int(np.argmin(d))
        ahead = np.flatnonzero(d[i:] >= self.lookahead_m)
        target = self.path[i + ahead[0]] if len(ahead) else self.path[-1]
        err = (math.atan2(target[1] - u.pos[1], target[0] - u.pos[0]) - u.heading + math.pi) % (2 * math.pi) - math.pi
        if self.action_mode == "discrete":
            if abs(err) > 0.7:
                return 3 if err > 0 else 4
            if abs(err) > 0.2:
                return 1 if err > 0 else 2
            return 0
        v = max(0.0, math.cos(err)) ** 2
        w = float(np.clip(2.0 * err / u.cfg["w_max"], -1, 1))
        return np.array([2 * v - 1, w], dtype=np.float32)
