"""Moving people (nurses, patients, visitors).

Each person walks along an A* path to a random destination, pauses briefly,
then picks a new destination. People are not on any map, which is what makes
the problem hard for a classical planner that only knows the static map.

Yielding (env config `humans_yield`, default True)
    A person does not step INTO the robot: if their next position would touch it,
    they wait; after `patience` blocked steps they give up and walk somewhere else.
    Environment v1 had no yielding, and the error analysis showed ~95% of person
    collisions were people walking into a robot that was often standing still -
    unavoidable for ANY policy, so they capped every agent and hid real differences.
    `humans_yield: false` keeps the old behaviour as a stress test ("careless people").

`start_step` supports the non-stationary test: people stand still at first and
start walking mid-episode, so we can check whether the agent adapts.
"""

from __future__ import annotations

import numpy as np

from .planning import astar


class Human:
    radius = 0.25

    patience = 10   # steps a person waits for the robot before choosing a new destination

    def __init__(self, xy, layout, rng, speed=0.5, start_step=0, yield_to_robot=True):
        self.pos = np.asarray(xy, dtype=float).copy()
        self.layout = layout
        self.rng = rng
        self.speed = float(speed)
        self.start_step = int(start_step)
        self.yield_to_robot = bool(yield_to_robot)
        self.blocked = 0
        self.waypoints: list[np.ndarray] = []
        self.pause = 0
        self._new_destination()

    def _new_destination(self):
        reachable = np.argwhere(np.isfinite(self.layout.geodesic))
        for _ in range(10):
            target = tuple(reachable[self.rng.integers(len(reachable))])
            path = astar(self.layout.inflated, self.layout.cell(self.pos), target)
            if path and len(path) > 3:
                self.waypoints = [self.layout.centre(c) for c in path[1:]]
                self._destination = target
                return
        self.waypoints = []
        self._destination = None

    @property
    def velocity_hint(self) -> np.ndarray:
        if not self.waypoints:
            return np.zeros(2)
        d = self.waypoints[0] - self.pos
        n = np.linalg.norm(d)
        return d / n * self.speed if n > 1e-9 else np.zeros(2)

    def step(self, dt: float, t: int, robot_pos=None, robot_radius: float = 0.2):
        if t < self.start_step or self.speed <= 0:
            return
        if self.pause > 0:
            self.pause -= 1
            return
        before = self.pos.copy()
        self._walk(dt)
        if self.yield_to_robot and robot_pos is not None:
            # 5 cm margin: a person stops just short of the robot instead of touching it
            if np.linalg.norm(self.pos - robot_pos) < self.radius + robot_radius + 0.05:
                self.pos = before          # wait this step (the walked waypoints are re-planned below)
                self.blocked += 1
                if self.blocked >= self.patience:
                    self.blocked = 0
                    self._new_destination()
                else:
                    self._resume_from(before)
                return
        self.blocked = 0

    def _resume_from(self, xy):
        """Re-plan to the current destination from xy (waypoints may have been consumed by _walk)."""
        dest = self._destination
        if dest is None:
            return
        path = astar(self.layout.inflated, self.layout.cell(xy), dest)
        self.waypoints = [self.layout.centre(c) for c in path[1:]] if path else []

    def _walk(self, dt: float):
        remaining = self.speed * dt
        while remaining > 0 and self.waypoints:
            d = self.waypoints[0] - self.pos
            dist = np.linalg.norm(d)
            if dist <= remaining:
                self.pos = self.waypoints.pop(0)
                remaining -= dist
            else:
                self.pos = self.pos + d / dist * remaining
                remaining = 0
        if not self.waypoints:
            self.pause = int(self.rng.integers(2, 10))
            self._new_destination()
