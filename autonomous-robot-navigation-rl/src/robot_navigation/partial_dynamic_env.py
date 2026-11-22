"""Independent five-action environment; diagnostics never enter the policy info."""

from copy import deepcopy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .core import GridMap, digest
from .partial_contract import MOVES, configuration, signature, validate_config
from .partial_memory import AgentMemory
from .partial_observation import encode, SPACE
from .partial_sensor import scan


def make_scenario(
    rows,
    start,
    goal,
    events=(),
    *,
    scenario_id="manual",
    layout_id="handwritten",
    event_seed=0,
    condition=None,
    split="manual"
):
    record = dict(
        id=scenario_id,
        layout_id=layout_id,
        rows=list(rows),
        start=list(start),
        goal=list(goal),
        events=deepcopy(list(events)),
        event_seed=event_seed,
        condition=condition or ("familiar-dynamic" if events else "familiar-static"),
        split=split,
        layout_hash=digest(list(rows)),
    )
    validate_scenario(record)
    return record


def validate_scenario(record):
    grid = GridMap(record["layout_id"], tuple(record["rows"]))
    if grid.shape != (10, 10) or record["layout_hash"] != digest(list(grid.rows)):
        raise ValueError("Partial scenario needs its matching 10x10 layout")
    for name in ("start", "goal"):
        cell = record[name]
        if (
            len(cell) != 2
            or any(type(v) is not int for v in cell)
            or not grid.is_free(cell)
        ):
            raise ValueError("Start/goal must be free integer cells")
    if record["start"] == record["goal"] or tuple(record["goal"]) not in grid.distances(
        record["start"]
    ):
        raise ValueError("A distinct goal must be reachable in base geometry")
    if type(record["event_seed"]) is not int or record["event_seed"] < 0:
        raise ValueError("Event seed must be nonnegative")
    ids, windows = set(), {}
    for event in record["events"]:
        cell = event["cell"]
        if (
            len(cell) != 2
            or any(type(v) is not int for v in cell)
            or not grid.is_free(cell)
            or cell in (record["start"], record["goal"])
        ):
            raise ValueError("Events require base-free cells excluding start and goal")
        if (
            type(event["start"]) is not int
            or type(event["clear"]) is not int
            or not 0 <= event["start"] < event["clear"]
        ):
            raise ValueError("Events need a finite, nonempty integer time window")
        if not isinstance(event["id"], str) or not event["id"] or event["id"] in ids:
            raise ValueError("Event IDs must be nonempty and unique")
        ids.add(event["id"])
        for start, clear in windows.setdefault(tuple(cell), []):
            if max(start, event["start"]) < min(clear, event["clear"]):
                raise ValueError("Overlapping events at the same cell are unsupported")
        windows[tuple(cell)].append((event["start"], event["clear"]))
    return grid


@dataclass
class _TrueState:
    grid: GridMap
    permanent: np.ndarray
    blocked: np.ndarray
    position: tuple
    goal: tuple
    step: int = 0


class PartialDynamicEnv(gym.Env):
    """Agents receive only (observation, reward, flags, public outcome/time info).

    Natural terminals scan the executed occupancy without applying new events.
    External truncation receives the real continuing t+1 state for bootstrapping:
    apply its events and scan before returning truncated=True. No timeout bonus.
    """

    metadata = {"render_modes": []}

    def __init__(self, scenarios, config=None):
        super().__init__()
        self.config = validate_config(config if config is not None else configuration())
        self.signature = signature(self.config)
        self._scenarios = deepcopy(list(scenarios))
        if not self._scenarios:
            raise ValueError("At least one declared scenario is required")
        for scenario in self._scenarios:
            validate_scenario(scenario)
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = deepcopy(SPACE)
        self._ended = True

    def _apply_events(self):
        state = self._true_state
        active = [
            e for e in self._scenario["events"] if e["start"] <= state.step < e["clear"]
        ]
        applied = [e for e in active if tuple(e["cell"]) != state.position]
        state.blocked = state.permanent.copy()
        for event in applied:
            state.blocked[tuple(event["cell"])] = True
        self._event_log.append(
            {
                "step": state.step,
                "nominal_active": [e["id"] for e in active],
                "applied": [e["id"] for e in applied],
                "suppressed": [
                    e["id"] for e in active if tuple(e["cell"]) == state.position
                ],
            }
        )

    def _observe(self):
        state = self._true_state
        self._reading = scan(state.blocked, state.position)
        self._scan_update = self.agent_memory.update(self._reading, state.step)
        return encode(
            self.agent_memory,
            state.position,
            state.goal,
            self._reading.distances,
            state.step,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        if scenario is None:
            scenario = self._scenarios[
                int(self.np_random.integers(len(self._scenarios)))
            ]
        self._scenario = deepcopy(scenario)
        grid = validate_scenario(self._scenario)
        permanent = np.array([[c == "#" for c in row] for row in grid.rows], dtype=bool)
        self._true_state = _TrueState(
            grid,
            permanent,
            permanent.copy(),
            tuple(scenario["start"]),
            tuple(scenario["goal"]),
        )
        self.agent_memory = AgentMemory(scenario["start"])
        self._event_log, self._ended, self._outcome = [], False, "running"
        self._apply_events()
        return self._observe(), {"outcome": "running", "steps": 0}

    def step(self, action):
        if self._ended:
            raise RuntimeError("Reset before stepping an ended episode")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(
            action
        ):
            raise ValueError("Action must be an integer from 0 to 4")
        action = int(action)
        state = self._true_state
        dr, dc = MOVES[action]
        attempted = (state.position[0] + dr, state.position[1] + dc)
        inside = all(0 <= v < 10 for v in attempted)
        collision = not inside or bool(state.blocked[attempted])
        if not collision and action != 4:
            state.position = attempted
            self.agent_memory.enter(state.position)
        success = state.position == state.goal
        reward = self.config["step_reward"] + (
            self.config["collision_penalty"]
            if collision
            else self.config["success_bonus"] if success else 0
        )
        state.step += 1
        terminated = bool(collision or success)
        truncated = bool(not terminated and state.step >= self.config["max_steps"])
        self._outcome = (
            "collision"
            if collision
            else "success" if success else "timeout" if truncated else "running"
        )
        if not terminated:
            self._apply_events()
        observation = self._observe()
        self._ended = terminated or truncated
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            {"outcome": self._outcome, "steps": state.step},
        )

    def diagnostic_snapshot(self):
        """Trusted evaluator/debug channel. Never pass this object to a policy."""
        state = self._true_state
        return deepcopy(
            {
                "scenario_id": self._scenario["id"],
                "position": list(state.position),
                "goal": list(state.goal),
                "step": state.step,
                "outcome": self._outcome,
                "true_rows": list(state.grid.rows),
                "occupancy": state.blocked.tolist(),
                "nominal_events": self._scenario["events"],
                "event_log": self._event_log,
                "scan": {
                    "distances": list(self._reading.distances),
                    "cells": [list(c) for c in self._reading.cells],
                },
                "memory": self.agent_memory.snapshot(),
                "scan_update": self._scan_update,
            }
        )
