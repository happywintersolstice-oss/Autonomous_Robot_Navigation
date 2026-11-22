"""The single source of truth for maps, task identity and environment dynamics."""
from collections import deque
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import gymnasium as gym
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("Up", "Down", "Left", "Right")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    for attempt in range(6):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM, errno.EINVAL) or attempt == 5:
                raise
            time.sleep(.02 * 2**attempt)


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            file.write(payload)
        # Windows readers or indexing tools can briefly hold the destination.
        # Replace a complete file so readers always see a complete JSON record.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM, errno.EINVAL) or attempt == 5:
                    raise
                time.sleep(.02 * 2**attempt)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class GridMap:
    map_id: str
    rows: tuple[str, ...]

    def __post_init__(self):
        if not self.rows or not self.rows[0] or len({len(r) for r in self.rows}) != 1:
            raise ValueError("Map must be a nonempty rectangle")
        if any(c not in ".#" for r in self.rows for c in r):
            raise ValueError("Only '.' (free) and '#' (blocked) are valid")
        if not self.free_cells:
            raise ValueError("Map has no free cells")

    @property
    def shape(self):
        return len(self.rows), len(self.rows[0])

    @property
    def free_cells(self):
        return [(r, c) for r, row in enumerate(self.rows) for c, v in enumerate(row) if v == "."]

    @property
    def hash(self):
        return digest({"map_id": self.map_id, "rows": self.rows})

    def is_free(self, cell):
        r, c = cell
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1] and self.rows[r][c] == "."

    def distances(self, start):
        start = tuple(start)
        if not self.is_free(start):
            raise ValueError("BFS start must be a free cell")
        distances, queue = {start: 0}, deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES:
                cell = (r + dr, c + dc)
                if self.is_free(cell) and cell not in distances:
                    distances[cell] = distances[(r, c)] + 1
                    queue.append(cell)
        return distances

    def path(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        distances = self.distances(goal)
        if start not in distances:
            raise ValueError("Goal is unreachable from start")
        route = [start]
        while route[-1] != goal:
            r, c = route[-1]
            for dr, dc in MOVES:
                cell = (r + dr, c + dc)
                if distances.get(cell) == distances[(r, c)] - 1:
                    route.append(cell)
                    break
        return route


@dataclass(frozen=True)
class Task:
    grid: GridMap
    config: dict

    @property
    def signature(self):
        return {"map_id": self.grid.map_id, "map_hash": self.grid.hash,
                "task_hash": digest(self.config),
                **{k: self.config[k] for k in ("environment_version", "observation_version", "action_version", "reward_version")}}

    def check(self, signature):
        if signature != self.signature:
            raise ValueError("Artifact task/map/version mismatch; a matching fixed-map artifact is required")


def load_task(root=ROOT):
    config = yaml.safe_load((root / "configs/environment.yaml").read_text())
    data = read_json(root / f"data/maps/{config['map_id']}.json")
    if data["map_id"] != config["map_id"]:
        raise ValueError("Map ID mismatch")
    if not isinstance(config["max_steps"], int) or config["max_steps"] < 1:
        raise ValueError("max_steps must be a positive integer")
    if not all(np.isfinite(config[k]) for k in ("step_reward", "goal_reward", "collision_reward")):
        raise ValueError("Rewards must be finite")
    return Task(GridMap(data["map_id"], tuple(data["rows"])), config)


def validate_scenario(grid, scenario):
    start, goal = tuple(scenario["start"]), tuple(scenario["goal"])
    if (len(start) != 2 or len(goal) != 2 or
            any(not isinstance(x, (int, np.integer)) for x in (*start, *goal))):
        raise ValueError("Coordinates must be integer row/column pairs")
    if start == goal or not grid.is_free(start) or not grid.is_free(goal):
        raise ValueError("Start and goal must be distinct free cells")
    length = len(grid.path(start, goal)) - 1
    if "shortest_length" in scenario and scenario["shortest_length"] != length:
        raise ValueError("Incorrect reference shortest length")
    return start, goal


class NavigationEnv(gym.Env):
    """Coordinate-only observation; no masking, planner fallback or early loop stop.

    The external action cap is implemented here, with terminal-event priority;
    truncations retain next-state bootstrap value in the learner.
    """
    metadata = {"render_modes": []}

    def __init__(self, task, scenarios):
        super().__init__()
        self.task, self.grid = task, task.grid
        self.scenarios = list(scenarios)
        if not self.scenarios:
            raise ValueError("At least one reset scenario is required")
        for scenario in self.scenarios:
            validate_scenario(self.grid, scenario)
        h, w = self.grid.shape
        self.observation_space = gym.spaces.MultiDiscrete([h, w, h, w])
        self.action_space = gym.spaces.Discrete(4)
        self.ended = True

    def _observation(self):
        return np.array((*self.position, *self.goal), dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        if scenario is None:
            scenario = self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        self.position, self.goal = validate_scenario(self.grid, scenario)
        self.scenario = scenario
        self.steps, self.ended = 0, False
        return self._observation(), {"scenario_id": scenario["id"], "outcome": "running"}

    def step(self, action):
        if self.ended:
            raise RuntimeError("Reset before stepping a completed episode")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer in [0, 3]")
        dr, dc = MOVES[int(action)]
        attempted = (self.position[0] + dr, self.position[1] + dc)
        collision = not self.grid.is_free(attempted)
        if not collision:
            self.position = attempted
        goal = self.position == self.goal
        self.steps += 1
        terminated = bool(collision or goal)
        truncated = bool(self.steps >= self.task.config["max_steps"] and not terminated)
        reward = self.task.config["step_reward"]
        reward += self.task.config["collision_reward"] if collision else self.task.config["goal_reward"] if goal else 0
        outcome = "collision" if collision else "success" if goal else "timeout" if truncated else "running"
        self.ended = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, {
            "attempted_position": list(attempted), "outcome": outcome,
            "scenario_id": self.scenario["id"], "steps": self.steps}
