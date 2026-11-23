"""Procedural hospital floor generator.

Why this exists
---------------
Supervised learning has a train/test split of rows. RL has no rows, so we split
*maps* instead: every map is generated deterministically from an integer seed,
and seeds come from three disjoint ranges:

    train       seeds 0 .. train_map_pool-1      (agents learn here)
    validation  seeds 1_000_000 + i              (model selection, tuning, early stopping)
    test        seeds 2_000_000 + i              (final, never-seen maps for the report)

A map is a ward floor: outer walls, interior partition walls with doorways
(ward boundaries), and rectangular clutter (beds, trolleys). Start (pharmacy)
and goal (ward bed) are placed so the goal is reachable and at a geodesic
distance that matches the difficulty level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .planning import geodesic_field, inflate

SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}

# Curriculum / difficulty levels. Higher level = more walls, clutter, people and longer trips.
DIFFICULTY = {
    0: dict(n_walls=0, n_obstacles=3, n_humans=0, goal_dist=(2.0, 4.0)),
    1: dict(n_walls=1, n_obstacles=5, n_humans=1, goal_dist=(3.0, 6.0)),
    2: dict(n_walls=2, n_obstacles=7, n_humans=2, goal_dist=(4.0, 8.0)),
    3: dict(n_walls=3, n_obstacles=9, n_humans=4, goal_dist=(5.0, 11.0)),
}


def map_seed(split: str, index: int) -> int:
    """Seed of the `index`-th map in a split (keeps the three splits disjoint)."""
    return SPLIT_OFFSETS[split] + int(index)


@dataclass
class Layout:
    """Everything that defines one episode's static world."""

    grid: np.ndarray            # bool [rows, cols], True = wall/obstacle
    resolution: float           # metres per cell
    start: np.ndarray           # (x, y) metres
    start_heading: float        # radians
    goal: np.ndarray            # (x, y) metres
    human_spawns: list = field(default_factory=list)  # list of (x, y)
    seed: int = 0
    difficulty: int = 0
    geodesic: np.ndarray | None = None   # distance-to-goal field (cells) on inflated grid
    inflated: np.ndarray | None = None   # grid inflated by robot radius

    @property
    def size_m(self) -> float:
        return self.grid.shape[1] * self.resolution

    def cell(self, xy) -> tuple:
        """World (x, y) -> grid (row, col), clipped inside the map."""
        r = int(np.clip(xy[1] / self.resolution, 0, self.grid.shape[0] - 1))
        c = int(np.clip(xy[0] / self.resolution, 0, self.grid.shape[1] - 1))
        return r, c

    def centre(self, cell) -> np.ndarray:
        """Grid (row, col) -> world (x, y) of the cell centre."""
        return np.array([(cell[1] + 0.5) * self.resolution, (cell[0] + 0.5) * self.resolution])

    def geodesic_m(self, xy) -> float:
        """Geodesic distance to goal in metres.

        The field is computed on the INFLATED grid, so a robot hugging a wall (legal:
        it collides only when its body touches the raw grid) can sit in a cell whose
        field value is +inf. Switching to straight-line distance there would make the
        progress reward jump by metres in a single step, so instead we take the
        nearest reachable cell within a small window and add the hop to it.
        Straight-line distance is used only if nothing reachable is nearby.
        """
        r, c = self.cell(xy)
        d = self.geodesic[r, c]
        if np.isfinite(d):
            return float(d * self.resolution)
        k = 3  # search window radius in cells (~0.75 m at 0.25 m resolution)
        r0, r1 = max(0, r - k), min(self.geodesic.shape[0], r + k + 1)
        c0, c1 = max(0, c - k), min(self.geodesic.shape[1], c + k + 1)
        window = self.geodesic[r0:r1, c0:c1]
        finite = np.argwhere(np.isfinite(window))
        if len(finite):
            cells = finite + [r0, c0]
            centres = (cells[:, ::-1] + 0.5) * self.resolution          # (row, col) -> (x, y)
            hop = np.linalg.norm(centres - np.asarray(xy, dtype=float), axis=1)
            return float(np.min(window[finite[:, 0], finite[:, 1]] * self.resolution + hop))
        return float(np.linalg.norm(np.asarray(xy) - self.goal))


def _add_partition_walls(grid, rng, n_walls):
    """Full-length interior walls with 1-2 doorways (each >= 1 m wide)."""
    n = grid.shape[0]
    door = max(4, int(round(1.0 / 0.25)))  # doorway width in cells (~1 m)
    for _ in range(n_walls):
        pos = int(rng.integers(n // 4, 3 * n // 4))
        n_doors = int(rng.integers(1, 3))
        line = np.ones(n, dtype=bool)
        line[0] = line[-1] = True
        for _ in range(n_doors):
            d0 = int(rng.integers(1, n - door - 1))
            line[d0: d0 + door] = False
        if rng.random() < 0.5:
            grid[pos, :] |= line          # horizontal wall
        else:
            grid[:, pos] |= line          # vertical wall


def _add_clutter(grid, rng, n_obstacles):
    """Rectangular beds / trolleys of 0.5 - 1.25 m per side."""
    n = grid.shape[0]
    for _ in range(n_obstacles):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        r, c = int(rng.integers(1, n - h - 1)), int(rng.integers(1, n - w - 1))
        grid[r: r + h, c: c + w] = True


def generate_layout(
    seed: int,
    difficulty: int = 1,
    size_m: float = 10.0,
    resolution: float = 0.25,
    robot_radius: float = 0.2,
    n_obstacles: int | None = None,
    n_humans: int | None = None,
    n_walls: int | None = None,
    max_tries: int = 200,
) -> Layout:
    """Deterministically build a reachable layout from `seed`.

    `n_obstacles` / `n_humans` / `n_walls` override the difficulty preset; this is
    how the robustness tests create harder-than-training conditions.
    """
    spec = dict(DIFFICULTY[int(np.clip(difficulty, 0, max(DIFFICULTY)))])
    if n_obstacles is not None:
        spec["n_obstacles"] = n_obstacles
    if n_humans is not None:
        spec["n_humans"] = n_humans
    if n_walls is not None:
        spec["n_walls"] = n_walls
    # Longer trips are possible on bigger maps (robustness test: unseen map size).
    scale = size_m / 10.0
    lo, hi = spec["goal_dist"][0] * scale, spec["goal_dist"][1] * scale

    rng = np.random.default_rng(seed)
    n = int(round(size_m / resolution))
    inflate_cells = int(np.ceil(robot_radius / resolution))

    for _ in range(max_tries):
        grid = np.zeros((n, n), dtype=bool)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
        _add_partition_walls(grid, rng, spec["n_walls"])
        _add_clutter(grid, rng, spec["n_obstacles"])
        inflated = inflate(grid, inflate_cells)
        free = np.argwhere(~inflated)
        if len(free) < 20:
            continue

        goal_cell = tuple(free[rng.integers(len(free))])
        field_ = geodesic_field(inflated, goal_cell)
        dist_m = field_ * resolution
        candidates = np.argwhere(np.isfinite(dist_m) & (dist_m >= lo) & (dist_m <= hi))
        if len(candidates) == 0:
            continue
        start_cell = tuple(candidates[rng.integers(len(candidates))])

        layout = Layout(
            grid=grid,
            resolution=resolution,
            start=np.array([(start_cell[1] + 0.5) * resolution, (start_cell[0] + 0.5) * resolution]),
            start_heading=float(rng.uniform(-np.pi, np.pi)),
            goal=np.array([(goal_cell[1] + 0.5) * resolution, (goal_cell[0] + 0.5) * resolution]),
            seed=seed,
            difficulty=difficulty,
            geodesic=field_,
            inflated=inflated,
        )

        # People spawn in reachable free space, away from the robot start and the goal.
        reachable = np.argwhere(np.isfinite(field_))
        spawns = []
        for _ in range(spec["n_humans"] * 20):
            if len(spawns) >= spec["n_humans"]:
                break
            xy = layout.centre(tuple(reachable[rng.integers(len(reachable))]))
            if np.linalg.norm(xy - layout.start) > 1.5 and np.linalg.norm(xy - layout.goal) > 1.0:
                spawns.append(xy)
        layout.human_spawns = spawns
        return layout
    raise RuntimeError(f"Could not generate a valid layout for seed {seed}")
