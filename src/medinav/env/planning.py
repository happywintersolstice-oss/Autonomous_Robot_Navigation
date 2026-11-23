"""Classical grid planning used by the project.

Why this exists
---------------
RL needs a *reference point*. A* with full map knowledge gives the shortest
collision-free path, which we use for three things:

1. The "A* oracle" baseline agent (classical robotics approach we compare against).
2. Path efficiency metric = optimal length / length the RL robot actually drove.
3. Geodesic reward shaping and the hybrid "A* waypoint + RL" agent (Part B).

Everything here is pure numpy / heapq so it runs anywhere without extra packages.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

# 8-connected moves with their step cost (diagonals cost sqrt(2)).
_MOVES = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def inflate(grid: np.ndarray, cells: int) -> np.ndarray:
    """Grow every obstacle by `cells` in all directions (a disc-shaped dilation).

    The robot is a circle, not a point. Planning on an inflated map lets us treat
    the robot as a point while still guaranteeing clearance.
    """
    grid = np.asarray(grid, dtype=bool)
    if cells <= 0:
        return grid.copy()
    out = grid.copy()
    h, w = grid.shape
    for dr in range(-cells, cells + 1):
        for dc in range(-cells, cells + 1):
            # (cells + 0.5)^2 keeps diagonal neighbours: a cell's CORNER can be closer
            # than its centre, and the robot must clear corners too.
            if dr * dr + dc * dc > (cells + 0.5) ** 2:
                continue
            shifted = np.zeros_like(grid)
            src = grid[max(0, -dr): h - max(0, dr), max(0, -dc): w - max(0, dc)]
            shifted[max(0, dr): h - max(0, -dr), max(0, dc): w - max(0, -dc)] = src
            out |= shifted
    return out


def _neighbours(grid, r, c):
    h, w = grid.shape
    for dr, dc, cost in _MOVES:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < h and 0 <= cc < w) or grid[rr, cc]:
            continue
        # No corner cutting: a diagonal move needs both side cells free.
        if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):
            continue
        yield rr, cc, cost


def astar(grid: np.ndarray, start: tuple, goal: tuple):
    """Shortest 8-connected path between two free cells, or None if unreachable."""
    grid = np.asarray(grid, dtype=bool)
    start, goal = tuple(start), tuple(goal)
    if grid[start] or grid[goal]:
        return None

    def h(cell):  # octile distance: admissible heuristic for 8-connected grids
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

    open_heap = [(h(start), 0.0, start)]
    came_from = {start: None}
    g_cost = {start: 0.0}
    while open_heap:
        _, g, cell = heapq.heappop(open_heap)
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1]
        if g > g_cost[cell]:
            continue  # stale heap entry
        for rr, cc, cost in _neighbours(grid, *cell):
            ng = g + cost
            if ng < g_cost.get((rr, cc), math.inf):
                g_cost[(rr, cc)] = ng
                came_from[(rr, cc)] = cell
                heapq.heappush(open_heap, (ng + h((rr, cc)), ng, (rr, cc)))
    return None


def geodesic_field(grid: np.ndarray, goal: tuple) -> np.ndarray:
    """Dijkstra from the goal: distance-to-goal (in cells) for every free cell.

    Unlike straight-line distance, this "walks around" walls, so a reward based on
    it never pulls the robot into a dead end. Unreachable cells are +inf.
    """
    grid = np.asarray(grid, dtype=bool)
    field = np.full(grid.shape, np.inf)
    goal = tuple(goal)
    if grid[goal]:
        return field
    field[goal] = 0.0
    heap = [(0.0, goal)]
    while heap:
        d, cell = heapq.heappop(heap)
        if d > field[cell]:
            continue
        for rr, cc, cost in _neighbours(grid, *cell):
            nd = d + cost
            if nd < field[rr, cc]:
                field[rr, cc] = nd
                heapq.heappush(heap, (nd, (rr, cc)))
    return field


def descend(field: np.ndarray, cell: tuple, steps: int) -> tuple:
    """Follow the geodesic field downhill for `steps` cells (used as an A* waypoint)."""
    r, c = cell
    h, w = field.shape
    for _ in range(steps):
        best, best_val = None, field[r, c]
        for dr, dc, _ in _MOVES:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and field[rr, cc] < best_val:
                best, best_val = (rr, cc), field[rr, cc]
        if best is None:
            break
        r, c = best
    return r, c


def path_length(cells, resolution: float) -> float:
    """Length in metres of a cell path."""
    if not cells or len(cells) < 2:
        return 0.0
    arr = np.asarray(cells, dtype=float)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum() * resolution)
