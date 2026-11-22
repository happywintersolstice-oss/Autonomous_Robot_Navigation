"""Perfect two-cell occupancy rays. Only this trusted sensor reads occupancy."""

from dataclasses import dataclass

import numpy as np

RAYS = ((-1, 0), (1, 0), (0, 1), (0, -1))  # N, S, E, W (different from action order)


@dataclass(frozen=True)
class Scan:
    distances: tuple[float, ...]
    cells: tuple[tuple[int, int, int], ...]


def scan(blocked, position):
    blocked = np.asarray(blocked)
    if blocked.shape != (10, 10) or not np.isin(blocked, (False, True)).all():
        raise ValueError("Sensor requires a 10x10 occupancy array")
    if len(position) != 2 or any(
        not isinstance(v, (int, np.integer)) for v in position
    ):
        raise ValueError("Position must be an integer pair")
    r, c = position
    if not (0 <= r < 10 and 0 <= c < 10) or blocked[r, c]:
        raise ValueError("Sensor origin must be a free in-bounds cell")
    cells, distances = [(int(r), int(c), 0)], []
    for dr, dc in RAYS:
        distance = 0.0
        for step in (1, 2):
            rr, cc = int(r + dr * step), int(c + dc * step)
            if not (0 <= rr < 10 and 0 <= cc < 10):
                distance = step / 2  # Virtual boundary; never a memory cell.
                break
            value = int(blocked[rr, cc])
            cells.append((rr, cc, value))
            if value:
                distance = step / 2
                break
        distances.append(distance)
    return Scan(tuple(distances), tuple(cells))
