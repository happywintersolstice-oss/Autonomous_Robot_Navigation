"""Neutral map and atomic JSON utilities; no navigation policy/environment."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import errno, hashlib, json, os, tempfile, time

ROOT = Path(__file__).resolve().parents[2]
_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path):
    for attempt in range(6):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            if (
                exc.errno not in (errno.EACCES, errno.EPERM, errno.EINVAL)
                or attempt == 5
            ):
                raise
            time.sleep(0.02 * 2**attempt)


def write_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(payload)
        # Windows readers or indexing tools can briefly hold the destination.
        # Replace a complete file so readers always see a complete JSON record.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if (
                    exc.errno not in (errno.EACCES, errno.EPERM, errno.EINVAL)
                    or attempt == 5
                ):
                    raise
                time.sleep(0.02 * 2**attempt)
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
        return [
            (r, c)
            for r, row in enumerate(self.rows)
            for c, v in enumerate(row)
            if v == "."
        ]

    @property
    def hash(self):
        return digest({"map_id": self.map_id, "rows": self.rows})

    def is_free(self, cell):
        r, c = cell
        return (
            0 <= r < self.shape[0] and 0 <= c < self.shape[1] and self.rows[r][c] == "."
        )

    def distances(self, start):
        start = tuple(start)
        if not self.is_free(start):
            raise ValueError("BFS start must be a free cell")
        distances, queue = {start: 0}, deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _CARDINAL:
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
            for dr, dc in _CARDINAL:
                cell = (r + dr, c + dc)
                if distances.get(cell) == distances[(r, c)] - 1:
                    route.append(cell)
                    break
        return route
