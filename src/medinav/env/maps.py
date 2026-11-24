"""Procedural hospital floor generator.

What a floor looks like
-----------------------
A rectangular hospital wing (default 24 m x 16 m): one main corridor runs along the
middle, with a row of rooms on each side. Every room opens onto the corridor through
a doorway. Rooms have a job and furniture that matches it:

    Pharmacy       shelves along the back wall + a dispensing counter  -> the robot STARTS here
    Ward A, B, ... beds against the back wall, numbered A-1, A-2, ...  -> deliveries GO to a bed
    Nurse station  a desk island
    Lobby          rows of waiting-room chairs
    Supply         shelving

Trolleys are parked along the corridor walls (more on harder levels) and people walk
between rooms (see humans.py). One delivery = "Pharmacy -> Ward C, bed 3".

Why procedural?
---------------
Supervised learning splits rows into train/test. RL has no rows, so we split FLOORS:
every floor is generated deterministically from an integer seed, and seeds come from
three disjoint ranges:

    train       seeds 0 .. train_map_pool-1      (agents learn here)
    validation  seeds 1_000_000 + i              (model selection, tuning, early stopping)
    test        seeds 2_000_000 + i              (final, never-seen floors for the report)

Room count, room widths, corridor width/position, door positions, which room is the
pharmacy, furniture and trolleys all change with the seed, so a test floor is a
genuinely new building, not a shuffled copy of a training floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .planning import geodesic_field, inflate

SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}

# Difficulty / curriculum levels.
#   goal_rank  which ward bed is the target, as a fraction of the sorted route lengths
#              (0 = the closest bed to the pharmacy, 1 = the farthest one)
#   n_humans   people walking around       n_obstacles  trolleys parked in the corridor
DIFFICULTY = {
    0: dict(n_humans=0, n_obstacles=0, goal_rank=(0.0, 0.25)),
    1: dict(n_humans=2, n_obstacles=1, goal_rank=(0.0, 0.5)),
    2: dict(n_humans=4, n_obstacles=3, goal_rank=(0.0, 1.0)),
    3: dict(n_humans=7, n_obstacles=5, goal_rank=(0.5, 1.0)),
}

ROOM_KINDS = ("pharmacy", "ward", "nurses", "lobby", "supply")


def map_seed(split: str, index: int) -> int:
    """Seed of the `index`-th floor in a split (keeps the three splits disjoint)."""
    return SPLIT_OFFSETS[split] + int(index)


@dataclass
class Room:
    name: str                    # "Pharmacy", "Ward C", ...
    kind: str                    # one of ROOM_KINDS
    rect: tuple                  # (x0, y0, x1, y1) metres, interior (inside the walls)
    door: tuple                  # (x0, x1, y) metres: doorway span on the corridor wall
    beds: list = field(default_factory=list)   # [(label, (x, y) delivery point)] for wards


@dataclass
class Layout:
    """Everything that defines one episode's static world."""

    grid: np.ndarray            # bool [rows, cols], True = wall/furniture
    resolution: float           # metres per cell
    start: np.ndarray           # (x, y) metres
    start_heading: float        # radians
    goal: np.ndarray            # (x, y) metres
    human_spawns: list = field(default_factory=list)  # list of (x, y)
    seed: int = 0
    difficulty: int = 0
    geodesic: np.ndarray | None = None   # distance-to-goal field (cells) on inflated grid
    inflated: np.ndarray | None = None   # grid inflated by robot radius
    rooms: list = field(default_factory=list)          # [Room]
    corridor: tuple = (0.0, 0.0, 0.0, 0.0)             # (x0, y0, x1, y1) metres
    trolleys: list = field(default_factory=list)       # [(x0, y0, x1, y1)] metres
    start_room: str = ""
    goal_room: str = ""
    goal_label: str = ""        # e.g. "C-3"

    @property
    def width_m(self) -> float:
        return self.grid.shape[1] * self.resolution

    @property
    def height_m(self) -> float:
        return self.grid.shape[0] * self.resolution

    @property
    def mission(self) -> str:
        return f"{self.start_room} → {self.goal_room}, bed {self.goal_label}"

    def cell(self, xy) -> tuple:
        """World (x, y) -> grid (row, col), clipped inside the map."""
        r = int(np.clip(xy[1] / self.resolution, 0, self.grid.shape[0] - 1))
        c = int(np.clip(xy[0] / self.resolution, 0, self.grid.shape[1] - 1))
        return r, c

    def centre(self, cell) -> np.ndarray:
        """Grid (row, col) -> world (x, y) of the cell centre."""
        return np.array([(cell[1] + 0.5) * self.resolution, (cell[0] + 0.5) * self.resolution])

    def room_at(self, xy) -> str:
        """Name of the room containing xy ('Corridor' / '' if none)."""
        x, y = xy
        for room in self.rooms:
            x0, y0, x1, y1 = room.rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return room.name
        cx0, cy0, cx1, cy1 = self.corridor
        return "Corridor" if cx0 <= x <= cx1 and cy0 <= y <= cy1 else ""

    def geodesic_m(self, xy) -> float:
        """Geodesic (walking-route) distance to the goal in metres.

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
        # Walls are only one cell thick, so a plain square window could "see" a cell in the
        # NEXT room. Instead flood-fill up to k cells over open floor (raw grid) only.
        k = 3  # ~0.75 m at 0.25 m resolution
        h, w = self.grid.shape
        xy = np.asarray(xy, dtype=float)
        best = np.inf
        frontier, seen = [(r, c)], {(r, c)}
        for _ in range(k):
            nxt = []
            for rr, cc in frontier:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and not self.grid[nr, nc]:
                        seen.add((nr, nc))
                        nxt.append((nr, nc))
                        g = self.geodesic[nr, nc]
                        if np.isfinite(g):
                            hop = np.linalg.norm(self.centre((nr, nc)) - xy)
                            best = min(best, g * self.resolution + hop)
            frontier = nxt
        if np.isfinite(best):
            return float(best)
        return float(np.linalg.norm(xy - self.goal))


# ------------------------------------------------------------------------- building blocks
class _Painter:
    """Draws rectangles given in metres onto the occupancy grid."""

    def __init__(self, grid, res):
        self.grid, self.res = grid, res
        self.keepout = np.zeros_like(grid)   # door approaches: furniture may never go here

    def cells(self, x0, y0, x1, y1):
        r = self.res
        c0, c1 = int(np.floor(min(x0, x1) / r + 1e-9)), int(np.ceil(max(x0, x1) / r - 1e-9))
        r0, r1 = int(np.floor(min(y0, y1) / r + 1e-9)), int(np.ceil(max(y0, y1) / r - 1e-9))
        h, w = self.grid.shape
        return slice(max(r0, 0), min(r1, h)), slice(max(c0, 0), min(c1, w))

    def solid(self, x0, y0, x1, y1):
        self.grid[self.cells(x0, y0, x1, y1)] = True

    def clear(self, x0, y0, x1, y1):
        self.grid[self.cells(x0, y0, x1, y1)] = False

    def furniture(self, x0, y0, x1, y1) -> bool:
        """Place furniture unless it would touch a door approach. Returns True if placed."""
        sl = self.cells(x0, y0, x1, y1)
        if self.keepout[sl].any():
            return False
        self.grid[sl] = True
        return True


def _split(rng, total, n, min_w):
    """Split [0, total] into n widths >= min_w, randomly."""
    spare = total - n * min_w
    cuts = np.sort(rng.uniform(0, spare, size=n - 1)) if n > 1 else np.array([])
    edges = np.concatenate([[0.0], cuts, [spare]])
    return np.diff(edges) + min_w


def _furnish(p: _Painter, room: Room, rng, back_is_top: bool, ward_letter: str):
    """Add furniture to a room. `back_is_top`: the wall opposite the door is at the top."""
    x0, y0, x1, y1 = room.rect
    depth_dir = -1 if back_is_top else 1          # direction from the back wall towards the door
    back = y1 if back_is_top else y0

    def band(d0, d1):
        """y-interval from `d0` to `d1` metres measured from the back wall."""
        a, b = back + depth_dir * d0, back + depth_dir * d1
        return min(a, b), max(a, b)

    if room.kind == "ward":
        bed_w, bed_l, gap = 0.9, 2.0, 0.8
        n_beds = int((x1 - x0 - 0.4 + gap) // (bed_w + gap))
        used = n_beds * bed_w + (n_beds - 1) * gap
        bx = x0 + (x1 - x0 - used) / 2
        for i in range(n_beds):
            ya, yb = band(0.15, 0.15 + bed_l)
            if p.furniture(bx, ya, bx + bed_w, yb):
                # delivery point: 0.6 m beyond the foot of the bed, on its centre line
                fy = band(0.15 + bed_l + 0.6, 0.15 + bed_l + 0.6)[0]
                room.beds.append((f"{ward_letter}-{len(room.beds) + 1}", (bx + bed_w / 2, fy)))
            bx += bed_w + gap
    elif room.kind == "pharmacy":
        ya, yb = band(0.0, 0.5)
        p.furniture(x0 + 0.3, ya, x1 - 0.3, yb)                       # shelves on the back wall
        cx = (x0 + x1) / 2
        ya, yb = band(1.6, 2.2)
        p.furniture(cx - 0.9, ya, cx + 0.9, yb)                       # dispensing counter
    elif room.kind == "supply":
        ya, yb = band(0.0, 0.6)
        p.furniture(x0 + 0.2, ya, x1 - 0.2, yb)
        if x1 - x0 > 3.5:
            ya, yb = band(1.6, 2.2)
            p.furniture(x0 + 0.8, ya, (x0 + x1) / 2 - 0.4, yb)        # free-standing rack
    elif room.kind == "nurses":
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        p.furniture(cx - 1.0, cy - 0.5 + 0.4 * depth_dir, cx + 1.0, cy + 0.5 + 0.4 * depth_dir)
    elif room.kind == "lobby":
        for d in (0.8, 2.2):                                          # two rows of chairs
            ya, yb = band(d, d + 0.5)
            x = x0 + 0.6
            while x + 0.5 < x1 - 0.6:
                p.furniture(x, ya, x + 0.5, yb)
                x += 0.8


def _assign_kinds(rng, n_rooms):
    """Pharmacy + nurse station always; >= 2 wards; lobby/supply when there is room."""
    kinds = ["pharmacy", "nurses"]
    extras = []
    if n_rooms >= 7:
        extras.append("lobby")
    if n_rooms >= 8:
        extras.append("supply")
    kinds += extras
    kinds += ["ward"] * (n_rooms - len(kinds))
    rng.shuffle(kinds)
    return kinds


# ------------------------------------------------------------------------- main generator
def generate_layout(
    seed: int,
    difficulty: int = 1,
    width_m: float = 24.0,
    height_m: float = 16.0,
    resolution: float = 0.25,
    robot_radius: float = 0.2,
    n_obstacles: int | None = None,
    n_humans: int | None = None,
    max_tries: int = 50,
) -> Layout:
    """Deterministically build a hospital floor + one delivery mission from `seed`.

    `n_obstacles` (corridor trolleys) / `n_humans` override the difficulty preset; this
    is how the robustness tests create harder-than-training conditions.
    """
    return _generate_cached(int(seed), int(difficulty), float(width_m), float(height_m), float(resolution),
                            float(robot_radius), n_obstacles, n_humans, int(max_tries))


@lru_cache(maxsize=256)
def _generate_cached(seed, difficulty, width_m, height_m, resolution, robot_radius, n_obstacles, n_humans,
                     max_tries) -> Layout:
    # Cached because evaluation replays the same floors for every agent; the Layout is never
    # modified after creation (people keep their own positions in humans.py).
    spec = dict(DIFFICULTY[int(np.clip(difficulty, 0, max(DIFFICULTY)))])
    if n_obstacles is not None:
        spec["n_obstacles"] = n_obstacles
    if n_humans is not None:
        spec["n_humans"] = n_humans
    rng = np.random.default_rng(seed)
    rows, cols = int(round(height_m / resolution)), int(round(width_m / resolution))
    wall = resolution                                    # walls are one cell thick
    inflate_cells = int(np.ceil(robot_radius / resolution))

    for _ in range(max_tries):
        grid = np.zeros((rows, cols), dtype=bool)
        p = _Painter(grid, resolution)
        p.solid(0, 0, width_m, wall)
        p.solid(0, height_m - wall, width_m, height_m)
        p.solid(0, 0, wall, height_m)
        p.solid(width_m - wall, 0, width_m, height_m)

        # --- corridor: 2.0-2.75 m wide, centre jittered +-1 m
        cw = float(rng.choice([2.0, 2.25, 2.5, 2.75]))
        cy = height_m / 2 + float(rng.uniform(-1.0, 1.0))
        cy0 = round((cy - cw / 2) / resolution) * resolution
        cy1 = cy0 + cw
        corridor = (wall, cy0, width_m - wall, cy1)
        p.solid(0, cy0 - wall, width_m, cy0)            # wall below the corridor
        p.solid(0, cy1, width_m, cy1 + wall)            # wall above the corridor

        # --- rooms on both sides
        rooms_geo = []                                  # (x0, x1, y0, y1, back_is_top)
        for side_top in (True, False):
            inner_w = width_m - 2 * wall
            n = int(rng.integers(3, 6))
            n = min(n, int(inner_w // 4.0))
            widths = _split(rng, inner_w, n, 4.0)
            x = wall
            for i, w in enumerate(widths):
                x0, x1 = x, x + w
                x = x1
                if i < n - 1:                           # partition wall between neighbours
                    xs = round(x1 / resolution) * resolution
                    if side_top:
                        p.solid(xs - wall, cy1, xs, height_m)
                    else:
                        p.solid(xs - wall, 0, xs, cy0)
                    x1 = xs - wall
                if side_top:
                    rooms_geo.append((x0, x1, cy1 + wall, height_m - wall, True))
                else:
                    rooms_geo.append((x0, x1, wall, cy0 - wall, False))

        kinds = _assign_kinds(rng, len(rooms_geo))
        rooms, ward_i = [], 0
        for (x0, x1, y0, y1, top), kind in zip(rooms_geo, kinds):
            # doorway 1.0-1.25 m on the corridor wall, >= 0.4 m from the room corners
            dw = float(rng.choice([1.0, 1.25]))
            dx0 = round(float(rng.uniform(x0 + 0.4, x1 - 0.4 - dw)) / resolution) * resolution
            door_y = cy1 if top else cy0 - wall
            p.clear(dx0, door_y, dx0 + dw, door_y + wall)
            # keep 1.2 m on both sides of the doorway free of furniture and trolleys
            p.keepout[p.cells(dx0 - 0.25, door_y - 1.2, dx0 + dw + 0.25, door_y + wall + 1.2)] = True
            if kind == "ward":
                letter = "ABCDEFGH"[ward_i]
                ward_i += 1
                name = f"Ward {letter}"
            else:
                letter = ""
                name = {"pharmacy": "Pharmacy", "nurses": "Nurse station", "lobby": "Lobby",
                        "supply": "Supply"}[kind]
            rooms.append(Room(name, kind, (x0, y0, x1, y1), (dx0, dx0 + dw, door_y + wall / 2)))
            rooms[-1]._top, rooms[-1]._letter = top, letter

        for room in rooms:
            _furnish(p, room, rng, back_is_top=room._top, ward_letter=room._letter)

        # --- trolleys parked against the corridor walls (0.6 x 1.0 m)
        trolleys = []
        for _ in range(spec["n_obstacles"] * 10):
            if len(trolleys) >= spec["n_obstacles"]:
                break
            tx = float(rng.uniform(wall + 0.3, width_m - wall - 1.3))
            along_top = rng.random() < 0.5
            ty0 = cy1 - 0.6 if along_top else cy0
            box = (tx, ty0, tx + 1.0, ty0 + 0.6)
            if p.furniture(*box):
                trolleys.append(box)
                p.keepout[p.cells(tx - 1.0, cy0, tx + 2.0, cy1)] = True   # trolleys never block in pairs

        inflated = inflate(grid, inflate_cells)
        free = ~inflated

        # --- start: inside the pharmacy, 1.6 m in from its doorway (in front of the counter)
        pharmacy = next(r for r in rooms if r.kind == "pharmacy")
        py0, py1 = pharmacy.rect[1], pharmacy.rect[3]
        door_x = (pharmacy.door[0] + pharmacy.door[1]) / 2
        start_cell = _nearest_free(free, np.array([door_x, _towards(pharmacy.door[2], (py0 + py1) / 2, 1.6)]),
                                   resolution)
        if start_cell is None:
            continue

        # --- goal: a bed in a ward, chosen by route length according to difficulty
        field_from_start = geodesic_field(inflated, start_cell)
        candidates = []
        for room in rooms:
            for label, pt in room.beds:
                cell = _nearest_free(free, np.asarray(pt), resolution, max_r=3)
                if cell is not None and np.isfinite(field_from_start[cell]):
                    candidates.append((field_from_start[cell], room.name, label, cell))
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda c: c[0])
        lo, hi = spec["goal_rank"]
        i0 = int(np.floor(lo * (len(candidates) - 1)))
        i1 = max(i0, int(np.ceil(hi * (len(candidates) - 1))))
        _, goal_room, goal_label, goal_cell = candidates[int(rng.integers(i0, i1 + 1))]

        layout = Layout(
            grid=grid, resolution=resolution,
            start=np.array([(start_cell[1] + 0.5) * resolution, (start_cell[0] + 0.5) * resolution]),
            start_heading=float(rng.uniform(-np.pi, np.pi)),
            goal=np.array([(goal_cell[1] + 0.5) * resolution, (goal_cell[0] + 0.5) * resolution]),
            seed=seed, difficulty=difficulty,
            geodesic=geodesic_field(inflated, goal_cell), inflated=inflated,
            rooms=rooms, corridor=corridor, trolleys=trolleys,
            start_room=pharmacy.name, goal_room=goal_room, goal_label=goal_label,
        )

        # --- people: mostly in the corridor (staff walking between rooms), some in rooms;
        #     never within 1.5 m of the robot start or 1 m of the delivery point
        reachable = np.argwhere(np.isfinite(layout.geodesic))
        in_corr = [c for c in reachable if layout.room_at(layout.centre(tuple(c))) == "Corridor"]
        spawns = []
        for _ in range(spec["n_humans"] * 30):
            if len(spawns) >= spec["n_humans"]:
                break
            pool = in_corr if (rng.random() < 0.7 and in_corr) else reachable
            xy = layout.centre(tuple(pool[rng.integers(len(pool))]))
            if (np.linalg.norm(xy - layout.start) > 1.5 and np.linalg.norm(xy - layout.goal) > 1.0
                    and all(np.linalg.norm(xy - s) > 0.8 for s in spawns)):
                spawns.append(xy)
        layout.human_spawns = spawns
        for room in rooms:                          # drop helper attributes (keep Room a clean record)
            del room._top, room._letter
        return layout
    raise RuntimeError(f"Could not generate a valid hospital floor for seed {seed}")


def _towards(door_y, centre_y, dist):
    """A y coordinate `dist` metres from the doorway, into the room."""
    return door_y + np.sign(centre_y - door_y) * dist


def _nearest_free(free: np.ndarray, xy: np.ndarray, res: float, max_r: int = 8):
    """Free (non-inflated) cell closest to xy, searching up to max_r cells away."""
    r, c = int(xy[1] / res), int(xy[0] / res)
    h, w = free.shape
    best, best_d = None, np.inf
    for dr in range(-max_r, max_r + 1):
        for dc in range(-max_r, max_r + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and free[rr, cc]:
                d = dr * dr + dc * dc
                if d < best_d:
                    best, best_d = (rr, cc), d
    return best
