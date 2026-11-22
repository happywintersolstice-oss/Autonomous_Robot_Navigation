"""New development-only layouts/schedules; no final-set creation or accessor."""

from collections import deque
from pathlib import Path
import random

import numpy as np

from .core import ROOT, GridMap, digest, read_json, write_json
from .release import sha
from .partial_contract import (
    BENCHMARK,
    MOVES,
    configuration,
    signature,
    validate_config,
)
from .partial_dynamic_env import make_scenario, validate_scenario

GENERATOR = "partial-connected-d4-development-v1"
CONDITIONS = ("familiar-static", "unseen-static", "familiar-dynamic", "unseen-dynamic")


def geometry_hash(rows):
    array = np.array([list(row) for row in rows])
    return digest(
        min(
            tuple("".join(row) for row in variant)
            for rotation in range(4)
            for variant in (
                np.rot90(array, rotation),
                np.fliplr(np.rot90(array, rotation)),
            )
        )
    )


def historical_exclusions(root=ROOT):
    """Hash already inspected historical geometry, without evaluating old cohorts."""
    sources, identities = {}, set()

    def visit(value):
        if isinstance(value, dict):
            rows = value.get("rows")
            if (
                isinstance(rows, list)
                and len(rows) == 10
                and all(
                    isinstance(r, str) and len(r) == 10 and set(r) <= {".", "#"}
                    for r in rows
                )
            ):
                identities.add(geometry_hash(rows))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for folder in ("data/maps", "data/unseen/unseen-maps-v1", "data/navigation-lab"):
        for path in sorted((root / folder).rglob("*.json")):
            visit(read_json(path))
            sources[path.relative_to(root).as_posix()] = sha(path)
    return identities, sources


def event_schedule(grid, start, goal, seed, config):
    """Nominal schedule uses only its declared seed/geometry, never agent actions."""
    rng = random.Random(seed)
    path = grid.path(start, goal)
    available = [c for c in grid.free_cells if c not in (tuple(start), tuple(goal))]
    preferred = path[1:-1]
    cells = rng.sample(
        available, min(config["events_per_dynamic_route"], len(available))
    )
    if preferred and cells:
        chosen = preferred[min(1, len(preferred) - 1)]
        cells = [chosen] + [c for c in cells if c != chosen]
        cells = cells[: config["events_per_dynamic_route"]]
    low, high = config["event_duration"]
    if not (type(low) is int and type(high) is int and 1 <= low <= high <= 50):
        raise ValueError("Invalid provisional event duration range")
    return [
        {
            "id": f"event-{i}",
            "cell": list(cell),
            "start": (on := rng.randint(0, 12) + i * 6),
            "clear": on + rng.randint(low, high),
        }
        for i, cell in enumerate(cells)
    ]


def path_without(grid, start, goal, blocked):
    """Trusted preparation/evaluation utility, never imported by a policy."""
    todo, seen = deque([(tuple(start), 0)]), {tuple(start)}
    while todo:
        cell, length = todo.popleft()
        if cell == tuple(goal):
            return length
        for dr, dc in MOVES[:4]:
            nxt = (cell[0] + dr, cell[1] + dc)
            if grid.is_free(nxt) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                todo.append((nxt, length + 1))
    return None


def layout_statistics(grid):
    degrees = {
        c: sum(grid.is_free((c[0] + dr, c[1] + dc)) for dr, dc in MOVES[:4])
        for c in grid.free_cells
    }
    return {
        "free_cells": len(degrees),
        "dead_ends": sum(d == 1 for d in degrees.values()),
        "junctions": sum(d >= 3 for d in degrees.values()),
        "diameter": max(max(grid.distances(c).values()) for c in grid.free_cells),
    }


def prepare(root=ROOT, config=None):
    root = Path(root)
    config = validate_config(config or configuration(root))
    if set(config["layout_counts"]) != {"train", "validation"}:
        raise ValueError("Only training/development layouts may be generated")
    if any(type(v) is not int or v < 1 for v in config["layout_counts"].values()):
        raise ValueError("Positive layout counts required")
    if (
        type(config["routes_per_layout"]) is not int
        or config["routes_per_layout"] < 3
        or config["routes_per_layout"] % 3
        or type(config["events_per_dynamic_route"]) is not int
        or not 1 <= config["events_per_dynamic_route"] <= 6
        or type(config["blocked_cells"]) is not int
        or not 1 <= config["blocked_cells"] <= 50
    ):
        raise ValueError("Invalid route/event/obstacle generation counts")
    folder = root / "data/partial" / BENCHMARK
    if folder.exists():
        reader = DevelopmentBenchmark(root)
        if reader.record["configuration"] != config:
            raise ValueError(
                "Existing development configuration changed; use a new benchmark version"
            )
        reader.audit()
        return reader.record
    excluded, sources = historical_exclusions(root)
    occupied = set(excluded)
    rng = random.Random(config["generation_seed"])
    splits, attempts = {}, 0
    for split, count in config["layout_counts"].items():
        layouts = []
        while len(layouts) < count:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError("New layout rejection limit exceeded")
            walls = set(rng.sample(range(100), config["blocked_cells"]))
            rows = [
                "".join("#" if r * 10 + c in walls else "." for c in range(10))
                for r in range(10)
            ]
            identity = geometry_hash(rows)
            if identity in occupied:
                continue
            grid = GridMap(f"{BENCHMARK}-{split}-{len(layouts)+1:03}", tuple(rows))
            if len(grid.distances(grid.free_cells[0])) != len(grid.free_cells):
                continue
            stats = layout_statistics(grid)
            if stats["diameter"] < config["minimum_diameter"]:
                continue
            bands = {"short": [], "medium": [], "long": []}
            for start in grid.free_cells:
                for goal, length in grid.distances(start).items():
                    if length:
                        bands[
                            (
                                "short"
                                if length <= 6
                                else "medium" if length <= 12 else "long"
                            )
                        ].append((start, goal, length))
            if not all(bands.values()):
                continue
            routes = []
            for i in range(config["routes_per_layout"]):
                band = ("short", "medium", "long")[i % 3]
                pair = rng.choice(bands[band])
                if any((r["start"], r["goal"]) == pair[:2] for r in routes):
                    choices = [
                        p
                        for p in bands[band]
                        if all((r["start"], r["goal"]) != p[:2] for r in routes)
                    ]
                    pair = rng.choice(choices)
                routes.append(
                    dict(start=pair[0], goal=pair[1], base_length=pair[2], band=band)
                )
            cases = []
            for i, route in enumerate(routes):
                seed = rng.randrange(2**31)
                events = event_schedule(
                    grid, route["start"], route["goal"], seed, config
                )
                impact = [
                    path_without(
                        grid, route["start"], route["goal"], {tuple(e["cell"])}
                    )
                    for e in events
                ]
                for dynamic in (False, True):
                    condition = ("familiar" if split == "train" else "unseen") + (
                        "-dynamic" if dynamic else "-static"
                    )
                    case = make_scenario(
                        rows,
                        route["start"],
                        route["goal"],
                        events if dynamic else [],
                        scenario_id=f"{grid.map_id}-route{i:02}-{condition}",
                        layout_id=grid.map_id,
                        event_seed=seed,
                        condition=condition,
                        split=split,
                    )
                    case["pair_id"] = f"{grid.map_id}-route{i:02}"
                    case["difficulty"] = {
                        "distance_band": route["band"],
                        "base_length": route["base_length"],
                        "single_event_detour_lengths": impact if dynamic else [],
                        **stats,
                    }
                    cases.append(case)
            layouts.append(
                dict(
                    layout_id=grid.map_id,
                    rows=rows,
                    geometry_hash=identity,
                    statistics=stats,
                    scenarios=cases,
                )
            )
            occupied.add(identity)
        splits[split] = {"benchmark_id": BENCHMARK, "split": split, "layouts": layouts}
    folder.mkdir(parents=True, exist_ok=False)
    for split, record in splits.items():
        write_json(folder / f"{split}.json", record)
    record = {
        "benchmark_id": BENCHMARK,
        "generator_version": GENERATOR,
        "configuration": config,
        "task_signature": signature(config),
        "configuration_hash": digest(config),
        "files": {f"{s}.json": sha(folder / f"{s}.json") for s in splits},
        "generation_attempts": attempts,
        "historical_excluded_geometry_hashes": sorted(excluded),
        "historical_sources": sources,
        "generator_sha256": sha(Path(__file__)),
        "final_status": "Not generated; ten fresh final layouts proposed; separate authorization/freeze required",
    }
    write_json(folder / "manifest.json", record)
    DevelopmentBenchmark(root).audit()
    return record


class DevelopmentBenchmark:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.folder = self.root / "data/partial" / BENCHMARK
        self.record = read_json(self.folder / "manifest.json")
        self.config = validate_config(self.record["configuration"])
        if (
            self.record["benchmark_id"] != BENCHMARK
            or self.record["generator_version"] != GENERATOR
            or self.record["configuration_hash"] != digest(self.config)
            or self.record["task_signature"] != signature(self.config)
            or self.record["generator_sha256"] != sha(Path(__file__))
        ):
            raise ValueError("Partial benchmark identity/configuration/source mismatch")

    def layouts(self, split):
        if split not in ("train", "validation"):
            raise ValueError("Final layouts are not generated or accessible")
        name = f"{split}.json"
        if sha(self.folder / name) != self.record["files"][name]:
            raise ValueError("Partial split checksum mismatch")
        record = read_json(self.folder / name)
        if record["benchmark_id"] != BENCHMARK or record["split"] != split:
            raise ValueError("Partial split identity mismatch")
        return record["layouts"]

    def scenarios(self, split, dynamic=None):
        return [
            case
            for layout in self.layouts(split)
            for case in layout["scenarios"]
            if dynamic is None or bool(case["events"]) == dynamic
        ]

    def audit(self):
        identities = set(self.record["historical_excluded_geometry_hashes"])
        count, case_ids = 0, set()
        for name, value in self.record["historical_sources"].items():
            if sha(self.root / name) != value:
                raise ValueError("Historical exclusion input changed")
        for split in ("train", "validation"):
            layouts = self.layouts(split)
            if len(layouts) != self.config["layout_counts"][split]:
                raise ValueError("Wrong layout count")
            for layout in layouts:
                identity = geometry_hash(layout["rows"])
                if identity in identities or identity != layout["geometry_hash"]:
                    raise ValueError("Duplicate/D4-equivalent geometry")
                identities.add(identity)
                if len(layout["scenarios"]) != self.config["routes_per_layout"] * 2:
                    raise ValueError("Wrong per-layout scenario count")
                for case in layout["scenarios"]:
                    grid = validate_scenario(case)
                    if list(grid.rows) != layout["rows"] or case["split"] != split:
                        raise ValueError("Scenario layout/split mismatch")
                    if case["id"] in case_ids:
                        raise ValueError("Duplicate scenario identity")
                    case_ids.add(case["id"])
                    dynamic = case["condition"].endswith("-dynamic")
                    condition = ("familiar" if split == "train" else "unseen") + (
                        "-dynamic" if dynamic else "-static"
                    )
                    expected = (
                        event_schedule(
                            grid,
                            case["start"],
                            case["goal"],
                            case["event_seed"],
                            self.config,
                        )
                        if dynamic
                        else []
                    )
                    if case["condition"] != condition or case["events"] != expected:
                        raise ValueError(
                            "Condition/schedule does not match declared split/seed"
                        )
                    count += 1
        return {
            "layouts": len(identities)
            - len(self.record["historical_excluded_geometry_hashes"]),
            "scenarios": count,
            "final_generated": False,
        }
