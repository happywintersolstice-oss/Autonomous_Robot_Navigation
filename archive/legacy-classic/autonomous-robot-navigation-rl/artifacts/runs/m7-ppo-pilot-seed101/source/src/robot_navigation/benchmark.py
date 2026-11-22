"""Deterministic, inspectable scenario preparation. No policy execution here."""
from collections import Counter, defaultdict
import random

from .core import ROOT, digest, load_task, read_json, write_json

GENERATOR_VERSION = "goal-distance-stratified-v1"


def distance_band(length):
    return "short" if length <= 5 else "medium" if length <= 10 else "long"


def prepare(root=ROOT):
    task = load_task(root)
    grid = task.grid
    if grid.shape != (10, 10) or len(grid.free_cells) < 30:
        raise ValueError("Benchmark requires a 10x10 map and >=30 free cells")
    remaining, components = set(grid.free_cells), []
    while remaining:
        component = set(grid.distances(min(remaining)))
        components.append(len(component))
        remaining -= component
    if len(components) != 1:
        raise ValueError("Benchmark free cells must form one connected component")
    groups = defaultdict(list)
    histogram = Counter()
    for goal in grid.free_cells:
        for start, distance in sorted(grid.distances(goal).items()):
            if start == goal:
                continue
            histogram[distance] += 1
            groups[(goal, distance_band(distance))].append({
                "id": f"r{start[0]}c{start[1]}-r{goal[0]}c{goal[1]}",
                "start": list(start), "goal": list(goal), "shortest_length": distance,
                "distance_band": distance_band(distance)})
    rng = random.Random(task.config["split_seed"])
    splits = {k: [] for k in ("train", "validation", "test")}
    for group in groups.values():
        rng.shuffle(group)
        # Rounding is per goal/distance stratum; small strata stay in training.
        n_holdout = int(len(group) * .15 + .5) if len(group) >= 4 else 0
        n_train = len(group) - 2 * n_holdout
        splits["train"].extend(group[:n_train])
        splits["validation"].extend(group[n_train:n_train + n_holdout])
        splits["test"].extend(group[n_train + n_holdout:])
    for split, scenarios in splits.items():
        scenarios.sort(key=lambda s: s["id"])
        write_json(root / f"data/scenarios/{split}.json", {
            "signature": task.signature, "generator_version": GENERATOR_VERSION,
            "split_seed": task.config["split_seed"], "split": split, "scenarios": scenarios})
    # Distance-stratified round robin, fixed before training, <=50 validation tasks.
    buckets = [[s for s in splits["validation"] if s["distance_band"] == band]
               for band in ("short", "medium", "long")]
    for bucket in buckets:
        rng.shuffle(bucket)
    monitoring = []
    limit = min(50, task.config["monitoring_count"], len(splits["validation"]))
    while len(monitoring) < limit:
        for bucket in buckets:
            if bucket and len(monitoring) < limit:
                monitoring.append(bucket.pop())
    write_json(root / "data/scenarios/monitoring.json", {
        "signature": task.signature, "split": "validation", "scenarios": monitoring})
    report = {"signature": task.signature, "generator_version": GENERATOR_VERSION,
              "status": "provisional benchmark; main-study freeze requires learning-pilot review",
              "free_cells": len(grid.free_cells), "blocked_cells": 100 - len(grid.free_cells),
              "component_sizes": components, "distance_bins": {"short": "1..5", "medium": "6..10", "long": ">=11"},
              "shortest_length_histogram": dict(sorted(histogram.items())),
              "split_seed": task.config["split_seed"], "monitoring_ids": [s["id"] for s in monitoring],
              "splits": {name: {"count": len(items), "fraction": len(items) / sum(histogram.values()),
                                "bands": dict(Counter(s["distance_band"] for s in items)),
                                "goal_frequencies": dict(Counter(f"{s['goal'][0]},{s['goal'][1]}" for s in items)),
                                "scenario_hash": digest(items)} for name, items in splits.items()}}
    write_json(root / "data/scenarios/benchmark_report.json", report)
    return report


def load_scenarios(task, split, root=ROOT):
    # No final-test evaluation entrypoint is shipped in milestone 1.
    if split not in ("train", "validation", "monitoring"):
        raise ValueError("Final test is sealed in this milestone; only train/validation are permitted")
    record = read_json(root / f"data/scenarios/{split}.json")
    task.check(record["signature"])
    return record["scenarios"]
