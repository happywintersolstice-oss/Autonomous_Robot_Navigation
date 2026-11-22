"""Explicit reward derivations, reproducible search candidates and ranking."""
from dataclasses import replace
import itertools
import random

import numpy as np

from .core import read_json


def reward_task(base, variant="step-cost"):
    if variant == "step-cost":
        return base
    if variant != "terminal-only":
        raise ValueError(f"Unsupported reward variant: {variant}")
    return replace(base, config={**base.config, "step_reward": 0., "reward_version": "terminal-only-v1"})


def validate_task_parent(task, parent):
    if not any(task.signature == reward_task(parent, variant).signature for variant in ("step-cost", "terminal-only")):
        raise ValueError("Task is not an approved reward-only derivation of the scenario task")


def task_for_run(base, directory):
    configuration = read_json(directory / "configuration.json")
    task = reward_task(base, configuration["training"].get("reward_variant", "step-cost"))
    task.check(read_json(directory / "manifest.json")["signature"])
    if configuration["environment"] != task.config:
        raise ValueError("Run environment does not match the approved reward variant")
    return task


def search_candidates(plan):
    keys = list(plan["search_space"])
    grid = [dict(zip(keys, values)) for values in itertools.product(*(plan["search_space"][k] for k in keys))]
    selected = random.Random(plan["search_seed"]).sample(grid, plan["search_candidates"])
    return [{"candidate_id": f"c{i:02d}", "parameters": values} for i, values in enumerate(selected, 1)]


def selection_rank(rows, identity):
    return (-float(np.mean([r["validation"]["success_rate"] for r in rows])),
            -float(np.mean([r["validation"]["spl"] for r in rows])),
            float(np.std([r["validation"]["success_rate"] for r in rows], ddof=1)), identity)
