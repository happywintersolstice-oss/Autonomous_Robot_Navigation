"""Read-only evaluation, shared by training monitoring and the inference CLI."""
from collections import defaultdict
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .agents import QLearning
from .core import ROOT, NavigationEnv, digest
from .episodes import aggregate, rollout


def source_hash(root):
    sources = {}
    for folder, extension in (("src", "*.py"), ("app", "*.py"), ("configs", "*.yaml")):
        for path in sorted((root / folder).rglob(extension)):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(sources)


def evaluate(task, scenarios, policy, seed, scope, visitation=None, trajectory_path=None):
    env = NavigationEnv(task, scenarios)
    fingerprint = getattr(policy, "state_fingerprint", None)
    before = fingerprint() if fingerprint else policy.q.copy() if isinstance(policy, QLearning) else None
    metrics, records, decision_times = [], [], []
    by_band = defaultdict(list)
    started = perf_counter()
    with ExitStack() as stack:
        output = None
        if trajectory_path is not None:
            path = Path(trajectory_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(gzip.open(path, "xt", encoding="utf-8"))
        for scenario in scenarios:
            episode = rollout(env, policy, scenario, seed)
            if visitation is not None:
                episode["initial_state_visited_in_training"] = bool(visitation[tuple(episode["initial_observation"])].sum())
            metrics.append({"scenario_id": scenario["id"], **episode["metrics"]})
            by_band[scenario.get("distance_band", "fixture")].append(episode["metrics"])
            outcome = episode["metrics"]["outcome"]
            if outcome not in [r["metrics"]["outcome"] for r in records]:
                records.append(episode)
            if output:
                output.write(json.dumps(episode, allow_nan=False) + "\n")
    elapsed = perf_counter() - started
    if before is not None and ((fingerprint and fingerprint() != before) or
                               (not fingerprint and not np.array_equal(before, policy.q))):
        raise AssertionError("Validation changed learned parameters")
    observation = np.array((*scenarios[0]["start"], *scenarios[0]["goal"]), dtype=np.int64)
    for _ in range(100):
        start = perf_counter()
        policy.act(observation)
        decision_times.append(perf_counter() - start)
    summary = {**aggregate(metrics), "scope": scope, "seed": seed,
               "by_distance_band": {band: aggregate(items) for band, items in by_band.items()},
               "evaluation_seconds": elapsed, "interactions_per_second": sum(m["steps"] for m in metrics) / elapsed,
               "mean_decision_microseconds": float(np.mean(decision_times) * 1e6),
               "timing_method": "perf_counter; includes reset/recording and optional gzip export, excludes env construction/rendering; decision mean over 100 queries"}
    if visitation is not None:
        overlap = sum(bool(visitation[(*s["start"], *s["goal"])].sum()) for s in scenarios)
        summary["initial_states_visited_in_training"] = overlap
        summary["initial_state_overlap_fraction"] = overlap / len(scenarios)
    return summary, metrics, records


def train_smoke(config_path=None, root=ROOT):
    """Compatibility entrypoint; both algorithms use the same training loop."""
    from .training import train_tabular
    return train_tabular(config_path=config_path, root=root)
