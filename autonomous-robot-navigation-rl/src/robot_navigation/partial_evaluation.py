"""Development-only evaluator. Privileged diagnostics are confined to this module."""

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from .core import GridMap, digest
from .release import sha
from .partial_benchmark import CONDITIONS, path_without
from .partial_contract import MOVES
from .partial_dynamic_env import PartialDynamicEnv


def code_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("partial_*.py"))}


def record_episode(case, policy, config=None, evaluation_seed=0):
    if case.get("split") not in ("train", "validation", "manual"):
        raise ValueError("Only development/manual scenarios may be evaluated")
    env = PartialDynamicEnv([case], config)
    observation, _ = env.reset(seed=evaluation_seed)
    fingerprint = getattr(policy, "state_fingerprint", lambda: None)
    before = fingerprint()
    record = {
        "schema_version": 1,
        "task_signature": env.signature,
        "configuration": env.config,
        "scenario": deepcopy(case),
        "scenario_hash": digest(case),
        "evaluation_seed": evaluation_seed,
        "policy_id": policy.policy_id,
        "algorithm": policy.algorithm,
        "training_seed": getattr(policy, "training_seed", None),
        "checkpoint_sha256": getattr(policy, "checkpoint_sha256", None),
        "parameter_fingerprint": before,
        "code_hashes": code_hashes(),
        "initial_observation": observation.tolist(),
        "initial_snapshot": env.diagnostic_snapshot(),
        "transitions": [],
        "inference_seconds": 0.0,
    }
    while True:
        started = perf_counter()
        # The sole policy call receives a copy of permitted information. No env,
        # scenario ID, evaluator info, event seed or snapshot is supplied.
        action = policy.act(observation.copy())
        record["inference_seconds"] += perf_counter() - started
        nxt, reward, terminal, cutoff, info = env.step(action)
        record["transitions"].append(
            {
                "observation": observation.tolist(),
                "action": int(action),
                "reward": reward,
                "next_observation": nxt.tolist(),
                "terminated": terminal,
                "truncated": cutoff,
                "info": info,
                "snapshot": env.diagnostic_snapshot(),
            }
        )
        observation = nxt
        if terminal or cutoff:
            break
    if before != fingerprint():
        raise ValueError("Evaluation changed learned parameters")
    record["metrics"] = episode_metrics(record)
    return record


def episode_metrics(record):
    transitions = record["transitions"]
    if not transitions or not (
        transitions[-1]["terminated"] or transitions[-1]["truncated"]
    ):
        raise ValueError("A complete episode is required")
    outcome = transitions[-1]["info"]["outcome"]
    if outcome not in ("success", "collision", "timeout"):
        raise ValueError("Invalid outcome")
    snapshots = [record["initial_snapshot"]] + [t["snapshot"] for t in transitions]
    positions = [tuple(s["position"]) for s in snapshots]
    seen, repeats, moved = {positions[0]}, 0, 0
    waits = longest_wait = streak = no_progress = longest_no_progress = 0
    best_distance = sum(
        abs(a - b) for a, b in zip(positions[0], record["scenario"]["goal"])
    )
    posthoc_open_waits = 0
    grid = GridMap(record["scenario"]["layout_id"], tuple(record["scenario"]["rows"]))
    for i, transition in enumerate(transitions):
        waiting = transition["action"] == 4
        waits += waiting
        streak = streak + 1 if waiting else 0
        longest_wait = max(longest_wait, streak)
        if positions[i + 1] != positions[i]:
            moved += 1
            repeats += positions[i + 1] in seen
            seen.add(positions[i + 1])
        distance = sum(
            abs(a - b) for a, b in zip(positions[i + 1], record["scenario"]["goal"])
        )
        no_progress = no_progress + 1 if distance >= best_distance else 0
        best_distance = min(best_distance, distance)
        longest_no_progress = max(longest_no_progress, no_progress)
        if waiting:
            blocked = {
                tuple(map(int, cell)) for cell in np.argwhere(snapshots[i]["occupancy"])
            }
            posthoc_open_waits += (
                path_without(grid, positions[i], record["scenario"]["goal"], blocked)
                is not None
            )
    events = {e["id"]: e for e in record["scenario"]["events"]}
    detected, relevant, recovered = {}, {}, {}
    for snapshot in snapshots:
        observed_blocked = {
            (r, c) for r, c, blocked in snapshot["scan"]["cells"] if blocked
        }
        active = snapshot["event_log"][-1]["applied"]
        path = set(grid.path(snapshot["position"], record["scenario"]["goal"]))
        for event_id in active:
            cell = tuple(events[event_id]["cell"])
            if cell in observed_blocked:
                detected.setdefault(event_id, snapshot["step"])
                if cell in path:
                    relevant.setdefault(event_id, snapshot["step"])
        for event_id, step in relevant.items():
            if snapshot["step"] > step and (
                tuple(snapshot["position"]) == tuple(events[event_id]["cell"])
                or snapshot["outcome"] == "success"
            ):
                recovered.setdefault(event_id, snapshot["step"] - step)
    final = snapshots[-1]
    applied = {e for tick in final["event_log"] for e in tick["applied"]}
    base_length = (
        len(grid.path(record["scenario"]["start"], record["scenario"]["goal"])) - 1
    )
    return {
        "outcome": outcome,
        "actions": len(transitions),
        "moves": moved,
        "return": float(sum(t["reward"] for t in transitions)),
        "wait_count": int(waits),
        "wait_proportion": waits / len(transitions),
        "longest_wait_streak": longest_wait,
        "repeated_movement_visits": int(repeats),
        "two_cell_oscillation_windows": sum(
            positions[i] == positions[i + 2]
            and positions[i + 1] == positions[i + 3]
            and positions[i] != positions[i + 1]
            for i in range(len(positions) - 3)
        ),
        "longest_no_manhattan_improvement": longest_no_progress,
        "discovered_cell_count": int(
            np.count_nonzero(np.array(final["memory"]["occupancy"]) != -1)
        ),
        "discovered_grid_fraction": float(
            np.mean(np.array(final["memory"]["occupancy"]) != -1)
        ),
        "sensor_cell_refreshes": sum(
            s["scan_update"]["sensed_cells"] for s in snapshots
        ),
        "occupancy_corrections": sum(
            s["scan_update"]["occupancy_corrections"] for s in snapshots
        ),
        "nominal_event_count": len(events),
        "applied_event_count": len(applied),
        "applied_event_ticks": sum(len(t["applied"]) for t in final["event_log"]),
        "suppressed_activation_ticks": sum(
            len(t["suppressed"]) for t in final["event_log"]
        ),
        "detected_event_count": len(detected),
        "events_never_observed": len(events) - len(detected),
        "applied_events_never_observed": len(applied - set(detected)),
        "recovery_opportunities": len(relevant),
        "recoveries": len(recovered),
        "observed_to_recovery_delays": list(recovered.values()),
        "posthoc_waits_with_open_route": int(posthoc_open_waits),
        "posthoc_successful_movement_excess_over_base_path": (
            max(0, moved - base_length) if outcome == "success" else None
        ),
    }


def validate_replay(record):
    if (
        record.get("schema_version") != 1
        or digest(record["scenario"]) != record["scenario_hash"]
    ):
        raise ValueError("Replay scenario/record identity mismatch")
    for name, value in record["code_hashes"].items():
        if sha(Path(__file__).parent / name) != value:
            raise ValueError(f"Replay engine changed: {name}")
    env = PartialDynamicEnv([record["scenario"]], record["configuration"])
    if env.signature != record["task_signature"]:
        raise ValueError("Replay task mismatch")
    observation, _ = env.reset(seed=record["evaluation_seed"])
    if (
        observation.tolist() != record["initial_observation"]
        or env.diagnostic_snapshot() != record["initial_snapshot"]
    ):
        raise ValueError("Replay initial state mismatch")
    for i, transition in enumerate(record["transitions"]):
        if observation.tolist() != transition["observation"]:
            raise ValueError(f"Replay before-state mismatch: {i}")
        observation, reward, terminal, cutoff, info = env.step(transition["action"])
        if (
            observation.tolist() != transition["next_observation"]
            or reward != transition["reward"]
            or terminal != transition["terminated"]
            or cutoff != transition["truncated"]
            or info != transition["info"]
            or env.diagnostic_snapshot() != transition["snapshot"]
        ):
            raise ValueError(f"Replay transition mismatch: {i}")
    if episode_metrics(record) != record["metrics"]:
        raise ValueError("Replay metrics mismatch")
    return record


def aggregate(records):
    if not records:
        raise ValueError("Cannot aggregate zero episodes")
    metrics = [r["metrics"] for r in records]
    counts = {
        o: sum(m["outcome"] == o for m in metrics)
        for o in ("success", "collision", "timeout")
    }
    if sum(counts.values()) != len(records):
        raise ValueError("Outcomes do not reconcile")
    opportunities = sum(m["recovery_opportunities"] for m in metrics)
    recovered = sum(m["recoveries"] for m in metrics)
    delays = [delay for m in metrics for delay in m["observed_to_recovery_delays"]]
    return {
        "episodes": len(records),
        "counts": counts,
        **{f"{k}_rate": v / len(records) for k, v in counts.items()},
        "mean_actions": float(np.mean([m["actions"] for m in metrics])),
        "mean_return": float(np.mean([m["return"] for m in metrics])),
        "total_actions": sum(m["actions"] for m in metrics),
        "total_waits": sum(m["wait_count"] for m in metrics),
        "recovery_opportunities": opportunities,
        "recoveries": recovered,
        "recovery_rate": recovered / opportunities if opportunities else None,
        "mean_observed_to_recovery_delay": float(np.mean(delays)) if delays else None,
        "mean_decision_microseconds": 1e6
        * sum(r["inference_seconds"] for r in records)
        / sum(m["actions"] for m in metrics),
    }


def grouped_summary(records):
    """Keep conditions/seeds/layouts/difficulty distinct; no pooled headline."""
    result = {}
    for field in (
        "algorithm",
        "condition",
        "layout_id",
        "difficulty_band",
        "training_seed",
    ):
        groups = {}
        for record in records:
            case = record["scenario"]
            value = (
                case.get("difficulty", {}).get("distance_band", "manual")
                if field == "difficulty_band"
                else (
                    record[field]
                    if field in ("algorithm", "training_seed")
                    else case[field]
                )
            )
            groups.setdefault(str(value), []).append(record)
        result[field] = {key: aggregate(rows) for key, rows in groups.items()}
    joint = {}
    for record in records:
        case = record["scenario"]
        key = (
            record["algorithm"],
            record["training_seed"],
            case["condition"],
            case["layout_id"],
            case.get("difficulty", {}).get("distance_band", "manual"),
        )
        joint.setdefault(key, []).append(record)
    result["joint_strata"] = [
        dict(
            algorithm=k[0],
            training_seed=k[1],
            condition=k[2],
            layout_id=k[3],
            difficulty_band=k[4],
            metrics=aggregate(rows),
        )
        for k, rows in joint.items()
    ]
    return result
