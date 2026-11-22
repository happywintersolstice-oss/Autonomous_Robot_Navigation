"""Product development/final scoring. Truth is confined to this evaluator.
All policy decisions receive only409 public values. Stored truth is human-only.
"""

from collections import deque
from pathlib import Path
from statistics import mean, median
import csv, gzip, json, zipfile
import numpy as np
from .core import ROOT, digest, write_json, read_json
from .partial_contract import configuration, MOVES
from .partial_dynamic_env import PartialDynamicEnv
from .partial_evaluation import episode_metrics
from .partial_baselines import OnlinePlanner, RandomPolicy
from .app_registry import load_model, RUNS
from .app_service import identity
from .phase3_features import PublicHistory
from .phase3_data import shortest
from .release import sha


def record(case, policy, seed=0):
    """Execute one evaluation episode; only a copied public409 reaches policy.act.

    Truth snapshots are retained for human playback and post-hoc scoring only.
    The parameter fingerprint must not change during inference."""
    if hasattr(policy, "reset"):
        policy.reset()
    env = PartialDynamicEnv([case], configuration())
    obs, _ = env.reset(seed=seed)
    fingerprint = getattr(policy, "state_fingerprint", lambda: None)
    before = fingerprint()
    r = {
        "schema_version": 1,
        "scope": "product-phase3",
        "task_signature": env.signature,
        "configuration": env.config,
        "scenario": case,
        "scenario_hash": digest(case),
        "evaluation_seed": seed,
        "policy_id": policy.policy_id,
        "algorithm": policy.algorithm,
        "policy_signature": getattr(policy, "signature", None),
        "training_seed": getattr(policy, "training_seed", None),
        "checkpoint_sha256": getattr(policy, "checkpoint_sha256", None),
        "parameter_fingerprint": before,
        "initial_observation": obs.tolist(),
        "initial_snapshot": env.diagnostic_snapshot(),
        "transitions": [],
    }
    while True:
        action = policy.act(
            obs.copy()
        )  # The sole decision interface; no truth/events/info.
        nxt, reward, term, trunc, info = env.step(action)
        r["transitions"].append(
            {
                "observation": obs.tolist(),
                "action": int(action),
                "reward": reward,
                "next_observation": nxt.tolist(),
                "terminated": term,
                "truncated": trunc,
                "info": info,
                "snapshot": env.diagnostic_snapshot(),
            }
        )
        obs = nxt
        if term or trunc:
            break
    if before != fingerprint():
        raise ValueError("Inference changed policy parameters")
    r["metrics"] = episode_metrics(r)
    r["behavior"] = behavior(r)
    return r


def behavior(record):
    """Diagnose recorded Waits using public observations, never obstacle identity."""
    history = PublicHistory()
    blocked = alternatives = known_route = 0
    for t in record["transitions"]:
        obs = np.asarray(t["observation"], dtype=np.float32)
        action = t["action"]
        if action == 4:
            pos = tuple(np.rint(obs[:2] * 9).astype(int))
            goal = tuple(np.rint(obs[2:4] * 9).astype(int))
            distance = sum(abs(pos[j] - goal[j]) for j in range(2))
            goal_blocked = False
            free_progress = False
            for a, (dr, dc) in enumerate(MOVES[:4]):
                cell = (pos[0] + dr, pos[1] + dc)
                if sum(abs(cell[j] - goal[j]) for j in range(2)) < distance:
                    occupied = (
                        not all(0 <= v < 10 for v in cell)
                        or obs[[4, 5, 7, 6][a]] == 0.5
                    )
                    goal_blocked |= occupied
                    free_progress |= not occupied
            blocked += bool(goal_blocked and not free_progress)
            alternatives += bool(
                history.waits >= 3 and history.observed_alternative(obs)
            )
            memory = obs[8:108].reshape(10, 10)
            rows = [
                "".join("." if value == 0 else "#" for value in row) for row in memory
            ]
            known_route += bool(
                rows[goal[0]][goal[1]] == "." and shortest(rows, pos, goal)
            )
        history.remember(obs, action)
    return {
        "wait_blocked_frontier": blocked,
        "unproductive_waits_observed_alternative": alternatives,
        "wait_known_free_memory_route": known_route,
    }


def optimum(case):
    """Post-hoc evaluator-only time-expanded BFS; never used in learning/selection."""
    rows = case["rows"]
    start = tuple(case["start"])
    goal = tuple(case["goal"])
    todo = deque([(start, 0)])
    seen = {(start, 0)}
    while todo:
        pos, t = todo.popleft()
        if pos == goal:
            return t
        if t >= 200:
            continue
        blocked = {
            tuple(e["cell"]) for e in case["events"] if e["start"] <= t < e["clear"]
        }
        for dr, dc in MOVES:
            nxt = (pos[0] + dr, pos[1] + dc)
            state = (nxt, t + 1)
            if (
                all(0 <= v < 10 for v in nxt)
                and rows[nxt[0]][nxt[1]] != "#"
                and nxt not in blocked
                and state not in seen
            ):
                seen.add(state)
                todo.append(state)
    return None


def case_row(label, kind, item, record, replay, optimal=None):
    m = record["metrics"]
    b = record["behavior"]
    c = record["scenario"]
    success = m["outcome"] == "success"
    return {
        "policy": label,
        "kind": kind,
        "checkpoint_id": item["id"] if isinstance(item, dict) else item,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "feature_version": (
            getattr(item, "feature_version", None)
            if not isinstance(item, dict)
            else item.get("feature_version")
        ),
        "training_actions": item["actions"] if isinstance(item, dict) else 0,
        "case_id": c["id"],
        "case_hash": digest(c),
        "layout_hash": c["layout_hash"],
        "structural_hash": c.get("structural_hash"),
        "topology_family": c.get("topology_family", "demo"),
        "event_seed": c["event_seed"],
        "evaluation_seed": 0,
        "condition": "dynamic" if c["events"] else "static",
        "distance_band": c.get("distance_band", "manual"),
        "blockage_kind": c.get("blockage_kind", "manual"),
        "base_distance": c.get("base_distance"),
        "outcome": m["outcome"],
        "success": int(success),
        "steps": m["actions"],
        "successful_steps": m["actions"] if success else None,
        "reward": m["return"],
        "wait_count": m["wait_count"],
        "wait_fraction": m["wait_proportion"],
        "moving_revisits": m["repeated_movement_visits"],
        "two_cell_loops": m["two_cell_oscillation_windows"],
        "recovery_opportunities": m["recovery_opportunities"],
        "recoveries": m["recoveries"],
        "posthoc_optimal_actions": optimal,
        "posthoc_efficiency": (
            optimal / m["actions"] if success and optimal is not None else None
        ),
        **b,
        "replay": replay,
    }


def policy_for(item):
    if item == "Observed-Map Replanner":
        return OnlinePlanner()
    if item == "Random reference":
        return RandomPolicy(0)
    return load_model(item)


def score_candidate(item, cases):
    """Score the supplied development cases under official reward; do not train."""
    policy = policy_for(item)
    rows = []
    for case in cases:
        rec = record(case, policy)
        rows.append(
            case_row(item["algorithm"], "development candidate", item, rec, None)
        )
    successful = [r for r in rows if r["success"]]
    long = [r for r in rows if r["distance_band"] == "long"]
    return {
        "checkpoint_id": item["id"],
        "checkpoint_sha256": item["checkpoint_sha256"],
        "feature_version": item["feature_version"],
        "actions": item["actions"],
        "cases": len(rows),
        "successes": len(successful),
        "long_successes": sum(r["success"] for r in long),
        "long_cases": len(long),
        "collisions": sum(r["outcome"] == "collision" for r in rows),
        "timeouts": sum(r["outcome"] == "timeout" for r in rows),
        "unproductive_waits": sum(
            r["unproductive_waits_observed_alternative"] for r in rows
        ),
        "mean_return": mean(r["reward"] for r in rows),
        "mean_successful_steps": (
            mean(r["steps"] for r in successful) if successful else None
        ),
        "waits": sum(r["wait_count"] for r in rows),
        "all_actions": sum(r["steps"] for r in rows),
        "rows": rows,
        "heldout_cases_evaluated": 0,
    }


def rank(result):
    """Apply the registered lexicographic development-only checkpoint ranking."""
    return (
        result["successes"],
        result["long_successes"],
        -result["collisions"],
        -result["unproductive_waits"],
        result["mean_return"],
        -(
            result["mean_successful_steps"]
            if result["mean_successful_steps"] is not None
            else 1000
        ),
        -result["actions"],
    )


def summarize(rows):
    """Aggregate matched outcomes with separate all-action and successful-only Wait denominators."""
    result = []
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    families = sorted({r["topology_family"] for r in rows})
    for policy in policies:
        for subset in [
            "all",
            "static",
            "dynamic",
            "short",
            "long",
            "success",
            "collision",
            "timeout",
            *families,
        ]:
            part = [
                r
                for r in rows
                if r["policy"] == policy
                and (
                    subset == "all"
                    or subset
                    in (
                        r["condition"],
                        r["distance_band"],
                        r["topology_family"],
                        r["outcome"],
                    )
                )
            ]
            if not part:
                continue
            success = [r for r in part if r["success"]]
            n = len(part)
            actions = sum(r["steps"] for r in part)
            wait = sum(r["wait_count"] for r in part)
            sactions = sum(r["steps"] for r in success)
            swait = sum(r["wait_count"] for r in success)
            opportunities = sum(r["recovery_opportunities"] for r in part)
            eff = [
                r["posthoc_efficiency"]
                for r in success
                if r["posthoc_efficiency"] is not None
            ]
            result.append(
                {
                    "Policy": policy,
                    "Type": part[0]["kind"],
                    "Subset": subset,
                    "Cases": n,
                    "Successes": len(success),
                    "Collisions": sum(r["outcome"] == "collision" for r in part),
                    "Timeouts": sum(r["outcome"] == "timeout" for r in part),
                    "Success %": 100 * len(success) / n,
                    "Mean successful steps": (
                        mean(r["steps"] for r in success) if success else None
                    ),
                    "Median successful steps": (
                        median(r["steps"] for r in success) if success else None
                    ),
                    "Mean official return": mean(r["reward"] for r in part),
                    "Mean posthoc efficiency": mean(eff) if eff else None,
                    "Wait actions": wait,
                    "All actions": actions,
                    "Wait %": 100 * wait / actions if actions else None,
                    "Successful Wait actions": swait,
                    "Successful actions": sactions,
                    "Successful Wait %": 100 * swait / sactions if sactions else None,
                    "Wait blocked frontier": sum(
                        r["wait_blocked_frontier"] for r in part
                    ),
                    "Wait observed alternative": sum(
                        r["unproductive_waits_observed_alternative"] for r in part
                    ),
                    "Wait known-free memory route": sum(
                        r["wait_known_free_memory_route"] for r in part
                    ),
                    "Revisits": sum(r["moving_revisits"] for r in part),
                    "Two-cell loops": sum(r["two_cell_loops"] for r in part),
                    "Recoveries": sum(r["recoveries"] for r in part),
                    "Recovery opportunities": opportunities,
                    "Recovery %": (
                        100 * sum(r["recoveries"] for r in part) / opportunities
                        if opportunities
                        else None
                    ),
                }
            )
    return result


def export_comparison(choices, cases, folder, scope, metadata=None, with_optimum=False):
    """Record a new comparison and immutable CSV/JSON/replay/ZIP evidence.

    Duplicate checkpoint IDs can reuse an already recorded case. Optional true
    optimum paths are evaluator-only and never enter a policy or selection."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "episodes").mkdir()
    rows = []
    failures = []
    aliases = {}
    executed = 0
    actions = 0
    cached = {}
    optimal = {c["id"]: optimum(c) for c in cases} if with_optimum else {}
    for label, item, kind in choices:
        key = item["id"] if isinstance(item, dict) else item
        policy = None
        for case in cases:
            cache_key = (key, case["id"])
            if cache_key in cached:
                path = cached[cache_key]
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    rec = json.load(f)
                aliases[label] = key
            else:
                if policy is None:
                    policy = policy_for(item)
                rec = record(case, policy)
                path = folder / "episodes" / f"{executed:05d}.json.gz"
                with gzip.open(path, "xt", encoding="utf-8") as f:
                    json.dump(rec, f, allow_nan=False)
                cached[cache_key] = path
                executed += 1
                actions += len(rec["transitions"])
            relative = path.relative_to(folder).as_posix()
            row = case_row(label, kind, item, rec, relative, optimal.get(case["id"]))
            rows.append(row)
            if row["outcome"] != "success":
                tail = [
                    {
                        "position": t["observation"][:2],
                        "goal": t["observation"][2:4],
                        "rays_NSEW": t["observation"][4:8],
                        "action": t["action"],
                        "reward": t["reward"],
                        "terminated": t["terminated"],
                        "truncated": t["truncated"],
                    }
                    for t in rec["transitions"][-8:]
                ]
                failures.append(
                    {
                        "policy": label,
                        "checkpoint_id": key,
                        "case_id": case["id"],
                        "outcome": row["outcome"],
                        "steps": row["steps"],
                        "replay": relative,
                        "last8_public_decisions": tail,
                    }
                )
        print("RECORDED " + scope + " " + label, flush=True)
    summary = summarize(rows)
    for name, data in [("case-results.csv", rows), ("summary.csv", summary)]:
        with (folder / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    write_json(
        folder / "failure-traces.json",
        {
            "scope": scope,
            "failures": failures,
            "interpretation": "Observed alternatives are safe adjacent cells, not guaranteed complete detours. Known-free routes use remembered occupancy only, which can be stale. No label implies the robot knew permanence or clearing time.",
        },
    )
    manifest = {
        "schema": "product-phase3-comparison-v1",
        "scope": scope,
        "purpose": (
            "Product-only held-out comparison; not historical scientific final/unseen accuracy"
            if scope == "product-heldout"
            else "Development comparison used for selection"
        ),
        "panel_case_count": len(cases),
        "case_count": len(rows),
        "unique_executed_episodes": executed,
        "unique_executed_actions": actions,
        "case_ids": [c["id"] for c in cases],
        "case_hashes": [digest(c) for c in cases],
        "cases": cases,
        "evaluation_seed": 0,
        "inference_only": True,
        "aliased_policy_ids": aliases,
        "summary": summary,
        "training_provenance": {
            label: {k: v for k, v in item.items() if k != "path"}
            for label, item, _ in choices
            if isinstance(item, dict)
        },
        "code_sha256": {
            name: sha(ROOT / "src/robot_navigation" / name)
            for name in (
                "phase3_features.py",
                "phase3_evaluation.py",
                "phase3_training.py",
                "phase3_data.py",
            )
        },
        "posthoc_optimum": (
            "Time-expanded true occupancy BFS with Wait; evaluator only, never selection/training input"
            if with_optimum
            else None
        ),
        **(metadata or {}),
    }
    write_json(folder / "manifest.json", manifest)
    inventory = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            inventory[p.relative_to(folder).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": sha(p),
            }
    write_json(folder / "export-inventory.json", {"files": inventory, "scope": scope})
    with zipfile.ZipFile(
        folder / "results-and-replays.zip", "x", zipfile.ZIP_DEFLATED, compresslevel=3
    ) as z:
        for name in [*inventory, "export-inventory.json"]:
            z.write(folder / name, name)
    return {"folder": str(folder), "manifest": manifest, "rows": rows}


def load_saved(folder):
    folder = Path(folder)
    with (folder / "case-results.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "folder": str(folder),
        "manifest": read_json(folder / "manifest.json"),
        "rows": rows,
    }
