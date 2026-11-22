"""Execute the predeclared development matrix and audit every saved trajectory."""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import yaml

from robot_navigation.agents import load_policy
from robot_navigation.core import ROOT, load_task, read_json, write_json
from robot_navigation.episodes import aggregate, validate_replay
from robot_navigation.training import train_tabular
from robot_navigation.workflow import source_hash


def diagnose_original_smoke():
    """Read existing failures; leave all milestone-1 evidence untouched."""
    run = ROOT / "artifacts/runs/smoke-seed7"
    unique_counts, timeout_q_ranges = [], []
    with (run / "q_learning_trajectories.jsonl").open(encoding="utf-8") as file:
        for line in file:
            episode = json.loads(line)
            if episode["metrics"]["outcome"] == "timeout":
                positions = [tuple(episode["initial_observation"][:2])] + [tuple(t["next_observation"][:2]) for t in episode["transitions"]]
                unique_counts.append(len(set(positions)))
                scores = np.array([t["scores"] for t in episode["transitions"]])
                timeout_q_ranges.append((float(scores.min()), float(scores.max())))
    pilot = read_json(run / "pilot.json")
    return {"source_run": "smoke-seed7", "timeout_episodes": len(unique_counts),
        "timeout_unique_position_histogram": dict(Counter(unique_counts)),
        "timeout_q_min": min(v[0] for v in timeout_q_ranges),
        "timeout_q_max": max(v[1] for v in timeout_q_ranges),
        "undiscovered_goals": sorted(set(pilot["sampled_training_goals"]) - set(pilot["successful_discoveries_by_goal"])),
        "interpretation": "Repeated-state loops consume all 200 actions. Goal discoveries are sparse. This motivates a longer diagnostic budget, not rule changes or planner fallback."}


def audit_run(directory, task):
    manifest = read_json(directory / "manifest.json")
    assert manifest["status"] == "complete" and not manifest["final_test_evaluated"]
    assert source_hash(directory / "source") == manifest["source_hash"]
    agent = load_policy(directory / "checkpoint.npz", task)
    before = agent.q.copy()
    key = "q_learning" if agent.algorithm == "q-learning" else "sarsa"
    metrics, actions = [], 0
    with gzip.open(directory / f"{key}_trajectories.jsonl.gz", "rt", encoding="utf-8") as file:
        for line in file:
            episode = json.loads(line)
            validate_replay(task, episode)
            for transition in episode["transitions"]:
                assert transition["action"] == agent.act(transition["observation"])
                assert transition["scores"] == agent.scores(transition["observation"]).tolist()
            metrics.append(episode["metrics"])
            actions += len(episode["transitions"])
    expected = read_json(directory / f"{key}_episodes.json")
    assert metrics == [{k: v for k, v in m.items() if k != "scenario_id"} for m in expected]
    result = read_json(directory / "validation.json")[key]
    for field, value in aggregate(metrics).items():
        assert result[field] == value
    with np.load(directory / "visitation.npz", allow_pickle=False) as file:
        visits = int(file["state_action_counts"].sum())
    pilot = read_json(directory / "pilot.json")
    with (directory / "training.csv").open(newline="", encoding="utf-8") as file:
        logged_steps = sum(int(row["steps"]) for row in csv.DictReader(file))
    partial = pilot["incomplete_budget_cutoff_episode"]
    assert visits == logged_steps + (partial["steps"] if partial else 0) == manifest["training_interactions"]
    assert np.array_equal(before, agent.q)
    checkpoints = list((directory / "checkpoints").glob("*.npz"))
    for path in checkpoints:
        load_policy(path, task)
    return {"run_id": directory.name, "episodes_verified": len(metrics), "saved_actions_verified": actions,
            "training_interactions_reconciled": visits, "periodic_checkpoints_verified": len(checkpoints),
            "source_snapshot_matches": True, "parameters_unchanged": True, "final_test_executions": 0}


def compare_repeat(first, second):
    task = load_task()
    a, b = load_policy(first / "checkpoint.npz", task), load_policy(second / "checkpoint.npz", task)
    key = "q_learning" if a.algorithm == "q-learning" else "sarsa"
    equal_table = np.array_equal(a.q, b.q)
    equal_log = (first / "training.csv").read_bytes() == (second / "training.csv").read_bytes()
    equal_metrics = read_json(first / f"{key}_episodes.json") == read_json(second / f"{key}_episodes.json")
    equal_replays = True
    with gzip.open(first / f"{key}_trajectories.jsonl.gz", "rt", encoding="utf-8") as fa, gzip.open(second / f"{key}_trajectories.jsonl.gz", "rt", encoding="utf-8") as fb:
        for line_a, line_b in zip(fa, fb, strict=True):
            ea, eb = json.loads(line_a), json.loads(line_b)
            ea.pop("policy_id")
            eb.pop("policy_id")
            equal_replays &= ea == eb
    assert equal_table and equal_log and equal_metrics and equal_replays
    return {"algorithm": a.algorithm, "seed": 7, "equal_tables": equal_table, "equal_training_csv": equal_log,
            "equal_validation_metrics": equal_metrics, "equal_replay_records_except_policy_id": equal_replays}


def main():
    plan = yaml.safe_load((ROOT / "configs/tabular_milestone.yaml").read_text(encoding="utf-8"))
    study = ROOT / "artifacts/studies" / plan["suite_id"]
    study.mkdir(parents=True, exist_ok=False)
    write_json(study / "plan.json", plan)
    shutil.copy2(ROOT / "docs/milestone2_protocol.md", study / "protocol.md")
    write_json(study / "original_smoke_diagnostics.json", diagnose_original_smoke())
    started = perf_counter()
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "source_hash": source_hash(ROOT), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "final_test_evaluated": False, "completed_runs": []}
    write_json(study / "manifest.json", manifest)
    results, audits, repeats = [], [], []
    try:
        for algorithm in plan["algorithms"]:
            for seed, repeat in [(s, False) for s in plan["seeds"]] + [(plan["repeat_seed"], True)]:
                run_id = f"m2-{algorithm}-seed{seed}" + ("-repeat" if repeat else "")
                config = {**plan["training"], "algorithm": algorithm, "seed": seed, "run_id": run_id}
                print(f"Starting {run_id}", flush=True)
                directory, summaries, pilot = train_tabular(config=config)
                key = "q_learning" if algorithm == "q-learning" else "sarsa"
                result = {"run_id": run_id, "algorithm": algorithm, "seed": seed, "repeat": repeat,
                          "validation": summaries[key], "pilot": pilot}
                results.append(result)
                audits.append(audit_run(directory, load_task()))
                manifest["completed_runs"].append(run_id)
                write_json(study / "manifest.json", manifest)
                write_json(study / "run_results.json", results)
                if repeat:
                    repeats.append(compare_repeat(directory.with_name(f"m2-{algorithm}-seed{seed}"), directory))
                print(f"Finished {run_id}: {summaries[key]['counts']}", flush=True)
        summary = {}
        for algorithm in plan["algorithms"]:
            independent = [r for r in results if r["algorithm"] == algorithm and not r["repeat"]]
            summary[algorithm] = {"independent_seeds": [r["seed"] for r in independent],
                "seed_results": [{"seed": r["seed"], **r["validation"]} for r in independent],
                "aggregation_unit": "independent training seed; repeats excluded",
                "metrics": {field: {"mean": float(np.mean([r["validation"][field] for r in independent])),
                                    "sample_std": float(np.std([r["validation"][field] for r in independent], ddof=1))}
                            for field in ("success_rate", "collision_rate", "timeout_rate", "spl", "mean_return")}}
        write_json(study / "summary.json", summary)
        write_json(study / "verification.json", {"audits": audits, "reproducibility": repeats,
                   "final_test_policy_executions": 0, "browser_animation": "not verified by this runner"})
        manifest.update(status="complete", wall_seconds=perf_counter() - started,
                        training_interactions=sum(r["pilot"]["training_interactions"] for r in results))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        write_json(study / "manifest.json", manifest)
        raise
    write_json(study / "manifest.json", manifest)
    print(json.dumps({a: s["metrics"] for a, s in summary.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
