"""Run the frozen three-family comparison; never reads the final-test manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
from time import perf_counter

import yaml

from robot_navigation.agents import BFSPolicy, RandomPolicy
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, digest, load_task, read_json, write_json
from robot_navigation.part_a import train_part_a
from robot_navigation.workflow import evaluate, source_hash


def main():
    parser = argparse.ArgumentParser(description="Part A: frozen validation-only experiment matrix")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/part_a.yaml")
    parser.add_argument("--suite-id")
    parser.add_argument("--run-prefix", default="m3")
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    suite_id = args.suite_id or plan["suite_id"]
    if any(Path(v).name != v or v in (".", "..") for v in (suite_id, args.run_prefix)):
        parser.error("Output names must be single directory names")
    task = load_task()
    study = ROOT / "artifacts/studies" / suite_id
    study.mkdir(parents=True, exist_ok=False)
    write_json(study / "plan.json", {**plan, "suite_id": suite_id, "run_prefix": args.run_prefix})
    shutil.copy2(ROOT / "docs/milestone3_protocol.md", study / "protocol.md")
    shutil.copy2(__file__, study / "runner.py")
    pilot = read_json(ROOT / "artifacts/runs/m3-dqn-pilot-seed7/pilot.json")
    pilot_validation = read_json(ROOT / "artifacts/runs/m3-dqn-pilot-seed7/validation.json")["dqn"]
    pilot_manifest = read_json(ROOT / "artifacts/runs/m3-dqn-pilot-seed7/manifest.json")
    task.check(pilot_manifest["signature"])
    if pilot_manifest["status"] != "complete":
        raise ValueError("Complete the recorded DQN pilot before freezing the matrix")
    dqn_training_estimate = len(plan["seeds"]) * plan["common"]["interactions"] / pilot["training_interactions_per_second"] * 1.2
    monitoring_estimate = len(plan["seeds"]) * plan["common"]["interactions"] / plan["common"]["monitor_every"] * pilot_manifest["monitoring_seconds"]
    write_json(study / "benchmark_freeze.json", {"signature": task.signature,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "configuration_hash": digest(plan),
        "pilot_training_interactions": pilot["training_interactions"], "pilot_training_seconds": pilot["training_seconds"],
        "pilot_validation_seconds": pilot_validation["evaluation_seconds"], "pilot_validation_counts": pilot_validation["counts"],
        "pilot_discovered_goals": pilot["distinct_goals_discovered"],
        "estimated_dqn_training_seconds": dqn_training_estimate, "estimated_dqn_monitoring_seconds": monitoring_estimate,
        "estimated_dqn_full_validation_seconds": len(plan["seeds"]) * pilot_validation["evaluation_seconds"],
        "estimated_tabular_training_seconds": 110., "estimated_tabular_monitoring_and_validation_seconds": 80.,
        "estimated_artifact_storage_mb": 100.,
        "estimate_basis": "DQN measured 10k throughput with 20% warm-up allowance; tabular milestone-2 measurements; storage includes trajectories and periodic models",
        "selection": "monitoring success, SPL, earliest; final ranking on full validation, never final test",
        "task_adjustments_after_pilot": [], "final_test_evaluated": False})
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash(ROOT), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "signature": task.signature, "final_test_evaluated": False, "completed_runs": []}
    write_json(study / "manifest.json", manifest)
    results = []
    started = perf_counter()
    try:
        for algorithm in plan["algorithms"]:
            settings = plan["dqn"] if algorithm == "dqn" else plan["tabular"]
            for seed in plan["seeds"]:
                config = {**plan["common"], **settings, "algorithm": algorithm, "seed": seed,
                          "run_id": f"{args.run_prefix}-{algorithm}-seed{seed}"}
                print(f"Starting {config['run_id']}", flush=True)
                directory, summaries, pilot = train_part_a(config)
                key = algorithm.replace("-", "_")
                result = {"run_id": directory.name, "algorithm": algorithm, "seed": seed,
                          "validation": summaries[key], "pilot": pilot,
                          "selected_interactions": read_json(directory / "manifest.json")["selected_interactions"]}
                results.append(result)
                manifest["completed_runs"].append(directory.name)
                write_json(study / "run_results.json", results)
                write_json(study / "manifest.json", manifest)
                print(f"Finished {directory.name}: selected {result['selected_interactions']} / {config['interactions']}; {summaries[key]['counts']}", flush=True)
        validation = load_scenarios(task, "validation")
        references = {}
        for name, policy in [("bfs", BFSPolicy(task.grid))] + [(f"random-seed{s}", RandomPolicy(s)) for s in plan["seeds"]]:
            seed = int(name.removeprefix("random-seed")) if name.startswith("random") else 0
            summary, metrics, _ = evaluate(task, validation, policy, seed, "complete validation reference",
                                          trajectory_path=study / f"{name}_trajectories.jsonl.gz")
            references[name] = summary
            write_json(study / f"{name}_episodes.json", metrics)
            if name == "bfs" and (summary["success_rate"] != 1 or summary["spl"] != 1):
                raise AssertionError("BFS reference correctness failure")
        write_json(study / "references.json", references)
        manifest.update(status="complete", wall_seconds=perf_counter() - started,
                        training_interactions=sum(r["pilot"]["training_interactions"] for r in results),
                        validation_interactions=sum(r["validation"]["interactions"] for r in results),
                        reference_interactions=sum(r["interactions"] for r in references.values()))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        write_json(study / "manifest.json", manifest)
        raise
    write_json(study / "manifest.json", manifest)
    print("Completed matrix; run scripts/analyze_part_a.py to audit, plot and shortlist.", flush=True)


if __name__ == "__main__":
    main()
