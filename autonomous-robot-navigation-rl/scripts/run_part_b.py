"""Execute the predeclared paired tabular search, reward ablation and voting."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter, perf_counter_ns

import numpy as np
import yaml

from robot_navigation.agents import load_policy
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, digest, load_task, read_json, write_json
from robot_navigation.episodes import validate_replay
from robot_navigation.part_a import train_part_a
from robot_navigation.refinement import search_candidates, selection_rank
from robot_navigation.visualization import save_figure
from robot_navigation.voting import VotingPolicy
from robot_navigation.workflow import evaluate, source_hash


def config_score(rows, identity):
    return selection_rank(rows, identity)


def vote_experiment(study, plan, family, rows, prefix):
    task = load_task()
    directory = ROOT / "artifacts/runs" / f"{prefix}-voting-{family}"
    directory.mkdir(parents=True, exist_ok=False)
    chosen = [next(r for r in rows if r["stage"] == "tuned" and r["algorithm"] == family and r["seed"] == seed)
              for seed in plan["voting_member_seeds"]]
    paths = [ROOT / "artifacts/runs" / r["run_id"] / "selected.npz" for r in chosen]
    members = [load_policy(path, task) for path in paths]
    vote = VotingPolicy(task, members, plan["voting_member_seeds"], f"voting/{study.name}/{family}")
    vote.save(directory / "voting.json", paths)
    write_json(directory / "configuration.json", {"training": {"algorithm": "voting", "reward_variant": "step-cost",
        "member_seeds": plan["voting_member_seeds"], "seed": plan["single_policy_seed"]}, "environment": task.config,
        "signature": task.signature})
    for folder in ("src", "app", "configs"):
        shutil.copytree(ROOT / folder, directory / "source" / folder, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    manifest = {"status": "running", "run_id": directory.name, "algorithm": "voting", "stage": "voting",
        "signature": task.signature, "source_hash": source_hash(ROOT), "viewer_checkpoint": "voting.json",
        "member_run_ids": [r["run_id"] for r in chosen], "training_interactions": 0,
        "member_training_interactions": sum(r["pilot"]["training_interactions"] for r in chosen),
        "final_test_evaluated": False, "inference_only_checkpoint": True}
    write_json(directory / "manifest.json", manifest)
    try:
        validation = load_scenarios(task, "validation")
        summary, metrics, records = evaluate(task, validation, vote, plan["single_policy_seed"],
            "complete validation manifest; frozen voting", trajectory_path=directory / "voting_trajectories.jsonl.gz")
        write_json(directory / "validation.json", {"voting": summary})
        write_json(directory / "voting_episodes.json", metrics)
        for episode in records:
            validate_replay(task, episode)
            write_json(directory / "replays" / f"voting_{episode['metrics']['outcome']}.json", episode)
        save_figure(directory / "voting_trajectory.png", task.grid,
                    next((r for r in records if r["metrics"]["outcome"] == "success"), records[0]))
        write_json(directory / "pilot.json", {"training_interactions": 0, "successful_training_episodes": 0,
            "distinct_goals_discovered": 0, "note": "Wrapper has no training; member training cost is reported separately"})
        before = vote.state_fingerprint()
        timing = {}
        policies = [(f"member-seed{seed}", member) for seed, member in zip(plan["voting_member_seeds"], members)] + [("voting", vote)]
        observations = [np.array((*s["start"], *s["goal"]), dtype=np.int64) for s in validation]
        for name, policy in policies:
            samples = []
            for i in range(1000):
                start = perf_counter_ns()
                policy.act(observations[i % len(observations)])
                samples.append((perf_counter_ns() - start) / 1000)
            timing[name] = {"mean_microseconds": float(np.mean(samples)), "median_microseconds": float(np.median(samples))}
        assert vote.state_fingerprint() == before
        ties, actions, tie_episodes = 0, 0, 0
        with gzip.open(directory / "voting_trajectories.jsonl.gz", "rt", encoding="utf-8") as file:
            for line in file:
                episode = json.loads(line)
                count = sum(t["decision"]["three_way_tie"] for t in episode["transitions"])
                ties += count
                actions += len(episode["transitions"])
                tie_episodes += count > 0
        comparisons = {f"member-seed{r['seed']}": r["validation"] for r in chosen}
        single = next(r for r in chosen if r["seed"] == plan["single_policy_seed"])
        result = {"family": family, "run_id": directory.name, "member_seeds": plan["voting_member_seeds"],
            "member_run_ids": [r["run_id"] for r in chosen], "selected_single_seed": plan["single_policy_seed"],
            "selected_single_run_id": single["run_id"], "selected_single": single["validation"], "members": comparisons,
            "voting": summary, "differences": {name: {field: summary[field] - result[field]
                for field in ("success_rate", "collision_rate", "timeout_rate", "spl")} for name, result in comparisons.items()},
            "voting_table_memory_bytes": sum(m.q.nbytes for m in members), "single_table_memory_bytes": members[0].q.nbytes,
            "voting_member_checkpoint_bytes": sum(p.stat().st_size for p in paths), "single_checkpoint_bytes": paths[0].stat().st_size,
            "member_training_interactions": manifest["member_training_interactions"],
            "member_training_seconds": sum(r["pilot"]["training_seconds"] for r in chosen),
            "additional_voting_training_interactions": 0, "timing": timing,
            "timing_method": "perf_counter_ns; 1000 round-robin validation initial-state queries; no reset/rendering/learning",
            "three_way_tie_actions": ties, "total_voting_actions": actions, "episodes_with_three_way_ties": tie_episodes,
            "component_metrics_reused": "same saved 928-pair validation evaluations; selected single is seed101, not another independent sample"}
        write_json(study / "voting_comparison.json", result)
        manifest.update(status="complete", validation_interactions=summary["interactions"], inference_state_unchanged=True)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "manifest.json", manifest)
        raise
    write_json(directory / "manifest.json", manifest)
    return directory.name


def main():
    parser = argparse.ArgumentParser(description="Run the bounded Part B development study; final test remains sealed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/part_b.yaml")
    parser.add_argument("--suite-id")
    parser.add_argument("--run-prefix", default="m4")
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    suite_id = args.suite_id or plan["suite_id"]
    if any(Path(v).name != v or v in (".", "..") for v in (suite_id, args.run_prefix)):
        parser.error("Output names must be single directory names")
    if plan["algorithms"] != read_json(ROOT / "artifacts/studies/part-a-v1/shortlist.json")["shortlist"]:
        raise ValueError("Study must use the frozen Part A shortlist")
    candidates = search_candidates(plan)
    expected_runs = len(plan["algorithms"]) * (len(candidates) * len(plan["development_seeds"]) + 3 * len(plan["confirmation_seeds"]))
    assert expected_runs == plan["limits"]["training_runs"] == 62
    assert expected_runs * plan["common"]["interactions"] == plan["limits"]["training_interactions"]
    assert len(set(plan["confirmation_seeds"]) & set(plan["development_seeds"])) == 0
    assert plan["single_policy_seed"] == plan["voting_member_seeds"][0]
    study = ROOT / "artifacts/studies" / suite_id
    study.mkdir(parents=True, exist_ok=False)
    task = load_task()
    write_json(study / "plan.json", {**plan, "run_prefix": args.run_prefix, "suite_id": suite_id})
    write_json(study / "candidates.json", candidates)
    shutil.copy2(__file__, study / "runner.py")
    shutil.copy2(ROOT / "docs/milestone4_protocol.md", study / "protocol.md")
    old = [r for r in read_json(ROOT / "artifacts/studies/part-a-v1/run_results.json") if r["algorithm"] in plan["algorithms"]]
    old_manifests = [read_json(ROOT / "artifacts/runs" / r["run_id"] / "manifest.json") for r in old]
    write_json(study / "freeze.json", {"signature": task.signature, "source_hash": source_hash(ROOT),
        "plan_hash": digest(plan), "candidate_hash": digest(candidates), "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "estimates": {"training_seconds": expected_runs * float(np.mean([r["pilot"]["training_seconds"] for r in old])),
            "monitoring_seconds": expected_runs * float(np.mean([m["monitoring_seconds"] for m in old_manifests])),
            "full_validation_seconds": expected_runs * float(np.mean([r["validation"]["evaluation_seconds"] for r in old])),
            "storage_mib": 300, "basis": "measured Part A tabular training, monitoring and full-validation/export timings; separate conservative storage estimate"},
        "targets": plan["targets"], "limits": plan["limits"], "final_test_evaluated": False})
    manifest = {"status": "running", "phase": "search", "signature": task.signature, "source_hash": source_hash(ROOT),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "completed_training_runs": [],
        "started_utc": datetime.now(timezone.utc).isoformat(), "final_test_evaluated": False}
    write_json(study / "manifest.json", manifest)
    rows, storage_bytes = [], 0
    started = perf_counter()

    def run_trial(algorithm, seed, parameters, stage, candidate_id=None):
        nonlocal storage_bytes
        if (len(rows) >= plan["limits"]["training_runs"] or
                perf_counter() - started > plan["limits"]["wall_minutes"] * 60 or
                storage_bytes > plan["limits"]["run_storage_mib"] * 1024 * 1024):
            raise RuntimeError("Predeclared study limit reached before the next run")
        extra = f"-{candidate_id}" if candidate_id else ""
        run_id = f"{args.run_prefix}-{stage}-{algorithm}{extra}-seed{seed}"
        variant = "terminal-only" if stage == "terminal" else "step-cost"
        config = {**plan["common"], **parameters, "algorithm": algorithm, "seed": seed,
                  "run_id": run_id, "stage": stage, "reward_variant": variant}
        print(f"Starting {run_id}", flush=True)
        directory, summaries, pilot = train_part_a(config)
        row = {"run_id": run_id, "algorithm": algorithm, "seed": seed, "stage": stage,
               "candidate_id": candidate_id, "parameters": parameters, "reward_variant": variant,
               "validation": summaries[algorithm.replace("-", "_")], "pilot": pilot,
               "selected_interactions": read_json(directory / "manifest.json")["selected_interactions"]}
        rows.append(row)
        storage_bytes += sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        manifest.update(phase=stage, completed_training_runs=[r["run_id"] for r in rows],
                        run_storage_mib=storage_bytes / 1024 / 1024)
        write_json(study / "run_results.json", rows)
        write_json(study / "manifest.json", manifest)
        print(f"Finished {len(rows)}/{expected_runs} {run_id}: {row['validation']['counts']}", flush=True)

    try:
        for algorithm in plan["algorithms"]:
            for candidate in candidates:
                for seed in plan["development_seeds"]:
                    run_trial(algorithm, seed, candidate["parameters"], "search", candidate["candidate_id"])
        selected = {}
        for algorithm in plan["algorithms"]:
            chosen = min(candidates, key=lambda c: config_score(
                [r for r in rows if r["algorithm"] == algorithm and r["candidate_id"] == c["candidate_id"]], c["candidate_id"]))
            selected[algorithm] = chosen
        write_json(study / "selected_configurations.json", selected)
        print(f"Selected development settings: {selected}", flush=True)
        for algorithm in plan["algorithms"]:
            for seed in plan["confirmation_seeds"]:
                run_trial(algorithm, seed, plan["baseline"], "control")
                run_trial(algorithm, seed, selected[algorithm]["parameters"], "tuned")
        family_rows = {a: [r for r in rows if r["algorithm"] == a and r["stage"] == "tuned"] for a in plan["algorithms"]}
        def family_rank(algorithm):
            group = family_rows[algorithm]
            score = selection_rank(group, algorithm)
            return (*score[:3], sum(r["pilot"]["training_seconds"] for r in group), algorithm)
        family = min(plan["algorithms"], key=family_rank)
        write_json(study / "family_choice_before_voting.json", {"family": family, "member_seeds": plan["voting_member_seeds"],
            "single_policy_seed": plan["single_policy_seed"], "reward_variant": "step-cost",
            "ranking_rule": "confirmation mean success, mean SPL, lower seed SD, training seconds, name",
            "decision_saved_before_reward_ablation_and_voting": True})
        for algorithm in plan["algorithms"]:
            for seed in plan["confirmation_seeds"]:
                run_trial(algorithm, seed, selected[algorithm]["parameters"], "terminal")
        manifest["voting_run"] = vote_experiment(study, plan, family, rows, args.run_prefix)
        manifest.update(status="complete", phase="complete", wall_seconds=perf_counter() - started,
            training_runs=len(rows), training_interactions=sum(r["pilot"]["training_interactions"] for r in rows),
            monitoring_interactions=sum(read_json(ROOT / "artifacts/runs" / r["run_id"] / "manifest.json")["monitoring_interactions"] for r in rows),
            validation_interactions=sum(r["validation"]["interactions"] for r in rows))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete",
                        phase=manifest["phase"], error=f"{type(exc).__name__}: {exc}", wall_seconds=perf_counter() - started,
                        completed_training_runs=[r["run_id"] for r in rows], training_runs=len(rows))
        write_json(study / "manifest.json", manifest)
        raise
    write_json(study / "manifest.json", manifest)
    print("Completed Part B matrix and voting. Run scripts/analyze_part_b.py for audited comparisons.", flush=True)


if __name__ == "__main__":
    main()
