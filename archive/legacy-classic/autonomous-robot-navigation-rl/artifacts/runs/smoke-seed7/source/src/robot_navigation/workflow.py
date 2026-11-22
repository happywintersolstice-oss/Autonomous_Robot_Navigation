"""Small, recorded training pilot and read-only validation. Never opens test.json."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import yaml

from .agents import BFSPolicy, QLearning, RandomPolicy
from .benchmark import load_scenarios
from .core import ROOT, NavigationEnv, digest, load_task, write_json
from .episodes import aggregate, rollout, validate_replay
from .visualization import save_figure


def source_hash(root):
    sources = {}
    for folder, extension in (("src", "*.py"), ("app", "*.py"), ("configs", "*.yaml")):
        for path in sorted((root / folder).rglob(extension)):
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(sources)


def evaluate(task, scenarios, policy, seed, scope, visitation=None):
    env = NavigationEnv(task, scenarios)
    before = policy.q.copy() if isinstance(policy, QLearning) else None
    metrics, records, decision_times = [], [], []
    started = perf_counter()
    for scenario in scenarios:
        episode = rollout(env, policy, scenario, seed)
        metrics.append({"scenario_id": scenario["id"], **episode["metrics"]})
        # Retain a representative success and each failure type, not all 200-step loops.
        outcome = episode["metrics"]["outcome"]
        if outcome not in [r["metrics"]["outcome"] for r in records]:
            records.append(episode)
        if visitation is not None:
            episode["initial_state_visited_in_training"] = bool(visitation[tuple(episode["initial_observation"])].sum())
    elapsed = perf_counter() - started
    if before is not None and not np.array_equal(before, policy.q):
        raise AssertionError("Validation changed learned parameters")
    # Decision-only timing excludes reset, rendering and recording; 100 repeated
    # deterministic queries, or RNG advancement for random reference only.
    observation = np.array((*scenarios[0]["start"], *scenarios[0]["goal"]), dtype=np.int64)
    for _ in range(100):
        start = perf_counter()
        policy.act(observation)
        decision_times.append(perf_counter() - start)
    summary = {**aggregate(metrics), "scope": scope, "seed": seed,
               "evaluation_seconds": elapsed, "interactions_per_second": sum(m["steps"] for m in metrics) / elapsed,
               "mean_decision_microseconds": float(np.mean(decision_times) * 1e6),
               "timing_method": "perf_counter; wall time includes recording/reset; decision mean over 100 queries; no rendering"}
    if visitation is not None:
        overlap = sum(bool(visitation[(*s["start"], *s["goal"])].sum()) for s in scenarios)
        summary["initial_states_visited_in_training"] = overlap
        summary["initial_state_overlap_fraction"] = overlap / len(scenarios)
    return summary, metrics, records


def train_smoke(config_path=None, root=ROOT):
    config_path = Path(config_path) if config_path else root / "configs/smoke.yaml"
    config = yaml.safe_load(config_path.read_text())
    for key in ("interactions", "monitor_every"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not (0 < config["learning_rate"] <= 1 and 0 <= config["discount"] <= 1 and
            0 <= config["epsilon_end"] <= config["epsilon_start"] <= 1 and
            0 < config["epsilon_decay_fraction"] <= 1):
        raise ValueError("Invalid Q-learning parameters")
    run_id = config["run_id"]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    directory = root / "artifacts/runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    task = load_task(root)
    train = load_scenarios(task, "train", root)
    validation = load_scenarios(task, "validation", root)
    monitoring = load_scenarios(task, "monitoring", root)
    policy = QLearning(task, f"q-learning/{run_id}/final")
    env = NavigationEnv(task, train)
    observation, _ = env.reset(seed=config["seed"])
    rng = np.random.default_rng(config["seed"])
    visitation = np.zeros_like(policy.q, dtype=np.int64)
    goal_samples, goal_discoveries = {}, {}
    goal_key = lambda obs: f"{int(obs[2])},{int(obs[3])}"
    goal_samples[goal_key(observation)] = 1
    steps, total_return, episode_started = 0, 0.0, 0
    monitoring_log, log = [], []
    elapsed_training = 0.0
    write_json(directory / "configuration.json", {"training": config, "environment": task.config, "signature": task.signature})
    manifest = {"run_id": run_id, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "source_hash": source_hash(root), "python": sys.version,
                "platform": platform.platform(), "processor": platform.processor(),
                "dependencies": {p: version(p) for p in ("numpy", "gymnasium", "PyYAML", "matplotlib", "streamlit")},
                "training_seed": config["seed"], "signature": task.signature,
                "algorithm": "Q-learning", "training_scenario_ids": [s["id"] for s in train],
                "validation_scenario_ids": [s["id"] for s in validation],
                "selection_rule": "final-budget checkpoint; no checkpoint/configuration selection in this smoke pilot",
                "final_test_evaluated": False, "inference_only_checkpoint": True,
                "information": "Q-learning receives robot/goal coordinates only; no planner, masking or fallback"}
    write_json(directory / "manifest.json", manifest)
    log_path = directory / "training.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "scenario_id", "end_interaction", "steps", "return", "outcome", "epsilon"])
        writer.writeheader()
        start = perf_counter()
        for interaction in range(1, config["interactions"] + 1):
            progress = min(1.0, (interaction - 1) / (config["interactions"] * config["epsilon_decay_fraction"]))
            epsilon = config["epsilon_start"] + progress * (config["epsilon_end"] - config["epsilon_start"])
            action = policy.explore(observation, epsilon, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            visitation[(*observation, action)] += 1
            policy.update(observation, action, reward, next_obs, terminated, config["learning_rate"], config["discount"])
            steps += 1
            total_return += reward
            if info["outcome"] == "success":
                key = goal_key(observation)
                goal_discoveries[key] = goal_discoveries.get(key, 0) + 1
            observation = next_obs
            if terminated or truncated:
                row = {"episode": len(log) + 1, "scenario_id": info["scenario_id"], "end_interaction": interaction,
                       "steps": steps, "return": total_return, "outcome": info["outcome"], "epsilon": epsilon}
                writer.writerow(row)
                log.append(row)
                observation, _ = env.reset()
                key = goal_key(observation)
                goal_samples[key] = goal_samples.get(key, 0) + 1
                steps, total_return, episode_started = 0, 0.0, interaction
            if interaction % config["monitor_every"] == 0 or interaction == config["interactions"]:
                elapsed_training += perf_counter() - start
                summary, _, _ = evaluate(task, monitoring, policy, config["seed"], "fixed monitoring subset", visitation)
                monitoring_log.append({"training_interactions": interaction, **summary})
                write_json(directory / "monitoring.json", monitoring_log)
                # Recoverable inference snapshot, explicitly not exact-resume state.
                policy.save(directory / "checkpoint.npz", {"training_interactions": interaction, "config": config})
                file.flush()
                start = perf_counter()
        elapsed_training += perf_counter() - start
    np.savez_compressed(directory / "visitation.npz", state_action_counts=visitation)
    summaries = {}
    for name, agent in (("q_learning", policy), ("bfs", BFSPolicy(task.grid)), ("random", RandomPolicy(config["seed"]))):
        summary, metrics, records = evaluate(task, validation, agent, config["seed"], "complete validation manifest", visitation if name == "q_learning" else None)
        summaries[name] = summary
        write_json(directory / f"{name}_episodes.json", metrics)
        for episode in records:
            validate_replay(task, episode)
            outcome = episode["metrics"]["outcome"]
            write_json(directory / "replays" / f"{name}_{outcome}.json", episode)
        # Complete evaluation trajectories are stored separately below, without selecting only successes.
        replay_env = NavigationEnv(task, validation)
        replay_policy = RandomPolicy(config["seed"]) if name == "random" else agent
        with (directory / f"{name}_trajectories.jsonl").open("w", encoding="utf-8") as file:
            import json
            for scenario in validation:
                file.write(json.dumps(rollout(replay_env, replay_policy, scenario, config["seed"]), allow_nan=False) + "\n")
        if name == "bfs":
            assert summary["success_rate"] == 1 and summary["spl"] == 1, "BFS correctness gate failed"
            save_figure(directory / "bfs_trajectory.png", task.grid, records[0])
        if name == "q_learning":
            save_figure(directory / "q_learning_trajectory.png", task.grid, records[0])
    # Also check every training start/goal with BFS through the same environment.
    bfs_train, _, _ = evaluate(task, train, BFSPolicy(task.grid), config["seed"], "complete training manifest")
    assert bfs_train["success_rate"] == 1 and bfs_train["spl"] == 1
    summaries["bfs_training_correctness"] = bfs_train
    successful_episodes = sum(row["outcome"] == "success" for row in log)
    goal_report = {"sampled_training_goals": goal_samples, "successful_discoveries_by_goal": goal_discoveries,
                   "distinct_goals_sampled": len(goal_samples), "distinct_goals_discovered": len(goal_discoveries),
                   "successful_training_episodes": successful_episodes,
                   "visited_state_action_entries": int(np.count_nonzero(visitation)),
                   "total_state_action_visits": int(visitation.sum()),
                   "training_seconds": elapsed_training, "training_interactions": config["interactions"],
                   "training_interactions_per_second": config["interactions"] / elapsed_training,
                   "completed_training_episodes": len(log),
                   "incomplete_budget_cutoff_episode": {"steps": steps, "return": total_return,
                       "began_at_interaction": episode_started} if steps else None,
                   "training_seconds_excludes": "monitoring, checkpoint I/O, final validation and trajectory export"}
    write_json(directory / "pilot.json", goal_report)
    write_json(directory / "validation.json", summaries)
    manifest.update(status="complete", training_interactions=config["interactions"], training_seconds=elapsed_training,
                    validation_interactions=sum(v["interactions"] for k, v in summaries.items() if k != "bfs_training_correctness"),
                    monitoring_interactions=sum(m["interactions"] for m in monitoring_log),
                    trajectory_export_interactions=sum(v["interactions"] for k, v in summaries.items() if k != "bfs_training_correctness"),
                    bfs_training_check_interactions=bfs_train["interactions"],
                    checkpoint_bytes=(directory / "checkpoint.npz").stat().st_size)
    write_json(directory / "manifest.json", manifest)
    return directory, summaries, goal_report
