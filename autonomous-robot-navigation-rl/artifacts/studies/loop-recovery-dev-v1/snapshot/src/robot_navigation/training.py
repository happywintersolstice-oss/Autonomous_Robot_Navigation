"""A shared, configuration-driven tabular loop. Evaluation cannot update tables."""
import csv
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
import yaml

from .agents import BFSPolicy, QLearning, RandomPolicy, SARSA
from .benchmark import load_scenarios
from .core import ROOT, NavigationEnv, load_task, write_json
from .episodes import validate_replay
from .visualization import save_figure
from .workflow import evaluate, source_hash
from .training_traces import trace_limit, TrainingTraces


def epsilon_at(config, interaction_index):
    progress = min(1., interaction_index / (config["interactions"] * config["epsilon_decay_fraction"]))
    return config["epsilon_start"] + progress * (config["epsilon_end"] - config["epsilon_start"])


def validate_config(config):
    trace_limit(config)
    for key in ("interactions", "monitor_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("algorithm", "q-learning") not in ("q-learning", "sarsa"):
        raise ValueError("Only Q-learning and SARSA are implemented")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not (0 < config["learning_rate"] <= 1 and 0 <= config["discount"] <= 1 and
            0 <= config["epsilon_end"] <= config["epsilon_start"] <= 1 and
            0 < config["epsilon_decay_fraction"] <= 1):
        raise ValueError("Invalid tabular parameters")
    run_id = config["run_id"]
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")


def train_loop(task, scenarios, policy, config, on_monitor=None, on_episode=None, trace=None):
    """Return diagnostics; callbacks see the table but never training RNG/env.

    SARSA's continuation action is sampled pre-update and carried unchanged.
    Truncation samples a bootstrap action, then discards it when resetting.
    """
    env = NavigationEnv(task, scenarios)
    rng = np.random.default_rng(config["seed"])
    observation, _ = env.reset(seed=config["seed"])
    if trace:
        trace.start(observation, env.scenario, 0)
    action = policy.explore(observation, epsilon_at(config, 0), rng)
    visitation = np.zeros_like(policy.q, dtype=np.int64)
    samples, discoveries = {}, {}
    steps, episode_return, episodes, successes = 0, 0., 0, 0
    goal_key = lambda obs: f"{int(obs[2])},{int(obs[3])}"
    elapsed = 0.
    started = perf_counter()
    for index in range(config["interactions"]):
        if steps == 0:
            key = goal_key(observation)
            samples[key] = samples.get(key, 0) + 1
        next_obs, reward, terminated, truncated, info = env.step(action)
        if trace:
            trace.transition(observation, action, reward, next_obs, terminated, truncated, info, index+1)
        visitation[(*observation, action)] += 1
        next_epsilon = epsilon_at(config, index + 1)
        next_action = None
        if isinstance(policy, SARSA) and not terminated:
            next_action = policy.explore(next_obs, next_epsilon, rng)
        arguments = (observation, action, reward, next_obs, terminated,
                     config["learning_rate"], config["discount"])
        if isinstance(policy, SARSA):
            policy.update(*arguments, next_action=next_action)
        else:
            policy.update(*arguments)
        steps += 1
        episode_return += reward
        if info["outcome"] == "success":
            key = goal_key(observation)
            discoveries[key] = discoveries.get(key, 0) + 1
            successes += 1
        if terminated or truncated:
            episodes += 1
            if on_episode:
                on_episode({"episode": episodes, "scenario_id": info["scenario_id"],
                            "end_interaction": index + 1, "steps": steps, "return": episode_return,
                            "outcome": info["outcome"], "epsilon": epsilon_at(config, index)})
            steps, episode_return = 0, 0.
            if index + 1 < config["interactions"]:
                observation, _ = env.reset()
                if trace:
                    trace.start(observation, env.scenario, index+1)
                # Always choose on the NEW reset state, even after truncation.
                action = policy.explore(observation, next_epsilon, rng)
        else:
            observation = next_obs
            if index + 1 < config["interactions"]:
                action = next_action if isinstance(policy, SARSA) else policy.explore(observation, next_epsilon, rng)
        if (index + 1) % config["monitor_every"] == 0 or index + 1 == config["interactions"]:
            elapsed += perf_counter() - started
            if on_monitor:
                on_monitor(index + 1, visitation)
            started = perf_counter()
    elapsed += perf_counter() - started
    return visitation, {
        "sampled_training_goals": samples, "successful_discoveries_by_goal": discoveries,
        "distinct_goals_sampled": len(samples), "distinct_goals_discovered": len(discoveries),
        "successful_training_episodes": successes, "completed_training_episodes": episodes,
        "visited_state_action_entries": int(np.count_nonzero(visitation)),
        "total_state_action_visits": int(visitation.sum()), "training_interactions": config["interactions"],
        "incomplete_budget_cutoff_episode": {"steps": steps, "return": episode_return,
            "scenario_id": env.scenario["id"], "outcome": "budget_cutoff; not an environment terminal"} if steps else None,
        "training_seconds": elapsed, "training_interactions_per_second": config["interactions"] / elapsed,
        "training_seconds_excludes": "environment construction, monitoring/checkpoints, final validation/export, figures"}


def train_tabular(config_path=None, root=ROOT, config=None):
    config = dict(config) if config is not None else yaml.safe_load(
        (Path(config_path) if config_path else root / "configs/smoke.yaml").read_text(encoding="utf-8"))
    config.setdefault("algorithm", "q-learning")
    validate_config(config)
    task = load_task(root)
    train = load_scenarios(task, "train", root)
    validation = load_scenarios(task, "validation", root)
    monitoring = load_scenarios(task, "monitoring", root)
    directory = root / "artifacts/runs" / config["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    algorithm = config["algorithm"]
    key = "q_learning" if algorithm == "q-learning" else "sarsa"
    policy = {"q-learning": QLearning, "sarsa": SARSA}[algorithm](task, f"{algorithm}/{config['run_id']}/final")
    write_json(directory / "configuration.json", {"training": config, "environment": task.config, "signature": task.signature})
    for folder in ("src", "app", "configs"):
        shutil.copytree(root / folder, directory / "source" / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    manifest = {"run_id": config["run_id"], "status": "running", "algorithm": algorithm,
        "started_utc": datetime.now(timezone.utc).isoformat(), "signature": task.signature,
        "source_hash": source_hash(root), "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "dependencies": {p: version(p) for p in ("numpy", "gymnasium", "PyYAML", "matplotlib", "streamlit")},
        "training_seed": config["seed"], "training_scenario_ids": [s["id"] for s in train],
        "validation_scenario_ids": [s["id"] for s in validation],
        "selection_rule": "final-budget checkpoint; no selection/tuning in development pilots",
        "information": "robot/goal coordinates only; no masking/planner fallback",
        "final_test_evaluated": False, "inference_only_checkpoint": True}
    write_json(directory / "manifest.json", manifest)
    monitoring_log = []
    training_progress = 0
    trace = TrainingTraces(task, directory, config)
    try:
        with (directory / "training.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "scenario_id", "end_interaction", "steps", "return", "outcome", "epsilon"])
            writer.writeheader()

            def monitor(interaction, visitation):
                nonlocal training_progress
                training_progress = interaction
                summary, _, _ = evaluate(task, monitoring, policy, config["seed"], "fixed monitoring subset", visitation)
                monitoring_log.append({"training_interactions": interaction, **summary})
                write_json(directory / "monitoring.json", monitoring_log)
                metadata = {"training_interactions": interaction, "config": config}
                policy.save(directory / f"checkpoints/step-{interaction:07d}.npz", metadata)
                policy.save(directory / "checkpoint.npz", metadata)
                file.flush()

            visitation, pilot = train_loop(task, train, policy, config, monitor, writer.writerow, trace)
        manifest['training_traces'] = trace.finish()
        np.savez_compressed(directory / "visitation.npz", state_action_counts=visitation)
        write_json(directory / "pilot.json", pilot)
        agents = [(key, policy)]
        if config.get("include_references", True):
            agents.extend([("bfs", BFSPolicy(task.grid)), ("random", RandomPolicy(config["seed"]))])
        summaries = {}
        for name, agent in agents:
            summary, metrics, records = evaluate(task, validation, agent, config["seed"],
                "complete validation manifest", visitation if name == key else None,
                directory / f"{name}_trajectories.jsonl.gz")
            summaries[name] = summary
            write_json(directory / f"{name}_episodes.json", metrics)
            for episode in records:
                validate_replay(task, episode)
                write_json(directory / "replays" / f"{name}_{episode['metrics']['outcome']}.json", episode)
            save_figure(directory / f"{name}_trajectory.png", task.grid, records[0])
            if name == "bfs" and (summary["success_rate"] != 1 or summary["spl"] != 1):
                raise AssertionError("BFS correctness gate failed")
        write_json(directory / "validation.json", summaries)
        manifest.update(status="complete", training_interactions=config["interactions"],
            training_seconds=pilot["training_seconds"], monitoring_interactions=sum(m["interactions"] for m in monitoring_log),
            validation_interactions=sum(m["interactions"] for m in summaries.values()),
            trajectory_export_interactions=0, checkpoint_bytes=(directory / "checkpoint.npz").stat().st_size,
            finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest['training_traces'] = trace.finish('interrupted' if isinstance(exc, KeyboardInterrupt) else 'run_failed')
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", last_monitor_interaction=training_progress,
                        resume_supported=False)
        write_json(directory / "manifest.json", manifest)
        raise
    write_json(directory / "manifest.json", manifest)
    return directory, summaries, pilot
