"""Common Part A records, validation checkpoint selection, and SB3 training."""
from collections import Counter
import csv
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
from time import perf_counter

import gymnasium as gym
import numpy as np

from .agents import QLearning, SARSA, load_policy
from .benchmark import load_scenarios
from .core import ROOT, NavigationEnv, load_task, write_json
from .episodes import validate_replay
from .training import train_loop, validate_config
from .visualization import save_figure
from .workflow import evaluate, source_hash
from .training_traces import trace_limit, TrainingTraces


def checkpoint_rank(summary, interaction):
    return summary["success_rate"], summary["spl"], -interaction


class TrainingLedger(gym.Wrapper):
    """Observe the raw task for diagnostics; never change actions or outcomes."""
    def __init__(self, env, writer, trace=None):
        super().__init__(env)
        h, w = env.grid.shape
        self.visitation = np.zeros((h, w, h, w, 4), dtype=np.int64)
        self.writer, self.interactions, self.episodes, self.successes = writer, 0, 0, 0
        self.samples, self.discoveries = Counter(), Counter()
        self.steps, self.total_return = 0, 0.
        self.trace = trace

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous = observation
        self.steps, self.total_return = 0, 0.
        if self.trace:
            self.trace.start(observation, self.env.scenario, self.interactions)
        return observation, info

    def step(self, action):
        goal = f"{self.previous[2]},{self.previous[3]}"
        if self.steps == 0:
            self.samples[goal] += 1
        self.visitation[(*self.previous, int(action))] += 1
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.trace:
            self.trace.transition(self.previous, action, reward, observation, terminated, truncated, info, self.interactions+1)
        self.previous = observation
        self.interactions += 1
        self.steps += 1
        self.total_return += reward
        if info["outcome"] == "success":
            self.successes += 1
            self.discoveries[goal] += 1
        if terminated or truncated:
            self.episodes += 1
            self.writer({"episode": self.episodes, "scenario_id": info["scenario_id"],
                         "end_interaction": self.interactions, "steps": self.steps,
                         "return": self.total_return, "outcome": info["outcome"]})
            self.steps, self.total_return = 0, 0.
        return observation, reward, terminated, truncated, info

    def report(self, seconds):
        return {"training_interactions": self.interactions, "training_seconds": seconds,
            "training_interactions_per_second": self.interactions / seconds,
            "training_seconds_excludes": "construction, monitoring/checkpoint I/O, final validation/export/figures",
            "completed_training_episodes": self.episodes, "successful_training_episodes": self.successes,
            "sampled_training_goals": dict(self.samples), "successful_discoveries_by_goal": dict(self.discoveries),
            "distinct_goals_sampled": len(self.samples), "distinct_goals_discovered": len(self.discoveries),
            "visited_state_action_entries": int(np.count_nonzero(self.visitation)),
            "total_state_action_visits": int(self.visitation.sum()),
            "incomplete_budget_cutoff_episode": {"steps": self.steps, "return": self.total_return,
                "scenario_id": self.env.scenario["id"], "outcome": "budget_cutoff"} if self.steps else None}


class PartARecord:
    def __init__(self, task, config, root=ROOT, scenario_task=None):
        from .refinement import validate_task_parent
        scenario_task = scenario_task or task
        validate_task_parent(task, scenario_task)
        self.task, self.config, self.root = task, config, root
        self.directory = root / "artifacts/runs" / config["run_id"]
        self.directory.mkdir(parents=True, exist_ok=False)
        self.monitoring = load_scenarios(scenario_task, "monitoring", root)
        self.validation = load_scenarios(scenario_task, "validation", root)
        self.monitoring_log, self.best_rank = [], None
        self.monitor_seconds = 0.
        self.extension = ".zip" if config["algorithm"] in ("dqn", "ppo") else ".npz"
        self.key = config["algorithm"].replace("-", "_")
        packages = ["numpy", "gymnasium", "PyYAML", "matplotlib", "streamlit"]
        if config["algorithm"] in ("dqn", "ppo"):
            packages += ["torch", "stable-baselines3"]
        self.manifest = {"status": "running", "run_id": config["run_id"], "algorithm": config["algorithm"],
            "training_seed": config["seed"], "signature": task.signature, "source_hash": source_hash(root),
            "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "dependencies": {p: version(p) for p in packages}, "final_test_evaluated": False,
            "selection_rule": "maximum monitoring success, then SPL, then earliest interaction",
            "viewer_checkpoint": "selected" + self.extension, "inference_only_checkpoint": True,
            "scenario_signature": scenario_task.signature, "reward_variant": config.get("reward_variant", "step-cost"),
            "stage": config.get("stage", "part-a"),
            "training_scenario_ids": [s["id"] for s in load_scenarios(scenario_task, "train", root)],
            "validation_scenario_ids": [s["id"] for s in self.validation],
            "information": "robot/goal coordinates only; neural policies use fixed coordinate scaling; no masking or planner fallback"}
        write_json(self.directory / "configuration.json", {"training": config, "environment": task.config, "signature": task.signature})
        for folder in ("src", "app", "configs"):
            shutil.copytree(root / folder, self.directory / "source" / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        self.save_manifest()

    def save_manifest(self):
        write_json(self.directory / "manifest.json", self.manifest)

    def monitor(self, policy, interaction, visitation, phase="after tabular update"):
        started = perf_counter()
        policy.policy_id = f"{self.config['algorithm']}/{self.config['run_id']}/step-{interaction}"
        summary, _, _ = evaluate(self.task, self.monitoring, policy, self.config["seed"],
                                  "fixed monitoring subset", visitation)
        self.monitoring_log.append({"training_interactions": interaction, "weight_phase": phase, **summary})
        path = self.directory / f"checkpoints/step-{interaction:07d}{self.extension}"
        policy.save(path, {"training_interactions": interaction, "config": self.config, "weight_phase": phase})
        rank = checkpoint_rank(summary, interaction)
        if self.best_rank is None or rank > self.best_rank:
            self.best_rank = rank
            selected = self.directory / ("selected" + self.extension)
            shutil.copy2(path, selected)
            shutil.copy2(path.with_suffix(".json"), selected.with_suffix(".json"))
            self.manifest["selected_interactions"] = interaction
            np.savez_compressed(self.directory / "selected_visitation.npz", state_action_counts=visitation)
        if interaction == self.config["interactions"]:
            final = self.directory / ("checkpoint" + self.extension)
            shutil.copy2(path, final)
            shutil.copy2(path.with_suffix(".json"), final.with_suffix(".json"))
        self.manifest["last_monitor_interaction"] = interaction
        write_json(self.directory / "monitoring.json", self.monitoring_log)
        self.save_manifest()
        self.monitor_seconds += perf_counter() - started

    def finish(self, visitation, pilot):
        write_json(self.directory / "pilot.json", pilot)
        np.savez_compressed(self.directory / "visitation.npz", state_action_counts=visitation)
        selected_path = self.directory / self.manifest["viewer_checkpoint"]
        selected = load_policy(selected_path, self.task)
        with np.load(self.directory / "selected_visitation.npz", allow_pickle=False) as data:
            selected_visitation = data["state_action_counts"]
        summary, metrics, records = evaluate(self.task, self.validation, selected, self.config["seed"],
            "complete validation manifest; validation-selected checkpoint", selected_visitation,
            self.directory / f"{self.key}_trajectories.jsonl.gz")
        summary["initial_state_overlap_scope"] = "selected checkpoint training prefix"
        write_json(self.directory / "validation.json", {self.key: summary})
        write_json(self.directory / f"{self.key}_episodes.json", metrics)
        for episode in records:
            validate_replay(self.task, episode)
            write_json(self.directory / "replays" / f"{self.key}_{episode['metrics']['outcome']}.json", episode)
        illustration = next((r for r in records if r["metrics"]["outcome"] == "success"), records[0])
        save_figure(self.directory / f"{self.key}_trajectory.png", self.task.grid, illustration)
        self.manifest.update(status="complete", training_interactions=self.config["interactions"],
            training_seconds=pilot["training_seconds"], monitoring_seconds=self.monitor_seconds,
            monitoring_interactions=sum(m["interactions"] for m in self.monitoring_log),
            validation_interactions=summary["interactions"], trajectory_export_interactions=0,
            checkpoint_bytes=selected_path.stat().st_size, finished_utc=datetime.now(timezone.utc).isoformat())
        self.save_manifest()
        return self.directory, {self.key: summary}, pilot


def train_part_a(config, root=ROOT):
    trace_limit(config)
    if config["algorithm"] == "ppo":
        from .ppo_config import validate_config as validate_ppo
        validate_ppo(config)
    elif config["algorithm"] != "dqn":
        validate_config(config)
    else:
        from .observation_encoding import COORDINATES, DIMENSIONS
        if config.get('observation_encoding', COORDINATES) not in DIMENSIONS:
            raise ValueError('Unsupported DQN observation encoding')
        if (config["interactions"] <= 0 or config["monitor_every"] <= 0 or
                config["interactions"] % config["train_freq"] or config["monitor_every"] % config["train_freq"]):
            raise ValueError("DQN budgets and monitoring intervals must be positive multiples of train_freq")
        if Path(config["run_id"]).name != config["run_id"] or config["run_id"] in (".", ".."):
            raise ValueError("run_id must be one directory name")
    from .refinement import reward_task
    parent_task = load_task(root)
    task = reward_task(parent_task, config.get("reward_variant", "step-cost"))
    scenarios = load_scenarios(parent_task, "train", root)
    record = PartARecord(task, config, root, scenario_task=parent_task)
    record.trace = TrainingTraces(task, record.directory, config)
    try:
        with (record.directory / "training.csv").open("w", encoding="utf-8", newline="") as file:
            fields = ["episode", "scenario_id", "end_interaction", "steps", "return", "outcome"]
            if config["algorithm"] not in ("dqn", "ppo"):
                fields.append("epsilon")
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            if config["algorithm"] == "dqn":
                from stable_baselines3.common.callbacks import BaseCallback
                from .dqn import DQNPolicy, build_model
                ledger = TrainingLedger(NavigationEnv(task, scenarios), writer.writerow, record.trace)
                model = build_model(ledger, config)
                policy = DQNPolicy(task, model, f"dqn/{config['run_id']}")

                class Monitor(BaseCallback):
                    def _on_step(self):
                        if perf_counter()-started > config.get('max_wall_seconds', float('inf')):
                            raise TimeoutError('DQN run exceeded its declared wall-time limit')
                        if self.num_timesteps % config["monitor_every"] == 0 and self.num_timesteps < config["interactions"]:
                            record.monitor(policy, self.num_timesteps, ledger.visitation,
                                           "SB3 callback: before boundary transition storage/gradient/target update")
                            file.flush()
                            if config.get('log_progress'):
                                print(f"{config['run_id']}: {self.num_timesteps:,} interactions", flush=True)
                        return True

                    def _on_training_end(self):
                        record.monitor(policy, self.num_timesteps, ledger.visitation, "after final scheduled gradient update")

                started = perf_counter()
                model.learn(total_timesteps=config["interactions"], callback=Monitor())
                seconds = perf_counter() - started - record.monitor_seconds
                if model.num_timesteps != config["interactions"] or ledger.interactions != config["interactions"]:
                    raise AssertionError("DQN training budget mismatch")
                visitation, pilot = ledger.visitation, ledger.report(seconds)
                pilot.update(gradient_updates=model._n_updates, replay_size=model.replay_buffer.size(),
                             replay_capacity=model.replay_buffer.buffer_size, observation_encoding=policy.encoding,
                             input_dimensions=model.observation_space.shape[0],
                             online_parameter_count=sum(p.numel() for p in model.q_net.parameters()))
            elif config["algorithm"] == "ppo":
                from .ppo import train
                visitation, pilot = train(task, scenarios, config, record, file, writer)
            else:
                policy = {"q-learning": QLearning, "sarsa": SARSA}[config["algorithm"]](task)
                def monitor(interaction, visitation):
                    record.monitor(policy, interaction, visitation)
                    file.flush()
                visitation, pilot = train_loop(task, scenarios, policy, config, monitor, writer.writerow, record.trace)
        record.manifest['training_traces'] = record.trace.finish()
        return record.finish(visitation, pilot)
    except BaseException as exc:
        record.manifest['training_traces'] = record.trace.finish('interrupted' if isinstance(exc, KeyboardInterrupt) else 'run_failed')
        record.manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               error=f"{type(exc).__name__}: {exc}", resume_supported=False)
        record.save_manifest()
        raise
