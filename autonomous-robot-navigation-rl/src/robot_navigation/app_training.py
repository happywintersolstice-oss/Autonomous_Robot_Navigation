"""Explicit bounded CPU training for four native five-action RL algorithms."""

import argparse, csv, json, os, random, sys, uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from .core import ROOT, write_json, digest
from .partial_contract import configuration
from .partial_dynamic_env import PartialDynamicEnv
from .partial_agents import ALGORITHMS
from .partial_features_scaled import build_policy
from .app_registry import RUNS, registry
from .product_curriculum import (
    cases as curriculum_cases,
    CurriculumEnv,
    manifest as curriculum_manifest,
)
from . import product_features


def model_settings(preset):
    """Return legacy/Phase2 defaults; Phase3 has its separately versioned trainer."""
    return {
        "discount": 0.99,
        "linear_learning_rate": 0.1,
        "neural_learning_rate": 0.0003,
        "net_arch": [32, 32],
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "buffer_size": 4096 if preset == "quick" else 10000,
        "learning_starts": 32 if preset == "quick" else 500,
        "batch_size": 32 if preset == "quick" else 100,
        "train_freq": 4,
        "target_update_interval": 128 if preset == "quick" else 250,
        "n_steps": 32 if preset == "quick" else 200,
        "n_epochs": 2 if preset == "quick" else 5,
    }


def train(
    preset="quick",
    models=ALGORITHMS,
    actions=None,
    seed=74001,
    output=None,
    job_dir=None,
    features="product-v2",
    shaping="none",
    q_stability=True,
):
    """Route explicitly requested training to the correct versioned learner."""
    if features == "product-v3":
        from .phase3_training import train as phase3_train

        return phase3_train(
            preset, models, actions, seed, output, job_dir, objective=shaping
        )
    if preset not in ("quick", "standard"):
        raise ValueError("Unknown preset")
    models = list(models)
    actions = int((256 if preset == "quick" else 10000) if actions is None else actions)
    seed = int(seed)
    if (
        not models
        or len(set(models)) != len(models)
        or any(a not in ALGORITHMS for a in models)
    ):
        raise ValueError("Choose native RL algorithms")
    if not 1 <= actions <= 50000 or not 0 <= seed <= 2**31 - 1:
        raise ValueError(
            "Actions must be 1–50000 per model and seed a nonnegative 31-bit integer"
        )
    if features not in ("product-v2", "legacy-b1") or shaping not in (
        "none",
        "progress-v1",
    ):
        raise ValueError("Unknown versioned features or shaping")
    cfg = model_settings(preset)
    if features == "product-v2":
        cfg.update(
            net_arch=[64, 64],
            n_epochs=4,
            buffer_size=20000,
            target_update_interval=500,
            epsilon_end=0.05,
            exploration_fraction=0.65,
            dqn_batch_size=64,
        )
    if "ppo" in models and actions % cfg["n_steps"]:
        raise ValueError(
            f"PPO needs an exact multiple of {cfg['n_steps']} actions for this preset; no automatic budget overrun."
        )
    if "dqn" in models and actions % cfg["train_freq"]:
        raise ValueError("DQN actions must be a multiple of 4")
    base = Path(output or RUNS).resolve()
    forbidden = [
        ROOT / "artifacts",
        ROOT / "data",
        ROOT.parent / "archive",
        ROOT.parent / ".git",
    ]
    if any(base == p.resolve() or base.is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError(
            "Training output must be separate from archived evidence and inputs"
        )
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    folder = base / "training" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    cases = curriculum_cases("train")
    started = perf_counter()
    manifest = {
        "schema": "dynamic-app-training-v1",
        "run_id": run_id,
        "preset": preset,
        "seed": seed,
        "requested_actions_per_model": actions,
        "q_stability": "target-v1" if q_stability else "off",
        "representation": (
            product_features.VERSION if features == "product-v2" else "B1-relative-x4"
        ),
        "shaping": shaping,
        "shaping_formula": (
            "0.2 * (Manhattan(old_position,goal) - Manhattan(new_position,goal)); public observations only"
            if shaping == "progress-v1"
            else "none"
        ),
        "curriculum_version": "product-curriculum-v2",
        "development_case_hash": curriculum_manifest()["development_hash"],
        "scenario_split": "train",
        "case_ids": [c["id"] for c in cases],
        "case_hash": digest(cases),
        "settings": cfg,
        "models": {},
        "status": "running",
        "purpose": "Product development demonstration, not a new Phase 5 scientific study",
        "training_source_sha256": {
            name: __import__("hashlib")
            .sha256((ROOT / "src/robot_navigation" / name).read_bytes())
            .hexdigest()
            for name in [
                "app_training.py",
                "product_curriculum.py",
                "product_features.py",
            ]
        },
        "checkpoint_kind": "inference-only; exact resume is not supported",
    }
    write_json(folder / "training-cases.json", cases)
    job = Path(job_dir) if job_dir else folder
    job.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": "running",
        "run_id": run_id,
        "folder": str(folder),
        "algorithm": None,
        "completed_actions": 0,
        "total_actions": actions * len(models),
        "fraction": 0.0,
    }

    def publish():
        write_json(folder / "manifest.json", manifest)
        write_json(job / "progress.json", progress)

    def cancel():
        if (job / "cancel.request").exists():
            raise InterruptedError("Training cancelled by user; partial run preserved")
        if perf_counter() - started > 3600:
            raise TimeoutError("One-hour product training safety cap")

    publish()
    history = []
    try:
        for index, algorithm in enumerate(models):
            cancel()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = CurriculumEnv(actions, seed, shaping)
            policy = (
                product_features.build_policy(env, algorithm, cfg, seed)
                if features == "product-v2"
                else build_policy(env, algorithm, cfg, seed, "B1-relative-x4")
            )
            if algorithm == "q-learning" and features == "product-v2" and q_stability:
                policy.target_weights = policy.weights.copy()
            before = policy.state_fingerprint()
            initial = (
                policy.weights.copy()
                if hasattr(policy, "weights")
                else torch.cat(
                    [
                        p.detach().flatten().cpu()
                        for p in policy.model.policy.parameters()
                    ]
                )
                .numpy()
                .copy()
            )
            optimizer_steps = 0
            if hasattr(policy, "model"):
                original_step = policy.model.policy.optimizer.step

                def observed_step(*a, **kw):
                    nonlocal optimizer_steps
                    result = original_step(*a, **kw)
                    optimizer_steps += 1
                    return result

                policy.model.policy.optimizer.step = observed_step
            env.np_random = np.random.default_rng(seed)
            env.action_space.seed(seed)
            progress["algorithm"] = algorithm
            count = 0
            episodes = 0
            goals = 0
            armstart = perf_counter()
            if algorithm in ("q-learning", "sarsa"):
                rng = np.random.default_rng(seed)
                obs, _ = env.reset(seed=seed)
                action = None
                for step in range(actions):
                    cancel()
                    epsilon = 1.0 - 0.9 * min(step / (max(1, actions) * 0.8), 1.0)
                    if action is None:
                        action = policy.explore(obs, epsilon, rng)
                    nxt, reward, term, trunc, info = env.step(action)
                    count += 1
                    next_epsilon = 1.0 - 0.9 * min((step + 1) / (actions * 0.8), 1.0)
                    sampled_next = (
                        policy.explore(nxt, next_epsilon, rng)
                        if algorithm == "sarsa" and not term
                        else None
                    )
                    policy.update(
                        obs, action, reward, nxt, term, 0.1, 0.99, sampled_next
                    )
                    history.append(
                        {
                            "algorithm": algorithm,
                            "action_index": count,
                            "reward": reward,
                            "outcome": info["outcome"],
                            "official_reward": info["official_reward"],
                            "training_shaping": info["training_shaping"],
                        }
                    )
                    if term or trunc:
                        episodes += 1
                        goals += info["outcome"] == "success"
                        obs, _ = env.reset()
                        action = None
                    else:
                        obs = nxt
                        action = sampled_next if algorithm == "sarsa" else None
                    if count % 32 == 0 or count == actions:
                        progress.update(
                            completed_actions=index * actions + count,
                            fraction=(index * actions + count)
                            / (actions * len(models)),
                        )
                        publish()
            else:

                class Progress(BaseCallback):
                    def _on_step(self):
                        nonlocal count, episodes, goals
                        cancel()
                        count = int(self.model.num_timesteps)
                        for info, done, reward in zip(
                            self.locals["infos"],
                            self.locals["dones"],
                            self.locals["rewards"],
                        ):
                            history.append(
                                {
                                    "algorithm": algorithm,
                                    "action_index": count,
                                    "reward": float(reward),
                                    "outcome": info.get("outcome", "running"),
                                    "official_reward": info.get(
                                        "official_reward", float(reward)
                                    ),
                                    "training_shaping": info.get(
                                        "training_shaping", 0.0
                                    ),
                                }
                            )
                            if done:
                                episodes += 1
                                goals += info.get("outcome") == "success"
                        if count % 32 == 0 or count == actions:
                            progress.update(
                                completed_actions=index * actions + count,
                                fraction=(index * actions + count)
                                / (actions * len(models)),
                            )
                            publish()
                        return True

                policy.model.learn(
                    total_timesteps=actions,
                    callback=Progress(),
                    reset_num_timesteps=False,
                )
                if count != actions:
                    raise ValueError(
                        "Learner did not stop at the exact requested action boundary"
                    )
            target = folder / algorithm
            target.mkdir()
            checkpoint = target / (
                "policy.npz" if algorithm in ("q-learning", "sarsa") else "policy.zip"
            )
            policy.policy_id = f"{algorithm}/dynamic-app/{run_id}/seed{seed}"
            if hasattr(policy, "model"):
                policy.model.policy.optimizer.step = original_step
            final = (
                policy.weights
                if hasattr(policy, "weights")
                else torch.cat(
                    [
                        p.detach().flatten().cpu()
                        for p in policy.model.policy.parameters()
                    ]
                ).numpy()
            )
            change = float(np.linalg.norm(final - initial))
            if count != actions or change <= 0 or before == policy.state_fingerprint():
                raise ValueError("No actual parameter change or incorrect budget")
            evidence = {
                "target_weight_refreshes": getattr(policy, "target_updates", 0),
                "td_updates": count if algorithm in ("q-learning", "sarsa") else 0,
                "optimizer_steps": optimizer_steps,
                "sb3_update_counter": (
                    int(policy.model._n_updates) if hasattr(policy, "model") else None
                ),
                "parameter_l2_change": change,
                "initial_parameter_sha256": before,
                "final_parameter_sha256": policy.state_fingerprint(),
            }
            policy.save(
                checkpoint,
                {
                    "learning_evidence": evidence,
                    "shaping": shaping,
                    "run_id": run_id,
                    "training_actions": count,
                    "training_case_hash": digest(cases),
                    "purpose": manifest["purpose"],
                },
            )
            manifest["models"][algorithm] = {
                "status": "complete",
                "checkpoint": checkpoint.relative_to(folder).as_posix(),
                "actions": count,
                "training_goals": goals,
                "completed_episodes": episodes,
                "wall_seconds": perf_counter() - armstart,
                "learning_evidence": evidence,
                "effective_settings": product_features.effective_settings(
                    policy, "target-v1" if q_stability else "off"
                ),
            }
            publish()
        manifest["status"] = "complete"
        progress.update(
            status="complete", fraction=1.0, completed_actions=actions * len(models)
        )
        publish()
    except Exception as exc:
        manifest["status"] = (
            "cancelled" if isinstance(exc, InterruptedError) else "failed"
        )
        manifest["error"] = repr(exc)
        progress.update(status=manifest["status"], error=str(exc))
        publish()
        raise
    finally:
        with (folder / "training.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "algorithm",
                    "action_index",
                    "reward",
                    "outcome",
                    "official_reward",
                    "training_shaping",
                ],
            )
            w.writeheader()
            w.writerows(history)
    return {
        "run_id": run_id,
        "folder": str(folder),
        "status": manifest["status"],
        "actual_actions": sum(m["actions"] for m in manifest["models"].values()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["quick", "standard"], default="quick")
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--actions", type=int)
    p.add_argument("--seed", type=int, default=74001)
    p.add_argument("--output", type=Path)
    p.add_argument("--job-dir", type=Path)
    p.add_argument("--q-stability", choices=["target-v1", "off"], default="target-v1")
    p.add_argument(
        "--features",
        choices=["product-v3", "product-v2", "legacy-b1"],
        default="product-v2",
    )
    p.add_argument(
        "--shaping", choices=["progress-v1", "none", "liveness-v1"], default="none"
    )
    p.add_argument("--list-models", action="store_true")
    a = p.parse_args()
    if a.list_models:
        r = registry(a.output)
    else:
        selected = [s for part in a.models for s in part.split(",")]
        selected = list(ALGORITHMS) if selected == ["all"] else selected
        r = train(
            a.preset,
            selected,
            a.actions,
            a.seed,
            a.output,
            a.job_dir,
            a.features,
            a.shaping,
            a.q_stability == "target-v1",
        )
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
