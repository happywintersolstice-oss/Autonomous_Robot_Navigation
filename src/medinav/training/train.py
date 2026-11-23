"""Train every (algorithm x seed) listed in an experiment config.

Why several seeds?  RL results vary a lot between random seeds. One lucky run
proves nothing, so every algorithm is trained with N seeds and we report the
mean with a bootstrap confidence interval (the RL replacement for k-fold CV).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..agents.registry import DEFAULT_ACTION_MODE, TABULAR, build_model
from ..agents.tabular import TabularAgent
from ..env import make_env
from ..evaluation.rollout import evaluate, summarise


def resolve_env(cfg: dict, spec: dict) -> dict:
    """Experiment-wide env settings + per-run overrides + the algorithm's action mode."""
    env = dict(cfg.get("env") or {})
    env.update(spec.get("env_overrides") or {})
    env["action_mode"] = spec.get("action_mode", DEFAULT_ACTION_MODE.get(spec["algo"], "discrete"))
    return env


def _write_meta(run_dir: Path, meta: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def train_tabular(label, spec, env_cfg, tcfg, run_dir, seed):
    algo = spec["algo"]
    wrappers = spec.get("wrappers") or {}
    env = make_env({**env_cfg, "split": "train"}, wrappers)
    val_env = make_env({**env_cfg, "split": "validation"}, wrappers)
    agent = TabularAgent(algo, env_cfg.get("n_rays", 24), env_cfg.get("max_range", 3.0), env_cfg.get("dist_norm", 15.0),
                         guidance=env_cfg.get("guidance", False))
    p = spec.get("params") or {}
    t0 = time.time()
    curve, best = [], (-1, -1e9)
    run_dir.mkdir(parents=True, exist_ok=True)

    def on_eval(ep, steps, ag):
        nonlocal best
        s = summarise(evaluate(ag, val_env, "validation", tcfg.get("eval_episodes", 20)))
        curve.append({"timesteps": steps, "episodes": ep, "wall_s": round(time.time() - t0, 1), **s,
                      "difficulty": env_cfg.get("difficulty", 1)})
        if (s["success"], s["return"]) > best:
            best = (s["success"], s["return"])
            ag.save(run_dir)  # q_table.npy always holds the best-on-validation table
        print(f"  [{label} s{seed}] ep {ep:>6}  val success {s['success']:.2f}", flush=True)

    agent.train(env, int(spec.get("episodes", 3000)), seed=seed, callback=on_eval,
                callback_every=int(spec.get("eval_every_episodes", 250)), **p)
    _write_curve(run_dir, curve)
    return time.time() - t0


def train_deep(label, spec, env_cfg, tcfg, run_dir, seed, trial=None):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .callbacks import CurriculumCallback, ValidationCallback

    algo = spec["algo"]
    wrappers = spec.get("wrappers") or {}
    n_envs = int(spec.get("n_envs", tcfg.get("n_envs", 4)))

    def factory(rank):
        def _make():
            e = Monitor(make_env({**env_cfg, "split": "train"}, wrappers), info_keywords=("is_success",))
            e.reset(seed=seed * 1000 + rank)
            return e
        return _make

    vec_cls = SubprocVecEnv if tcfg.get("vec_env") == "subproc" else DummyVecEnv
    venv = vec_cls([factory(i) for i in range(n_envs)])
    val_env = make_env({**env_cfg, "split": "validation"}, wrappers)
    tb = str(run_dir.parent.parent / "tensorboard") if tcfg.get("tensorboard") and run_dir else None
    model = build_model(algo, venv, dict(spec.get("params") or {}), seed, spec.get("net_arch", [128, 128]), tb)

    callbacks = [ValidationCallback(val_env, run_dir, label, env_cfg["action_mode"],
                                    eval_freq=int(tcfg.get("eval_freq", 10_000)),
                                    n_episodes=int(tcfg.get("eval_episodes", 20)),
                                    recurrent=algo == "recurrent_ppo", trial=trial)]
    if spec.get("curriculum"):
        cur = spec["curriculum"] if isinstance(spec["curriculum"], dict) else {}
        callbacks.append(CurriculumCallback(**cur))
    t0 = time.time()
    model.learn(total_timesteps=int(spec.get("timesteps", 200_000)), callback=callbacks,
                tb_log_name=f"{label}_s{seed}")
    if run_dir:
        model.save(run_dir / "model.zip")
    venv.close()
    return time.time() - t0, callbacks[0]


def _write_curve(run_dir: Path, curve):
    if not curve:
        return
    with (run_dir / "curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)


def train_experiment(cfg: dict, out_root: Path, only=None, seeds=None, skip_existing=True):
    """Train all runs in the config. Returns list of run directories."""
    from ..config import algorithm_specs

    tcfg = cfg.get("training") or {}
    seeds = seeds if seeds is not None else tcfg.get("seeds", [0, 1, 2])
    specs = algorithm_specs(cfg, only)
    if only and not specs:
        raise ValueError(f"--only {only} matched no run label or algorithm in the config")
    done, failed = [], []
    for label, spec in specs.items():
        env_cfg = resolve_env(cfg, spec)
        for seed in seeds:
            run_dir = Path(out_root) / label / f"seed_{seed}"
            if skip_existing and (run_dir / "meta.json").exists():
                print(f"skip {label} seed {seed} (already trained)")
                done.append(run_dir)
                continue
            # A folder without meta.json is a crashed/interrupted run: clear its leftovers so a
            # stale best_model.zip or half-written curve can never be mixed with the new run.
            if run_dir.exists():
                for f in ("curve.csv", "best_model.zip", "model.zip", "q_table.npy", "meta.json"):
                    (run_dir / f).unlink(missing_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"== training {label} ({spec['algo']}) seed {seed}", flush=True)
            try:
                if spec["algo"] in TABULAR:
                    secs = train_tabular(label, spec, env_cfg, tcfg, run_dir, seed)
                else:
                    secs, _ = train_deep(label, spec, env_cfg, tcfg, run_dir, seed)
            except KeyboardInterrupt:
                raise  # let the user stop the whole experiment
            except Exception as exc:  # noqa: BLE001 - one broken run must not kill a multi-hour experiment
                # No meta.json is written, so the run counts as "not trained" and is retried next time.
                msg = f"{type(exc).__name__}: {exc}"
                print(f"!! {label} seed {seed} FAILED - {msg}", flush=True)
                failed.append({"label": label, "seed": seed, "error": msg})
                continue
            _write_meta(run_dir, {
                "label": label, "algo": spec["algo"], "owner": spec.get("owner", ""),
                "seed": seed, "action_mode": env_cfg["action_mode"], "env": env_cfg,
                "wrappers": spec.get("wrappers") or {}, "params": spec.get("params") or {},
                "net_arch": spec.get("net_arch"), "timesteps": spec.get("timesteps"),
                "episodes": spec.get("episodes"), "curriculum": spec.get("curriculum", False),
                "train_time_s": round(secs, 1),
            })
            done.append(run_dir)
    if failed:
        Path(out_root).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / "failed_runs.json").write_text(json.dumps(failed, indent=2))
        print(f"!! {len(failed)} run(s) failed - see {Path(out_root) / 'failed_runs.json'}", flush=True)
    return done
