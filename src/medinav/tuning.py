"""Hyper-parameter tuning with Optuna (Part B "smarter search").

Grid search is hopeless for RL (each trial = a full training run). We compare:

    random  RandomSampler       - the spec's "randomly exploring options"
    tpe     TPESampler          - Bayesian optimisation: models which regions of the
                                  search space gave good scores and samples there
+ MedianPruner: a trial whose validation success at a checkpoint is below the median
  of earlier trials at the same checkpoint is stopped early -> more trials per hour.

The objective is the best VALIDATION success rate (+ a tiny tie-break on return).
Test maps are never touched during tuning.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from .agents.registry import TABULAR
from .config import algorithm_specs
from .training.train import resolve_env


def _suggest(trial, name, space):
    kind = space["type"]
    if kind == "loguniform":
        return trial.suggest_float(name, float(space["low"]), float(space["high"]), log=True)
    if kind == "uniform":
        return trial.suggest_float(name, float(space["low"]), float(space["high"]))
    if kind == "int":
        return trial.suggest_int(name, int(space["low"]), int(space["high"]), log=space.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(space["choices"]))
    raise ValueError(kind)


def tune(cfg: dict, label: str, sampler: str, out_root: Path, n_trials=None):
    import optuna

    from .evaluation.rollout import evaluate, summarise

    tcfg = cfg.get("tuning") or {}
    specs = algorithm_specs(cfg, [label])
    if label not in specs:
        raise KeyError(f"label '{label}' is not in the config's algorithms: {sorted(algorithm_specs(cfg))}")
    spec = specs[label]
    space = (tcfg.get("search_space") or {}).get(spec["algo"])
    if not space:
        raise KeyError(f"no tuning.search_space entry for algorithm '{spec['algo']}'")
    n_trials = n_trials or int(tcfg.get("n_trials", 20))
    env_cfg = resolve_env(cfg, spec)
    train_cfg = cfg.get("training") or {}
    out = Path(out_root) / f"{label}_{sampler}"
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        params = dict(spec.get("params") or {})
        trial_spec = dict(spec)
        for name, sp in space.items():
            value = _suggest(trial, name, sp)
            if name == "net_arch":
                trial_spec["net_arch"] = [int(x) for x in str(value).split(",")]
            else:
                params[name] = value
        trial_spec["params"] = params
        if spec["algo"] in TABULAR:
            from .agents.tabular import TabularAgent
            from .env import make_env

            env = make_env({**env_cfg, "split": "train"}, spec.get("wrappers"))
            venv = make_env({**env_cfg, "split": "validation"}, spec.get("wrappers"))
            agent = TabularAgent(spec["algo"], env_cfg.get("n_rays", 24), env_cfg.get("max_range", 3.0),
                                 env_cfg.get("dist_norm", 15.0), guidance=env_cfg.get("guidance", False))
            scores = []

            def cb(ep, steps, ag):
                s = summarise(evaluate(ag, venv, "validation", train_cfg.get("eval_episodes", 20)))
                scores.append(s["success"] + 1e-3 * s["return"])
                trial.report(s["success"], ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            agent.train(env, int(tcfg.get("episodes", spec.get("episodes", 2000))), seed=trial.number,
                        callback=cb, callback_every=int(spec.get("eval_every_episodes", 250)), **params)
            return max(scores)

        from .training.train import train_deep

        trial_spec["timesteps"] = int(tcfg.get("timesteps", 100_000))
        with tempfile.TemporaryDirectory() as tmp:
            _, val_cb = train_deep(label, trial_spec, env_cfg, train_cfg, Path(tmp), seed=trial.number, trial=trial)
        return max(h["success"] + 1e-3 * h["return"] for h in val_cb.history)

    samplers = {"tpe": optuna.samplers.TPESampler(seed=0, n_startup_trials=5),
                "random": optuna.samplers.RandomSampler(seed=0)}
    study = optuna.create_study(
        direction="maximize", sampler=samplers[sampler],
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=f"{label}_{sampler}",
        storage=f"sqlite:///{(out / 'study.db').as_posix()}", load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, catch=(ValueError, RuntimeError))
    df = study.trials_dataframe()
    df.to_csv(out / "trials.csv", index=False)
    best = dict(study.best_params)
    snippet = {"params": {k: v for k, v in best.items() if k != "net_arch"}}
    if "net_arch" in best:
        snippet["net_arch"] = [int(x) for x in str(best["net_arch"]).split(",")]
    (out / "best_params.yaml").write_text(yaml.safe_dump({label + "_tuned": {"algo": spec["algo"], **snippet}}))
    (out / "summary.json").write_text(json.dumps({
        "label": label, "sampler": sampler, "best_value": study.best_value, "best_params": best,
        "n_trials": len(study.trials),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    }, indent=2))
    print(f"best {label} ({sampler}): {study.best_value:.3f} with {best}")
    return study
