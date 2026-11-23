"""Evaluate every trained run (+ baselines + ensembles) under one or more test conditions.

Output: one tidy CSV, one row per episode:
    agent, algo, seed, condition, split, success, collision, return, path_efficiency, ...
Every report table and figure is computed from this file, so results are
reproducible and nothing is copied by hand.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..agents import ensemble as ens
from ..agents.registry import baseline, find_runs, load_agent
from ..env import make_env
from .rollout import evaluate


def eval_env_config(meta_env: dict, eval_env: dict, condition: dict, action_mode: str, split: str) -> dict:
    """Env used at test time = training env + experiment eval settings + condition, same action mode."""
    cfg = {**meta_env, **(eval_env or {}), **(condition or {})}
    cfg["action_mode"] = action_mode
    cfg["split"] = split
    return cfg


def load_all(runs_roots, labels=None):
    """{label: [(agent, meta), ...one per seed]} from one folder or a list of folders."""
    from ..agents.registry import read_meta

    out = defaultdict(list)
    roots = runs_roots if isinstance(runs_roots, (list, tuple)) else [runs_roots]
    for rd in [r for root in roots for r in find_runs(root)]:
        try:
            meta = read_meta(rd)
            # filter BEFORE loading: loading every network just to discard it is slow
            if labels and meta["label"] not in labels and meta["algo"] not in labels:
                continue
            agent, meta = load_agent(rd)
        except Exception as exc:  # noqa: BLE001 - a corrupt/half-copied run must not stop evaluation
            print(f"  !! skipping {rd}: cannot load ({type(exc).__name__}: {exc})", flush=True)
            continue
        out[meta["label"]].append((agent, meta))
    for label in out:
        out[label].sort(key=lambda am: am[1]["seed"])
    return dict(out)


def build_ensembles(spec_list, trained):
    """Create ensemble agents from config entries. Returns [(agent, meta), ...]."""
    built = []
    for spec in spec_list or []:
        members = [m for m in spec["members"] if m in trained]
        if len(members) != len(spec["members"]):
            print(f"  ensemble {spec['name']}: missing members {set(spec['members']) - set(members)} - skipped")
            continue
        kind = spec["type"]
        # Combining agents only makes sense if they speak the same action language.
        needed = {"q_average": "discrete", "q_average_seeds": "discrete", "majority_vote": "discrete",
                  "action_average": "continuous"}.get(kind)
        modes = {trained[m][0][1]["action_mode"] for m in members}
        if needed and modes != {needed}:
            print(f"  ensemble {spec['name']}: '{kind}' needs {needed} members, got {sorted(modes)} - skipped")
            continue
        if kind.startswith("q_average") and any(trained[m][0][1]["algo"] not in
                                                 ("dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn")
                                                 for m in members):
            print(f"  ensemble {spec['name']}: Q-averaging needs DQN-family members - skipped")
            continue
        if spec.get("per_seed", True) and kind != "q_average_seeds":
            n = min(len(trained[m]) for m in members)
            groups = [[trained[m][i] for m in members] for i in range(n)]
        else:  # one ensemble of every seed of every member
            groups = [[am for m in members for am in trained[m]]]
        for i, group in enumerate(groups):
            agents = [a for a, _ in group]
            base_meta = dict(group[0][1])
            if kind in ("q_average", "q_average_seeds"):
                agent = ens.QAverageEnsemble(agents, spec["name"])
            elif kind == "majority_vote":
                agent = ens.MajorityVoteEnsemble(agents, spec["name"])
            elif kind == "action_average":
                agent = ens.ActionAverageEnsemble(agents, spec.get("weights"), spec["name"])
            elif kind == "shield":
                n_rays = base_meta["env"].get("n_rays", 24)
                agent = ens.SafetyShieldEnsemble(agents[0], n_rays, name=spec["name"])
            else:
                raise ValueError(f"unknown ensemble type {kind}")
            # all members must see identical observations; we use the first member's env/wrappers
            built.append((agent, {**base_meta, "label": spec["name"], "algo": f"ensemble:{kind}", "seed": i}))
    return built


def evaluate_experiment(cfg: dict, runs_root, conditions: dict | None = None, split="test",
                        n_episodes=50, labels=None, include_baselines=True, out_csv=None) -> pd.DataFrame:
    eval_cfg = cfg.get("evaluation") or {}
    eval_env = eval_cfg.get("env") or {}
    conditions = conditions or {"nominal": {}}
    trained = load_all(runs_root, labels)
    entries = [am for runs in trained.values() for am in runs]
    entries += build_ensembles(cfg.get("ensembles"), trained)
    if include_baselines:
        base_env = {k: v for k, v in (cfg.get("env") or {}).items()}
        for name in ("astar", "random"):
            entries.append((baseline(name), {"label": name, "algo": name, "seed": 0, "action_mode": "discrete",
                                             "env": base_env, "wrappers": {}}))

    if not entries:
        raise RuntimeError("Nothing to evaluate: no trained runs found and baselines disabled")
    frames = []
    for cond_name, cond in conditions.items():
        for agent, meta in entries:
            if meta["algo"] in ("astar", "random"):
                # Fresh baseline per condition: the random agent owns an RNG, and re-using one instance
                # would make its scores depend on how many episodes ran before (i.e. condition order).
                agent = baseline(meta["algo"])
            env_cfg = eval_env_config(meta["env"], eval_env, cond, meta["action_mode"], split)
            env = make_env(env_cfg, meta.get("wrappers"))
            rows = evaluate(agent, env, split, n_episodes)
            df = pd.DataFrame(rows)
            df.insert(0, "condition", cond_name)
            df.insert(0, "seed", meta["seed"])
            df.insert(0, "algo", meta["algo"])
            df.insert(0, "agent", meta["label"])
            frames.append(df)
            print(f"  {cond_name:>18} | {meta['label']:<22} seed {meta['seed']}: "
                  f"success {df['success'].mean():.2f}  collision {df['collision'].mean():.2f}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)
    return result


def _evaluate_one_condition(args):
    """Worker for evaluate_parallel. Module-level so Windows 'spawn' processes can import it;
    each worker loads its own copies of the (small) models instead of pickling them."""
    cfg, runs_root, cond_name, cond, split, n_episodes, labels = args
    return evaluate_experiment(cfg, runs_root, {cond_name: cond}, split, n_episodes, labels)


def evaluate_parallel(cfg: dict, runs_root, conditions: dict, split="test", n_episodes=50, labels=None,
                      jobs: int = 1, out_csv=None) -> pd.DataFrame:
    """Same output as evaluate_experiment, but each test condition runs in its own process.

    Conditions are independent (same agents, same map seeds, different world settings), so this is
    an embarrassingly parallel job: 10 conditions on a 20-thread laptop finish ~8x faster.
    Row order is fixed by condition order, so the CSV is identical to a sequential run.
    """
    from concurrent.futures import ProcessPoolExecutor

    jobs = max(1, min(int(jobs), len(conditions)))
    if jobs == 1:
        return evaluate_experiment(cfg, runs_root, conditions, split, n_episodes, labels, out_csv=out_csv)
    tasks = [(cfg, runs_root, name, cond, split, n_episodes, labels) for name, cond in conditions.items()]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        frames = list(pool.map(_evaluate_one_condition, tasks))   # map keeps the input order
    result = pd.concat(frames, ignore_index=True)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)
    return result
