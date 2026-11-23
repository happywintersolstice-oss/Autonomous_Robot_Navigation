"""Name -> algorithm lookup, plus saving/loading trained runs.

A trained run is a folder:
    runs/<experiment>/<label>/seed_<n>/
        meta.json        algorithm, action mode, env overrides, wrappers, hyper-parameters
        model.zip        (deep RL)   or   q_table.npy (tabular)
        best_model.zip   checkpoint with the best VALIDATION success rate
        curve.csv        validation learning curve
"""

from __future__ import annotations

import json
from pathlib import Path

from .baselines import AStarAgent, RandomAgent
from .base import SB3Agent
from .tabular import TabularAgent

TABULAR = ("q_learning", "sarsa")
DQN_FAMILY = ("dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn")
POLICY_GRADIENT = ("ppo", "a2c", "recurrent_ppo")
CONTINUOUS = ("sac", "td3")
BASELINES = ("random", "astar")
ALL = TABULAR + DQN_FAMILY + POLICY_GRADIENT + CONTINUOUS

# default action space per algorithm (PPO/A2C can be switched in the config)
DEFAULT_ACTION_MODE = {
    **{a: "discrete" for a in TABULAR + DQN_FAMILY},
    **{a: "continuous" for a in POLICY_GRADIENT + CONTINUOUS},
}


def family_module(algo: str):
    """Import lazily so numpy-only parts (tabular, baselines, app) work without torch."""
    if algo in DQN_FAMILY:
        from . import dqn_family as m
    elif algo in POLICY_GRADIENT:
        from . import policy_gradient as m
    elif algo in CONTINUOUS:
        from . import continuous_control as m
    else:
        raise KeyError(f"Unknown deep RL algorithm '{algo}'. Known: {ALL}")
    return m


def build_model(algo, env, params, seed, net_arch, tensorboard_log=None):
    return family_module(algo).build(algo, env, params, seed, net_arch, tensorboard_log)


def read_meta(run_dir) -> dict:
    return json.loads((Path(run_dir) / "meta.json").read_text())


def load_agent(run_dir, best=True):
    """Load any trained run as an `Agent`. Returns (agent, meta)."""
    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    algo = meta["algo"]
    if algo in TABULAR:
        agent = TabularAgent.load(run_dir, algo, meta.get("env", {}))
    else:
        path = run_dir / ("best_model.zip" if best and (run_dir / "best_model.zip").exists() else "model.zip")
        model = family_module(algo).load(algo, path)
        agent = SB3Agent(model, meta["label"], meta["action_mode"], recurrent=(algo == "recurrent_ppo"))
    agent.name = meta["label"]
    return agent, meta


def baseline(name: str, action_mode="discrete"):
    if name == "random":
        return RandomAgent(action_mode)
    if name == "astar":
        return AStarAgent(action_mode)
    raise KeyError(name)


def find_runs(root) -> list[Path]:
    """All run folders (those with a meta.json) under `root`."""
    return sorted(p.parent for p in Path(root).rglob("meta.json"))
