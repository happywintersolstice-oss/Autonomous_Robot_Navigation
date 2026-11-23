"""SAC and TD3 - off-policy continuous control  (owner: Member 4)

Both learn a critic Q(s, a) from a replay buffer (sample-efficient, like DQN)
and an actor that outputs smooth (v, w) commands - closer to how a real robot
is driven than 5 discrete moves.

TD3 (Fujimoto et al., 2018)
    Deterministic actor + three tricks against Q over-estimation:
    twin critics (take the min), delayed actor updates, target policy smoothing.
    Exploration comes from Gaussian noise added to the actions.

SAC (Haarnoja et al., 2018)
    Stochastic actor that maximises reward + alpha * entropy. The entropy bonus
    keeps exploring automatically (alpha is tuned automatically), which is often
    more robust to hyper-parameters than TD3.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

ALGORITHMS = ("sac", "td3")


def build(algo: str, env, params: dict, seed: int, net_arch, tensorboard_log=None):
    params = dict(params)
    if algo == "td3":
        sigma = float(params.pop("action_noise_std", 0.2))
        params["action_noise"] = NormalActionNoise(np.zeros(2), sigma * np.ones(2))
        return TD3("MlpPolicy", env, seed=seed, verbose=0, tensorboard_log=tensorboard_log,
                   policy_kwargs={"net_arch": list(net_arch)}, **params)
    return SAC("MlpPolicy", env, seed=seed, verbose=0, tensorboard_log=tensorboard_log,
               policy_kwargs={"net_arch": list(net_arch)}, **params)


def load(algo: str, path):
    return (SAC if algo == "sac" else TD3).load(path)
