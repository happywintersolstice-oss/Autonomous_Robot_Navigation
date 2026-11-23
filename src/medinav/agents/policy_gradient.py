"""PPO, A2C and Recurrent PPO  (owner: Member 3)

Policy-gradient / actor-critic methods learn the policy pi(a|s) directly
(the "actor") together with a value estimate V(s) (the "critic") that reduces
the variance of the gradient. They work with discrete or continuous actions;
we use continuous (v, w) by default so they can be ensembled with SAC/TD3.

A2C (synchronous Advantage Actor-Critic)
    One gradient step per small batch of experience, advantage
    A = r + gamma * V(s') - V(s). Fast but sensitive to step size.

PPO (Schulman et al., 2017)
    Re-uses each batch for several epochs but clips the policy ratio
    r = pi_new / pi_old to [1 - eps, 1 + eps] so a single update cannot move the
    policy too far. Usually the most stable on-policy method - our expected
    front-runner for the shortlist.

Recurrent PPO (sb3-contrib)
    PPO with an LSTM: the policy keeps a memory, which helps when sensors drop
    out (Part B missing-information experiment) and when people move.
"""

from __future__ import annotations

from stable_baselines3 import A2C, PPO

ALGORITHMS = ("ppo", "a2c", "recurrent_ppo")


def _recurrent_cls():
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:  # optional dependency
        raise ImportError("recurrent_ppo needs `pip install sb3-contrib`") from exc
    return RecurrentPPO


def build(algo: str, env, params: dict, seed: int, net_arch, tensorboard_log=None):
    arch = dict(pi=list(net_arch), vf=list(net_arch))
    if algo == "recurrent_ppo":
        return _recurrent_cls()("MlpLstmPolicy", env, seed=seed, verbose=0, tensorboard_log=tensorboard_log,
                                policy_kwargs={"net_arch": arch, "lstm_hidden_size": 64}, **params)
    cls = PPO if algo == "ppo" else A2C
    return cls("MlpPolicy", env, seed=seed, verbose=0, tensorboard_log=tensorboard_log,
               policy_kwargs={"net_arch": arch}, **params)


def load(algo: str, path):
    if algo == "recurrent_ppo":
        return _recurrent_cls().load(path)
    return (PPO if algo == "ppo" else A2C).load(path)
