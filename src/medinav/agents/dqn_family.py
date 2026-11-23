"""DQN, Double DQN and Dueling DQN  (owner: Member 2)

DQN (Mnih et al., 2015)
    A neural network approximates Q(s, a) directly from the raw 29-value
    observation - no hand-made discretisation like the tabular agents.
    Stabilisers: experience replay (break correlation between samples) and a
    target network (a slowly updated copy used to compute targets).
    target = r + gamma * max_a' Q_target(s', a')

Double DQN (van Hasselt et al., 2016)
    The max in DQN both *selects* and *evaluates* the next action with the same
    noisy network, which over-estimates Q-values. Double DQN selects with the
    online net and evaluates with the target net:
    target = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

Dueling DQN (Wang et al., 2016)
    Splits the head into a state value V(s) and an advantage A(s, a):
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    In navigation many states have similar values for all actions (open
    corridor), so learning V separately speeds learning up.

Stable-Baselines3 ships plain DQN only; the two subclasses below add the
Double and Dueling variants with minimal, readable changes.
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn

ALGORITHMS = ("dqn", "double_dqn", "dueling_dqn", "dueling_double_dqn")


class DoubleDQN(DQN):
    """DQN whose target uses online-net action selection (only `train` changes)."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(data, "discounts", None)
            gamma = discounts if discounts is not None else self.gamma
            with th.no_grad():
                # 1) online network picks the best next action ...
                next_actions = self.q_net(data.next_observations).argmax(dim=1, keepdim=True)
                # 2) ... target network says how good that action is
                next_q = th.gather(self.q_net_target(data.next_observations), 1, next_actions)
                target_q = data.rewards + (1 - data.dones) * gamma * next_q
            current_q = th.gather(self.q_net(data.observations), 1, data.actions.long())
            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


class DuelingQNetwork(QNetwork):
    """Shared MLP trunk, then separate value and advantage heads."""

    def __init__(self, observation_space, action_space, features_extractor, features_dim,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(observation_space, action_space, features_extractor, features_dim,
                         net_arch, activation_fn, normalize_images)
        net_arch = list(net_arch or [64, 64])
        n_actions = int(action_space.n)
        trunk = create_mlp(features_dim, -1, net_arch[:-1], activation_fn) if len(net_arch) > 1 else []
        trunk_out = net_arch[-2] if len(net_arch) > 1 else features_dim
        self.trunk = nn.Sequential(*trunk)
        self.value = nn.Sequential(nn.Linear(trunk_out, net_arch[-1]), activation_fn(), nn.Linear(net_arch[-1], 1))
        self.advantage = nn.Sequential(nn.Linear(trunk_out, net_arch[-1]), activation_fn(), nn.Linear(net_arch[-1], n_actions))
        del self.q_net  # replaced by trunk + heads

    def forward(self, obs: th.Tensor) -> th.Tensor:
        h = self.trunk(self.extract_features(obs, self.features_extractor))
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


def build(algo: str, env, params: dict, seed: int, net_arch, tensorboard_log=None):
    cls = DoubleDQN if "double" in algo else DQN
    policy = DuelingDQNPolicy if "dueling" in algo else "MlpPolicy"
    return cls(policy, env, seed=seed, policy_kwargs={"net_arch": list(net_arch)},
               tensorboard_log=tensorboard_log, verbose=0, **params)


def load(algo: str, path):
    cls = DoubleDQN if "double" in algo else DQN
    custom = {"policy_class": DuelingDQNPolicy} if "dueling" in algo else None
    return cls.load(path, custom_objects=custom)


def q_values(model, obs) -> np.ndarray:
    """Q-values for one observation (used by the Q-averaging ensemble and saliency)."""
    with th.no_grad():
        t = th.as_tensor(np.asarray(obs, dtype=np.float32)[None], device=model.device)
        return model.q_net(t).cpu().numpy()[0]
