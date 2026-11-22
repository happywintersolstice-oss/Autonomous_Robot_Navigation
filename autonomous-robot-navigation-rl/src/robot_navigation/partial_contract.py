"""Identity of the new task; no historical task or learned checkpoint is adapted."""

from copy import deepcopy

import numpy as np
import yaml

from .core import ROOT, digest

BENCHMARK = "partial-dynamic-navigation-v1"
ENVIRONMENT = "partial-dynamic-env-v1"
OBSERVATION = "partial-memory-409-v2"
SENSOR = "cardinal-rays-range2-v1"
ACTION = "up-down-left-right-wait-v1"
REWARD = "partial-step-goal-collision-v1"
FEATURE = "partial-visible-affine-v1"
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ACTION_NAMES = ("Up", "Down", "Left", "Right", "Wait")
VERSIONS = dict(
    benchmark_id=BENCHMARK,
    environment_version=ENVIRONMENT,
    observation_version=OBSERVATION,
    sensor_version=SENSOR,
    action_version=ACTION,
    reward_version=REWARD,
    feature_version=FEATURE,
)


def configuration(root=ROOT):
    return validate_config(
        yaml.safe_load(
            (root / "configs/partial_dynamic.yaml").read_text(encoding="utf-8")
        )
    )


def validate_config(config):
    config = deepcopy(config)
    if any(config.get(k) != v for k, v in VERSIONS.items()) or config.get("shape") != [
        10,
        10,
    ]:
        raise ValueError("Unsupported partial/dynamic contract")
    if type(config["max_steps"]) is not int or not 1 <= config["max_steps"] <= 200:
        raise ValueError(
            "External cap must be an integer from 1 to 200 for this encoding"
        )
    for key in (
        "step_reward",
        "success_bonus",
        "collision_penalty",
        "extra_timeout_penalty",
    ):
        if not np.isfinite(config[key]):
            raise ValueError("Rewards must be finite")
    if config["extra_timeout_penalty"] != 0:
        raise ValueError("This version has no timeout penalty")
    return config


def signature(config):
    config = validate_config(config)
    fields = {
        **VERSIONS,
        "shape": [10, 10],
        **{
            k: config[k]
            for k in (
                "max_steps",
                "step_reward",
                "success_bonus",
                "collision_penalty",
                "extra_timeout_penalty",
            )
        },
    }
    return {**VERSIONS, "task_hash": digest(fields)}
