"""The common 409-value agent-facing input, never a true-map encoder."""

import gymnasium as gym
import numpy as np

DIMENSIONS = 409
SPACE = gym.spaces.Box(-1.0, 1.0, shape=(DIMENSIONS,), dtype=np.float32)


def validate(observation):
    value = np.asarray(observation)
    if (
        value.shape != (DIMENSIONS,)
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError("Observation must be a finite float32 vector of length 409")
    if not SPACE.contains(value) or (value[:8] < 0).any() or (value[108:] < 0).any():
        raise ValueError("Partial observation is out of range")
    if (
        not np.isin(value[8:108], (-1, 0, 1)).all()
        or not np.isin(value[4:8], (0, 0.5, 1)).all()
    ):
        raise ValueError("Invalid occupancy or sensor encoding")
    coordinates = np.rint(value[:4] * 9).astype(int)
    if not np.allclose(value[:4], coordinates / 9, atol=1e-7, rtol=0):
        raise ValueError("Coordinates must denote grid cells")
    return value


def encode(memory, position, goal, distances, step):
    if memory.scan_step != step:
        raise ValueError("Observation requires the current scan")
    age = np.where(
        memory.last_seen_step >= 0, np.minimum(step - memory.last_seen_step, 20) / 20, 0
    )
    result = np.concatenate(
        (
            np.asarray((*position, *goal)) / 9,
            distances,
            memory.occupancy.ravel(),
            age.ravel(),
            np.minimum(memory.visit_count, 5).ravel() / 5,
            np.minimum(memory.consecutive_blocked_observations, 5).ravel() / 5,
            [min(step, 200) / 200],
        )
    ).astype(np.float32)
    return validate(result)
