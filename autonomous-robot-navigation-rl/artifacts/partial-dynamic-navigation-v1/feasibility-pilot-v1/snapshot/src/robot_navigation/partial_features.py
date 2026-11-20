"""Deterministic shared affine features derived exclusively from permitted input."""
import numpy as np

from .partial_contract import FEATURE
from .partial_observation import validate

FEATURE_COUNT = 412


def encode_features(observation):
    value = validate(observation)
    return np.concatenate(([1.], value, value[2:4]-value[:2])).astype(np.float64)
