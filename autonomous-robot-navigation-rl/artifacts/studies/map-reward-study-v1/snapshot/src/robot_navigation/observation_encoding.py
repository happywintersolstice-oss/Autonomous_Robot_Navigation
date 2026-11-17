"""Fixed neural feature maps; no new information, fitted statistics or map input."""
import numpy as np

COORDINATES = 'normalized-coordinates-v1'
ABSOLUTE_RELATIVE = 'absolute-relative-coordinates-v1'
DIMENSIONS = {COORDINATES: 4, ABSOLUTE_RELATIVE: 6}


def encode(observation, shape, encoding=COORDINATES):
    if encoding not in DIMENSIONS:
        raise ValueError('Unsupported observation encoding')
    h, w = shape
    scale = np.array([max(1, h-1), max(1, w-1)]*2, dtype=np.float32)
    absolute = np.asarray(observation, dtype=np.float32)/scale
    if encoding == COORDINATES:
        return absolute
    return np.concatenate((absolute, absolute[2:]-absolute[:2])).astype(np.float32)
