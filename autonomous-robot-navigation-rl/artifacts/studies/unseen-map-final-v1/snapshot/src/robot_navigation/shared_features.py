"""Full-map, robot-centered features shared by all four M11 algorithms."""
import numpy as np

from .core import MOVES
from .unseen_benchmark import table_key

FEATURE_VERSION = 'egocentric-map-goal-v1'
FEATURE_COUNT = 1234

# Precompute coordinates for every possible robot position. These tables are
# fixed geometry, independent of any training, validation or final layout.
RELATIVE = np.array([(r,c) for r in range(-9,10) for c in range(-9,10)])
POSITIONS = np.array([(r,c) for r in range(10) for c in range(10)])
ABSOLUTE = POSITIONS[:,None,:] + RELATIVE[None,:,:]
OUTSIDE = ((ABSOLUTE < 0) | (ABSOLUTE > 9)).any(axis=2)
LOOKUP = np.clip(ABSOLUTE[:,:,0],0,9)*10 + np.clip(ABSOLUTE[:,:,1],0,9)
NEIGHBORS = np.array([(dr+9)*19+dc+9 for dr,dc in MOVES])
RAYS = np.array([[(dr*step+9)*19+dc*step+9 for step in range(1,10)] for dr,dc in MOVES])
MASK_WEIGHTS = np.array([1,2,4,8])


def encode_features(observation):
    _, coordinates = table_key(observation)
    r,c,gr,gc = coordinates
    position = r*10+c
    outside = OUTSIDE[position]
    walls = np.asarray(observation)[4:][LOOKUP[position]] * ~outside
    blocked = np.maximum(walls,outside)
    clearance = np.cumprod(1-blocked[RAYS],axis=1).sum(axis=1)/9
    dr,dc = gr-r,gc-c
    sector = (int(np.sign(dr))+1)*3+int(np.sign(dc))+1
    local_mask = int(blocked[NEIGHBORS] @ MASK_WEIGHTS)
    features = np.zeros(FEATURE_COUNT,dtype=np.float64)
    features[:3] = (1,dr/9,dc/9)
    features[3:7] = clearance
    features[7+(dr+9)*19+dc+9] = 1
    features[368+sector*16+local_mask] = 1
    features[512:873] = walls/np.sqrt(21)
    features[873:] = outside/np.sqrt(261)
    return features
