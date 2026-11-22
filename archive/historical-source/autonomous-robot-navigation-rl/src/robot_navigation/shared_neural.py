"""Vectorized fixed features over raw 104-value SB3 observations."""
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.functional import one_hot

from .shared_features import FEATURE_COUNT, LOOKUP, OUTSIDE, NEIGHBORS, RAYS, MASK_WEIGHTS


class SharedMapExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space,features_dim=FEATURE_COUNT)
        self.register_buffer('lookup',torch.tensor(LOOKUP,dtype=torch.long))
        self.register_buffer('outside',torch.tensor(OUTSIDE,dtype=torch.float32))
        self.register_buffer('neighbors',torch.tensor(NEIGHBORS,dtype=torch.long))
        self.register_buffer('rays',torch.tensor(RAYS,dtype=torch.long))
        self.register_buffer('mask_weights',torch.tensor(MASK_WEIGHTS,dtype=torch.float32))

    def forward(self, observations):
        coordinates = torch.round(observations[:,:4]*9).long()
        position = coordinates[:,0]*10+coordinates[:,1]
        outside = self.outside[position]
        walls = torch.gather(observations[:,4:],1,self.lookup[position])*(1-outside)
        blocked = torch.maximum(walls,outside)
        clearance = torch.cumprod(1-blocked[:,self.rays],dim=2).sum(dim=2)/9
        offsets = coordinates[:,2:]-coordinates[:,:2]
        goal_index = (offsets[:,0]+9)*19+offsets[:,1]+9
        sector = (torch.sign(offsets[:,0])+1)*3+torch.sign(offsets[:,1])+1
        mask = (blocked[:,self.neighbors]*self.mask_weights).sum(dim=1).long()
        return torch.cat((torch.ones_like(observations[:,:1]), offsets.float()/9, clearance,
                          one_hot(goal_index,361).float(),one_hot(sector*16+mask,144).float(),
                          walls/np.sqrt(21),outside/np.sqrt(261)),dim=1)
