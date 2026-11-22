"""Versioned action-local goal features; every value comes from the public 409-vector.
No search, action mask, teacher, blocker type, future schedule or hidden geometry.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import DQN,PPO
from stable_baselines3.dqn.policies import DQNPolicy,QNetwork
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .partial_contract import signature,MOVES
from .partial_observation import validate
from .partial_agents import LinearPolicy,metadata
from .partial_neural import NeuralPolicy
from .core import read_json,write_json
from .release import sha
VERSION='partial-action-local-goal-v2'
ROW=96;FEATURE_COUNT=5*ROW

def task_signature(config):return {**signature(config),'feature_version':VERSION}

def torch_rows(observations):
    x=observations;batch=x.shape[0];pos=torch.round(x[:,:2]*9).long();goal=torch.round(x[:,2:4]*9).long();moves=torch.tensor(MOVES,device=x.device);candidate=pos[:,None,:]+moves[None,:,:];inside=((candidate>=0)&(candidate<10)).all(dim=2);idx=candidate[:,:,0].clamp(0,9)*10+candidate[:,:,1].clamp(0,9)
    occupancy=x[:,8:108].gather(1,idx);occupancy=torch.where(inside,occupancy,torch.full_like(occupancy,2));occ=F.one_hot((occupancy+1).long(),4).float()
    relative=(goal[:,None,:]-candidate).float();distance=relative.abs().sum(2);old=(goal-pos).abs().sum(1).float();progress=(old[:,None]-distance).clamp(-1,1);pc=F.one_hot((progress+1).long(),3).float()
    visits=x[:,208:308].gather(1,idx);age=x[:,108:208].gather(1,idx);blocked=x[:,308:408].gather(1,idx);wait=torch.zeros_like(distance);wait[:,4]=1
    direction=torch.eye(5,device=x.device)[None,:,:].expand(batch,-1,-1);sensor=torch.cat((x[:,[4,5,7,6]],torch.zeros_like(x[:,:1])),1)
    reached=(distance==0).float();free=(occupancy==0).float();any_blocked=(occupancy[:,:4]>=1).any(1).float()[:,None].expand(-1,5)
    scalars=torch.stack((torch.ones_like(distance),wait,distance/18,progress,reached,visits,age,blocked,sensor,wait*any_blocked,reached*free,visits*free,progress*free),2)
    result=torch.cat((scalars,direction,relative/9,occ,pc,(pc[:,:,:,None]*occ[:,:,None,:]).flatten(2),F.one_hot(distance.long().clamp(0,18),19).float(),F.one_hot(relative[:,:,0].long().clamp(-9,9)+9,19).float(),F.one_hot(relative[:,:,1].long().clamp(-9,9)+9,19).float()),2)
    return result

def rows(observation):
    value=validate(observation)
    with torch.no_grad():return torch_rows(torch.from_numpy(value[None,:])).numpy()[0].astype(np.float64)

class LocalExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        if observation_space.shape!=(409,):raise ValueError('Requires public partial observation')
        super().__init__(observation_space,features_dim=FEATURE_COUNT)
    def forward(self,observations):return torch_rows(observations).flatten(1)

class SharedQ(QNetwork):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.q_net=nn.Sequential(nn.Linear(ROW,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,obs):return self.q_net(self.extract_features(obs,self.features_extractor).reshape(-1,5,ROW)).squeeze(-1)

class LocalDQNPolicy(DQNPolicy):
    def make_q_net(self):return SharedQ(**self._update_features_extractor(self.net_args,features_extractor=None)).to(self.device)

class SharedActorValue(nn.Module):
    latent_dim_pi=5;latent_dim_vf=64
    def __init__(self):
        super().__init__();self.actor=nn.Sequential(nn.Linear(ROW,64),nn.Tanh(),nn.Linear(64,1));self.critic=nn.Sequential(nn.Linear(FEATURE_COUNT,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
    def forward_actor(self,features):return self.actor(features.reshape(-1,5,ROW)).squeeze(-1)
    def forward_critic(self,features):return self.critic(features)
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)

class LocalPPOPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):self.mlp_extractor=SharedActorValue().to(self.device)
    def _build(self,lr_schedule):
        super()._build(lr_schedule);self.action_net=nn.Identity();self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

class LocalLinearPolicy(LinearPolicy):
    def __init__(self,algorithm,config,seed):
        super().__init__(algorithm,config,seed);self.weights=np.zeros(ROW,dtype=np.float64);self.signature=task_signature(config);self.updates=0
    def scores(self,observation):return rows(observation)@self.weights
    def update(self,observation,action,reward,next_observation,terminated,alpha,gamma,next_action=None):
        phi=rows(observation)[action];target=float(reward)
        if not terminated:
            values=self.scores(next_observation)
            if self.algorithm=='sarsa':
                if next_action not in range(5):raise ValueError('SARSA requires sampled next behavior action')
                target+=gamma*values[next_action]
            else:target+=gamma*values.max()
        self.weights+=alpha*(target-phi@self.weights)*phi/max(1.,float(phi@phi));self.updates+=1
        if not np.isfinite(self.weights).all():raise FloatingPointError('Nonfinite TD weights')
    def save(self,path,extra):
        super().save(path,extra);_patch_metadata(path)

class LocalNeuralPolicy(NeuralPolicy):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.signature={**self.signature,'feature_version':VERSION}
    def save(self,path,extra):super().save(path,extra);_patch_metadata(path)

def _patch_metadata(path):
    p=Path(path).with_suffix('.json');m=read_json(p);write_json(p,{**m,'feature_version':VERSION,'feature_count':FEATURE_COUNT,'linear_action_features':ROW,'architecture_version':'shared-action-scorer-v1'})

def build_policy(env,algorithm,settings,seed):
    torch.set_num_threads(1)
    if algorithm in ('q-learning','sarsa'):return LocalLinearPolicy(algorithm,env.unwrapped.config,seed)
    common=dict(seed=seed,device='cpu',verbose=0,learning_rate=.0003,gamma=.99,policy_kwargs={'features_extractor_class':LocalExtractor})
    if algorithm=='dqn':model=DQN(LocalDQNPolicy,env,**common,learning_starts=settings['learning_starts'],buffer_size=20000,batch_size=64,train_freq=4,gradient_steps=1,target_update_interval=500,exploration_initial_eps=1.,exploration_final_eps=.05,exploration_fraction=.65)
    elif algorithm=='ppo':model=PPO(LocalPPOPolicy,env,**common,n_steps=settings['n_steps'],batch_size=settings['batch_size'],n_epochs=4,ent_coef=.02)
    else:raise ValueError('Unknown native algorithm')
    return LocalNeuralPolicy(algorithm,model,env.unwrapped.config,seed)

def load_policy(path,config):
    path=Path(path);m=read_json(path.with_suffix('.json'))
    if m.get('signature')!=task_signature(config) or m.get('feature_version')!=VERSION or m.get('feature_count')!=FEATURE_COUNT or m.get('actions')!=5 or m.get('observation_values')!=409 or m.get('architecture_version')!='shared-action-scorer-v1' or sha(path)!=m.get('checkpoint_sha256'):raise ValueError('Product feature/task/architecture/checksum mismatch')
    algorithm=m['algorithm']
    if algorithm in ('q-learning','sarsa'):
        p=LocalLinearPolicy(algorithm,config,m['training_seed'])
        with np.load(path,allow_pickle=False) as data:
            if data['weights'].shape!=(ROW,) or not np.isfinite(data['weights']).all():raise ValueError('Invalid local TD weights')
            p.weights[:]=data['weights']
    else:
        torch.set_num_threads(1);model={'dqn':DQN,'ppo':PPO}[algorithm].load(path,device='cpu')
        if model.observation_space.shape!=(409,) or model.action_space.n!=5 or model.policy_kwargs.get('features_extractor_class') is not LocalExtractor or not isinstance(model.policy,LocalDQNPolicy if algorithm=='dqn' else LocalPPOPolicy) or any(not torch.isfinite(v).all() for v in model.policy.state_dict().values()):raise ValueError('Invalid product neural architecture')
        p=LocalNeuralPolicy(algorithm,model,config,m['training_seed'])
    p.policy_id=m['policy_id'];p.checkpoint_sha256=m['checkpoint_sha256'];return p
