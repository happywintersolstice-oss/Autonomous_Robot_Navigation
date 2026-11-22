"""F3 state is built ONLY from current public409 and past public decisions.
No geometry/event/diagnostic argument exists in this module's policy interface.
"""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from stable_baselines3 import DQN,PPO
from stable_baselines3.dqn.policies import DQNPolicy,QNetwork
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .partial_contract import signature,MOVES
from .partial_observation import validate
from .partial_agents import LinearPolicy,metadata
from .partial_neural import NeuralPolicy
from .product_features import torch_rows as f2_rows
from .core import read_json,write_json
from .release import sha
VERSION='partial-action-public-history8-v3';HISTORY_VERSION='public-decision-history8-v1'
STATE=431;ROW=104;FEATURE_COUNT=520

def task_signature(config):return {**signature(config),'feature_version':VERSION,'agent_state_version':HISTORY_VERSION,'agent_state_values':STATE}

class PublicHistory:
    def __init__(self):self.positions=deque(maxlen=8);self.last_action=None;self.waits=0
    def state(self,observation):
        obs=validate(observation);positions=np.full((8,2),-1.,dtype=np.float32)
        if self.positions:positions[-len(self.positions):]=np.asarray(self.positions)
        action=np.zeros(5,dtype=np.float32)
        if self.last_action is not None:action[self.last_action]=1
        return np.concatenate((obs,positions.ravel(),action,[min(self.waits,20)/20])).astype(np.float32)
    def remember(self,observation,action):
        obs=validate(observation)
        if type(action) is not int or action not in range(5):raise ValueError('Five public actions required')
        self.positions.append(obs[:2].copy());self.last_action=action;self.waits=self.waits+1 if action==4 else 0
    def observed_alternative(self,observation):
        obs=validate(observation);pos=np.rint(obs[:2]*9).astype(int);past=[tuple(np.rint(p*9).astype(int)) for p in self.positions]
        for action,(dr,dc) in enumerate(MOVES[:4]):
            cell=(int(pos[0]+dr),int(pos[1]+dc));ray=obs[[4,5,7,6][action]]
            if all(0<=v<10 for v in cell) and ray!=.5 and obs[8+cell[0]*10+cell[1]]==0 and cell not in past:return True
        return False

def validate_state(value):
    s=np.asarray(value)
    if s.shape!=(STATE,) or s.dtype!=np.float32 or not np.isfinite(s).all():raise ValueError('F3 requires versioned431 public-history state')
    validate(s[:409])
    if (s[409:]<-1).any() or (s[409:]>1).any():raise ValueError('Invalid public history range')
    return s

def torch_rows(states):
    obs=states[:,:409];base=f2_rows(obs);batch=len(states);position=torch.round(obs[:,:2]*9).long();moves=torch.tensor(MOVES,device=states.device);candidate=position[:,None,:]+moves[None,:,:]
    past=torch.round(states[:,409:425].reshape(-1,8,2)*9).long();valid=(past>=0).all(2);matches=(candidate[:,:,None,:]==past[:,None,:,:]).all(3)&valid[:,None,:];recent=matches.float().sum(2)/8
    reversal=matches[:,:,-1].float();same_action=states[:,425:430];waits=states[:,430:431].expand(-1,5);is_wait=torch.zeros_like(waits);is_wait[:,4]=1
    old=(torch.round(obs[:,2:4]*9).long()-position).abs().sum(1);distance=(torch.round(obs[:,None,2:4]*9).long()-candidate).abs().sum(2);approaches=(distance<old[:,None]).float()
    idx=candidate[:,:,0].clamp(0,9)*10+candidate[:,:,1].clamp(0,9);free=(obs[:,8:108].gather(1,idx)==0).float()*((candidate>=0)&(candidate<10)).all(2).float()
    current=((past==position[:,None,:]).all(2)&valid).float().sum(1)[:,None].expand(-1,5)/8
    alternatives=((free[:,:4]>0)&(recent[:,:4]==0)).float().sum(1)[:,None].expand(-1,5)/4
    extra=torch.stack((recent,reversal,same_action,is_wait*waits,is_wait*alternatives,free*recent,waits,current),2)
    return torch.cat((base,extra),2)

def rows(state):
    value=validate_state(state)
    with torch.no_grad():return torch_rows(torch.from_numpy(value[None,:])).numpy()[0].astype(np.float64)

class HistoryExtractor(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        if observation_space.shape!=(STATE,):raise ValueError('F3 requires explicit agent-state space')
        super().__init__(observation_space,features_dim=FEATURE_COUNT)
    def forward(self,observations):return torch_rows(observations).flatten(1)

class HistoryQ(QNetwork):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.q_net=nn.Sequential(nn.Linear(ROW,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,obs):return self.q_net(self.extract_features(obs,self.features_extractor).reshape(-1,5,ROW)).squeeze(-1)

class HistoryDQNPolicy(DQNPolicy):
    def make_q_net(self):return HistoryQ(**self._update_features_extractor(self.net_args,features_extractor=None)).to(self.device)

class HistoryActorValue(nn.Module):
    latent_dim_pi=5;latent_dim_vf=64
    def __init__(self):
        super().__init__();self.actor=nn.Sequential(nn.Linear(ROW,64),nn.Tanh(),nn.Linear(64,1));self.critic=nn.Sequential(nn.Linear(FEATURE_COUNT,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
    def forward_actor(self,features):return self.actor(features.reshape(-1,5,ROW)).squeeze(-1)
    def forward_critic(self,features):return self.critic(features)
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)

class HistoryPPOPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):self.mlp_extractor=HistoryActorValue().to(self.device)
    def _build(self,lr_schedule):
        super()._build(lr_schedule);self.action_net=nn.Identity();self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

def patch_metadata(policy,path,extra):
    m=metadata(policy,path,extra);m.update(feature_version=VERSION,feature_count=FEATURE_COUNT,agent_state_values=STATE,agent_state_version=HISTORY_VERSION,linear_action_features=ROW,architecture_version='history-shared-action-scorer-v1');write_json(Path(path).with_suffix('.json'),m)

class HistoryLinearPolicy(LinearPolicy):
    def __init__(self,algorithm,config,seed=0):
        super().__init__(algorithm,config,seed);self.weights=np.zeros(ROW,dtype=np.float64);self.signature=task_signature(config);self.history=PublicHistory();self.updates=0;self.target_updates=0;self.target_weights=self.weights.copy() if algorithm=='q-learning' else None
    def reset(self):self.history=PublicHistory()
    def scores(self,state):return rows(state)@self.weights
    def act(self,observation):
        state=self.history.state(observation);action=int(np.argmax(self.scores(state)));self.history.remember(observation,action);return action
    def update(self,state,action,reward,next_state,terminated,alpha=.1,gamma=.99,next_action=None):
        phi=rows(state)[action];target=float(reward)
        if not terminated:
            scores=self.scores(next_state)
            if self.algorithm=='sarsa':
                if next_action not in range(5):raise ValueError('SARSA needs actual next behavior action')
                target+=gamma*scores[next_action]
            else:target+=gamma*(rows(next_state)[int(np.argmax(scores))]@self.target_weights)
        error=float(np.clip(target-phi@self.weights,-5,5));self.weights+=alpha*error*phi/max(1.,float(phi@phi));norm=float(np.linalg.norm(self.weights))
        if norm>100:self.weights*=100/norm
        self.updates+=1
        if self.target_weights is not None and self.updates%100==0:self.target_weights[:]=self.weights;self.target_updates+=1
        if not np.isfinite(self.weights).all():raise FloatingPointError('Nonfinite F3 TD weights')
    def save(self,path,extra):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix!='.npz':raise ValueError('F3 linear checkpoint requires npz')
        np.savez_compressed(path,weights=self.weights);patch_metadata(self,path,extra)

class HistoryNeuralPolicy(NeuralPolicy):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.signature={**self.signature,'feature_version':VERSION,'agent_state_version':HISTORY_VERSION,'agent_state_values':STATE};self.history=PublicHistory()
    def reset(self):self.history=PublicHistory()
    def act(self,observation):
        state=self.history.state(observation);action,_=self.model.predict(state,deterministic=True);action=int(np.asarray(action).item())
        if action not in range(5):raise ValueError('Invalid F3 action')
        self.history.remember(observation,action);return action
    def save(self,path,extra):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix!='.zip':raise ValueError('F3 neural checkpoint requires zip')
        self.model.save(path);patch_metadata(self,path,extra)

def build_policy(env,algorithm,seed,quick=False):
    torch.set_num_threads(1)
    if algorithm in ('q-learning','sarsa'):return HistoryLinearPolicy(algorithm,env.unwrapped.config,seed)
    common=dict(seed=seed,device='cpu',verbose=0,learning_rate=.0003,gamma=.99,policy_kwargs={'features_extractor_class':HistoryExtractor})
    if algorithm=='dqn':model=DQN(HistoryDQNPolicy,env,**common,learning_starts=32 if quick else 500,buffer_size=20000,batch_size=64,train_freq=4,gradient_steps=1,target_update_interval=250,exploration_initial_eps=1.,exploration_final_eps=.08,exploration_fraction=.8)
    elif algorithm=='ppo':model=PPO(HistoryPPOPolicy,env,**common,n_steps=32 if quick else 200,batch_size=32 if quick else 100,n_epochs=4,ent_coef=.015)
    else:raise ValueError('Unknown RL family')
    return HistoryNeuralPolicy(algorithm,model,env.unwrapped.config,seed)

def load_policy(path,config):
    path=Path(path);m=read_json(path.with_suffix('.json'))
    if m.get('signature')!=task_signature(config) or m.get('feature_version')!=VERSION or m.get('feature_count')!=FEATURE_COUNT or m.get('agent_state_values')!=STATE or m.get('agent_state_version')!=HISTORY_VERSION or m.get('architecture_version')!='history-shared-action-scorer-v1' or m.get('actions')!=5 or m.get('observation_values')!=409 or sha(path)!=m.get('checkpoint_sha256'):raise ValueError('F3 feature/history/task/architecture/checksum mismatch')
    algorithm=m['algorithm']
    if algorithm in ('q-learning','sarsa'):
        p=HistoryLinearPolicy(algorithm,config,m['training_seed'])
        with np.load(path,allow_pickle=False) as data:
            if data['weights'].shape!=(ROW,) or not np.isfinite(data['weights']).all():raise ValueError('Invalid F3 weights')
            p.weights[:]=data['weights']
    else:
        torch.set_num_threads(1);model={'dqn':DQN,'ppo':PPO}[algorithm].load(path,device='cpu')
        if model.observation_space.shape!=(STATE,) or model.action_space.n!=5 or model.policy_kwargs.get('features_extractor_class') is not HistoryExtractor or not isinstance(model.policy,HistoryDQNPolicy if algorithm=='dqn' else HistoryPPOPolicy) or any(not torch.isfinite(v).all() for v in model.policy.state_dict().values()):raise ValueError('Invalid F3 neural architecture')
        p=HistoryNeuralPolicy(algorithm,model,config,m['training_seed'])
    p.policy_id=m['policy_id'];p.checkpoint_sha256=m['checkpoint_sha256'];p.reset();return p
