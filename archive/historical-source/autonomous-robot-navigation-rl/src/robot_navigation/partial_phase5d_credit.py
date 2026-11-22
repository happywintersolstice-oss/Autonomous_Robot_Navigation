"""Forward-only scores and side-effect-free credit arithmetic, with provenance."""
from contextlib import contextmanager
import json
from unittest.mock import patch

import numpy as np
import torch

from .partial_features_scaled import encode_scaled
from .partial_phase5b_diagnostics import GROUPS,continuity_hash
from .partial_phase5c_coverage import score_record
from .partial_phase5d_states import state_hash


def forbidden(*args,**kwargs):
    raise RuntimeError('Phase 5D forbids training, policy actions, optimizer steps, environment steps and network access')


@contextmanager
def scientific_guard():
    from contextlib import ExitStack
    from stable_baselines3 import DQN,PPO
    from .partial_agents import LinearPolicy
    from .partial_features_scaled import ScaledLinearPolicy
    from .partial_dynamic_env import PartialDynamicEnv
    with ExitStack() as stack:
        for cls,name in ((DQN,'learn'),(DQN,'train'),(PPO,'learn'),(PPO,'train'),
                         (LinearPolicy,'update'),(ScaledLinearPolicy,'update'),(PartialDynamicEnv,'step')):
            stack.enter_context(patch.object(cls,name,forbidden))
        stack.enter_context(patch('socket.create_connection',forbidden))
        yield


@contextmanager
def protect_inference(policy):
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.object(policy,'act',forbidden))
        if hasattr(policy,'model'):
            stack.enter_context(patch.object(policy.model.policy.optimizer,'step',forbidden))
            buffer=getattr(policy.model,'replay_buffer',None)
            if buffer is not None: stack.enter_context(patch.object(buffer,'sample',forbidden))
        before=frozen_fingerprint(policy)
        yield
        if frozen_fingerprint(policy)!=before:
            raise RuntimeError('Inference mutated weights, optimizer or buffer')


def frozen_fingerprint(policy):
    state={'weights':policy.state_fingerprint()}
    if hasattr(policy,'model'):
        state['optimizer']=policy.model.policy.optimizer.state_dict()
        state['replay']=getattr(policy.model,'replay_buffer',None)
        state['rollout']=getattr(policy.model,'rollout_buffer',None)
    return continuity_hash(state)


class ForwardBudget:
    def __init__(self,path=None,ceiling=5000):
        self.ceiling,self.calls=ceiling,0
        self.file=path.open('x',encoding='utf-8',buffering=1) if path else None
    def claim(self,role,identity,observation_hash):
        if self.calls>=self.ceiling: raise RuntimeError('Forward-only call ceiling reached before prediction')
        self.calls+=1
        if self.file:
            self.file.write(json.dumps({'call':self.calls,'role':role,'model':identity,'observation_sha256':observation_hash})+'\n')
    def close(self):
        if self.file: self.file.close()


class FrozenScores:
    def __init__(self,policy,feature_version,budget,identity):
        self.policy,self.feature_version,self.budget,self.identity=policy,feature_version,budget,identity
        self.cache,self.reused= {},0
    def reuse(self,observation,scores):
        a=np.asarray(scores,dtype=np.float64)
        if a.shape!=(5,) or not np.isfinite(a).all(): raise ValueError('Invalid historical score reuse')
        self.cache['online',state_hash(observation)]=a
    def get(self,observation,role='online'):
        o=np.asarray(observation,dtype=np.float32);h=state_hash(o);key=role,h
        if key in self.cache:
            self.reused+=1;return self.cache[key]
        self.budget.claim(role,self.identity,h)
        p=self.policy
        if role=='components':
            if p.algorithm not in ('q-learning','sarsa'): raise ValueError('Neural scores do not have linear feature contributions')
            phi=encode_scaled(o,self.feature_version)
            value={name:(p.weights[:,lo:hi]@phi[lo:hi]).tolist() for name,(lo,hi) in GROUPS.items()}
        elif p.algorithm in ('q-learning','sarsa'):
            if role!='online': raise ValueError('No target network in a linear agent')
            value=p.scores(o)
        else:
            tensor=torch.as_tensor(o,dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                if p.algorithm=='dqn':
                    if role not in ('online','target'): raise ValueError('DQN has Q vectors, not PPO critic values')
                    net=p.model.q_net if role=='online' else p.model.q_net_target
                    value=net(tensor)[0].cpu().numpy().astype(float)
                elif role=='online':
                    value=p.model.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy().astype(float)
                elif role=='value': value=float(p.model.policy.predict_values(tensor).item())
                else: raise ValueError('PPO has no Q(s,a) or target Q network')
        if isinstance(value,dict): finite=all(np.isfinite(v).all() for v in value.values())
        else: finite=bool(np.isfinite(value).all())
        if not finite: raise FloatingPointError('Nonfinite snapshot calculation')
        self.cache[key]=value
        return value


def attribution(scores,components):
    reconstructed=np.sum(np.array(list(components.values())),axis=0)
    if not np.allclose(reconstructed,scores,atol=1e-9,rtol=1e-9):
        raise ValueError('Linear contribution sum does not reproduce Q')
    competitor=max((a for a in range(5) if a!=3),key=lambda a:scores[a])
    return {'components':components,'sum_error':float(np.max(np.abs(reconstructed-scores))),
            'Right_minus_best_other_components':{k:float(v[3]-v[competitor]) for k,v in components.items()},
            'competitor_action':competitor,'provenance':'snapshot exact linear algebra, not a causal training attribution'}


def snapshot_td(reward,terminated,truncated,next_target_values,current_chosen,checkpoint,transition_step):
    if not terminated and next_target_values is None: raise ValueError('Continuing transition needs target-network values')
    target=float(reward) if terminated else float(reward+.99*max(next_target_values))
    return {'target':target,'residual':target-float(current_chosen),'provenance':'snapshot',
        'checkpoint':checkpoint,'transition_step':transition_step,'terminated':bool(terminated),
        'truncated':bool(truncated),'historical_optimizer_target_verified':False,
        'scope':'Frozen target calculation, not the target used by any historical replay optimizer batch'}


def require_historical_target(value):
    if value.get('provenance') not in ('historical','reconstructable-contemporaneous'):
        raise ValueError('Snapshot target cannot be labelled a historical optimizer target')


def ppo_statistics(probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.shape!=(5,) or np.any(p<0) or not np.isfinite(p).all() or not np.isclose(p.sum(),1.,atol=1e-6):
        raise ValueError('Invalid PPO probability vector')
    nz=p>0;entropy=float(-np.sum(p[nz]*np.log(p[nz])))
    competitor=max(p[a] for a in range(5) if a!=3)
    return {'entropy':entropy,'Right_log_odds_vs_best_other':float(np.log(p[3]/competitor)) if p[3]>0 and competitor>0 else None,
            'score_kind':'probability','Q_values':None}


def audit_linear_logs(steps,logs,feature_version,algorithm):
    if len(steps)!=6000 or len(logs)!=6000: raise ValueError('Incomplete actual linear log')
    rows=[]
    for i,(s,log) in enumerate(zip(steps,logs)):
        if log['update']!=i+1: raise ValueError('Linear update order changed')
        obs=np.asarray(s['observation'],dtype=np.float32);phi=encode_scaled(obs,feature_version)
        scores=np.asarray(s['behavior_context']['decision']['action_values'])
        target=float(log['target']);error=target-scores[s['action']]
        expected_norm=.1*abs(error)/np.linalg.norm(phi)
        if (not log['finite'] or bool(log['terminated'])!=bool(s['terminated'])
                or not np.isclose(log['feature_norm'],np.linalg.norm(phi),atol=1e-9,rtol=1e-8)):
            raise ValueError('Recorded linear feature norm/terminal/finiteness mismatch')
        if not np.isclose(error,log['td_error'],atol=1e-8,rtol=1e-8) or not np.isclose(expected_norm,log['update_norm'],atol=1e-9,rtol=1e-7):
            raise ValueError('Actual logged linear error/update mechanics disagree')
        expected=None;status='historical target recorded; independent bootstrap value not observed'
        if s['terminated']:
            expected=s['reward'];status='reconstructable-contemporaneous'
        elif not s['truncated'] and i+1<len(steps) and steps[i+1]['episode']==s['episode']:
            following=steps[i+1]
            if following['observation']!=s['next_observation']: raise ValueError('Non-contiguous linear observations')
            nxt=np.asarray(s['next_observation'],dtype=np.float32);phi_next=encode_scaled(nxt,feature_version)
            before=np.asarray(following['behavior_context']['decision']['action_values'],dtype=float).copy()
            before[s['action']]-=.1*log['td_error']*float(phi@phi_next)/float(phi@phi)
            if algorithm=='sarsa':
                action=log['next_behavior_action']
                if type(action) is not int or action not in range(5): raise ValueError('Missing sampled SARSA next action')
                bootstrap=before[action]
            else: bootstrap=before.max()
            expected=s['reward']+.99*bootstrap;status='reconstructable-contemporaneous'
        if expected is not None and not np.isclose(expected,target,atol=1e-7,rtol=1e-8):
            raise ValueError('Backed-out actual Bellman target disagrees')
        rows.append({'step':i+1,'recorded_target':target,'recorded_error':log['td_error'],
            'computed_target':float(expected) if expected is not None else None,
            'sampled_next_action':log['next_behavior_action'],'provenance':status,
            'actual_goal_reward':s['reward'] if s['reward']>0 else None,
            'target_error':abs(float(expected)-target) if expected is not None else None})
    return rows


def audit_ppo_rollout(state,steps,gamma=.99,gae_lambda=.95):
    buffer=state['replay'];n=int(buffer.buffer_size if buffer.full else buffer.pos)
    if n==0: return {'saved_rollout_transitions':0,'full_history_GAE':'not recorded','boundary':'not-observed'}
    tail=steps[-n:];values=np.asarray(buffer.values[:n]).reshape(-1)
    probs=np.array([s['behavior_context']['decision']['action_probabilities'] for s in tail])
    recorded_v=np.array([s['behavior_context']['decision']['state_value'] for s in tail])
    actions=np.array([s['action'] for s in tail]);logs=np.log(probs[np.arange(n),actions])
    if not np.allclose(values,recorded_v,atol=1e-5,rtol=1e-5): raise ValueError('PPO collection-time values disagree')
    if not np.allclose(np.asarray(buffer.log_probs[:n]).reshape(-1),logs,atol=1e-5,rtol=1e-5): raise ValueError('PPO sampled log-probs disagree')
    advantages=np.asarray(buffer.advantages[:n]).reshape(-1);returns=np.asarray(buffer.returns[:n]).reshape(-1)
    rewards=np.asarray(buffer.rewards[:n]).reshape(-1);starts=np.asarray(buffer.episode_starts[:n]).reshape(-1)
    if not np.allclose(returns,advantages+values,atol=1e-5): raise ValueError('PPO return identity failed')
    errors=[]
    for i in range(n-1):
        mask=1-starts[i+1]
        expected=rewards[i]+gamma*mask*values[i+1]-values[i]+gamma*gae_lambda*mask*advantages[i+1]
        errors.append(abs(float(expected-advantages[i])))
    if errors and max(errors)>1e-4: raise ValueError('PPO interior GAE recurrence failed')
    terminal_boundary=bool(np.asarray(state['last_episode_starts']).item())
    if terminal_boundary and not np.isclose(advantages[-1],rewards[-1]-values[-1],atol=1e-4):
        raise ValueError('PPO terminal rollout boundary failed')
    timeout_indices=[i for i,s in enumerate(tail) if s.get('truncated',False) and not s.get('terminated',False)]
    unaugmented=[i for i in range(n) if i not in timeout_indices]
    if unaugmented and not np.allclose(rewards[unaugmented],[tail[i]['reward'] for i in unaugmented],atol=1e-5):
        raise ValueError('Unexpected PPO rollout reward transformation')
    return {'saved_rollout_transitions':n,'values_match_contemporaneous_log':True,'sampled_log_probs_match':True,
        'return_identity_verified':True,'interior_GAE_max_error':max(errors,default=0.),
        'boundary':'verified zero continuation after saved episode end; rollout reward may include timeout augmentation' if terminal_boundary else 'pre-update last bootstrap value not recorded',
        'timeout_augmented_records':len(timeout_indices),
        'timeout_reward_adjustments':[{'rollout_index':i,'raw_reward':tail[i]['reward'],'saved_rollout_reward':float(rewards[i]),
            'bootstrap_provenance':'contemporaneous bootstrap tensor not separately recorded; adjustment only'} for i in timeout_indices],
        'saved_reward_GAE_recurrence_verified':True,
        'full_saved_rollout_independent_GAE':terminal_boundary and not timeout_indices,
        'full_history_GAE':'not recorded; earlier rollout boundaries and minibatch credit path unavailable',
        'provenance':'historical last-rollout arrays, not a post-update critic reconstruction'}
