"""Disabled-by-default passive observers of the pinned SB3/linear execution.

No training entry point. No extra sampling, forward prediction or optimizer call.
Supported: exact pinned CPU SB3 2.9.2, 409 floats, Discrete(5), one environment,
one-step replay, no VecNormalize/SDE/memory-optimized replay. Reject otherwise.
"""
from contextlib import ExitStack,nullcontext
import hashlib
import inspect
from importlib.metadata import version
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import DQN,PPO
from stable_baselines3.common.buffers import ReplayBuffer,RolloutBuffer,BaseBuffer
from stable_baselines3.common.policies import ActorCriticPolicy

from .core import ROOT,read_json
from .partial_agents import LinearPolicy
from .partial_features_scaled import ScaledLinearPolicy
from .partial_phase5b_diagnostics import continuity_hash
from .partial_instrumentation_trace import numeric,ReplayIdentity


def methods():
    return {'DQN.train':DQN.train,'DQN._on_step':DQN._on_step,'ReplayBuffer.add':ReplayBuffer.add,
        'ReplayBuffer.sample':ReplayBuffer.sample,'ReplayBuffer._get_samples':ReplayBuffer._get_samples,
        'RolloutBuffer.add':RolloutBuffer.add,'RolloutBuffer.get':RolloutBuffer.get,
        'RolloutBuffer._get_samples':RolloutBuffer._get_samples,'RolloutBuffer.reset':RolloutBuffer.reset,
        'RolloutBuffer.compute_returns_and_advantage':RolloutBuffer.compute_returns_and_advantage,
        'PPO.train':PPO.train,'PPO.collect_rollouts':PPO.collect_rollouts,
        'ActorCriticPolicy.forward':ActorCriticPolicy.forward,'BaseBuffer.sample':BaseBuffer.sample,
        'LinearPolicy.update':LinearPolicy.update,'ScaledLinearPolicy.update':ScaledLinearPolicy.update}


def library_contract():
    return {'versions':{n:version(n) for n in ('stable-baselines3','torch','numpy','gymnasium')},
        'methods':{n:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for n,f in methods().items()}}


def check_compatibility(contract=None):
    expected=contract if contract is not None else read_json(ROOT/'configs/partial_phase5e_library_v1.json')
    if library_contract()!=expected:raise RuntimeError('UNSUPPORTED_INSTRUMENTATION_PATH: installed library/source mismatch')


def fingerprint(model):
    return continuity_hash(model.policy.state_dict())


def network_versions(model):
    if isinstance(model,DQN):return {'online':continuity_hash(model.q_net.state_dict()),'target':continuity_hash(model.q_net_target.state_dict())}
    return {'policy_and_value':fingerprint(model),'target':'not_applicable'}


def anchor(method,text):
    lines,start=inspect.getsourcelines(method);found=[start+i for i,line in enumerate(lines) if text in line]
    if len(found)!=1:raise RuntimeError('UNSUPPORTED_INSTRUMENTATION_PATH: ambiguous hook '+text)
    return found[0]


def observe(policy,sink=None,*,environment=None,case_id=None):
    """A missing sink is a true no-op: no hooks, file I/O or library calls."""
    return nullcontext() if sink is None else ActualObserver(policy,sink,environment=environment,case_id=case_id)


class ActualObserver:
    def __init__(self,policy,sink,environment=None,case_id=None):
        self.policy=policy;self.model=getattr(policy,'model',None);self.sink=sink
        self.environment=environment;self.case_id=case_id
        self.interactions=0;self.episode=-1;self.optimizer_steps=0;self.sync_id=0
        self.rollout_id=0;self.epoch=-1;self.minibatch=0;self.batch=None;self.sample=None
        self.training_call_id=0
        self.last_raw=None;self.last_obs=None;self.previous_weights=None;self.stats={}
        self.rollout_rows=[];self._stack=None
    def emit(self,event,**fields):self.sink.emit(event,**fields)
    def __enter__(self):
        check_compatibility()
        if sys.gettrace() is not None:raise RuntimeError('UNSUPPORTED_INSTRUMENTATION_PATH: active debugger/tracer')
        self._stack=ExitStack()
        try:
            if self.model is not None:
                m=self.model
                if (m.device.type!='cpu' or m.observation_space.shape!=(409,) or m.action_space.n!=5 or m.n_envs!=1
                    or m._vec_normalize_env is not None or getattr(m,'use_sde',False)):
                    raise RuntimeError('UNSUPPORTED_INSTRUMENTATION_PATH: observation/device/environment mode')
                if isinstance(m,DQN):
                    b=m.replay_buffer
                    if type(b) is not ReplayBuffer or b.optimize_memory_usage or b.pos or b.full:
                        raise RuntimeError('UNSUPPORTED_INSTRUMENTATION_PATH: replay must start empty, standard one-step mode')
                    self.identities=ReplayIdentity(b.buffer_size,b.n_envs)
                elif not isinstance(m,PPO):raise RuntimeError('UNSUPPORTED_INSTRUMENTATION_PATH: algorithm')
                if self.environment is None:self.environment=m.env.envs[0].unwrapped
                self._original_step=m.policy.optimizer.step
                self._stack.enter_context(patch.object(m.policy.optimizer,'step',self._optimizer_step))
            else:
                if not isinstance(self.policy,LinearPolicy):raise RuntimeError('UNSUPPORTED_INSTRUMENTATION_PATH: linear class')
                if self.environment is None:raise ValueError('A linear observer needs the caller-owned environment')
            self.codes={f.__code__:n for n,f in methods().items()}
            self.codes[type(self.environment).step.__code__]='environment.step'
            self.codes[type(self.environment).reset.__code__]='environment.reset'
            self.dqn_loss=anchor(DQN.train,'self.policy.optimizer.zero_grad()')
            self.ppo_loss=anchor(PPO.train,'if self.target_kl is not None and approx_kl_div')
            self.permutation_line=anchor(RolloutBuffer.get,'if not self.generator_ready:')
            self.norm_line=anchor(PPO.train,'advantages = (advantages - advantages.mean())')
            self.linear_lines={f.__code__:anchor(f,'self.weights[action] +=') for f in (LinearPolicy.update,ScaledLinearPolicy.update)}
            # Observe the actual normalization reductions once, never redo them.
            if isinstance(self.model,PPO):
                for name in ('mean','std'):
                    original=getattr(torch.Tensor,name)
                    def reduction(tensor,*args,_name=name,_original=original,**kwargs):
                        value=_original(tensor,*args,**kwargs);caller=inspect.currentframe().f_back
                        if caller.f_code is PPO.train.__code__ and caller.f_lineno==self.norm_line:
                            self.stats[_name]=numeric(value)
                        return value
                    self._stack.enter_context(patch.object(torch.Tensor,name,reduction))
            self.emit('observer_opened',algorithm=self.policy.algorithm,feature_signature=self.policy.signature,
                unsupported_absence={'hidden_map':'not_applicable','future_events':'not_applicable','sealed_final':'not_applicable'},
                scope='CPU, one environment, actual computations only; no extra predictions/samples')
            sys.settrace(self._trace)
            return self
        except BaseException:
            self._stack.close();raise
    def __exit__(self,kind,error,tb):
        sys.settrace(None)
        self._stack.close()
        if kind is None:self.emit('observer_closed',interactions=self.interactions,optimizer_steps=self.optimizer_steps,rollouts=self.rollout_id)
        else:
            try:self.emit('observer_failed',error=repr(error),interactions=self.interactions,optimizer_steps=self.optimizer_steps)
            except BaseException:pass
        return False
    def _optimizer_step(self,*args,**kwargs):
        m=self.model;before=network_versions(m);old=[p.detach().cpu().numpy().copy() for p in m.policy.parameters()]
        grads=[p.grad.detach().cpu().numpy().copy() for p in m.policy.parameters() if p.grad is not None]
        if not grads or not all(np.isfinite(g).all() for g in grads):raise ValueError('Nonfinite or absent actual gradient')
        pre_optimizer=continuity_hash(m.policy.optimizer.state_dict())
        result=self._original_step(*args,**kwargs)  # Exactly one original optimizer call.
        if not all(bool(torch.isfinite(p).all()) for p in m.policy.parameters()):
            raise ValueError('Nonfinite parameters after actual optimizer step')
        self.optimizer_steps+=1
        after=network_versions(m)
        delta=float(np.sqrt(sum(np.sum((p.detach().cpu().numpy().astype(float)-a)**2) for p,a in zip(m.policy.parameters(),old))))
        self.emit('optimizer',update_id=self.optimizer_steps,training_actions=int(m.num_timesteps),
            training_call_id=self.training_call_id,
            rollout_id=self.rollout_id if isinstance(m,PPO) else 'not_applicable',epoch=self.epoch,minibatch_id=self.minibatch,
            gradient_tensors=[numeric(g) for g in grads],clipped_gradient_norm=float(np.sqrt(sum(np.sum(g.astype(float)**2) for g in grads))),
            parameter_update_norm=delta,network_before=before,network_after=after,optimizer_before=pre_optimizer,
            optimizer_after=continuity_hash(m.policy.optimizer.state_dict()),learning_rates=[p['lr'] for p in m.policy.optimizer.param_groups])
        return result
    def _trace(self,frame,event,arg):
        name=self.codes.get(frame.f_code)
        if name is None:return None
        local=frame.f_locals;obj=local.get('self')
        if name.startswith('environment.') and obj is not self.environment:return None
        if name.startswith(('DQN.','PPO.')) and obj is not self.model:return None
        if name.startswith('ActorCriticPolicy.') and obj is not getattr(self.model,'policy',None):return None
        if name.startswith(('ReplayBuffer.','RolloutBuffer.')) and obj not in (getattr(self.model,'replay_buffer',None),getattr(self.model,'rollout_buffer',None)):
            return None
        if event=='call':
            if name=='environment.step':self._step_observation=self.last_obs.copy() if self.last_obs is not None else None
            elif name=='PPO.collect_rollouts':self.rollout_id+=1;self.rollout_rows=[];self.epoch=-1
            elif name in ('PPO.train','DQN.train'):self.training_call_id+=1;self.epoch=-1
            elif name in ('LinearPolicy.update','ScaledLinearPolicy.update'):
                self.previous_weights=self.policy.weights.copy()
            elif name=='ReplayBuffer.add':
                if self.last_raw is None:raise ValueError('Replay add missing actual transition identity')
                self.identities.add(obj.pos,0,self.last_raw['transition_id'],episode=self.last_raw['episode'],
                    terminated=self.last_raw['terminated'],truncated=self.last_raw['truncated'])
            elif name=='RolloutBuffer.add':
                if self.last_raw is None:raise ValueError('Rollout add missing raw environment record')
                row={'rollout_id':self.rollout_id,'row':int(obj.pos),'transition_id':self.last_raw['transition_id'],'env_index':0}
                self.rollout_rows.append(row)
                self.emit('rollout_add',**row,observation=numeric(local['obs']),action=numeric(local['action']),
                    raw_reward=self.last_raw['reward'],adjusted_reward=numeric(local['reward']),
                    episode_start=numeric(local['episode_start']),values=numeric(local['value']),log_probs=numeric(local['log_prob']),
                    terminated=self.last_raw['terminated'],truncated=self.last_raw['truncated'])
            elif name=='RolloutBuffer.compute_returns_and_advantage':
                self.emit('gae_input',rollout_id=self.rollout_id,rows=list(self.rollout_rows),
                    last_values=numeric(local['last_values']),dones=numeric(local['dones']),gamma=obj.gamma,gae_lambda=obj.gae_lambda,
                    values=numeric(obj.values),rewards=numeric(obj.rewards),episode_starts=numeric(obj.episode_starts))
            elif name=='DQN._on_step':self._sync_before=network_versions(self.model)
        elif event=='line':
            if name=='DQN.train' and frame.f_lineno==self.dqn_loss:
                if self.sample is None:raise ValueError('Actual replay identities missing')
                r=local['replay_data']
                self.emit('dqn_loss',sample=self.sample,update_id=self.optimizer_steps+1,network=network_versions(self.model),
                    observation=numeric(r.observations),next_observation=numeric(r.next_observations),actions=numeric(r.actions),
                    reward=numeric(r.rewards),consumed_dones=numeric(r.dones),discount=numeric(local['discounts']),
                    current_chosen_Q=numeric(local['current_q_values']),next_target_max_Q=numeric(local['next_q_values']),
                    actual_target=numeric(local['target_q_values']),loss=numeric(local['loss']),loss_reduction='mean smooth_l1',
                    epsilon=float(self.model.exploration_rate),training_actions=int(self.model.num_timesteps))
            elif name=='RolloutBuffer.get' and frame.f_lineno==self.permutation_line:
                self.epoch+=1;self.minibatch=0
                self.emit('ppo_permutation',rollout_id=self.rollout_id,training_call_id=self.training_call_id,epoch=self.epoch,indices=numeric(local['indices']))
            elif name=='PPO.train' and frame.f_lineno==self.ppo_loss:
                r=local['rollout_data'];early=bool(self.model.target_kl is not None and local['approx_kl_div']>1.5*self.model.target_kl)
                self.emit('ppo_loss',rollout_id=self.rollout_id,training_call_id=self.training_call_id,epoch=int(local['epoch']),minibatch_id=self.minibatch,
                    sample=self.batch,network=network_versions(self.model),optimizer_step=None if early else self.optimizer_steps+1,
                    early_stop=early,raw_advantages=numeric(r.advantages),normalized_advantages=numeric(local['advantages']),
                    normalization_statistics=dict(self.stats) if self.model.normalize_advantage and len(r.advantages)>1 else {'status':'not_applicable'},
                    old_log_probs=numeric(r.old_log_prob),new_log_probs=numeric(local['log_prob']),ratio=numeric(local['ratio']),
                    policy_loss_1=numeric(local['policy_loss_1']),policy_loss_2=numeric(local['policy_loss_2']),
                    values=numeric(local['values']),values_pred=numeric(local['values_pred']),returns=numeric(r.returns),
                    entropy=None if local['entropy'] is None else numeric(local['entropy']),policy_loss=numeric(local['policy_loss']),
                    value_loss=numeric(local['value_loss']),entropy_loss=numeric(local['entropy_loss']),total_loss=numeric(local['loss']),
                    approx_kl=numeric(local['approx_kl_div']),clip_fraction=float(local['clip_fraction']),clip_range=float(local['clip_range']),
                    vf_coef=self.model.vf_coef,ent_coef=self.model.ent_coef,Q_values='not_applicable')
            elif frame.f_code in self.linear_lines and frame.f_lineno==self.linear_lines[frame.f_code]:
                phi=local.get('phi',local.get('features'))
                self._linear_pending={'observation':numeric(local['observation']),'next_observation':numeric(local['next_observation']),
                    'action':int(local['action']),'reward':float(local['reward']),'terminated':bool(local['terminated']),
                    'sampled_next_action':local['next_action'],'alpha':local['alpha'],'gamma':local['gamma'],
                    'features':numeric(phi),'target':float(local['target']),'td_error':float(local['error']),
                    'weights_before':numeric(self.previous_weights),'preupdate_Q_from_live_weights':float(self.previous_weights[local['action']]@phi),
                    'next_values':None if 'values' not in local else numeric(local['values']),
                    'neural_fields':'not_applicable'}
        elif event=='return':
            if name=='environment.reset' and arg is not None:
                self.episode+=1;self.last_obs=np.asarray(arg[0]).copy()
                case=arg[1].get('case_id',self.case_id if self.case_id is not None else 'not_observed: caller did not provide case identifier')
                self.emit('reset',episode=self.episode,case_id=case,seed=local.get('seed'),observation=numeric(arg[0]))
            elif name=='environment.step' and arg is not None:
                obs,reward,terminated,truncated,info=arg;self.interactions+=1
                self.last_raw={'transition_id':self.interactions,'episode':self.episode,'action':int(local['action']),
                    'reward':float(reward),'terminated':bool(terminated),'truncated':bool(truncated)}
                self.last_obs=np.asarray(obs).copy()
                self.emit('environment_transition',**self.last_raw,env_index=0,observation=numeric(self._step_observation),
                    next_observation=numeric(obs),case_id=info.get('case_id',self.case_id or 'not_observed'))
            elif name=='ReplayBuffer._get_samples' and arg is not None:
                self.sample=self.identities.sampled(local['batch_inds'],local['env_indices'])
                self.emit('replay_sample',sample=self.sample,indices=numeric(local['batch_inds']),env_indices=numeric(local['env_indices']),
                    buffer_pos=int(obj.pos),buffer_full=bool(obj.full),mode='standard one-step replacement sampling',
                    raw_dones=numeric(obj.dones[local['batch_inds'],local['env_indices']]),
                    raw_timeouts=numeric(obj.timeouts[local['batch_inds'],local['env_indices']]))
            elif name=='RolloutBuffer._get_samples' and arg is not None:
                self.minibatch+=1
                self.batch=[dict(self.rollout_rows[int(i)]) for i in local['batch_inds']]
                self.emit('rollout_sample',rollout_id=self.rollout_id,training_call_id=self.training_call_id,epoch=self.epoch,minibatch_id=self.minibatch,
                    indices=numeric(local['batch_inds']),sample=self.batch)
            elif name=='RolloutBuffer.compute_returns_and_advantage':
                self.emit('gae_result',rollout_id=self.rollout_id,advantages=numeric(obj.advantages),returns=numeric(obj.returns))
            elif name=='ActorCriticPolicy.forward' and arg is not None:
                self.emit('ppo_behavior',rollout_id=self.rollout_id,transition_id=self.interactions+1,
                    observation=numeric(local['obs']),probabilities=numeric(local['distribution'].distribution.probs),
                    action=numeric(arg[0]),values=numeric(arg[1]),log_probs=numeric(arg[2]),Q_values='not_applicable')
            elif name=='PPO.collect_rollouts' and 'terminal_value' in local:
                # Per-timeout values are captured at the actual augmentation line below.
                pass
            elif name=='DQN._on_step':
                if self.model._n_calls%max(self.model.target_update_interval//self.model.n_envs,1)==0:
                    self.sync_id+=1;self.emit('target_sync',sync_id=self.sync_id,call=int(self.model._n_calls),
                        before=self._sync_before,after=network_versions(self.model),tau=self.model.tau,
                        epsilon=float(self.model.exploration_rate),training_actions=int(self.model.num_timesteps))
            elif name in ('LinearPolicy.update','ScaledLinearPolicy.update') and hasattr(self,'_linear_pending'):
                self.optimizer_steps+=1
                self.emit('linear_update',**self._linear_pending,update_id=self.optimizer_steps,
                    transition_id=self.interactions,weights_after=numeric(self.policy.weights),
                    update_norm=float(np.linalg.norm(self.policy.weights-self.previous_weights)))
                del self._linear_pending
        # Capture the already-computed timeout value immediately before reward augmentation.
        if event=='line' and name=='PPO.collect_rollouts' and 'terminal_value' in local:
            source=inspect.getframeinfo(frame).code_context
            if source and 'rewards[idx] += self.gamma * terminal_value' in source[0]:
                self.emit('ppo_timeout_bootstrap',rollout_id=self.rollout_id,transition_id=self.interactions,
                    terminal_observation=numeric(local['infos'][local['idx']]['terminal_observation']),
                    raw_reward=float(local['rewards'][local['idx']]),terminal_value=numeric(local['terminal_value']),gamma=self.model.gamma)
        return self._trace
