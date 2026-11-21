"""Synthetic-only bounded mechanical fixtures. Not a research trainer or evaluator."""
from contextlib import ExitStack
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from .core import ROOT,read_json,write_json
from .partial_contract import configuration
from .partial_features_scaled import build_policy,load_scaled_policy
from .partial_observation import SPACE,validate
from .partial_training import train_linear
from .partial_pilot_diagnostics import preserve_random_state
from .partial_phase5b_diagnostics import continuity_hash
from .partial_training_instrumentation import observe,network_versions,fingerprint
from .partial_instrumentation_trace import TraceWriter,MechanicalLedger,numeric,checksum


def rng_hash():
    return continuity_hash({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()})


def fixture_observation():
    # Legal fresh open-grid sensor/memory observation; repeated deliberately so
    # replay identities cannot be guessed from observation values.
    from .partial_dynamic_env import PartialDynamicEnv,make_scenario
    obs,_=PartialDynamicEnv([make_scenario(['.'*10]*10,[4,4],[4,5])]).reset(seed=23101)
    return obs


class MechanicalEnvironment(gym.Env):
    """Cyclic tensor/flag test stream, not robot navigation or a research task.

    Actions are recorded but cannot change this controlled stream. Repeated
    observations with continuing/goal/collision/timeout/continuing records test
    library bookkeeping; reward labels are not navigation outcomes.
    """
    def __init__(self,ledger,run):
        self.ledger,self.run=ledger,run;self.config=configuration();self.observation_space=deepcopy(SPACE)
        self.action_space=gym.spaces.Discrete(5);self.action_space.seed(23101)
        self.count=0;self.reset_count=0;self.observation=fixture_observation()
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.reset_count+=1
        return self.observation.copy(),{'case_id':'synthetic-duplicate-observation-stream','outcome':'mechanical'}
    def step(self,action):
        self.ledger.record('transition',self.run)  # Reserve before actual synthetic transition.
        assert self.action_space.contains(action)
        k=self.count%5;self.count+=1
        reward=( -.01,9.99,-5.01,-.01,-.01)[k];terminated=k in (1,2);truncated=k==3
        next_observation=self.observation.copy()
        if terminated or truncated:next_observation[408]=np.float32(.005)
        return next_observation,reward,terminated,truncated,{
            'case_id':'synthetic-duplicate-observation-stream','outcome':'synthetic_mechanical','steps':self.count}


def train_frame():
    frame=inspect.currentframe().f_back
    from stable_baselines3 import DQN,PPO
    while frame is not None:
        if frame.f_code in (DQN.train.__code__,PPO.train.__code__):return frame
        frame=frame.f_back
    raise ValueError('Optimizer not called by actual library training frame')


class EquivalenceWitness:
    """Independent always-on audit witness using profiling, not the logger trace.

    Wraps the original optimizer once in BOTH runs; observes actual frame locals
    and outputs. No extra sampler, forward, RNG or optimizer operation.
    """
    def __init__(self,policy,env,ledger,run):
        self.policy,self.env,self.ledger,self.run=policy,env,ledger,run
        self.rows=[];self.steps=[];self.samples=[];self.forward_counts={};self.train_returns=[];self.stack=ExitStack()
        self.model=getattr(policy,'model',None)
    def __enter__(self):
        from stable_baselines3.common.buffers import ReplayBuffer,RolloutBuffer
        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3 import PPO
        self.codes={type(self.env).step.__code__:'environment.step',type(self.env).reset.__code__:'environment.reset',
            ReplayBuffer._get_samples.__code__:'replay_sample',RolloutBuffer._get_samples.__code__:'rollout_sample',
            ActorCriticPolicy.forward.__code__:'actor_forward',ActorCriticPolicy.predict_values.__code__:'critic_forward'}
        self.codes[PPO.train.__code__]='ppo_train_return'
        if self.model is not None:
            if self.policy.algorithm=='dqn':self.codes[type(self.model.q_net).forward.__code__]='q_forward'
            original=self.model.policy.optimizer.step
            def step(*args,**kwargs):
                frame=train_frame();l=frame.f_locals
                self.ledger.record('optimizer',self.run)
                before=network_versions(self.model)
                gradients=[p.grad.detach().cpu().numpy().copy() for p in self.model.policy.parameters() if p.grad is not None]
                row={'loss':numeric(l['loss']),'network_before':before,'gradient_hash':continuity_hash(gradients),
                    'rng_before':rng_hash(),'sample_indices':list(self.samples[-1]['indices'])}
                result=original(*args,**kwargs)
                row['network_after']=network_versions(self.model);row['optimizer_after']=continuity_hash(self.model.policy.optimizer.state_dict())
                row['rng_after']=rng_hash();self.steps.append(row);return result
            self.stack.enter_context(patch.object(self.model.policy.optimizer,'step',step))
        else:
            original=self.policy.update
            def linear(*args,**kwargs):
                self.ledger.record('optimizer',self.run);before=self.policy.weights.copy();result=original(*args,**kwargs)
                self.steps.append({'weights_before':numeric(before),'weights_after':numeric(self.policy.weights),'rng':rng_hash()})
                return result
            self.stack.enter_context(patch.object(self.policy,'update',linear))
        if sys.getprofile() is not None:raise ValueError('Independent witness needs an unused profile hook')
        sys.setprofile(self.profile);return self
    def __exit__(self,*args):
        sys.setprofile(None);self.stack.close()
    def profile(self,frame,event,arg):
        if event!='return':return
        name=self.codes.get(frame.f_code)
        if name is None:return
        l=frame.f_locals;obj=l.get('self')
        if name=='ppo_train_return' and obj is self.model and 'continue_training' in l:
            self.train_returns.append({'early_stop':not l['continue_training'],'epoch':l['epoch'],
                'last_loss':numeric(l['loss']),'approx_kl':numeric(l['approx_kl_div']),'updates':self.model._n_updates})
            return
        if arg is None:return
        if name.startswith('environment.'):
            if obj is not self.env:return
            if name=='environment.step':
                self.rows.append({'kind':'step','action':int(l['action']),'observation':numeric(arg[0]),'reward':arg[1],
                    'terminated':arg[2],'truncated':arg[3],'info':deepcopy(arg[4])})
            else:self.rows.append({'kind':'reset','observation':numeric(arg[0]),'seed':l.get('seed'),'info':deepcopy(arg[1])})
        elif name in ('replay_sample','rollout_sample'):
            if obj not in (getattr(self.model,'replay_buffer',None),getattr(self.model,'rollout_buffer',None)):return
            self.samples.append({'kind':name,'indices':l['batch_inds'].tolist(),
                'env_indices':l['env_indices'].tolist() if 'env_indices' in l else None,
                'payload':{field:numeric(getattr(arg,field)) for field in arg._fields if getattr(arg,field) is not None}})
        elif name=='actor_forward' and obj is getattr(self.model,'policy',None):
            self.rows.append({'kind':'behavior','probabilities':numeric(l['distribution'].distribution.probs),
                'action':numeric(arg[0]),'value':numeric(arg[1]),'log_prob':numeric(arg[2])})
            self.forward_counts[name]=self.forward_counts.get(name,0)+1
        elif name=='critic_forward' and obj is getattr(self.model,'policy',None):
            self.forward_counts[name]=self.forward_counts.get(name,0)+1
        elif name=='q_forward' and obj in (self.model.q_net,self.model.q_net_target):
            key='target_forward' if obj is self.model.q_net_target else 'online_forward'
            self.forward_counts[key]=self.forward_counts.get(key,0)+1
    def result(self):
        return {'environment_and_behavior':self.rows,'optimizer_steps':self.steps,'sample_order':self.samples,
            'forward_counts':self.forward_counts,'train_returns':self.train_returns,'rng':rng_hash()}


def worker():
    """Only invoked by the sealed synthetic pytest harness, never a study run."""
    if os.environ.get('PHASE5E_MECHANICAL_ONLY')!='1':raise RuntimeError('Synthetic test harness authorization required')
    plan_path=Path(sys.argv[1]);plan=read_json(plan_path)
    if plan['source_kind']!='synthetic_mechanical' or plan['seed']!=23101:raise ValueError('Not the registered synthetic fixture')
    folder=Path(plan['folder']);folder.mkdir(parents=True,exist_ok=False)
    ledger=MechanicalLedger(plan['ledger']);run=plan['run']
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(1)
    env=MechanicalEnvironment(ledger,run)
    c=read_json(plan['model_config'])
    policy=build_policy(env,plan['algorithm'],c,plan['seed'],plan['arm'])
    metadata={'study_id':'phase5e-training-instrumentation-v1','run_id':run,'algorithm':plan['algorithm'],'arm':plan['arm'],
        'seed':plan['seed'],'source_kind':'synthetic_mechanical','source_hash':plan['source_hash'],
        'config_hash':checksum(c),'dependency_hash':plan['dependency_hash'],'split':'synthetic_mechanical'}
    sink=TraceWriter(folder/'trace',metadata) if plan['enabled'] else None
    before_mean,before_std=torch.Tensor.mean,torch.Tensor.std
    before_optimizer=policy.model.policy.optimizer.step if hasattr(policy,'model') else None
    started=perf_counter()
    try:
        with EquivalenceWitness(policy,env,ledger,run) as witness:
            with observe(policy,sink,environment=env,case_id='synthetic-duplicate-observation-stream'):
                if policy.algorithm in ('q-learning','sarsa'):train_linear(policy,env,8,plan['seed'],c)
                elif policy.algorithm=='dqn':
                    # Real production hyperparameters unchanged. Four collection
                    # steps are below warmup; this explicit unit call tests one
                    # real optimizer step on the collected four-record fixture.
                    policy.model.learn(total_timesteps=4)
                    policy.model.train(gradient_steps=1,batch_size=c['batch_size'])
                    for _ in range(c['target_update_interval']-policy.model._n_calls):policy.model._on_step()
                else:
                    policy.model.learn(total_timesteps=c['n_steps'])
                    # Explicit zero-environment mechanical early-stop injection,
                    # AFTER the normal configured collection/update. No config or
                    # historical model changes; reuses its existing actual rollout.
                    policy.model.target_kl=1e-12
                    policy.model.train()
            result=witness.result()
        if sink:sink.close()
        state={'weights':policy.state_fingerprint(),'rng':rng_hash(),'environment_rng':env.np_random.bit_generator.state,
            'action_rng':env.action_space.np_random.bit_generator.state,'transitions':env.count,'resets':env.reset_count}
        if hasattr(policy,'model'):
            m=policy.model
            state['optimizer']=continuity_hash(m.policy.optimizer.state_dict())
            state['buffer']=continuity_hash(m.replay_buffer if policy.algorithm=='dqn' else m.rollout_buffer)
            state['updates']=int(m._n_updates);state['timesteps']=int(m.num_timesteps)
            state['network']=network_versions(m)
        else:state['weights_tensor']=numeric(policy.weights)
        result['final_state']=state
        write_json(folder/'witness.json',result)
        path=folder/('policy.npz' if policy.algorithm in ('q-learning','sarsa') else 'policy.zip')
        policy.save(path,{'study_id':'phase5e-training-instrumentation-v1','classification':'synthetic_mechanical'})
        with preserve_random_state():
            loaded=load_scaled_policy(path,configuration(),plan['arm'])
            if loaded.state_fingerprint()!=policy.state_fingerprint():raise ValueError('Synthetic checkpoint loading changed weights/signature')
        write_json(folder/'receipt.json',{'classification':'synthetic_mechanical','transitions':env.count,
            'optimizer_steps':len(result['optimizer_steps']),'wall_seconds':perf_counter()-started,
            'witness_sha256':checksum(result),'checkpoint_loading':'passed','enabled':plan['enabled'],
            'scientific_actions':0,'research_assessment_actions':0,
            'hooks_restored':sys.gettrace() is None and sys.getprofile() is None and torch.Tensor.mean is before_mean and torch.Tensor.std is before_std
                and (before_optimizer is None or policy.model.policy.optimizer.step==before_optimizer)})
    finally:
        if sink:sink.close()


def paired(algorithm,arm,folder,ledger,source_hash,dependency_hash,model_config):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    for enabled in (False,True):
        run=algorithm+'-'+arm+('-on' if enabled else '-off')
        plan={'source_kind':'synthetic_mechanical','algorithm':algorithm,'arm':arm,'seed':23101,
            'folder':str(folder/('on' if enabled else 'off')),'enabled':enabled,'ledger':str(ledger),
            'source_hash':source_hash,'dependency_hash':dependency_hash,'run':run,'model_config':str(model_config)}
        path=folder/(('on' if enabled else 'off')+'-plan.json');write_json(path,plan)
        environment=dict(os.environ,PHASE5E_MECHANICAL_ONLY='1',PYTHONPATH=str(ROOT/'src'))
        process=subprocess.run([sys.executable,'-c','from robot_navigation.partial_phase5e_mechanics import worker; worker()',str(path)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        (folder/('on.log' if enabled else 'off.log')).write_text(process.stdout+process.stderr,encoding='utf-8')
        if process.returncode:raise RuntimeError('Synthetic worker failed; partial evidence retained: '+process.stderr[-3000:])
    off=read_json(folder/'off/witness.json');on=read_json(folder/'on/witness.json')
    if off!=on:
        for key in off:
            if off[key]!=on[key]:raise ValueError('On/off bitwise mismatch in '+key+' for '+algorithm+'/'+arm)
    receipt={'algorithm':algorithm,'arm':arm,'bitwise_equal':True,'tolerance':0,'independent_witness_sha256':checksum(off),
        'transitions_per_run':off['final_state']['transitions'],'optimizer_steps_per_run':len(off['optimizer_steps']),
        'compared':['actions/probabilities/reset sequence','global/env/action RNG','sample order and duplicates','buffer contents',
            'losses/clipped gradients','optimizer and online/target parameters','forward-call counts','GAE/returns via actual rollout samples']}
    write_json(folder/'equivalence.json',receipt);return receipt
