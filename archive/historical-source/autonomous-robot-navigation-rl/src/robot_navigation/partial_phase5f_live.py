"""The single registered real Fixture A pilot. No evaluation or retry path."""
from contextlib import ExitStack
import gc
import inspect
import random
import sys
from time import perf_counter
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import BaseBuffer,ReplayBuffer,RolloutBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy

from .core import write_json
from .partial_contract import configuration
from .partial_dynamic_env import PartialDynamicEnv
from .partial_features_scaled import build_policy
from .partial_training_instrumentation import observe,network_versions
from .partial_instrumentation_trace import TraceWriter,numeric,checksum,quota_bytes
from .partial_phase5b_diagnostics import continuity_hash
from .partial_phase5f_io import STUDY,ALGORITHMS,Journal,Budget,observed_hash


def rng_signature(env):
    return {'python':continuity_hash(random.getstate()),'numpy':continuity_hash(np.random.get_state()),
        'torch':continuity_hash(torch.get_rng_state()),'case_rng':continuity_hash(env.np_random.bit_generator.state),
        'action_rng':continuity_hash(env.action_space.np_random.bit_generator.state)}


class Sink:
    """Use the unchanged Phase 5E transport, adding a trace hash chain/context."""
    def __init__(self,writer,budget,algorithm,case_ids):
        self.writer=writer;self.budget=budget;self.algorithm=algorithm;self.case_ids=set(case_ids)
        self.case_id='not_observed';self.episode=-1;self.previous='0'*64
    def emit(self,event,**fields):
        self.budget.check()
        if event=='reset':
            if fields['case_id'] not in self.case_ids:raise ValueError('Reset outside registered Fixture A training cases')
            self.case_id=fields['case_id'];self.episode=fields['episode']
        fields['phase5f_context']={'algorithm':self.algorithm,'registered_case_id':self.case_id,'episode':self.episode,
            'accepted_or_reserved_action':self.budget.counts[self.algorithm],'source_kind':'live_actual'}
        fields['phase5f_previous_sha256']=self.previous
        digest=checksum({'event':event,'fields':fields})
        self.writer.emit(event,**fields,phase5f_chain_sha256=digest)
        self.previous=digest


class ActionRecorder(gym.Wrapper):
    def __init__(self,base,budget,algorithm,journal,sink):
        super().__init__(base);self.budget=budget;self.algorithm=algorithm;self.journal=journal;self.sink=sink
        self.previous=None;self.choice={'mode':'not_observed'};self.model=None
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self.previous=obs.copy()
        self.journal.append('reset',algorithm=self.algorithm,episode=self.sink.episode,case_id=self.sink.case_id,
            seed=kwargs.get('seed'),observation_sha256=observed_hash(obs))
        return obs,info
    def step(self,action):
        n=self.budget.action(self.algorithm)  # Flushed before the one real env.step.
        before=observed_hash(self.previous)
        choice=dict(self.choice)
        context={'mode':'stochastic_categorical','source_kind':'live_actual'} if self.algorithm=='ppo' else choice
        context['model_num_timesteps_before']=int(self.model.num_timesteps)
        context['library_updates_before']=int(self.model._n_updates)
        context['epsilon']=float(self.model.exploration_rate) if self.algorithm=='dqn' else 'not_applicable'
        obs,reward,terminated,truncated,info=self.env.step(action)
        self.journal.append('action_accepted',algorithm=self.algorithm,transition_id=n,episode=self.sink.episode,
            case_id=self.sink.case_id,action=int(action),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
            outcome=info['outcome'],observation_sha256=before,next_observation_sha256=observed_hash(obs),
            selection_context=context)
        self.previous=obs.copy()
        return obs,reward,terminated,truncated,info


class Calls:
    """Passive independent call accounting, plus once-only durable operation taps."""
    def __init__(self,policy,env,budget,journal,sink):
        self.model=policy.model;self.env=env;self.budget=budget;self.journal=journal;self.sink=sink
        self.algorithm=policy.algorithm;self.counts={};self.stack=ExitStack();self.generations={};self.policy_branch=False
    def __enter__(self):
        if sys.getprofile() is not None:raise RuntimeError('Unsupported existing profiling hook')
        m=self.model;self.codes={BaseBuffer.sample.__code__:'replay_sample_calls',ReplayBuffer._get_samples.__code__:'replay_consumptions',
            RolloutBuffer._get_samples.__code__:'rollout_consumptions',RolloutBuffer.get.__code__:'rollout_get',
            ActorCriticPolicy.forward.__code__:'actor_forwards',ActorCriticPolicy.predict_values.__code__:'critic_forwards',
            ActorCriticPolicy.evaluate_actions.__code__:'policy_training_evaluations',
            OffPolicyAlgorithm._sample_action.__code__:'action_selection'}
        self.codes[inspect.unwrap(torch.optim.Adam.step).__code__]='native_optimizer_steps'
        if self.algorithm=='dqn':
            self.codes[type(m.q_net).forward.__code__]='q_forward'
            self.codes[m.policy.predict.__func__.__code__]='policy_prediction'
            original_add=m.replay_buffer.add
            def add(*args,**kwargs):
                b=m.replay_buffer;slot=int(b.pos);generation=self.generations.get(slot,0)+1
                actual=inspect.signature(original_add).bind(*args,**kwargs).arguments
                result=original_add(*args,**kwargs)  # One original insertion, not a second sample.
                self.generations[slot]=generation
                self.sink.emit('phase5f_replay_insert',transition_id=self.budget.counts['dqn'],slot=slot,generation=generation,env_index=0,
                    observation_sha256=observed_hash(actual['obs'][0]),next_observation_sha256=observed_hash(actual['next_obs'][0]),
                    actions=numeric(actual['action']),rewards=numeric(actual['reward']),stored_dones=numeric(b.dones[slot]),
                    stored_timeouts=numeric(b.timeouts[slot]),buffer_pos_after=int(b.pos),buffer_full_after=bool(b.full))
                return result
            self.stack.enter_context(patch.object(m.replay_buffer,'add',add))
        original_step=m.policy.optimizer.step
        def step(*args,**kwargs):
            n=self.budget.optimizer(self.algorithm)
            result=original_step(*args,**kwargs)  # The one actual optimizer call.
            self.journal.append('optimizer_finished',algorithm=self.algorithm,update_id=n,training_actions=int(m.num_timesteps))
            return result
        self.stack.enter_context(patch.object(m.policy.optimizer,'step',step))
        sys.setprofile(self.profile);return self
    def __exit__(self,*args):sys.setprofile(None);self.stack.close()
    def profile(self,frame,event,arg):
        name=self.codes.get(frame.f_code)
        if name is None:return
        obj=frame.f_locals.get('self');m=self.model
        if event=='call':
            if name=='action_selection' and obj is m:
                self.policy_branch=False
                self.env.choice={'mode':'warmup_action_space_sample' if m.num_timesteps<m.learning_starts else 'epsilon_branch_pending',
                    'learning_starts':int(m.learning_starts),'source_kind':'live_actual'}
            elif name=='policy_prediction' and obj is m.policy:self.policy_branch=True
            elif name=='q_forward' and obj in (m.q_net,m.q_net_target):
                key='online_forwards' if obj is m.q_net else 'target_forwards';self.counts[key]=self.counts.get(key,0)+1
            elif name=='native_optimizer_steps' and obj is m.policy.optimizer:self.counts[name]=self.counts.get(name,0)+1
            elif name in ('actor_forwards','critic_forwards','policy_training_evaluations') and obj is m.policy:
                self.counts[name]=self.counts.get(name,0)+1
            elif name in ('replay_sample_calls','replay_consumptions') and obj is getattr(m,'replay_buffer',None):
                self.counts[name]=self.counts.get(name,0)+1
            elif name=='rollout_consumptions' and obj is getattr(m,'rollout_buffer',None):
                self.counts[name]=self.counts.get(name,0)+1
        elif event=='return':
            if name=='action_selection' and obj is m and arg is not None:
                if m.num_timesteps>=m.learning_starts:
                    self.env.choice['mode']='policy_predict_branch' if self.policy_branch else 'epsilon_action_space_sample'
                self.counts['selected_actions']=self.counts.get('selected_actions',0)+1
            elif name=='rollout_get' and arg is not None and obj is getattr(m,'rollout_buffer',None) and frame.f_locals.get('start_idx')==0:
                self.counts['rollout_permutations']=self.counts.get('rollout_permutations',0)+1


def run_models(folder,manifest,cases,model_config):
    """Called exactly once, only after the immutable registration/tests gate."""
    journal=Journal(folder/'training-ledger.jsonl',quota_root=folder)
    budget=Budget(journal);results=[]
    try:
        for algorithm in ALGORITHMS:
            budget.start(algorithm);started=perf_counter()
            arm=folder/algorithm;arm.mkdir(exist_ok=False)
            random.seed(23121);np.random.seed(23121);torch.manual_seed(23121)
            base=PartialDynamicEnv(cases,configuration())
            metadata={'study_id':STUDY,'run_id':algorithm+'-B1-seed23121','algorithm':algorithm,'arm':'B1-relative-x4',
                'seed':23121,'source_kind':'live_actual','source_hash':manifest['source_digest'],
                'config_hash':checksum(model_config),'dependency_hash':checksum(manifest['library_contract']),'split':'train'}
            writer=TraceWriter(arm/'trace',metadata,byte_cap=536870912,quota_roots=[folder])
            sink=Sink(writer,budget,algorithm,[c['id'] for c in cases])
            env=ActionRecorder(base,budget,algorithm,journal,sink)
            policy=build_policy(env,algorithm,model_config,23121,'B1-relative-x4');env.model=policy.model
            base.np_random=np.random.default_rng(23121);base.action_space.seed(23121)
            policy.policy_id=f'{algorithm}/{STUDY}/B1-relative-x4/seed23121'
            before_rng=rng_signature(base);before_network=network_versions(policy.model)
            original_optimizer=policy.model.policy.optimizer.step;original_mean=torch.Tensor.mean;original_std=torch.Tensor.std
            try:
                with Calls(policy,env,budget,journal,sink) as calls:
                    with observe(policy,sink=sink,environment=base):
                        policy.model.learn(total_timesteps=500,reset_num_timesteps=False)
                budget.check()
                if budget.counts[algorithm]!=500 or policy.model.num_timesteps!=500:raise ValueError('Incomplete exact 500-action arm')
                if budget.updates[algorithm]!={'dqn':100,'ppo':50}[algorithm]:raise ValueError('Unexpected actual optimizer count')
                hooks=(sys.gettrace() is None and sys.getprofile() is None and torch.Tensor.mean is original_mean and
                    torch.Tensor.std is original_std and policy.model.policy.optimizer.step==original_optimizer)
                if not hooks:raise ValueError('Observer hooks were not restored')
                # One final inference/optimizer checkpoint, never loaded/scored for evaluation.
                if quota_bytes([folder])+4194304>536870912:raise RuntimeError('Artifact reservation before checkpoint')
                policy.save(arm/'policy.zip',{'study_id':STUDY,'classification':'live_actual','training_actions':500,
                    'manifest_sha256':manifest['manifest_sha256'],'checkpoint_role':'Final provenance only; no selection/evaluation'})
                buffer=policy.model.replay_buffer if algorithm=='dqn' else policy.model.rollout_buffer
                if algorithm=='dqn':
                    np.savez_compressed(arm/'replay-prefix.npz',observations=buffer.observations[:500],
                        next_observations=buffer.next_observations[:500],actions=buffer.actions[:500],rewards=buffer.rewards[:500],
                        dones=buffer.dones[:500],timeouts=buffer.timeouts[:500])
                result={'algorithm':algorithm,'training_actions':500,'optimizer_steps':budget.updates[algorithm],
                    'library_update_counter':int(policy.model._n_updates),'profile_counts':calls.counts,
                    'network_before':before_network,'network_after':network_versions(policy.model),
                    'optimizer_final_sha256':continuity_hash(policy.model.policy.optimizer.state_dict()),
                    'buffer_final_sha256':continuity_hash(buffer),'rng_before':before_rng,'rng_after':rng_signature(base),
                    'observer_hooks_restored':hooks,'wall_seconds':perf_counter()-started,
                    'live_bitwise_on_off_comparison':'not_observed: uninstrumented research run not authorized',
                    'research_assessment_actions':0,'final_unseen_evaluations':0}
                write_json(arm/'run-receipt.json',result);results.append(result)
                journal.append('arm_finished',algorithm=algorithm,training_actions=500,optimizer_steps=budget.updates[algorithm])
            finally:writer.close()
            del policy,env,base;gc.collect()
        budget.check()
        journal.append('pilot_finished',training_actions=1000,optimizer_steps=sum(budget.updates.values()))
        return {'status':'runs_completed','arms':results,'scientific_training_actions':sum(budget.counts.values()),
            'optimizer_steps':sum(budget.updates.values()),'wall_seconds':perf_counter()-budget.started,
            'research_evaluation_actions':0,'final_unseen_evaluations':0}
    except BaseException as exc:
        try:journal.append('pilot_failed',error=repr(exc),counts=budget.counts,updates=budget.updates)
        except BaseException:pass
        raise
    finally:journal.close()
