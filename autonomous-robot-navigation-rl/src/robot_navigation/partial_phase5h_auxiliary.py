"""Small auxiliary head and selective, once-only taps of unchanged SB3 DQN.train."""
from contextlib import ExitStack
import hashlib,inspect
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import DQN
from .partial_training_instrumentation import check_compatibility,network_versions
from .partial_phase5b_diagnostics import continuity_hash
from .partial_instrumentation_trace import numeric
from .partial_phase5f_io import observed_hash

SEED=23141

def labels(observations):
    """Public grid-coordinate delta; vertical wins magnitude ties; goal = -1."""
    coordinates=torch.round(observations[:,:4]*9).long()
    delta=coordinates[:,2:4]-coordinates[:,:2];dr,dc=delta[:,0],delta[:,1]
    vertical=dr.abs()>=dc.abs()
    result=torch.where(vertical,torch.where(dr<0,0,1),torch.where(dc<0,2,3))
    return torch.where((dr==0)&(dc==0),-1,result)

def attach(model):
    layers=model.q_net.q_net
    if (len(layers)!=5 or not isinstance(layers[0],nn.Linear) or layers[0].in_features!=412 or
        layers[0].out_features!=32 or layers[2].in_features!=32 or layers[2].out_features!=32 or
        not isinstance(layers[3],nn.ReLU) or layers[4].in_features!=32 or layers[4].out_features!=5):
        raise RuntimeError('INTERVENTION_BLOCKED: unsupported shared latent architecture')
    before=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED);head=nn.Linear(32,4).to('cpu')
    if not torch.equal(before,torch.get_rng_state()):raise RuntimeError('PAIRING_INVALID: head initialization RNG')
    model.policy.add_module('goal_direction_head',head)
    model.policy.optimizer.param_groups[0]['params'].extend(list(head.parameters()))
    return head

def parameter_counts(model):
    return {'online_q':sum(p.numel() for p in model.q_net.parameters()),
        'target_q':sum(p.numel() for p in model.q_net_target.parameters()),
        'auxiliary':sum(p.numel() for p in getattr(model.policy,'goal_direction_head',nn.Identity()).parameters()),
        'optimizer_parameters':sum(p.numel() for g in model.policy.optimizer.param_groups for p in g['params'])}

def gradient_norm(parameters):
    return float(np.sqrt(sum(float(torch.sum(p.grad.detach().double()**2)) for p in parameters if p.grad is not None)))

class LeanLearning:
    """No replacement learner, second replay draw, additional Q forward or update."""
    def __init__(self,model,coefficient,emit,reserve_optimizer,latest_transition):
        self.model=model;self.coefficient=coefficient;self.emit=emit;self.reserve=reserve_optimizer;self.latest=latest_transition
        self.stack=ExitStack();self.slots={};self.batch=None;self.latent=None;self.pending=None;self.updates=0;self.syncs=0
    def __enter__(self):
        check_compatibility();m=self.model
        if m.device.type!='cpu' or m.n_envs!=1 or m.observation_space.shape!=(409,) or m.action_space.n!=5 or m.replay_buffer.optimize_memory_usage:
            raise RuntimeError('INTERVENTION_BLOCKED: unsupported execution path')
        head=getattr(m.policy,'goal_direction_head',None)
        if self.coefficient and head is None:raise RuntimeError('INTERVENTION_BLOCKED: missing head')
        self.head=head
        original_add=m.replay_buffer.add
        def add(*args,**kwargs):
            slot=int(m.replay_buffer.pos);row=self.latest();result=original_add(*args,**kwargs)
            self.slots[slot]={'transition_id':row['transition_id'],'generation':self.slots.get(slot,{}).get('generation',0)+1}
            return result
        self.stack.enter_context(patch.object(m.replay_buffer,'add',add))
        original_sample=m.replay_buffer._get_samples
        def sample(batch_inds,env=None):
            result=original_sample(batch_inds,env)
            self.batch=result;self.ids=[self.slots[int(i)]['transition_id'] for i in batch_inds]
            self.indices=np.asarray(batch_inds).copy();self.latent=None
            return result
        self.stack.enter_context(patch.object(m.replay_buffer,'_get_samples',sample))
        def latent(module,args,result):
            if self.batch is not None and torch.is_grad_enabled():self.latent=result
        self.stack.callback(m.q_net.q_net[-2].register_forward_hook(latent).remove)
        original_loss=F.smooth_l1_loss
        def loss(current,target,*args,**kwargs):
            frame=inspect.currentframe().f_back
            if frame.f_code is not DQN.train.__code__ or frame.f_locals.get('self') is not m:
                return original_loss(current,target,*args,**kwargs)
            if self.batch is None or self.latent is None:raise RuntimeError('Missing actual replay/latent')
            td=original_loss(current,target,*args,**kwargs);r=self.batch
            lab=labels(r.observations);valid=lab>=0;logits=None;ce=td.new_zeros(())
            if head is not None:
                logits=head(self.latent);ce=F.cross_entropy(logits[valid],lab[valid]) if bool(valid.any()) else logits.sum()*0
            total=td+self.coefficient*ce
            loc=frame.f_locals
            self.pending={'event':'update','update_id':self.updates+1,'training_actions':int(m.num_timesteps),'indices':self.indices.tolist(),
                'sample_ids':self.ids,'generations':[self.slots[int(i)]['generation'] for i in self.indices],
                'sample_observation_sha256':[observed_hash(x) for x in r.observations.detach().numpy()],
                'sample_next_sha256':[observed_hash(x) for x in r.next_observations.detach().numpy()],
                'public_coordinates':numeric(r.observations[:,:4]),'rewards':numeric(r.rewards.flatten()),'dones':numeric(r.dones.flatten()),
                'current_Q':numeric(current.flatten()),'next_target_max':numeric(loc['next_q_values'].flatten()),
                'actual_target':numeric(target.flatten()),'discount':float(loc['discounts']),
                'td_loss':float(td.detach()),'auxiliary_loss':float(ce.detach()),'total_loss':float(total.detach()),'lambda':self.coefficient,
                'direction_labels':lab.detach().tolist(),'direction_logits':None if logits is None else numeric(logits),
                'auxiliary_valid':int(valid.sum()),'auxiliary_correct':None if logits is None else int((logits[valid].argmax(1)==lab[valid]).sum()),
                'network_before':network_versions(m),'optimizer_before':continuity_hash(m.policy.optimizer.state_dict())}
            self.pending['auxiliary_before']='not_applicable' if head is None else continuity_hash(head.state_dict())
            return total
        self.stack.enter_context(patch.object(F,'smooth_l1_loss',loss))
        original_step=m.policy.optimizer.step
        def step(*args,**kwargs):
            if self.pending is None:raise RuntimeError('Unobserved optimizer operation')
            self.reserve();r=self.pending
            r['clipped_gradient_norm']=gradient_norm(m.policy.parameters())
            r['trunk_gradient_norm']=gradient_norm(list(m.q_net.q_net[0].parameters())+list(m.q_net.q_net[2].parameters()))
            r['Q_head_gradient_norm']=gradient_norm(m.q_net.q_net[4].parameters())
            r['aux_head_gradient_norm']=0. if head is None else gradient_norm(head.parameters())
            result=original_step(*args,**kwargs);self.updates+=1
            r['network_after']=network_versions(m);r['optimizer_after']=continuity_hash(m.policy.optimizer.state_dict())
            r['auxiliary_after']='not_applicable' if head is None else continuity_hash(head.state_dict())
            self.emit(r);self.pending=None;self.batch=None;self.latent=None
            return result
        self.stack.enter_context(patch.object(m.policy.optimizer,'step',step))
        original_on_step=m._on_step
        def on_step():
            before=network_versions(m);result=original_on_step()
            if m._n_calls%m.target_update_interval==0:
                self.syncs+=1;self.emit({'event':'target_sync','training_actions':int(m.num_timesteps),'before':before,'after':network_versions(m)})
            return result
        self.stack.enter_context(patch.object(m,'_on_step',on_step));return self
    def __exit__(self,*args):self.stack.close()
