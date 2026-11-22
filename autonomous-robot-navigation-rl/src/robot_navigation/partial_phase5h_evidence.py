"""Compact ledgers, detached assessments and one read-only H validation pass."""
import gzip,hashlib,json,os,random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import gymnasium as gym
from .core import ROOT,read_json,write_json
from .release import sha
from .partial_contract import configuration,MOVES
from .partial_dynamic_env import PartialDynamicEnv
from .partial_features_scaled import build_policy
from .partial_phase5b_diagnostics import continuity_hash
from .partial_instrumentation_trace import canonical,checksum,finite_tree,array
from .partial_phase5f_io import observed_hash
from .partial_phase5g_io import cases,direction
from .partial_phase5g_assess import first_metrics,public_position,global_rng
from .partial_phase5h_auxiliary import attach,parameter_counts,SEED

STUDY='phase5h-fast-goal-conditioning-v1';BASE='33f5843aa6bf3a7188757b6b13cff361de697ecf'
ARMS=('H0','H1');CHECKPOINTS=(0,2000)
def location(root=ROOT):return root/'artifacts/partial-dynamic-navigation-v1'/STUDY
def require(v,message):
    if not v:raise ValueError(message)

class Stream:
    """Constant-cost byte reservations, not directory scans per transition."""
    def __init__(self,path,cap=32*1024**2):
        self.file=gzip.open(path,'xb',compresslevel=3);self.cap=cap;self.bytes=0;self.n=0;self.previous='0'*64
    def write(self,row):
        finite_tree(row);body={'sequence':self.n+1,'previous':self.previous,**row};h=checksum(body)
        payload=canonical({'record':body,'sha256':h})+b'\n'
        if self.bytes+len(payload)>self.cap:raise RuntimeError('Evidence stream quota before write')
        self.file.write(payload);self.file.flush();self.bytes+=len(payload);self.n+=1;self.previous=h
    def close(self):self.file.close()

def records(path):
    previous='0'*64
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            item=json.loads(line);r=item['record'];finite_tree(r)
            require(r['sequence']==i and r['previous']==previous and checksum(r)==item['sha256'],'Compact ledger hash/order')
            previous=item['sha256'];yield r

class TrainingEnv(gym.Wrapper):
    def __init__(self,base,stream,arm,cap=2000,wall=1200):
        super().__init__(base);self.stream=stream;self.arm=arm;self.cap=cap;self.wall=wall;self.started=perf_counter()
        self.n=0;self.episode=-1;self.last=None;self.obs=None;self.model=None;self.case_id=None
        self.case_pairs={(tuple(c['start']),tuple(c['goal'])):c['id'] for c in cases()['training_cases']}
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self.episode+=1;self.obs=obs.copy()
        pos=public_position(obs);goal=np.rint(obs[2:4]*9).astype(int).tolist()
        self.case_id=self.case_pairs[(tuple(pos),tuple(goal))]
        self.stream.write({'event':'reset','arm':self.arm,'episode':self.episode,'case_id':self.case_id,'seed':kwargs.get('seed'),'observation_sha256':observed_hash(obs)})
        return obs,info
    def step(self,action):
        if self.n>=self.cap or perf_counter()-self.started>=self.wall:raise RuntimeError('Training action/time ceiling before step')
        n=self.n+1;self.stream.write({'event':'reserve_action','arm':self.arm,'transition_id':n});self.n=n
        before=self.obs;obs,reward,term,trunc,info=self.env.step(action)
        row={'event':'step','arm':self.arm,'transition_id':n,'episode':self.episode,'case_id':self.case_id,
            'position':public_position(before),'goal':np.rint(before[2:4]*9).astype(int).tolist(),'next_position':public_position(obs),
            'action':int(action),'reward':float(reward),'terminated':bool(term),'truncated':bool(trunc),'outcome':info['outcome'],
            'observation_sha256':observed_hash(before),'next_observation_sha256':observed_hash(obs),
            'epsilon':float(self.model.exploration_rate) if self.model is not None else None}
        self.stream.write(row);self.last=row;self.obs=obs.copy();return obs,reward,term,trunc,info

def model(env,arm,config):
    p=build_policy(env,'dqn',config,SEED,'B1-relative-x4')
    if arm=='H1':attach(p.model)
    return p

def save_checkpoint(policy,env,path,arm,cp,plan_sha):
    m=policy.model;before={'policy':continuity_hash(m.policy.state_dict()),'optimizer':continuity_hash(m.policy.optimizer.state_dict()),
        'rng':global_rng(),'case_rng':continuity_hash(env.unwrapped.np_random.bit_generator.state),
        'action_rng':continuity_hash(env.action_space.np_random.bit_generator.state)}
    torch.save({'policy':m.policy.state_dict(),'optimizer':m.policy.optimizer.state_dict()},path)
    after={'policy':continuity_hash(m.policy.state_dict()),'optimizer':continuity_hash(m.policy.optimizer.state_dict()),
        'rng':global_rng(),'case_rng':continuity_hash(env.unwrapped.np_random.bit_generator.state),
        'action_rng':continuity_hash(env.action_space.np_random.bit_generator.state)}
    require(before==after,'Checkpoint changed state/RNG')
    write_json(path.with_suffix('.json'),{'arm':arm,'training_actions':cp,'plan_sha256':plan_sha,'sha256':sha(path),
        'state':before,'parameter_counts':parameter_counts(m),'num_timesteps':int(m.num_timesteps),
        'optimizer_updates':int(m._n_updates),'state_unchanged_by_save':True})

def assess(arm,offset):
    folder=location();plan=read_json(folder/'plan.json');require((folder/'run.lock').exists(),'One registered run')
    launch=read_json(folder/f'launch-{arm}.json');require(launch['offset']==offset,'Registered assessment offset')
    started=perf_counter();panels=cases();ordered=panels['original_cases']+panels['expanded_cases'];rows=[];n=offset
    stream=Stream(folder/f'assessment-{arm}.jsonl.gz',cap=16*1024**2);torch.set_num_threads(1)
    try:
        for cp in CHECKPOINTS:
            file=folder/f'{arm}-{cp:04d}.pt';meta=read_json(file.with_suffix('.json'));require(sha(file)==meta['sha256'],'Checkpoint hash before load')
            env=PartialDynamicEnv(panels['training_cases'],configuration());p=model(env,arm,plan['model'])
            saved=torch.load(file,map_location='cpu',weights_only=True);p.model.policy.load_state_dict(saved['policy']);p.model.policy.optimizer.load_state_dict(saved['optimizer'])
            require(continuity_hash(p.model.policy.state_dict())==meta['state']['policy'] and continuity_hash(p.model.policy.optimizer.state_dict())==meta['state']['optimizer'],'Checkpoint/load equality')
            before=global_rng();state=continuity_hash(p.model.policy.state_dict());optim=continuity_hash(p.model.policy.optimizer.state_dict())
            for ix,c in enumerate(ordered):
                e=PartialDynamicEnv([c],configuration());obs,_=e.reset(seed=0);trajectory=[];path=[public_position(obs)]
                with torch.inference_mode():q=p.model.q_net(torch.as_tensor(obs).unsqueeze(0))[0].tolist()
                while True:
                    require(n<16000 and len(trajectory)<200,'Assessment cap before step')
                    require(perf_counter()-started+launch['elapsed_before']<1800,'Assessment wall cap')
                    action=p.act(obs);n+=1;stream.write({'event':'reserve','arm':arm,'checkpoint':cp,'case_id':c['id'],'global_action':n})
                    nxt,r,term,trunc,info=e.step(action)
                    t={'action':action,'reward':float(r),'terminated':bool(term),'truncated':bool(trunc),'outcome':info['outcome'],
                        'position':public_position(obs),'next_position':public_position(nxt),'global_action':n}
                    stream.write({'event':'step','arm':arm,'checkpoint':cp,'case_id':c['id'],**t});trajectory.append(t);path.append(t['next_position']);obs=nxt
                    if term or trunc:break
                metrics=first_metrics(c,trajectory[0]['action'],{'action_values':q},'dqn')
                rows.append({'arm':arm,'checkpoint':cp,'case_id':c['id'],'panel':'original' if ix<4 else 'expanded','direction':direction(c),
                    'Q':q,'first_action':trajectory[0]['action'],'first_metrics':metrics,'outcome':trajectory[-1]['outcome'],
                    'steps':len(trajectory),'path':path,'trajectory':trajectory})
            require(state==continuity_hash(p.model.policy.state_dict()) and optim==continuity_hash(p.model.policy.optimizer.state_dict()) and before==global_rng(),'Assessment state/RNG mutation')
        stream.write({'event':'closed','arm':arm,'actions':n-offset,'cases':40,'inference_only':True})
        result={'arm':arm,'cases':rows,'actions':n-offset,'wall_seconds':perf_counter()-started,'checkpoint_load_equal':True,'model_optimizer_rng_preserved':True}
        write_json(folder/f'cases-{arm}.json',result);return result
    finally:stream.close()

def numerical_check(r):
    reward=array(r['rewards']);done=array(r['dones']);q=array(r['current_Q']);nxt=array(r['next_target_max']);target=array(r['actual_target'])
    require(np.allclose(target,reward+(1-done)*r['discount']*nxt,atol=2e-5,rtol=2e-6),'TD target arithmetic')
    error=target.astype(float)-q;absolute=abs(error);td=float(np.where(absolute<1,.5*error**2,absolute-.5).mean())
    require(np.isclose(td,r['td_loss'],atol=2e-5,rtol=2e-6),'TD Huber loss')
    coords=np.rint(array(r['public_coordinates'])*9).astype(int);delta=coords[:,2:]-coords[:,:2];dr,dc=delta.T
    lab=np.where(abs(dr)>=abs(dc),np.where(dr<0,0,1),np.where(dc<0,2,3));lab[(dr==0)&(dc==0)]=-1
    require(lab.tolist()==r['direction_labels'],'Public direction sign/tie/at-goal labels')
    ce=0.
    if r['direction_logits'] is not None:
        logits=array(r['direction_logits']).astype(float);valid=lab>=0;x=logits[valid];y=lab[valid]
        if len(x):ce=float((np.log(np.exp(x-x.max(1,keepdims=True)).sum(1))+x.max(1)-x[np.arange(len(x)),y]).mean())
        require(r['auxiliary_valid']==int(valid.sum()) and r['auxiliary_correct']==int((x.argmax(1)==y).sum()),'Replay-only accuracy')
    require(np.isclose(ce,r['auxiliary_loss'],atol=2e-5,rtol=2e-6) and np.isclose(td+r['lambda']*ce,r['total_loss'],atol=2e-5,rtol=2e-6),'Auxiliary/total objective arithmetic')
    require(np.isfinite(r['clipped_gradient_norm']) and r['clipped_gradient_norm']<=10.00002,'Clipped actual gradient')

def validate(folder,plan):
    """One offline reconciliation; no construction, forward, action or training."""
    summary={};total_eval=0;panel=cases();allowed={c['id'] for c in panel['training_cases']};all_cases=[]
    initial={}
    for arm in ARMS:
        training=list(records(folder/f'train-{arm}.jsonl.gz'));steps=[r for r in training if r['event']=='step'];res=[r for r in training if r['event']=='reserve_action']
        require(training[-1]['event']=='closed' and len(steps)==len(res)==2000,'Exact training action closure')
        require([r['transition_id'] for r in steps]==list(range(1,2001)),'Training IDs')
        amap={r['transition_id']:r for r in steps};resets=[r for r in training if r['event']=='reset'];previous=None
        for z,a in zip(res,steps):
            require(z['transition_id']==a['transition_id'] and z['sequence']<a['sequence'],'Pre-action reservation')
            require(a['case_id'] in allowed,'Training split')
            if previous and not(previous['terminated'] or previous['truncated']):require(a['observation_sha256']==previous['next_observation_sha256'] and a['episode']==previous['episode'],'Training continuity')
            previous=a
        require(len(resets)==sum(s['terminated'] or s['truncated'] for s in steps)+1,'Reset terminal linkage')
        reserves=[r for r in training if r['event']=='reserve_optimizer']
        require(len(reserves)==475 and [r['update_id'] for r in reserves]==list(range(1,476)),'All optimizer operations pre-reserved')
        for a in steps:
            require(a['outcome'] in ('running','success','collision','timeout'),'Actual environment outcome')
            require(a['terminated']==(a['outcome'] in ('success','collision')) and a['truncated']==(a['outcome']=='timeout'),'Raw train terminal versus timeout')
            require(np.isclose(a['reward'],{'success':9.99,'collision':-5.01,'running':-.01,'timeout':-.01}[a['outcome']],atol=1e-10),'Raw train reward')
            epsilon=0. if a['transition_id']==1 else max(.1,1.-.9*(a['transition_id']-1)/1600)
            require(np.isclose(a['epsilon'],epsilon,atol=1e-10),'Original horizon epsilon')
        updates=list(records(folder/f'updates-{arm}.jsonl.gz'));ops=[r for r in updates if r['event']=='update'];sync=[r for r in updates if r['event']=='target_sync']
        require(len(ops)==475 and [o['update_id'] for o in ops]==list(range(1,476)) and [s['training_actions'] for s in sync]==list(range(250,2001,250)),'Optimizer/target schedule')
        saved0=torch.load(folder/f'{arm}-0000.pt',map_location='cpu',weights_only=True)
        def network(saved):return {name:continuity_hash({k[len(prefix):]:v for k,v in saved['policy'].items() if k.startswith(prefix)}) for name,prefix in [('online','q_net.'),('target','q_net_target.')]}
        def auxiliary(saved):
            values={k[len('goal_direction_head.'):]:v for k,v in saved['policy'].items() if k.startswith('goal_direction_head.')}
            return continuity_hash(values) if values else 'not_applicable'
        goal_draws=[];goal_updates=set();accuracy=[0,0];version=network(saved0);opt=continuity_hash(saved0['optimizer']);aux=auxiliary(saved0)
        component_losses={'td':[],'auxiliary':[],'total':[]};example=None
        for row in updates:
            if row['event']=='closed':continue
            if row['event']=='target_sync':
                if version is not None:require(row['before']==version,'Target lineage')
                version=row['after'];require(version['online']==version['target'],'Hard target copy');continue
            numerical_check(row);require(row['lambda']==(0. if arm=='H0' else .05),'Frozen coefficient')
            for k,field in [('td','td_loss'),('auxiliary','auxiliary_loss'),('total','total_loss')]:component_losses[k].append(row[field])
            if version is not None:require(row['network_before']==version,'Online lineage')
            if opt is not None:require(row['optimizer_before']==opt and row['auxiliary_before']==aux,'Optimizer/head lineage')
            version=row['network_after'];opt=row['optimizer_after'];aux=row['auxiliary_after']
            require(row['training_actions']==100+4*row['update_id'],'One original 2000-step horizon')
            for i,n in enumerate(row['sample_ids']):
                a=amap[n];require(n<=row['training_actions'] and row['indices'][i]==n-1 and row['generations'][i]==1,'Actual prior replay identity')
                require(row['sample_observation_sha256'][i]==a['observation_sha256'] and row['sample_next_sha256'][i]==a['next_observation_sha256'],'Consumed replay hashes')
                require(np.isclose(array(row['rewards'])[i],a['reward'],atol=2e-5) and array(row['dones'])[i]==a['terminated'],'Replay reward/timeout mask')
                require(np.array_equal(np.rint(array(row['public_coordinates'])[i]*9).astype(int),a['position']+a['goal']),'Public label source')
                if a['outcome']=='success':
                    goal_draws.append(n);goal_updates.add(row['update_id'])
                    if example is None:example={'transition_id':n,'case_id':a['case_id'],'update_id':row['update_id'],'sample_row':i,
                        'used_at':row['training_actions'],'actual_target':float(array(row['actual_target'])[i]),'chosen_Q':float(array(row['current_Q'])[i])}
            if arm=='H1':accuracy[0]+=row['auxiliary_correct'];accuracy[1]+=row['auxiliary_valid']
        for cp in CHECKPOINTS:
            file=folder/f'{arm}-{cp:04d}.pt';meta=read_json(file.with_suffix('.json'));require(sha(file)==meta['sha256'] and meta['state_unchanged_by_save'],'Checkpoint hashes/save')
            require(meta['num_timesteps']==cp and meta['optimizer_updates']==(0 if cp==0 else 475),'Exact checkpoint')
            if cp==0:initial[arm]=meta
            saved=saved0 if cp==0 else torch.load(file,map_location='cpu',weights_only=True)
            require(continuity_hash(saved['policy'])==meta['state']['policy'] and continuity_hash(saved['optimizer'])==meta['state']['optimizer'],'Saved policy/optimizer state')
            if cp==2000:require(network(saved)==version and auxiliary(saved)==aux and continuity_hash(saved['optimizer'])==opt,'Final checkpoint/update lineage')
        eval_rows=list(records(folder/f'assessment-{arm}.jsonl.gz'));accepted=[r for r in eval_rows if r['event']=='step'];reserved=[r for r in eval_rows if r['event']=='reserve']
        result=read_json(folder/f'cases-{arm}.json');require(len(result['cases'])==40 and len(accepted)==len(reserved)==result['actions'] and eval_rows[-1]['event']=='closed','Assessment accounting')
        require(result['checkpoint_load_equal'] and result['model_optimizer_rng_preserved'],'Detached inference state')
        for z,a in zip(reserved,accepted):require(z['global_action']==a['global_action'] and z['sequence']<a['sequence'],'Assessment pre-reservation')
        require([a['global_action'] for a in accepted]==list(range(total_eval+1,total_eval+len(accepted)+1)),'Global evaluation IDs');total_eval+=len(accepted)
        nodes={};cursor=0
        ordered=panel['original_cases']+panel['expanded_cases']
        for cp in CHECKPOINTS:
            part=[c for c in result['cases'] if c['checkpoint']==cp];require([c['case_id'] for c in part]==[c['id'] for c in ordered],'Fixed panel order')
            for r,c in zip(part,ordered):
                require(r['first_metrics']==first_metrics(c,r['first_action'],{'action_values':r['Q']},'dqn'),'First-action vector/metrics')
                require(r['first_action']==r['trajectory'][0]['action'],'Actual first action');pos=c['start'];path=[pos];seen={tuple(pos)};revisits=0;wait=0
                for i,t in enumerate(r['trajectory'],1):
                    dr,dc=MOVES[t['action']];dest=[pos[0]+dr,pos[1]+dc];collision=not all(0<=v<10 for v in dest);nxt=pos if collision else dest
                    success=not collision and nxt==c['goal'];timeout=not collision and not success and i==200
                    outcome='collision' if collision else 'success' if success else 'timeout' if timeout else 'running'
                    require(i<=200 and t['position']==pos and t['next_position']==nxt and t['outcome']==outcome and t['terminated']==(collision or success) and t['truncated']==timeout,'Public open-grid path/flags')
                    require(np.isclose(t['reward'],9.99 if success else -5.01 if collision else -.01,atol=1e-10),'Unchanged reward')
                    require(all(accepted[cursor][k]==v for k,v in t.items()),'Case/ledger join');cursor+=1
                    if nxt!=pos:revisits+=tuple(nxt) in seen;seen.add(tuple(nxt))
                    wait+=t['action']==4;pos=nxt;path.append(pos)
                    require(i==len(r['trajectory']) or outcome=='running','No actions after terminal')
                require(r['steps']==len(r['trajectory']) and r['outcome']==outcome!='running' and r['path']==path,'Complete case outcome')
                distance=sum(abs(x-y) for x,y in zip(c['start'],c['goal']));r.update(wait_count=wait,moving_revisits=int(revisits),
                    ABAB=sum(path[i]==path[i+2] and path[i+1]==path[i+3] and path[i]!=path[i+1] for i in range(len(path)-3)),
                    shortest_path=distance,inefficiency=r['steps']/distance if outcome=='success' else None)
            expanded=part[4:];actions=sum(r['steps'] for r in part);wait=sum(r['wait_count'] for r in part)
            nodes[str(cp)]={'original_success':sum(r['outcome']=='success' for r in part[:4]),'expanded_success':sum(r['outcome']=='success' for r in expanded),
                'directions':{d:sum(r['outcome']=='success' for r in expanded if r['direction']==d) for d in ('Up','Down','Left','Right')},
                'first_aligned':sum(r['first_metrics']['aligned'] for r in expanded),'first_aligned_denominator':16,'wait_count':wait,'actions':actions,'wait_fraction':wait/actions,
                'timeouts':sum(r['outcome']=='timeout' for r in part),'collisions':sum(r['outcome']=='collision' for r in part),
                'moving_revisits':sum(r['moving_revisits'] for r in part),'ABAB':sum(r['ABAB'] for r in part)}
            all_cases.extend(part)
        require(cursor==len(accepted),'All assessment actions joined')
        summary[arm]={'checkpoints':nodes,'goal_encounters':sum(a['outcome']=='success' for a in steps),'goal_sample_draws':len(goal_draws),
            'distinct_goals_sampled':len(set(goal_draws)),'updates_with_goals':len(goal_updates),'goal_example':example,
            'optimizer_steps':475,'training_actions':2000,'auxiliary_correct':accuracy[0] if arm=='H1' else None,
            'auxiliary_training_draws':accuracy[1] if arm=='H1' else None,'accuracy':accuracy[0]/accuracy[1] if accuracy[1] else None,
            'losses':{k:{'mean':float(np.mean(v)),'min':float(min(v)),'max':float(max(v))} for k,v in component_losses.items()}}
    h0=torch.load(folder/'H0-0000.pt',map_location='cpu',weights_only=True)['policy'];h1=torch.load(folder/'H1-0000.pt',map_location='cpu',weights_only=True)['policy']
    require(all(torch.equal(v,h1[k]) for k,v in h0.items()),'Actual initial common tensors equal')
    require(total_eval<=16000,'Evaluation cap');pair=read_json(folder/'pairing.json');require(pair['common_initial_equal'] and pair['initial_rng_equal'],'PAIRING_INVALID')
    a,b=summary['H0']['checkpoints']['2000'],summary['H1']['checkpoints']['2000']
    checks={'at_least_12':b['expanded_success']>=12,'each_direction_2':all(n>=2 for n in b['directions'].values()),
        'success_advantage_3':b['expanded_success']-a['expanded_success']>=3,'alignment_advantage_3':b['first_aligned']-a['first_aligned']>=3,
        'wait_within_10_points':b['wait_fraction']<=a['wait_fraction']+.1,
        'no_new_complete_direction_collapse':not(any(v>0 for v in b['directions'].values()) and sum(v>0 for v in b['directions'].values())==1 and sum(v>0 for v in a['directions'].values())!=1)}
    favorable=b['expanded_success']>a['expanded_success'] or b['first_aligned']>a['first_aligned']
    verdict='PROMISING_REPLICATION_REQUIRED' if all(checks.values()) else 'INCONCLUSIVE_SMALL_PILOT' if favorable else 'INTERVENTION_NOT_SUPPORTED'
    did=(b['expanded_success']-summary['H1']['checkpoints']['0']['expanded_success'])-(a['expanded_success']-summary['H0']['checkpoints']['0']['expanded_success'])
    return {'verdict':verdict,'gates':{k:'passed' for k in ('G0','G1','G2','G3','G4')},'arms':summary,'decision_checks':checks,'difference_in_differences':did,
        'training_actions':4000,'optimizer_steps':950,'assessment_actions':total_eval,'assessment_episodes':80,'case_rows':all_cases,
        'full_regression_suite':'DEFERRED_TO_MAJOR_MILESTONE','not_claimed':['full gradient tensors','all five Q values per training transition','complete forward/RNG call census','Phase5G full tensor trace equivalence','causal mechanism','generalization or four-model ranking']}
