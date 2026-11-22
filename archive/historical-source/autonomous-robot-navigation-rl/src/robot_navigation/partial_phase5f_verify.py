"""Independent Phase 5F arithmetic and lineage, reading closed evidence only."""
from collections import Counter,defaultdict
import hashlib
import numpy as np

from .core import read_json
from .release import sha
from .partial_instrumentation_trace import read_trace,array,checksum
from .partial_instrumentation_audit import audit_records,near
from .partial_phase5f_io import ALGORITHMS,read_journal


def require(condition,message):
    if not condition:raise ValueError(message)


def tensor_hash(value):return hashlib.sha256(np.ascontiguousarray(value,dtype=np.float32).tobytes()).hexdigest()


def stats(value):
    v=np.asarray(value,dtype=float).reshape(-1)
    return {'n':len(v),'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),'std':float(v.std())} if len(v) else {'n':0,'status':'not_observed'}


def events(rows,event):return [r for r in rows if r['event']==event]


def chain(rows,manifest,algorithm):
    previous='0'*64
    for r in rows:
        fields={k:v for k,v in r.items() if k not in ('schema','sequence','event','source_kind','classification','run','phase5f_chain_sha256')}
        require(fields['phase5f_previous_sha256']==previous,'Trace chain predecessor')
        require(checksum({'event':r['event'],'fields':fields})==r['phase5f_chain_sha256'],'Trace chain digest')
        previous=r['phase5f_chain_sha256'];run=r['run']
        require(run['source_hash']==manifest['source_digest'] and run['algorithm']==algorithm and run['seed']==23121 and
            run['split']=='train' and run['arm']=='B1-relative-x4' and run['source_kind']=='live_actual','Trace provenance')
        require(run['config_hash']==checksum(manifest['model_config']) and run['dependency_hash']==checksum(manifest['library_contract']),'Trace dependency/config')
    return previous


def common(folder,manifest,algorithm,journal):
    rows=read_trace(folder/algorithm/'trace');last_hash=chain(rows,manifest,algorithm)
    receipt=read_json(folder/algorithm/'run-receipt.json')
    accepted=events([j for j in journal if j.get('algorithm')==algorithm],'action_accepted')
    reserved=events([j for j in journal if j.get('algorithm')==algorithm],'action_reserved')
    env=events(rows,'environment_transition');resets=events(rows,'reset');jr=events([j for j in journal if j.get('algorithm')==algorithm],'reset')
    require(len(accepted)==len(reserved)==len(env)==500,'Exact accepted/reserved/environment count')
    require([r['transition_id'] for r in accepted]==list(range(1,501)) and [r['transition_id'] for r in reserved]==list(range(1,501)),'Action IDs')
    require(len(jr)==len(resets),'Reset correspondence');allowed=set(manifest['case_ids']);reset_map={}
    for index,(r,j) in enumerate(zip(resets,jr)):
        require(r['episode']==j['episode']==index and r['case_id']==j['case_id'] and r['case_id'] in allowed,'Reset/episode/train case')
        require(r['seed']==j['seed']==(23121 if index==0 else None),'Original seeded first reset')
        require(tensor_hash(array(r['observation']))==j['observation_sha256'],'Reset observation')
        reset_map[index]=r
    previous=None;terminal_episodes=[];counts=Counter()
    for i,(a,e,z) in enumerate(zip(accepted,env,reserved),1):
        require(z['sequence']<a['sequence'] and e['transition_id']==a['transition_id']==i,'Action reservation/order')
        for k in ('episode','action','reward','terminated','truncated'):require(a[k]==e[k],'Raw action field '+k)
        require(a['case_id'] in allowed and a['case_id']==reset_map[a['episode']]['case_id']==e['phase5f_context']['registered_case_id'],'Step split/context')
        require(a['observation_sha256']==tensor_hash(array(e['observation'])) and a['next_observation_sha256']==tensor_hash(array(e['next_observation'])),'Step observation hash')
        first=previous is None or previous['terminated'] or previous['truncated']
        require(a['episode']==(0 if previous is None else previous['episode']+int(first)),'Episode linkage')
        expected= tensor_hash(array(reset_map[a['episode']]['observation'])) if first else previous['next_observation_sha256']
        require(a['observation_sha256']==expected,'Observation continuity')
        outcome=a['outcome'];counts[outcome]+=1
        require(outcome in ('running','success','collision','timeout'),'Public outcome')
        require(a['terminated']==(outcome in ('success','collision')) and a['truncated']==(outcome=='timeout'),'Terminal semantics')
        near(a['reward'],{'running':-.01,'success':9.99,'collision':-5.01,'timeout':-.01}[outcome],1e-10)
        require(a['selection_context']['model_num_timesteps_before']==i-1,'Real training timestep context')
        if a['terminated'] or a['truncated']:terminal_episodes.append(a['episode'])
        previous=a
    # SB3 autoresets after the final terminal action, potentially leaving a zero-action reset.
    require(len(resets)==len(terminal_episodes)+1,'Complete reset/terminal linkage')
    optim=events(rows,'optimizer');expected_updates={'dqn':100,'ppo':50}[algorithm]
    ores=events([j for j in journal if j.get('algorithm')==algorithm],'optimizer_reserved')
    odone=events([j for j in journal if j.get('algorithm')==algorithm],'optimizer_finished')
    require(len(optim)==len(ores)==len(odone)==expected_updates,'All actual optimizer operations accounted')
    version=receipt['network_before'];old_optimizer=None
    for r in rows:
        if r['event']=='optimizer':
            n=r['update_id'];require(r['network_before']==version,'Network continuity before optimizer')
            if old_optimizer is not None:require(r['optimizer_before']==old_optimizer,'Optimizer continuity')
            require(ores[n-1]['update_id']==odone[n-1]['update_id']==n and ores[n-1]['sequence']<odone[n-1]['sequence'],'Optimizer reservation/order')
            require(ores[n-1]['training_actions']==odone[n-1]['training_actions']==r['training_actions'],'Optimizer/action join')
            version=r['network_after'];old_optimizer=r['optimizer_after']
        elif r['event']=='target_sync':
            require(r['before']==version,'Target sync prior version');version=r['after']
            require(version['online']==version['target'] and r['tau']==1.,'Actual hard target copy')
    require(version==receipt['network_after'] and old_optimizer==receipt['optimizer_final_sha256'],'Final versions')
    require(receipt['observer_hooks_restored'] and receipt['training_actions']==500 and receipt['optimizer_steps']==expected_updates,'Final receipt')
    require(receipt['wall_seconds']<900 and receipt['research_assessment_actions']==receipt['final_unseen_evaluations']==0,'Arm scope/time')
    require(rows[-1]['interactions']==500 and rows[-1]['optimizer_steps']==expected_updates,'Observer completion')
    meta=read_json(folder/algorithm/'policy.json')
    require(meta['checkpoint_sha256']==sha(folder/algorithm/'policy.zip') and meta['manifest_sha256']==manifest['manifest_sha256'],'Checkpoint provenance bytes')
    math=audit_records(rows)
    return rows,accepted,receipt,{'training_actions':500,'optimizer_steps':len(optim),'terminal_outcomes':dict(counts),
        'resets':len(resets),'completed_episodes':len(terminal_episodes),'incomplete_episode_actions':
            sum(a['episode']==accepted[-1]['episode'] for a in accepted) if not (accepted[-1]['terminated'] or accepted[-1]['truncated']) else 0,
        'empty_final_reset':resets[-1]['episode']>accepted[-1]['episode'],'trace_records':len(rows),'trace_chain_sha256':last_hash,
        'checkpoint_sha256':meta['checkpoint_sha256'],'arithmetic':math,'wall_seconds':receipt['wall_seconds']}


def dqn(rows,actions,receipt,folder):
    inserted={};latest=None;draws=[];targets=[];residuals=[];goal_rows=[];sample_count=0;loss_count=0
    action_map={a['transition_id']:a for a in actions};insert_events=events(rows,'phase5f_replay_insert')
    for r in rows:
        if r['event']=='phase5f_replay_insert':
            n=r['transition_id'];a=action_map[n]
            require(n==len(inserted)+1 and r['slot']==n-1 and r['generation']==1 and r['env_index']==0,'Actual insertion slot/generation')
            require(r['observation_sha256']==a['observation_sha256'] and r['next_observation_sha256']==a['next_observation_sha256'],'Insertion actual observation')
            near(array(r['actions']),a['action']);near(array(r['rewards']),a['reward'])
            near(array(r['stored_dones']),a['terminated'] or a['truncated']);near(array(r['stored_timeouts']),a['truncated'])
            require(r['buffer_pos_after']==n and not r['buffer_full_after'],'Buffer position')
            inserted[n]=r
        elif r['event']=='replay_sample':
            sample_count+=1;latest=r;indices=array(r['indices']);envs=array(r['env_indices'])
            require(len(r['sample'])==len(indices)==50 and len(envs)==50,'Replay batch dimensions')
            for i,(s,idx,env) in enumerate(zip(r['sample'],indices,envs)):
                n=s['transition_id'];require(n in inserted,'Sample before insertion or unknown')
                require(s['slot']==int(idx)==inserted[n]['slot'] and s['generation']==inserted[n]['generation'] and s['env_index']==env==0,'Ordered replay identity')
                a=action_map[n];require(s['terminated']==a['terminated'] and s['truncated']==a['truncated'] and s['episode']==a['episode'],'Replay flags')
                near(array(r['raw_dones'])[i],a['terminated'] or a['truncated']);near(array(r['raw_timeouts'])[i],a['truncated'])
        elif r['event']=='dqn_loss':
            loss_count+=1;require(latest is not None and latest['sample']==r['sample'],'Consumed sample matches actual sampler')
            require(r['update_id']==loss_count and r['training_actions']==100+4*loss_count,'Original DQN schedule')
            require(r['observation']['shape']==r['next_observation']['shape']==[50,409],'Consumed observation shape')
            for i,s in enumerate(r['sample']):
                n=s['transition_id'];a=action_map[n];ins=inserted[n];draws.append(n)
                require(tensor_hash(array(r['observation'])[i])==ins['observation_sha256'] and tensor_hash(array(r['next_observation'])[i])==ins['next_observation_sha256'],'Consumed observations resolve insertion')
                near(array(r['actions'])[i],a['action']);near(array(r['reward'])[i],a['reward']);near(array(r['consumed_dones'])[i],a['terminated'])
                if a['outcome']=='success':goal_rows.append({'transition_id':n,'update_id':r['update_id'],'sample_row':i,'trace_sequence':r['sequence'],
                    'reward':a['reward'],'actual_target':float(array(r['actual_target'])[i].item()),'chosen_Q':float(array(r['current_chosen_Q'])[i].item()),
                    'residual_snapshot':float(array(r['td_residual'])[i].item()),'case_id':a['case_id'],'slot':s['slot'],'generation':s['generation']})
            targets.extend(array(r['actual_target']).ravel());residuals.extend(array(r['td_residual']).ravel());latest=None
    require(len(inserted)==500 and sample_count==loss_count==100,'DQN signal coverage')
    require([r['training_actions'] for r in events(rows,'target_sync')]==[250,500],'Target sync schedule')
    for a in actions:
        n=a['transition_id'];expected=0. if n==1 else max(.1,1.-.9*((n-1)/500)/.8)
        near(a['selection_context']['epsilon'],expected,1e-10)
        require(a['selection_context']['mode'] in ('warmup_action_space_sample','policy_predict_branch','epsilon_action_space_sample'),'Actual DQN action branch')
    predictions=sum(a['selection_context']['mode']=='policy_predict_branch' for a in actions)
    c=receipt['profile_counts']
    for k,v in {'native_optimizer_steps':100,'replay_sample_calls':100,'replay_consumptions':100,'selected_actions':500,'target_forwards':100,'online_forwards':100+predictions}.items():
        require(c.get(k,0)==v,'Independent DQN call witness '+k)
    with np.load(folder/'dqn/replay-prefix.npz',allow_pickle=False) as b:
        for i,a in enumerate(actions):
            require(tensor_hash(b['observations'][i,0])==a['observation_sha256'] and tensor_hash(b['next_observations'][i,0])==a['next_observation_sha256'],'Final replay prefix observation')
            near(b['actions'][i,0],a['action']);near(b['rewards'][i,0],a['reward']);near(b['dones'][i,0],a['terminated'] or a['truncated']);near(b['timeouts'][i,0],a['truncated'])
    require(receipt['library_update_counter']==100,'DQN library counter')
    goals=[a['transition_id'] for a in actions if a['outcome']=='success'];frequencies=Counter(g['transition_id'] for g in goal_rows)
    return {'recorded_goal_experiences':len(goals),'goal_sample_draws':len(goal_rows),'distinct_sampled_goals':len(frequencies),
        'goal_draw_frequencies':dict(frequencies),'all_sample_draws':len(draws),'distinct_sampled_experiences':len(set(draws)),
        'actual_target':stats(targets),'residual_snapshot_reconstructed':stats(residuals),'target_syncs':2,
        'reward_to_update_example':goal_rows[0] if goal_rows else {'status':'not_observed','recorded_goals':len(goals),'goal_draws':0},
        'profile_counts':c,'required_missing_fields':[]}


def ppo(rows,actions,receipt):
    amap={a['transition_id']:a for a in actions};behavior={r['transition_id']:r for r in events(rows,'ppo_behavior')}
    additions=events(rows,'rollout_add');gaei={r['rollout_id']:r for r in events(rows,'gae_input')};gaer={r['rollout_id']:r for r in events(rows,'gae_result')}
    timeouts={r['transition_id']:r for r in events(rows,'ppo_timeout_bootstrap')};byrow={};perms={};sample=None
    require(len(behavior)==len(additions)==500 and len(gaei)==len(gaer)==5,'PPO collector/GAE coverage')
    for add in additions:
        n=add['transition_id'];a=amap[n];b=behavior[n];roll=(n-1)//100+1;idx=(n-1)%100
        require(add['rollout_id']==b['rollout_id']==roll and add['row']==idx and add['env_index']==0,'Actual rollout row')
        require(tensor_hash(array(add['observation']))==a['observation_sha256']==tensor_hash(array(b['observation'])),'Behavior/row observation')
        near(array(add['action']),a['action']);near(array(b['action']),a['action'])
        near(array(add['values']),array(b['values']));near(array(add['log_probs']),array(b['log_probs']))
        probs=array(b['probabilities']);require(probs.shape==(1,5) and (probs>0).all(),'Categorical probabilities')
        near(probs.sum(1),1);near(array(b['log_probs']),np.log(probs[0,a['action']]))
        require(add['raw_reward']==a['reward'] and add['terminated']==a['terminated'] and add['truncated']==a['truncated'],'Raw collector fields')
        if a['truncated']:
            require(n in timeouts,'Missing actual timeout boundary signal');t=timeouts[n]
            require(t['raw_reward']==a['reward'] and tensor_hash(array(t['terminal_observation']))==a['next_observation_sha256'],'Timeout terminal observation/reward')
            near(array(add['adjusted_reward']),a['reward']+t['gamma']*array(t['terminal_value']))
        else:near(array(add['adjusted_reward']),a['reward'])
        g=gaei[roll];identity={'rollout_id':roll,'row':idx,'transition_id':n,'env_index':0}
        require(g['rows'][idx]==identity,'GAE row identity');near(array(g['values'])[idx],array(add['values']))
        near(array(g['rewards'])[idx],array(add['adjusted_reward']));near(array(g['episode_starts'])[idx],array(add['episode_start']))
        expected_start=n==1 or amap[n-1]['terminated'] or amap[n-1]['truncated'];near(array(add['episode_start']),expected_start)
        byrow[roll,idx]=add
    for roll,g in gaei.items():near(array(g['dones']),amap[100*roll]['terminated'] or amap[100*roll]['truncated'])
    goal_uses=[];draws=[];all_adv=[];all_returns=[];losses=0
    for r in rows:
        if r['event']=='ppo_permutation':
            key=(r['rollout_id'],r['epoch']);require(key not in perms,'Duplicate PPO epoch');perms[key]=array(r['indices']).tolist()
        elif r['event']=='rollout_sample':
            sample=r;key=(r['rollout_id'],r['epoch']);start=(r['minibatch_id']-1)*50
            require(key in perms and array(r['indices']).tolist()==perms[key][start:start+50],'Actual PPO permutation slice')
            require(len(r['sample'])==50 and [s['row'] for s in r['sample']]==array(r['indices']).tolist(),'PPO row/sample order')
        elif r['event']=='ppo_loss':
            losses+=1;require(sample is not None and sample['sample']==r['sample'] and not r['early_stop'] and r['optimizer_step']==losses,'PPO actual update/row join')
            roll=r['rollout_id'];g=gaer[roll]
            for i,s in enumerate(r['sample']):
                idx=s['row'];add=byrow[roll,idx];n=add['transition_id'];require(n==s['transition_id'],'PPO exact consumed transition')
                near(array(r['raw_advantages'])[i],array(g['advantages'])[idx]);near(array(r['returns'])[i],array(g['returns'])[idx]);near(array(r['old_log_probs'])[i],array(add['log_probs']))
                draws.append(n)
                if amap[n]['outcome']=='success':goal_uses.append({'transition_id':n,'rollout_id':roll,'row':idx,'epoch':r['epoch'],
                    'minibatch_id':r['minibatch_id'],'update_id':losses,'trace_sequence':r['sequence'],'case_id':amap[n]['case_id'],
                    'raw_reward':amap[n]['reward'],'raw_advantage':float(array(r['raw_advantages'])[i]),'normalized_advantage':float(array(r['normalized_advantages'])[i]),
                    'return':float(array(r['returns'])[i]),'collection_action_probability':float(array(behavior[n]['probabilities'])[0,amap[n]['action']]),
                    'actual_update_action_probability':float(np.exp(array(r['new_log_probs'])[i]))})
            sample=None
    require(len(perms)==25 and losses==50 and set(perms)=={(r,e) for r in range(1,6) for e in range(5)},'Complete PPO epochs/updates')
    require(Counter(draws)==Counter({n:5 for n in range(1,501)}),'PPO exact repeated exposure')
    for g in gaer.values():all_adv.extend(array(g['advantages']).ravel());all_returns.extend(array(g['returns']).ravel())
    c=receipt['profile_counts']
    for k,v in {'actor_forwards':500,'critic_forwards':5+len(timeouts),'policy_training_evaluations':50,'rollout_consumptions':50,'rollout_permutations':25,'native_optimizer_steps':50}.items():require(c.get(k,0)==v,'Independent PPO call witness '+k)
    require(receipt['library_update_counter']==25,'PPO epoch counter')
    goals=sum(a['outcome']=='success' for a in actions)
    return {'recorded_goal_experiences':goals,'goal_sample_draws':len(goal_uses),'distinct_sampled_goals':len(set(g['transition_id'] for g in goal_uses)),
        'all_sample_draws':len(draws),'distinct_sampled_experiences':len(set(draws)),'advantages':stats(all_adv),'returns':stats(all_returns),
        'timeout_bootstraps':len(timeouts),'rollouts':5,'epochs':25,'Q_values':'not_applicable',
        'reward_to_update_example':goal_uses[0] if goal_uses else {'status':'not_observed','recorded_goals':goals,'goal_draws':0},
        'profile_counts':c,'required_missing_fields':[]}


def verify_records(folder,manifest):
    journal=read_journal(folder/'training-ledger.jsonl');require(journal[-1]['event']=='pilot_finished','Pilot journal closure')
    require([r['algorithm'] for r in events(journal,'arm_started')]==list(ALGORITHMS),'One-shot arm order')
    require([r['total'] for r in events(journal,'action_reserved')]==list(range(1,1001)),'Aggregate action ledger')
    results={}
    for algorithm in ALGORITHMS:
        rows,actions,receipt,summary=common(folder,manifest,algorithm,journal)
        summary['signals']=dqn(rows,actions,receipt,folder) if algorithm=='dqn' else ppo(rows,actions,receipt)
        results[algorithm]=summary
        del rows,actions,receipt
    run=read_json(folder/'run.json');require(run['wall_seconds']<1800 and run['scientific_training_actions']==1000,'Combined runtime/actions')
    return {'status':'passed','arms':results,'scientific_training_actions':1000,'actual_optimizer_steps':150,
        'research_evaluation_actions':0,'new_synthetic_transitions':0,'new_synthetic_optimizer_steps':0,
        'live_bitwise_on_off_equivalence':'not_observed: matched extra research run not authorized',
        'inherited_evidence':'Phase 5E synthetic bitwise equivalence only','required_missing_fields':[]}
