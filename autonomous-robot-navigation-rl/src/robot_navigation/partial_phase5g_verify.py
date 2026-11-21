"""Phase 5G independent closed-record lineage and arithmetic; no learner calls.

The G checks extend the reviewed F verifier to the registered 2000-action envelope.
Observer/transport and numerical routines are imported unchanged from E/F.
"""
from collections import Counter
import hashlib
import numpy as np
from .core import read_json
from .release import sha
from .partial_instrumentation_trace import read_trace,array,checksum
from .partial_instrumentation_audit import audit_records,near
from .partial_phase5f_verify import require,tensor_hash,stats,events,chain
from .partial_phase5g_io import ALGORITHMS,CHECKPOINTS,read_journal,cases,direction,UPDATES

def common(folder,manifest,algorithm,journal):
    rows=read_trace(folder/algorithm/'trace');last_hash=chain(rows,manifest,algorithm)
    receipt=read_json(folder/algorithm/'run-receipt.json')
    accepted=events([j for j in journal if j.get('algorithm')==algorithm],'action_accepted')
    reserved=events([j for j in journal if j.get('algorithm')==algorithm],'action_reserved')
    env=events(rows,'environment_transition');resets=events(rows,'reset');jr=events([j for j in journal if j.get('algorithm')==algorithm],'reset')
    require(len(accepted)==len(reserved)==len(env)==2000,'Exact accepted/reserved/environment count')
    require([r['transition_id'] for r in accepted]==list(range(1,2001)) and [r['transition_id'] for r in reserved]==list(range(1,2001)),'Action IDs')
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
    optim=events(rows,'optimizer');expected_updates={'dqn':475,'ppo':200}[algorithm]
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
    require(receipt['observer_hooks_restored'] and receipt['training_actions']==2000 and receipt['optimizer_steps']==expected_updates,'Final receipt')
    require(receipt['wall_seconds']<1800 and receipt['research_assessment_actions_during_training']==receipt['final_unseen_evaluations']==0,'Arm scope/time')
    require(rows[-1]['interactions']==2000 and rows[-1]['optimizer_steps']==expected_updates,'Observer completion')
    meta=read_json(folder/algorithm/'checkpoints/2000/policy.json')
    require(meta['checkpoint_sha256']==sha(folder/algorithm/'checkpoints/2000/policy.zip') and meta['manifest_sha256']==manifest['manifest_sha256'],'Checkpoint provenance bytes')
    math=audit_records(rows)
    return rows,accepted,receipt,{'training_actions':2000,'optimizer_steps':len(optim),'terminal_outcomes':dict(counts),
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
    require(len(inserted)==2000 and sample_count==loss_count==475,'DQN signal coverage')
    require([r['training_actions'] for r in events(rows,'target_sync')]==list(range(250,2001,250)),'Target sync schedule')
    for a in actions:
        n=a['transition_id'];expected=0. if n==1 else max(.1,1.-.9*((n-1)/2000)/.8)
        near(a['selection_context']['epsilon'],expected,1e-10)
        require(a['selection_context']['mode'] in ('warmup_action_space_sample','policy_predict_branch','epsilon_action_space_sample'),'Actual DQN action branch')
    predictions=sum(a['selection_context']['mode']=='policy_predict_branch' for a in actions)
    c=receipt['profile_counts']
    for k,v in {'native_optimizer_steps':475,'replay_sample_calls':475,'replay_consumptions':475,'selected_actions':2000,'target_forwards':475,'online_forwards':475+predictions}.items():
        require(c.get(k,0)==v,'Independent DQN call witness '+k)
    with np.load(folder/'dqn/checkpoints/2000/buffer.npz',allow_pickle=False) as b:
        for i,a in enumerate(actions):
            require(tensor_hash(b['observations'][i,0])==a['observation_sha256'] and tensor_hash(b['next_observations'][i,0])==a['next_observation_sha256'],'Final replay prefix observation')
            near(b['actions'][i,0],a['action']);near(b['rewards'][i,0],a['reward']);near(b['dones'][i,0],a['terminated'] or a['truncated']);near(b['timeouts'][i,0],a['truncated'])
    require(receipt['library_update_counter']==475,'DQN library counter')
    goals=[a['transition_id'] for a in actions if a['outcome']=='success'];frequencies=Counter(g['transition_id'] for g in goal_rows)
    return {'recorded_goal_experiences':len(goals),'goal_sample_draws':len(goal_rows),'distinct_sampled_goals':len(frequencies),
        'goal_draw_frequencies':dict(frequencies),'all_sample_draws':len(draws),'distinct_sampled_experiences':len(set(draws)),
        'actual_target':stats(targets),'residual_snapshot_reconstructed':stats(residuals),'target_syncs':8,
        'reward_to_update_example':goal_rows[0] if goal_rows else {'status':'not_observed','recorded_goals':len(goals),'goal_draws':0},
        'profile_counts':c,'required_missing_fields':[]}

def ppo(rows,actions,receipt):
    amap={a['transition_id']:a for a in actions};behavior={r['transition_id']:r for r in events(rows,'ppo_behavior')}
    additions=events(rows,'rollout_add');gaei={r['rollout_id']:r for r in events(rows,'gae_input')};gaer={r['rollout_id']:r for r in events(rows,'gae_result')}
    timeouts={r['transition_id']:r for r in events(rows,'ppo_timeout_bootstrap')};byrow={};perms={};sample=None
    require(len(behavior)==len(additions)==2000 and len(gaei)==len(gaer)==20,'PPO collector/GAE coverage')
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
    require(len(perms)==100 and losses==200 and set(perms)=={(r,e) for r in range(1,21) for e in range(5)},'Complete PPO epochs/updates')
    require(Counter(draws)==Counter({n:5 for n in range(1,2001)}),'PPO exact repeated exposure')
    for g in gaer.values():all_adv.extend(array(g['advantages']).ravel());all_returns.extend(array(g['returns']).ravel())
    c=receipt['profile_counts']
    for k,v in {'actor_forwards':2000,'critic_forwards':20+len(timeouts),'policy_training_evaluations':200,'rollout_consumptions':200,'rollout_permutations':100,'native_optimizer_steps':200}.items():require(c.get(k,0)==v,'Independent PPO call witness '+k)
    require(receipt['library_update_counter']==100,'PPO epoch counter')
    goals=sum(a['outcome']=='success' for a in actions)
    return {'recorded_goal_experiences':goals,'goal_sample_draws':len(goal_uses),'distinct_sampled_goals':len(set(g['transition_id'] for g in goal_uses)),
        'all_sample_draws':len(draws),'distinct_sampled_experiences':len(set(draws)),'advantages':stats(all_adv),'returns':stats(all_returns),
        'timeout_bootstraps':len(timeouts),'rollouts':20,'epochs':100,'Q_values':'not_applicable',
        'reward_to_update_example':goal_uses[0] if goal_uses else {'status':'not_observed','recorded_goals':goals,'goal_draws':0},
        'profile_counts':c,'required_missing_fields':[]}


def checkpoint_audit(folder,algorithm,rows,actions,run,journal):
    entries=events([r for r in journal if r.get('algorithm')==algorithm],'checkpoint_saved')
    require([r['training_actions'] for r in entries]==list(CHECKPOINTS),'All exact checkpoints in order')
    optim=events(rows,'optimizer');adds=events(rows,'rollout_add');gaes={r['rollout_id']:r for r in events(rows,'gae_result')}
    out={}
    for cp,j in zip(CHECKPOINTS,entries):
        target=folder/algorithm/'checkpoints'/f'{cp:04d}';r=read_json(target/'receipt.json');meta=read_json(target/'policy.json')
        expected_steps=max(0,(cp-100)//4) if algorithm=='dqn' else cp//10
        require(r['training_actions']==r['model_num_timesteps']==cp and r['optimizer_steps']==expected_steps,'Exact post-update checkpoint count')
        require(r['total_timesteps']==(0 if cp==0 else 2000),'One learn horizon, no split learn calls')
        require(r['library_updates']==(expected_steps if algorithm=='dqn' else cp//20),'Checkpoint library counter')
        require(r['rng_before']==r['rng_after'] and r['state_unchanged'],'Save RNG/state preserved')
        require(r['checkpoint_sha256']==meta['checkpoint_sha256']==j['checkpoint_sha256']==sha(target/'policy.zip'),'Checkpoint byte hash')
        require(r['buffer_file_sha256']==sha(target/'buffer.npz'),'Checkpoint buffer bytes')
        expected=run['network_before'] if cp==0 else [o for o in optim if o['training_actions']==cp][-1]['network_after']
        require(r['network']==j['network']==expected and r['optimizer_sha256']==j['optimizer_sha256'],'Checkpoint version join')
        require(r['optimizer_sha256']==(optim[0]['optimizer_before'] if cp==0 else [o for o in optim if o['training_actions']==cp][-1]['optimizer_after']),'Checkpoint optimizer lineage')
        require(checksum(r['rng_before'])==j['rng_sha256'],'Saved RNG chain')
        require(meta['feature_version']=='partial-visible-relative-scale4-v1' and meta['observation_values']==409 and meta['feature_count']==412 and meta['actions']==5,'Checkpoint feature identity')
        if cp==0:require(r['rng_before']==run['rng_before'],'Initial RNG provenance')
        with np.load(target/'buffer.npz',allow_pickle=False) as b:
            for key in b.files:require(np.isfinite(b[key]).all(),'Finite checkpoint buffer')
            if algorithm=='dqn':
                require(b['observations'].shape==(cp,1,409) and r['buffer_pos']==cp and not r['buffer_full'],'Replay checkpoint prefix')
                for i,a in enumerate(actions[:cp]):
                    require(tensor_hash(b['observations'][i,0])==a['observation_sha256'] and tensor_hash(b['next_observations'][i,0])==a['next_observation_sha256'],'Snapshot replay/action identity')
                    near(b['actions'][i,0],a['action']);near(b['rewards'][i,0],a['reward']);near(b['dones'][i,0],a['terminated'] or a['truncated']);near(b['timeouts'][i,0],a['truncated'])
            elif cp:
                roll=cp//100;part=[a for a in adds if a['rollout_id']==roll]
                require(len(part)==100 and r['buffer_pos']==100 and r['buffer_full'],'Rollout checkpoint full')
                for idx,a in enumerate(part):
                    require(tensor_hash(b['observations'][idx])==tensor_hash(array(a['observation'])),'Rollout checkpoint observation')
                    for name,field in [('actions','action'),('values','values'),('log_probs','log_probs'),('episode_starts','episode_start'),('rewards','adjusted_reward')]:near(b[name][idx],array(a[field]))
                near(b['advantages'].reshape(-1),array(gaes[roll]['advantages']).reshape(-1));near(b['returns'].reshape(-1),array(gaes[roll]['returns']).reshape(-1))
        out[str(cp)]={'checkpoint_sha256':r['checkpoint_sha256'],'optimizer_steps':expected_steps,'network':r['network'],'rng_save_unchanged':True}
    return out


def reward_coverage(rows,actions,training_cases,algorithm):
    amap={a['transition_id']:a for a in actions};directions={c['id']:direction(c) for c in training_cases}
    ops={o['update_id']:o['training_actions'] for o in events(rows,'optimizer')};uses=[]
    for r in events(rows,'dqn_loss' if algorithm=='dqn' else 'ppo_loss'):
        update=r['update_id'] if algorithm=='dqn' else r['optimizer_step'];at=ops[update]
        for idx,s in enumerate(r['sample']):
            a=amap[s['transition_id']]
            if a['outcome']!='success':continue
            sample={'transition_id':a['transition_id'],'case_id':a['case_id'],'direction':directions[a['case_id']],
                'experienced_at':a['transition_id'],'used_at':at,'update_id':update,'sample_row':idx,'trace_sequence':r['sequence'],'raw_goal_reward':a['reward']}
            if algorithm=='dqn':sample.update(actual_target=float(array(r['actual_target'])[idx].item()),chosen_Q=float(array(r['current_chosen_Q'])[idx].item()),residual_snapshot_reconstructed=float(array(r['td_residual'])[idx].item()))
            else:sample.update(rollout_id=r['rollout_id'],row=s['row'],epoch=r['epoch'],minibatch_id=r['minibatch_id'],
                raw_advantage=float(array(r['raw_advantages'])[idx]),normalized_advantage=float(array(r['normalized_advantages'])[idx]),
                return_value=float(array(r['returns'])[idx]),action_probability_at_update=float(np.exp(array(r['new_log_probs'])[idx])))
            uses.append(sample)
    def summarize(lo,hi):
        encountered=[a for a in actions if lo<a['transition_id']<=hi and a['outcome']=='success']
        draws=[u for u in uses if lo<u['used_at']<=hi]
        result={'training_action_interval':[lo,hi],'goal_experiences':len(encountered),'goal_sample_draws':len(draws),
            'distinct_goals_sampled':len({u['transition_id'] for u in draws}),'actual_updates_using_goals':len({u['update_id'] for u in draws}),'by_direction':{}}
        for d in ('Up','Down','Left','Right'):
            ee=[a for a in encountered if directions[a['case_id']]==d];dd=[u for u in draws if u['direction']==d]
            result['by_direction'][d]={'goal_experiences':len(ee),'goal_sample_draws':len(dd),'distinct_goals_sampled':len({u['transition_id'] for u in dd}),
                'actual_updates_using_goals':len({u['update_id'] for u in dd}),'earliest_link':dd[0] if dd else {'status':'not_observed','goal_experiences':len(ee)},
                'actual_target' if algorithm=='dqn' else 'raw_advantage':stats([u['actual_target' if algorithm=='dqn' else 'raw_advantage'] for u in dd])}
        return result
    return {'cumulative':{str(cp):summarize(0,cp) for cp in CHECKPOINTS},'intervals':[summarize(lo,hi) for lo,hi in zip(CHECKPOINTS,CHECKPOINTS[1:])],
        'all_goal_use_links':uses}


def evaluate_records(folder,manifest):
    from .partial_phase5g_assess import first_metrics,reducing_actions
    from .partial_contract import MOVES
    panels=cases();ordered=panels['original_cases']+panels['expanded_cases'];all_global=[];result={}
    for algorithm in ALGORITHMS:
        output={};ledger=read_journal(folder/algorithm/'development/assessment-ledger.jsonl')
        require(ledger[-1]['event']=='assessment_closed','Assessment ledger closed')
        reserved=events(ledger,'assessment_reserved');accepted=events(ledger,'assessment_accepted');starts=events(ledger,'case_started');ends=events(ledger,'case_finished')
        require(len(reserved)==len(accepted) and len(starts)==len(ends)==80,'Assessment operation/case counts')
        for a,z in zip(accepted,reserved):
            require(a['assessment_action_id']==z['assessment_action_id'] and z['sequence']<a['sequence'],'Assessment reservation before execution')
            for k in ('algorithm','checkpoint','case_id'):require(a[k]==z[k],'Assessment reservation identity')
        all_global.extend(a['assessment_action_id'] for a in accepted);cursor=0
        for cp in CHECKPOINTS:
            node=read_json(folder/algorithm/'development'/f'{cp:04d}.json');rows=node['cases']
            require(len(rows)==20 and [r['case_id'] for r in rows]==[c['id'] for c in ordered],'Identical 4+16 ordered panels')
            require(node['model_optimizer_before']==node['model_optimizer_after'] and node['global_rng_before']==node['global_rng_after'],'Pure child inference/RNG')
            checkpoint=read_json(folder/algorithm/'checkpoints'/f'{cp:04d}'/'receipt.json')
            require(node['checkpoint_sha256']==checkpoint['checkpoint_sha256'],'Assessed exact snapshot')
            assessments=[]
            for ix,(r,c) in enumerate(zip(rows,ordered)):
                initial=array(r['initial_observation']);require(initial.shape==(409,),'Public observation shape')
                near(initial[:4],np.asarray(c['start']+c['goal'])/9,1e-7)
                require(r['panel']==('original' if ix<4 else 'expanded') and r['direction']==direction(c),'Assessment panel/direction')
                require(r['first_action']==r['trajectory'][0]['action'] and r['first_metrics']==first_metrics(c,r['first_action'],r['first_scores'],algorithm),'First-action diagnostic arithmetic')
                require(r['checkpoint_sha256']==checkpoint['checkpoint_sha256'] and r['inference_only'],'Case checkpoint provenance')
                pos=list(c['start']);path=[pos];prev=tensor_hash(initial);wait=0;revisits=0;seen={tuple(pos)};invalid=0
                for step,s in enumerate(r['trajectory'],1):
                    require(1<=step<=200 and s['step']==step and s['position']==pos and s['observation_sha256']==prev,'Assessment path/hash continuity')
                    require(isinstance(s['action'],int) and s['action'] in range(5),'Assessment action contract')
                    dr,dc=MOVES[s['action']];dest=[pos[0]+dr,pos[1]+dc];collision=not all(0<=v<10 for v in dest)
                    nxt=pos if collision else dest;success=not collision and nxt==c['goal'];timeout=not collision and not success and step==200
                    outcome='collision' if collision else 'success' if success else 'timeout' if timeout else 'running'
                    require(s['next_position']==nxt and s['outcome']==outcome and s['terminated']==(collision or success) and s['truncated']==timeout,'Open-grid actual termination/path')
                    near(s['reward'],9.99 if success else -5.01 if collision else -.01,1e-10)
                    require(step==len(r['trajectory']) or outcome=='running','No action after terminal')
                    actual=accepted[cursor];require(all(actual[k]==v for k,v in s.items()),'Trajectory/assessment ledger exact join');cursor+=1
                    require(actual['case_id']==c['id'] and actual['checkpoint']==cp,'Case/action join')
                    if nxt!=pos:revisits+=tuple(nxt) in seen;seen.add(tuple(nxt))
                    wait+=s['action']==4;invalid+=collision;pos=list(nxt);path.append(pos);prev=s['next_observation_sha256']
                require(path==r['path'] and r['steps']==len(r['trajectory']) and r['outcome']==outcome!='running','Complete case path')
                oscillations=sum(path[i]==path[i+2] and path[i+1]==path[i+3] and path[i]!=path[i+1] for i in range(len(path)-3))
                distance=sum(abs(a-b) for a,b in zip(c['start'],c['goal']))
                assessments.append({'case_id':c['id'],'panel':r['panel'],'direction':direction(c),'steps':r['steps'],'success':outcome=='success','outcome':outcome,
                    'first_action':r['first_action'],**r['first_metrics'],'wait_count':wait,'moving_revisits':int(revisits),'oscillation_windows':int(oscillations),
                    'invalid_moves':invalid,'shortest_reference':distance,'success_step_inefficiency':r['steps']/distance if outcome=='success' else 'not_applicable'})
            def panel_stats(part):return {'successes':sum(r['success'] for r in part),'cases':len(part),'first_aligned':sum(r['aligned'] for r in part),'alignment_denominator':len(part)}
            totalsteps=sum(r['steps'] for r in assessments);right=[r for r in assessments if r['direction']=='Right']
            output[str(cp)]={'original':panel_stats(assessments[:4]),'expanded':panel_stats(assessments[4:]),'combined':panel_stats(assessments),
                'expanded_by_direction':{d:panel_stats([r for r in assessments[4:] if r['direction']==d]) for d in ('Up','Down','Left','Right')},
                'combined_by_direction':{d:panel_stats([r for r in assessments if r['direction']==d]) for d in ('Up','Down','Left','Right')},
                'assessment_actions':totalsteps,'wait_count':sum(r['wait_count'] for r in assessments),'wait_rate':sum(r['wait_count'] for r in assessments)/totalsteps,
                'collisions':sum(r['outcome']=='collision' for r in assessments),'timeouts':sum(r['outcome']=='timeout' for r in assessments),
                'moving_revisits':sum(r['moving_revisits'] for r in assessments),'oscillation_windows':sum(r['oscillation_windows'] for r in assessments),
                'first_wait':sum(r['wait'] for r in assessments),'first_invalid':sum(r['invalid_move'] for r in assessments),'first_ties':sum(r['tie'] for r in assessments),
                'mean_goal_directed_margin':float(np.mean([r['goal_directed_margin'] for r in assessments])),
                'first_right_margin':float(np.mean([r['right_margin'] for r in right])),
                'right_margin_denominator':len(right),'first_right_log_odds':'not_applicable' if algorithm=='dqn' else float(np.mean([r['right_log_odds'] for r in right])),
                'case_metrics':assessments,'checkpoint_sha256':checkpoint['checkpoint_sha256']}
        require(cursor==len(accepted),'No omitted/extra assessment actions')
        receipt=read_json(folder/algorithm/'development/receipt.json')
        require(receipt['evaluation_actions']==len(accepted)==receipt['native_calls']['environment_step_calls'] and receipt['native_calls']['optimizer_calls']==0,'Independent assessment call counts')
        final=output['2000'];gain=final['expanded']['successes']-output['0']['expanded']['successes']
        checks={'expanded_gain_at_least_4':gain>=4,'expanded_at_least_12':final['expanded']['successes']>=12,
            'each_direction_at_least_2':all(x['successes']>=2 for x in final['expanded_by_direction'].values())}
        result[algorithm]={'checkpoints':output,'screen':{'verdict':'BASIC_NAVIGATION_SCREEN_MET' if all(checks.values()) else 'BASIC_NAVIGATION_SCREEN_NOT_MET',
            'checks':checks,'expanded_gain':gain},'evaluation_receipt':receipt}
    require(all_global==list(range(1,len(all_global)+1)) and len(all_global)<=32000,'Aggregate assessment limit/order')
    isolation=read_json(folder/'assessment-parent-isolation.json');require(isolation['before']==isolation['after'],'Parent RNG/training evidence unchanged by children')
    return {'arms':result,'assessment_actions':len(all_global),'assessment_cases':160,'training_actions':0,'optimizer_steps':0,'isolation':'passed'}


def verify_records(folder,manifest):
    journal=read_journal(folder/'training-ledger.jsonl');require(journal[-1]['event']=='training_finished','Training completion')
    require([r['algorithm'] for r in events(journal,'arm_started')]==list(ALGORITHMS),'One DQN then PPO')
    require([r['total'] for r in events(journal,'action_reserved')]==list(range(1,4001)),'Total training 4000')
    training=cases()['training_cases'];arms={}
    for algorithm in ALGORITHMS:
        rows,actions,receipt,summary=common(folder,manifest,algorithm,journal)
        require(all(r['run']['study_id']=='phase5g-longer-horizon-v1' for r in rows),'G run metadata')
        summary['signals']=dqn(rows,actions,receipt,folder) if algorithm=='dqn' else ppo(rows,actions,receipt)
        summary['checkpoints']=checkpoint_audit(folder,algorithm,rows,actions,receipt,journal)
        summary['goal_coverage']=reward_coverage(rows,actions,training,algorithm);arms[algorithm]=summary
        del rows,actions,receipt
    run=read_json(folder/'training.json');require(run['scientific_training_actions']==4000 and run['wall_seconds']<3600,'Combined training wall/actions')
    assessment=evaluate_records(folder,manifest)
    return {'verdict':'PHASE5G_MEASUREMENT_VALIDATED','gates':{k:'passed' for k in ('G0','G1','G2','G3','G4','G5')},'training':arms,
        'assessment':assessment,'scientific_training_actions':4000,'optimizer_steps':675,'assessment_actions':assessment['assessment_actions'],
        'new_synthetic_transitions':0,'new_synthetic_optimizer_steps':0,'required_missing_fields':[],
        'live_bitwise_on_off_equivalence':'not_observed; no matched uninstrumented research run authorized',
        'F_comparison':'Separate fresh run, different DQN horizon; never a combined trajectory'}
