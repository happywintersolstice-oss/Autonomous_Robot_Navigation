"""Predeclared neural screening, fresh paired confirmations and read-only audits."""
import csv
import gzip
import json

import numpy as np

from .agents import load_policy, viewer_checkpoint
from .benchmark import load_scenarios
from .core import load_task, read_json
from .dashboard import training_episode, training_paths
from .episodes import aggregate, validate_replay
from .release import sha
from .workflow import source_hash

STUDY = 'neural-tuning-v1'
METRICS = ('success_rate','collision_rate','timeout_rate','spl','mean_return','mean_successful_route_length')


def validate_plan(plan):
    if (plan['algorithms'] != ['dqn','ppo'] or plan['workers'] != 2
            or plan['search_interactions'] != 100000 or plan['confirmation_interactions'] != 200000
            or plan['pilot_interactions'] != 10000 or plan['monitor_every'] != 10000):
        raise ValueError('Neural study requires its declared algorithms, budgets and worker bound')
    seeds = [plan['pilot_seed'],plan['development_seed'],*plan['confirmation_seeds']]
    if len(plan['confirmation_seeds']) != 3 or len(set(seeds)) != 5 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Pilot, development and three confirmation seeds must be distinct')
    allowed={'dqn':{'learning_rate','discount','target_update_interval','epsilon_end','epsilon_decay_fraction'},
             'ppo':{'learning_rate','discount','n_steps','ent_coef','clip_range'}}
    for family in plan['algorithms']:
        candidates=plan['candidates'][family]
        if list(candidates) != ['c00','c01','c02','c03'] or candidates['c00'] != {}:
            raise ValueError('Four candidates must include the unchanged c00 baseline')
        if any(set(parameters)-allowed[family] for parameters in candidates.values()):
            raise ValueError('Candidate changes a factor outside the declared tuning scope')
    if (plan['limits']['max_runs_including_pilots'] != 22
            or plan['limits']['max_training_interactions_including_pilots'] != 3220000):
        raise ValueError('Neural study count/interaction limits do not reconcile')


def run_config(plan, family, stage, seed, candidate='c00', root=None):
    validate_plan(plan)
    if stage not in ('pilot','search','control','tuned') or family not in plan['algorithms']:
        raise ValueError('Unknown neural study stage or algorithm')
    expected_seeds = [plan['pilot_seed']] if stage=='pilot' else [plan['development_seed']] if stage=='search' else plan['confirmation_seeds']
    if seed not in expected_seeds or (stage in ('pilot','control') and candidate!='c00'):
        raise ValueError('Run differs from the predeclared seed or baseline condition')
    baseline=read_json(root/'artifacts/runs'/plan['baseline_runs'][family]/'configuration.json')
    load_task(root).check(baseline['signature'])
    config=baseline['training'].copy()
    if config['algorithm'] != family:
        raise ValueError('Baseline family mismatch')
    config.update(plan['candidates'][family][candidate])
    budget=plan['pilot_interactions'] if stage=='pilot' else plan['search_interactions'] if stage=='search' else plan['confirmation_interactions']
    identity=f"{plan['run_prefix']}-{stage}-{family}"+(f'-{candidate}' if stage=='search' else '')+f'-seed{seed}'
    config.update(run_id=identity,stage=stage,seed=seed,interactions=budget,monitor_every=plan['monitor_every'],
                  training_trace_episodes=10,training_trace_starts=[0,budget//4,budget//2,3*budget//4,19*budget//20],
                  max_wall_seconds=plan['limits']['run_wall_seconds'],log_progress=True)
    return config


def choose_candidate(rows, family, plan):
    selected=[r for r in rows if r['stage']=='search' and r['algorithm']==family]
    if (len(selected)!=4 or {r['candidate'] for r in selected}!=set(plan['candidates'][family])
            or any(r['seed']!=plan['development_seed'] for r in selected)):
        raise ValueError('Candidate selection requires the complete declared search')
    return min(selected,key=lambda r:(-r['validation']['success_rate'],-r['validation']['spl'],
                                      r['validation']['collision_rate'],r['candidate']))['candidate']


def stats(values):
    values=[v for v in values if v is not None]
    return {'mean':float(np.mean(values)) if values else None,
            'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'contributing_seeds':len(values)}


def summarize(rows, selected, plan, signature):
    systems={}
    for family in plan['algorithms']:
        groups={stage:sorted([r for r in rows if r['algorithm']==family and r['stage']==stage],key=lambda r:r['seed'])
                for stage in ('control','tuned')}
        if selected[family]=='c00':
            if groups['tuned']:
                raise ValueError('A retained baseline must not create duplicate confirmation replicates')
            groups['tuned']=groups['control']
        for stage,group in groups.items():
            if [r['seed'] for r in group]!=plan['confirmation_seeds']:
                raise ValueError('Confirmation matrix is incomplete or not seed-matched')
            if any(r['pilot']['training_interactions']!=200000 or r['validation']['episodes']!=928 for r in group):
                raise ValueError('Confirmation budgets or validation support differ')
        paired=[{'seed':before['seed'],**{m:after['validation'][m]-before['validation'][m]
                    for m in METRICS if m!='mean_successful_route_length'}}
                for before,after in zip(groups['control'],groups['tuned'],strict=True)]
        systems[family]={'selected_candidate':selected[family],'baseline_retained':selected[family]=='c00',
                        'seed_count':3,'seeds':plan['confirmation_seeds'],
                        'conditions':{stage:{'metrics':{m:stats([r['validation'][m] for r in group]) for m in METRICS},
                           'costs':{m:stats([r['costs'][m] for r in group]) for m in group[0]['costs']},
                           'seed_results':group} for stage,group in groups.items()},
                        'paired_seed_differences':paired,
                        'paired_change':{m:stats([r[m] for r in paired]) for m in paired[0] if m!='seed'},
                        'wins_losses_ties':{key:sum((r['success_rate']>0 if key=='wins' else r['success_rate']<0 if key=='losses' else r['success_rate']==0) for r in paired)
                                            for key in ('wins','losses','ties')}}
    return {'status':'verified','signature':signature,'systems':systems,
            'scope':'preliminary three-seed matched confirmation on development validation; one-seed screening',
            'validation_scenarios_per_seed':928,'final_test_evaluated':False,
            'original_final_selection_changed':False,'aggregation_unit':'independent paired training seed; sample SD, not a confidence interval'}


def audit_run(row, expected_config, frozen_source, root):
    task=load_task(root);directory=root/'artifacts/runs'/row['run_id']
    manifest=read_json(directory/'manifest.json');config=read_json(directory/'configuration.json')
    task.check(manifest['signature'])
    if (manifest['status']!='complete' or manifest['final_test_evaluated'] or config['training']!=expected_config
            or config['environment']!=task.config or manifest['source_hash']!=frozen_source
            or source_hash(directory/'source')!=frozen_source):
        raise ValueError('Run configuration, completion or source identity differs from the frozen matrix')
    budget=expected_config['interactions'];family=row['algorithm'];key=family.replace('-','_')
    if (manifest['training_seed']!=row['seed'] or manifest['training_interactions']!=budget
            or manifest['algorithm']!=family or manifest['stage']!=row['stage']):
        raise ValueError('Training seed or interaction count mismatch')
    scenarios=load_scenarios(task,'validation',root)
    metrics=read_json(directory/f'{key}_episodes.json')
    if [m['scenario_id'] for m in metrics]!=[s['id'] for s in scenarios]:
        raise ValueError('Validation scenario coverage mismatch')
    measured=aggregate(metrics)
    if any(row['validation'][name]!=value for name,value in measured.items()):
        raise ValueError('Validation aggregate does not reconcile')
    with (directory/'training.csv').open(newline='',encoding='utf-8') as file:
        ledger=list(csv.DictReader(file))
    pilot=read_json(directory/'pilot.json');partial=pilot['incomplete_budget_cutoff_episode']
    if sum(int(r['steps']) for r in ledger)+(partial['steps'] if partial else 0)!=budget:
        raise ValueError('Training ledger budget mismatch')
    with np.load(directory/'visitation.npz',allow_pickle=False) as data:
        if int(data['state_action_counts'].sum())!=budget:
            raise ValueError('Visitation budget mismatch')
    monitor=read_json(directory/'monitoring.json')
    if [m['training_interactions'] for m in monitor]!=list(range(10000,budget+1,10000)):
        raise ValueError('Monitoring checkpoint schedule differs')
    best=max(monitor,key=lambda m:(m['success_rate'],m['spl'],-m['training_interactions']))
    if best['training_interactions']!=manifest['selected_interactions'] or best['training_interactions']!=row['selected_interactions']:
        raise ValueError('Checkpoint selection rule mismatch')
    policy=load_policy(viewer_checkpoint(directory),task)
    if policy.model.observation_space.shape!=(4,):
        raise ValueError('Neural tuning must retain the original four-coordinate input')
    fingerprint=policy.state_fingerprint();cache={};actions=0
    with gzip.open(directory/f'{key}_trajectories.jsonl.gz','rt',encoding='utf-8') as file:
        for scenario,line,metric in zip(scenarios,file,metrics,strict=True):
            episode=json.loads(line)
            if episode['scenario']!=scenario or episode['mode']!='validation':
                raise ValueError('Validation replay identity mismatch')
            validate_replay(task,episode)
            if episode['metrics']!={k:v for k,v in metric.items() if k!='scenario_id'}:
                raise ValueError('Validation trajectory metrics mismatch')
            for transition in episode['transitions']:
                state=tuple(transition['observation'])
                if state not in cache:
                    if family=='dqn':
                        action,scores=policy.action_and_scores(state);cache[state]=(action,scores.tolist())
                    else:
                        cache[state]=policy.decision(state)
                if family=='dqn':
                    if (transition['action'],transition['scores'])!=cache[state]:
                        raise ValueError('DQN action or Q score does not match the saved checkpoint')
                elif transition.get('decision')!=cache[state] or transition['action']!=cache[state]['action'] or transition['scores'] is not None:
                    raise ValueError('PPO action or probability does not match the saved checkpoint')
                actions+=1
    if fingerprint!=policy.state_fingerprint():
        raise ValueError('Inference audit changed the learned model')
    checkpoints=sorted((directory/'checkpoints').glob('*.zip'))
    if len(checkpoints)!=budget//10000:
        raise ValueError('Missing periodic checkpoints')
    for checkpoint in checkpoints:
        metadata=read_json(checkpoint.with_suffix('.json'))
        if sha(checkpoint)!=metadata['sha256'] or metadata['training_interactions']!=int(checkpoint.stem.split('-')[-1]):
            raise ValueError('Periodic checkpoint checksum or step mismatch')
    training_rows=[dict(r) for r in ledger]
    if partial and partial['steps']:
        training_rows.append({'end_interaction':budget,'steps':partial['steps'],'scenario_id':partial['scenario_id'],
                              'return':partial['return'],'outcome':'budget_cutoff'})
    expected=[];starts=expected_config['training_trace_starts']
    for window,start in enumerate(starts):
        stop=starts[window+1] if window+1<len(starts) else budget
        candidates=[r for r in training_rows if start<=int(r['end_interaction'])-int(r['steps'])<stop]
        expected.extend((window,r) for r in candidates[:2])
    paths=training_paths(directory)
    if len(paths)!=len(expected) or len(paths)>10:
        raise ValueError('Recorded training window coverage mismatch')
    trace_actions=0
    for path,(window,recorded) in zip(paths,expected,strict=True):
        episode=training_episode(task,directory,path,root)
        if (episode['recording_window_index']!=window
                or episode['complete']!=(recorded['outcome']!='budget_cutoff')
                or episode['training_start_interaction']!=int(recorded['end_interaction'])-int(recorded['steps'])
                or episode['training_end_interaction']!=int(recorded['end_interaction'])
                or episode['scenario']['id']!=recorded['scenario_id']
                or not np.isclose(sum(t['reward'] for t in episode['transitions']),float(recorded['return']),rtol=0,atol=1e-12)):
            raise ValueError('Recording differs from its chronological ledger entry')
        trace_actions+=len(episode['transitions'])
    return {'run_id':row['run_id'],'training_steps':budget,'validation_episodes':len(metrics),
            'validation_actions_replayed':actions,'recorded_training_actions_replayed':trace_actions,
            'checkpoints_verified':len(checkpoints),'inference_unchanged':True,'source_snapshot_matches':True}


def saved_summary(root):
    study=root/'artifacts/studies'/STUDY
    if not (study/'verification.json').is_file():
        return None
    verification=read_json(study/'verification.json')
    if (verification['status']!='passed'
            or set(verification['output_hashes'])!={'summary.json','paired_neural_tuning.png','neural_tuning_learning.png'}
            or any(sha(study/name)!=value for name,value in verification['output_hashes'].items())):
        raise ValueError('Neural tuning publication verification failed')
    return read_json(study/'summary.json')


def figures(study, summary, root):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,(family,system) in zip(axes,summary['systems'].items()):
        for before,after in zip(system['conditions']['control']['seed_results'],system['conditions']['tuned']['seed_results'],strict=True):
            ax.plot([0,1],[before['validation']['success_rate'],after['validation']['success_rate']],'o-',label=str(before['seed']))
        ax.set(xticks=[0,1],xticklabels=['Fresh baseline','Selected settings'],ylim=(-.02,1.02),title=family.upper())
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.25);ax.legend(title='Training seed',fontsize=8)
    fig.supxlabel('Preliminary: three fresh matched seeds; 200k steps each; 928 development-validation scenarios',fontsize=8)
    fig.savefig(study/'paired_neural_tuning.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,(family,system) in zip(axes,summary['systems'].items()):
        for stage,color in [('control','#2463a6'),('tuned','#208474')]:
            curves=[read_json(root/'artifacts/runs'/r['run_id']/'monitoring.json') for r in system['conditions'][stage]['seed_results']]
            x=[r['training_interactions'] for r in curves[0]]
            values=np.array([[r['success_rate'] for r in curve] for curve in curves]);mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
            ax.plot(x,mean,color=color,label=stage);ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=color,alpha=.15)
        ax.set(title=family.upper(),xlabel='Training steps',ylim=(-.02,1.02));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.legend()
    fig.supxlabel('Fixed 30-pair monitoring subset; mean ± sample training-seed SD; not confidence intervals',fontsize=8)
    fig.savefig(study/'neural_tuning_learning.png',dpi=160);plt.close(fig)
