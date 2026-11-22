"""Paired representation statistics and audits; no final-test loader."""
import csv
import gzip
import json

import numpy as np

from .agents import load_policy, viewer_checkpoint
from .benchmark import load_scenarios
from .core import ROOT, NavigationEnv, read_json, write_json
from .dashboard import scenario_labels
from .episodes import aggregate, validate_replay
from .four_way import audit_run
from .four_way_analysis import grouped_metrics, outcome_totals, stats
from .observation_encoding import COORDINATES, ABSOLUTE_RELATIVE
from .release import sha
from .workflow import source_hash

STUDY = 'representation-v1'
SEEDS = [101,211,307,401,503]
VARIANTS = {'control': COORDINATES, 'relative': ABSOLUTE_RELATIVE}
METRICS = ('success_rate','collision_rate','timeout_rate','spl','mean_return','mean_successful_route_length')


def summarize(rows, signature):
    if len(rows) != 10:
        raise ValueError('Exactly ten complete main runs are required')
    systems = {}
    for variant, encoding in VARIANTS.items():
        selected = sorted([r for r in rows if r['variant']==variant],key=lambda r:r['seed'])
        if [r['seed'] for r in selected] != SEEDS:
            raise ValueError('All five unique paired seeds are required per treatment')
        if any(r['encoding']!=encoding or r['pilot']['training_interactions']!=200000
               or r['validation']['episodes']!=928 for r in selected):
            raise ValueError('Representation encoding, budget or validation support mismatch')
        for row in selected:
            if set(row['validation']['counts'])!={'success','collision','timeout'} or sum(row['validation']['counts'].values())!=928:
                raise ValueError('Validation outcome counts do not reconcile')
        systems[variant] = {'encoding':encoding,'seed_count':5,'seed_results':selected,
                           'metrics':{m:stats([r['validation'][m] for r in selected]) for m in METRICS},
                           'training_seconds':stats([r['pilot']['training_seconds'] for r in selected])}
    deltas = [{'seed':b['seed'],**{m:a['validation'][m]-b['validation'][m]
               for m in ('success_rate','collision_rate','timeout_rate','spl','mean_return')}}
              for b,a in zip(systems['control']['seed_results'],systems['relative']['seed_results'],strict=True)]
    return {'status':'verified','signature':signature,'systems':systems,'paired_seed_differences':deltas,
            'paired_change':{m:stats([d[m] for d in deltas]) for m in deltas[0] if m!='seed'},
            'aggregation_unit':'paired independent training seed; sample SD, not confidence intervals',
            'scope':'post-release exploratory development validation; no hyperparameter search',
            'final_test_evaluated':False,'original_final_selection_changed':False}


def validate_training_prefix(task, record):
    if record['mode']!='training' or not record['transitions']:
        raise ValueError('A training trace must contain actual training actions')
    task.check(record['signature'])
    if record['training_end_interaction']-record['training_start_interaction']!=len(record['transitions']):
        raise ValueError('Training trace interaction bounds do not reconcile')
    if record['complete']:
        validate_replay(task,record)
        return
    if 'metrics' in record or record.get('cutoff_reason') not in ('budget_cutoff','interrupted','run_failed'):
        raise ValueError('Incomplete training prefixes cannot contain terminal metrics')
    env=NavigationEnv(task,[record['scenario']])
    observation,_=env.reset(seed=record['seed'],options={'scenario':record['scenario']})
    if observation.tolist()!=record['initial_observation']:
        raise ValueError('Training prefix initial state mismatch')
    for transition in record['transitions']:
        if transition['observation']!=observation.tolist():
            raise ValueError('Training prefix before-state mismatch')
        observation,reward,terminated,truncated,info=env.step(transition['action'])
        if (transition['next_observation']!=observation.tolist() or transition['reward']!=reward
                or transition['terminated']!=terminated or transition['truncated']!=truncated
                or transition['info']!=info or terminated or truncated):
            raise ValueError('Training prefix transition/ending mismatch')


def audit_traces(task,directory,manifest,scenarios):
    traces=manifest['training_traces']
    records=traces['records']
    if traces['requested_episodes']!=3 or len(records)!=3 or traces['extra_environment_interactions']!=0:
        raise ValueError('Registered chronological training traces are missing')
    with (directory/'training.csv').open(newline='',encoding='utf-8') as file:
        ledger=list(csv.DictReader(file))[:3]
    actions=0
    for index,(item,row) in enumerate(zip(records,ledger,strict=True),1):
        path=directory/item['path']
        if item['path']!=f'training_traces/episode-{index:03d}.json' or sha(path)!=item['sha256']:
            raise ValueError('Training trace identity/checksum mismatch')
        record=read_json(path)
        validate_training_prefix(task,record)
        if (not record['complete'] or record['seed']!=manifest['training_seed']
                or record['scenario']!=scenarios.get(row['scenario_id'])
                or len(record['transitions'])!=int(row['steps'])
                or record['training_end_interaction']!=int(row['end_interaction'])
                or record['metrics']['outcome']!=row['outcome']
                or record['metrics']['return']!=float(row['return'])
                or item['actions']!=len(record['transitions']) or not item['complete']):
            raise ValueError('Training trace does not match the first chronological ledger rows')
        if any(t['scores'] is not None or t.get('decision') is not None for t in record['transitions']):
            raise ValueError('Exploratory traces must not invent frozen-policy scores')
        actions+=len(record['transitions'])
    return {'records':len(records),'actions':actions,'chronological_ledger_verified':True}


def audit(task,row,study_manifest,plan):
    directory=ROOT/'artifacts/runs'/row['run_id']
    manifest=read_json(directory/'manifest.json')
    result=audit_run(task,row)
    if source_hash(directory/'source')!=manifest['source_hash'] or manifest['source_hash']!=study_manifest['source_hash']:
        raise ValueError('Representation source snapshot mismatch')
    training=read_json(directory/'configuration.json')['training']
    expected={**plan['training'],'seed':row['seed'],'run_id':row['run_id'],'observation_encoding':row['encoding']}
    if training!=expected:
        raise ValueError('Run configuration differs from registered one-factor treatment')
    policy=load_policy(viewer_checkpoint(directory),task)
    if policy.encoding!=row['encoding']:
        raise ValueError('Loaded inference encoding differs from treatment')
    dimensions=4 if row['variant']=='control' else 6
    parameters=sum(p.numel() for p in policy.model.q_net.parameters())
    if row['pilot']['input_dimensions']!=dimensions or row['pilot']['online_parameter_count']!=parameters:
        raise ValueError('Recorded representation dimensions/parameter count mismatch')
    before=policy.state_fingerprint();cache={};actions=0
    scenarios=load_scenarios(task,'validation')
    metrics=read_json(directory/'dqn_episodes.json')
    with gzip.open(directory/'dqn_trajectories.jsonl.gz','rt',encoding='utf-8') as file:
        for scenario,line,metric in zip(scenarios,file,metrics,strict=True):
            episode=json.loads(line)
            if episode['scenario']!=scenario or episode['mode']!='validation':
                raise ValueError('Representation validation scenario mismatch')
            validate_replay(task,episode)
            if episode['metrics']!={k:v for k,v in metric.items() if k!='scenario_id'}:
                raise ValueError('Representation validation metric mismatch')
            for transition in episode['transitions']:
                state=tuple(transition['observation'])
                if state not in cache:
                    action,scores=policy.action_and_scores(state)
                    cache[state]=(action,scores.tolist())
                if (transition['action'],transition['scores'])!=cache[state]:
                    raise ValueError('Representation checkpoint action/score mismatch')
                actions+=1
    if before!=policy.state_fingerprint():
        raise ValueError('Representation inference audit changed learned state')
    checkpoints=sorted((directory/'checkpoints').glob('*.zip'))
    if len(checkpoints)!=20:
        raise ValueError('Representation monitoring checkpoint coverage mismatch')
    for path in checkpoints:
        load_policy(path,task)
    result.update(policy_actions_replayed=actions,checkpoints_verified=20,
                  training_traces=audit_traces(task,directory,manifest,
                      {s['id']:s for s in load_scenarios(task,'train')}),inference_unchanged=True)
    return result


def add_groups(summary,task):
    labels={s['id']:scenario_labels(task.grid,s) for s in load_scenarios(task,'validation')}
    if labels!=read_json(ROOT/'artifacts/studies/milestone5-v1/scenario_labels.json')['scenarios']:
        raise ValueError('Historical robustness labels changed')
    for system in summary['systems'].values():
        groups={}
        for row in system['seed_results']:
            metrics=read_json(ROOT/'artifacts/runs'/row['run_id']/'dqn_episodes.json')
            for group in grouped_metrics(metrics,labels):
                groups.setdefault((group['dimension'],group['group']),[]).append({'seed':row['seed'],'metrics':group['metrics']})
        system['groups']=[{'dimension':d,'group':g,'scenarios_per_seed':rows[0]['metrics']['episodes'],
                          'metrics':{m:stats([r['metrics'][m] for r in rows]) for m in METRICS},
                          'counts':outcome_totals(rows),'seed_results':rows} for (d,g),rows in groups.items()]


def figures(study,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, FuncFormatter
    colors={'control':'#2463a6','relative':'#208474'}
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    for ax,metric in zip(axes,('success_rate','spl')):
        for before,after in zip(summary['systems']['control']['seed_results'],summary['systems']['relative']['seed_results']):
            ax.plot([0,1],[before['validation'][metric],after['validation'][metric]],'o-',alpha=.65,label=str(before['seed']))
        ax.set(xticks=[0,1],xticklabels=['Absolute (4)','Absolute + relative (6)'],ylim=(-.02,1.02),title=metric.replace('_',' ').title())
        ax.grid(axis='y',alpha=.2)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[1].legend(title='Training seed',fontsize=8)
    fig.supxlabel('Paired fresh seeds; 200k interactions each; 928 validation pairs; exploratory input ablation',fontsize=9)
    fig.savefig(study/'paired_representation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    for variant,system in summary['systems'].items():
        curves=[read_json(ROOT/'artifacts/runs'/r['run_id']/'monitoring.json') for r in system['seed_results']]
        x=[r['training_interactions'] for r in curves[0]]
        for ax,metric in zip(axes,('success_rate','spl')):
            values=np.array([[r[metric] for r in curve] for curve in curves]);mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
            ax.plot(x,mean,label=variant,color=colors[variant]);ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.15,color=colors[variant])
    for ax,metric in zip(axes,('Success','SPL')):
        ax.set(title=f'Monitoring {metric}',xlabel='Training interactions',ylim=(-.02,1.02),xticks=[0,50000,100000,150000,200000]);ax.grid(alpha=.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:g}k'))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[1].legend()
    fig.supxlabel('Fixed 30-pair monitor; mean ± sample seed SD, clipped to [0,1]; not confidence intervals',fontsize=9)
    fig.savefig(study/'representation_learning.png',dpi=160);plt.close(fig)


def saved_summary(root=ROOT):
    study=root/'artifacts/studies'/STUDY
    if not (study/'verification.json').is_file():
        return None
    verification=read_json(study/'verification.json')
    if verification['status']!='passed' or any(sha(study/name)!=value for name,value in verification['output_hashes'].items()):
        raise ValueError('Representation publication verification failed')
    return read_json(study/'summary.json')
