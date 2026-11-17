"""Four-family development-validation summaries; never opens the final split."""
from collections import Counter
import csv
import gzip
import json

import numpy as np

from .benchmark import load_scenarios
from .core import ROOT,read_json,write_json
from .episodes import aggregate,validate_replay
from .release import sha

FAMILIES={'q-learning':'Q-learning','sarsa':'SARSA','dqn':'DQN','ppo':'PPO'}
SEEDS=(7,19,43,67,91)
STUDY='four-rl-v1'
METRICS=('success_rate','collision_rate','timeout_rate','spl','mean_return')


def summarize(rows,signature,scenario_ids):
    systems={}
    for algorithm,label in FAMILIES.items():
        selected=sorted((r for r in rows if r['algorithm']==algorithm),key=lambda r:r['seed'])
        if [r['seed'] for r in selected]!=list(SEEDS):
            raise ValueError(f'{label} requires all five unique declared seeds')
        if any(r['pilot']['training_interactions']!=200000 or r['validation']['episodes']!=len(scenario_ids)
               for r in selected):
            raise ValueError('Four-way comparison requires matching budgets and validation support')
        for row in selected:
            counts=row['validation']['counts']
            if set(counts)!={'success','collision','timeout'} or sum(counts.values())!=len(scenario_ids):
                raise ValueError('Outcome counts do not reconcile')
        systems[algorithm]={'label':label,'seed_count':len(selected),'seeds':list(SEEDS),
            'metrics':{m:{'mean':float(np.mean([r['validation'][m] for r in selected])),
                          'sample_sd':float(np.std([r['validation'][m] for r in selected],ddof=1))} for m in METRICS},
            'counts':{outcome:sum(r['validation']['counts'][outcome] for r in selected)
                      for outcome in ('success','collision','timeout')},
            'training_interactions':sum(r['pilot']['training_interactions'] for r in selected),
            'training_seconds':sum(r['pilot']['training_seconds'] for r in selected),
            'seed_results':selected}
    if len(rows)!=20:
        raise ValueError('Exactly 20 main runs are required; pilot and tuned runs are excluded')
    return {'status':'verified','signature':signature,'systems':systems,'validation_scenarios':len(scenario_ids),
        'aggregation_unit':'independent training seed; sample SD, not confidence intervals',
        'scope':'post-release exploratory development validation; historical Part A baselines plus newly trained PPO',
        'ppo_hyperparameter_tuning':False,'final_test_evaluated':False,'original_final_selection_changed':False}


def saved_summary(root=ROOT):
    study=root/'artifacts/studies'/STUDY
    if not (study/'summary.json').is_file() or not (study/'verification.json').is_file():
        return None
    summary=read_json(study/'summary.json')
    if summary['status']!='verified' or read_json(study/'verification.json')['status']!='passed':
        raise ValueError('Four-algorithm comparison has not passed verification')
    return summary


def audit_run(task,row,root=ROOT,ppo_inference=False):
    from .agents import load_policy,viewer_checkpoint
    from .workflow import source_hash
    directory=root/'artifacts/runs'/row['run_id']
    manifest=read_json(directory/'manifest.json')
    task.check(manifest['signature'])
    scenarios=load_scenarios(task,'validation',root)
    training=load_scenarios(task,'train',root)
    if (manifest['status']!='complete' or manifest['final_test_evaluated'] or
            manifest['training_interactions']!=200000 or
            manifest['validation_scenario_ids']!=[s['id'] for s in scenarios] or
            manifest['training_scenario_ids']!=[s['id'] for s in training]):
        raise ValueError('Run identity, budget or scenario mismatch')
    config=read_json(directory/'configuration.json')
    if config['environment']!=task.config:
        raise ValueError('Historical/new environments differ')
    key=row['algorithm'].replace('-','_')
    metrics=read_json(directory/f'{key}_episodes.json')
    if [m['scenario_id'] for m in metrics]!=[s['id'] for s in scenarios]:
        raise ValueError('Validation episode coverage mismatch')
    for k,v in aggregate(metrics).items():
        if row['validation'][k]!=v:
            raise ValueError(f'Saved aggregate mismatch: {k}')
    with (directory/'training.csv').open(newline='',encoding='utf-8') as file:
        logged=sum(int(r['steps']) for r in csv.DictReader(file))
    partial=row['pilot']['incomplete_budget_cutoff_episode']
    if logged+(partial['steps'] if partial else 0)!=200000:
        raise ValueError('Training ledger does not reconcile')
    monitor=read_json(directory/'monitoring.json')
    if [m['training_interactions'] for m in monitor]!=list(range(10000,200001,10000)):
        raise ValueError('Monitoring schedule mismatch')
    best=max(monitor,key=lambda m:(m['success_rate'],m['spl'],-m['training_interactions']))
    if best['training_interactions']!=row['selected_interactions'] or best['training_interactions']!=manifest['selected_interactions']:
        raise ValueError('Checkpoint selection mismatch')
    result={'run_id':row['run_id'],'training_ledger_reconciled':True,
            'validation_episodes':len(metrics),'selection_verified':True,
            'policy_actions_replayed':0,'scope':'saved aggregate, ledger and task verification'}
    if ppo_inference:
        if source_hash(directory/'source')!=manifest['source_hash']:
            raise ValueError('PPO source snapshot mismatch')
        policy=load_policy(viewer_checkpoint(directory),task)
        before=policy.state_fingerprint();cache={};actions=0
        with gzip.open(directory/'ppo_trajectories.jsonl.gz','rt',encoding='utf-8') as file:
            for scenario,line,metric in zip(scenarios,file,metrics,strict=True):
                episode=json.loads(line)
                if episode['scenario']!=scenario or episode['mode']!='validation':
                    raise ValueError('PPO trajectory scenario mismatch')
                validate_replay(task,episode)
                if episode['metrics']!={k:v for k,v in metric.items() if k!='scenario_id'}:
                    raise ValueError('PPO trajectory metrics mismatch')
                for step in episode['transitions']:
                    state=tuple(step['observation'])
                    if state not in cache:
                        cache[state]=policy.decision(state)
                    if step['decision']!=cache[state] or step['scores'] is not None or step['action']!=cache[state]['action']:
                        raise ValueError('PPO action/probability mismatch')
                    actions+=1
        if policy.state_fingerprint()!=before:
            raise ValueError('Audit mutated PPO state')
        checkpoints=sorted((directory/'checkpoints').glob('*.zip'))
        if len(checkpoints)!=20:
            raise ValueError('PPO checkpoint count mismatch')
        for path in checkpoints:
            load_policy(path,task)
        result.update(policy_actions_replayed=actions,source_snapshot_matches=True,
                      inference_unchanged=True,checkpoints_verified=20)
    return result


def figures(study,summary,root=ROOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    colors=('#2463a6','#b87816','#ae4c79','#208474')
    for i,(algorithm,system) in enumerate(summary['systems'].items()):
        for ax,metric in zip(axes,('success_rate','spl')):
            values=[r['validation'][metric] for r in system['seed_results']]
            ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),fmt='o',capsize=5,color=colors[i])
            ax.scatter(i+np.linspace(-.08,.08,5),values,marker='x',color=colors[i])
    for ax,title in zip(axes,('Validation success','Validation SPL')):
        ax.set(xticks=range(4),xticklabels=list(FAMILIES.values()),ylim=(-.02,1.02),title=title)
        ax.grid(axis='y',alpha=.2)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    fig.supxlabel('928 development pairs per seed; 200,000 training interactions each; mean ± sample seed SD; exploratory extension',fontsize=8)
    fig.savefig(study/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    curve_data={}
    for color,(algorithm,system) in zip(colors,summary['systems'].items()):
        curves=[read_json(root/'artifacts/runs'/r['run_id']/'monitoring.json') for r in system['seed_results']]
        x=[m['training_interactions'] for m in curves[0]]
        curve_data[algorithm]={'interactions':x}
        for ax,metric in zip(axes,('success_rate','spl')):
            values=np.array([[m[metric] for m in curve] for curve in curves])
            mean,sd=values.mean(axis=0),values.std(axis=0,ddof=1)
            curve_data[algorithm][metric]={'mean':mean.tolist(),'sample_sd':sd.tolist()}
            ax.plot(x,mean,label=system['label'],color=color)
            ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=color,alpha=.1)
    for ax,title in zip(axes,('Monitoring success','Monitoring SPL')):
        ax.set(title=title,xlabel='Training interactions',ylim=(-.02,1.02));ax.grid(alpha=.2)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[1].legend()
    fig.supxlabel('Fixed 30-pair validation monitor; five-seed means ± sample SD (clipped to [0,1]); not confidence intervals',fontsize=8)
    fig.savefig(study/'learning.png',dpi=160);plt.close(fig)
    write_json(study/'curve_data.json',curve_data)
