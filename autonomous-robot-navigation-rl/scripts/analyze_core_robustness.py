"""Milestone 5: outcome-independent labels and saved-validation reanalysis."""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robot_navigation.agents import load_policy, viewer_checkpoint
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, digest, load_task, read_json, write_json
from robot_navigation.dashboard import scenario_labels, system_runs, frozen_episode
from robot_navigation.episodes import aggregate, validate_replay
from robot_navigation.refinement import task_for_run

METRICS = ('success_rate', 'collision_rate', 'timeout_rate', 'spl', 'mean_return', 'mean_successful_route_length')
EXEMPLARS = ('m3-q-learning-seed7', 'm3-sarsa-seed7', 'm3-dqn-seed7',
             'm4-tuned-q-learning-seed101', 'm4-tuned-sarsa-seed101')


def mean_sd(values):
    values = [v for v in values if v is not None]
    return {'mean': float(np.mean(values)) if values else None,
            'sd': float(np.std(values, ddof=1)) if len(values)>1 else None,
            'seed_count': len(values)}


def summarize(rows):
    return {metric: mean_sd([r['metrics'][metric] for r in rows]) for metric in METRICS}


def plot(study, summary):
    folder = study / 'figures'
    folder.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, metric in zip(axes, ('success_rate', 'spl')):
        for name, system in summary['systems'].items():
            rows = [r for r in system['groups'] if r['dimension']=='distance']
            rows = sorted(rows, key=lambda r: ('short','medium','long').index(r['group']))
            ax.plot([r['group'] for r in rows], [r['aggregate'][metric]['mean'] for r in rows], marker='o', label=name)
        ax.set_ylim(0, 1.03)
        ax.set_title('Validation success' if metric=='success_rate' else 'Validation SPL')
        ax.grid(axis='y', alpha=.2)
        if metric=='success_rate':
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8)
    fig.supxlabel('Fixed validation distance groups; means over declared seeds; voting is one ensemble', fontsize=9)
    fig.savefig(folder / 'distance.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5.6), layout='constrained')
    names = list(summary['systems'])
    groups = [(d,g) for d in ('corridor','wall proximity','region crossing')
              for g in sorted({r['group'] for r in summary['systems'][names[0]]['groups'] if r['dimension']==d})]
    values = [[next(r for r in summary['systems'][n]['groups'] if (r['dimension'],r['group'])==(d,g))['aggregate']['success_rate']['mean']
               for d,g in groups] for n in names]
    im=ax.imshow(values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(len(groups)), [g for d,g in groups], rotation=20, ha='right')
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j,i,f'{value:.1%}',ha='center',va='center',color='white' if value>.55 else '#172554')
    fig.colorbar(im, ax=ax, label='Mean validation success')
    ax.set_title('Predefined fixed-map robustness groups (overlapping dimensions)')
    fig.savefig(folder / 'groups.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax, metric in zip(axes, ('success_rate','spl')):
        for name, system in summary['systems'].items():
            curves = system['learning_curve']
            if not curves:
                continue
            x=np.array([r['interactions'] for r in curves])
            y=np.array([r[metric]['mean'] for r in curves])
            sd=np.array([r[metric]['sd'] or 0 for r in curves])
            line,=ax.plot(x,y,label=name)
            ax.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color=line.get_color(),alpha=.1)
        ax.set(xlabel='Training interactions',ylabel=metric,ylim=(0,1.03))
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.supxlabel('Saved 30-pair monitoring curves; mean ± sample seed SD, not confidence intervals', fontsize=9)
    fig.savefig(folder / 'learning.png',dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite-id',default='milestone5-v1')
    args=parser.parse_args()
    if Path(args.suite_id).name != args.suite_id or args.suite_id in ('.','..'):
        raise ValueError('suite-id must be one directory name')
    study=ROOT / 'artifacts/studies' / args.suite_id
    study.mkdir(parents=True, exist_ok=False)
    write_json(study / 'status.json', {'status':'running','final_test_policy_executions':0})
    started=perf_counter()
    task=load_task()
    scenarios=load_scenarios(task,'validation')
    labels={s['id']:scenario_labels(task.grid,s) for s in scenarios}
    label_record={'signature':task.signature,'scope':'canonical validation only','scenarios':labels}
    write_json(study / 'scenario_labels.json',label_record)
    # Label identity is frozen on disk before outcome aggregation.
    inputs={}
    def track(path):
        inputs[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return read_json(path)
    summary={'signature':task.signature,'labels_hash':digest(label_record),'systems':{},
             'scope':'Development validation reanalysis on a familiar fixed map; final test sealed',
             'new_training_interactions':0,'new_benchmark_evaluation_interactions':0}
    verified_episodes=0
    for name, runs in system_runs().items():
        seed_rows=[]
        all_groups={}
        learning=[]
        for run in runs:
            directory=ROOT / 'artifacts/runs' / run
            matching_task=task_for_run(task,directory)
            task.check(matching_task.signature)
            manifest=track(directory / 'manifest.json')
            if manifest['status'] != 'complete':
                raise ValueError(f'Incomplete run: {run}')
            checkpoint=viewer_checkpoint(directory)
            load_policy(checkpoint,task)  # verifies task and model checksum
            inputs[checkpoint.relative_to(ROOT).as_posix()]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            key={'q-learning':'q_learning','sarsa':'sarsa','dqn':'dqn','voting':'voting'}[track(directory / 'configuration.json')['training']['algorithm']]
            metrics=track(directory / f'{key}_episodes.json')
            if len(metrics)!=len(labels) or {r['scenario_id'] for r in metrics}!=set(labels):
                raise ValueError(f'Scenario coverage mismatch: {run}')
            overall=aggregate(metrics)
            stored=track(directory / 'validation.json')[key]
            for k,v in overall.items():
                if isinstance(v,float):
                    assert np.isclose(v,stored[k],atol=1e-12,rtol=0), (run,k)
                else:
                    assert v==stored[k], (run,k)
            seed=track(directory / 'configuration.json')['training']['seed']
            seed_rows.append({'run_id':run,'seed':seed,'metrics':overall,
                'training_interactions':manifest.get('training_interactions',0),
                'selected_interactions':manifest.get('selected_interactions'),
                'training_seconds':manifest.get('training_seconds'),
                'evaluation_seconds':stored['evaluation_seconds'],
                'mean_decision_microseconds':stored['mean_decision_microseconds']})
            for dimension in next(iter(labels.values())):
                for group in sorted({v[dimension] for v in labels.values()}):
                    selected=[r for r in metrics if labels[r['scenario_id']][dimension]==group]
                    all_groups.setdefault((dimension,group),[]).append({'run_id':run,'seed':seed,'metrics':aggregate(selected)})
            monitor=directory / 'monitoring.json'
            if monitor.exists():
                learning.append(track(monitor))
            verified_episodes+=len(metrics)
        curves=[]
        if learning:
            steps=[r['training_interactions'] for r in learning[0]]
            assert all([r['training_interactions'] for r in curve]==steps for curve in learning)
            for i,step in enumerate(steps):
                curves.append({'interactions':step,**{m:mean_sd([c[i][m] for c in learning]) for m in ('success_rate','spl')}})
        summary['systems'][name]={'seed_results':seed_rows,'aggregate':summarize(seed_rows),
            'seed_count':len(seed_rows),'learning_curve':curves,
            'groups':[{'dimension':d,'group':g,'scenarios_per_seed':rows[0]['metrics']['episodes'],
                       'seed_results':rows,'aggregate':summarize(rows)} for (d,g),rows in all_groups.items()]}
    paired=[]
    for family in ('Q-learning','SARSA'):
        before=summary['systems'][f'{family} Baseline']
        after=summary['systems'][f'{family} Refined']
        for b,a in zip(before['groups'],after['groups']):
            assert (b['dimension'],b['group'])==(a['dimension'],a['group'])
            deltas=[]
            for br,ar in zip(b['seed_results'],a['seed_results']):
                assert br['seed']==ar['seed']
                deltas.append({'seed':br['seed'],**{m:ar['metrics'][m]-br['metrics'][m] for m in ('success_rate','collision_rate','timeout_rate','spl')}})
            paired.append({'family':family,'dimension':b['dimension'],'group':b['group'],
                           'seed_differences':deltas,**{m:mean_sd([r[m] for r in deltas]) for m in ('success_rate','collision_rate','timeout_rate','spl')}})
    write_json(study / 'paired_groups.json',paired)
    # Archive early models without editing their original experiment directories.
    early={}
    for run in EXEMPLARS:
        directory=ROOT / 'artifacts/runs' / run
        checkpoints=sorted(directory.glob('checkpoints/step-*.npz'))+sorted(directory.glob('checkpoints/step-*.zip'))
        if not checkpoints:
            early[run]={'available':False,'reason':'Periodic checkpoints absent; inference-only selected model retained'}
            continue
        source=checkpoints[0]
        target=study / 'early_checkpoints' / run / source.name
        target.parent.mkdir(parents=True)
        for path in (source,source.with_suffix('.json')):
            shutil.copy2(path,target.with_suffix(path.suffix))
        early[run]={'available':True,'checkpoint':target.relative_to(ROOT).as_posix(),
                    'source_checkpoint':source.relative_to(ROOT).as_posix(),
                    'interactions':10000,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    write_json(study / 'early_checkpoints.json',early)
    # Independent action replays for declared early/selected examples and actual failure.
    examples=[]
    selected_scenario=next(s for s in scenarios if s['shortest_length']>=11)
    for run in EXEMPLARS:
        directory=ROOT / 'artifacts/runs' / run
        choices=[('selected',None)]
        if early[run]['available']:
            choices.append(('early',ROOT / early[run]['checkpoint']))
        for stage,checkpoint in choices:
            record=frozen_episode(task,directory,selected_scenario,checkpoint)
            target=study / 'replays' / f'{run}-{stage}.json'
            write_json(target,record)
            examples.append({'run_id':run,'stage':stage,'scenario_id':record['scenario']['id'],
                             'actions':len(record['transitions']),'outcome':record['metrics']['outcome']})
    failure=read_json(ROOT / 'artifacts/runs/m4-control-q-learning-seed101/replays/q_learning_collision.json')
    validate_replay(task,failure)
    write_json(study / 'replays/control-collision.json',failure)
    vote=frozen_episode(task,ROOT / 'artifacts/runs/m4-voting-q-learning',selected_scenario)
    write_json(study / 'replays/voting-selected.json',vote)
    write_json(study / 'summary.json',summary)
    rows=[]
    for name,system in summary['systems'].items():
        for group in system['groups']:
            for seed in group['seed_results']:
                rows.append({'system':name,'dimension':group['dimension'],'group':group['group'],
                             'run_id':seed['run_id'],'seed':seed['seed'],**{k:v for k,v in seed['metrics'].items() if k!='counts'}})
    with (study / 'groups.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(study,summary)
    shutil.copy2(__file__,study / 'analyzer.py')
    inputs['docs/milestone5_protocol.md']=hashlib.sha256((ROOT / 'docs/milestone5_protocol.md').read_bytes()).hexdigest()
    write_json(study / 'inputs.json',inputs)
    write_json(study / 'verification.json',{'status':'complete','saved_episode_metrics_reconciled':verified_episodes,
        'systems':len(summary['systems']),'training_runs':25,'voting_systems':1,
        'scenario_count':len(labels),'group_rows':len(rows),'label_hash':digest(label_record),
        'example_replays':examples,'additional_display_replay_actions':sum(r['actions'] for r in examples)+len(vote['transitions']),
        'existing_collision_replay_actions_verified':len(failure['transitions']),
        'new_training_interactions':0,'new_benchmark_evaluation_interactions':0,
        'final_test_policy_executions':0,'analysis_seconds':perf_counter()-started,
        'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    write_json(study / 'status.json', {'status':'complete','final_test_policy_executions':0})
    print(f'Completed {study.name}: {verified_episodes} saved episode metrics; {len(rows)} group/seed rows; final test untouched')


if __name__=='__main__':
    main()
