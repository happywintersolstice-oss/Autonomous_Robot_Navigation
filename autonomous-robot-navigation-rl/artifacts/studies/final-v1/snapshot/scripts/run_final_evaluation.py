"""Freeze, evaluate once, and audit the predeclared final matrix. No training."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robot_navigation.core import ROOT, digest, read_json, write_json
from robot_navigation.dashboard import scenario_labels
from robot_navigation.episodes import aggregate, validate_replay
from robot_navigation.release import freeze_release, verify_freeze, frozen_policy, load_final_scenarios
from robot_navigation.workflow import evaluate
from robot_navigation.visualization import save_figure

METRICS=('success_rate','collision_rate','timeout_rate','spl','mean_return','mean_successful_route_length')


def stats(values):
    values=[v for v in values if v is not None]
    return {'mean':float(np.mean(values)) if values else None,
            'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'replicates':len(values)}


def fingerprint(policy):
    if hasattr(policy,'state_fingerprint'):
        return policy.state_fingerprint()
    if hasattr(policy,'q'):
        return policy.q.tobytes()
    return None


def audit_record(study,record,task,scenarios,directory):
    policy=frozen_policy(study,record,task)
    before=fingerprint(policy)
    metrics=[];actions=0
    with gzip.open(directory/'trajectories.jsonl.gz','rt',encoding='utf-8') as f:
        for i,line in enumerate(f):
            episode=json.loads(line)
            if episode['scenario']!=scenarios[i] or episode['mode']!='final evaluation':
                raise ValueError('Final replay scenario/order mismatch')
            validate_replay(task,episode,allow_final=True)
            for transition in episode['transitions']:
                obs=np.array(transition['observation'])
                if hasattr(policy,'decision'):
                    decision=policy.decision(obs)
                    assert decision==transition['decision']
                    action=decision['action'];scores=None
                elif hasattr(policy,'action_and_scores'):
                    action,scores=policy.action_and_scores(obs)
                else:
                    scores=policy.scores(obs) if hasattr(policy,'scores') else None
                    action=policy.act(obs)
                assert action==transition['action']
                if scores is not None:
                    assert np.array_equal(scores,np.array(transition['scores']))
                else:
                    assert transition['scores'] is None
                actions+=1
            metrics.append({'scenario_id':episode['scenario']['id'],**episode['metrics']})
    assert len(metrics)==len(scenarios)
    assert metrics==read_json(directory/'episodes.json')
    assert before==fingerprint(policy)
    return {'id':record['id'],'episodes':len(metrics),'actions':actions,'all_transitions_and_scores_verified':True,
            'inference_state_unchanged':True}


def figures(output,summary,episodes):
    folder=output/'figures';folder.mkdir()
    names=list(summary['systems'])
    fig,axes=plt.subplots(1,2,figsize=(13,6.5),layout='constrained')
    for ax,metric in zip(axes,('success_rate','spl')):
        values=[summary['systems'][n]['metrics'][metric]['mean'] for n in names]
        errors=[summary['systems'][n]['metrics'][metric]['sample_sd'] or 0 for n in names]
        ax.barh(names,values,xerr=errors,color='#2463a6',capsize=3)
        ax.set_xlim(0,1.05);ax.set_xlabel('Success fraction' if metric=='success_rate' else 'SPL')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Frozen final evaluation · means ± sample replicate SD; one vote and one BFS')
    fig.savefig(folder/'comparison.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for name in ('Q-learning Baseline','Q-learning Refined','SARSA Refined','Part A DQN','Q-learning Voting'):
        groups=summary['systems'][name]['groups']
        lookup={g['group']:g['metrics']['success_rate']['mean'] for g in groups if g['dimension']=='distance'}
        values=[lookup.get(band) for band in ('short','medium','long')]
        ax.plot(('short','medium','long'),values,marker='o',label=name)
    ax.set(ylim=(0,1.04),ylabel='Mean final-test success',title='Frozen core robustness by reference route length')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(folder/'distance.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    bottom=np.zeros(len(names))
    for outcome,color in (('success','#15803d'),('collision','#dc2626'),('timeout','#64748b')):
        values=[summary['systems'][n]['metrics'][f'{outcome}_rate']['mean'] for n in names]
        ax.barh(names,values,left=bottom,label=outcome,color=color);bottom+=values
    ax.set(xlim=(0,1),xlabel='Mean outcome fraction',title='Final outcome partition; failures retained')
    ax.legend(loc='lower right');fig.savefig(folder/'outcomes.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    data=[];labels=[]
    for name in names:
        values=[m['steps'] for record in summary['systems'][name]['seed_results'] for m in episodes[record['id']] if m['outcome']=='success']
        if values:
            data.append(values);labels.append(f'{name} (n={len(values)})')
    ax.boxplot(data,orientation='horizontal',tick_labels=labels,showfliers=False)
    ax.set(xlabel='Actions on successful episodes only; pooled fixed replicates, not uncertainty',
           title='Conditional successful route lengths; read with outcome fractions')
    fig.savefig(folder/'routes.png',dpi=145);plt.close(fig)


def analyze(output,freeze,rows,labels,episodes):
    systems={}
    for group in dict.fromkeys(r['group'] for r in rows):
        selected=[r for r in rows if r['group']==group]
        groups=[]
        for dimension in next(iter(labels.values())):
            for label in sorted({r[dimension] for r in labels.values()}):
                per_seed=[]
                for record in selected:
                    metrics=aggregate([m for m in episodes[record['id']] if labels[m['scenario_id']][dimension]==label])
                    per_seed.append({'id':record['id'],'seed':record['seed'],'metrics':metrics})
                groups.append({'dimension':dimension,'group':label,'scenarios_per_replicate':per_seed[0]['metrics']['episodes'],
                    'seed_results':per_seed,'metrics':{m:stats([r['metrics'][m] for r in per_seed]) for m in METRICS}})
        systems[group]={'seed_results':selected,'replicates':len(selected),
            'replicate_unit':'RNG stream' if group=='Random reference' else 'one ensemble' if 'Voting' in group else 'one reference' if group=='BFS reference' else 'training seed',
            'metrics':{m:stats([r['evaluation'][m] for r in selected]) for m in METRICS},'groups':groups}
    by_id={r['id']:r for r in rows}
    paired={}
    for family in ('Q-learning','SARSA'):
        control=systems[f'{family} Baseline']['seed_results'];tuned=systems[f'{family} Refined']['seed_results']
        differences=[]
        for b,t in zip(control,tuned):
            assert b['seed']==t['seed']
            differences.append({'seed':b['seed'],**{m:t['evaluation'][m]-b['evaluation'][m] for m in ('success_rate','collision_rate','timeout_rate','spl')}})
        paired[family]={'seed_differences':differences,**{m:stats([r[m] for r in differences]) for m in ('success_rate','collision_rate','timeout_rate','spl')}}
    vote=by_id['m4-voting-q-learning']['evaluation']
    voting={str(seed):{m:vote[m]-by_id[f'm4-tuned-q-learning-seed{seed}']['evaluation'][m] for m in ('success_rate','collision_rate','timeout_rate','spl')} for seed in (101,211,307)}
    targets={}
    for name in ('Q-learning Refined','SARSA Refined'):
        values=systems[name]['metrics']
        targets[name]={'success':values['success_rate']['mean']>=.8,'collision':values['collision_rate']['mean']<=.02,'spl':values['spl']['mean']>=.75}
    summary={'signature':freeze['signature'],'scope':'Frozen final-test initial conditions on a familiar map',
        'selected_system':freeze['selected_system'],'selected_control':freeze['selected_control'],
        'systems':systems,'paired_changes':paired,'voting_minus_member':voting,'targets':targets,
        'primary_single':{k:by_id[k]['evaluation'] for k in (freeze['selected_control'],freeze['selected_system'])},
        'no_test_based_selection':True}
    write_json(output/'summary.json',summary)
    figures(output,summary,episodes)
    return summary


def run(study,expected,root=ROOT):
    freeze=verify_freeze(study,expected,root)
    output=study/'evaluation';output.mkdir(exist_ok=False)
    write_json(study/'status.json',{'status':'running','final_evaluation_complete':False,'freeze_digest':expected})
    started=perf_counter();rows=[];audits=[];episodes={}
    try:
        task,scenarios=load_final_scenarios(study,freeze)
        labels={s['id']:scenario_labels(task.grid,s) for s in scenarios}
        write_json(output/'scenario_labels.json',{'signature':task.signature,'labels':labels})
        for record in freeze['records']:
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if perf_counter()-started>freeze['limits']['wall_seconds'] or size>freeze['limits']['evaluation_storage_bytes']:
                raise RuntimeError('Predeclared evaluation limit reached before next record; incomplete matrix retained')
            directory=output/'runs'/record['id'];directory.mkdir(parents=True)
            policy=frozen_policy(study,record,task)
            before=fingerprint(policy)
            summary,metrics,replays=evaluate(task,scenarios,policy,record['evaluation_seed'],
                'frozen complete final-test manifest; no learning or checkpoint selection',
                trajectory_path=directory/'trajectories.jsonl.gz',mode='final evaluation',allow_final=True)
            assert before==fingerprint(policy)
            if record['kind']=='bfs':
                assert summary['counts']['success']==len(scenarios) and summary['spl']==1
            write_json(directory/'summary.json',summary);write_json(directory/'episodes.json',metrics)
            for replay in replays:
                outcome=replay['metrics']['outcome'];write_json(directory/'replays'/f'{outcome}.json',replay)
            save_figure(directory/'trajectory.png',task.grid,replays[0])
            episodes[record['id']]=metrics
            rows.append({**record,'evaluation':summary})
            write_json(output/'run_results.json',rows)
            print(f"{record['id']}: {summary['counts']}",flush=True)
        evaluation_seconds=perf_counter()-started
        audit_started=perf_counter()
        for record in freeze['records']:
            audits.append(audit_record(study,record,task,scenarios,output/'runs'/record['id']))
        summary=analyze(output,freeze,rows,labels,episodes)
        verify_freeze(study,expected,root)
        verification={'status':'complete','freeze_digest':expected,'frozen_files_unchanged':True,
            'evaluation_records':len(rows),'evaluation_episodes':sum(r['evaluation']['episodes'] for r in rows),
            'evaluation_interactions':sum(r['evaluation']['interactions'] for r in rows),
            'evaluation_seconds':evaluation_seconds,'audit_and_analysis_seconds':perf_counter()-audit_started,
            'audit_episode_replays':sum(r['episodes'] for r in audits),'audit_actions':sum(r['actions'] for r in audits),
            'audits':audits,'bfs_all_optimal':True,'new_training_interactions':0,
            'no_selection_after_test':True,'analysis_source_frozen':True}
        write_json(output/'verification.json',verification)
        # Publish only aggregate final evidence to the dashboard after full audit.
        write_json(study/'summary.json',summary)
        write_json(study/'status.json',{'status':'complete','final_evaluation_complete':True,'freeze_digest':expected})
    except BaseException as exc:
        write_json(study/'status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            'final_evaluation_complete':False,'freeze_digest':expected,'completed_records':len(rows),
            'error':f'{type(exc).__name__}: {exc}','retry_or_selection_performed':False})
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','evaluate'))
    parser.add_argument('--suite-id',default='final-v1')
    parser.add_argument('--freeze-digest')
    args=parser.parse_args()
    if Path(args.suite_id).name!=args.suite_id or args.suite_id in ('.','..'):
        parser.error('suite-id must be one directory name')
    study=ROOT/'artifacts/studies'/args.suite_id
    if args.command=='freeze':
        print(freeze_release(study))
    else:
        if not args.freeze_digest:
            parser.error('evaluate requires the saved --freeze-digest')
        run(study,args.freeze_digest)


if __name__=='__main__':
    main()
