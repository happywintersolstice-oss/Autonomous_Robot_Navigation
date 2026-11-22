"""Bounded, two-process neural tuning with a frozen search/confirmation protocol."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import shutil
import sys
import time

import yaml

from robot_navigation.core import ROOT, digest, load_task, read_json, write_json
from robot_navigation.neural_tuning import (STUDY, audit_run, choose_candidate, figures,
                                           run_config, summarize, validate_plan)
from robot_navigation.part_a import train_part_a
from robot_navigation.release import sha
from robot_navigation.workflow import source_hash


def worker(path):
    job=read_json(path);config=job['config']
    directory,results,pilot=train_part_a(config,ROOT)
    result=results[config['algorithm'].replace('-','_')]
    manifest=read_json(directory/'manifest.json')
    row={'run_id':directory.name,'algorithm':config['algorithm'],'seed':config['seed'],
         'stage':config['stage'],'candidate':job['candidate'],'validation':result,'pilot':pilot,
         'selected_interactions':manifest['selected_interactions'],
         'costs':{'training_seconds':pilot['training_seconds'],'monitoring_seconds':manifest['monitoring_seconds'],
                  'validation_seconds':result['evaluation_seconds'],
                  'decision_microseconds':result['mean_decision_microseconds'],
                  'checkpoint_bytes':manifest['checkpoint_bytes']}}
    output=path.parent.parent/'results'/f'{directory.name}.json'
    if output.exists():
        raise FileExistsError('Worker result already exists')
    write_json(output,row)


def run():
    root=ROOT;study=root/'artifacts/studies'/STUDY
    plan=yaml.safe_load((root/'configs/neural_tuning.yaml').read_text(encoding='utf-8'))
    validate_plan(plan)
    if study.exists() or list((root/'artifacts/runs').glob('m9-*')):
        raise FileExistsError('Neural tuning must use fresh suite/run identities; no silent overwrite or training resume')
    historical={p.relative_to(root).as_posix():sha(p) for folder in ('artifacts','data')
                for p in (root/folder).rglob('*') if p.is_file()}
    old_seeds={read_json(p).get('training_seed') for p in (root/'artifacts/runs').glob('*/manifest.json')}
    if old_seeds & {plan['pilot_seed'],plan['development_seed'],*plan['confirmation_seeds']}:
        raise ValueError('The declared seeds are not fresh')
    study.mkdir(parents=True);source=source_hash(root);started=time.perf_counter()
    write_json(study/'historical_hashes.json',historical)
    write_json(study/'plan.json',plan)
    shutil.copy2(Path(__file__),study/'runner.py')
    shutil.copy2(root/'docs/milestone9_protocol.md',study/'protocol.md')
    write_json(study/'preflight.json',{'plan_hash':digest(plan),'source_hash':source,
        'runner_sha256':sha(Path(__file__)),'protocol_sha256':sha(root/'docs/milestone9_protocol.md'),
        'started_utc':datetime.now(timezone.utc).isoformat(),'signature':load_task().signature,
        'workers':2,'scope':'preliminary three-seed neural tuning; no new final evaluation'})
    status={'status':'running','phase':'pilots','completed_runs':[],'running_runs':[],
            'new_final_test_policy_executions':0,'original_final_selection_changed':False}
    rows=[];jobs={};active={}

    def save():
        status.update(elapsed_seconds=time.perf_counter()-started,completed_runs=[r['run_id'] for r in rows],
                      training_interactions=sum(r['pilot']['training_interactions'] for r in rows),
                      running_runs=list(active))
        write_json(study/'run_results.json',rows);write_json(study/'manifest.json',status)

    def protected():
        if any(not (root/p).is_file() or sha(root/p)!=value for p,value in historical.items()):
            raise ValueError('Historical artifact/data bytes changed')

    def stage_jobs(configs):
        pending=list(configs)
        while pending or active:
            if time.perf_counter()-started>plan['limits']['suite_wall_seconds']:
                raise TimeoutError('Neural suite reached its predeclared wall-time cap')
            while pending and len(active)<plan['workers']:
                config,candidate=pending.pop(0);identity=config['run_id']
                if source_hash(root)!=source:
                    raise ValueError('Source changed during the frozen run matrix')
                if len(jobs)>=plan['limits']['max_runs_including_pilots']:
                    raise ValueError('Declared training-run cap exceeded')
                if sum(j['config']['interactions'] for j in jobs.values())+config['interactions']>plan['limits']['max_training_interactions_including_pilots']:
                    raise ValueError('Declared interaction cap exceeded')
                spec={'config':config,'candidate':candidate};jobs[identity]=spec
                path=study/'jobs'/f'{identity}.json';write_json(path,spec)
                (study/'logs').mkdir(exist_ok=True)
                log=(study/'logs'/f'{identity}.log').open('x',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker',str(path)],
                    cwd=root,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active[identity]=(process,log,time.perf_counter())
                print(f'Started {identity} ({config["interactions"]:,} steps)',flush=True);save()
            for identity,(process,log,run_started) in list(active.items()):
                if time.perf_counter()-run_started>plan['limits']['run_wall_seconds']:
                    raise TimeoutError(f'{identity} exceeded the run wall cap')
                code=process.poll()
                if code is None:
                    continue
                log.close();del active[identity]
                if code:
                    raise RuntimeError(f'{identity} failed (exit {code}); inspect its saved log')
                row=read_json(study/'results'/f'{identity}.json');rows.append(row);save()
                print(f"Completed {identity}: {row['validation']['counts']}",flush=True)
            time.sleep(.5)
        protected()

    try:
        save()
        stage_jobs([(run_config(plan,f,'pilot',plan['pilot_seed'],root=root),'c00') for f in plan['algorithms']])
        pilots={r['algorithm']:r for r in rows}
        estimate=0
        for family,pilot in pilots.items():
            # Both algorithms use fixed one-thread learners. Scaling measured
            # pilot work is approximate; validation outcome mix can change.
            training=pilot['pilot']['training_seconds']*1.3
            monitoring=pilot['costs']['monitoring_seconds']
            validation=pilot['costs']['validation_seconds']*2
            estimate+=4*(training*10+monitoring*10+validation)+6*(training*20+monitoring*20+validation)
        estimate=estimate/plan['workers']+180
        if estimate+time.perf_counter()-started>plan['limits']['suite_wall_seconds']:
            raise RuntimeError('Measured pilot estimate exceeds the frozen wall cap; do not begin the main matrix')
        write_json(study/'freeze.json',{'plan_hash':digest(plan),'source_hash':source,'pilot_run_ids':[r['run_id'] for r in rows],
            'main_wall_seconds_estimate':estimate,'estimate_note':'measured pilots, 30% training and 2x validation allowances; two concurrent one-thread processes; outcomes can change cost',
            'limits':plan['limits'],'frozen_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint_selection':'monitor success, SPL, earliest step',
            'candidate_selection':'complete validation success, SPL, lower collision, candidate ID',
            'confirmation_seed_count':3,'preliminary':True})
        status['phase']='search';save()
        stage_jobs([(run_config(plan,f,'search',plan['development_seed'],c,root),c)
                    for c in ('c00','c01','c02','c03') for f in plan['algorithms']])
        selected={f:choose_candidate(rows,f,plan) for f in plan['algorithms']}
        write_json(study/'selected_configurations.json',{'selected':selected,'frozen_before_confirmation':True,
            'selection_uses':'one development training seed, 100k budget, 928 development-validation pairs',
            'baseline_retention_rule':'c00 winner reuses control confirmations, without duplicate training replicates'})
        print(f'Frozen development choices: {selected}',flush=True)
        status['phase']='confirmation';save()
        confirmations=[]
        for family in plan['algorithms']:
            for seed in plan['confirmation_seeds']:
                confirmations.append((run_config(plan,family,'control',seed,root=root),'c00'))
                if selected[family]!='c00':
                    confirmations.append((run_config(plan,family,'tuned',seed,selected[family],root),selected[family]))
        stage_jobs(confirmations)
        status['phase']='audit';save()
        audits=[]
        for row in rows:
            audits.append(audit_run(row,jobs[row['run_id']]['config'],source,root))
            print(f"Audited {row['run_id']}",flush=True)
            if time.perf_counter()-started>plan['limits']['suite_wall_seconds']:
                raise TimeoutError('Neural suite reached its wall cap during verification')
        summary=summarize(rows,selected,plan,load_task().signature)
        summary.update(search_results=[r for r in rows if r['stage']=='search'],
                       actual_training_interactions=sum(r['pilot']['training_interactions'] for r in rows),
                       unique_training_runs=len(rows),workers=2)
        figures(study,summary,root)
        write_json(study/'summary.json',summary)
        protected()
        artifact_bytes=sum(p.stat().st_size for folder in [study,*[root/'artifacts/runs'/r['run_id'] for r in rows]]
                           for p in folder.rglob('*') if p.is_file())
        if artifact_bytes>plan['limits']['new_artifact_bytes'] or time.perf_counter()-started>plan['limits']['suite_wall_seconds']:
            raise RuntimeError('Neural suite exceeded its predeclared storage or wall-time limit')
        write_json(study/'verification.json',{'status':'passed','audits':audits,
            'historical_files_verified':len(historical),'historical_inputs_unchanged':True,
            'new_final_test_policy_executions':0,'original_final_selection_changed':False,
            'validation_actions_replayed':sum(a['validation_actions_replayed'] for a in audits),
            'recorded_training_actions_replayed':sum(a['recorded_training_actions_replayed'] for a in audits),
            'output_hashes':{name:sha(study/name) for name in ('summary.json','paired_neural_tuning.png','neural_tuning_learning.png')},
            'new_artifact_bytes':artifact_bytes})
        status.update(status='verified',phase='complete',selected_candidates=selected,new_artifact_bytes=artifact_bytes)
        save();print(status,flush=True)
    except BaseException as exc:
        for identity,(process,log,_) in active.items():
            process.terminate();process.wait(timeout=10);log.close()
            path=root/'artifacts/runs'/identity/'manifest.json'
            if path.exists():
                manifest=read_json(path)
                if manifest['status']=='running':
                    manifest.update(status='interrupted',error='Study worker stopped after a suite failure',resume_supported=False)
                    write_json(path,manifest)
        active.clear()
        status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'incomplete',error=f'{type(exc).__name__}: {exc}')
        save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    worker(args.worker) if args.worker else run()
