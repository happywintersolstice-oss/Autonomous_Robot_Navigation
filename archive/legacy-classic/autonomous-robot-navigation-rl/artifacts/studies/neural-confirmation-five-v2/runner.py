"""Run the predeclared two-seed addition and publish combined five-seed evidence."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from robot_navigation.core import ROOT, digest, load_task, read_json, write_json
from robot_navigation.neural_confirmation import STUDY, combined_summary, prepare
from robot_navigation.neural_tuning import audit_run, figures
from robot_navigation.release import sha
from robot_navigation.workflow import source_hash


def run():
    plan=yaml.safe_load((ROOT/'configs/neural_confirmation.yaml').read_text(encoding='utf-8'))
    previous,jobs,aggregate_plan,engine_hashes=prepare(plan,ROOT)
    reused=[]
    recovery=ROOT/'artifacts/studies'/plan['recovery_source_study']
    old_status=read_json(recovery/'manifest.json')
    if old_status['status']!='incomplete' or set(old_status['completed_runs'])!=set(plan['reused_completed_runs']):
        raise ValueError('Recovery does not match the preserved incomplete attempt')
    for identity in plan['reused_completed_runs']:
        row=read_json(recovery/'results'/f'{identity}.json')
        expected=next(job['config'] for job in jobs if job['config']['run_id']==identity)
        directory=ROOT/'artifacts/runs'/identity
        if read_json(directory/'configuration.json')['training']!=expected:
            raise ValueError('Preserved completed DQN configuration differs')
        if read_json(directory/'manifest.json')['status']!='complete':
            raise ValueError('Recovery tried to reuse an incomplete training run')
        reused.append(row)
    jobs=[job for job in jobs if job['config']['run_id'] not in plan['reused_completed_runs']]
    if len(jobs)!=plan['limits']['max_new_runs'] or sum(job['config']['interactions'] for job in jobs)!=plan['limits']['max_new_training_interactions']:
        raise ValueError('Recovery work does not reconcile with its cap')
    jobs.sort(key=lambda job:(job['config']['algorithm'],job['config']['seed'],job['config']['stage']))
    study=ROOT/'artifacts/studies'/STUDY
    if study.exists() or any((ROOT/'artifacts/runs'/job['config']['run_id']).exists() for job in jobs):
        raise FileExistsError('Extension needs fresh output identities; overwriting/resume is unsupported')
    allowed=set(plan['reused_completed_runs'])|{'m9-control-ppo-seed7607','m9-tuned-ppo-seed7607'}
    if any(read_json(path).get('training_seed') in plan['additional_seeds'] and path.parent.name not in allowed
           for path in (ROOT/'artifacts/runs').glob('*/manifest.json')):
        raise ValueError('Unexpected previous attempt for an additional seed')
    estimate=sum(1.35*row['costs']['training_seconds']+2*row['costs']['monitoring_seconds']
                 +2*row['costs']['validation_seconds'] for row in previous if row['algorithm']=='ppo')/3*2/plan['workers']+180
    if estimate>plan['limits']['suite_wall_seconds']:
        raise RuntimeError('Prior measured costs exceed the declared extension wall cap')
    print('Hashing existing evidence before the extension',flush=True)
    historical={path.relative_to(ROOT).as_posix():sha(path) for folder in ('artifacts','data')
                for path in (ROOT/folder).rglob('*') if path.is_file()}
    study.mkdir(parents=True)
    source=source_hash(ROOT)
    write_json(study/'historical_hashes.json',historical)
    write_json(study/'extension_plan.json',plan)
    write_json(study/'plan.json',aggregate_plan)
    shutil.copy2(Path(__file__),study/'runner.py')
    shutil.copy2(ROOT/'docs/neural_confirmation_protocol.md',study/'protocol.md')
    prior_study=ROOT/'artifacts/studies'/plan['source_study']
    shutil.copy2(prior_study/'selected_configurations.json',study/'selected_configurations.json')
    write_json(study/'freeze.json',{'plan_hash':digest(plan),'source_hash':source,'engine_hashes':engine_hashes,
        'frozen_utc':datetime.now(timezone.utc).isoformat(),'runner_sha256':sha(Path(__file__)),
        'worker_sha256':sha(ROOT/'scripts/run_neural_tuning.py'),
        'protocol_sha256':sha(study/'protocol.md'),'jobs':jobs,'main_wall_seconds_estimate':estimate,
        'original_results_already_available':True,'reused_completed_runs':plan['reused_completed_runs'],
        'json_io_repair_before_restarted_ppo':True,'new_final_test_policy_executions':0})
    started=time.perf_counter()
    additions=list(reused)
    active={}
    pending=list(jobs)
    status={'status':'running','phase':'additional seeds','completed_runs':[],'running_runs':[]}

    def save():
        status.update(elapsed_seconds=time.perf_counter()-started,
                      completed_runs=[row['run_id'] for row in additions],running_runs=list(active),
                      added_seed_training_interactions=sum(row['pilot']['training_interactions'] for row in additions),
                      newly_executed_training_interactions=sum(row['pilot']['training_interactions'] for row in additions if row['run_id'] not in plan['reused_completed_runs']))
        write_json(study/'additional_run_results.json',additions)
        write_json(study/'manifest.json',status)

    def protected():
        if any(not (ROOT/path).is_file() or sha(ROOT/path)!=value for path,value in historical.items()):
            raise ValueError('Historical evidence changed')

    try:
        save()
        while pending or active:
            if time.perf_counter()-started>plan['limits']['suite_wall_seconds']:
                raise TimeoutError('Extension reached the suite wall cap')
            while pending and len(active)<plan['workers']:
                if source_hash(ROOT)!=source:
                    raise ValueError('Source changed during added-seed training')
                job=pending.pop(0)
                identity=job['config']['run_id']
                path=study/'jobs'/f'{identity}.json'
                write_json(path,job)
                (study/'logs').mkdir(exist_ok=True)
                log=(study/'logs'/f'{identity}.log').open('x',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'scripts/run_neural_tuning.py'),
                                          '--worker',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active[identity]=(process,log,time.perf_counter())
                print(f'Started {identity}',flush=True)
                save()
            for identity,(process,log,run_started) in list(active.items()):
                if time.perf_counter()-run_started>plan['limits']['run_wall_seconds']:
                    raise TimeoutError(f'{identity} exceeded its run wall cap')
                code=process.poll()
                if code is None:
                    continue
                log.close()
                del active[identity]
                if code:
                    raise RuntimeError(f'{identity} failed with exit {code}; inspect its log')
                row=read_json(study/'results'/f'{identity}.json')
                additions.append(row)
                print(f"Completed {identity}. {row['validation']['counts']}",flush=True)
                save()
            time.sleep(.5)
        protected()
        status['phase']='audit'
        save()
        audits=[]
        for row in previous+additions:
            path=ROOT/'artifacts/runs'/row['run_id']
            config=read_json(path/'configuration.json')['training']
            expected_source=read_json(path/'manifest.json')['source_hash']
            if row in additions and row['run_id'] not in plan['reused_completed_runs']:
                expected=next(job['config'] for job in jobs if job['config']['run_id']==row['run_id'])
                if config!=expected or expected_source!=source:
                    raise ValueError('Added run differs from its frozen job')
            elif row in additions:
                if expected_source!=read_json(recovery/'freeze.json')['source_hash']:
                    raise ValueError('Reused DQN source differs from the preserved attempt')
            else:
                frozen=read_json(ROOT/'artifacts/studies'/plan['source_study']/'jobs'/f"{row['run_id']}.json")
                if config!=frozen['config']:
                    raise ValueError('Original confirmation configuration differs from its frozen job')
            audits.append(audit_run(row,config,expected_source,ROOT))
            print(f"Audited {row['run_id']}",flush=True)
            if time.perf_counter()-started>plan['limits']['suite_wall_seconds']:
                raise TimeoutError('Extension reached its wall cap during audit')
        summary=combined_summary(previous,additions,plan,aggregate_plan,load_task().signature)
        summary.update(search_results=read_json(prior_study/'summary.json')['search_results'],
                       unique_confirmation_runs=len(previous+additions),new_training_runs=len(additions),workers=2,
                       reused_completed_addition_runs=len(reused),newly_executed_runs=len(jobs),
                       newly_executed_training_interactions=sum(job['config']['interactions'] for job in jobs),
                       interrupted_ppo_attempts_preserved=True,json_io_repair_before_restarted_ppo=True)
        figures(study,summary,ROOT)
        write_json(study/'summary.json',summary)
        write_json(study/'run_results.json',previous+additions)
        protected()
        size=sum(path.stat().st_size for folder in [study,*[ROOT/'artifacts/runs'/job['config']['run_id'] for job in jobs]]
                 for path in folder.rglob('*') if path.is_file())
        if size>plan['limits']['new_artifact_bytes'] or time.perf_counter()-started>plan['limits']['suite_wall_seconds']:
            raise RuntimeError('Extension exceeded its declared storage or wall cap')
        write_json(study/'verification.json',{'status':'passed','audits':audits,
            'historical_files_verified':len(historical),'historical_inputs_unchanged':True,
            'new_final_test_policy_executions':0,'original_final_selection_changed':False,
            'validation_actions_replayed':sum(a['validation_actions_replayed'] for a in audits),
            'recorded_training_actions_replayed':sum(a['recorded_training_actions_replayed'] for a in audits),
            'output_hashes':{name:sha(study/name) for name in ('summary.json','paired_neural_tuning.png','neural_tuning_learning.png','plan.json')},
            'new_artifact_bytes':size})
        status.update(status='verified',phase='complete',new_artifact_bytes=size)
        save()
        print(status,flush=True)
    except BaseException as exc:
        for identity,(process,log,_) in active.items():
            process.terminate()
            process.wait(timeout=10)
            log.close()
            path=ROOT/'artifacts/runs'/identity/'manifest.json'
            if path.exists():
                manifest=read_json(path)
                if manifest['status']=='running':
                    manifest.update(status='interrupted',error='Extension worker stopped after suite failure',resume_supported=False)
                    write_json(path,manifest)
        active.clear()
        status.update(status='incomplete',error=f'{type(exc).__name__}. {exc}')
        save()
        raise


if __name__=='__main__':
    run()
