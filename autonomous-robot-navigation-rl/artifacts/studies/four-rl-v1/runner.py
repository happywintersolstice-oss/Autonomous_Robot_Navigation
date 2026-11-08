"""Add PPO without rewriting the original Part A or frozen final study."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
from time import perf_counter

import yaml

from robot_navigation.core import ROOT,digest,load_task,read_json,write_json
from robot_navigation.part_a import train_part_a
from robot_navigation.release import sha
from robot_navigation.workflow import source_hash


def historical_files():
    paths=[ROOT/'artifacts/studies/part-a-v1'/n for n in ('run_results.json','summary.json','shortlist.json','references.json')]
    paths.extend(ROOT/'data/scenarios'/n for n in ('train.json','validation.json','monitoring.json'))
    paths.append(ROOT/'configs/part_a.yaml')
    for algorithm in ('q-learning','sarsa','dqn'):
        for seed in (7,19,43,67,91):
            run=ROOT/f'artifacts/runs/m3-{algorithm}-seed{seed}'
            paths.extend(run/n for n in ('manifest.json','configuration.json','validation.json','pilot.json','training.csv',f'{algorithm.replace("-","_")}_episodes.json'))
            checkpoint=read_json(run/'manifest.json')['viewer_checkpoint']
            paths.extend((run/checkpoint,(run/checkpoint).with_suffix('.json')))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('pilot','run'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ppo_comparison.yaml')
    args=parser.parse_args()
    plan=yaml.safe_load(args.config.read_text('utf-8'))
    # Fail before creating a study/run when the approved neural runtime cannot load.
    from robot_navigation.ppo import PPOPolicy
    for key in ('suite_id','run_prefix'):
        if Path(plan[key]).name!=plan[key] or plan[key] in ('.','..'):
            raise ValueError('Study/run output must be a single directory name')
    study=ROOT/'artifacts/studies'/plan['suite_id']
    if args.command=='pilot':
        study.mkdir(parents=True,exist_ok=False)
        write_json(study/'plan.json',plan)
        shutil.copy2(ROOT/'docs/ppo_comparison_protocol.md',study/'protocol.md')
        shutil.copy2(__file__,study/'runner.py')
        write_json(study/'historical_hashes.json',historical_files())
        write_json(study/'pretraining.json',{'signature':load_task().signature,'plan_hash':digest(plan),
            'source_hash':source_hash(ROOT),'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'post-release exploratory development validation; no final-test execution',
            'new_configuration_candidates':1,'owner_requested_fourth_algorithm':'ppo'})
        config={**plan['training'],'seed':plan['pilot_seed'],'interactions':plan['pilot_interactions'],
                'run_id':f"{plan['run_prefix']}-ppo-pilot-seed{plan['pilot_seed']}"}
        started=perf_counter()
        directory,summaries,pilot=train_part_a(config)
        manifest=read_json(directory/'manifest.json')
        estimated_run=(pilot['training_seconds']*plan['training']['interactions']/config['interactions']+
            manifest['monitoring_seconds']*plan['training']['interactions']/config['interactions']+
            summaries['ppo']['evaluation_seconds'])
        result={'run_id':directory.name,'wall_seconds':perf_counter()-started,'pilot':pilot,
            'validation':summaries['ppo'],'estimated_main_matrix_seconds':len(plan['seeds'])*estimated_run,
            'estimate_basis':'measured pilot training/monitoring extrapolation plus one full validation per seed',
            'settings_changed_after_pilot':False}
        write_json(study/'pilot.json',result)
        print(result,flush=True)
        return
    if read_json(study/'plan.json')!=plan:
        raise ValueError('PPO configuration changed after pretraining registration')
    if historical_files()!=read_json(study/'historical_hashes.json'):
        raise ValueError('Historical comparison inputs changed')
    pilot=read_json(study/'pilot.json')
    if pilot['estimated_main_matrix_seconds']>plan['limits']['suite_wall_seconds']:
        raise ValueError('Pilot estimate exceeds the declared suite wall limit')
    if (study/'manifest.json').exists():
        raise FileExistsError('This matrix already started; use a new registered study, no silent overwrite')
    started=perf_counter();results=[]
    manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
              'signature':load_task().signature,'plan_hash':digest(plan),'source_hash':source_hash(ROOT),
              'final_test_evaluated':False,'exploratory_post_release':True,'completed_runs':[]}
    write_json(study/'manifest.json',manifest)
    try:
        for seed in plan['seeds']:
            if perf_counter()-started>plan['limits']['suite_wall_seconds']:
                raise TimeoutError('PPO matrix wall-time limit reached')
            config={**plan['training'],'seed':seed,'run_id':f"{plan['run_prefix']}-ppo-seed{seed}"}
            print(f"Starting {config['run_id']}",flush=True)
            directory,summaries,pilot=train_part_a(config)
            results.append({'run_id':directory.name,'algorithm':'ppo','seed':seed,'validation':summaries['ppo'],
                'pilot':pilot,'selected_interactions':read_json(directory/'manifest.json')['selected_interactions']})
            manifest['completed_runs'].append(directory.name)
            write_json(study/'run_results.json',results)
            write_json(study/'manifest.json',manifest)
            print(f"Finished {directory.name}: {summaries['ppo']['counts']}",flush=True)
            directories=[ROOT/'artifacts/runs'/name for name in manifest['completed_runs']]
            directories.extend((ROOT/'artifacts/runs'/f"{plan['run_prefix']}-ppo-pilot-seed{plan['pilot_seed']}",study))
            size=sum(p.stat().st_size for r in directories
                     for p in r.rglob('*') if p.is_file())
            if size>plan['limits']['new_artifact_bytes']:
                raise RuntimeError('PPO artifact storage limit exceeded')
            if perf_counter()-started>plan['limits']['suite_wall_seconds']:
                raise TimeoutError('PPO matrix wall-time limit exceeded')
        if historical_files()!=read_json(study/'historical_hashes.json'):
            raise ValueError('Historical artifacts changed during the PPO matrix')
        manifest.update(status='complete',wall_seconds=perf_counter()-started,
                        training_interactions=sum(r['pilot']['training_interactions'] for r in results),
                        new_run_artifact_bytes=size,finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}',wall_seconds=perf_counter()-started)
        write_json(study/'manifest.json',manifest)
        raise
    write_json(study/'manifest.json',manifest)
    print('PPO matrix complete. Analyze the four-algorithm validation comparison.',flush=True)


if __name__=='__main__':
    main()
