"""One-shot Phase 5F registration/run and read-only verify/report CLI."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import yaml

from .core import ROOT,read_json,write_json
from .release import sha
from .partial_contract import configuration
from .partial_features_scaled import feature_signature,SCALED
from .partial_training_instrumentation import check_compatibility,library_contract
from .partial_instrumentation_trace import checksum,quota_bytes
from .partial_phase5e import installed_audit,test_counts
from .partial_phase5f_io import STUDY,BASE,ALGORITHMS,location,git,inputs,training_cases,preserve

BRANCH='feature/phase5f-instrumented-learning-pilot-v1'


def settings(root=ROOT):
    c=yaml.safe_load((root/'configs/partial_phase5f_v1.yaml').read_text())
    expected={'study_id':STUDY,'base_commit':BASE,'branch':BRANCH,'algorithms':list(ALGORITHMS),'primary_seed':23121,
        'arm':'B1-relative-x4','feature_version':SCALED,'per_algorithm_actions':500,'total_training_actions':1000,
        'per_algorithm_wall_seconds':900,'suite_wall_seconds':1800,'artifact_byte_ceiling':536870912,
        'research_evaluation_actions':0,'final_unseen_evaluations':0,'new_synthetic_transition_cap':0,'new_synthetic_optimizer_cap':0,
        'expected_optimizer_steps':{'dqn':100,'ppo':50},'enabled_by_default':False,'absolute_tolerance':2e-5,'relative_tolerance':2e-6}
    if c!=expected:raise ValueError('Frozen F envelope mismatch')
    return c


def branch_gate(root=ROOT):
    if git(root,'rev-parse','HEAD').strip()!=BASE or git(root,'branch','--show-current').strip()!=BRANCH:
        raise ValueError('Registration/run require exact published E parent on new F branch')


def tests(stage,root=ROOT):
    """New tests perform no environment/optimizer operations; preserve every receipt."""
    folder=root/'.runtime/phase5f-tests';folder.mkdir(exist_ok=True)
    n=len(list(folder.glob('attempt-*.json')))+1;prefix=folder/f'attempt-{n:04d}'
    source=inputs(root);started=perf_counter()
    command=[sys.executable,'-m','pytest','developer/tests/test_partial_phase5f.py','-q','-o','markers=pilot: read-only integration on existing research evidence','-m',
        'pilot' if stage=='post' else 'not pilot','--junitxml='+str(prefix.with_suffix('.xml'))]
    env=dict(os.environ,PYTHONPATH=str(root/'src'))
    with prefix.with_suffix('.stdout.log').open('x') as out,prefix.with_suffix('.stderr.log').open('x') as err:
        p=subprocess.run(command,cwd=root,env=env,stdout=out,stderr=err)
    receipt={'stage':stage,'exit_code':p.returncode,'source_inputs':source,'source_digest':checksum(source),
        'test_counts':test_counts(prefix.with_suffix('.xml')),'wall_seconds':perf_counter()-started,
        'new_synthetic_transitions':0,'new_synthetic_optimizer_steps':0,'physical_operation_cap':{'transitions':0,'optimizer':0},
        'scope':'Pure budget/config/hash tests, or read-only integration on actual pilot records; no learner/env construction',
        'command':command}
    write_json(prefix.with_suffix('.json'),receipt)
    print(json.dumps(receipt,indent=2))
    if p.returncode:raise RuntimeError('Targeted tests failed; retained attempt '+str(prefix))
    return receipt


def register(root=ROOT):
    branch_gate(root);c=settings(root);check_compatibility();preservation=preserve(root)
    source=inputs(root);digest=checksum(source)
    testpaths=sorted((root/'.runtime/phase5f-tests').glob('attempt-*.json'))
    good=[p for p in testpaths if read_json(p)['stage']=='pre' and read_json(p)['exit_code']==0 and read_json(p)['source_digest']==digest]
    if not good:raise ValueError('Current frozen sources need passing pre-tests before registration')
    legacy=root/'.runtime/phase5f-parent-regressions';lreceipt=read_json(legacy/'receipt.json')
    counts=test_counts(legacy/'regressions.xml')
    if lreceipt['exit_code']!=0 or counts!={'tests':464,'failures':0,'errors':0,'skipped':0}:raise ValueError('Full unchanged parent regressions prerequisite')
    if read_json(legacy/'plan.json')['driver_source_sha256']!=sha(root/'developer/commands/phase5f_parent_regressions.py'):
        raise ValueError('Legacy regression driver changed after execution')
    preflight=read_json(root/'.runtime/phase5f-parent-preflight.json')
    if preflight['exact_parent']!=BASE or preflight['scoped_read_only_verification']['status']!='passed':raise ValueError('Parent read-only verification prerequisite')
    cases=training_cases(root);model=yaml.safe_load((root/'configs/partial_feasibility_v1.yaml').read_text())['model']
    expected_model={'discount':.99,'linear_learning_rate':.1,'neural_learning_rate':.0003,'net_arch':[32,32],
        'epsilon_start':1.,'epsilon_end':.1,'buffer_size':10000,'learning_starts':100,'batch_size':50,'train_freq':4,
        'target_update_interval':250,'n_steps':100,'n_epochs':5}
    if model!=expected_model:raise ValueError('Original model configuration mismatch')
    # JSON array payloads plus all 150 gradients/metadata, conservatively 400 MiB.
    # DQN 100*2*50*409 floats and PPO 500*2*409 floats dominate; reserve
    # 16 bytes per numeric JSON value and 300 MiB for all other tensors/metadata.
    estimate=(100*2*50*409+500*2*409)*16+300*1024**2
    if estimate>=c['artifact_byte_ceiling']:raise ValueError('Expected complete trace above quota')
    folder=location(root);folder.mkdir(exist_ok=False)
    (folder/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    for name in ('base','original'):shutil.copyfile(root/'.runtime'/f'phase5f-{name}-inventory.json',folder/f'{name}-inventory.json')
    shutil.copyfile(root/'.runtime/phase5f-parent-preflight.json',folder/'parent-preflight.json')
    dest=folder/'legacy-regressions';dest.mkdir()
    for name in ('plan.json','receipt.json','parent-process-operations.json','child-mechanics.jsonl','regressions.xml','stdout.log','stderr.log'):
        shutil.copyfile(legacy/name,dest/name)
    testdir=folder/'tests';testdir.mkdir()
    for p in (root/'.runtime/phase5f-tests').glob('attempt-*'):shutil.copyfile(p,testdir/p.name)
    audit=installed_audit();write_json(folder/'installed-library-audit.json',audit)
    manifest={'study_id':STUDY,'registered_utc':datetime.now(timezone.utc).isoformat(),'parent_commit':BASE,
        'parent_tree':git(root,'rev-parse',BASE+'^{tree}').strip(),'branch':BRANCH,'settings':c,
        'source_inputs':source,'source_digest':digest,'library_contract':library_contract(),'library_audit_sha256':sha(folder/'installed-library-audit.json'),
        'hardware':{'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'python':sys.version,'device':'cpu','n_envs':1,'torch_threads':1},
        'model_config':model,'model_config_source_sha256':sha(root/'configs/partial_feasibility_v1.yaml'),
        'task_feature_signature':feature_signature(configuration(root),SCALED),'case_ids':[s['id'] for s in cases],
        'case_sha256':{s['id']:checksum(s) for s in cases},'case_source':'Phase5B training_cases array only; original uniform reset sampler',
        'case_source_sha256':sha(root/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/manifest.json'),
        'source_access':'Only training_cases array decoded; historical/final artifacts byte-hashed, not parsed for science',
        'order':list(ALGORITHMS),'expected_uncompressed_upper_bytes':estimate,'new_synthetic_caps':{'transitions':0,'optimizer':0},
        'legacy_counts':counts,'pre_tests_sha256':sha(good[-1]),'preservation':preservation,
        'stop_policy':'one attempt; no retries; failure preserved; stop after F',
        'measurement_gates':['action_ledger','train_split','trace_chain','consumed_tensor_lineage','numeric_DQN','numeric_PPO_GAE',
            'optimizer_and_model_continuity','independent_call_counts','observer_closed','source_library_hashes','regressions','no_evaluation']}
    write_json(folder/'manifest.json',manifest)
    (folder/'manifest.sha256').write_text(sha(folder/'manifest.json')+'\n',encoding='ascii')
    write_json(folder/'registration-receipt.json',{'status':'sealed_before_research','manifest_sha256':sha(folder/'manifest.json'),
        'scientific_actions':0,'synthetic_transitions':0,'synthetic_optimizer_steps':0})
    return {'status':'registered','folder':str(folder),'source_digest':digest,'expected_upper_bytes':estimate}


def checked(root=ROOT):
    folder=location(root);manifest=read_json(folder/'manifest.json')
    if sha(folder/'manifest.json')!=(folder/'manifest.sha256').read_text().strip():raise ValueError('Registration seal mismatch')
    if inputs(root)!=manifest['source_inputs'] or checksum(inputs(root))!=manifest['source_digest']:raise ValueError('Frozen source changed')
    settings(root);check_compatibility();preserve(root)
    require_contract=library_contract()==manifest['library_contract']
    if not require_contract:raise ValueError('Library contract changed')
    audit=read_json(folder/'installed-library-audit.json')
    if sha(folder/'installed-library-audit.json')!=manifest['library_audit_sha256']:raise ValueError('Library audit seal')
    for name,h in audit['dependency_file_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('Installed runtime binary/source changed: '+name)
    cases=training_cases(root)
    if {s['id']:checksum(s) for s in cases}!=manifest['case_sha256']:raise ValueError('Training case source changed')
    manifest['manifest_sha256']=sha(folder/'manifest.json')
    return folder,manifest,cases


def run(root=ROOT):
    branch_gate(root);folder,m,cases=checked(root)
    with (folder/'run.lock').open('x') as stream:stream.write(json.dumps({'manifest_sha256':m['manifest_sha256'],'attempt':1,'retries':0}))
    try:
        from .partial_phase5f_live import run_models
        result=run_models(folder,m,cases,m['model_config']);write_json(folder/'run.json',result)
        return result
    except BaseException as exc:
        write_json(folder/'failure.json',{'verdict':'INSTRUMENTATION_BLOCKED' if 'UNSUPPORTED' in str(exc) else 'PILOT_INCOMPLETE',
            'exception':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'retry_authorized':False})
        raise


def verify(root=ROOT,require_post=True):
    folder,m,_=checked(root)
    if (folder/'failure.json').exists():return read_json(folder/'failure.json')
    from .partial_phase5f_verify import verify_records
    result=verify_records(folder,m)
    if require_post:
        paths=sorted((folder/'tests').glob('attempt-*.json'))
        good=[p for p in paths if read_json(p)['stage']=='post' and read_json(p)['exit_code']==0 and read_json(p)['source_digest']==m['source_digest']]
        if not good:raise ValueError('Post-pilot read-only targeted tests missing')
        result['post_tests']=read_json(good[-1])['test_counts']
    if (folder/'output-hashes.json').exists():
        for n,h in read_json(folder/'output-hashes.json')['files'].items():
            if sha(folder/n)!=h:raise ValueError('Output hash changed: '+n)
    if quota_bytes([folder])>=536870912:raise ValueError('Aggregate artifact quota')
    result['preservation']=preserve(root);result['legacy_tests']=m['legacy_counts']
    result['verdict']='INSTRUMENTED_LEARNING_TRACES_VALIDATED'
    return result


def report(root=ROOT):
    from .partial_phase5f_report import render
    folder=location(root)
    # Copy every test attempt, including genuine failures, without rewriting science.
    for p in (root/'.runtime/phase5f-tests').glob('attempt-*'):
        target=folder/'tests'/p.name
        if target.exists() and sha(target)!=sha(p):raise ValueError('Cannot replace test history')
        if not target.exists():shutil.copyfile(p,target)
    try:result=verify(root)
    except Exception as exc:
        result={'verdict':'PILOT_INCOMPLETE','error':str(exc),'exception':type(exc).__name__}
    write_json(folder/'verification.json',result);render(root,result)
    write_json(folder/'preservation.json',preserve(root))
    write_json(folder/'output-hashes.json',{'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name not in ('output-hashes.json','verification.json','preservation.json')},
        'reports':{p.relative_to(root).as_posix():sha(p) for p in (root/'docs/partial_phase5f_results.md',root/'docs/partial_phase5f_diagnosis.md')}})
    return {'verdict':result['verdict'],'artifact_bytes':quota_bytes([folder])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['test-pre','register','run','test-post','verify','report']);a=p.parse_args()
    if a.command.startswith('test-'):result=tests('pre' if a.command=='test-pre' else 'post')
    else:result=globals()[a.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
