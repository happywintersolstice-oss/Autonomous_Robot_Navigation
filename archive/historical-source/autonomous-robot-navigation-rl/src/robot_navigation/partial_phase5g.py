"""Phase 5G: register, one-shot run, closed-record verify and data-only report."""
import argparse,gzip,json,os,platform,shutil,subprocess,sys,traceback
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import yaml

from .core import ROOT,read_json,write_json
from .release import sha
from .partial_contract import configuration
from .partial_features_scaled import feature_signature,SCALED
from .partial_training_instrumentation import check_compatibility,library_contract
from .partial_instrumentation_trace import checksum,quota_bytes
from .partial_phase5e import installed_audit,test_counts
from .partial_phase5g_io import BASE,BRANCH,STUDY,CHECKPOINTS,ALGORITHMS,UPDATES,location,git,inputs,cases,preserve


def settings(root=ROOT):
    c=yaml.safe_load((root/'configs/partial_phase5g_v1.yaml').read_text())
    expected={'study_id':STUDY,'base_commit':BASE,'branch':BRANCH,'algorithms':list(ALGORITHMS),'seed':23121,
        'arm':'B1-relative-x4','feature_version':SCALED,'per_algorithm_actions':2000,'training_action_cap':4000,'checkpoints':list(CHECKPOINTS),
        'expected_optimizer_steps':UPDATES,'assessment_action_cap':32000,'assessment_case_steps':200,'assessment_seed':0,
        'development_panels':{'original':4,'expanded':16},'synthetic_transition_cap':0,'synthetic_optimizer_cap':0,
        'arm_wall_seconds':1800,'training_wall_seconds':3600,'artifact_byte_cap':536870912,'absolute_tolerance':2e-5,'relative_tolerance':2e-6,
        'readiness':{'expanded_gain':4,'expanded_successes':12,'successes_per_direction':2},'enabled_by_default':False,'final_unseen_accesses':0}
    if c!=expected:raise ValueError('Registered Phase 5G scope changed')
    return c


def branch_gate(root=ROOT):
    if git(root,'rev-parse','HEAD').strip()!=BASE or git(root,'branch','--show-current').strip()!=BRANCH:raise ValueError('Wrong exact parent/new branch')


def storage_plan(root=ROOT):
    """Offline F trace sizing; no sampler, model, environment or extra training."""
    base=root/'artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1';arms={}
    for algorithm in ALGORITHMS:
        maximum=defaultdict(int);total=0;compressed=0
        for p in (base/algorithm/'trace').glob('*.jsonl.gz'):
            compressed+=p.stat().st_size
            with gzip.open(p,'rb') as stream:
                for line in stream:
                    row=json.loads(line)['record'];total+=len(line);maximum[row['event']]=max(maximum[row['event']],len(line))
        counts={'observer_opened':1,'observer_closed':1,'reset':2001,'environment_transition':2000,'phase5f_replay_insert':2000,
            'target_sync':8,'replay_sample':475,'dqn_loss':475,'optimizer':475} if algorithm=='dqn' else {
            'observer_opened':1,'observer_closed':1,'reset':2001,'environment_transition':2000,'ppo_behavior':2000,'rollout_add':2000,
            'gae_input':20,'gae_result':20,'ppo_permutation':100,'rollout_sample':200,'ppo_loss':200,'optimizer':200,'ppo_timeout_bootstrap':10}
        uncompressed=int(sum(n*maximum.get(e,4096) for e,n in counts.items())*1.25)
        arms[algorithm]={'F_uncompressed_bytes':total,'F_compressed_bytes':compressed,'F_event_max_bytes':dict(maximum),
            'G_max_event_counts':counts,'G_uncompressed_padded_estimate':uncompressed,
            'G_compressed_estimate_twice_observed_ratio':int(uncompressed*compressed/total*2)}
        if uncompressed>=536870912:raise ValueError('Expected arm exceeds inherited uncompressed transport reservation cap')
    estimate=sum(a['G_compressed_estimate_twice_observed_ratio'] for a in arms.values())+(64+128+16)*1024**2
    if estimate>=536870912:raise ValueError('Expected complete logs/checkpoints/assessment exceed study quota; stop before research')
    return {'arms':arms,'checkpoint_reservation_bytes':64*1024**2,'assessment_reservation_bytes':128*1024**2,
        'metadata_test_reservation_bytes':16*1024**2,'aggregate_padded_estimate':estimate,'hard_cap':536870912,
        'scope':'Conservative projected storage, not an absolute numerical-string worst case; actual aggregate and per-arm uncompressed guards remain active'}


def tests(stage,root=ROOT):
    folder=root/'.runtime/phase5g-tests';folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*.json')))+1;prefix=folder/f'attempt-{n:04d}'
    source=inputs(root);started=perf_counter()
    cmd=[sys.executable,'-m','pytest','developer/tests/test_partial_phase5g.py','-q','-o','markers=pilot: read-only G evidence integration',
        '-m','pilot' if stage=='post' else 'not pilot','--junitxml='+str(prefix.with_suffix('.xml'))]
    env=dict(os.environ,PYTHONPATH=str(root/'src'),GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0='*')
    with prefix.with_suffix('.stdout.log').open('x') as out,prefix.with_suffix('.stderr.log').open('x') as err:
        process=subprocess.run(cmd,cwd=root,env=env,stdout=out,stderr=err)
    receipt={'stage':stage,'exit_code':process.returncode,'source_inputs':source,'source_digest':checksum(source),
        'test_counts':test_counts(prefix.with_suffix('.xml')),'wall_seconds':perf_counter()-started,'new_synthetic_transitions':0,
        'new_synthetic_optimizer_steps':0,'declared_caps':{'transitions':0,'optimizer':0},'command':cmd,
        'scope':'Offline formulas, fake ledger/control checks or closed G records; no environment/model construction, training or scoring'}
    write_json(prefix.with_suffix('.json'),receipt)
    if process.returncode:raise RuntimeError('Targeted tests failed; retained '+str(prefix))
    return receipt


def register(root=ROOT):
    branch_gate(root);config=settings(root);check_compatibility();preservation=preserve(root)
    source=inputs(root);digest=checksum(source);attempts=sorted((root/'.runtime/phase5g-tests').glob('attempt-*.json'))
    passed=[p for p in attempts if read_json(p)['stage']=='pre' and read_json(p)['exit_code']==0 and read_json(p)['source_digest']==digest]
    if not passed:raise ValueError('Current frozen source needs passing offline pretests')
    legacy=root/'.runtime/phase5g-parent-regressions';receipt=read_json(legacy/'receipt.json');counts=test_counts(legacy/'regressions.xml')
    if receipt['exit_code'] or counts!={'tests':479,'failures':0,'errors':0,'skipped':0}:raise ValueError('Complete unchanged parent suite prerequisite')
    if read_json(legacy/'plan.json')['driver_source_sha256']!=sha(root/'developer/commands/phase5g_parent_regressions.py'):raise ValueError('Changed legacy regression driver')
    preflight=read_json(root/'.runtime/phase5g-parent-preflight.json')
    if preflight['exact_parent']!=BASE or preflight['scoped_parent_verification']['verdict']!='INSTRUMENTED_LEARNING_TRACES_VALIDATED':raise ValueError('Parent read-only verification failed')
    panel=cases(root);model=yaml.safe_load((root/'configs/partial_feasibility_v1.yaml').read_text())['model']
    if model!={'discount':.99,'linear_learning_rate':.1,'neural_learning_rate':.0003,'net_arch':[32,32],'epsilon_start':1.,'epsilon_end':.1,
        'buffer_size':10000,'learning_starts':100,'batch_size':50,'train_freq':4,'target_update_interval':250,'n_steps':100,'n_epochs':5}:raise ValueError('Historical model settings mismatch')
    storage=storage_plan(root);folder=location(root);folder.mkdir(exist_ok=False);(folder/'.gitattributes').write_bytes(b'* -text\n')
    for name in ('base','original'):shutil.copyfile(root/'.runtime'/f'phase5g-{name}-inventory.json',folder/f'{name}-inventory.json')
    for name in ('phase5g-parent-preflight.json','phase5g-preflight-git-ownership-failure.log'):
        shutil.copyfile(root/'.runtime'/name,folder/name)
    dest=folder/'legacy-regressions';dest.mkdir()
    for name in ('plan.json','receipt.json','parent-process-operations.json','child-mechanics.jsonl','regressions.xml','stdout.log','stderr.log'):shutil.copyfile(legacy/name,dest/name)
    (folder/'tests').mkdir()
    for p in (root/'.runtime/phase5g-tests').glob('attempt-*'):shutil.copyfile(p,folder/'tests'/p.name)
    audit=installed_audit();write_json(folder/'installed-library-audit.json',audit)
    manifest={'study_id':STUDY,'parent_commit':BASE,'parent_tree':git(root,'rev-parse',BASE+'^{tree}').strip(),'branch':BRANCH,
        'registered_utc':datetime.now(timezone.utc).isoformat(),'settings':config,'source_inputs':source,'source_digest':digest,
        'library_contract':library_contract(),'library_audit_sha256':sha(folder/'installed-library-audit.json'),'model_config':model,
        'model_source_sha256':sha(root/'configs/partial_feasibility_v1.yaml'),'task_feature_signature':feature_signature(configuration(root),SCALED),
        'case_ids':[c['id'] for c in panel['training_cases']],
        'panels':{k:[{'id':c['id'],'sha256':checksum(c)} for c in value] for k,value in panel.items()},
        'case_source_sha256':sha(root/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/manifest.json'),
        'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'device':'cpu','n_envs':1,'torch_threads':1},
        'order':list(ALGORITHMS),'checkpoints':list(CHECKPOINTS),'snapshot_boundary':'after completed original train return; initialized before reset at0',
        'schedule':'single fresh learn total_timesteps2000, exploration_fraction.8; not a continuation of F',
        'evaluation':'one detached child per algorithm after training models destroyed, all four checkpoint copies, fixed original4+expanded16, deterministic inference, seed0',
        'storage_plan':storage,'preservation':preservation,'legacy_counts':counts,'pretests_sha256':sha(passed[-1]),
        'new_synthetic_caps':{'transitions':0,'optimizer':0},'ledger_schema':'phase5f-journal-v1 reused, study identity in G namespace/manifest',
        'trace_schema':'partial-learning-live-v1 with unchanged F passive relay names','quality_gates':['G0','G1','G2','G3','G4','G5'],
        'stop':'No retries, extra arms/seeds, tuning, full training, final/unseen or Phase6; stop after G'}
    write_json(folder/'manifest.json',manifest);(folder/'manifest.sha256').write_bytes((sha(folder/'manifest.json')+'\n').encode())
    write_json(folder/'registration-receipt.json',{'status':'sealed_before_training_or_evaluation','manifest_sha256':sha(folder/'manifest.json'),
        'training_actions':0,'assessment_actions':0,'new_synthetic_transitions':0,'new_synthetic_optimizer_steps':0})
    return {'status':'registered','folder':str(folder),'source_digest':digest,'storage_estimate':storage['aggregate_padded_estimate']}


def checked(root=ROOT):
    folder=location(root);m=read_json(folder/'manifest.json')
    if sha(folder/'manifest.json')!=(folder/'manifest.sha256').read_text().strip():raise ValueError('Manifest seal mismatch')
    if inputs(root)!=m['source_inputs']:raise ValueError('Frozen G source modified')
    settings(root);check_compatibility();preserve(root)
    if library_contract()!=m['library_contract']:raise ValueError('Library mismatch')
    audit=read_json(folder/'installed-library-audit.json')
    if sha(folder/'installed-library-audit.json')!=m['library_audit_sha256']:raise ValueError('Library audit seal')
    for p,h in audit['dependency_file_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('Runtime dependency changed: '+p)
    panel=cases(root)
    if {k:[{'id':c['id'],'sha256':checksum(c)} for c in v] for k,v in panel.items()}!=m['panels']:raise ValueError('Cases/order changed')
    if sha(root/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/manifest.json')!=m['case_source_sha256']:raise ValueError('Source manifest changed')
    m['manifest_sha256']=sha(folder/'manifest.json');return folder,m,panel


def run(root=ROOT):
    branch_gate(root);folder,m,panel=checked(root)
    with (folder/'run.lock').open('x') as stream:stream.write(json.dumps({'attempt':1,'manifest_sha256':m['manifest_sha256'],'retries':0}))
    try:
        from .partial_phase5g_live import run_models
        training=run_models(folder,m,panel['training_cases']);write_json(folder/'training.json',training)
        from .partial_phase5g_assess import global_rng
        def isolated_state():
            return {'parent_global_rng':global_rng(),'training_ledger_sha256':sha(folder/'training-ledger.jsonl'),
                'training_receipt_sha256':sha(folder/'training.json'),
                'checkpoints':{p.relative_to(folder).as_posix():sha(p) for p in folder.glob('*/checkpoints/*/policy.zip')}}
        before=isolated_state();assessments=[];offset=0
        with (folder/'assessments.lock').open('x') as stream:stream.write(m['manifest_sha256'])
        for algorithm in ALGORITHMS:
            write_json(folder/f'assessment-launch-{algorithm}.json',{'algorithm':algorithm,'offset':offset,'manifest_sha256':m['manifest_sha256'],
                'training_receipt_sha256':sha(folder/'training.json'),'parent_process_id':os.getpid()})
            env=dict(os.environ,PYTHONPATH=str(root/'src'))
            with (folder/f'assessment-{algorithm}.stdout.log').open('x') as out,(folder/f'assessment-{algorithm}.stderr.log').open('x') as err:
                p=subprocess.run([sys.executable,'-m','robot_navigation.partial_phase5g_assess',algorithm,str(offset)],cwd=root,env=env,stdout=out,stderr=err)
            if p.returncode:raise RuntimeError('Detached assessment failed; no retry: '+algorithm)
            r=read_json(folder/algorithm/'development/receipt.json');offset+=r['evaluation_actions'];assessments.append(r)
        after=isolated_state();write_json(folder/'assessment-parent-isolation.json',{'before':before,'after':after,'unchanged':before==after})
        if before!=after:raise RuntimeError('Assessment changed parent state/evidence')
        result={'status':'completed','training_actions':4000,'actual_optimizer_steps':675,'assessment_actions':offset,'assessment_cases':160,
            'training_wall_seconds':training['wall_seconds'],'assessment_wall_seconds':sum(r['wall_seconds'] for r in assessments),
            'new_synthetic_transitions':0,'new_synthetic_optimizer_steps':0,'final_unseen_accesses':0}
        write_json(folder/'run.json',result);return result
    except BaseException as exc:
        write_json(folder/'failure.json',{'verdict':'PHASE5G_INCOMPLETE_OR_BLOCKED','exception':type(exc).__name__,'error':str(exc),
            'traceback':traceback.format_exc(),'retry_authorized':False})
        raise


def verify(root=ROOT,require_post=True):
    folder,m,_=checked(root)
    if (folder/'failure.json').exists():return read_json(folder/'failure.json')
    from .partial_phase5g_verify import verify_records
    result=verify_records(folder,m)
    if require_post:
        good=[p for p in (folder/'tests').glob('attempt-*.json') if read_json(p)['stage']=='post' and read_json(p)['exit_code']==0 and read_json(p)['source_digest']==m['source_digest']]
        if not good:raise ValueError('Post integration tests missing')
        result['post_tests']=read_json(sorted(good)[-1])['test_counts']
    if (folder/'output-hashes.json').exists():
        inventory=read_json(folder/'output-hashes.json')
        for n,h in inventory['files'].items():
            if sha(folder/n)!=h:raise ValueError('Output hash mismatch: '+n)
        for n,h in inventory.get('reports',{}).items():
            if sha(root/n)!=h:raise ValueError('Report hash mismatch: '+n)
    result['preservation']=preserve(root);result['legacy_tests']=m['legacy_counts'];result['artifact_bytes_at_verification']=quota_bytes([folder])
    if result['artifact_bytes_at_verification']>=536870912:raise ValueError('Artifact cap breached')
    return result


def report(root=ROOT):
    folder=location(root)
    for p in (root/'.runtime/phase5g-tests').glob('attempt-*'):
        target=folder/'tests'/p.name
        if target.exists() and sha(target)!=sha(p):raise ValueError('Cannot replace failed test history')
        if not target.exists():shutil.copyfile(p,target)
    try:result=verify(root)
    except Exception as e:result={'verdict':'PHASE5G_INCOMPLETE_OR_BLOCKED','error':str(e),'traceback':traceback.format_exc()}
    write_json(folder/'verification.json',result)
    from .partial_phase5g_report import render
    render(root,result);write_json(folder/'preservation.json',preserve(root))
    write_json(folder/'output-hashes.json',{'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name not in ('output-hashes.json','verification.json','preservation.json')},
        'reports':{p.relative_to(root).as_posix():sha(p) for p in [root/'docs/partial_phase5g_results.md',root/'docs/partial_phase5g_diagnosis.md']}})
    return {'verdict':result['verdict'],'artifact_bytes':quota_bytes([folder])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['test-pre','register','run','test-post','verify','report']);a=p.parse_args()
    result=tests('pre' if a.command=='test-pre' else 'post') if a.command.startswith('test-') else globals()[a.command]()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
