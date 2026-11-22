"""Fast Phase 5H: focused test, register, one paired run, analyze once, verify."""
import argparse,csv,gc,hashlib,inspect,json,os,random,subprocess,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from .core import ROOT,read_json,write_json
from .release import sha
from .partial_contract import configuration
from .partial_dynamic_env import PartialDynamicEnv
from .partial_training_instrumentation import check_compatibility,library_contract,network_versions
from .partial_phase5g_io import cases
from .partial_phase5g_assess import global_rng
from .partial_phase5h_auxiliary import LeanLearning,parameter_counts,SEED
from .partial_phase5h_evidence import STUDY,BASE,ARMS,CHECKPOINTS,location,require,Stream,TrainingEnv,model,save_checkpoint,validate,assess

BRANCH='feature/'+STUDY
MAIN=Path('C:/Users/User/Desktop/Autonomous_Robot_Navigation')

def git(root,*args):
    owner=root if (root/'.git').exists() else root.parent
    return subprocess.check_output(['git','-c','core.longpaths=true','-c','safe.directory='+str(owner),*args],cwd=root,text=True)

def own_sources():
    paths=list((ROOT/'src/robot_navigation').glob('partial_phase5h*.py'))+[ROOT/'configs/partial_phase5h_v1.yaml',
        ROOT/'developer/tests/test_partial_phase5h.py',ROOT/'docs/partial_phase5h_protocol.md',ROOT.parent/'CODEX_PHASE5H_FAST_GOAL_CONDITIONING_EXPERIMENT.md']
    return {p.relative_to(ROOT.parent).as_posix():sha(p) for p in paths}

def dependencies():
    names=['core','release','partial_features_scaled','partial_neural','partial_agents','partial_dynamic_env','partial_contract','partial_features','partial_observation',
        'partial_memory','partial_sensor','partial_phase5b_diagnostics','partial_training_instrumentation','partial_instrumentation_trace',
        'partial_phase5f_io','partial_phase5f_verify','partial_phase5g_io','partial_phase5g_assess']
    paths=[ROOT/'src/robot_navigation'/f'{n}.py' for n in names]+[ROOT/'configs/partial_dynamic.yaml',ROOT/'configs/partial_feasibility_v1.yaml',
        ROOT/'configs/partial_phase5e_library_v1.json',ROOT/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/manifest.json']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def main_state():
    return {'head':git(MAIN,'rev-parse','HEAD').strip(),'status_sha256':hashlib.sha256(git(MAIN,'status','--porcelain').encode()).hexdigest(),
        'index_sha256':hashlib.sha256(git(MAIN,'diff','--cached','--binary').encode()).hexdigest()}

def settings():
    c=yaml.safe_load((ROOT/'configs/partial_phase5h_v1.yaml').read_text())
    require(c=={'study_id':STUDY,'seed':23141,'arms':['H0','H1'],'lambda':.05,'actions_per_arm':2000,'training_cap':4000,
        'checkpoints':[0,2000],'assessment_episodes':80,'assessment_cap':16000,'case_steps':200,'smoke_transitions':20,
        'mechanical_optimizer_cap':6,'training_wall_per_arm':1200,'assessment_wall':1800,'artifact_cap':134217728,
        'full_regression_suite':'DEFERRED_TO_MAJOR_MILESTONE','thresholds':{'successes':12,'per_direction':2,'success_advantage':3,'alignment_advantage':3,'wait_tolerance':.1}},'Frozen H envelope')
    return c

def focused():
    settings();folder=ROOT/'.runtime/phase5h-focused';folder.mkdir(exist_ok=False)
    command=[sys.executable,'-m','pytest','developer/tests/test_partial_phase5h.py','-q','--junitxml='+str(folder/'tests.xml')]
    started=perf_counter();env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PHASE5H_MECHANICS=str(folder))
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:p=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    receipt={'exit_code':p.returncode,'source_inputs':own_sources(),'wall_seconds':perf_counter()-started,
        'full_regression_suite':'DEFERRED_TO_MAJOR_MILESTONE','mechanical_counts':read_json(folder/'mechanics.json') if (folder/'mechanics.json').exists() else 'not_observed: failed before receipt'}
    write_json(folder/'receipt.json',receipt);require(p.returncode==0,'Essential focused gate failed; stop, no scientific run')
    return receipt

def register():
    settings();check_compatibility();require(git(ROOT,'rev-parse','HEAD').strip()==BASE and git(ROOT,'branch','--show-current').strip()==BRANCH,'Expected parent and H branch')
    require(git(ROOT,'diff','--name-only')=='' and git(ROOT,'diff','--cached','--name-only')=='','Historical files must be unchanged')
    old=ROOT/'artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1';v=read_json(old/'verification.json');p=read_json(old/'preservation.json')
    require(v['verdict']=='PHASE5G_MEASUREMENT_VALIDATED' and p['status']=='passed','Existing G receipts')
    receipt=read_json(ROOT/'.runtime/phase5h-focused/receipt.json');require(receipt['exit_code']==0 and receipt['source_inputs']==own_sources(),'Frozen focused tests')
    cfg=yaml.safe_load((ROOT/'configs/partial_feasibility_v1.yaml').read_text())['model'];panel=cases()
    folder=location();folder.mkdir(exist_ok=False);(folder/'.gitattributes').write_bytes(b'* -text\n')
    import shutil
    for name in ('receipt.json','mechanics.json','mechanics-evidence.json','mechanical-ledger.jsonl','smoke-actions.jsonl.gz','stdout.log','stderr.log','tests.xml'):
        shutil.copyfile(ROOT/'.runtime/phase5h-focused'/name,folder/('focused-'+name))
    plan={'study_id':STUDY,'parent':BASE,'branch':BRANCH,'settings':settings(),'sources':own_sources(),'dependencies':dependencies(),
        'library_contract':library_contract(),'loss_source_sha256':{n:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for n,f in {'smooth_l1':F.smooth_l1_loss,'cross_entropy':F.cross_entropy}.items()},
        'model':cfg,'panels':{k:[{'id':c['id'],'sha256':__import__('robot_navigation.partial_instrumentation_trace',fromlist=['checksum']).checksum(c)} for c in val] for k,val in panel.items()},
        'inherited_receipts':{'verification_sha256':sha(old/'verification.json'),'preservation_sha256':sha(old/'preservation.json'),'verdict':v['verdict']},
        'original_main':main_state(),'order':['H0','H1'],'initialization':'fresh seed23141; auxiliary Linear32x4 in fork_rng with seed23141, restored immediately',
        'label_rule':'round public coordinates*9, goal-position; greatest magnitude; vertical tie; exact goal excluded',
        'gradient_rule':'original TD to trunk and Q head; lambda CE to trunk and auxiliary head; original policy-wide gradient clipping includes head',
        'decision_rule':'all six fixed gates; complete direction collapse means successes confined to one direction, newly absent in H0',
        'inconclusive_rule':'at least one favorable H1 final expanded success or first-alignment difference, but full promising gates missed',
        'storage':'per-stream uncompressed caps train8MiB each/update24MiB each/assessment16MiB each =96MiB; checkpoints8MiB plus metadata24MiB =128MiB',
        'full_regression_suite':'DEFERRED_TO_MAJOR_MILESTONE','historical_audit':'Inherited G receipts; specific dependency hashes and Git scope only; no 10k-file rehash'}
    write_json(folder/'plan.json',plan);(folder/'plan.sha256').write_bytes((sha(folder/'plan.json')+'\n').encode())
    return {'status':'sealed_before_research','folder':str(folder),'focused_tests':receipt['mechanical_counts']}

def checked():
    folder=location();plan=read_json(folder/'plan.json');require(sha(folder/'plan.json')==(folder/'plan.sha256').read_text().strip(),'Plan seal')
    require(own_sources()==plan['sources'] and dependencies()==plan['dependencies'],'Specific frozen inputs');check_compatibility()
    require({n:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for n,f in {'smooth_l1':F.smooth_l1_loss,'cross_entropy':F.cross_entropy}.items()}==plan['loss_source_sha256'],'Loss hook sources restored/pinned')
    require(main_state()==plan['original_main'],'Original dirty main status/index/HEAD changed')
    require(git(ROOT,'diff','--name-only')=='','Existing tracked files changed')
    return folder,plan

def run():
    folder,plan=checked()
    with (folder/'run.lock').open('x') as f:f.write(sha(folder/'plan.json'))
    start=perf_counter();receipts={};pair={};initialized={}
    try:
        for arm in ARMS:
            stream=Stream(folder/f'train-{arm}.jsonl.gz',cap=8*1024**2);updates=Stream(folder/f'updates-{arm}.jsonl.gz',cap=24*1024**2)
            env=TrainingEnv(PartialDynamicEnv(cases()['training_cases'],configuration()),stream,arm);nopt=[0]
            try:
                random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);policy=model(env,arm,plan['model']);m=policy.model;env.model=m
                env.unwrapped.np_random=np.random.default_rng(SEED);env.action_space.seed(SEED)
                initialized[arm]={'Q':network_versions(m),'rng':global_rng(),'case_rng':__import__('robot_navigation.partial_phase5b_diagnostics',fromlist=['continuity_hash']).continuity_hash(env.unwrapped.np_random.bit_generator.state),
                    'action_rng':__import__('robot_navigation.partial_phase5b_diagnostics',fromlist=['continuity_hash']).continuity_hash(env.action_space.np_random.bit_generator.state)}
                if arm=='H1':
                    pair={'common_initial_equal':initialized['H0']['Q']==initialized['H1']['Q'],
                        'initial_rng_equal':all(initialized['H0'][k]==initialized['H1'][k] for k in ('rng','case_rng','action_rng')),'initial_states':initialized}
                    write_json(folder/'pairing.json',pair);require(pair['common_initial_equal'] and pair['initial_rng_equal'],'PAIRING_INVALID')
                save_checkpoint(policy,env,folder/f'{arm}-0000.pt',arm,0,sha(folder/'plan.json'))
                def reserve():
                    require(nopt[0]<475 and perf_counter()-env.started<1200,'Optimizer/time cap before operation')
                    stream.write({'event':'reserve_optimizer','arm':arm,'update_id':nopt[0]+1,'training_actions':env.n});nopt[0]+=1
                counts={'native_optimizer':0,'native_replay_consumption':0};adam=inspect.unwrap(torch.optim.Adam.step).__code__
                def profile(frame,event,arg):
                    if event=='call' and frame.f_code is adam and frame.f_locals.get('self') is m.policy.optimizer:counts['native_optimizer']+=1
                    if event=='return' and frame.f_code is ReplayBuffer._get_samples.__code__ and frame.f_locals.get('self') is m.replay_buffer and arg is not None:counts['native_replay_consumption']+=1
                require(sys.getprofile() is None,'Unexpected profiling hook')
                sys.setprofile(profile)
                try:
                    with LeanLearning(m,0. if arm=='H0' else .05,updates.write,reserve,lambda:env.last) as observer:m.learn(total_timesteps=2000,reset_num_timesteps=False)
                finally:sys.setprofile(None)
                require(env.n==2000 and m.num_timesteps==2000 and nopt[0]==observer.updates==m._n_updates==475,'Exact training/update count')
                require(counts=={'native_optimizer':475,'native_replay_consumption':475},'No excess optimizer/replay calls')
                save_checkpoint(policy,env,folder/f'{arm}-2000.pt',arm,2000,sha(folder/'plan.json'))
                require(perf_counter()-env.started<1200,'Arm wall limit')
                stream.write({'event':'closed','arm':arm,'actions':env.n,'optimizer_steps':nopt[0]});updates.write({'event':'closed','optimizer_steps':observer.updates})
                receipts[arm]={'actions':env.n,'optimizer_steps':nopt[0],'native_calls':counts,'wall_seconds':perf_counter()-env.started,'parameter_counts':parameter_counts(m)}
            finally:stream.close();updates.close()
            del observer,policy,m,env;gc.collect()
        write_json(folder/'train-receipts.json',receipts)
        before={'rng':global_rng(),'training_hashes':{p.name:sha(p) for p in folder.glob('train-*.jsonl.gz')}}
        assessment_start=perf_counter();offset=0
        for arm in ARMS:
            write_json(folder/f'launch-{arm}.json',{'offset':offset,'elapsed_before':perf_counter()-assessment_start})
            envvars=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
            with (folder/f'assess-{arm}.stdout.log').open('x') as out,(folder/f'assess-{arm}.stderr.log').open('x') as err:
                p=subprocess.run([sys.executable,'-m','robot_navigation.partial_phase5h','assess','--arm',arm,'--offset',str(offset)],cwd=ROOT,env=envvars,stdout=out,stderr=err)
            require(p.returncode==0,'Detached assessment failed; no retry');offset+=read_json(folder/f'cases-{arm}.json')['actions']
        require(perf_counter()-assessment_start<=1800,'Assessment total wall cap')
        after={'rng':global_rng(),'training_hashes':{p.name:sha(p) for p in folder.glob('train-*.jsonl.gz')}};require(before==after,'Assessment changed parent RNG/training evidence')
        write_json(folder/'resource-ledger.json',{'training_actions':4000,'optimizer_steps':950,'assessment_actions':offset,'assessment_episodes':80,
            'training_seconds':{k:r['wall_seconds'] for k,r in receipts.items()},'assessment_seconds':perf_counter()-assessment_start,
            'total_run_seconds':perf_counter()-start,'assessment_parent_isolation':{'before':before,'after':after},
            'mechanical':read_json(folder/'focused-mechanics.json'),'full_regression_suite':'DEFERRED_TO_MAJOR_MILESTONE'})
        return {'status':'runs_completed','training_actions':4000,'assessment_actions':offset,'total_seconds':perf_counter()-start}
    except BaseException as exc:
        write_json(folder/'failure.json',{'verdict':'PAIRING_INVALID' if 'PAIRING_INVALID' in str(exc) else 'INCOMPLETE_OR_INVALID','error':repr(exc),'retry':False});raise

def analyze():
    folder,plan=checked();require(not(folder/'analysis.lock').exists(),'Analyze once; no duplicate verification/evaluation')
    (folder/'analysis.lock').write_bytes(b'one offline analysis\n')
    try:
        require(not(folder/'failure.json').exists(),'Research attempt incomplete')
        result=validate(folder,plan);rows=result.pop('case_rows')
        ledger=read_json(folder/'resource-ledger.json');require(ledger['assessment_parent_isolation']['before']==ledger['assessment_parent_isolation']['after'],'Parent isolation')
        for arm,r in read_json(folder/'train-receipts.json').items():require(r['native_calls']=={'native_optimizer':475,'native_replay_consumption':475} and r['wall_seconds']<=1200,'Measured calls/wall')
        require(ledger['assessment_seconds']<=1800,'Assessment wall limit')
        fields=['arm','checkpoint','case_id','panel','direction','outcome','steps','first_action','wait_count','moving_revisits','ABAB','shortest_path','inefficiency']
        with (folder/'case-results.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields+['first_aligned','goal_margin','initial_Q']);w.writeheader()
            for r in rows:w.writerow({**{k:r[k] for k in fields},'first_aligned':r['first_metrics']['aligned'],'goal_margin':r['first_metrics']['goal_directed_margin'],'initial_Q':json.dumps(r['Q'])})
        flat=[]
        for arm,r in result['arms'].items():
            for cp,n in r['checkpoints'].items():flat.append({'arm':arm,'checkpoint':cp,**{k:v for k,v in n.items() if k!='directions'},**n['directions']})
        with (folder/'checkpoint-metrics.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
        write_json(folder/'optimizer-summary.json',{a:{k:v for k,v in n.items() if k!='checkpoints'} for a,n in result['arms'].items()})
        write_json(folder/'validation.json',result)
    except BaseException as exc:
        result={'verdict':'INCOMPLETE_OR_INVALID','error':repr(exc)};write_json(folder/'validation.json',result)
    preservation={'parent':BASE,'main_before':plan['original_main'],'main_after':main_state(),'scope':'new H paths only; inherited G receipts; no full historical byte audit'}
    require(preservation['main_before']==preservation['main_after'],'Original main changed');write_json(folder/'preservation.json',preservation)
    from .partial_phase5h_report import render
    render(ROOT,result)
    files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='hashes.json'}
    write_json(folder/'hashes.json',{'files':files,'reports':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'docs/partial_phase5h_results.md',ROOT/'docs/partial_phase5h_diagnosis.md')}})
    require(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())<=134217728,'Artifact cap')
    return {'verdict':result['verdict'],'training_actions':result.get('training_actions'),'assessment_actions':result.get('assessment_actions')}

def verify():
    """Read-only published-file validation; numerical analysis was done once."""
    folder,_=checked();inventory=read_json(folder/'hashes.json')
    for n,h in inventory['files'].items():require(sha(folder/n)==h,'Output hash: '+n)
    for n,h in inventory['reports'].items():require(sha(ROOT/n)==h,'Report hash: '+n)
    require(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())<=134217728,'Artifact cap')
    v=read_json(folder/'validation.json');return {'verdict':v['verdict'],'output_hashes':'passed','full_regression_suite':'DEFERRED_TO_MAJOR_MILESTONE'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['test','register','run','analyze','verify','assess']);p.add_argument('--arm',choices=ARMS);p.add_argument('--offset',type=int,default=0);a=p.parse_args()
    result=focused() if a.command=='test' else assess(a.arm,a.offset) if a.command=='assess' else globals()[a.command]()
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_inputs')},indent=2))
if __name__=='__main__':main()
