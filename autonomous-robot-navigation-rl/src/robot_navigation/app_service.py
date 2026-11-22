"""One-time episode generation, passive replay and fair development comparison."""
import argparse,csv,gzip,io,json,os,subprocess,sys,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
from .core import ROOT,digest,read_json,write_json
from .partial_contract import configuration
from .partial_baselines import OnlinePlanner,RandomPolicy
from .partial_evaluation import record_episode
from .app_registry import RUNS,load_model,newest_by_algorithm
from .app_scenarios import development_cases

_JOBS = {}

def identity():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]

def policy_for(selection,seed=0):
    if selection=='Observed-Map Replanner':return OnlinePlanner()
    if selection=='Random reference':return RandomPolicy(seed)
    if isinstance(selection,dict):return load_model(selection)
    raise ValueError('Choose a trained native model or an explicitly labeled non-RL reference')

def generate(case,selection,seed=0,output=None):
    policy=policy_for(selection,seed);record=record_episode(case,policy,configuration(),seed)
    folder=Path(output or RUNS)/'episodes';folder.mkdir(parents=True,exist_ok=True);path=folder/(identity()+'.json.gz')
    with gzip.open(path,'xt',encoding='utf-8') as f:json.dump(record,f,allow_nan=False)
    return record,path

def read_replay(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def frame(record,index):
    """Only index recorded data. No policy call and no environment stepping."""
    i=max(0,min(int(index),len(record['transitions'])))
    return record['initial_snapshot'] if i==0 else record['transitions'][i-1]['snapshot']

def cursor(index,command,length,seek=None):
    if command=='reset':return 0
    if command=='seek':return max(0,min(length,int(seek)))
    if command in ('step','play'):return min(length,index+1)
    if command=='back':return max(0,index-1)
    return max(0,min(length,index))

def compare(items=None,cases=None,include_random=False,output=None):
    selected=items or newest_by_algorithm();cases=cases or development_cases()
    required={'q-learning','sarsa','dqn','ppo'}
    if set(selected)!=required:raise ValueError('Comparison needs one native checkpoint for each of Q-learning, SARSA, DQN and PPO. Train the missing models first.')
    folder=Path(output or RUNS)/'comparisons'/identity();folder.mkdir(parents=True,exist_ok=False)
    choices=[(a,selected[a],'RL') for a in ('q-learning','sarsa','dqn','ppo')]+[('Observed-Map Replanner','Observed-Map Replanner','Non-RL reference')]
    if include_random:choices.append(('Random reference','Random reference','Non-RL reference'))
    rows=[];replays=[]
    for label,selection,kind in choices:
        for case in cases:
            record,path=generate(case,selection,seed=0,output=folder)
            m=record['metrics'];base=record['scenario'];success=m['outcome']=='success'
            row={'policy':label,'kind':kind,'case_id':base['id'],'scenario_hash':digest(base),'event_seed':base['event_seed'],
                'condition':'dynamic' if base['events'] else 'static','outcome':m['outcome'],'success':int(success),
                'steps':m['actions'],'reward':m['return'],'wait_count':m['wait_count'],'wait_fraction':m['wait_proportion'],
                'moving_revisits':m['repeated_movement_visits'],'two_cell_loops':m['two_cell_oscillation_windows'],
                'recovery_opportunities':m['recovery_opportunities'],'recoveries':m['recoveries'],
                'successful_movement_excess':m['posthoc_successful_movement_excess_over_base_path'],
                'training_seed':record['training_seed'],'checkpoint_sha256':record['checkpoint_sha256'],'replay':path.relative_to(folder).as_posix()}
            rows.append(row);replays.append(path)
    with (folder/'case-results.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for label,_,kind in choices:
        for subset in ('all','static','dynamic'):
            part=[r for r in rows if r['policy']==label and (subset=='all' or r['condition']==subset)]
            if not part:continue
            n=len(part);actions=sum(r['steps'] for r in part);opportunities=sum(r['recovery_opportunities'] for r in part)
            summary.append({'Policy':label,'Type':kind,'Subset':subset,'Cases':n,'Successes':sum(r['success'] for r in part),
                'Success %':100*sum(r['success'] for r in part)/n,'Collision %':100*sum(r['outcome']=='collision' for r in part)/n,
                'Timeout %':100*sum(r['outcome']=='timeout' for r in part)/n,'Mean steps':actions/n,'Mean reward':sum(r['reward'] for r in part)/n,
                'Wait %':100*sum(r['wait_count'] for r in part)/actions if actions else None,
                'Revisits':sum(r['moving_revisits'] for r in part),'Loops':sum(r['two_cell_loops'] for r in part),
                'Recoveries':sum(r['recoveries'] for r in part),'Recovery opportunities':opportunities,
                'Recovery %':100*sum(r['recoveries'] for r in part)/opportunities if opportunities else None})
    manifest={'schema':'dynamic-app-comparison-v1','purpose':'Product development/manual comparison; no historical scores or final/unseen cohorts',
        'case_ids':[c['id'] for c in cases],'case_hashes':[digest(c) for c in cases],'evaluation_seed':0,
        'checkpoint_ids':{a:i['id'] for a,i in selected.items()},'inference_only':True,'case_count':len(rows),'total_actions':sum(r['steps'] for r in rows),'summary':summary}
    write_json(folder/'manifest.json',manifest)
    with zipfile.ZipFile(folder/'results-and-replays.zip','x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in [folder/'case-results.csv',folder/'manifest.json',*replays]:z.write(p,p.relative_to(folder).as_posix())
    return {'folder':str(folder),'manifest':manifest,'rows':rows}

def start_job(preset,models,actions,seed):
    job=ROOT/'.runtime/jobs'/identity();job.mkdir(parents=True,exist_ok=False)
    args=[sys.executable,'-m','robot_navigation.app_training','--preset',preset,'--models',*models,'--actions',str(actions),'--seed',str(seed),'--job-dir',str(job)]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
    with (job/'stdout.log').open('w') as out,(job/'stderr.log').open('w') as err:
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    write_json(job/'job.json',{'pid':child.pid,'command':args})
    _JOBS[str(job)] = child
    return str(job)

def job_status(job):
    path=Path(job)/'progress.json'
    status=read_json(path) if path.is_file() else {'status':'starting','fraction':0.}
    error=Path(job)/'stderr.log'
    child=_JOBS.get(str(job))
    if child is not None and child.poll() is not None and status['status'] in ('starting','running'):
        status.update(status='failed',error=f'Worker exited ({child.returncode}) before completion. '+(error.read_text(errors='replace')[-2000:] if error.exists() else ''))
    return status

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compare',action='store_true');p.add_argument('--include-random',action='store_true');a=p.parse_args()
    if a.compare:r=compare(include_random=a.include_random);print(json.dumps({'folder':r['folder'],**r['manifest']},indent=2))
    else:p.print_help()
if __name__=='__main__':main()
