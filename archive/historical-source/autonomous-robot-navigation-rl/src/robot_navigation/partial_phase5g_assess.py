"""Detached deterministic inference on only the frozen 4+16 development cases."""
import argparse,inspect,json,random,sys
from time import perf_counter
import numpy as np
import torch

from .core import ROOT,read_json,write_json
from .release import sha
from .partial_contract import configuration,MOVES
from .partial_dynamic_env import PartialDynamicEnv
from .partial_features_scaled import load_scaled_policy
from .partial_pilot_diagnostics import decision_values
from .partial_phase5b_diagnostics import continuity_hash
from .partial_instrumentation_trace import numeric,checksum,quota_bytes
from .partial_phase5f_verify import require
from .partial_phase5g_io import location,cases,inputs,CHECKPOINTS,Journal,AssessmentBudget,observed_hash,direction


def global_rng():return {'python':continuity_hash(random.getstate()),'numpy':continuity_hash(np.random.get_state()),'torch':continuity_hash(torch.get_rng_state())}

def public_position(obs):return np.rint(np.asarray(obs[:2])*9).astype(int).tolist()

def reducing_actions(position,goal):
    old=sum(abs(x-y) for x,y in zip(position,goal));result=[]
    for a,(dr,dc) in enumerate(MOVES):
        n=position[0]+dr,position[1]+dc
        if all(0<=x<10 for x in n) and sum(abs(x-y) for x,y in zip(n,goal))<old:result.append(a)
    return result

def first_metrics(case,action,scores,algorithm):
    key='action_values' if algorithm=='dqn' else 'action_probabilities'
    values=np.asarray(scores[key],dtype=float);require(values.shape==(5,) and np.isfinite(values).all(),'First scores')
    require(action==int(values.argmax()),'Deterministic first-action argmax')
    reducing=reducing_actions(case['start'],case['goal']);ties=np.flatnonzero(values==values.max()).tolist()
    other=max(values[i] for i in range(5) if i!=3)
    result={'aligned':action in reducing,'reducing_actions':reducing,'argmax_ties':ties,'tie':len(ties)>1,
        'wait':action==4,'invalid_move':not all(0<=x<10 for x in (case['start'][0]+MOVES[action][0],case['start'][1]+MOVES[action][1])),
        'goal_directed_margin':float(max(values[i] for i in reducing)-max(values[i] for i in range(5) if i not in reducing)),
        'right_margin':float(values[3]-other),'right_margin_scale':'Q units' if algorithm=='dqn' else 'probability difference',
        'right_log_odds':'not_applicable' if algorithm=='dqn' else float(np.log(values[3]/other))}
    return result


def evaluate(algorithm,offset):
    folder=location();m=read_json(folder/'manifest.json');panel=cases();ordered=panel['original_cases']+panel['expanded_cases']
    require(algorithm in ('dqn','ppo'),'Allowed model')
    require(sha(folder/'manifest.json')==(folder/'manifest.sha256').read_text().strip() and inputs()==m['source_inputs'],'Frozen assessment source/seal')
    require({k:[{'id':c['id'],'sha256':checksum(c)} for c in v] for k,v in panel.items()}==m['panels'],'Frozen panels')
    require((folder/'run.lock').exists() and (folder/'assessments.lock').exists(),'Only launched after training')
    launch=read_json(folder/f'assessment-launch-{algorithm}.json')
    require(launch['algorithm']==algorithm and launch['offset']==offset and launch['training_receipt_sha256']==sha(folder/'training.json'),'Registered child/offset')
    require(read_json(folder/'training.json')['scientific_training_actions']==4000,'Both training arms complete before evaluation')
    arm=folder/algorithm;out=arm/'development';out.mkdir(exist_ok=False)
    journal=Journal(out/'assessment-ledger.jsonl',activity='fixed_development_assessment',quota_root=folder)
    budget=AssessmentBudget(journal);budget.count=offset;started=perf_counter();counts={'environment_step_calls':0,'optimizer_calls':0}
    code=inspect.unwrap(torch.optim.Adam.step).__code__
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is code:
            counts['optimizer_calls']+=1;raise RuntimeError('Optimizer forbidden in assessment')
        if event=='return' and frame.f_code is PartialDynamicEnv.step.__code__ and arg is not None:counts['environment_step_calls']+=1
    require(sys.getprofile() is None and sys.gettrace() is None,'Assessment hooks disabled initially')
    torch.set_num_threads(1);all_cases=[]
    try:
        sys.setprofile(profile)
        for cp in CHECKPOINTS:
            target=arm/'checkpoints'/f'{cp:04d}';checkpoint_sha=sha(target/'policy.zip')
            policy=load_scaled_policy(target/'policy.zip',configuration(),'B1-relative-x4')
            state=lambda:{'network':continuity_hash(policy.model.policy.state_dict()),'optimizer':continuity_hash(policy.model.policy.optimizer.state_dict())}
            before=state();rng_before=global_rng();rows=[]
            for index,case in enumerate(ordered):
                budget.start_case();env=PartialDynamicEnv([case],configuration());obs,_=env.reset(seed=0)
                initial=obs.copy();path=[public_position(obs)];steps=[];scores=decision_values(policy,obs)
                journal.append('case_started',algorithm=algorithm,checkpoint=cp,case_id=case['id'],panel='original' if index<4 else 'expanded',
                    seed=0,initial_observation=numeric(obs),checkpoint_sha256=checkpoint_sha)
                while True:
                    action=policy.act(obs)
                    ident=budget.reserve(algorithm=algorithm,checkpoint=cp,case_id=case['id'])
                    before_hash=observed_hash(obs);position=public_position(obs)
                    next_obs,reward,terminated,truncated,info=env.step(action)
                    step={'assessment_action_id':ident,'step':len(steps)+1,'action':action,'reward':float(reward),'terminated':bool(terminated),
                        'truncated':bool(truncated),'outcome':info['outcome'],'position':position,'next_position':public_position(next_obs),
                        'observation_sha256':before_hash,'next_observation_sha256':observed_hash(next_obs)}
                    journal.append('assessment_accepted',algorithm=algorithm,checkpoint=cp,case_id=case['id'],**step)
                    steps.append(step);path.append(step['next_position']);obs=next_obs
                    if terminated or truncated:break
                initial_action=steps[0]['action'];fm=first_metrics(case,initial_action,scores,algorithm)
                require(before==state(),'Assessment mutated model/optimizer')
                result={'algorithm':algorithm,'checkpoint':cp,'case_id':case['id'],'panel':'original' if index<4 else 'expanded',
                    'direction':direction(case),'initial_observation':numeric(initial),'first_action':initial_action,'first_scores':scores,
                    'first_metrics':fm,'steps':len(steps),'outcome':steps[-1]['outcome'],'trajectory':steps,'path':path,
                    'checkpoint_sha256':checkpoint_sha,'inference_only':True}
                rows.append(result);journal.append('case_finished',algorithm=algorithm,checkpoint=cp,case_id=case['id'],steps=len(steps),outcome=result['outcome'])
            require(before==state() and rng_before==global_rng(),'Assessment state/global RNG mutation')
            if quota_bytes([folder])+4*1024**2>=536870912:raise RuntimeError('Quota before evaluation result')
            write_json(out/f'{cp:04d}.json',{'algorithm':algorithm,'checkpoint':cp,'cases':rows,'model_optimizer_before':before,'model_optimizer_after':state(),
                'global_rng_before':rng_before,'global_rng_after':global_rng(),'checkpoint_sha256':checkpoint_sha,'training_actions':0,'optimizer_steps':0})
            all_cases.extend(rows)
        journal.append('assessment_closed',algorithm=algorithm,cases=len(all_cases),accepted_actions=budget.count-offset)
        require(counts['environment_step_calls']==budget.count-offset and counts['optimizer_calls']==0,'Assessment native call witness')
        result={'status':'completed','algorithm':algorithm,'evaluation_cases':len(all_cases),'evaluation_actions':budget.count-offset,
            'global_action_offset':offset,'global_action_end':budget.count,'native_calls':counts,'wall_seconds':perf_counter()-started,
            'training_actions':0,'optimizer_steps':0,'final_unseen_accesses':0,'process_id':__import__('os').getpid()}
        write_json(out/'receipt.json',result);return result
    except BaseException as exc:
        try:journal.append('assessment_failed',algorithm=algorithm,error=repr(exc),reserved_actions=budget.count-offset)
        except BaseException:pass
        raise
    finally:sys.setprofile(None);journal.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('algorithm',choices=['dqn','ppo']);p.add_argument('offset',type=int);a=p.parse_args()
    print(json.dumps(evaluate(a.algorithm,a.offset),indent=2))

if __name__=='__main__':main()
