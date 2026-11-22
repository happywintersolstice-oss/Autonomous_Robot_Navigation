"""Registered, one-shot Phase 5D read-only diagnosis. Never starts a trainer."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import yaml

from .core import ROOT, read_json, write_json
from .release import sha
from .partial_contract import configuration, ACTION_NAMES
from .partial_features_scaled import ARMS, load_scaled_policy
from .partial_phase5b import junit
from .partial_phase5c_coverage import coordinates, direction, score_record, state_bins
from .partial_pilot_diagnostics import preserve_random_state
from .partial_phase5b_diagnostics import continuity_hash
from .partial_phase5d_states import state_hash, relation, select_cohort, match_states, check_goal_only_pair
from .partial_phase5d_credit import (ForwardBudget, FrozenScores, scientific_guard, protect_inference,
    attribution, snapshot_td, ppo_statistics, audit_linear_logs, audit_ppo_rollout)

STUDY='phase5d-goal-credit-diagnosis-v1'
BASE='c1505cc1dab0623b9c7eb86b7bfae9ac3216fa4d'
FAMILIES=('q-learning','sarsa','dqn','ppo')
CHECKPOINTS=(0,2000,6000)
DIRECTIONS=('Up','Down','Left','Right')
BRIEF='CODEX_PHASE5D_GOAL_CONDITIONING_CREDIT_ASSIGNMENT_DIAGNOSIS.md'


def location(root=ROOT): return root/'artifacts/partial-dynamic-navigation-v1'/STUDY
def historical(root,study): return root/'artifacts/partial-dynamic-navigation-v1'/study
def b_location(root): return historical(root,'phase5b-a-only-v1')
def c_location(root): return historical(root,'phase5c-directional-coverage-v1')
def arm_path(root,family,arm): return b_location(root)/'training-and-evaluation-evidence'/family/arm


def settings(root=ROOT):
    c=yaml.safe_load((root/'configs/partial_phase5d_v1.yaml').read_text())
    expected={'study_id':STUDY,'version':'goal-credit-read-only-v1','reviewed_commit':BASE,
        'algorithms':list(FAMILIES),'arms':list(ARMS),'checkpoints':list(CHECKPOINTS),
        'cohort_per_direction_and_outcome':6,'cohort_window':6000,'approximate_raw_mean_l1_max':.05,
        'forward_call_ceiling':5000,'declared_worst_case_calls':4752,'analysis_wall_seconds':1200,
        'artifact_byte_ceiling':134217728,'new_training_actions':0,'new_policy_environment_actions':0,'new_probes':0}
    if c!=expected: raise ValueError('Registered resource/protocol envelope changed')
    return c


def source_inputs(root=ROOT):
    paths=list((root/'src/robot_navigation').glob('*.py'))+list((root/'developer/tests').glob('test_partial_*.py'))
    paths+=list((root/'configs').glob('partial*.yaml'))
    paths += [root/'docs/partial_phase5d_protocol.md',root/'pyproject.toml',root/'requirements-lock.txt']
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


def checked_hash(path,expected):
    if not path.is_file() or sha(path)!=expected: raise ValueError('Hash mismatch: '+str(path))


def preserve_history(root=ROOT):
    saved=location(root)/'base-inventory.json'
    inventory=read_json(saved if saved.exists() else root/'.runtime/phase5d-base-inventory.json')
    for p,h in inventory['files'].items(): checked_hash(root.parent/p,h)
    counts={}
    for study in ('feasibility-pilot-v1','phase5b-a-only-v1','phase5c-directional-coverage-v1'):
        folder=historical(root,study); pins=read_json(folder/'delivery-manifest.json');count=0
        for p,h in pins['artifact_files'].items(): checked_hash(folder/p,h);count+=1
        for section,base in [('active_files',root),('implementation_files',root.parent),('new_application_files',root)]:
            for p,h in pins.get(section,{}).items():
                data=(base/p).read_bytes()
                if p in pins.get('normalized_implementation_paths',{}): data=data.replace(b'\r\n',b'\n')
                if hashlib.sha256(data).hexdigest()!=h: raise ValueError('Historical source mismatch: '+p)
                count+=1
        counts[study]=count
    return {'status':'passed','base_files':len(inventory['files']),'historical_pins':counts,'changed':[]}


class Evidence:
    """Allow only explicitly sealed Phase 5B/5C development evidence, never split paths."""
    def __init__(self,root,pins): self.root,self.pins=root,pins
    def path(self,path):
        p=Path(path).resolve();r=self.root.resolve()
        if not p.is_relative_to(r): raise ValueError('Evidence escaped root')
        name=p.relative_to(r).as_posix()
        if any(part.lower() in ('final','test','tests','held-out','heldout') for part in p.parts):
            raise ValueError('Forbidden final/test evidence read')
        if not any(name.startswith('artifacts/partial-dynamic-navigation-v1/'+s+'/') for s in
                   ('phase5b-a-only-v1','phase5c-directional-coverage-v1')) or name not in self.pins:
            raise ValueError('Evidence not on registered allowlist: '+name)
        checked_hash(p,self.pins[name]);return p
    def json(self,path): return read_json(self.path(path))
    def records(self,path):
        with gzip.open(self.path(path),'rt',encoding='utf-8') as stream: return [json.loads(line) for line in stream]
    def state(self,path):
        # Trusted local pickle, verified against the sealed SHA before deserialization.
        return pickle.loads(gzip.decompress(self.path(path).read_bytes()))


def allowed_inputs(root):
    paths=[c_location(root)/'probe-manifest.json',c_location(root)/'replay-audit.json']
    for family in FAMILIES:
        for arm in ARMS:
            base=arm_path(root,family,arm)
            paths.append(base/'training-actions.jsonl.gz')
            paths.append(base/('linear-updates.jsonl.gz' if family in FAMILIES[:2] else 'optimizer-updates.jsonl.gz'))
            for cp in CHECKPOINTS:
                tag=f'{family}-{arm}-{cp:05d}.json'
                paths.extend(c_location(root)/sub/tag for sub in ('probes','decisions','coverage'))
                folder=base/f'{cp:05d}'
                paths.extend(folder/name for name in ('summary.json','policy.json','evaluation.jsonl.gz'))
                paths.append(folder/('policy.npz' if family in FAMILIES[:2] else 'policy.zip'))
                if family in FAMILIES[2:]: paths.append(folder/'continuation-state.pkl.gz')
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


def rng_fingerprint():
    import random
    import torch
    return continuity_hash({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()})


def availability():
    return {
      'linear':{'actual_target_TD_error_sampled_SARSA_action_update_norm':'historical',
        'continuing_bootstrap':'reconstructable-contemporaneous from adjacent logged scores and exact recorded increment',
        'truncated_or_last_continuing_bootstrap':'not-observed independently; actual target remains recorded',
        'feature_contributions':'snapshot exact algebra'},
      'dqn':{'online_and_target_weights_0_2k_6k':'historical saved checkpoints',
        'training_preaction_online_Q':'historical','replay_contents':'historical saved buffer',
        'optimizer_norms_counts':'historical','per_batch_sample_IDs_target_tensors_network_trajectory':'not-observed; not recorded',
        'frozen_TD_target_residual':'snapshot; never historical optimizer target'},
      'ppo':{'all_training_preaction_probabilities_V':'historical',
        'last_rollout_values_log_probs_returns_advantages_2k_6k':'historical; only last 100 transitions each',
        'sampled_log_prob':'reconstructable-contemporaneous from logged action probability',
        'full_GAE_history_minibatch_normalized_advantages_shuffle':'not-observed; not recorded',
        'last_boundary_bootstrap':'not-observed unless saved boundary is terminal; inspect each archive',
        'frozen_probabilities_entropy_log_odds_V':'snapshot','Q_values':'not-applicable'},
      'all':{'full_gradient_arrays':'not-observed; norm/finiteness summaries only',
        'assessment_episodes':'historical; 480 reused, zero executed',
        'legal_40_probes':'synthetic-score-only; exact Phase 5C reuse',
        'hidden_world_or_final_test':'not-accessed'}}


def library_semantics():
    from stable_baselines3 import DQN,PPO
    from stable_baselines3.common.buffers import ReplayBuffer,RolloutBuffer
    methods={'DQN.train':DQN.train,'ReplayBuffer._get_samples':ReplayBuffer._get_samples,
             'RolloutBuffer.compute_returns_and_advantage':RolloutBuffer.compute_returns_and_advantage,'PPO.train':PPO.train,
             'PPO.collect_rollouts':PPO.collect_rollouts}
    return {name:{'sha256':hashlib.sha256(inspect.getsource(method).encode()).hexdigest(),
                  'source':inspect.getsource(method)} for name,method in methods.items()}


def validate(root=ROOT,post=False):
    before=source_inputs(root);target=root/'.runtime'/('phase5d-post-tests.xml' if post else 'phase5d-tests.xml')
    start=perf_counter()
    result=subprocess.run([sys.executable,'-m','pytest','-q',f'--junitxml={target}'],cwd=root,capture_output=True,text=True)
    target.with_suffix('.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout+result.stderr,end='')
    if result.returncode or source_inputs(root)!=before: raise ValueError('Regression/source gate failed')
    record={'counts':junit(target),'wall_seconds':perf_counter()-start,'source_inputs':before,'sha256':sha(target)}
    write_json(target.with_suffix('.json'),record);return record['counts']


def prepare(root=ROOT):
    c=settings(root);preserved=preserve_history(root)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
    if head!=BASE or branch!='feature/'+STUDY: raise ValueError('Incorrect parent/branch')
    if junit(root/'.runtime/phase5d-baseline-tests.xml')!={'tests':399,'failures':0,'errors':0,'skipped':0}:
        raise ValueError('Inherited baseline failed')
    receipt=read_json(root/'.runtime/phase5d-tests.json')
    if receipt['source_inputs']!=source_inputs(root): raise ValueError('Stale preregistration tests')
    pins=allowed_inputs(root);reader=Evidence(root,pins);cohorts={}
    for family in FAMILIES:
        for arm in ARMS:
            rows=reader.records(arm_path(root,family,arm)/'training-actions.jsonl.gz')
            steps=[r for r in rows if r['kind']=='step']
            if [s['global_training_interaction'] for s in steps]!=list(range(1,6001)): raise ValueError('Missing training records')
            cohorts[family+'/'+arm]=select_cohort(steps,c['cohort_per_direction_and_outcome'])
    folder=location(root);folder.mkdir(parents=True,exist_ok=False)
    (folder/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n',encoding='utf-8',newline='\n')
    for name,target in [('phase5d-base-inventory.json','base-inventory.json'),('phase5d-original-inventory.json','original-inventory.json'),
                        ('phase5d-baseline-tests.xml','baseline-regressions.xml'),('phase5d-tests.xml','preregistration-regressions.xml')]:
        shutil.copy2(root/'.runtime'/name,folder/target)
    shutil.copy2(c_location(root)/'replay-audit.json',folder/'parent-verification.json')
    history=folder/'validation-history';history.mkdir()
    for name in ('phase5d-focused-initial-failure.xml','phase5d-focused-tests.xml','phase5d-prerefinement-tests.xml'):
        path=root/'.runtime'/name
        if path.exists():shutil.copy2(path,history/name)
    source=folder/'source';source.mkdir()
    new=list((root/'src/robot_navigation').glob('partial_phase5d*.py'))+[root/'configs/partial_phase5d_v1.yaml',
        root/'docs/partial_phase5d_protocol.md',root/'developer/tests/test_partial_phase5d.py',root.parent/BRIEF]
    for p in new: shutil.copy2(p,source/p.name)
    write_json(folder/'evidence-availability.json',availability())
    write_json(folder/'library-semantics.json',library_semantics())
    plan={'study_id':STUDY,'registered_utc':datetime.now(timezone.utc).isoformat(),'base_commit':head,
        'base_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip(),'branch':branch,
        'settings':c,'source_inputs':source_inputs(root),'evidence_inputs':pins,'cohorts':cohorts,
        'source_snapshot':{p.name:sha(p) for p in source.iterdir()},'availability_hash':sha(folder/'evidence-availability.json'),
        'library_semantics_hash':sha(folder/'library-semantics.json'),'preservation':preserved,'pretests':receipt,
        'brief_raw_sha256':sha(root.parent/BRIEF),
        'validation_history':{'initial_focused':'23 passed, 2 failed due to synthetic fixture float64; corrected before registration',
            'focused_final':'25 passed; no learning calls in Phase 5D unit fixtures'},
        'runtime':{'python':platform.python_version(),'dependencies':{n:version(n) for n in ('numpy','torch','stable-baselines3','gymnasium','pytest')}},
        'randomness':'No sampling RNG; fixed evenly spaced distinct-state cohorts. Global RNG preserved around model loads/inference.',
        'read_policy':'Only sealed Phase 5B/5C development evidence; historical inventory uses bytes only, never parses final/test.',
        'case_count_per_checkpoint':20,'probe_count_per_checkpoint':40,'declared_worst_case_calls':4752}
    write_json(folder/'analysis-plan.json',plan);write_json(folder/'analysis-seal.json',{'sha256':sha(folder/'analysis-plan.json')})
    return {'status':'prepared','plan_sha256':sha(folder/'analysis-plan.json'),'allowed_files':len(pins)}


def checked_plan(root=ROOT):
    folder=location(root);checked_hash(folder/'analysis-plan.json',read_json(folder/'analysis-seal.json')['sha256'])
    plan=read_json(folder/'analysis-plan.json')
    checked_hash(root.parent/BRIEF,plan['brief_raw_sha256'])
    if plan['source_inputs']!=source_inputs(root): raise ValueError('Registered sources changed')
    for p,h in plan['source_snapshot'].items(): checked_hash(folder/'source'/p,h)
    for name,key in [('evidence-availability.json','availability_hash'),('library-semantics.json','library_semantics_hash')]: checked_hash(folder/name,plan[key])
    if library_semantics()!=read_json(folder/'library-semantics.json'):raise ValueError('Installed library implementation changed')
    for p,h in plan['evidence_inputs'].items(): checked_hash(root/p,h)
    return plan


def unique_training_total(counts):
    if len(counts)!=8 or any(n!=6000 for n in counts): raise ValueError('Count unique 6k arms only; never sum nested checkpoints')
    return sum(counts)


def historical_alignment(records,cp):
    steps=[r for r in records if r['kind']=='step' and r['global_training_interaction']<=cp]
    resets=[r for r in records if r['kind']=='reset' and r['episode']<=max((s['episode'] for s in steps),default=-1)]
    # The zero-action checkpoint includes its recorded initial reset, not later resets.
    if cp==0: resets=[r for r in records if r['kind']=='reset'][:1]
    rows=[]
    for d in DIRECTIONS:
        current=[s for s in steps if relation(s['observation'])[0]==d]
        near=[s for s in current if relation(s['observation'])[1]==1]
        reset_count=sum(relation(r['observation'])[0]==d for r in resets)
        quality=[score_record(s['observation'],s['behavior_context']['decision']) for s in near]
        rows.append({'direction':d,'checkpoint':cp,'episode_starts':reset_count,'axis_decisions':len(current),
          'axis_distinct_observations':len({state_hash(s['observation']) for s in current}),
          'axis_distinct_position_goal_pairs':len({tuple(coordinates(s['observation'])) for s in current}),
          'near1_decisions':len(near),'near1_distinct_observations':len({state_hash(s['observation']) for s in near}),
          'reducing_attempts':sum(s['action'] in q['reducing_actions'] for s,q in zip(near,quality)),
          'away_attempts':sum(s['action']!=4 and s['action'] not in q['reducing_actions'] for s,q in zip(near,quality)),
          'wait_attempts':sum(s['action']==4 for s in near),'goal_arrivals':sum(s['reward']>0 for s in near),
          'contemporaneous_greedy_reducing':sum(q['first_argmax'] in q['reducing_actions'] for q in quality),
          'ties':sum(len(q['maximizers'])>1 for q in quality),'training_prefix':cp,
          'provenance':'historical; nested prefixes, exploration actions differ from deterministic argmax'})
    return rows


def dump_csv(path,rows):
    import csv
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def measured_score(engine,observation,family,cp,step=None):
    values=engine.get(observation);decision={('action_probabilities' if family=='ppo' else 'action_values'):np.asarray(values).tolist()}
    row=score_record(observation,decision)
    row['Right_margin']=float(values[3]-max(values[a] for a in (0,1,2,4)))
    row['state_bins']=state_bins(observation);row['provenance']='snapshot'
    if family in FAMILIES[:2]: row['linear_attribution']=attribution(values,engine.get(observation,'components'))
    elif family=='dqn':
        target=engine.get(observation,'target');row['target_scores']=np.asarray(target).tolist()
        if step is not None:
            nxt=None if step['terminated'] else engine.get(step['next_observation'],'target')
            row['next_target_scores']=None if nxt is None else np.asarray(nxt).tolist()
            row['snapshot_credit']=snapshot_td(step['reward'],step['terminated'],step['truncated'],nxt,values[step['action']],cp,step['global_training_interaction'])
    else:
        row['ppo']=ppo_statistics(values);row['state_value']=engine.get(observation,'value')
    return row


def counterfactuals(probes,rows):
    byid={p['id']:p for p in probes};scores={r['case_id']:r for r in rows};result=[]
    resets=[p for p in probes if p['kind']=='reset-goal-distance-translation']
    for i,a in enumerate(resets):
        for b in resets[i+1:]:
            if a['position']!=b['position']:continue
            check_goal_only_pair(a['observation'],b['observation'])
            sa,sb=scores[a['id']],scores[b['id']]
            result.append({'kind':'legal goal-only','first':a['id'],'second':b['id'],
                'score_mean_abs_change':float(np.mean(np.abs(np.array(sa['scores'])-sb['scores']))),
                'argmax_changed':sa['first_argmax']!=sb['first_argmax'],
                'first_tie_count':len(sa['maximizers']),'second_tie_count':len(sb['maximizers']),
                'provenance':'synthetic-score-only; same legal visible history, descriptive sensitivity'})
    for p in probes:
        if 'action_permutation' not in p: continue
        base=scores[p['base_id']];current=scores[p['id']];perm=p['action_permutation']
        transported=np.empty(5)
        for action,new in enumerate(perm):transported[new]=base['scores'][action]
        result.append({'kind':'D4 legal history','first':p['base_id'],'second':p['id'],
            'score_mean_abs_change':float(np.mean(np.abs(transported-current['scores']))),
            'argmax_equivariant':perm[base['first_argmax']]==current['first_argmax'],
            'tie_set_equivariant':sorted(perm[a] for a in base['maximizers'])==sorted(current['maximizers']),
            'provenance':'synthetic-score-only; transformed history also changes position/memory, not an isolated memory effect'})
    return result


def analyze(root=ROOT):
    plan=checked_plan(root);preserve_history(root);folder=location(root);c=settings(root)
    (folder/'analysis.lock').open('x').close()  # Persistent: an interrupted analysis cannot silently retry.
    start=perf_counter();budget=ForwardBudget(folder/'forward-calls.jsonl',c['forward_call_ceiling'])
    reader=Evidence(root,plan['evidence_inputs']);score_rows=[];cases=[];matches=[];directions=[];credit=[];cf=[];linear=[];buffers=[];counts=[]
    rng_checks=[]
    try:
        probes=reader.json(c_location(root)/'probe-manifest.json')['probes']
        if len(probes)!=40:raise ValueError('Incorrect fixed probe count')
        with scientific_guard():
            for family in FAMILIES:
                for arm in ARMS:
                    print('Analyzing',family,arm,flush=True)
                    base=arm_path(root,family,arm);records=reader.records(base/'training-actions.jsonl.gz')
                    steps=[r for r in records if r['kind']=='step'];counts.append(len(steps))
                    if [s['global_training_interaction'] for s in steps]!=list(range(1,6001)):raise ValueError('Training stream changed')
                    if select_cohort(steps)!=plan['cohorts'][family+'/'+arm]:raise ValueError('Cohort changed')
                    selected=[steps[row['global_step']-1] for row in plan['cohorts'][family+'/'+arm]['selected']]
                    if family in FAMILIES[:2]:
                        logs=reader.records(base/'linear-updates.jsonl.gz')
                        linear.extend(dict(row,family=family,arm=arm) for row in audit_linear_logs(steps,logs,ARMS[arm],family))
                    else:
                        logs=reader.records(base/'optimizer-updates.jsonl.gz')
                        if not logs or not all(r['finite'] for r in logs):raise ValueError('Invalid optimizer summaries')
                    for cp in CHECKPOINTS:
                        if perf_counter()-start>c['analysis_wall_seconds']:raise TimeoutError('Analysis wall ceiling reached')
                        if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>c['artifact_byte_ceiling']:raise RuntimeError('Artifact byte cap reached')
                        tag=f'{family}-{arm}-{cp:05d}.json';checkpoint=base/f'{cp:05d}'
                        oldprobes=reader.json(c_location(root)/'probes'/tag);oldcases=reader.json(c_location(root)/'decisions'/tag)
                        oldcoverage=reader.json(c_location(root)/'coverage'/tag)['coverage']
                        summary=reader.json(checkpoint/'summary.json');path=reader.path(checkpoint/summary['checkpoint'])
                        evaluations=reader.records(checkpoint/'evaluation.jsonl.gz')
                        if len(evaluations)!=20 or len(oldcases)!=20 or len(oldprobes)!=40:raise ValueError('Incomplete case/probe panel')
                        if oldcoverage['total']['decisions']!=cp:raise ValueError('Historical prefix count changed')
                        aligned=historical_alignment(records,cp)
                        for row in aligned:
                            d=row['direction'];near=oldcoverage['near_goal']['1'][d]
                            for fresh,old in [('near1_decisions','decisions'),('goal_arrivals','goal_rewards'),('contemporaneous_greedy_reducing','greedy_reducing')]:
                                if row[fresh]!=near[old]:raise ValueError('Historical Phase5C coverage disagreement')
                            # Phase5C retains an empty autoreset at the exact cutoff when present.
                            row['episode_starts']=oldcoverage['reset_directions'][d]
                            directions.append(dict(row,family=family,arm=arm))
                        reset_observations=[(r['case_id'],r['initial_observation']) for r in oldcases]
                        matches.extend(dict(row,family=family,arm=arm,checkpoint=cp) for row in match_states(steps[:cp],reset_observations))
                        rng_before=rng_fingerprint()
                        with preserve_random_state():
                            # Loader verifies checkpoint metadata, task signatures and 409/412 dimensions.
                            policy=load_scaled_policy(path,configuration(root),arm)
                            if policy.state_fingerprint()!=summary['parameter_fingerprint']:raise ValueError('Checkpoint fingerprint mismatch')
                            with protect_inference(policy):
                                engine=FrozenScores(policy,ARMS[arm],budget,tag)
                                for p,old in zip(probes,oldprobes):
                                    if p['id']!=old['probe_id']:raise ValueError('Probe order changed')
                                    engine.reuse(p['observation'],old['scores'])
                                for old in oldcases:engine.reuse(old['initial_observation'],old['first_decision']['scores'])
                                local=[]
                                for p in probes:
                                    row=dict(measured_score(engine,p['observation'],family,cp),family=family,arm=arm,checkpoint=cp,
                                        kind='legal-probe',case_id=p['id'],observation_sha256=state_hash(p['observation']),
                                        observation_ref='phase5c-directional-coverage-v1/probe-manifest.json#'+p['id'],probe_provenance='synthetic-score-only')
                                    score_rows.append(row);local.append(row)
                                cf.extend(dict(row,family=family,arm=arm,checkpoint=cp) for row in counterfactuals(probes,local))
                                for old,evaluation in zip(oldcases,evaluations):
                                    if evaluation['scenario']['id']!=old['case_id']:raise ValueError('Case order changed')
                                    o=old['initial_observation'];row=measured_score(engine,o,family,cp)
                                    row=dict(row,family=family,arm=arm,checkpoint=cp,case_id=old['case_id'],panel=old['panel'],
                                        direction=old['direction'],distance=old['distance'],initial_observation=o,
                                        position_goal=coordinates(o).tolist(),rays_North_South_East_West=o[4:8],
                                        known_cells=int(np.count_nonzero(np.array(o[8:108])!=-1)),historical_outcome=old['outcome'],
                                        historical_first_action=evaluation['transitions'][0]['action'],historical_actions=old['metrics']['actions'],
                                        trace_ref=(checkpoint/'evaluation.jsonl.gz').relative_to(root).as_posix()+'#'+old['case_id'],
                                        outcome_provenance='historical; zero new episode actions')
                                    cases.append(row)
                                for s in selected:
                                    row=dict(measured_score(engine,s['observation'],family,cp,s),family=family,arm=arm,checkpoint=cp,
                                        kind='fixed-historical-cohort',global_step=s['global_training_interaction'],
                                        historical_action=s['action'],historical_reward=s['reward'],terminated=s['terminated'],truncated=s['truncated'],
                                        observation_sha256=state_hash(s['observation']),direction=relation(s['observation'])[0],
                                        cohort='goal' if s['reward']>0 else 'other-near1',
                                        historical_decision=s['behavior_context']['decision'],
                                        trace_ref=(base/'training-actions.jsonl.gz').relative_to(root).as_posix())
                                    score_rows.append(row)
                                    credit.append({k:v for k,v in row.items() if k not in ('initial_observation',)})
                        if rng_fingerprint()!=rng_before:raise ValueError('Global inference RNG changed')
                        rng_checks.append({'model':tag,'weights_optimizer_buffer_unchanged':True,'global_RNG_restored':True,'rng_sha256':rng_before})
                        if family in FAMILIES[2:]:
                            state=reader.state(checkpoint/'continuation-state.pkl.gz')
                            if state['parameters']!=summary['parameter_fingerprint'] or state['timesteps']!=cp:raise ValueError('Continuation mismatch')
                            if family=='ppo':result=audit_ppo_rollout(state,steps[:cp])
                            else:
                                replay=state['replay'];n=int(replay.buffer_size if replay.full else replay.pos)
                                if n!=cp:raise ValueError('DQN replay count mismatch')
                                if n and (not np.allclose(np.asarray(replay.rewards[:n]).reshape(-1),[s['reward'] for s in steps[:cp]],atol=1e-6)
                                          or not np.allclose(np.asarray(replay.observations[:n]).reshape(n,409),[s['observation'] for s in steps[:cp]],atol=0,rtol=0)):
                                    raise ValueError('Replay content disagreement')
                                result={'saved_replay_transitions':n,'sampled_batch_IDs_and_actual_targets':'not recorded',
                                    'provenance':'historical buffer; no replay.sample or optimization executed'}
                            buffers.append(dict(result,family=family,arm=arm,checkpoint=cp))
        total=unique_training_total(counts)
        if len(cases)!=480 or len([r for r in score_rows if r['kind']=='legal-probe'])!=960:raise ValueError('Incomplete score panels')
        actual={'historical_unique_training_actions':total,'historical_assessment_actions':sum(r['historical_actions'] for r in cases),
            'historical_assessment_episodes':len(cases),'historical_phase5c_probe_calls_reused':960,'historical_case_vectors_reused':480,
            'new_forward_calls':budget.calls,'new_training_actions':0,'new_policy_environment_actions':0,'new_probes':0}
        if actual['historical_assessment_actions']!=41705:raise ValueError('Historical assessment action reconciliation failed')
        preservation=preserve_history(root)
        for name,rows in [('credit-metrics.csv',credit),('matched-states.csv',matches),('case-decisions.csv',cases),
            ('direction-coverage.csv',directions),('linear-credit-audit.csv',linear),('counterfactual-comparisons.csv',cf)]:dump_csv(folder/name,rows)
        write_json(folder/'scores.json',score_rows);write_json(folder/'buffer-audit.json',buffers);write_json(folder/'inference-preservation.json',rng_checks)
        summary={'study_id':STUDY,'verdict':'DIAGNOSTIC_COMPLETED_INSUFFICIENT_HISTORICAL_SIGNAL',
            'reason':'Missing DQN actual optimizer sample/target trajectory, incomplete PPO historical GAE, and development-state support limit unique causal attribution. Descriptive frozen sensitivity is not causal credit assignment.',
            'accounting':actual,'linear_updates_audited':len(linear),
            'linear_independent_targets':sum(r['computed_target'] is not None for r in linear),
            'wall_seconds':perf_counter()-start,'preservation':preservation,'resource_caps':c,
            'scientific_scope':'Fixture A only; no new trajectories; all 8 arms and all 3 snapshots retained',
            'followup':'Before further learning, register instrumentation for replay sample IDs, online/target fingerprints and actual TD targets; PPO collection bootstrap values, rollout IDs, returns/advantages and batch membership. No new training authorized.'}
        if summary['wall_seconds']>1200:raise TimeoutError('Analysis wall ceiling exceeded')
        write_json(folder/'diagnostic-summary.json',summary)
        write_json(folder/'source-and-evidence-manifest.json',{'base_commit':BASE,'base_tree':plan['base_tree'],
            'plan_sha256':sha(folder/'analysis-plan.json'),'dependencies':plan['source_inputs'],'evidence':plan['evidence_inputs']})
        write_json(folder/'validation-gates.json',{'G0':'passed','G1':'passed','G2':'passed','G3':'passed','G4':'passed','G5':'pending postrun tests and independent verification'})
        return summary
    except Exception:
        write_json(folder/'failure.json',{'verdict':'IMPLEMENTATION_BLOCKED','traceback':traceback.format_exc(),
            'new_forward_calls':budget.calls,'wall_seconds':perf_counter()-start,'retry_authorized':False})
        raise
    finally: budget.close()


def verify(root=ROOT):
    plan=checked_plan(root);folder=location(root);preserved=preserve_history(root)
    if (folder/'failure.json').exists():raise ValueError('Analysis blocked; no retry authorized')
    post=read_json(root/'.runtime/phase5d-post-tests.json')
    if post['source_inputs']!=plan['source_inputs'] or post['counts']['failures'] or post['counts']['errors']:raise ValueError('Postrun tests stale/failed')
    summary=read_json(folder/'diagnostic-summary.json');scores=read_json(folder/'scores.json')
    if (folder/'output-hashes.json').exists():
        pins=read_json(folder/'output-hashes.json')
        for p,h in pins['files'].items():checked_hash(folder/p,h)
        for p,h in pins['reports'].items():checked_hash(root/p,h)
        # Completed delivery verification is read-only and leaves published hashes stable.
        return read_json(folder/'verification.json')
    calls=[json.loads(x) for x in (folder/'forward-calls.jsonl').read_text().splitlines()]
    if [r['call'] for r in calls]!=list(range(1,len(calls)+1)) or len(calls)!=summary['accounting']['new_forward_calls'] or len(calls)>4752:
        raise ValueError('Forward call ledger mismatch')
    # Independent arithmetic on saved derived scores, never a second inference run.
    for row in scores:
        values=np.asarray(row['scores']);expected=score_record_from_values(row,values)
        if expected!=row['first_argmax']:raise ValueError('Derived argmax mismatch')
        if 'linear_attribution' in row:attribution(values,row['linear_attribution']['components'])
        if row['family']=='ppo':ppo_statistics(values)
        if 'snapshot_credit' in row:
            credit=row['snapshot_credit']
            if credit['provenance']!='snapshot' or credit['historical_optimizer_target_verified']:raise ValueError('Snapshot mislabelled as historic')
            computed=snapshot_td(row['historical_reward'],row['terminated'],row['truncated'],row['next_target_scores'],
                values[row['historical_action']],row['checkpoint'],row['global_step'])
            if computed!=credit:raise ValueError('Snapshot target/residual arithmetic changed')
    reader=Evidence(root,plan['evidence_inputs']);recount=[];independent_linear=0
    from .partial_phase5d_report import records as csv_records
    case_rows=csv_records(folder/'case-decisions.csv');match_rows=csv_records(folder/'matched-states.csv')
    coverage_rows=csv_records(folder/'direction-coverage.csv')
    for family in FAMILIES:
        for arm in ARMS:
            base=arm_path(root,family,arm);rows=reader.records(base/'training-actions.jsonl.gz')
            steps=[r for r in rows if r['kind']=='step'];recount.append(len(steps))
            if select_cohort(steps)!=plan['cohorts'][family+'/'+arm]:raise ValueError('Independent cohort mismatch')
            if family in FAMILIES[:2]:
                audited=audit_linear_logs(steps,reader.records(base/'linear-updates.jsonl.gz'),ARMS[arm],family)
                independent_linear+=sum(r['computed_target'] is not None for r in audited)
            for cp in CHECKPOINTS:
                tag=f'{family}-{arm}-{cp:05d}.json'
                oldcases=reader.json(c_location(root)/'decisions'/tag)
                oldprobes=reader.json(c_location(root)/'probes'/tag)
                newcases=[r for r in case_rows if (r['family'],r['arm'],r['checkpoint'])==(family,arm,cp)]
                if len(newcases)!=20:raise ValueError('Independent case count mismatch')
                for old,new in zip(oldcases,newcases):
                    if (old['case_id']!=new['case_id'] or old['initial_observation']!=new['initial_observation']
                        or old['outcome']!=new['historical_outcome'] or old['first_decision']['scores']!=new['scores']):
                        raise ValueError('Historical development case changed in derived evidence')
                newprobes=[r for r in scores if (r['family'],r['arm'],r['checkpoint'],r['kind'])==(family,arm,cp,'legal-probe')]
                if [(r['probe_id'],r['scores']) for r in oldprobes]!=[(r['case_id'],r['scores']) for r in newprobes]:
                    raise ValueError('Historical probe score reuse mismatch')
                expected=match_states(steps[:cp],[(r['case_id'],r['initial_observation']) for r in oldcases])
                actual=[{k:v for k,v in r.items() if k not in ('family','arm','checkpoint')} for r in match_rows
                        if (r['family'],r['arm'],r['checkpoint'])==(family,arm,cp)]
                if actual!=expected:raise ValueError('Independent matched-state support mismatch')
                expected=historical_alignment(rows,cp)
                actual=[r for r in coverage_rows if (r['family'],r['arm'],r['checkpoint'])==(family,arm,cp)]
                for e,a in zip(expected,actual):
                    for key in e:
                        if key!='episode_starts' and e[key]!=a[key]:raise ValueError('Independent historical alignment mismatch')
    if unique_training_total(recount)!=48000 or independent_linear!=summary['linear_independent_targets']:raise ValueError('Independent historical accounting mismatch')
    if len([r for r in scores if r['kind']=='legal-probe'])!=960:raise ValueError('Probe rows missing')
    shutil.copy2(root/'.runtime/phase5d-post-tests.xml',folder/'postrun-regressions.xml')
    write_json(folder/'postrun-test-receipt.json',post)
    write_json(folder/'validation-gates.json',{key:'passed' for key in ('G0','G1','G2','G3','G4','G5')})
    result={'status':'passed','verification_new_score_calls':0,'historical_preservation':preserved,
        'forward_calls':len(calls),'full_tests':post['counts'],'independent_checks':'Sealed inputs, source/dependencies, finite five-vectors, argmax, exact linear sums, PPO normalization, snapshot provenance and call ledger',
        'new_training_actions':0,'new_policy_environment_actions':0,'independent_linear_targets':independent_linear}
    write_json(folder/'verification.json',result);return result


def score_record_from_values(row,values):
    if values.shape!=(5,) or not np.isfinite(values).all():raise ValueError('Invalid derived scores')
    return int(values.argmax())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('validate','postvalidate','prepare','analyze','verify','report'))
    command=parser.parse_args().command
    if command=='report':
        from .partial_phase5d_report import report
        result=report()
    elif command in ('validate','postvalidate'): result=validate(post=command=='postvalidate')
    else:result=globals()[command]()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
