"""Phase 5C C0 only: frozen historical coverage audit, with no training command."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import pickle
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import yaml

from .core import ROOT,digest,read_json,write_json
from .release import sha
from .partial_agents import ALGORITHMS
from .partial_contract import configuration,signature,ACTION_NAMES
from .partial_commands import read_records
from .partial_training import validate_training_trace
from .partial_evaluation import validate_replay
from .partial_features_scaled import ARMS,load_scaled_policy
from .partial_pilot_diagnostics import decision_values,preserve_random_state
from .partial_phase5b import junit,_resource_usage
from .partial_phase5b_fixtures import strata
from .partial_phase5c_fixtures import probe_manifest,uniform_schedule
from .partial_phase5c_coverage import coverage,score_record,direction,DIRECTIONS,sampling_gate,schedule_comparison,coordinates

STUDY='phase5c-directional-coverage-v1'
BASE='ae3ae73bf7ba5cbe36dcf879125f15fd14c43f4e'


def location(root=ROOT): return root/'artifacts/partial-dynamic-navigation-v1'/STUDY
def old_location(root=ROOT): return root/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1'


def settings(root=ROOT):
    c=yaml.safe_load((root/'configs/partial_phase5c_v1.yaml').read_text(encoding='utf-8'))
    if (c['study_id']!=STUDY or c['reviewed_commit']!=BASE or c['historical_checkpoints']!=[0,2000,6000]
            or c['sampling_gate']!={'feature_arm':'B1-relative-x4','checkpoint':6000,'minimum_episode_starts':64,
                'underexposed_direction_share_below':.15,'underexposed_case_share_below':.0625,
                'impaired_expanded_goals_at_most':1,'sparse_near1_direction_decisions_below':50,
                'sparse_direction_goal_rewards_below':10,'schedule_direction_share_gain_at_least':.05,
                'schedule_case_share_gain_at_least':.03,'simulation_seed':23041}
            or c['analysis_wall_seconds']!=900 or c['artifact_byte_ceiling']!=134217728
            or c['simulation_prefixes']!=[8,32,64,128,256,512]
            or c['conditional_training']!={'feature_version':'partial-visible-relative-scale4-v1','seed':23041,
                'fixture_seed':23042,'actions_per_arm':4000,'total_training_ceiling':32000,
                'checkpoints':[0,2000,4000],'evaluation_action_ceiling':200000,'suite_wall_seconds':1800,
                'per_arm_wall_seconds':300,'artifact_byte_ceiling':536870912,
                'screen':{'new_goals':12,'new_gain':3,'minimum_per_direction':2,'expanded_goals':10,'expanded_gain':2,'original_goals':2}}):
        raise ValueError('Predeclared C0 protocol changed')
    return c


def source_inputs(root=ROOT):
    paths=list((root/'src/robot_navigation').glob('*.py'))
    paths+=list((root/'developer/tests').glob('test_partial_*.py'))
    paths+=[root/p for p in ['configs/partial_phase5c_v1.yaml','configs/partial_phase5b_v1.yaml',
        'configs/partial_feasibility_v1.yaml','configs/partial_dynamic.yaml','docs/partial_phase5c_protocol.md',
        'pyproject.toml','requirements-lock.txt']]
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


class Evidence:
    def __init__(self,root): self.root,self.files=root,{}
    def data(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()): raise ValueError('Evidence path escaped application')
        payload=path.read_bytes(); self.files[path.relative_to(self.root).as_posix()]=hashlib.sha256(payload).hexdigest()
        return payload
    def json(self,path): return json.loads(self.data(path).decode('utf-8'))
    def records(self,path): return [json.loads(line) for line in gzip.decompress(self.data(path)).decode('utf-8').splitlines()]
    def state(self,path): return pickle.loads(gzip.decompress(self.data(path)))  # trusted, hash-pinned local archive only


def preserve_history(root=ROOT):
    stored=location(root)/'base-inventory.json'
    inventory=read_json(stored if stored.exists() else root/'.runtime/phase5c-base-inventory.json')
    changed=[p for p,h in inventory['files'].items() if not (root.parent/p).is_file() or sha(root.parent/p)!=h]
    if changed: raise ValueError('A previous file changed: '+repr(changed))
    checked={}
    for study in ('feasibility-pilot-v1','phase5b-a-only-v1'):
        folder=root/'artifacts/partial-dynamic-navigation-v1'/study
        pins=read_json(folder/'delivery-manifest.json'); count=0
        for name,h in pins['artifact_files'].items():
            if sha(folder/name)!=h: raise ValueError('Historical artifact changed: '+name)
            count+=1
        for section,base in [('active_files',root),('implementation_files',root.parent)]:
            for name,h in pins.get(section,{}).items():
                payload=(base/name).read_bytes()
                if name in pins.get('normalized_implementation_paths',{}): payload=payload.replace(b'\r\n',b'\n')
                if hashlib.sha256(payload).hexdigest()!=h: raise ValueError('Historical implementation changed: '+name)
                count+=1
        checked[study]=count
    return {'status':'passed','base_files_checked':len(inventory['files']),'changed':[], 'delivery_pins_checked':checked}


def validate_tests(root=ROOT):
    before=source_inputs(root); target=root/'.runtime/phase5c-tests.xml'
    result=subprocess.run([sys.executable,'-m','pytest','-q',f'--junitxml={target}'],cwd=root,capture_output=True,text=True)
    (root/'.runtime/phase5c-tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout+result.stderr,end='')
    if result.returncode or before!=source_inputs(root): raise ValueError('Full regression/source gate failed')
    counts=junit(target); write_json(root/'.runtime/phase5c-test-inputs.json',before)
    return counts


def prepare(root=ROOT):
    c=settings(root); preserve_history(root)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
    if head!=BASE or branch!='feature/'+STUDY: raise ValueError('Wrong reviewed parent or new branch')
    if junit(root/'.runtime/phase5c-baseline-tests.xml')['tests']!=366: raise ValueError('Unexpected historical baseline test count')
    tests=junit(root/'.runtime/phase5c-tests.xml'); inputs=source_inputs(root)
    if inputs!=read_json(root/'.runtime/phase5c-test-inputs.json'): raise ValueError('Tests/source stale')
    folder=location(root); folder.mkdir(parents=True,exist_ok=False)
    for name in ('phase5c-base-inventory.json','phase5c-original-inventory.json'):
        # Inventory manifests only, no duplicate historical file trees.
        shutil.copy2(root/'.runtime'/name,folder/('base-inventory.json' if 'base-' in name else 'original-inventory.json'))
    shutil.copy2(root/'.runtime/phase5c-tests.xml',folder/'preaudit-regressions.xml')
    shutil.copy2(root/'.runtime/phase5c-baseline-tests.xml',folder/'baseline-regressions.xml')
    probes=probe_manifest(); write_json(folder/'probe-manifest.json',probes)
    write_json(folder/'data-dictionary.json',{
        'version':'phase5c-coverage-denominators-v1',
        'episode_starts':'Every logged reset, including empty autoresets; not equivalent to action decisions.',
        'prefixes':'0,2000,6000 actions per old arm; 2000 is contained in 6000. Unique old total is 48000.',
        'states':'SHA256 of full pre-action float32 409 vector, including history and elapsed time.',
        'current_directions':list(DIRECTIONS),'diagonals':'A separate disjoint category, not two axis samples.',
        'actual_vs_attempted':'Actual distance uses returned coordinates. A collision can preserve actual distance while attempting an increase.',
        'near1':'Distance exactly one: selected reducing action immediately reaches the goal on open A.',
        'near2':'Distance two: reducing action is progress, not immediate arrival; diagonal cases may have two reducing choices.',
        'joint_bins':'direction|quadrant|boundary|distance|visits|known; unobserved bins include impossible combinations.',
        'zero_samples':'observed=false, count=0, rate=null. Not an estimate of zero probability.',
        'greedy_comparison':'First argmax and full maximum tie set are distinct. No realized epsilon coin was logged.',
        'history':'Clipped current-cell visits, known-cell count, elapsed band and blockage streak are derived only from permitted observation.',
        'PPO_signal':'Only the last saved rollout is available; positive bootstrapped timeout rewards are not labelled goal arrivals.',
        'probes':'Scores only, no policy.act, trainer.learn, policy.update or env.step; scripted unit-test mechanics are separate.'})
    source=folder/'source'; source.mkdir()
    for path in list((root/'src/robot_navigation').glob('partial_phase5c*.py'))+[root/'developer/tests/test_partial_phase5c.py',root/'configs/partial_phase5c_v1.yaml',root/'docs/partial_phase5c_protocol.md']:
        shutil.copy2(path,source/path.name)
    shutil.copy2(root.parent/'CODEX_PHASE5C_DIRECTIONAL_COVERAGE_DIAGNOSIS.md',source/'brief.md')
    plan={'study_id':STUDY,'registered_utc':datetime.now(timezone.utc).isoformat(),'settings':c,
        'source_inputs':inputs,'probe_hash':sha(folder/'probe-manifest.json'),'probe_count':len(probes['probes']),
        'brief_raw_sha256':sha(source/'brief.md'),'brief_lf_sha256':hashlib.sha256((source/'brief.md').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'baseline_commit':BASE,'baseline_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=root,text=True).strip(),
        'branch':branch,'source_snapshot':{p.name:sha(p) for p in source.iterdir()},'tests':tests,
        'runtime':{'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
            'dependencies':{n:version(n) for n in ('numpy','torch','stable-baselines3','gymnasium','pytest')}},
        'new_training_actions':0,'new_policy_environment_actions':0,'conditional_training_registered':False}
    write_json(folder/'analysis-plan.json',plan); write_json(folder/'analysis-seal.json',{'sha256':sha(folder/'analysis-plan.json')})
    return {'status':'prepared','analysis_plan_sha256':sha(folder/'analysis-plan.json'),'probes':len(probes['probes'])}


def checked_plan(root=ROOT):
    folder=location(root); plan=read_json(folder/'analysis-plan.json')
    if sha(folder/'analysis-plan.json')!=read_json(folder/'analysis-seal.json')['sha256'] or plan['source_inputs']!=source_inputs(root):
        raise ValueError('Analysis plan/source changed')
    if sha(folder/'probe-manifest.json')!=plan['probe_hash']: raise ValueError('Probe manifest changed')
    brief=root.parent/'CODEX_PHASE5C_DIRECTIONAL_COVERAGE_DIAGNOSIS.md'
    if hashlib.sha256(brief.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=plan['brief_lf_sha256']:
        raise ValueError('Work order changed')
    for name,h in plan['source_snapshot'].items():
        if sha(folder/'source'/name)!=h: raise ValueError('Minimal source snapshot changed')
    return plan


def learning_signal(reader,arm_folder,count,trace,cp):
    state=reader.state(arm_folder/f'{count:05}/continuation-state.pkl.gz')
    model=cp['numerical']; signal={'model_numerical':model,'parameter_fingerprint':cp['parameter_fingerprint']}
    steps=[r for r in trace if r['kind']=='step']
    if 'replay' not in state: return signal
    buffer=state['replay']; n=int(buffer.buffer_size if buffer.full else buffer.pos)
    if cp['algorithm']=='dqn':
        positive=int(np.count_nonzero(buffer.rewards[:n]>0))
        if positive!=sum(s['reward']>0 for s in steps) or n!=count: raise ValueError('DQN stored reward/count mismatch')
        signal['DQN_replay']={'transitions':n,'positive_goal_rewards':positive,'timeout_markers':int(np.count_nonzero(buffer.timeouts[:n]))}
    else:
        tail=steps[-n:] if n else []
        goals=sum(s['reward']>0 for s in tail)
        if int(np.count_nonzero(np.isclose(buffer.rewards[:n],9.99)))!=goals:
            raise ValueError('PPO last-rollout goal reward mismatch')
        signal['PPO_last_rollout']={'transitions':n,'raw_goal_rewards':goals,'raw_timeouts':sum(s['truncated'] for s in tail),
            'total_positive_buffer_rewards':int(np.count_nonzero(buffer.rewards[:n]>0)),
            'scope':'Last rollout only; positive timeout bootstrap rewards are not goal transitions',
            'returns_finite':bool(np.isfinite(buffer.returns[:n]).all())}
    return signal


def analyze(root=ROOT):
    plan=checked_plan(root); folder=location(root); started=perf_counter(); reader=Evidence(root)
    with (folder/'analysis.lock').open('x') as f: f.write(datetime.now(timezone.utc).isoformat())
    result={'status':'running','new_training_actions':0,'new_policy_environment_actions':0,
            'historical_unique_training_actions':0,'historical_assessment_episodes':0,'historical_assessment_actions':0,
            'checkpoint_prefix_sum_note':'2k is contained in 6k; overlapping prefix totals are not independent samples.',
            'systems':{},'score_probe_calls':0,'discrepancies':[]}
    def guard():
        if perf_counter()-started>plan['settings']['analysis_wall_seconds']: raise TimeoutError('C0 wall ceiling')
        if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>plan['settings']['artifact_byte_ceiling']:
            raise RuntimeError('C0 artifact ceiling')
    try:
        preserve_history(root)
        old=old_location(root); original=reader.json(old/'manifest.json'); old_summary=reader.json(old/'summary.json')
        if original['settings']['primary_seed']!=23021 or original['case_audit']['pair_overlap']!=0:
            raise ValueError('Unexpected original source protocol')
        if old_summary['status']!='completed' or old_summary['budget']['training_total']!=48000:
            raise ValueError('Incomplete or inconsistent old study')
        if reader.json(old/'final-verdict.json')['verdict']!='SCALING_HYPOTHESIS_NOT_SUPPORTED': raise ValueError('Wrong previous verdict')
        reader.data(old/'learning-curves.csv'); reader.data(old/'paired-results.csv')
        if junit(old/'postrun-regressions.xml')['tests']!=366: raise ValueError('Unexpected saved tests')
        reader.data(old/'postrun-regressions.xml')
        probes=read_json(folder/'probe-manifest.json')['probes']; finals={}; pairs=[]
        (folder/'coverage').mkdir(); (folder/'probes').mkdir(); (folder/'decisions').mkdir()
        for family in ALGORITHMS:
            result['systems'][family]={}
            for arm in ARMS:
                guard(); arm_folder=old/'training-and-evaluation-evidence'/family/arm
                raw=reader.data(arm_folder/'training-actions.jsonl.gz')
                records=[json.loads(line) for line in gzip.decompress(raw).decode('utf-8').splitlines()]
                if validate_training_trace(records,configuration(root))!=6000: raise ValueError('Historical trace mismatch')
                if [r['global_training_interaction'] for r in records if r['kind']=='step']!=list(range(1,6001)):
                    raise ValueError('Wrong historical interaction sequence')
                stream=reader.json(arm_folder/'training-case-stream.json')
                actual_stream=[{'episode':r['episode'],'case_id':r['scenario']['id'],'seed':r['seed']} for r in records if r['kind']=='reset']
                if stream!=actual_stream: raise ValueError('Recorded case stream mismatch')
                updates=reader.records(arm_folder/('linear-updates.jsonl.gz' if family in ('q-learning','sarsa') else 'optimizer-updates.jsonl.gz'))
                if any(not r['finite'] for r in updates): raise ValueError('Nonfinite old optimizer log')
                result['historical_unique_training_actions']+=6000; checkpoints=[]
                for count in (0,2000,6000):
                    guard(); target=arm_folder/f'{count:05}'; cp=reader.json(target/'summary.json')
                    prefix=raw[:cp['training_prefix_bytes']]
                    if prefix and hashlib.sha256(prefix).hexdigest()!=cp['training_prefix_sha256']: raise ValueError('Prefix checksum')
                    trace=[json.loads(line) for line in gzip.decompress(prefix).decode('utf-8').splitlines()] if prefix else []
                    cv=coverage(trace,original['training_cases'])
                    if cv['total']['decisions']!=count or cv['reset_indices']!=uniform_schedule(cv['reset_count'],23021):
                        raise ValueError('Uniform case sampling/count mismatch')
                    expected_goals=cp['training']['overall']['counts']['success'] if count else 0
                    if cv['total']['goal_rewards']!=expected_goals: raise ValueError('Completed goal signal mismatch')
                    signal=learning_signal(reader,arm_folder,count,trace,cp)
                    key='update' if family in ('q-learning','sarsa') else 'training_actions'
                    included=[r for r in updates if r[key]<=count]
                    expected=count if family in ('q-learning','sarsa') else max(cp['numerical']['optimizer_steps'],default=0)
                    if len(included)!=expected: raise ValueError('Optimizer count mismatch')
                    signal['recorded_optimizer_steps']=len(included)
                    path=target/cp['checkpoint']; reader.data(path); reader.data(path.with_suffix('.json'))
                    evaluated=reader.records(target/'evaluation.jsonl.gz'); quality=[]; probe_values=[]
                    with preserve_random_state():
                        policy=load_scaled_policy(path,configuration(root),arm); before=policy.state_fingerprint()
                        for entry in evaluated:
                            validate_replay(entry)
                            if entry['feature_signature']!=policy.signature: raise ValueError('Feature signature mismatch')
                            for t in entry['transitions']:
                                obs=np.asarray(t['observation'],dtype=np.float32); values=decision_values(policy,obs)
                                if score_record(obs,values)['first_argmax']!=t['action']: raise ValueError('Saved greedy action mismatch')
                                if any(not np.allclose(v,t['decision'][k],atol=1e-7,rtol=1e-6) for k,v in values.items()):
                                    raise ValueError('Saved score mismatch')
                            result['historical_assessment_episodes']+=1; result['historical_assessment_actions']+=len(entry['transitions'])
                            d=score_record(np.asarray(entry['initial_observation'],dtype=np.float32),entry['transitions'][0]['decision'])
                            quality.append({'case_id':entry['scenario']['id'],**strata(entry['scenario']),
                                'panel':'original' if len(quality)<4 else 'expanded','outcome':entry['metrics']['outcome'],
                                'metrics':entry['metrics'],'initial_observation':entry['initial_observation'],'first_decision':d})
                        for p in probes:
                            values=decision_values(policy,np.asarray(p['observation'],dtype=np.float32))
                            probe_values.append({'probe_id':p['id'],**score_record(np.asarray(p['observation'],dtype=np.float32),values)})
                            result['score_probe_calls']+=1
                        if before!=policy.state_fingerprint() or before!=cp['parameter_fingerprint']:
                            raise ValueError('Score pass changed policy weights')
                    if [q['case_id'] for q in quality]!=[c['id'] for c in original['original_cases']+original['expanded_cases']]:
                        raise ValueError('Wrong development matrix')
                    counts={d:sum(q['outcome']=='success' for q in quality[4:] if q['direction']==d) for d in DIRECTIONS[:4]}
                    node={'checkpoint':count,'coverage':cv,'learning_signal':signal,'expanded_direction_goals':counts,
                          'source_checkpoint_sha256':sha(path),'assessment_case_count':len(quality)}
                    stem=f'{family}-{arm}-{count:05}'
                    write_json(folder/'coverage'/f'{stem}.json',node)
                    write_json(folder/'decisions'/f'{stem}.json',quality)
                    # D4 discrepancy is descriptive, not required learned symmetry.
                    by_id={v['probe_id']:v for v in probe_values}
                    for p,v in zip(probes,probe_values):
                        if p['kind']=='D4-history':
                            base=np.asarray(by_id[p['base_id']]['scores']); perm=np.asarray(p['action_permutation'])
                            v['D4_score_linf']=float(np.max(np.abs(np.asarray(v['scores'])[perm]-base)))
                            v['D4_argmax_equivariant']=v['first_argmax']==p['action_permutation'][by_id[p['base_id']]['first_argmax']]
                    write_json(folder/'probes'/f'{stem}.json',probe_values)
                    checkpoints.append({'checkpoint':count,'coverage_file':f'coverage/{stem}.json',
                        'decision_file':f'decisions/{stem}.json','probe_file':f'probes/{stem}.json',
                        'actions':count,'resets':cv['reset_count'],'goal_rewards':cv['total']['goal_rewards'],
                        'expanded_direction_goals':counts})
                    if arm=='B1-relative-x4' and count==6000: finals[family]=node
                result['systems'][family][arm]=checkpoints
        result['sampling_gate']=sampling_gate(finals,plan['settings'])
        lengths=sorted(set(plan['settings']['simulation_prefixes']+[n['coverage']['reset_count'] for n in finals.values()]))
        write_json(folder/'schedule-simulations.json',{'seed':23041,'new_training_actions':0,
            'prefixes':[schedule_comparison(n,23041) for n in lengths],
            'scope':'Episode-start index schedules only, no learned actions or coverage guarantees.'})
        result['coverage_intervention_justified']=result['sampling_gate']['coverage_intervention_justified']
        result['historical_preservation']=preserve_history(root); guard()
        result.update(status='completed',verdict='TRAINING_CANDIDATE_REQUIRES_SEPARATE_REGISTRATION' if result['coverage_intervention_justified'] else 'DIAGNOSIS_ONLY_SAMPLING_NOT_JUSTIFIED')
    except BaseException as exc:
        result.update(status='failed',verdict='IMPLEMENTATION_BLOCKED',exception=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    result.update(elapsed_seconds=perf_counter()-started,resources=_resource_usage())
    write_json(folder/'coverage-audit.json',result)
    write_json(folder/'source-and-evidence-manifest.json',{'analysis_plan_sha256':sha(folder/'analysis-plan.json'),
        'base_commit':BASE,'base_tree':plan['baseline_tree'],'dependencies':plan['source_inputs'],
        'historical_evidence_read':reader.files,'minimal_source_snapshot':plan['source_snapshot'],
        'probe_manifest_sha256':sha(folder/'probe-manifest.json')})
    if not result.get('coverage_intervention_justified',False):
        write_json(folder/'final-verdict.json',{'verdict':result['verdict'],'coverage_intervention_justified':False,
            'new_training_actions':0,'new_policy_environment_actions':0,
            'reason':result.get('sampling_gate',{}).get('reason',result.get('exception'))})
    write_json(folder/'output-hashes.json',{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
        if p.is_file() and p.name!='output-hashes.json'})
    return {k:result[k] for k in ('status','verdict','new_training_actions','new_policy_environment_actions','elapsed_seconds')}


def verify(root=ROOT):
    plan=checked_plan(root); folder=location(root); result=read_json(folder/'coverage-audit.json')
    if result['status']!='completed': raise ValueError('No successful C0 audit to verify')
    for name,h in read_json(folder/'output-hashes.json').items():
        if sha(folder/name)!=h: raise ValueError('Audit output changed: '+name)
    manifest=read_json(folder/'source-and-evidence-manifest.json')
    for name,h in manifest['historical_evidence_read'].items():
        if sha(root/name)!=h: raise ValueError('Historical source changed: '+name)
    old=old_location(root); original=read_json(old/'manifest.json'); total=0; finals={}
    for family,arms in result['systems'].items():
        for arm,cps in arms.items():
            records=read_records(old/'training-and-evaluation-evidence'/family/arm/'training-actions.jsonl.gz')
            total+=sum(r['kind']=='step' for r in records)
            for cp in cps:
                node=read_json(folder/cp['coverage_file']); n=cp['checkpoint']; steps=[r for r in records if r['kind']=='step' and r['global_training_interaction']<=n]
                counts=Counter(direction(np.rint(np.asarray(s['observation'][2:4])*9).astype(int)-np.rint(np.asarray(s['observation'][:2])*9).astype(int)) for s in steps)
                if len(steps)!=node['coverage']['total']['decisions']: raise ValueError('Independent count mismatch')
                if any(counts[d]!=node['coverage']['marginals']['direction'][d]['decisions'] for d in DIRECTIONS):
                    raise ValueError('Independent current-direction mismatch')
                if node['coverage']['total']['goal_rewards']!=sum(s['reward']>0 for s in steps): raise ValueError('Independent goal count mismatch')
                if arm=='B1-relative-x4' and n==6000: finals[family]=node
    if total!=48000 or total!=result['historical_unique_training_actions']: raise ValueError('Lost historical actions')
    if sampling_gate(finals,plan['settings'])!=result['sampling_gate']: raise ValueError('Gate result changed')
    preserve_history(root)
    receipt={'status':'passed','historical_training_actions_independently_recounted':total,
        'checkpoint_prefixes':24,'new_training_actions':0,'new_policy_environment_actions':0,
        'historical_assessment_actions_replayed':result['historical_assessment_actions'],
        'historical_assessment_episodes_replayed':result['historical_assessment_episodes'],
        'analysis_plan_sha256':sha(folder/'analysis-plan.json')}
    write_json(folder/'replay-audit.json',receipt)
    write_json(folder/'gate-results.json',{'source_and_evidence':'passed','coverage_reconciliation':'passed',
        'saved_score_replay':'passed','sampling_intervention_justified':result['coverage_intervention_justified'],
        'training':'not run' if not result['coverage_intervention_justified'] else 'requires separate registration'})
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('validate','prepare','analyze','verify','report'))
    action=parser.parse_args().command
    if action=='report':
        from .partial_phase5c_report import report
        result=report()
    else: result={'validate':validate_tests,'prepare':prepare,'analyze':analyze,'verify':verify}[action]()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
