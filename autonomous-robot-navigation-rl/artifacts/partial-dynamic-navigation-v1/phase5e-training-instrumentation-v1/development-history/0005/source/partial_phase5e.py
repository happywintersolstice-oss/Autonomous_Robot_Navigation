"""Phase 5E engineering validation only: validate, audit, verify. No research run."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
import xml.etree.ElementTree as ET

import yaml

from .core import ROOT,read_json,write_json
from .release import sha
from .partial_training_instrumentation import library_contract,methods,check_compatibility
from .partial_instrumentation_trace import MechanicalLedger,read_trace,checksum
from .partial_instrumentation_audit import audit_records

STUDY='phase5e-training-instrumentation-v1'
BASE='fe200034e816b10352453471f9c680b7c77d6ca4'
BRIEF='CODEX_PHASE5E_TRAINING_INSTRUMENTATION.md'


def location(root=ROOT):return root/'artifacts/partial-dynamic-navigation-v1'/STUDY


def git(root,*args):
    return subprocess.check_output(['git','-c','core.longpaths=true','-c','safe.directory='+str(root.resolve()),*args],cwd=root,text=True)


def sources(root=ROOT):
    paths=list((root/'src/robot_navigation').glob('partial_*instrumentation*.py'))+list((root/'src/robot_navigation').glob('partial_phase5e*.py'))
    paths+=[root/'developer/tests/test_partial_phase5e.py',root/'docs/partial_phase5e_protocol.md',
        root/'configs/partial_phase5e_v1.yaml',root/'configs/partial_phase5e_library_v1.json',root.parent/BRIEF]
    return {p.relative_to(root.parent).as_posix():sha(p) for p in sorted(set(paths))}


def checked_input(root,path,pins):
    p=(root/path).resolve()
    if not p.is_relative_to(root.resolve()) or any(s.lower() in ('final','test','unseen','heldout','held-out') for s in p.parts):
        raise ValueError('Forbidden final/unseen or escaping input')
    name=p.relative_to(root).as_posix()
    if name not in pins:raise ValueError('Input not registered')
    if sha(p)!=pins[name]:raise ValueError('Input checksum mismatch')
    return p


def preserve(root=ROOT):
    folder=location(root)
    base=read_json(folder/'base-inventory.json' if (folder/'base-inventory.json').exists() else root/'.runtime/phase5e-base-inventory.json')
    for name,h in base['files'].items():
        if sha(root.parent/name)!=h:raise ValueError('Previous source/evidence changed: '+name)
    d=root/'artifacts/partial-dynamic-navigation-v1/phase5d-goal-credit-diagnosis-v1'
    for name,h in read_json(d/'output-hashes.json')['files'].items():
        if sha(d/name)!=h:raise ValueError('Historical Phase 5D evidence changed: '+name)
    # The old source-glob guards cannot accept new modules. Check their recorded
    # per-file dependencies, not a rewritten or rerun historical one-shot study.
    plan=read_json(d/'analysis-plan.json')
    for name,h in plan['source_inputs'].items():
        if sha(root/name)!=h:raise ValueError('Historical dependency changed: '+name)
    original=read_json(folder/'original-inventory.json' if (folder/'original-inventory.json').exists() else root/'.runtime/phase5e-original-inventory.json')
    common=Path(git(root,'rev-parse','--git-common-dir').strip())
    original_root=(root/common).resolve().parent if not common.is_absolute() else common.parent
    for name,h in original['files'].items():
        if sha(original_root/name)!=h:raise ValueError('Original dirty workspace file changed: '+name)
    if git(original_root,'rev-parse','HEAD').strip()!=original['head']:raise ValueError('Main HEAD changed')
    if git(original_root,'status','--porcelain','--untracked-files=all')!=original['status']:raise ValueError('Main working-tree status changed')
    if git(original_root,'diff','--cached','--binary')!=original['index']:raise ValueError('Main index changed')
    for name,commit in original['historical_refs'].items():
        if git(original_root,'rev-parse',name).strip()!=commit:raise ValueError('Historical branch changed: '+name)
    return {'status':'passed','parent_commit':base['head'],'parent_tree':base['tree'],'parent_files_checked':len(base['files']),
        'original_files_checked':len(original['files']),'main_head':original['head'],'historical_refs':original['historical_refs'],
        'base_inventory_sha256':sha(folder/'base-inventory.json' if (folder/'base-inventory.json').exists() else root/'.runtime/phase5e-base-inventory.json'),
        'original_inventory_sha256':sha(folder/'original-inventory.json' if (folder/'original-inventory.json').exists() else root/'.runtime/phase5e-original-inventory.json'),
        'historical_data_parsing':'none; byte checks only; no old study rerun'}


def installed_audit():
    import torch,numpy,gymnasium
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
    from stable_baselines3.common.vec_env import DummyVecEnv
    functions=dict(methods(),**{'OffPolicyAlgorithm._store_transition':OffPolicyAlgorithm._store_transition,
        'DummyVecEnv.step_wait':DummyVecEnv.step_wait})
    rows={};files={}
    for name,function in functions.items():
        path=Path(inspect.getsourcefile(function));text=inspect.getsource(function)
        rows[name]={'sha256':hashlib.sha256(text.encode()).hexdigest(),'source':text,'path':str(path),'first_line':inspect.getsourcelines(function)[1]}
        files[str(path)]=sha(path)
    for module in (torch,torch._C,numpy,numpy.random.mtrand,gymnasium):
        path=Path(module.__file__);files[str(path)]=sha(path)
    return {'contract':library_contract(),'call_paths':rows,'dependency_file_sha256':files}


def test_counts(path):
    tree=ET.parse(path).getroot();suites=[tree] if tree.tag=='testsuite' else list(tree.findall('testsuite'))
    return {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}


def settings(root=ROOT):
    c=yaml.safe_load((root/'configs/partial_phase5e_v1.yaml').read_text())
    if (c['base_commit']!=BASE or c['branch']!='feature/'+STUDY or c['enabled_by_default'] is not False or
        [c[k] for k in ('scientific_training_actions','research_assessment_actions','sealed_final_evaluations')]!=[0,0,0] or
        [c[k] for k in ('synthetic_transition_cap','synthetic_optimizer_cap','synthetic_wall_seconds','artifact_byte_cap')]!=[1000,500,1200,134217728]
        or c['expected_paired_transitions']!=480 or c['expected_paired_optimizer_steps']!=108 or c['equality_tolerance']!=0):
        raise ValueError('Registered Phase 5E envelope changed')
    return c


def checked_attempt(folder,root=ROOT):
    plan=read_json(folder/'attempt-plan.json')
    if sha(folder/'attempt-plan.json')!=read_json(folder/'attempt-seal.json')['sha256']:raise ValueError('Attempt seal changed')
    if plan['sources']!=sources(root) or plan['library']!=library_contract():raise ValueError('Attempt source/library changed')
    if sha(folder/'library-audit.json')!=plan['library_audit_sha256']:raise ValueError('Library source audit changed')
    for name,h in read_json(folder/'library-audit.json')['dependency_file_sha256'].items():
        if sha(Path(name))!=h:raise ValueError('Installed dependency file changed: '+name)
    for name,h in plan['source_snapshot'].items():
        if sha(folder/'source'/name)!=h:raise ValueError('Minimal source snapshot changed')
    return plan


def validate(root=ROOT,development=False,development_pair=None):
    c=settings(root);check_compatibility();preservation=preserve(root)
    if git(root,'rev-parse','HEAD').strip()!=BASE or git(root,'branch','--show-current').strip()!='feature/'+STUDY:
        raise ValueError('Wrong reviewed base or new feature branch')
    ledger_path=root/'.runtime/phase5e-operations.jsonl';ledger=MechanicalLedger(ledger_path);totals=ledger.totals()
    selected=[(a,b) for a in ('q-learning','sarsa','dqn','ppo') for b in ('B0-original','B1-relative-x4')]
    if development_pair:
        if not development:raise ValueError('Targeted checks are development only; final always validates every pair')
        selected=[tuple(development_pair.split(':'))]
        if selected[0] not in [(a,b) for a in ('q-learning','sarsa','dqn','ppo') for b in ('B0-original','B1-relative-x4')]:raise ValueError('Unknown mechanical pair')
    needed_transitions=sum({'q-learning':16,'sarsa':16,'dqn':8,'ppo':200}[a] for a,b in selected)
    needed_updates=sum({'q-learning':16,'sarsa':16,'dqn':2,'ppo':20}[a] for a,b in selected)
    if totals['transitions']+needed_transitions>1000 or totals['optimizer_steps']+needed_updates>500:
        raise RuntimeError('INCOMPLETE_VALIDATION: insufficient remaining cumulative budget for paired validation')
    if development:
        parent=root/'.runtime/phase5e-development';parent.mkdir(exist_ok=True)
        folder=parent/f'{len(list(parent.iterdir()))+1:04d}'
    else:folder=location(root)
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n',encoding='utf-8',newline='\n')
    for name,target in [('phase5e-base-inventory.json','base-inventory.json'),('phase5e-original-inventory.json','original-inventory.json')]:
        shutil.copy2(root/'.runtime'/name,folder/target)
    library=installed_audit();write_json(folder/'library-audit.json',library)
    model=yaml.safe_load((root/'configs/partial_feasibility_v1.yaml').read_text())['model']
    write_json(folder/'model-config.json',model)
    inputs=sources(root);snapshot=folder/'source';snapshot.mkdir()
    for name in inputs:shutil.copy2(root.parent/name,snapshot/Path(name).name)
    old_tests=git(root,'ls-tree','--full-tree','-r','--name-only',BASE,'--','autonomous-robot-navigation-rl/developer/tests').splitlines()
    old_tests=[str((root.parent/name).resolve()) for name in old_tests if Path(name).name.startswith('test_') and name.endswith('.py')]
    if len(old_tests)<20 or any(Path(name).name=='test_partial_phase5e.py' for name in old_tests):raise ValueError('Unchanged parent test inventory is incomplete')
    plan={'study_id':STUDY,'development':development,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'base_commit':BASE,'base_tree':preservation['parent_tree'],'settings':c,'sources':inputs,'source_digest':checksum(inputs),
        'library':library_contract(),'library_digest':checksum(library_contract()),'library_audit_sha256':sha(folder/'library-audit.json'),
        'source_snapshot':{p.name:sha(p) for p in snapshot.iterdir()},'preservation':preservation,
        'legacy_test_files':old_tests,'selected_pairs':selected,'budget_before':totals,'user_scope_clarification':'Legacy regressions unchanged and separately counted; cumulative new Phase 5E caps apply only to Phase 5E tests'}
    write_json(folder/'attempt-plan.json',plan);write_json(folder/'attempt-seal.json',{'sha256':sha(folder/'attempt-plan.json')})
    environment=dict(__import__('os').environ,PYTHONPATH=str(root/'src'),PHASE5E_OUTPUT=str(folder),PHASE5E_LEDGER=str(ledger_path))
    start=perf_counter();ledger.record('session_start',str(folder),time=__import__('time').time())
    try:
        test_nodes=['developer/tests/test_partial_phase5e.py']
        if development_pair:
            a,b=selected[0]
            test_nodes=[f'developer/tests/test_partial_phase5e.py::test_independent_on_off_bitwise_equality_all_models_and_arms[{b}-{a}]']
            if a=='dqn':test_nodes.append(f'developer/tests/test_partial_phase5e.py::test_actual_dqn_samples_targets_timeouts_duplicates_and_sync[{b}]')
            elif a=='ppo':test_nodes.append(f'developer/tests/test_partial_phase5e.py::test_actual_ppo_bootstrap_GAE_permutation_normalization_losses_stop[{b}]')
            else:test_nodes.append(f'developer/tests/test_partial_phase5e.py::test_actual_linear_normalized_update_and_sampled_SARSA_action[{b}-{a}]')
        result=subprocess.run([sys.executable,'-m','pytest','-q',*test_nodes,
            '--junitxml='+str(folder/'focused-regressions.xml')],cwd=root,env=environment,capture_output=True,text=True)
        (folder/'focused.log').write_text(result.stdout+result.stderr,encoding='utf-8');print(result.stdout+result.stderr,end='',flush=True)
    finally:
        ledger.record('session_end',str(folder),seconds=perf_counter()-start)
        shutil.copy2(ledger_path,folder/'operations.jsonl')
    counts=test_counts(folder/'focused-regressions.xml')
    if result.returncode or any(counts[k] for k in ('failures','errors','skipped')):
        write_json(folder/'validation-failure.json',{'verdict':'INSTRUMENTATION_BLOCKED','stage':'focused mechanics',
            'counts':counts,'cumulative_operations':ledger.totals(),'source_fixes_allowed_before_final_seal':development,
            'attempt_retry_allowed':False})
        return {'verdict':'INSTRUMENTATION_BLOCKED','folder':str(folder),'counts':counts,'operations':ledger.totals()}
    checked_attempt(folder,root)
    if not development:
        legacy_plan={'test_files':old_tests,'junit':str(folder/'legacy-regressions.xml'),'receipt':str(folder/'legacy-operations.json')}
        write_json(folder/'legacy-plan.json',legacy_plan)
        legacy=subprocess.run([sys.executable,'-c','import sys; from robot_navigation.partial_phase5e_legacy import run; sys.exit(run(sys.argv[1]))',str(folder/'legacy-plan.json')],
            cwd=root,env=environment,capture_output=True,text=True)
        (folder/'legacy.log').write_text(legacy.stdout+legacy.stderr,encoding='utf-8');print(legacy.stdout+legacy.stderr,end='',flush=True)
        if legacy.returncode:
            write_json(folder/'validation-failure.json',{'verdict':'INSTRUMENTATION_BLOCKED','stage':'unchanged legacy regressions','exit_code':legacy.returncode})
            return {'verdict':'INSTRUMENTATION_BLOCKED','stage':'legacy regressions'}
    checked_attempt(folder,root);preserve(root)
    if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>c['artifact_byte_cap']:raise RuntimeError('Validation artifact ceiling')
    write_json(folder/'validation-complete.json',{'status':'passed','focused':counts,'operations':ledger.totals(),'development':development})
    return {'status':'passed','folder':str(folder),'focused':counts,'operations':ledger.totals(),'development':development}


def audit(root=ROOT):
    folder=location(root);checked_attempt(folder,root)
    if (folder/'validation-failure.json').exists():raise ValueError('Final validation blocked; no retry')
    complete=read_json(folder/'validation-complete.json');pairs=[];numerical=[]
    for path in sorted((folder/'pairs').iterdir()):
        off=read_json(path/'off/witness.json');on=read_json(path/'on/witness.json')
        if off!=on:raise ValueError('Independent witness mismatch')
        equivalence=read_json(path/'equivalence.json')
        if checksum(off)!=equivalence['independent_witness_sha256']:raise ValueError('Witness content changed')
        rows=read_trace(path/'on/trace');checks=audit_records(rows)
        pairs.append(equivalence);numerical.append({'pair':path.name,**checks})
    ledger=MechanicalLedger(root/'.runtime/phase5e-operations.jsonl');t=ledger.totals()
    if t['transitions']>1000 or t['optimizer_steps']>500 or t['wall_seconds']>1200:raise ValueError('Cumulative budget exceeded')
    legacy=read_json(folder/'legacy-operations.json');legacy_tests=test_counts(folder/'legacy-regressions.xml')
    if any(legacy_tests[k] for k in ('failures','errors','skipped')):raise ValueError('Legacy regression gate failed')
    history=root/'.runtime/phase5e-development'
    if history.exists():
        target=folder/'development-history';target.mkdir(exist_ok=False)
        # Retain compact genuine failures and receipts; reference source hashes,
        # do not copy duplicate models or full historical experiment trees.
        for attempt in sorted(history.iterdir()):
            dest=target/attempt.name;dest.mkdir()
            for name in ('attempt-plan.json','attempt-seal.json','focused.log','focused-regressions.xml','validation-failure.json','validation-complete.json','operations.jsonl'):
                if (attempt/name).exists():shutil.copy2(attempt/name,dest/name)
    write_json(folder/'preservation.json',preserve(root));write_json(folder/'numerical-audit.json',numerical)
    write_json(folder/'equivalence-summary.json',pairs)
    summary={'verdict':'INSTRUMENTATION_VALIDATED_NO_RESEARCH_TRAINING','gates':{g:'passed' for g in ('G0','G1','G2','G3','G4','G5')},
        'new_synthetic_cumulative':t,'final_focused_tests':complete['focused'],'legacy_tests':legacy_tests,'legacy_activity':legacy,
        'scientific_training_actions':0,'research_assessment_actions':0,'sealed_final_evaluations':0,
        'bitwise_on_off_pairs':len(pairs),'unsupported':['unmatched library versions/source','GPU','multiple environments','VecNormalize','SDE','memory-optimized/n-step replay','prefilled replay without identity sidecar','active debugger trace'],
        'future_proposal':'Separately register a Fixture A B1 DQN/PPO instrumentation pilot: seed 23121, 500 actions each, no evaluation or algorithm change. Requires new user authorization; not started.'}
    write_json(folder/'summary.json',summary)
    from .partial_phase5e_report import render
    render(summary,pairs,numerical,root)
    write_json(folder/'output-hashes.json',{'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='output-hashes.json'},
        'reports':{p.relative_to(root).as_posix():sha(p) for p in (root/'docs/partial_phase5e_results.md',root/'docs/partial_phase5e_diagnosis.md')}})
    return summary


def verify(root=ROOT):
    folder=location(root);plan=checked_attempt(folder,root);pins=read_json(folder/'output-hashes.json')
    for name,h in pins['files'].items():
        if sha(folder/name)!=h:raise ValueError('Final evidence changed: '+name)
    for name,h in pins['reports'].items():
        if sha(root/name)!=h:raise ValueError('Final report changed: '+name)
    for path in (folder/'pairs').iterdir():
        if read_json(path/'on/witness.json')!=read_json(path/'off/witness.json'):raise ValueError('Paired equivalence changed')
        audit_records(read_trace(path/'on/trace'))
    preserve(root)
    return {'status':'passed','verdict':read_json(folder/'summary.json')['verdict'],'new_scientific_actions':0,
        'verification_environment_actions':0,'verification_optimizer_steps':0,'verification_model_predictions':0,
        'frozen_source_digest':plan['source_digest'],'evidence_files':len(pins['files'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('validate','audit','verify'))
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--development-pair',choices=[a+':'+b for a in ('q-learning','sarsa','dqn','ppo') for b in ('B0-original','B1-relative-x4')])
    args=parser.parse_args()
    if args.development and args.command!='validate':parser.error('--development applies only to mechanical validation')
    if args.development_pair and not args.development:parser.error('--development-pair requires --development')
    try:result=validate(development=args.development,development_pair=args.development_pair) if args.command=='validate' else globals()[args.command]()
    except BaseException:
        # Preserve a genuine partial final receipt, never turn an exception into a pass.
        folder=location()
        if folder.exists() and not (folder/'summary.json').exists():
            write_json(folder/'interruption.json',{'verdict':'INCOMPLETE_VALIDATION','traceback':traceback.format_exc()})
        raise
    print(json.dumps(result,indent=2,allow_nan=False))
    if result.get('verdict') in ('INSTRUMENTATION_BLOCKED','INCOMPLETE_VALIDATION'):sys.exit(1)


if __name__=='__main__':main()
