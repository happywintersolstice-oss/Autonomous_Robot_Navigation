"""All unchanged Phase 5F parent tests, separately counted legacy mechanics."""
from pathlib import Path
import json,os,subprocess,sys
from time import perf_counter
from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_instrumentation_trace import MechanicalLedger
from robot_navigation.partial_phase5f_io import git

BASE='6fb6aac124067b00016f5514ebbc66ef777dea33'

def run():
    folder=ROOT/'.runtime/phase5g-parent-regressions';folder.mkdir(exist_ok=False)
    parent=ROOT/'artifacts/partial-dynamic-navigation-v1/phase5e-training-instrumentation-v1';old=read_json(parent/'attempt-plan.json')
    names=git(ROOT,'ls-tree','--full-tree','-r','--name-only',BASE,'--','autonomous-robot-navigation-rl/developer/tests').splitlines()
    tests=[str(ROOT.parent/n) for n in names if Path(n).name.startswith('test_') and n.endswith('.py')]
    assert len(tests)==52 and any(Path(p).name=='test_partial_phase5f.py' for p in tests)
    lease=folder/'phase5e-regression-fixtures';lease.mkdir()
    write_json(lease/'attempt-plan.json',{'selected_pairs':old['selected_pairs'],'source_digest':old['source_digest'],
        'library_digest':old['library_digest'],'scope':'Fresh legacy regression fixtures; no historical study reopened'})
    (lease/'model-config.json').write_bytes((parent/'model-config.json').read_bytes())
    plan={'test_files':tests,'junit':str(folder/'regressions.xml'),'receipt':str(folder/'parent-process-operations.json'),
        'base_commit':BASE,'driver_source_sha256':sha(Path(__file__)),'research_actions':0,
        'scope':'Complete unchanged parent suite, including E mechanical fixtures and F offline trace tests; no scientific rerun'}
    write_json(folder/'plan.json',plan)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PHASE5E_OUTPUT=str(lease),PHASE5E_LEDGER=str(folder/'child-mechanics.jsonl'),
        GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='core.excludesFile',GIT_CONFIG_VALUE_0=str(ROOT/'.runtime/phase5g-parent-scope.exclude'),
        GIT_CONFIG_KEY_1='safe.directory',GIT_CONFIG_VALUE_1='*')
    started=perf_counter()
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        p=subprocess.run([sys.executable,'-u','-c','import sys; from robot_navigation.partial_phase5e_legacy import run; sys.exit(run(sys.argv[1]))',str(folder/'plan.json')],cwd=ROOT,env=env,stdout=out,stderr=err)
    counts=read_json(folder/'parent-process-operations.json');child=MechanicalLedger(folder/'child-mechanics.jsonl').totals()
    result={'exit_code':p.returncode,'classification':'unchanged_parent_regression','wall_seconds':perf_counter()-started,
        'parent_process':counts,'child_fixture_mechanics':child,'scientific_actions':0,'new_Phase5G_synthetic_actions':0,
        'combined_environment_transitions':counts['operations']['completed_environment_transitions']+child['transitions'],
        'combined_neural_optimizer_steps':counts['operations']['completed_neural_optimizer_steps']+child['optimizer_steps'],
        'partial_linear_updates':counts['operations']['completed_linear_updates']}
    write_json(folder/'receipt.json',result);print(json.dumps(result,indent=2))
    if p.returncode:raise SystemExit(p.returncode)

if __name__=='__main__':run()
