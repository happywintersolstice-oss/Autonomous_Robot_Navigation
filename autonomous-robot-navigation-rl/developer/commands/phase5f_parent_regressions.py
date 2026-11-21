"""Unchanged parent regressions in a new regression-only lease, never old CLIs."""
from pathlib import Path
import json
import os
import subprocess
import sys
from time import perf_counter

from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_instrumentation_trace import checksum,MechanicalLedger

BASE='192ba8dca9957ac6d6b38965956b06f05b694548'


def run():
    folder=ROOT/'.runtime/phase5f-parent-regressions';folder.mkdir(exist_ok=False)
    parent=ROOT/'artifacts/partial-dynamic-navigation-v1/phase5e-training-instrumentation-v1'
    old=read_json(parent/'attempt-plan.json')
    names=subprocess.check_output(['git','-c','core.longpaths=true','ls-tree','--full-tree','-r','--name-only',BASE,'--',
        'autonomous-robot-navigation-rl/developer/tests'],cwd=ROOT,text=True).splitlines()
    tests=[str(ROOT.parent/n) for n in names if Path(n).name.startswith('test_') and n.endswith('.py')]
    assert len(tests)==51 and any(Path(p).name=='test_partial_phase5e.py' for p in tests)
    lease=folder/'phase5e-regression-fixtures';lease.mkdir()
    write_json(lease/'attempt-plan.json',{'selected_pairs':old['selected_pairs'],'source_digest':old['source_digest'],
        'library_digest':old['library_digest'],'scope':'Regression-only fixture lease, not the sealed Phase 5E validation/namespace'})
    (lease/'model-config.json').write_bytes((parent/'model-config.json').read_bytes())
    plan={'test_files':tests,'junit':str(folder/'regressions.xml'),'receipt':str(folder/'parent-process-operations.json'),
        'base_commit':BASE,'driver_source_sha256':sha(Path(__file__)),'research_actions':0,
        'scope':'All unchanged parent tests, including Phase5E tests, in fresh regression-only scratch; no old one-shot CLI/namespace is rerun'}
    write_json(folder/'plan.json',plan)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PHASE5E_OUTPUT=str(lease),PHASE5E_LEDGER=str(folder/'child-mechanics.jsonl'))
    started=perf_counter()
    with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
        process=subprocess.run([sys.executable,'-u','-c',
            'import sys; from robot_navigation.partial_phase5e_legacy import run; sys.exit(run(sys.argv[1]))',str(folder/'plan.json')],
            cwd=ROOT,env=env,stdout=out,stderr=err)
    parent_counts=read_json(folder/'parent-process-operations.json')
    child=MechanicalLedger(folder/'child-mechanics.jsonl').totals()
    write_json(folder/'receipt.json',{'exit_code':process.returncode,'classification':'unchanged_parent_regression',
        'wall_seconds':perf_counter()-started,'parent_process':parent_counts,'child_fixture_mechanics':child,
        'scientific_actions':0,'new_Phase5F_synthetic_actions':0,'old_Phase5E_validation_rerun':False,
        'combined_environment_transitions':parent_counts['operations']['completed_environment_transitions']+child['transitions'],
        'combined_optimizer_and_partial_linear_steps':parent_counts['operations']['completed_neural_optimizer_steps']+
            parent_counts['operations']['completed_linear_updates']+child['optimizer_steps']})
    if process.returncode:raise SystemExit(process.returncode)
    print(json.dumps(read_json(folder/'receipt.json'),indent=2))


if __name__=='__main__':run()
