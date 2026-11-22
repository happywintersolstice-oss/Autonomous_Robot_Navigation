"""Continue delivery after runner interruption; never rerun Phase 5E mechanics.

The sealed focused tests have already completed. This helper only executes the
unchanged legacy test list, persists its separate counts, and completes the
existing validation receipt. Its own code is registered separately.
"""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys

from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5e import location,checked_attempt,preserve,test_counts
from robot_navigation.partial_instrumentation_trace import MechanicalLedger


def complete():
    folder=location();plan=checked_attempt(folder)
    if (folder/'validation-complete.json').exists():raise ValueError('Already completed; use read-only verify')
    (folder/'continuation.lock').open('x').close()
    focused=test_counts(folder/'focused-regressions.xml')
    if focused!={'tests':32,'failures':0,'errors':0,'skipped':0}:raise ValueError('Completed sealed focused validation required')
    ledger=MechanicalLedger(ROOT/'.runtime/phase5e-operations.jsonl');before=ledger.totals()
    if (before['transitions'],before['optimizer_steps'])!=(812,158):raise ValueError('Unexpected saved operation totals')
    own=Path(__file__).resolve();snapshot=folder/'source/phase5e_complete_validation.py'
    snapshot.write_bytes(own.read_bytes())
    write_json(folder/'continuation-plan.json',{'scope':'Unchanged legacy regression completion and receipts only',
        'utc':datetime.now(timezone.utc).isoformat(),'source_path':own.relative_to(ROOT).as_posix(),'source_sha256':sha(own),
        'original_scientific_source_digest':plan['source_digest'],'focused':focused,'new_operations_before':before,
        'rerun_focused':False,'extra_scientific_training':0,'extra_Phase5E_synthetic_operations':0})
    write_json(folder/'runner-interruption.json',{'interim_verdict':'INCOMPLETE_VALIDATION',
        'tool_result':'runner error: runner pipe closed before exit','tool_exit_code':-1,
        'last_saved_new_validation_result':'32 passed; focused-regressions.xml and focused.log preserved',
        'legacy_initial_completed_operations':'not_observed: runner closed before legacy receipt was written',
        'legacy_initial_test_result':'not_observed; no JUnit or legacy receipt survived',
        'new_Phase5E_synthetic_retried':False,'remaining_processes_when_checked':0,
        'resolution':'Complete unchanged parent suite separately; do not repeat passed focused tests'})
    legacy_plan=folder/'legacy-plan.json'
    if read_json(legacy_plan)['test_files']!=plan['legacy_test_files']:raise ValueError('Parent test list changed')
    stdout=folder/'legacy-continuation.log';stderr=folder/'legacy-continuation-stderr.log'
    with stdout.open('x',encoding='utf-8') as out,stderr.open('x',encoding='utf-8') as err:
        process=subprocess.run([sys.executable,'-u','-c',
            'import sys; from robot_navigation.partial_phase5e_legacy import run; sys.exit(run(sys.argv[1]))',str(legacy_plan)],
            cwd=ROOT,stdout=out,stderr=err)
    if process.returncode:raise ValueError('Legacy regressions failed; preserve partial receipts')
    counts=test_counts(folder/'legacy-regressions.xml')
    if any(counts[k] for k in ('failures','errors','skipped')):raise ValueError('Legacy regression gate failed')
    checked_attempt(folder);preserve()
    if ledger.totals()!=before:raise ValueError('Legacy continuation changed the Phase 5E mechanical ledger')
    write_json(folder/'validation-complete.json',{'status':'passed','focused':focused,'operations':before,
        'development':False,'legacy':counts,'continuation':'Only unchanged legacy suite resumed after runner interruption',
        'initial_legacy_partial_counts':'not_observed; explicitly retained in runner-interruption.json'})
    write_json(folder/'continuation-verification.json',{'status':'passed','focused_rerun':False,
        'new_Phase5E_operations_before':before,'new_Phase5E_operations_after':ledger.totals(),
        'source_sha256':sha(own),'source_seal_unchanged':True,'legacy_tests':counts})
    print({'status':'validation completed','focused':focused,'legacy':counts,'new_operations':before})


if __name__=='__main__':complete()
