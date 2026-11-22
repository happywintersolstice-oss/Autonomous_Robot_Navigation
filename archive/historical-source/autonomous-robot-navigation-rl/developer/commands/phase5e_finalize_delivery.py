"""Audit and seal completed Phase 5E data; zero new learning or environment calls.

Separately registered delivery code records the runner interruption honestly.
The original scientific source/attempt seal and focused test results are intact.
"""
from pathlib import Path
import json

from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5e import location,checked_attempt,audit,verify
from robot_navigation.partial_instrumentation_trace import MechanicalLedger,quota_bytes


def finalize():
    folder=location();plan=checked_attempt(folder)
    if read_json(folder/'continuation-verification.json')['status']!='passed':raise ValueError('Completed legacy continuation required')
    (folder/'publication-finalize.lock').open('x').close()
    own=Path(__file__).resolve();continuation=ROOT/'developer/commands/phase5e_complete_validation.py'
    expected=read_json(folder/'continuation-plan.json')['source_sha256']
    if sha(continuation)!=expected:raise ValueError('Continuation source changed')
    (folder/'source/phase5e_finalize_delivery.py').write_bytes(own.read_bytes())
    write_json(folder/'publication-delivery-plan.json',{
        'scope':'Audit existing captured tensors and append runner-continuation disclosure; no new tests, model loads or learning',
        'sources':{p.relative_to(ROOT).as_posix():sha(p) for p in (own,continuation)},
        'original_source_digest':plan['source_digest'],'original_attempt_seal':sha(folder/'attempt-seal.json')})
    before=MechanicalLedger(ROOT/'.runtime/phase5e-operations.jsonl').totals()
    summary=audit()
    footer=('\n\n## Validation runner continuation\n\n'
        'The runner connection closed after the final 32-test Phase 5E suite passed, while the separate legacy regression process was active. '
        'No legacy JUnit or operation receipt survived that interrupted process, so its partial counts/time are explicitly not_observed; '
        'they are not included in the exact completed legacy-run counts above. The complete unchanged parent suite was subsequently run to completion '
        'using the separately registered continuation helper. The passed Phase 5E fixtures were not rerun, and the sealed source, captured tensors '
        'and cumulative new-mechanics ledger stayed unchanged. The exact total legacy activity including the interrupted process is unavailable. '
        'See runner-interruption.json, continuation-plan.json, continuation-verification.json and publication-delivery-plan.json in the evidence namespace. '
        'Both delivery helpers are separately source-hashed and make no scientific training, final evaluation or new Phase 5E mechanical calls.\n')
    for p in (ROOT/'docs/partial_phase5e_results.md',ROOT/'docs/partial_phase5e_diagnosis.md'):
        p.write_text(p.read_text(encoding='utf-8')+footer,encoding='utf-8',newline='\n')
    (folder/'results.md').write_bytes((ROOT/'docs/partial_phase5e_results.md').read_bytes())
    after=MechanicalLedger(ROOT/'.runtime/phase5e-operations.jsonl').totals()
    if before!=after:raise ValueError('Delivery changed new mechanical operation ledger')
    checked_attempt(folder)
    write_json(folder/'publication-delivery-verification.json',{'status':'passed','verdict':summary['verdict'],
        'scientific_training_actions':0,'research_assessment_actions':0,'sealed_final_evaluations':0,
        'additional_Phase5E_mechanical_operations':0,'new_synthetic_cumulative':after,
        'original_source_digest':plan['source_digest'],'completed_legacy_tests':summary['legacy_tests'],
        'initial_interrupted_legacy_activity':'not_observed; not fabricated or counted as a completed test run'})
    pins={'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='output-hashes.json'},
        'reports':{p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'docs/partial_phase5e_results.md',ROOT/'docs/partial_phase5e_diagnosis.md')},
        'delivery_sources':{p.relative_to(ROOT).as_posix():sha(p) for p in (own,continuation)}}
    write_json(folder/'output-hashes.json',pins)
    verified=verify()
    for name,h in pins['delivery_sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Delivery source hash changed')
    size=quota_bytes([folder,ROOT/'.runtime/phase5e-development'])
    if size>134217728:raise ValueError('Cumulative artifact quota exceeded')
    return {'status':verified['status'],'verdict':summary['verdict'],'focused':summary['final_focused_tests'],
        'completed_legacy_tests':summary['legacy_tests'],'new_operations':after,'cumulative_artifact_bytes':size}


if __name__=='__main__':print(json.dumps(finalize(),indent=2))
