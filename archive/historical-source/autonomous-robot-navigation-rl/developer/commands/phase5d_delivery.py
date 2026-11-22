"""Delivery-only CSV boolean correction; never reruns the sealed Phase 5D analysis.

The registered source/snapshot is deliberately unchanged. See the delivery note.
"""
from pathlib import Path
import json
import shutil
from unittest.mock import patch

from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5d import location,checked_plan
from robot_navigation import partial_phase5d_report as renderer

ORIGINAL_DECODE=renderer.decode


def decode(value):
    if value=='True':return True
    if value=='False':return False
    return ORIGINAL_DECODE(value)


def report(root=ROOT):
    folder=location(root);checked_plan(root)
    if read_json(folder/'verification.json')['status']!='passed':raise ValueError('Scientific verification required')
    (folder/'delivery.lock').open('x').close()
    own=Path(__file__).resolve();tests=root/'developer/tests/test_phase5d_delivery.py'
    note=root/'docs/partial_phase5d_delivery_note.md'
    evidence=root/'.runtime/phase5d-delivery-tests.xml'
    from robot_navigation.partial_phase5b import junit
    if junit(evidence)!={'tests':8,'failures':0,'errors':0,'skipped':0}:raise ValueError('Delivery regression gate failed')
    shutil.copy2(evidence,folder/'delivery-regressions.xml')
    shutil.copy2(own,folder/'source/phase5d_delivery.py')
    shutil.copy2(tests,folder/'source/test_phase5d_delivery.py')
    shutil.copy2(note,folder/'source/partial_phase5d_delivery_note.md')
    write_json(folder/'report-failure.json',{
        'stage':'first report rendering, after successful scientific verification',
        'exception':"TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        'cause':'csv.DictWriter writes Python booleans True/False; JSON-only decoder left these as strings',
        'scientific_analysis_retried':False,'new_score_calls':0,'report_written_before_failure':False,
        'frozen_analysis_plan_unchanged':True})
    inputs={'analysis-plan.json':sha(folder/'analysis-plan.json'),'scores.json':sha(folder/'scores.json'),
        'forward-calls.jsonl':sha(folder/'forward-calls.jsonl'),'diagnostic-summary.json':sha(folder/'diagnostic-summary.json')}
    write_json(folder/'delivery-plan.json',{'scope':'Boolean CSV decoding and report delivery only, registered after scoring; not an experimental protocol amendment',
        'scientific_source_changes':0,'scientific_retries':0,'additional_score_calls':0,'inputs':inputs,
        'source':{str(p.relative_to(root).as_posix()):sha(p) for p in (own,tests,note)},'tests':junit(evidence)})
    with patch.object(renderer,'decode',decode):result=renderer.report(root)
    # Archive the first rendered delivery before appending its transparent erratum.
    shutil.copy2(folder/'output-hashes.json',folder/'delivery-original-output-hashes.json')
    shutil.copy2(root/'docs/partial_phase5d_results.md',folder/'delivery-original-results.md')
    suffix=('\n\n## Report delivery correction\n\n'
        'The first report-rendering command failed before writing a report because CSV booleans were decoded as strings. '
        'The scientific analysis and independent verification had already passed and were not rerun. '
        'A separately recorded delivery adapter converts only literal True/False CSV values to booleans. '
        'All frozen scientific source, cohorts, scores, input hashes and the 3,786-call ledger remain unchanged. '
        'Eight additional delivery-only tests pass; the full scientific regression suite remains 424 passing tests. '
        'See [the delivery note](partial_phase5d_delivery_note.md). '
        'The frozen original report CLI retains the parsing defect; use the delivered report or the separately recorded adapter. '
        'The adapter itself is exclusive and refuses another delivery attempt.\n')
    for p in (root/'docs/partial_phase5d_results.md',root/'docs/partial_phase5d_diagnosis.md',folder/'results.md'):
        p.write_text(p.read_text(encoding='utf-8')+suffix,encoding='utf-8',newline='\n')
    for name,h in inputs.items():
        if sha(folder/name)!=h:raise ValueError('Delivery changed scientific evidence')
    checked_plan(root)
    write_json(folder/'delivery-verification.json',{'status':'passed','additional_score_calls':0,
        'scientific_inputs_unchanged':inputs,'delivery_tests':junit(evidence),
        'source_registration':'Separate delivery manifest; original scientific seal unchanged'})
    # Final delivery seal includes the preserved failed attempt and initial-render hashes.
    pins={'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='output-hashes.json'},
        'reports':{p.relative_to(root).as_posix():sha(p) for p in (root/'docs/partial_phase5d_results.md',root/'docs/partial_phase5d_diagnosis.md',note)},
        'delivery_source':{p.relative_to(root).as_posix():sha(p) for p in (own,tests)},
        'hash_manifest_self_excluded':True}
    write_json(folder/'output-hashes.json',pins)
    size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if size>134217728:raise ValueError('Artifact cap exceeded')
    return dict(result,artifact_bytes=size,delivery_tests=8,additional_score_calls=0)


if __name__=='__main__':print(json.dumps(report(),indent=2))
