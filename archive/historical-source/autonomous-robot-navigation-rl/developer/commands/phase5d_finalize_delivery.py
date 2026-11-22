"""Seal already-rendered Phase 5D reports after an external preservation receipt.

No scientific analysis or report renderer is invoked. No model is loaded.
"""
from pathlib import Path
import json
import shutil

from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5b import junit
from robot_navigation.partial_phase5d import location,checked_plan


def finalize(root=ROOT):
    folder=location(root);checked_plan(root)
    if read_json(folder/'original-workspace-preservation.json')['status']!='passed':raise ValueError('Original preservation receipt required')
    tests=root/'.runtime/phase5d-delivery-full-tests.xml'
    if junit(tests)!={'tests':432,'failures':0,'errors':0,'skipped':0}:raise ValueError('Final full regressions required')
    (folder/'delivery-finalize.lock').open('x').close()
    # The renderer wrote all three reports before the Git ownership check failed.
    if (folder/'output-hashes.json').exists():raise ValueError('Unexpected prior completed delivery')
    previous=read_json(folder/'delivery-plan.json')
    for name,h in previous['inputs'].items():
        if sha(folder/name)!=h:raise ValueError('Scientific evidence changed during delivery')
    for name,h in previous['source'].items():
        if sha(root/name)!=h:raise ValueError('Delivery adapter source changed')
    shutil.copy2(tests,folder/'delivery-full-regressions.xml')
    for p in (root/'docs/partial_phase5d_results.md',root/'docs/partial_phase5d_diagnosis.md'):
        if not p.is_file():raise ValueError('Expected already-rendered report missing')
        shutil.copy2(p,folder/('delivery-original-'+p.name))
    own=Path(__file__).resolve();note=root/'docs/partial_phase5d_delivery_completion.md'
    shutil.copy2(own,folder/'source/phase5d_finalize_delivery.py')
    shutil.copy2(note,folder/'source/partial_phase5d_delivery_completion.md')
    write_json(folder/'delivery-completion-plan.json',{
        'scope':'Finalize reports already rendered; no renderer retry, analysis retry or score call',
        'source':{p.relative_to(root).as_posix():sha(p) for p in (own,note)},
        'prior_delivery_plan_sha256':sha(folder/'delivery-plan.json'),
        'scientific_inputs_unchanged':previous['inputs'],'final_full_tests':junit(tests)})
    write_json(folder/'delivery-postcondition-failure.json',{
        'stage':'after all reports rendered, before original-repository Git postcondition',
        'exception':'Git exit 128: dubious ownership under CodexSandboxOffline',
        'followup_status_difference':'Default status collapsed untracked directories; initial inventory used --untracked-files=all',
        'resolution':'Separate read-only preservation check in appropriate context, using original status options; all 9646 files, main HEAD, index and status match',
        'renderer_retried':False,'scientific_analysis_retried':False,'additional_score_calls':0})
    footer=('\n\n## Delivery validation and known renderer limitation\n\n'
        'The complete report above was rendered from unchanged scientific records. The first renderer attempt failed on CSV boolean parsing; '
        'the separately recorded adapter corrected literal True/False parsing and rendered these reports. Its final Git postcondition then encountered '
        'sandbox ownership restrictions and a mismatch between collapsed and expanded untracked-file status formats. '
        'A separate read-only check using the original --untracked-files=all options confirmed all 9,646 original files, main HEAD, index and working-tree status unchanged. '
        'The renderer and scientific analysis were not retried. All 432 final full regressions pass, including eight delivery-only checks. '
        'All original scientific source, protocol/seal, cohorts, scores and the 3,786-call journal remain byte-identical. '
        'The frozen original report CLI retains its parsing/status-check limitations; this completed report is the delivery. '
        'See [delivery details](partial_phase5d_delivery_completion.md), [parser correction](partial_phase5d_delivery_note.md) and the separately hashed delivery manifests. '
        'No new training, navigation actions or model scoring occurred during delivery.\n')
    for p in (root/'docs/partial_phase5d_results.md',root/'docs/partial_phase5d_diagnosis.md',folder/'results.md'):
        p.write_text(p.read_text(encoding='utf-8')+footer,encoding='utf-8',newline='\n')
    checked_plan(root)
    write_json(folder/'delivery-verification.json',{'status':'passed','final_full_tests':junit(tests),
        'new_score_calls':0,'new_training_actions':0,'new_policy_environment_actions':0,
        'scientific_inputs_unchanged':previous['inputs'],'original_preservation':'passed; 9646 files and Git state',
        'report_generation':'completed before Git postcondition failure; finalizer did not rerender'})
    docs=[root/'docs'/name for name in ('partial_phase5d_results.md','partial_phase5d_diagnosis.md','partial_phase5d_delivery_note.md','partial_phase5d_delivery_completion.md')]
    sources=[root/name for name in previous['source']]+[own,note]
    write_json(folder/'output-hashes.json',{'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='output-hashes.json'},
        'reports':{p.relative_to(root).as_posix():sha(p) for p in docs},
        'delivery_source':{p.relative_to(root).as_posix():sha(p) for p in sources},'hash_manifest_self_excluded':True})
    size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if size>134217728:raise ValueError('Artifact byte cap exceeded')
    return {'status':'completed','verdict':read_json(folder/'diagnostic-summary.json')['verdict'],
        'artifact_bytes':size,'final_full_tests':432,'new_forward_calls':3786,'new_training_actions':0,'new_policy_environment_actions':0}


if __name__=='__main__':print(json.dumps(finalize(),indent=2))
