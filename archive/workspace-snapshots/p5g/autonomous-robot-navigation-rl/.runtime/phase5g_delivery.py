"""Report-only publication appendix. Never executes a model or environment."""
import json
from pathlib import Path
from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5g import checked
from robot_navigation.partial_phase5g_io import inputs
from robot_navigation.partial_instrumentation_trace import checksum,quota_bytes

folder,manifest,_=checked()
result=read_json(folder/'verification.json')
assert result['verdict']=='PHASE5G_MEASUREMENT_VALIDATED'
inventory=read_json(folder/'output-hashes.json')
for n,h in inventory['files'].items():assert sha(folder/n)==h,n
for n,h in inventory['reports'].items():assert sha(ROOT/n)==h,n
history=json.loads((ROOT/'.runtime/phase5g-delivery-history.json').read_text(encoding='utf-8-sig'))
write_json(folder/'setup-history.json',history)
report=ROOT/'docs/partial_phase5g_results.md';before=sha(report)
with report.open('a',encoding='utf-8',newline='\n') as f:
    f.write('\n## Delivery-only appendix\n\nThe [complete preflight correction history](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/setup-history.json) records the Git ownership checks and the CRLF/LF inventory assertion as well as the scoped historical-status exception. These occurred before registration and research. The [hashed delivery appendix](../artifacts/partial-dynamic-navigation-v1/phase5g-longer-horizon-v1/delivery-appendix.json) records this report-only addition; no frozen scientific code, tests, case data, checkpoints, traces or assessment records changed.\n\nVerification includes the preserved original dirty-workspace inventory and pinned runtime paths, so run it from this preserved worktree. Root instruction-brief hashes are byte-sensitive; automatic line-ending conversion fails closed. In PowerShell from the application folder with the pinned Python environment active: `$env:PYTHONPATH="src"; python -m robot_navigation.partial_phase5g verify`. Do not invoke the one-shot study again.\n')
assert checksum(inputs())==manifest['source_digest']
write_json(folder/'delivery-appendix.json',{'classification':'report_only_delivery','delivery_script_sha256':sha(Path(__file__)),
    'frozen_source_digest':manifest['source_digest'],'generated_report_sha256_before_appendix':before,
    'published_report_sha256':sha(report),'setup_history_sha256':sha(folder/'setup-history.json'),
    'scientific_inputs_changed':False,'scientific_records_changed':False,'training_or_assessment_retries':0,
    'new_training_actions':0,'new_assessment_actions':0,'new_synthetic_transitions':0,'new_optimizer_steps':0})
for name in ('setup-history.json','delivery-appendix.json'):inventory['files'][name]=sha(folder/name)
inventory['reports']['docs/partial_phase5g_results.md']=sha(report)
write_json(folder/'output-hashes.json',inventory)
assert quota_bytes([folder])<536870912
print(json.dumps({'verdict':result['verdict'],'artifact_bytes':quota_bytes([folder]),'frozen_source_unchanged':True,
    'screens':{a:r['screen'] for a,r in result['assessment']['arms'].items()},
    'training_goals':{a:r['signals']['recorded_goal_experiences'] for a,r in result['training'].items()}},indent=2))
