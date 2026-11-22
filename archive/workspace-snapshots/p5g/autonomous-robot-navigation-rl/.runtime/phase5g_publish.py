"""Delivery-only staging and immutable-byte validation, no scientific execution."""
import hashlib,json,subprocess
from pathlib import Path
from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5g import checked
from robot_navigation.partial_phase5g_io import BASE,BRANCH
from robot_navigation.partial_instrumentation_trace import checksum,quota_bytes

repo=ROOT.parent
def git(*args):return subprocess.check_output(['git','-c','core.longpaths=true','-c','safe.directory='+str(repo),*args],cwd=repo)
def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def staged():
    changes=git('diff','--cached','--name-status','-z').decode().split('\0')[:-1]
    assert changes and all(changes[i]=='A' for i in range(0,len(changes),2))
    names=changes[1::2]
    entries={line.split('\t',1)[1]:line.split()[1] for line in git('ls-files','--stage').decode().splitlines()}
    for name in names:assert entries[name]==blob((repo/name).read_bytes()),name
    return {n:entries[n] for n in names}

assert git('rev-parse','HEAD').decode().strip()==BASE and git('branch','--show-current').decode().strip()==BRANCH
folder,manifest,_=checked();v=read_json(folder/'verification.json')
assert v['verdict']=='PHASE5G_MEASUREMENT_VALIDATED' and all(x=='passed' for x in v['gates'].values())
inventory=read_json(folder/'output-hashes.json')
for name,h in inventory['files'].items():assert sha(folder/name)==h,name
for name,h in inventory['reports'].items():assert sha(ROOT/name)==h,name
before=staged()
write_json(folder/'publication-validation.json',{'classification':'delivery_only_byte_validation','script_sha256':sha(Path(__file__)),
    'parent_commit':BASE,'branch':BRANCH,'frozen_source_digest':manifest['source_digest'],
    'verification_receipt_sha256':sha(folder/'verification.json'),'staged_files_before_this_receipt':len(before),
    'staged_name_blob_manifest_sha256':checksum(before),'all_changes_are_new_files':True,'staged_blob_bytes_equal_registered_worktree_bytes':True,
    'legacy_tests':v['legacy_tests'],'post_tests':v['post_tests'],'preservation':v['preservation'],
    'new_training_actions':0,'new_assessment_actions':0,'new_optimizer_steps':0})
inventory['files']['publication-validation.json']=sha(folder/'publication-validation.json');write_json(folder/'output-hashes.json',inventory)
git('add','--',str((folder/'publication-validation.json').relative_to(repo)),str((folder/'output-hashes.json').relative_to(repo)))
after=staged()
assert quota_bytes([folder])<536870912
print(json.dumps({'staged_files':len(after),'all_new':True,'raw_byte_stage_matches':True,'verdict':v['verdict'],
    'training_actions':v['scientific_training_actions'],'optimizer_steps':v['optimizer_steps'],'assessment_actions':v['assessment_actions'],
    'artifact_bytes':quota_bytes([folder]),'learning_signals':{a:{k:r['signals'][k] for k in ('recorded_goal_experiences','goal_sample_draws','distinct_sampled_goals')} for a,r in v['training'].items()}},indent=2))
