"""Delivery-only byte-preserving staging; no research execution."""
import hashlib,json,subprocess
from pathlib import Path
from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5f_io import location,preserve

repo=ROOT.parent;folder=location()
def git(*args,**kwargs):return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=repo,**kwargs)
def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

changes=git('diff','--cached','--name-status','-z').decode().split('\0')[:-1]
assert len(changes)%2==0 and all(changes[i]=='A' for i in range(0,len(changes),2))
names=changes[1::2];repairs=[]
entries={x.split('\t',1)[1]:x.split()[1] for x in git('ls-files','--stage').decode().splitlines()}
for n in names:
    data=(repo/n).read_bytes()
    if entries[n]!=blob(data):
        assert n in ('autonomous-robot-navigation-rl/docs/partial_phase5f_results.md','autonomous-robot-navigation-rl/docs/partial_phase5f_diagnosis.md'),n
        h=git('hash-object','-w','--no-filters','--',n).decode().strip()
        git('update-index','--cacheinfo','100644,'+h+','+n)
        repairs.append({'path':n,'worktree_sha256':sha(repo/n),'original_filtered_blob':entries[n],'raw_preserved_blob':h})
write_json(folder/'delivery-appendix.json',{'scope':'Publication transport only; no scientific source, records, tests or model changed',
    'delivery_script_sha256':sha(Path(__file__)),'raw_byte_staging':repairs,
    'checkout_note':'For portable byte-verification of root instruction briefs use git -c core.autocrlf=false clone; inherited parent attributes/settings remain unchanged',
    'new_research_actions':0,'new_optimizer_steps':0})
inventory=read_json(folder/'output-hashes.json');inventory['files']['delivery-appendix.json']=sha(folder/'delivery-appendix.json')
write_json(folder/'output-hashes.json',inventory)
git('add','--',str((folder/'delivery-appendix.json').relative_to(repo)),str((folder/'output-hashes.json').relative_to(repo)))
entries={x.split('\t',1)[1]:x.split()[1] for x in git('ls-files','--stage').decode().splitlines()}
changes=git('diff','--cached','--name-status','-z').decode().split('\0')[:-1]
assert all(changes[i]=='A' for i in range(0,len(changes),2))
for n in changes[1::2]:assert entries[n]==blob((repo/n).read_bytes()),n
for n,h in inventory['files'].items():assert sha(folder/n)==h,n
for n,h in inventory['reports'].items():assert sha(ROOT/n)==h,n
print(json.dumps({'staged_files':len(changes)//2,'raw_byte_stage_matches':True,'delivery_adjustments':repairs,'preservation':preserve()},indent=2))
