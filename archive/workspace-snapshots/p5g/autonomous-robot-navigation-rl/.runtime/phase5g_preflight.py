import difflib,hashlib,json,os,subprocess
from pathlib import Path
from robot_navigation.core import ROOT,read_json,write_json
from robot_navigation.release import sha
from robot_navigation.partial_phase5f import verify
from robot_navigation.partial_training_instrumentation import check_compatibility,library_contract
os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0='*')

MAIN=Path('C:/Users/User/Desktop/Autonomous_Robot_Navigation')
F=MAIN/'autonomous-robot-navigation-rl/.runtime/p5f'
def git(root,*args):return subprocess.check_output(['git','-c','core.longpaths=true','-c','safe.directory='+str(root.resolve()),*args],cwd=root,text=True)
def inventory(root):
    names=git(root,'ls-files','--cached','--others','--exclude-standard').splitlines()
    return {n:sha(root/n) for n in sorted(set(names)) if (root/n).is_file()}
assert git(F,'rev-parse','HEAD').strip()=='6fb6aac124067b00016f5514ebbc66ef777dea33'
assert git(F,'status','--porcelain')==''
parent_files={};repairs=[]
for line in git(F,'ls-files','--stage').splitlines():
    fields,n=line.split('\t',1);expected=fields.split()[1];data=(F/n).read_bytes()
    normalized=data.replace(b'\r\n',b'\n')
    assert any(hashlib.sha1(b'blob '+str(len(v)).encode()+b'\0'+v).hexdigest()==expected for v in (data,normalized)),n
    parent_files[n]=hashlib.sha256(data).hexdigest()
    dest=ROOT.parent/n
    if dest.read_bytes()!=data:
        assert dest.read_bytes().replace(b'\r\n',b'\n')==data.replace(b'\r\n',b'\n'),n
        dest.write_bytes(data);repairs.append(n)
base={'head':git(F,'rev-parse','HEAD').strip(),'tree':git(F,'rev-parse','HEAD^{tree}').strip(),'files':parent_files}
write_json(ROOT/'.runtime/phase5g-base-inventory.json',base)
refs=['main','feature/phase5b-basic-navigation-diagnosis-v1','feature/phase5c-directional-coverage-v1',
    'feature/phase5d-goal-credit-diagnosis-v1','feature/phase5e-training-instrumentation-v1','feature/phase5f-instrumented-learning-pilot-v1']
original={'head':git(MAIN,'rev-parse','HEAD').strip(),'status':git(MAIN,'status','--porcelain','--untracked-files=all'),
    'index':git(MAIN,'diff','--cached','--binary'),'refs':{n:git(MAIN,'rev-parse',n).strip() for n in refs},'files':inventory(MAIN)}
write_json(ROOT/'.runtime/phase5g-original-inventory.json',original)
pre={'exact_parent':base['head'],'parent_tree':base['tree'],'restored_checkout_bytes_only':repairs,'library':library_contract()}
check_compatibility()
try:pre['plain_parent_verification']=verify()
except Exception as e:pre['plain_parent_verification']={'status':'failed','error':repr(e)}
old=read_json(ROOT/'artifacts/partial-dynamic-navigation-v1/phase5f-instrumented-learning-pilot-v1/original-inventory.json')
pre['original_status_difference']=list(difflib.unified_diff(old['status'].splitlines(),original['status'].splitlines(),fromfile='F registered',tofile='current main'))
assert original['status'].replace('?? CODEX_PHASE5G_LONGER_HORIZON_LEARNING_DIAGNOSIS.md\n','')==old['status']
exclude=ROOT/'.runtime/phase5g-parent-scope.exclude'
prior=git(MAIN,'config','--get','core.excludesFile').strip() if subprocess.run(['git','config','--get','core.excludesFile'],cwd=MAIN,capture_output=True).returncode==0 else ''
text=Path(os.path.expanduser(prior)).read_text() if prior and Path(os.path.expanduser(prior)).is_file() else ''
exclude.write_text(text+'\n/CODEX_PHASE5G_LONGER_HORIZON_LEARNING_DIAGNOSIS.md\n',encoding='utf-8')
os.environ.update(GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='core.excludesFile',GIT_CONFIG_VALUE_0=str(exclude),GIT_CONFIG_KEY_1='safe.directory',GIT_CONFIG_VALUE_1='*')
pre['scoped_parent_verification']=verify()
pre['scope']='Only newly supplied G brief is excluded from old F original-status comparison in this process; no old source/evidence/Git config modified. G inventory includes full current main.'
write_json(ROOT/'.runtime/phase5g-parent-preflight.json',pre)
print(json.dumps({'parent_files':len(parent_files),'original_files':len(original['files']),'repairs':repairs,'parent_verdict':pre['scoped_parent_verification']['verdict']},indent=2))
