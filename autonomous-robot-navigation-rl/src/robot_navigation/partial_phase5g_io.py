"""Frozen G inputs and budget controls. No physical operations in this module."""
import json,subprocess
from pathlib import Path
from time import perf_counter
from .core import ROOT,read_json
from .release import sha
from .partial_phase5f_io import Journal,read_journal,observed_hash
from .partial_instrumentation_trace import checksum

BASE='6fb6aac124067b00016f5514ebbc66ef777dea33'
BRANCH='feature/phase5g-longer-horizon-diagnosis-v1'
STUDY='phase5g-longer-horizon-v1'
BRIEF='CODEX_PHASE5G_LONGER_HORIZON_LEARNING_DIAGNOSIS.md'
ALGORITHMS=('dqn','ppo');CHECKPOINTS=(0,500,1000,2000)
UPDATES={'dqn':475,'ppo':200}

def git(root,*args):
    owner=Path(root).resolve()
    if not (owner/'.git').exists():owner=owner.parent
    return subprocess.check_output(['git','-c','core.longpaths=true','-c','safe.directory='+str(owner),*args],cwd=root,text=True)

def location(root=ROOT):return root/'artifacts/partial-dynamic-navigation-v1'/STUDY

def inputs(root=ROOT):
    paths=list((root/'src/robot_navigation').glob('partial_phase5g*.py'))
    paths += [root/'developer/tests/test_partial_phase5g.py',root/'developer/commands/phase5g_parent_regressions.py',
        root/'configs/partial_phase5g_v1.yaml',root/'docs/partial_phase5g_protocol.md',root.parent/BRIEF]
    return {p.relative_to(root.parent).as_posix():sha(p) for p in sorted(paths)}

def cases(root=ROOT):
    text=(root/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/manifest.json').read_text(encoding='utf-8')
    result={}
    for key,count in [('training_cases',8),('original_cases',4),('expanded_cases',16)]:
        marker='"'+key+'":';start=text.index(marker)+len(marker)
        while text[start].isspace():start+=1
        result[key],_=json.JSONDecoder().raw_decode(text,start)
        if len(result[key])!=count:raise ValueError('Frozen panel count')
        for c in result[key]:
            if c['fixture_category']!='A' or c['events'] or c['rows']!=['.'*10]*10:raise ValueError('Not static open Fixture A')
            wanted={'training_cases':'train','original_cases':'evaluation','expanded_cases':'expanded-development'}[key]
            if c['fixture_split']!=wanted:raise ValueError('Panel/split mismatch')
    flat=sum(result.values(),[])
    if len({c['id'] for c in flat})!=28 or len({(tuple(c['start']),tuple(c['goal'])) for c in flat})!=28:raise ValueError('Panels overlap')
    return result

def direction(case):
    dr=case['goal'][0]-case['start'][0];dc=case['goal'][1]-case['start'][1]
    if (dr==0)==(dc==0):raise ValueError('Expected axial historical goals')
    return 'Up' if dr<0 else 'Down' if dr>0 else 'Left' if dc<0 else 'Right'

def preserve(root=ROOT):
    folder=location(root)
    def inv(name):
        p=folder/(name+'-inventory.json')
        return read_json(p if p.exists() else root/'.runtime'/('phase5g-'+name+'-inventory.json'))
    base=inv('base');original=inv('original')
    for n,h in base['files'].items():
        if sha(root.parent/n)!=h:raise ValueError('Historical parent file changed: '+n)
    common=Path(git(root,'rev-parse','--git-common-dir').strip())
    main=common.parent if common.is_absolute() else (root/common).resolve().parent
    for n,h in original['files'].items():
        if sha(main/n)!=h:raise ValueError('Original dirty file changed: '+n)
    for args,expected in [(('rev-parse','HEAD'),original['head']+'\n'),(('status','--porcelain','--untracked-files=all'),original['status']),
        (('diff','--cached','--binary'),original['index'])]:
        if git(main,*args)!=expected:raise ValueError('Original main HEAD/status/index changed')
    for n,h in original['refs'].items():
        if git(main,'rev-parse',n).strip()!=h:raise ValueError('Historical ref changed: '+n)
    return {'status':'passed','parent':base['head'],'parent_tree':base['tree'],'parent_files':len(base['files']),
        'original_files':len(original['files']),'main_head':original['head'],'historical_refs':original['refs'],
        'base_inventory_sha256':checksum(base),'original_inventory_sha256':checksum(original)}

class Budget:
    def __init__(self,journal,clock=perf_counter):
        self.journal=journal;self.clock=clock;self.started=clock();self.arm_started=None;self.arm=None
        self.counts=dict.fromkeys(ALGORITHMS,0);self.updates=dict.fromkeys(ALGORITHMS,0);self.entered=[]
    def check(self):
        if self.clock()-self.started>=3600:raise TimeoutError('Combined training 3600-second ceiling')
        if self.arm_started is not None and self.clock()-self.arm_started>=1800:raise TimeoutError('Arm training 1800-second ceiling')
    def start(self,algorithm):
        if algorithm not in ALGORITHMS or algorithm in self.entered or algorithm!=ALGORITHMS[len(self.entered)]:raise RuntimeError('One DQN then PPO, no retry')
        if self.arm is not None and self.counts[self.arm]!=2000:raise RuntimeError('Previous arm incomplete')
        self.check();self.arm=algorithm;self.arm_started=self.clock();self.entered.append(algorithm)
        self.journal.append('arm_started',algorithm=algorithm,seed=23121)
    def action(self,algorithm):
        self.check()
        if algorithm!=self.arm or self.counts[algorithm]>=2000 or sum(self.counts.values())>=4000:raise RuntimeError('Training cap before step')
        n=self.counts[algorithm]+1
        self.journal.append('action_reserved',algorithm=algorithm,transition_id=n,total=sum(self.counts.values())+1)
        self.counts[algorithm]=n;return n
    def optimizer(self,algorithm):
        self.check();n=self.updates[algorithm]+1
        if algorithm!=self.arm or n>UPDATES[algorithm]:raise RuntimeError('Optimizer ceiling before operation')
        self.journal.append('optimizer_reserved',algorithm=algorithm,update_id=n,training_actions=self.counts[algorithm])
        self.updates[algorithm]=n;return n

class AssessmentBudget:
    def __init__(self,journal):self.journal=journal;self.count=0;self.case_steps=0
    def start_case(self):self.case_steps=0
    def reserve(self,**fields):
        if self.count>=32000 or self.case_steps>=200:raise RuntimeError('Assessment cap before env.step')
        n=self.count+1;self.journal.append('assessment_reserved',assessment_action_id=n,case_step=self.case_steps+1,**fields)
        self.count=n;self.case_steps+=1;return n
