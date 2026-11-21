"""Phase 5F immutable inputs, chained journals and hard budgets; no training."""
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from .core import ROOT,read_json
from .release import sha
from .partial_instrumentation_trace import checksum,canonical,finite_tree,quota_bytes

STUDY='phase5f-instrumented-learning-pilot-v1'
BASE='192ba8dca9957ac6d6b38965956b06f05b694548'
BRIEF='CODEX_PHASE5F_INSTRUMENTED_LEARNING_SIGNAL_PILOT.md'
ALGORITHMS=('dqn','ppo')


def location(root=ROOT):return root/'artifacts/partial-dynamic-navigation-v1'/STUDY


def git(root,*args):
    return subprocess.check_output(['git','-c','core.longpaths=true','-c','safe.directory='+str(root.resolve()),*args],cwd=root,text=True)


def inputs(root=ROOT):
    paths=list((root/'src/robot_navigation').glob('partial_phase5f*.py'))
    paths+=[root/'developer/tests/test_partial_phase5f.py',root/'developer/commands/phase5f_parent_regressions.py',
        root/'configs/partial_phase5f_v1.yaml',root/'docs/partial_phase5f_protocol.md',root.parent/BRIEF]
    return {p.relative_to(root.parent).as_posix():sha(p) for p in sorted(paths)}


def training_cases(root=ROOT):
    """Parse only the registered training_cases array, never other case panels."""
    path=root/'artifacts/partial-dynamic-navigation-v1/phase5b-a-only-v1/manifest.json'
    text=path.read_text(encoding='utf-8');marker='"training_cases":';start=text.index(marker)+len(marker)
    while text[start].isspace():start+=1
    cases,end=json.JSONDecoder().raw_decode(text,start)
    if len(cases)!=8 or len({c['id'] for c in cases})!=8:raise ValueError('Wrong training-case count')
    for c in cases:
        if (c['fixture_category']!='A' or c['fixture_split']!='train' or c['events'] or c['rows']!=['.'*10]*10
            or '-A-train-' not in c['id']):raise ValueError('Not original open-grid Fixture A training')
    return cases


def preserve(root=ROOT):
    folder=location(root)
    def inventory(name):
        path=folder/(name+'-inventory.json')
        return read_json(path if path.exists() else root/'.runtime'/('phase5f-'+name+'-inventory.json'))
    base=inventory('base');original=inventory('original')
    for n,h in base['files'].items():
        if sha(root.parent/n)!=h:raise ValueError('Parent source/evidence changed: '+n)
    common=Path(git(root,'rev-parse','--git-common-dir').strip())
    original_root=common.parent if common.is_absolute() else (root/common).resolve().parent
    for n,h in original['files'].items():
        if sha(original_root/n)!=h:raise ValueError('Original workspace changed: '+n)
    if git(original_root,'rev-parse','HEAD').strip()!=original['head']:raise ValueError('Main HEAD changed')
    if git(original_root,'status','--porcelain','--untracked-files=all')!=original['status']:raise ValueError('Original status changed')
    if git(original_root,'diff','--cached','--binary')!=original['index']:raise ValueError('Original index changed')
    for n,h in original['refs'].items():
        if git(original_root,'rev-parse',n).strip()!=h:raise ValueError('Historical branch changed: '+n)
    return {'status':'passed','parent':base['head'],'tree':base['tree'],'parent_files':len(base['files']),
        'original_files':len(original['files']),'main_head':original['head'],
        'base_inventory_sha256':checksum(base),'original_inventory_sha256':checksum(original),'historical_refs':original['refs']}


class Journal:
    """Exclusive append-only, flushed before operations; hash chain detects edits."""
    def __init__(self,path,activity='research_training',byte_cap=536870912,quota_root=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('x',encoding='utf-8',buffering=1)
        self.activity=activity;self.sequence=0;self.previous='0'*64
        self.byte_cap=byte_cap;self.quota_root=Path(quota_root) if quota_root else self.path.parent
    def append(self,event,**fields):
        finite_tree(fields)
        row={'schema':'phase5f-journal-v1','sequence':self.sequence+1,'previous_sha256':self.previous,
            'source_kind':'live_actual','activity':self.activity,'event':event,**fields}
        digest=checksum(row);payload=canonical({'record':row,'sha256':digest})+b'\n'
        if quota_bytes([self.quota_root])+len(payload)+65536>self.byte_cap:raise RuntimeError('Artifact quota before journal append')
        self.file.write(payload.decode());self.file.flush();self.sequence+=1;self.previous=digest
        return row
    def close(self):self.file.close()


def read_journal(path):
    result=[];previous='0'*64
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        item=json.loads(line);r=item['record']
        if (r['sequence']!=len(result)+1 or r['previous_sha256']!=previous or checksum(r)!=item['sha256']):
            raise ValueError('Journal corruption/order/chain failure')
        finite_tree(r);previous=item['sha256'];result.append(r)
    return result


class Budget:
    def __init__(self,journal,clock=perf_counter):
        self.journal=journal;self.clock=clock;self.started=clock();self.arm_started=None
        self.arm=None;self.counts={'dqn':0,'ppo':0};self.updates={'dqn':0,'ppo':0};self.entered=[]
    def start(self,algorithm):
        if algorithm not in ALGORITHMS or algorithm in self.entered or algorithm!=ALGORITHMS[len(self.entered)]:
            raise RuntimeError('Exactly one DQN then one PPO; no retries/extra arms')
        self.check();self.arm=algorithm;self.arm_started=self.clock();self.entered.append(algorithm)
        self.journal.append('arm_started',algorithm=algorithm,seed=23121)
    def check(self):
        if self.clock()-self.started>=1800:raise TimeoutError('Combined pilot wall ceiling')
        if self.arm_started is not None and self.clock()-self.arm_started>=900:raise TimeoutError('Per-arm wall ceiling')
    def action(self,algorithm):
        self.check()
        if algorithm!=self.arm or self.counts[algorithm]>=500 or sum(self.counts.values())>=1000:
            raise RuntimeError('Scientific action ceiling before env.step')
        n=self.counts[algorithm]+1
        self.journal.append('action_reserved',algorithm=algorithm,transition_id=n,total=sum(self.counts.values())+1)
        self.counts[algorithm]=n;return n
    def optimizer(self,algorithm):
        self.check();n=self.updates[algorithm]+1
        if algorithm!=self.arm or n>{'dqn':100,'ppo':50}[algorithm]:raise RuntimeError('Unregistered optimizer operation')
        self.journal.append('optimizer_reserved',algorithm=algorithm,update_id=n,training_actions=self.counts[algorithm])
        self.updates[algorithm]=n;return n


def observed_hash(observation):
    import numpy as np
    from .partial_observation import validate
    value=validate(np.asarray(observation,dtype=np.float32))
    return hashlib.sha256(value.tobytes()).hexdigest()
