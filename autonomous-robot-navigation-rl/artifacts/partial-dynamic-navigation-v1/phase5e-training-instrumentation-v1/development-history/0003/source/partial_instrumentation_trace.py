"""Versioned numeric trace transport and identity. No learner or environment calls."""
from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

SCHEMA='partial-learning-live-v1'
KINDS={'live_actual','historical_logged','snapshot_reconstructed','synthetic_mechanical','not_observed','not_applicable'}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def checksum(value):return hashlib.sha256(canonical(value)).hexdigest()


def numeric(value):
    if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
    a=np.asarray(value)
    if a.dtype.kind not in 'biuf' or not np.isfinite(a).all():raise ValueError('Nonfinite or nonnumeric live tensor')
    return {'shape':list(a.shape),'dtype':a.dtype.str,'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest(),'data':a.tolist()}


def array(record):
    a=np.asarray(record['data'],dtype=np.dtype(record['dtype'])).reshape(record['shape'])
    if not np.isfinite(a).all() or hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=record['sha256']:
        raise ValueError('Numeric payload checksum/finiteness mismatch')
    return a


def finite_tree(value):
    if isinstance(value,dict):
        if set(('shape','dtype','sha256','data')).issubset(value):array(value)
        else:
            for v in value.values():finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)
    elif isinstance(value,(float,np.floating)) and not np.isfinite(value):raise ValueError('Nonfinite scalar')


class TraceWriter:
    """Exclusive bounded gzip shards; digest each record and retain failed attempts."""
    def __init__(self,folder,metadata,byte_cap=134217728,shard_bytes=262144):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        if metadata.get('source_kind') not in KINDS:raise ValueError('Explicit measurement provenance required')
        if any(k not in metadata for k in ('study_id','run_id','algorithm','arm','seed','source_hash','config_hash','dependency_hash')):
            raise ValueError('Shared trace metadata missing')
        if metadata.get('split') not in ('train','synthetic_mechanical'):raise ValueError('Forbidden final/unseen trace source')
        self.metadata=metadata;self.byte_cap=byte_cap;self.shard_bytes=shard_bytes
        self.sequence=0;self.reserved_bytes=0;self.shard=0;self.shard_size=0;self.stream=None
        self.closed=False
    def emit(self,kind,**fields):
        if self.closed:raise ValueError('Trace is closed')
        if set(fields)&{'schema','sequence','event','source_kind','run','classification'}:raise ValueError('Reserved provenance/schema fields')
        finite_tree(fields)
        body={'schema':SCHEMA,'sequence':self.sequence+1,'event':kind,'source_kind':'live_actual',
              'classification':self.metadata['source_kind'],'run':self.metadata,**fields}
        payload=canonical({'record':body,'sha256':checksum(body)})+b'\n'
        # Reserve uncompressed bytes, conservatively bounding compressed storage too.
        if self.reserved_bytes+len(payload)+4096>self.byte_cap:raise RuntimeError('Trace quota exhausted before record')
        if self.stream is None or self.shard_size+len(payload)>self.shard_bytes:
            if self.stream:self.stream.close()
            self.shard+=1;self.shard_size=0
            self.stream=gzip.open(self.folder/f'{self.shard:05d}.jsonl.gz','xb',compresslevel=3)
        self.stream.write(payload);self.stream.flush()
        self.sequence+=1;self.reserved_bytes+=len(payload);self.shard_size+=len(payload)
    def close(self):
        if self.stream:self.stream.close();self.stream=None
        self.closed=True


def read_trace(folder):
    rows=[]
    for p in sorted(Path(folder).glob('*.jsonl.gz')):
        with gzip.open(p,'rt',encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line);r=item['record']
                if checksum(r)!=item['sha256'] or r['sequence']!=len(rows)+1 or r['schema']!=SCHEMA:
                    raise ValueError('Trace order/checksum/schema mismatch')
                finite_tree(r);rows.append(r)
    if not rows or rows[-1]['event']!='observer_closed':raise ValueError('Interrupted or empty trace; completion not observed')
    return rows


class ReplayIdentity:
    def __init__(self,size,n_envs=1):
        self.size=size;self.n_envs=n_envs;self.slots={};self.generations={};self.last_id=0
    def add(self,slot,env,transition_id,**metadata):
        if not 0<=slot<self.size or not 0<=env<self.n_envs or transition_id<=self.last_id:
            raise ValueError('Invalid slot/environment or duplicated transition identity')
        key=(int(slot),int(env));generation=self.generations.get(key,0)+1
        self.generations[key]=generation;self.last_id=int(transition_id)
        self.slots[key]={'slot':key[0],'env_index':key[1],'generation':generation,'transition_id':int(transition_id),**metadata}
        return dict(self.slots[key])
    def sampled(self,indices,env_indices):
        if len(indices)!=len(env_indices):raise ValueError('Mismatched replay index arrays')
        result=[]
        for s,e in zip(indices,env_indices):
            if not isinstance(s,(int,np.integer)) or not isinstance(e,(int,np.integer)) or (int(s),int(e)) not in self.slots:
                raise ValueError('Malformed or uninitialized replay sample index')
            result.append(dict(self.slots[int(s),int(e)]))
        return result


class MechanicalLedger:
    """Shared across sequential processes and all development/final Phase 5E tests."""
    def __init__(self,path):self.path=Path(path)
    def rows(self):return [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
    def totals(self):
        rows=self.rows()
        return {'transitions':sum(r['kind']=='transition' for r in rows),'optimizer_steps':sum(r['kind']=='optimizer' for r in rows),
                'wall_seconds':sum(r.get('seconds',0) for r in rows if r['kind']=='session_end')}
    def record(self,kind,run='',**fields):
        t=self.totals();rows=self.rows();active=[r for r in rows if r['kind']=='session_start']
        ends=sum(r['kind']=='session_end' for r in rows)
        if active and len(active)>ends and time.time()-active[-1]['time']+t['wall_seconds']>1200:
            raise RuntimeError('Synthetic validation wall ceiling')
        if kind=='transition' and t['transitions']>=1000:raise RuntimeError('Synthetic transition ceiling')
        if kind=='optimizer' and t['optimizer_steps']>=500:raise RuntimeError('Synthetic optimizer ceiling')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'sequence':len(rows)+1,'kind':kind,'run':run,**fields})+'\n')
