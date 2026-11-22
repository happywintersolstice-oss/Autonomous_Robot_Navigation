"""Independent product-only structural generators. Freeze before model work.
Training/development readers never open the held-out file. Geometry is used here
only for reachability, structural audits and evaluator strata, not policy input.
"""
from pathlib import Path
from collections import Counter,deque
import numpy as np
from .core import ROOT,digest,read_json,write_json,GridMap
from .partial_dynamic_env import make_scenario
from .release import sha
VERSION='product-phase3-holdout-v1'
TRAIN_FAMILIES=('comb_yard','radial_pods','parallel_rails','three_room_line','tree_carving','four_court_star')
DEV_FAMILIES=('triangular_yards','lollipop_court','pitchfork_gates','diagonal_rooms')
TEST_FAMILIES=('irregular_room_chain','serpentine_spurs','bilateral_deadends','double_gate_ring','destination_enclosure','forced_bridge','island_loops','switchback_spine')
DEMO_NAMES=('New static maze','Permanent dead end with detour','Temporary bridge - Wait','Temporary gate - detour')
TEST_PATH=ROOT/'data/product_phase3_holdout_v1.json';TRAIN_PATH=ROOT/'data/product_phase3_train_dev_v1.json';AUDIT_PATH=ROOT/'data/product_phase3_split_audit_v1.json'

def canonical(rows):
    a=np.array([list(r) for r in rows]);variants=[]
    for k in range(4):
        b=np.rot90(a,k);variants+=[''.join(b.ravel()),''.join(np.fliplr(b).ravel())]
    return digest(min(variants))

def structure(rows):
    free={(r,c) for r in range(10) for c in range(10) if rows[r][c]!='#'};neighbors={p:sorted(q for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)) if q in free) for p in free};labels={p:str(len(n)) for p,n in neighbors.items()}
    for _ in range(5):labels={p:digest([labels[p],sorted(labels[q] for q in neighbors[p])]) for p in free}
    edges=sum(map(len,neighbors.values()))//2
    return digest({'vertices':len(free),'edges':edges,'degree_counts':sorted(Counter(map(len,neighbors.values())).items()),'wl5':sorted(Counter(labels.values()).items())})

def geometry(family,rng):
    a=np.full((10,10),'#',dtype='<U1')
    def room(r0,r1,c0,c1):a[r0:r1,c0:c1]='.'
    def path(points):
        for (r,c),(s,t) in zip(points,points[1:]):
            assert r==s or c==t
            a[min(r,s):max(r,s)+1,min(c,t):max(c,t)+1]='.'
    if family=='irregular_room_chain':
        w=int(rng.integers(2,4));h=int(rng.integers(2,4));room(1,1+h,1,1+w);room(1,4,6,9);room(6,9,5,9);room(6,9,1,3)
        path([(2,2),(2,7),(7,7),(7,2)]);a[6,int(rng.integers(5,8))]='#'
    elif family=='serpentine_spurs':
        for row in (1,3,5,7):room(row,row+1,1,9)
        path([(1,8),(3,8)]);path([(3,1),(5,1)]);path([(5,8),(7,8)])
        for c in rng.choice(np.arange(2,8),2,replace=False):room(7,int(rng.integers(9,11)),int(c),int(c)+1)
    elif family=='bilateral_deadends':
        room(1,9,4,5)
        for row in (1,4,7):room(row,row+1,4-int(rng.integers(1,4)),5+int(rng.integers(1,5)))
    elif family=='double_gate_ring':
        room(1,2,1,9);room(8,9,1,9);room(1,9,1,2);room(1,9,8,9);h=int(rng.integers(2,5));w=int(rng.integers(2,5));room(3,3+h,3,3+w)
        c=int(rng.integers(3,3+w));d=int(rng.integers(3,3+w));path([(1,c),(3,c)]);path([(2+h,d),(8,d)])
    elif family=='destination_enclosure':
        room(1,9,1,9);a[2,2:8]='#';a[2:8,2]='#';a[2:8,7]='#';a[7,2:8]='#';a[7,int(rng.integers(3,7))]='.'
        a[1,int(rng.integers(3,7))]='#';a[8,int(rng.integers(2,7))]='#'
    elif family=='forced_bridge':
        room(1,4,1,4);room(6,9,6,9);col=int(rng.integers(4,7));path([(3,2),(3,col),(7,col),(7,7)])
        a[1,int(rng.integers(1,4))]='#';a[8,int(rng.integers(6,9))]='#'
    elif family=='island_loops':
        room(1,9,1,9);a[2:2+int(rng.integers(2,5)),2:4]='#';a[5:5+int(rng.integers(2,4)),6:8]='#';a[int(rng.integers(2,5)),4]='#';a[int(rng.integers(5,8)),5]='#'
    elif family=='switchback_spine':
        for c in (1,4,7):room(1,9,c,c+1)
        path([(1,1),(1,4)]);path([(8,4),(8,7)]);row=int(rng.integers(2,7));room(row,row+2,7,10);a[int(rng.integers(3,7)),0]='.'
    elif family=='comb_yard':
        col=int(rng.integers(1,4));room(1,9,col,col+1)
        for row in (1,3,5,7):room(row,row+1,col,col+int(rng.integers(3,7)))
    elif family=='radial_pods':
        path([(1,4),(8,4)]);path([(4,1),(4,8)])
        for rr,cc in ((1,4),(8,4),(4,1),(4,8)):
            if rng.random()<.75:room(max(0,rr-1),min(10,rr+int(rng.integers(1,3))),max(0,cc-1),min(10,cc+int(rng.integers(1,3))))
    elif family=='parallel_rails':
        top=int(rng.integers(1,4));bottom=int(rng.integers(6,9));room(top,top+1,1,9);room(bottom,bottom+1,1,9)
        for c in rng.choice(np.arange(1,9),3,replace=False):room(top,bottom+1,int(c),int(c)+1)
    elif family=='three_room_line':
        row=int(rng.integers(3,7));path([(row,1),(row,8)])
        for c in (1,4,7):room(row-int(rng.integers(1,3)),row+int(rng.integers(1,3)),c,min(10,c+2))
    elif family=='tree_carving':
        start=(1,1);visited={start};stack=[start];a[start]='.'
        while stack:
            r,c=stack[-1];opts=[(r+dr,c+dc) for dr,dc in ((-2,0),(2,0),(0,-2),(0,2)) if 1<=r+dr<=7 and 1<=c+dc<=7 and (r+dr,c+dc) not in visited]
            if not opts:stack.pop();continue
            nxt=opts[int(rng.integers(len(opts)))];a[(r+nxt[0])//2,(c+nxt[1])//2]='.';a[nxt]='.';visited.add(nxt);stack.append(nxt)
    elif family=='four_court_star':
        path([(2,4),(7,4)]);path([(4,2),(4,7)])
        for rr,cc in ((1,3),(6,3),(3,1),(3,6)):room(rr,min(10,rr+int(rng.integers(2,4))),cc,min(10,cc+int(rng.integers(2,4))))
    elif family=='triangular_yards':
        room(1,int(rng.integers(3,5)),1,int(rng.integers(3,5)));room(1,int(rng.integers(3,5)),6,int(rng.integers(8,10)));room(6,int(rng.integers(8,10)),4,int(rng.integers(6,9)));path([(2,2),(2,7),(7,7),(7,5),(7,2),(2,2)]);a[2,int(rng.integers(4,6))]='.';a[5,int(rng.integers(1,3))]='.'
    elif family=='lollipop_court':
        room(1,2,3,9);room(6,7,3,9);room(1,7,3,4);room(1,7,8,9);path([(6,5),(9,5)]);a[int(rng.integers(2,6)),9]='.'
        if rng.random()<.5:room(8,10,4,7)
    elif family=='pitchfork_gates':
        room(8,9,1,9)
        for col in (1,4,8):room(int(rng.integers(1,5)),9,col,col+1)
        row=int(rng.integers(4,7));room(row,row+1,4,7)
    elif family=='diagonal_rooms':
        room(0,int(rng.integers(2,4)),0,int(rng.integers(2,4)));room(3,int(rng.integers(5,7)),3,int(rng.integers(5,7)));room(7,10,7,10);path([(2,2),(2,4),(4,4),(4,8),(8,8)]);a[3,int(rng.integers(3,6))]='.';a[int(rng.integers(7,10)),6]='.'
    else:raise ValueError('Unknown structural family')
    k=int(rng.integers(4));return [''.join(row) for row in np.rot90(a,k)]

def shortest(rows,start,goal,remove=None):
    todo=deque([tuple(start)]);previous={tuple(start):None}
    while todo:
        p=todo.popleft()
        if p==tuple(goal):
            result=[]
            while p is not None:result.append(p);p=previous[p]
            return result[::-1]
        for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)):
            if all(0<=v<10 for v in q) and rows[q[0]][q[1]]!='#' and q!=remove and q not in previous:previous[q]=p;todo.append(q)
    return None

def _build(split,families,per_family,seed,forbidden_layouts=(),forbidden_structures=()):
    rng=np.random.default_rng(seed);used=set(forbidden_layouts);blocked_structures=set(forbidden_structures);result=[]
    for family in families:
        for index in range(per_family):
            long=index%2==1
            for attempt in range(5000):
                rows=geometry(family,rng);layout=canonical(rows);topology=structure(rows)
                if layout in used or topology in blocked_structures:continue
                grid=GridMap('generator',tuple(rows));free=[(r,c) for r in range(10) for c in range(10) if rows[r][c]!='#'];start=free[int(rng.integers(len(free)))];distances=grid.distances(start)
                possible=[p for p,d in distances.items() if (8<=d<=40 if long else 2<=d<=7)]
                if not possible:continue
                goal=possible[int(rng.integers(len(possible)))];path=shortest(rows,start,goal);dynamic=(family=='forced_bridge' or (index//2)%2==0)
                events=[];block_kind='none'
                if dynamic:
                    bottlenecks=[(i,p) for i,p in enumerate(path[1:-1],1) if shortest(rows,start,goal,remove=p) is None]
                    alternatives=[(i,p) for i,p in enumerate(path[1:-1],1) if shortest(rows,start,goal,remove=p) is not None]
                    candidates=bottlenecks if family=='forced_bridge' else (bottlenecks if family in ('serpentine_spurs','bilateral_deadends','switchback_spine','comb_yard','tree_carving','pitchfork_gates') and bottlenecks else alternatives or bottlenecks)
                    if not candidates:continue
                    at,cell=candidates[0];clear=min(60,at+int(rng.integers(6,19)));events=[{'id':'product-occupancy-blockage','cell':list(cell),'start':0,'clear':clear}];block_kind='forced-wait' if shortest(rows,start,goal,remove=cell) is None else 'detour-available'
                case=make_scenario(rows,list(start),list(goal),events,scenario_id=f'p3-{split}-{len(result):03d}',layout_id=f'p3-{family}-{index:03d}',event_seed=seed+len(result),split='product-heldout' if split=='holdout' else 'validation' if split=='development' else 'train',condition=family)
                degree=[sum(q in set(free) for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))) for p in free]
                case.update(topology_family=family,canonical_layout_hash=layout,structural_hash=topology,base_distance=len(path)-1,distance_band='long' if long else 'short',condition_type='dynamic' if events else 'static',blockage_kind=block_kind,dead_end_count=sum(d==1 for d in degree),curriculum_stage=0 if len(path)-1<=4 else 1 if len(path)-1<=9 else 2,product_scope='phase3-'+split)
                used.add(layout);result.append(case);break
            else:raise RuntimeError(f'Could not generate independent case {split}/{family}/{index}')
    return result

def freeze():
    if TEST_PATH.exists() or TRAIN_PATH.exists():raise FileExistsError('Product Phase 3 datasets already frozen; never regenerate in place')
    # Held-out structures are drawn first, without opening old scientific assets.
    held=_build('holdout',TEST_FAMILIES,8,93217)
    from .product_curriculum import manifest as old_manifest
    old=old_manifest();phase2=old['training_cases']+old['development_cases'];old_layout={canonical(c['rows']) for c in phase2};old_structure={structure(c['rows']) for c in phase2}
    assert not old_layout&{c['canonical_layout_hash'] for c in held},'Unexpected Phase 2 exact structural reuse; do not adapt test difficulty'
    assert not old_structure&{c['structural_hash'] for c in held},'Unexpected Phase 2 structural overlap'
    test_l={c['canonical_layout_hash'] for c in held};test_s={c['structural_hash'] for c in held}
    training=_build('train',TRAIN_FAMILIES,48,93317,old_layout|test_l,old_structure|test_s)
    train_l={c['canonical_layout_hash'] for c in training};train_s={c['structural_hash'] for c in training}
    dev=_build('development',DEV_FAMILIES,8,93417,old_layout|test_l|train_l,old_structure|test_s|train_s)
    payload={'version':VERSION,'seed':93217,'scope':'New product-only held-out suite; not historical scientific final/unseen data','cases':held,'case_hashes':[digest(c) for c in held],'inclusion':'64 base-reachable cases; finite initial occupancy events clear within60; base path<=40, so waiting until all clear then following a base path is feasible below200. No outcomes were used during construction.'};payload['suite_hash']=digest(payload)
    audit={'version':'phase3-structural-audit-v1','phase2_cases':len(phase2),'train_cases':len(training),'development_cases':len(dev),'holdout_cases':len(held),'holdout_families':list(TEST_FAMILIES),'train_families':list(TRAIN_FAMILIES),'development_families':list(DEV_FAMILIES),'layout_hash_overlap':sorted(test_l&(old_layout|train_l|{c['canonical_layout_hash'] for c in dev})),'structural_hash_overlap':sorted(test_s&(old_structure|train_s|{c['structural_hash'] for c in dev})),'family_overlap':sorted(set(TEST_FAMILIES)&(set(TRAIN_FAMILIES)|set(DEV_FAMILIES)|{c['product_kind'] for c in phase2})),'train_development_structural_overlap':sorted(train_s&{c['structural_hash'] for c in dev}),'canonicalization':'Minimum of all8 rotations/reflections of full occupancy; no start/goal dependence','structural_fingerprint':'Free-cell graph vertex/edge/degree census and five-round Weisfeiler-Lehman label histogram, independent of coordinates; not a full graph-isomorphism proof'}
    write_json(TEST_PATH,payload);TEST_PATH.with_suffix('.sha256').write_text(sha(TEST_PATH)+'\n',encoding='utf-8')
    write_json(TRAIN_PATH,{'version':'phase3-train-dev-v1','seeds':{'train':93317,'development':93417},'training':training,'development':dev,'heldout_suite_hash':payload['suite_hash'],'heldout_evaluation_for_selection':False});write_json(AUDIT_PATH,audit)
    return {'holdout_suite_hash':payload['suite_hash'],'heldout_file_sha256':sha(TEST_PATH),'audit':audit}

def train_dev():return read_json(TRAIN_PATH)

def verify_frozen():
    expected=TEST_PATH.with_suffix('.sha256').read_text().strip()
    if sha(TEST_PATH)!=expected:raise ValueError('Frozen product holdout checksum mismatch')
    return read_json(TEST_PATH)

if __name__=='__main__':
    import json
    print(json.dumps(freeze(),indent=2))
