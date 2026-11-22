"""A separately versioned warehouse delivery demonstration, not the old benchmark."""
import argparse
from copy import deepcopy
import gzip
import json
from time import perf_counter

import numpy as np
import yaml

from .agents import QLearning
from .core import ROOT,GridMap,Task,NavigationEnv,read_json,write_json,validate_scenario,digest
from .episodes import rollout,aggregate,validate_replay
from .release import sha

MAP_ID = 'warehouse-delivery-v1'
RUN_ID = 'warehouse-delivery-q-learning-v1'


def definition():
    height,width = 13,17
    rows = [['#' if r in (0,height-1) or c in (0,width-1) else '.' for c in range(width)] for r in range(height)]
    fixtures = [
        {'name':'Rack A','kind':'storage rack','bounds':[2,3,6,4]},
        {'name':'Rack B','kind':'storage rack','bounds':[2,7,6,8]},
        {'name':'Rack C','kind':'storage rack','bounds':[2,11,6,12]},
        {'name':'Packing bench','kind':'work bench','bounds':[9,7,9,9]},
        {'name':'Inbound pallets','kind':'staging pallets','bounds':[9,2,9,3]},
    ]
    for item in fixtures:
        r1,c1,r2,c2 = item['bounds']
        for r in range(r1,r2+1):
            for c in range(c1,c2+1):
                rows[r][c] = '#'
    stations = [
        {'id':'receiving','name':'Receiving bay','cell':[10,2],'purpose':'Collect incoming stock'},
        {'id':'rack-a','name':'Rack A pickup','cell':[4,2],'purpose':'Collect stored items from Rack A'},
        {'id':'rack-b','name':'Rack B pickup','cell':[4,6],'purpose':'Collect stored items from Rack B'},
        {'id':'rack-c','name':'Rack C pickup','cell':[4,10],'purpose':'Collect stored items from Rack C'},
        {'id':'packing','name':'Packing station','cell':[10,8],'purpose':'Deliver items for packing'},
        {'id':'dispatch','name':'Dispatch bay','cell':[10,14],'purpose':'Deliver packed parcels for shipping'},
        {'id':'charging','name':'Charging dock','cell':[1,14],'purpose':'Return the robot for charging'},
    ]
    jobs = [
        {'id':'pick-b','name':'Pick an order: Rack B to packing','from':'rack-b','to':'packing',
         'description':'Collect an item beside Rack B, leave the storage aisle and bring it to the packing station.'},
        {'id':'receive-a','name':'Put away stock: receiving to Rack A','from':'receiving','to':'rack-a',
         'description':'Move incoming stock from the receiving bay to the access point beside Rack A.'},
        {'id':'receive-c','name':'Put away stock: receiving to Rack C','from':'receiving','to':'rack-c',
         'description':'Carry incoming stock across the main aisle to the access point beside Rack C.'},
        {'id':'pick-a','name':'Pick an order: Rack A to packing','from':'rack-a','to':'packing',
         'description':'Collect an item beside Rack A and deliver it to the packing station.'},
        {'id':'pick-c','name':'Pick an order: Rack C to packing','from':'rack-c','to':'packing',
         'description':'Collect an item beside Rack C and deliver it to the packing station.'},
        {'id':'ship','name':'Ship a parcel: packing to dispatch','from':'packing','to':'dispatch',
         'description':'Move a packed parcel from the packing station to the dispatch bay.'},
        {'id':'charge','name':'Return to the charging dock','from':'dispatch','to':'charging',
         'description':'After dispatch, return through the perimeter aisle to the charging dock.'},
    ]
    return {'map_id':MAP_ID,'name':'Warehouse delivery floor','rows':[''.join(row) for row in rows],
        'description':'A small warehouse: receiving and dispatch bays face a main cross-aisle, three storage racks and a packing work area.',
        'fixtures':fixtures,'stations':stations,'jobs':jobs,
        'zones':[{'name':'RECEIVING','bounds':[8,1,11,4],'color':'#17313b'},
                 {'name':'PACKING','bounds':[8,6,11,10],'color':'#18352e'},
                 {'name':'DISPATCH','bounds':[8,12,11,15],'color':'#2a2c3c'}],
        'simulation_limits':'Fixed floor plan, point robot and static fixtures. No people, pallet handling, robot footprint or live sensors are simulated.'}


def config(root=ROOT):
    return yaml.safe_load((root/'configs/warehouse_demo.yaml').read_text(encoding='utf-8'))


def task_from_map(layout,settings):
    return Task(GridMap(MAP_ID,tuple(layout['rows'])),{
        'map_id':MAP_ID,'environment_version':'grid-v1','observation_version':'coordinates-v1',
        'action_version':'up-down-left-right-v1','reward_version':'warehouse-step-goal-collision-v1',
        **{k:settings[k] for k in ('max_steps','discount','step_reward','goal_reward','collision_reward')}})


def scenario(grid,start,goal):
    item = {'id':f'r{start[0]}c{start[1]}-r{goal[0]}c{goal[1]}','start':list(start),'goal':list(goal),
            'shortest_length':len(grid.path(start,goal))-1}
    validate_scenario(grid,item)
    return item


def prepare(root=ROOT):
    directory = root/'data/warehouse'/MAP_ID
    settings = config(root)
    layout = definition()
    task = task_from_map(layout,settings)
    stations = {item['id']:item for item in layout['stations']}
    for item in layout['stations']:
        if not task.grid.is_free(item['cell']):
            raise ValueError('A warehouse station is blocked')
    if len(task.grid.distances(layout['stations'][0]['cell']))!=len(task.grid.free_cells):
        raise ValueError('Warehouse floor is disconnected')
    jobs = [{**job,'scenario':scenario(task.grid,stations[job['from']]['cell'],stations[job['to']]['cell'])}
            for job in layout['jobs']]
    held_out = {job['scenario']['id'] for job in jobs}
    rng = np.random.default_rng(settings['seed'])
    train,validation = [],[]
    for station in layout['stations']:
        pairs = [scenario(task.grid,cell,station['cell']) for cell in task.grid.free_cells if list(cell)!=station['cell']]
        rng.shuffle(pairs)
        others = [s for s in pairs if s['id'] not in held_out]
        count = round(len(others)*settings['validation_fraction'])
        validation.extend([s for s in pairs if s['id'] in held_out]+others[:count])
        train.extend(others[count:])
    content = {'map.json':layout,'routes.json':{'signature':task.signature,'jobs':jobs,
        'train':sorted(train,key=lambda s:s['id']),'validation':sorted(validation,key=lambda s:s['id']),
        'scope':'Warehouse demonstration development split; named delivery jobs are excluded from training starts. No independent final assessment.'}}
    if directory.exists():
        for name,value in content.items():
            if read_json(directory/name)!=value:
                raise ValueError('Warehouse version already exists with different data')
    else:
        for name,value in content.items():
            write_json(directory/name,value)
    return task,layout,content['routes.json']


def load(root=ROOT):
    directory = root/'data/warehouse'/MAP_ID
    layout,routes = read_json(directory/'map.json'),read_json(directory/'routes.json')
    task = task_from_map(layout,config(root))
    task.check(routes['signature'])
    return task,layout,routes


def train(root=ROOT):
    task,layout,routes = prepare(root)
    settings = config(root)
    directory = root/'artifacts/warehouse'/RUN_ID
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'status.json',{'status':'training'})
    write_json(directory/'configuration.json',settings)
    write_json(directory/'inputs.json',{'map':sha(root/'data/warehouse'/MAP_ID/'map.json'),
        'routes':sha(root/'data/warehouse'/MAP_ID/'routes.json'),'configuration':sha(root/'configs/warehouse_demo.yaml')})
    policy = QLearning(task,f'q-learning/{RUN_ID}-seed{settings["seed"]}')
    env = NavigationEnv(task,routes['train'])
    rng = np.random.default_rng(settings['seed'])
    obs,_ = env.reset(seed=settings['seed'])
    started = perf_counter()
    monitoring = []
    monitor = [job['scenario'] for job in routes['jobs']]
    for step in range(settings['interactions']):
        epsilon = settings['epsilon_start']+(settings['epsilon_end']-settings['epsilon_start'])*min(1,step/(.8*settings['interactions']))
        action = policy.explore(obs,epsilon,rng)
        nxt,reward,end,cut,_ = env.step(action)
        policy.update(obs,action,reward,nxt,end,settings['alpha'],settings['discount'])
        obs = env.reset()[0] if end or cut else nxt
        if (step+1)%settings['monitor_every']==0:
            scores = aggregate([rollout(NavigationEnv(task,monitor),policy,s)['metrics'] for s in monitor])
            monitoring.append({'step':step+1,**scores})
            print(f'Warehouse: {step+1:,} training actions; named-job monitoring success {scores["success_rate"]:.1%}',flush=True)
    seconds = perf_counter()-started
    policy.save(directory/'final.npz',{'training_seed':settings['seed'],'training_interactions':settings['interactions']})
    reloaded = QLearning.load(directory/'final.npz',task)
    before = reloaded.q.copy()
    records = [rollout(NavigationEnv(task,routes['validation']),reloaded,s) for s in routes['validation']]
    for episode in records:
        validate_replay(task,episode)
    if not np.array_equal(before,reloaded.q):
        raise ValueError('Warehouse validation changed the trained table')
    with gzip.open(directory/'validation.jsonl.gz','wt',encoding='utf-8') as file:
        for item in records:
            file.write(json.dumps(item,allow_nan=False)+'\n')
    summary = {'signature':task.signature,'run_id':RUN_ID,'training_interactions':settings['interactions'],
        'training_seconds':seconds,'training_pairs':len(routes['train']),
        'validation':aggregate([e['metrics'] for e in records]),
        'named_jobs':{job['id']:next(e['metrics'] for e in records if e['scenario']['id']==job['scenario']['id']) for job in routes['jobs']},
        'scope':routes['scope'],'selection':'Fixed final checkpoint after the declared budget; no checkpoint selection.',
        'limitations':'Single training seed on this familiar warehouse; development evidence, not a new final benchmark. Named jobs were monitored during training.'}
    write_json(directory/'monitoring.json',monitoring)
    write_json(directory/'summary.json',summary)
    write_json(directory/'outputs.json',{p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='status.json'})
    write_json(directory/'status.json',{'status':'verified'})
    return summary


def saved(root=ROOT):
    task,layout,routes = load(root)
    directory = root/'artifacts/warehouse'/RUN_ID
    if read_json(directory/'status.json')['status']!='verified':
        raise ValueError('Warehouse model is not verified')
    for name,value in read_json(directory/'outputs.json').items():
        if sha(directory/name)!=value:
            raise ValueError(f'Changed warehouse artifact: {name}')
    pins = read_json(directory/'inputs.json')
    for key,path in (('map',root/'data/warehouse'/MAP_ID/'map.json'),
                     ('routes',root/'data/warehouse'/MAP_ID/'routes.json'),
                     ('configuration',root/'configs/warehouse_demo.yaml')):
        if sha(path)!=pins[key]:
            raise ValueError(f'Changed warehouse input: {key}')
    with gzip.open(directory/'validation.jsonl.gz','rt',encoding='utf-8') as file:
        records = [json.loads(line) for line in file]
    return task,layout,routes,read_json(directory/'summary.json'),records,directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train'))
    args = parser.parse_args()
    result = train() if args.command=='train' else prepare()[2]
    print(json.dumps({k:v for k,v in result.items() if k not in ('train','validation','jobs')},indent=2))


if __name__=='__main__':
    main()
