"""Real core transitions distinguish cycles, collisions, detours and route groups."""
from itertools import cycle
import csv

import numpy as np
import pytest

from robot_navigation.core import GridMap,NavigationEnv,Task,load_task,ROOT,read_json
from robot_navigation.episodes import rollout,validate_replay
from robot_navigation.unseen_diagnostics import diagnose,grouped,STUDY,saved


def recording(actions,*,start=(1,1),goal=(3,3),rows=('....',)*4,max_steps=200):
    grid=GridMap('diagnostic-fixture',rows)
    task=Task(grid,{**load_task(ROOT).config,'map_id':grid.map_id,'max_steps':max_steps})
    scenario={'id':'fixture-route','start':list(start),'goal':list(goal),
              'shortest_length':len(grid.path(start,goal))-1}
    sequence=cycle(actions)
    class Script:
        policy_id='diagnostic-fixture'
        def act(self,observation):
            return next(sequence)
    episode=rollout(NavigationEnv(task,[scenario]),Script(),scenario)
    validate_replay(task,episode)
    return task,episode


def test_two_cell_timeout_and_geometric_annotations():
    task,episode=recording([0,1])
    result=diagnose(task,episode)
    assert result=={'category':'two-cell timeout','cycle_start':0,'first_return':2,'period':2,
        'actions_after_first_return':198,'distinct_cells':2,'minimum_goal_distance':4,
        'final_goal_distance':4,'closest_approach_action':0,'failure_cell':[1,1],'event_action':2}


def test_longer_cycle_and_transient_prefix_use_the_first_return():
    task,episode=recording([0,3,1,2])
    result=diagnose(task,episode)
    assert result['category']=='longer-cycle timeout' and result['period']==4
    assert result['minimum_goal_distance']==3 and result['closest_approach_action']==3
    # A prefix visits three distinct states before the two-state suffix begins.
    task,episode=recording([0,0,3]+[0,1]*99,start=(3,0))
    result=diagnose(task,episode)
    assert result['cycle_start']==3 and result['first_return']==5 and result['period']==2
    assert result['actions_after_first_return']==195 and result['failure_cell']==[1,1]
    task,episode=recording([0,0,3,0,1],start=(3,0))
    # The cyclic script eventually changes an action at an old state, so a frozen
    # deterministic-policy diagnosis must reject it instead of inventing a cycle.
    with pytest.raises(ValueError,match='inconsistent state actions'):
        diagnose(task,episode)


@pytest.mark.parametrize('action,start,rows,category',[
    (0,(1,1),('.#..','....','....','....'),'wall collision'),
    (0,(0,0),('....',)*4,'boundary collision')])
def test_collision_terminal_self_return_is_not_a_cycle(action,start,rows,category):
    task,episode=recording([action],start=start,rows=rows)
    result=diagnose(task,episode)
    assert result['category']==category and result['period'] is None
    assert result['failure_cell']==list(start) and result['event_action']==1


@pytest.mark.parametrize('actions,category',[
    ([3,3],'optimal success'),([0,3,3,1],'detour success')])
def test_successful_route_lengths_are_separate_from_timeouts(actions,category):
    task,episode=recording(actions,goal=(1,3))
    result=diagnose(task,episode)
    assert result['category']==category and result['failure_cell'] is None
    assert result['final_goal_distance']==0 and result['period'] is None


def test_nonperiodic_short_timeout_retains_zero_cycle_fields():
    task,episode=recording([3,3,3,1,1,2],start=(0,0),max_steps=6)
    result=diagnose(task,episode)
    assert result['category']=='non-periodic timeout' and result['period'] is None
    assert result['distinct_cells']==7 and result['actions_after_first_return']==0


def test_rates_weight_seeds_and_preserve_zero_success_groups():
    def row(seed,category,outcome):
        return {'dataset':'final','algorithm':'q-learning','training_seed':seed,'category':category,
                'outcome':outcome,'steps':2 if outcome=='success' else 200,
                'actions_after_first_return':198 if outcome=='timeout' else 0,
                'period':2 if outcome=='timeout' else None,'cycle_start':0 if outcome=='timeout' else None}
    result=grouped([row(7,'optimal success','success'),*[row(9,'two-cell timeout','timeout') for _ in range(3)]])[0]
    assert result['success_mean']==.5 and np.isclose(result['success_sd'],np.sqrt(.5))
    assert sum(result['counts'].values())==4 and result['periodic_timeouts']==3
    failed=grouped([row(7,'two-cell timeout','timeout'),row(9,'two-cell timeout','timeout')])[0]
    assert failed['success_mean']==failed['success_sd']==0 and failed['counts']['optimal success']==0


def test_saved_diagnostics_reconcile_all_sources_without_policy_queries(monkeypatch):
    path=ROOT/'artifacts/studies'/STUDY/'summary.json'
    if not path.exists() or read_json(path)['status']!='verified':
        pytest.skip('The diagnostic study is not built yet')
    import robot_navigation.unseen_agents as agents
    import robot_navigation.unseen_final as final
    def forbidden(*args,**kwargs):
        raise AssertionError('Saved diagnosis cannot query a policy')
    monkeypatch.setattr(agents,'load_map_policy',forbidden)
    monkeypatch.setattr(final,'load_map_policy',forbidden)
    from robot_navigation.unseen_diagnostics import calculate
    observed=calculate(ROOT)
    summary=saved(ROOT)
    assert len(observed['rows'])==summary['episodes']==8064
    assert observed['groups']==summary['groups'] and observed['examples']==summary['examples']
    from robot_navigation.unseen_diagnostics import CATEGORIES
    with (path.parent/'seeds.csv').open(newline='',encoding='utf-8') as file:
        seeds=list(csv.DictReader(file))
    assert len(seeds)==24
    for row in seeds:
        cases=[r for r in observed['rows'] if (r['dataset'],r['algorithm'],r['training_seed'])==
               (row['dataset'],row['algorithm'],int(row['training_seed']))]
        assert int(row['episodes'])==len(cases)
        assert all(int(row[category])==sum(r['category']==category for r in cases) for category in CATEGORIES)


def test_source_and_output_changes_are_rejected_without_modifying_evidence(monkeypatch):
    path=ROOT/'artifacts/studies'/STUDY/'manifest.json'
    if not path.exists():
        pytest.skip('The diagnostic study is not built yet')
    import robot_navigation.unseen_diagnostics as diagnostics
    original=diagnostics.sha
    for name in ('episodes.csv','validation.jsonl.gz','milestone13_protocol.md'):
        with monkeypatch.context() as patch:
            patch.setattr(diagnostics,'sha',lambda path: 'changed' if path.name==name else original(path))
            with pytest.raises(ValueError):
                saved(ROOT)
