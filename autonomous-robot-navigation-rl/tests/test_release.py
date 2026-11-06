"""Final gate tests use tiny fixtures, never execute the real final manifest."""
import shutil

import pytest

from robot_navigation.agents import BFSPolicy
from robot_navigation.core import ROOT, GridMap, NavigationEnv, Task, digest, load_task, read_json, write_json
from robot_navigation.episodes import rollout, validate_replay
from robot_navigation.release import registry, sha, verify_freeze, final_summary


def test_final_mode_requires_explicit_evaluation_context():
    base=load_task()
    task=Task(GridMap('fixture',('...',)),{**base.config,'max_steps':4})
    scenario={'id':'fixture','start':[0,0],'goal':[0,2],'shortest_length':2}
    policy=BFSPolicy(task.grid)
    with pytest.raises(ValueError):
        rollout(NavigationEnv(task,[scenario]),policy,scenario,mode='final evaluation')
    episode=rollout(NavigationEnv(task,[scenario]),policy,scenario,mode='final evaluation',allow_final=True)
    with pytest.raises(ValueError):
        validate_replay(task,episode)
    assert validate_replay(task,episode,allow_final=True)['metrics']['spl']==1


@pytest.mark.parametrize('target',['original','snapshot','seal'])
def test_freeze_rejects_changed_inputs(tmp_path,target):
    root=tmp_path/'project';root.mkdir()
    file=root/'input.txt';file.write_text('frozen',encoding='utf-8')
    study=tmp_path/'study';snapshot=study/'snapshot';snapshot.mkdir(parents=True)
    shutil.copy2(file,snapshot/file.name)
    record={'files':{file.name:sha(file)}}
    signature=digest(record)
    write_json(study/'freeze.json',record);write_json(study/'seal.json',{'freeze_digest':signature})
    assert verify_freeze(study,signature,root)==record
    if target=='seal':
        write_json(study/'seal.json',{'freeze_digest':'changed'})
    else:
        (file if target=='original' else snapshot/file.name).write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='changed|mismatch'):
        verify_freeze(study,signature,root)


def test_registry_retains_all_declared_seeds_and_references(tmp_path):
    rows=registry()
    assert len(rows)==42 and len({r['id'] for r in rows})==42
    assert sum(r['kind']=='learned' for r in rows)==35
    assert sum(r['kind']=='random' for r in rows)==5
    assert [r['seed'] for r in rows if r['group']=='Q-learning Refined']==[101,211,307,401,503]
    assert final_summary(tmp_path) is None


def test_final_pipeline_on_synthetic_manifest_only(tmp_path):
    """Exercise serialization/audit/publication with fixture BFS stand-ins.

    This is a one-route synthetic release, not learned-family performance data.
    No real final scenario file is read or any saved learned policy executed.
    """
    import importlib.util
    import yaml
    spec=importlib.util.spec_from_file_location('final_runner_fixture',ROOT/'scripts/run_final_evaluation.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    root=tmp_path/'project';study=root/'artifacts/studies/fixture-final'
    config={**load_task().config,'map_id':'fixture-release','max_steps':4}
    (root/'configs').mkdir(parents=True)
    (root/'configs/environment.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    write_json(root/'data/maps/fixture-release.json',{'map_id':'fixture-release','rows':['...']})
    task=load_task(root)
    scenarios=[{'id':'fixture-only','start':[0,0],'goal':[0,2],'shortest_length':2,'distance_band':'short'}]
    write_json(root/'data/scenarios/test.json',{'signature':task.signature,'split':'test','scenarios':scenarios})
    files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    for relative in files:
        destination=study/'snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,destination)
    records=[{**r,'kind':'bfs'} for r in registry()]
    freeze={'files':files,'signature':task.signature,'test_scenario_hash':digest(scenarios),'test_scenario_count':1,
        'records':records,'limits':{'wall_seconds':900,'evaluation_storage_bytes':500*1024**2},
        'selected_system':'m4-tuned-q-learning-seed101','selected_control':'m4-control-q-learning-seed101'}
    signature=digest(freeze)
    write_json(study/'freeze.json',freeze);write_json(study/'seal.json',{'freeze_digest':signature})
    runner.run(study,signature,root)
    assert read_json(study/'status.json')['final_evaluation_complete']
    audit=read_json(study/'evaluation/verification.json')
    assert audit['evaluation_episodes']==audit['audit_episode_replays']==42
    assert audit['evaluation_interactions']==audit['audit_actions']==84
    assert audit['bfs_all_optimal']
    with pytest.raises(FileExistsError):
        runner.run(study,signature,root)
