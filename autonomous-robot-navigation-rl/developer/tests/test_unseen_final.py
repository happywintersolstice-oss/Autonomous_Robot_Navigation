"""Synthetic sealed releases exercise final access without opening real holdouts."""
from copy import deepcopy
import numpy as np
import pytest

from test_unseen import benchmark
from robot_navigation.shared_agents import LinearMapPolicy
from robot_navigation.unseen_audit import _audit_validation, load_recording, replay_task
from robot_navigation.unseen_agents import load_map_policy
from robot_navigation.unseen_benchmark import MapNavigationEnv
from robot_navigation.unseen_final import (STUDY, FinalBenchmark, analyze, evaluate_record,
                                          publish_freeze, validate_result)
from robot_navigation.unseen_study import evaluate


@pytest.fixture
def final_release(benchmark,tmp_path):
    model = tmp_path/'model/selected.npz'
    LinearMapPolicy(benchmark,'q-learning','fixture/q').save(model,{})
    record = {'run_id':'fixture-q','kind':'learned','algorithm':'q-learning','training_seed':7,
              'evaluation_seed':0,'checkpoint':'models/selected.npz'}
    files = {f'data/unseen/{benchmark.directory.name}/{p.name}':p for p in benchmark.directory.glob('*.json')}
    files.update({'models/selected.npz':model,'models/selected.json':model.with_suffix('.json')})
    study = tmp_path/'final-study'
    publish_freeze(study,{'study_id':STUDY,'signature':benchmark.signature,'records':[record]},files)
    return study,FinalBenchmark(study),record


def test_final_access_requires_matching_seal_and_explicit_evaluation(final_release,benchmark):
    study,final,record = final_release
    with pytest.raises(ValueError,match='sealed'):
        benchmark.layouts('test')
    cases = final.cases('test')
    with pytest.raises(ValueError,match='sealed'):
        evaluate(final,cases,'bfs',0,split='test')
    with pytest.raises(ValueError,match='sealed final benchmark'):
        evaluate(benchmark,cases,'bfs',0,split='test',final=True)
    with pytest.raises(ValueError,match='declared training'):
        MapNavigationEnv(final,cases)
    summary,rows,episodes = evaluate(final,cases,'bfs',0,split='test',final=True)
    assert summary['episodes']==6 and summary['success_rate']==summary['spl']==1
    assert all(e['mode']=='final evaluation' and e['release_digest']==final.release_digest for e in episodes)
    with pytest.raises(ValueError,match='Final replay access'):
        replay_task(final,episodes[0])
    replay_task(final,episodes[0],allow_final=True)
    altered = deepcopy(episodes[0])
    altered['release_digest']='another-release'
    with pytest.raises(ValueError,match='matching sealed release'):
        replay_task(final,altered,allow_final=True)
    path = study/'snapshot/models/selected.npz'
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='Frozen input changed'):
        FinalBenchmark(study)


def test_pending_attempts_are_retained_and_complete_results_are_reused(final_release,monkeypatch):
    study,final,record = final_release
    pending = study/'pending'/record['run_id']
    pending.mkdir(parents=True)
    (pending/'partial.txt').write_text('retained',encoding='utf-8')
    def forbidden(*args,**kwargs):
        raise AssertionError('Final evaluation cannot train')
    monkeypatch.setattr(LinearMapPolicy,'update',forbidden)
    result = evaluate_record(study,final,record)
    assert result['status']=='complete' and result['final']['episodes']==6
    assert len(list((study/'incomplete').glob('*/partial.txt')))==1
    directory = study/'runs'/record['run_id']
    policy = load_map_policy(study/'snapshot'/record['checkpoint'],final)
    actions = _audit_validation(final,directory/'trajectories.jsonl.gz',policy,'test',result['final'],final=True)
    assert actions==result['final']['interactions']
    for item in result['replays']:
        _,episode = load_recording(final,directory,item)
        assert episode['mode']=='final evaluation'
    import robot_navigation.unseen_final as runner
    monkeypatch.setattr(runner,'evaluate',forbidden)
    assert evaluate_record(study,final,record)==result
    path = directory/'episodes.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='checksum'):
        validate_result(directory,record,final.release_digest)


def test_expired_evaluation_keeps_partial_results_unpublished(final_release):
    study,final,record = final_release
    with pytest.raises(TimeoutError,match='wall limit'):
        evaluate_record(study,final,record,deadline=0)
    assert not (study/'runs'/record['run_id']).exists()
    assert (study/'pending'/record['run_id']).exists()


def test_cluster_intervals_resample_layouts_and_keep_undefined_lengths():
    def result(algorithm,rates):
        maps = {}
        for i,rate in enumerate(rates):
            maps[str(i)]={'episodes':10,'counts':{'success':int(10*rate),'collision':int(10*(1-rate)),'timeout':0},
                          'success_rate':rate,'collision_rate':1-rate,'timeout_rate':0.,'spl':rate}
        return {'record':{'algorithm':algorithm},'evaluation_seconds':1.,'final':{
            'episodes':20,'counts':{k:sum(m['counts'][k] for m in maps.values()) for k in ('success','collision','timeout')},
            'success_rate':float(np.mean(rates)),'collision_rate':1-float(np.mean(rates)),
            'timeout_rate':0.,'spl':float(np.mean(rates)),'mean_return':0.,
            'mean_successful_route_length':2. if sum(rates) else None,'by_map':maps}}
    config = {'algorithms':['q-learning'],'bootstrap_seed':13411,'bootstrap_samples':5000,'confidence_level':.95}
    rows = [result('q-learning',[0.,1.]),result('bfs',[1.,1.]),result('random',[0.,0.])]
    systems = analyze(rows,config)
    assert systems==analyze(rows,config)
    assert systems[0]['statistics']['success_rate']['layout_ci95']==[0.,1.]
    assert systems[1]['statistics']['success_rate']['layout_ci95']==[1.,1.]
    assert systems[2]['statistics']['mean_successful_route_length']['defined_seeds']==0
    varied = analyze([result('q-learning',[0.,0.]),result('q-learning',[1.,1.]),*rows[1:]],config)
    stats = varied[0]['statistics']
    assert np.isclose(stats['success_rate']['sd'],np.sqrt(.5))
    assert stats['success_rate']['layout_ci95']==[.5,.5]
    assert stats['mean_successful_route_length']=={'mean':2.,'sd':None,'defined_seeds':1}
