"""Milestone 5 group definitions, frozen comparisons and integrated UI behavior."""
import copy

import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.agents import BFSPolicy
from robot_navigation.benchmark import load_scenarios
from robot_navigation.core import ROOT, GridMap, NavigationEnv, load_task, read_json
from robot_navigation.dashboard import PAGES, evidence, scenario_labels, frozen_episode, early_checkpoint
from robot_navigation.episodes import ComparisonPlayback, rollout


def box(app,label):
    return next(item for item in app.selectbox if item.label==label)


def test_geometry_labels_and_validation_partition():
    corridor=GridMap('corridor',('#####','#...#','#####'))
    labels=scenario_labels(corridor,{'start':[1,1],'goal':[1,3],'shortest_length':2})
    assert labels['corridor']=='reference corridor' and labels['wall proximity']=='near wall'
    opened=GridMap('open',('...','...','...'))
    labels=scenario_labels(opened,{'start':[1,1],'goal':[0,2],'shortest_length':2})
    assert labels['corridor']=='no reference corridor' and labels['wall proximity']=='open start'
    task=load_task()
    labels=[scenario_labels(task.grid,s) for s in load_scenarios(task,'validation')]
    assert len(labels)==928
    assert {r['distance'] for r in labels}=={'short','medium','long'}
    assert {r['start region'] for r in labels}=={'NW','NE','SW','SE'}
    assert all(len(r)==6 for r in labels)


def test_comparison_holds_terminal_frames_and_rejects_mismatches():
    task=load_task()
    scenario=load_scenarios(task,'validation')[0]
    success=rollout(NavigationEnv(task,[scenario]),BFSPolicy(task.grid),scenario)
    class Collision:
        policy_id='collision fixture'
        def act(self,observation):
            return 0  # first validation scenario starts in the top row
    failure=rollout(NavigationEnv(task,[scenario]),Collision(),scenario)
    assert len(failure['transitions'])==1 < len(success['transitions'])
    player=ComparisonPlayback([failure,success])
    player.play();player.tick();player.tick()
    assert player.current[0]['index']==1 and player.current[0]['status']=='collision'
    assert player.current[1]['index']==2
    player.seek(1000)
    assert not player.playing and player.current[1]['status']=='success'
    player.reset();assert player.index==0 and not player.playing
    altered=copy.deepcopy(success);altered['mode']='custom demonstration'
    with pytest.raises(ValueError,match='matching'):
        ComparisonPlayback([failure,altered])


def test_group_evidence_reconciles_support_and_seed_matching():
    saved=evidence()
    assert saved['new_training_interactions']==saved['new_benchmark_evaluation_interactions']==0
    for name,system in saved['systems'].items():
        assert system['seed_count']==(1 if 'Voting' in name else 5)
        for dimension in {r['dimension'] for r in system['groups']}:
            groups=[r for r in system['groups'] if r['dimension']==dimension]
            assert sum(r['scenarios_per_seed'] for r in groups)==928
            for i,seed in enumerate(system['seed_results']):
                assert sum(r['seed_results'][i]['metrics']['counts']['success'] for r in groups)==seed['metrics']['counts']['success']
    paired=read_json(ROOT / 'artifacts/studies/milestone5-v1/paired_groups.json')
    assert all([r['seed'] for r in g['seed_differences']]==[101,211,307,401,503] for g in paired)


def test_frozen_matched_replays_reject_nonvalidation_and_use_early_model():
    task=load_task();scenario=load_scenarios(task,'validation')[0]
    directory=ROOT / 'artifacts/runs/m4-tuned-q-learning-seed101'
    early=early_checkpoint(directory)
    first=frozen_episode(task,directory,scenario,early)
    later=frozen_episode(task,directory,scenario)
    assert first['display_checkpoint']!=later['display_checkpoint']
    assert first['scenario']==later['scenario']
    changed={**scenario,'start':[0,1]}
    with pytest.raises(ValueError,match='canonical validation'):
        frozen_episode(task,directory,changed)


@pytest.mark.parametrize('page',[p for p in PAGES if p!='Navigation simulator'])
def test_all_dashboard_evidence_pages(page):
    app=AppTest.from_file(str(ROOT / 'app/streamlit_app.py'),default_timeout=20).run()
    box(app,'Dashboard view').set_value(page).run()
    assert not app.exception
    assert app.title[0].value==page
    if page!='Methodology':
        assert app.dataframe or app.json


def test_overlays_scores_event_seeking_preserve_replay():
    app=AppTest.from_file(str(ROOT / 'app/streamlit_app.py'),default_timeout=15).run()
    box(app,'Saved run').set_value(ROOT / 'artifacts/runs/m4-tuned-q-learning-seed101').run()
    box(app,'Recorded episode').set_value(ROOT / 'artifacts/runs/m4-tuned-q-learning-seed101/replays/q_learning_success.json').run()
    next(b for b in app.button if b.label=='Next step').click().run()
    identity=app.session_state['episode_identity']
    for checkbox in app.checkbox:
        if checkbox.label!='Traveled trail':
            checkbox.set_value(True)
    app.run()
    assert not app.exception
    assert app.session_state['episode_identity']==identity and app.session_state['playback'].index==1
    assert any('Q score' in list(df.value.columns) for df in app.dataframe)
    box(app,'Seek displayed event').set_value(0).run()
    assert app.session_state['playback'].index==0 and app.session_state['seek']==0


def test_matched_comparison_and_early_checkpoint_controls():
    app=AppTest.from_file(str(ROOT / 'app/streamlit_app.py'),default_timeout=20).run()
    directory=ROOT / 'artifacts/runs/m4-tuned-q-learning-seed101'
    box(app,'Saved run').set_value(directory).run()
    next(r for r in app.radio if r.label=='Episode source').set_value('Matched comparison').run()
    box(app,'Compare with run').set_value(ROOT / 'artifacts/runs/m4-tuned-sarsa-seed101').run()
    assert not app.exception
    player=app.session_state['comparison_playback']
    assert player.episodes[0]['scenario']==player.episodes[1]['scenario']
    next(b for b in app.button if b.label=='Next step').click().run()
    assert app.session_state['comparison_playback'].index==1
    next(b for b in app.button if b.label=='Play').click().run()
    app.session_state['comparison_tick']=0.;app.run()
    assert app.session_state['comparison_playback'].index==2
    next(b for b in app.button if b.label=='Pause').click().run()
    assert not app.session_state['comparison_playback'].playing
    next(r for r in app.radio if r.label=='Comparison type').set_value('Early versus selected checkpoint').run()
    assert not app.exception
    assert 'step-0010000' in app.session_state['comparison_playback'].episodes[0]['display_checkpoint']
    assert app.session_state['comparison_playback'].index==0


def test_missing_dashboard_artifacts_are_reported(tmp_path):
    # AppTest wraps a standalone page with a deliberately absent evidence root.
    script=tmp_path / 'missing.py'
    script.write_text('from pathlib import Path\nfrom robot_navigation.core import load_task\n'
                      'from robot_navigation.dashboard_ui import render_page\n'
                      f'render_page("Overview",load_task(),Path({str(tmp_path)!r}))\n',encoding='utf-8')
    app=AppTest.from_file(str(script)).run()
    assert not app.exception
    assert any('evidence unavailable' in info.value for info in app.info)
