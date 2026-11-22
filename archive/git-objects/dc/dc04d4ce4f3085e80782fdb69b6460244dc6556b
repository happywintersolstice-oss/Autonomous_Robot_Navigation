"""Diagnostic links preserve source identity and reuse the centered replay player."""
import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT,read_json
from robot_navigation.unseen_diagnostics import STUDY


def test_diagnostic_event_links_cover_both_cohorts_and_all_training_seeds(monkeypatch):
    path=ROOT/'artifacts/studies'/STUDY/'summary.json'
    if not path.exists() or read_json(path)['status']!='verified':
        pytest.skip('The diagnostic study is not verified yet')
    import robot_navigation.unseen_agents as agents
    import robot_navigation.unseen_final as final
    def forbidden(*args,**kwargs):
        raise AssertionError('Diagnostic playback cannot query a policy')
    monkeypatch.setattr(agents,'load_map_policy',forbidden)
    monkeypatch.setattr(final,'load_map_policy',forbidden)
    summary=read_json(path)
    viewer=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    next(w for w in viewer.selectbox if w.label=='Dashboard view').set_value('Unseen maps').run()
    for dataset,source in (('development','unseen-map-main-v1'),('final','unseen-map-final-v1')):
        next(w for w in viewer.selectbox if w.label=='Unseen-map study').set_value(source).run()
        for algorithm in ('q-learning','sarsa','dqn','ppo'):
            for seed in (12101,12203,12301):
                viewer.session_state['unseen_view']='Failure analysis'
                viewer.run()
                next(w for w in viewer.selectbox if w.label=='Diagnostic algorithm').set_value(algorithm).run()
                assert not viewer.exception and len(viewer.dataframe[0].value)==4
                example=next(e for e in summary['examples'] if
                    (e['dataset'],e['algorithm'],e['training_seed'])==(dataset,algorithm,seed) and e['period'])
                next(w for w in viewer.selectbox if w.label=='Illustrative recording').set_value(example).run()
                next(b for b in viewer.button if b.label=='Replay this event').click().run()
                assert not viewer.exception
                assert viewer.session_state['unseen_view']=='Replay'
                player=viewer.session_state['unseen_playback']
                assert player.index==example['event_action'] and not player.playing
                assert any('Saved diagnosis' in c.value for c in viewer.caption)
                assert player.episode['scenario']['id']==example['scenario_id']
                assert player.episode['layout_id']==example['layout_id']
                transition=player.episode['transitions'][player.index-1]
                assert player.current['position']==transition['next_observation'][:2]
                assert player.current['return']==sum(t['reward'] for t in player.episode['transitions'][:player.index])
    next(b for b in viewer.button if b.label=='Play').click().run()
    previous=viewer.session_state['unseen_playback']
    assert previous.playing
    viewer.session_state['unseen_view']='Failure analysis'
    viewer.run()
    assert not viewer.exception and not previous.playing
    assert next(w for w in viewer.selectbox if w.label=='Diagnostic algorithm').value==algorithm
    assert next(w for w in viewer.selectbox if w.label=='Illustrative recording').value==example


def test_stale_diagnostic_seek_cannot_move_another_recording():
    viewer=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    next(w for w in viewer.selectbox if w.label=='Dashboard view').set_value('Unseen maps').run()
    viewer.session_state['unseen_view']='Replay'
    viewer.run()
    next(b for b in viewer.button if b.label=='Next step').click().run()
    player=viewer.session_state['unseen_playback']
    index=player.index
    viewer.session_state['unseen_requested_seek']={'identity':'another-recording','index':100}
    viewer.run()
    assert not viewer.exception and viewer.session_state['unseen_playback'].index==index


def test_quick_selection_and_event_click_use_the_submitted_example():
    path=ROOT/'artifacts/studies'/STUDY/'summary.json'
    if not path.exists() or read_json(path)['status']!='verified':
        pytest.skip('The diagnostic study is not verified yet')
    examples=read_json(path)['examples']
    viewer=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    next(w for w in viewer.selectbox if w.label=='Dashboard view').set_value('Unseen maps').run()
    for category in ('longer-cycle timeout','boundary collision'):
        viewer.session_state['unseen_view']='Failure analysis'
        viewer.run()
        next(w for w in viewer.selectbox if w.label=='Diagnostic algorithm').set_value('ppo').run()
        example=next(e for e in examples if e['dataset']=='final' and e['algorithm']=='ppo' and e['category']==category)
        # Submit the selection together with the button instead of running in between.
        next(w for w in viewer.selectbox if w.label=='Illustrative recording').set_value(example)
        next(b for b in viewer.button if b.label=='Replay this event').click().run()
        assert not viewer.exception
        player=viewer.session_state['unseen_playback']
        assert player.index==example['event_action']
        assert player.episode['scenario']['id']==example['scenario_id']
        assert viewer.session_state['unseen_diagnostic_context']['category']==category
