"""Exercise the map-aware pilot through the existing dashboard entry point."""
from streamlit.testing.v1 import AppTest
import pytest

from robot_navigation.core import ROOT


def test_unseen_results_and_four_learning_players_use_saved_actions(monkeypatch):
    import robot_navigation.unseen_agents as agents
    viewer = AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    next(w for w in viewer.selectbox if w.label=='Dashboard view').set_value('Unseen maps').run()
    assert not viewer.exception
    assert len(viewer.dataframe[0].value) == 4

    def forbidden(*args, **kwargs):
        raise AssertionError('Recorded playback must not query a trained map-aware policy')
    monkeypatch.setattr(agents,'load_map_policy',forbidden)
    viewer.session_state['unseen_view'] = 'Replay'
    viewer.run()
    next(w for w in viewer.selectbox if w.label=='Recording type').set_value('Recorded learning').run()
    for algorithm in ('q-learning','sarsa','dqn','ppo'):
        next(w for w in viewer.selectbox if w.label=='Map-aware algorithm').set_value(algorithm).run()
        assert not viewer.exception
        player = viewer.session_state['unseen_playback']
        assert player.index == 0 and player.episode['mode'] == 'training'
        next(b for b in viewer.button if b.label=='Next step').click().run()
        assert not viewer.exception
        player = viewer.session_state['unseen_playback']
        assert player.current['position'] == player.episode['transitions'][0]['next_observation'][:2]
        assert player.current['return'] == player.episode['transitions'][0]['reward']
    next(w for w in viewer.selectbox if w.label=='Map-aware algorithm').set_value('dqn').run()
    next(b for b in viewer.button if b.label=='Play').click().run()
    previous = viewer.session_state['unseen_playback']
    assert previous.playing
    next(w for w in viewer.selectbox if w.label=='Dashboard view').set_value('Overview').run()
    assert not viewer.exception and not previous.playing


def test_main_study_replays_all_training_seeds_and_pauses_on_study_change(monkeypatch):
    from robot_navigation.core import read_json
    from robot_navigation.unseen_main import STUDY
    path = ROOT/'artifacts/studies'/STUDY/'summary.json'
    if not path.exists() or read_json(path)['status']!='verified':
        pytest.skip('The registered main study has not completed verification yet')
    import robot_navigation.unseen_agents as agents
    def forbidden(*args,**kwargs):
        raise AssertionError('Playback cannot load or query a trained policy')
    monkeypatch.setattr(agents,'load_map_policy',forbidden)
    viewer = AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=30).run()
    next(w for w in viewer.selectbox if w.label=='Dashboard view').set_value('Unseen maps').run()
    next(w for w in viewer.selectbox if w.label=='Unseen-map study').set_value(STUDY).run()
    assert len(viewer.dataframe[0].value)==4
    viewer.session_state['unseen_view'] = 'Replay'
    viewer.run()
    for kind in ('Recorded learning','Trained policy'):
        next(w for w in viewer.selectbox if w.label=='Recording type').set_value(kind).run()
        for algorithm in ('q-learning','sarsa','dqn','ppo'):
            next(w for w in viewer.selectbox if w.label=='Map-aware algorithm').set_value(algorithm).run()
            for seed in (12101,12203,12301):
                next(w for w in viewer.selectbox if w.label=='Training seed').set_value(seed).run()
                assert not viewer.exception
                player = viewer.session_state['unseen_playback']
                assert player.index==0 and f'{algorithm}-seed{seed}' in player.episode['policy_id']
                next(b for b in viewer.button if b.label=='Next step').click().run()
                player = viewer.session_state['unseen_playback']
                assert player.current['position']==player.episode['transitions'][0]['next_observation'][:2]
    next(b for b in viewer.button if b.label=='Reset').click().run()
    next(b for b in viewer.button if b.label=='Play').click().run()
    previous = viewer.session_state['unseen_playback']
    assert previous.playing
    next(w for w in viewer.selectbox if w.label=='Unseen-map study').set_value('unseen-map-pilot-v1').run()
    assert not viewer.exception and not previous.playing
