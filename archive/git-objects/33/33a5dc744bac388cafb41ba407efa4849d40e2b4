"""PPO dashboard wiring against the real saved software-pilot checkpoint."""
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT


def box(app,label):
    return next(widget for widget in app.selectbox if widget.label==label)


def test_verified_four_family_results_are_exposed_in_dashboard():
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    box(app,'Dashboard view').set_value('Agent comparison').run()
    assert not app.exception
    frame=app.dataframe[0].value
    assert list(frame['Algorithm'])==['Q-learning','SARSA','DQN','PPO']
    assert list(frame['Seeds'])==[5,5,5,5]
    assert any('Frozen final evaluation' in heading.value for heading in app.subheader)
    box(app,'Dashboard view').set_value('Learning analysis').run()
    assert not app.exception
    assert any('Four-algorithm validation monitoring' in heading.value for heading in app.subheader)


def test_real_ppo_replay_probabilities_custom_demo_and_comparison():
    directory=ROOT/'artifacts/runs/m7-ppo-pilot-seed101'
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=20).run()
    box(app,'Trained model').set_value(directory).run()
    box(app,'Recorded episode').set_value(directory/'replays/ppo_success.json').run()
    next(button for button in app.button if button.label=='Next step').click().run()
    assert not app.exception
    player=app.session_state['playback']
    assert player.episode['policy_id'].startswith('ppo/') and player.index==1
    assert any('Action probability' in table.value.columns for table in app.dataframe)
    assert player.current['position']==player.episode['transitions'][0]['next_observation'][:2]
    app.session_state['replay_tab']='Try a route'; app.run()
    box(app,'Frozen policy').set_value('PPO')
    next(button for button in app.button if button.label=='Generate demonstration').click().run()
    assert not app.exception
    assert app.session_state['playback'].episode['mode']=='custom demonstration'
    assert app.session_state['playback'].episode['policy_id'].startswith('ppo/')
    app.session_state['replay_tab']='Compare agents'; app.run()
    box(app,'Second trained model').set_value(ROOT/'artifacts/runs/m3-dqn-seed7').run()
    assert not app.exception
    next(button for button in app.button if button.label=='Next step').click().run()
    player=app.session_state['comparison_playback']
    assert player.index==1 and player.episodes[0]['scenario']==player.episodes[1]['scenario']
    assert player.episodes[0]['policy_id'].startswith('ppo/')
    assert player.episodes[1]['policy_id'].startswith('dqn/')
