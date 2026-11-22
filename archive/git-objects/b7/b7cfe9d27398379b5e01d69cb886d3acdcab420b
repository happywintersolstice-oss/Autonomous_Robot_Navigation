"""Real recordings, tab isolation and the four-family shared action clock."""
from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT, load_task, read_json
from robot_navigation.dashboard import baseline_lineup, training_episode, training_paths
from robot_navigation.episodes import ComparisonPlayback


def app():
    return AppTest.from_file(str(ROOT/'app/streamlit_app.py'), default_timeout=30).run()


def switch(viewer, label):
    # AppTest exposes tab labels but not a tab-click method. Set the tracked
    # widget state to exercise the same active-tab branch and pause boundary.
    viewer.session_state['replay_tab'] = label
    return viewer.run()


def select(viewer, label):
    return next(w for w in viewer.selectbox if w.label == label)


def click(viewer, label):
    return next(w for w in viewer.button if w.label == label).click().run()


def test_sidebar_is_navigation_only_and_tabs_explain_their_jobs():
    viewer = app()
    assert not viewer.exception
    assert [t.label for t in viewer.tabs] == ['Trained replay','Learning replay','Compare agents','Try a route']
    assert [w.label for w in viewer.sidebar.selectbox] == ['Dashboard view']
    assert not viewer.sidebar.radio and not viewer.sidebar.number_input and not viewer.sidebar.checkbox
    assert any('does not learn during replay' in text.value for text in viewer.markdown)
    assert viewer.session_state['playback'].episode['policy_id'].startswith('q-learning/')


def test_learning_plays_actual_chronological_training_actions_without_policy_queries(monkeypatch):
    viewer = app()
    import robot_navigation.simulator_ui as ui
    def forbidden(*args, **kwargs):
        pytest.fail('A training recording must not query the trained policy')
    monkeypatch.setattr(ui, 'load_policy', forbidden)
    switch(viewer, 'Learning replay')
    assert not viewer.exception
    run = select(viewer, 'Training run').value
    path = select(viewer, 'Training episode').value
    expected = read_json(path)
    assert viewer.session_state['playback'].episode == expected
    assert expected['mode'] == 'training' and expected['training_start_interaction'] == 0
    assert training_paths(run)
    click(viewer, 'Next step')
    player = viewer.session_state['playback']
    assert player.current['position'] == expected['transitions'][0]['next_observation'][:2]
    assert player.current['return'] == expected['transitions'][0]['reward']
    assert any('Decision scores were not recorded' in c.value for c in viewer.caption)
    assert any('Older Q-learning, SARSA, DQN and PPO' in c.value for c in viewer.caption)
    assert all('Q score' not in table.value.columns and 'Action probability' not in table.value.columns
               for table in viewer.dataframe)  # Step history is still available.
    select(viewer, 'Training episode').set_value(training_paths(run)[1]).run()
    assert not viewer.exception and viewer.session_state['playback'].index == 0
    assert viewer.session_state['playback'].episode == read_json(training_paths(run)[1])


def test_switching_tabs_pauses_and_separates_training_and_trained_replays():
    viewer = app()
    click(viewer, 'Play')
    original = viewer.session_state['playback']
    switch(viewer, 'Learning replay')
    assert not original.playing and not viewer.exception
    training = viewer.session_state['playback']
    assert training.episode['mode'] == 'training' and training.index == 0
    click(viewer, 'Play')
    switch(viewer, 'Trained replay')
    assert not training.playing and not viewer.exception
    assert viewer.session_state['playback'].episode['mode'] == 'validation'
    assert not viewer.session_state['playback'].playing


def test_training_reader_rejects_tampered_or_unregistered_files(tmp_path):
    import shutil
    run = ROOT/'artifacts/runs/m8-control-dqn-seed101'
    directory = tmp_path/'run'
    shutil.copytree(run/'training_traces', directory/'training_traces')
    shutil.copy2(run/'manifest.json', directory/'manifest.json')
    path = training_paths(directory)[0]
    expected = training_episode(load_task(), directory, path)
    assert expected == read_json(path)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='checksum'):
        training_episode(load_task(), directory, path)
    with pytest.raises(ValueError, match='not declared'):
        training_episode(load_task(), directory, tmp_path/'unknown.json')


def test_training_cutoff_replay_does_not_invent_a_terminal_outcome(monkeypatch):
    from robot_navigation.representation import validate_training_prefix
    import robot_navigation.simulator_ui as ui
    record = read_json(ROOT/'artifacts/runs/m8-control-dqn-seed101/training_traces/episode-001.json')
    record['transitions'] = record['transitions'][:1]
    record.pop('metrics')
    record.update(complete=False, cutoff_reason='budget_cutoff',
                  training_end_interaction=record['training_start_interaction']+1)
    validate_training_prefix(load_task(), record)
    viewer = app()
    monkeypatch.setattr(ui, 'training_episode', lambda *args:deepcopy(record))
    switch(viewer, 'Learning replay')
    click(viewer, 'Next step')
    assert not viewer.exception
    assert viewer.session_state['playback'].current['status']=='running'
    assert any('cut short' in i.value for i in viewer.info)
    assert not any('Completed SPL' in c.value for c in viewer.caption)


def test_four_algorithms_hold_each_terminal_frame_and_reset_when_seed_changes():
    viewer = switch(app(), 'Compare agents')
    next(w for w in viewer.radio if w.label=='What would you like to compare?').set_value('Four algorithms').run()
    assert not viewer.exception
    player = viewer.session_state['comparison_playback']
    assert [e['policy_id'].split('/')[0] for e in player.episodes] == ['q-learning','sarsa','dqn','ppo']
    assert all(e['scenario']==player.episodes[0]['scenario'] for e in player.episodes)
    click(viewer, 'Play')
    viewer.session_state['comparison_tick'] = 0.
    viewer.run()
    assert viewer.session_state['comparison_playback'].index == 1
    click(viewer, 'Pause')
    viewer.slider[0].set_value(player.length).run()
    player = viewer.session_state['comparison_playback']
    assert not viewer.exception and not player.playing
    assert [state['index'] for state in player.current] == [len(e['transitions']) for e in player.episodes]
    assert all(state['status'] in ('success','collision','timeout') for state in player.current)
    table = viewer.dataframe[0].value
    assert list(table['Actions']) == [len(e['transitions']) for e in player.episodes]
    click(viewer, 'Reset')
    assert viewer.session_state['comparison_playback'].index == 0
    select(viewer, 'Training seed').set_value(19).run()
    assert not viewer.exception and viewer.session_state['comparison_playback'].index == 0
    assert all('seed19' in e['policy_id'] for e in viewer.session_state['comparison_playback'].episodes)
    click(viewer, 'Play')
    previous = viewer.session_state['comparison_playback']
    switch(viewer, 'Trained replay')
    assert not viewer.exception and not previous.playing


def test_four_clock_rejects_any_mismatched_task_or_replay_seed():
    record = read_json(ROOT/'artifacts/runs/m4-tuned-q-learning-seed101/replays/q_learning_success.json')
    for field, value in [('mode','training'), ('seed',999), ('scenario',{'id':'other'})]:
        records = [deepcopy(record) for _ in range(4)]
        records[3][field] = value
        with pytest.raises(ValueError, match='matching'):
            ComparisonPlayback(records)
    assert len(baseline_lineup(7)) == 4
    with pytest.raises(ValueError, match='declared'):
        baseline_lineup(101)


def test_run_filter_empty_state_is_clear_and_recoverable():
    viewer = app()
    select(viewer, 'Experiment').set_value('DQN input representation').run()
    select(viewer, 'Algorithm').set_value('PPO').run()
    assert not viewer.exception
    assert any('No saved runs match' in i.value for i in viewer.info)
    select(viewer, 'Algorithm').set_value('DQN').run()
    assert not viewer.exception and select(viewer, 'Trained model').value.name.startswith('m8-')
