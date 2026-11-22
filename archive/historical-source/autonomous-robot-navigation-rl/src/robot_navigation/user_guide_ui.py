"""Plain-language explanations backed by the saved evaluation evidence."""
import streamlit as st

from .evidence import EXPECTED_ERRORS
from .release import final_summary


def render_guide():
    st.title('Your guide to robot navigation')
    st.write('Help a simulated robot travel from its starting square to a destination while avoiding walls.')
    for heading, text in (
        ('1. Choose a map', 'Open Demo and use Navigation map. Try the dead-end maze to see tight passages, or rooms and doorways for a more open layout.'),
        ('2. Watch the robot', 'Press Play. The blue circle marked R is the robot, S is its start and G is the goal. Dark squares are walls. The blue line shows where it has travelled.'),
        ('3. Explore a route', 'Pause whenever you want. Next step shows one move, Reset starts the same replay again, and the timeline jumps to a step. Choose a route lets you set a different start and destination.'),
        ('4. Compare behaviour', 'Compare robots runs two saved models on the same route. Check whether each reaches the goal and how many moves it takes. A single example is not an overall performance score.'),
    ):
        st.subheader(heading)
        st.write(text)
    st.subheader('What do the outcomes mean?')
    st.write('**Goal reached:** the robot arrived. **Collision:** it tried to move into a wall or outside the map. '
             '**Step limit reached:** it used all 200 moves without arriving. A repeating route means it has become stuck in a pattern.')
    st.subheader('Why use Q-learning?')
    st.write('Q-learning stores a score for each possible move at each robot-and-goal position. During training it tries moves and updates those scores from rewards. '
             'This is a practical fit for a small grid: its decisions are quick and its scores can be inspected. '
             'The tuned version was selected using development results on the original map. The newer maps have their own Q-learning models.')
    st.info('The robot already learned before you opened this app. Pressing Play shows its decisions; it does not train it or automatically rescue it when it gets stuck.')
    st.subheader('Other methods you may see')
    st.write('**SARSA, DQN and PPO** are alternative learning methods available for the original map. '
             '**BFS** is a route planner that knows the full map; it provides a shortest-path reference. '
             '**Random** chooses moves without learning. These are different starting points, so their results need context.')
    st.subheader('How should I read the results?')
    st.write('Success rate means the share of routes completed. Collision rate means the share ending in a wall or boundary attempt. '
             'Timeout rate means the share that ran out of moves. Route efficiency (SPL) rewards both arriving and using a short path; higher is better, up to 1.')
    st.write('Results from the original map, new structured maps and previously unseen maps are reported separately. '
             'This is a grid simulation; performance here does not establish safety or readiness for a physical robot.')


def render_results_intro(root):
    st.title('How well does the robot navigate?')
    st.write('Start with the saved assessment on the original map. Open the research details below for comparisons and other map groups.')
    try:
        summary = final_summary(root)
        if summary is None:
            st.info('The original-map assessment is not available in this installation.')
            return
        selected = next(row for system in summary['systems'].values() for row in system['seed_results']
                        if row['id'] == summary['selected_system'])
        scores = selected['evaluation']
        a,b,c = st.columns(3)
        a.metric('Routes completed', f"{100*scores['success_rate']:.1f}%")
        b.metric('Routes ending in collision', f"{100*scores['collision_rate']:.1f}%")
        c.metric('Routes that ran out of moves', f"{100*scores['timeout_rate']:.1f}%")
        st.caption(f"Selected Q-learning model · {scores['episodes']:,} held-out start–goal routes on the familiar original map. "
                   'These saved final results are separate from the example shown in Demo.')
        st.write('The robot completes most assessed routes, but some still fail or repeat until the move limit. '
                 'A good example replay does not guarantee success on another route.')
    except (*EXPECTED_ERRORS, StopIteration) as exc:
        st.info(f'This assessment could not be verified: {exc}. Other saved research pages remain available below.')
