"""Plain-language warehouse comparison guide and archived evidence introduction."""
import streamlit as st

from .evidence import EXPECTED_ERRORS
from .release import final_summary


def render_guide():
    st.title('Your guide to warehouse navigation')
    st.write('**Autonomous Warehouse Robot Navigation: A Comparative Study of Q-learning, SARSA, DQN and PPO**')
    st.write('Our question is how four learning algorithms compare when a robot delivers items around the same warehouse. Racks, staged pallets and a packing bench form obstacles; receiving, rack pickup points, packing, dispatch and charging give each route a purpose.')
    for heading,text in (
        ('1. Watch a delivery', 'Open Simulator. Choose an Algorithm, Training seed and Delivery job, then press Play. R is the robot, S is its start and G is its goal. Dark cells block movement. Pause, Next step, Reset and the timeline let you inspect each move.'),
        ('2. Compare all four algorithms', 'Open Compare robots. All four robots receive the same pickup and destination, and advance on one shared move clock. A finished robot holds its final position. Change the delivery job or seed to see a different case. Browse every assessed route includes all 188 routes, including failures.'),
        ('3. See how learning happened', 'Learning history plays actions actually recorded during training. Early actions include exploration and mistakes. Select a training stage and inspect its learning curve. These recordings are samples, not the entire training history.'),
        ('4. Assign your own delivery', 'Choose a route lets you select pickup and destination stations. Create delivery route uses the selected trained algorithm, or BFS for a shortest-route reference. Playback does not change the trained model.'),
        ('5. Read the full comparison', 'Results reports averages and sample standard deviations across three independent training seeds per algorithm. Compare all routes, rather than judging a method from one successful delivery. Download the CSV for the numerical table.'),
    ):
        st.subheader(heading)
        st.write(text)
    st.subheader('Why compare these four methods?')
    st.write('**Q-learning** learns a table of move scores using the best estimated next move. **SARSA** learns a similar table using the next move actually chosen during exploration. Comparing them tests two related learning rules. **DQN** uses a neural network to estimate move scores, with replay memory and a target network. **PPO** learns a probability distribution over moves and a value estimate using limited policy updates. Together they compare table-based and neural learning, and value-based and policy-based approaches.')
    st.write('No method was declared the winner in advance. Q-learning appears first by display order. Each method has three seeds and 600,000 training actions. The floor, coordinate observations, rewards, training routes and assessment routes are shared; their update rules and compute needs differ.')
    st.subheader('What do the outcomes mean?')
    st.write('**Goal reached:** delivery completed. **Collision:** a move hit a wall or boundary and ended the route. **Step limit reached:** the robot used 200 moves without arriving. A repeated pattern can cause a timeout; playback never silently rescues a stuck robot.')
    st.write('**Completion rate** is the fraction of routes completed. **Collision rate** and **timeout rate** describe failed routes. **Route efficiency (SPL)** rewards arriving by a short path, from 0 to 1. **Seed SD** describes variability between the three training runs; it is not a confidence interval. BFS knows the full map and supplies a reference, rather than a fifth learned algorithm.')
    st.subheader('What can this study establish?')
    st.info('The comparison uses 188 development routes on one familiar warehouse. These routes were inspected in earlier work. Results do not establish unseen-warehouse generalization, physical-robot safety or a fresh final holdout score.')
    st.write('Movement is a four-direction grid simulation. Physical parcel loading, robot dimensions, sensors, moving people and battery use are not modeled. Temporary 3D map preview changes the view of this same simulation. Historical maps and results remain available in clearly labelled expandable sections.')


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
                   'These saved final results are separate from the warehouse comparison.')
        st.write('The robot completes most assessed routes, but some still fail or repeat until the move limit. '
                 'A good example replay does not guarantee success on another route.')
    except (*EXPECTED_ERRORS, StopIteration) as exc:
        st.info(f'This assessment could not be verified: {exc}. Other saved research pages remain available below.')
