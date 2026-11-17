"""Presentation-only helpers for the local navigation control room.

All map positions, scores and counters come from existing episode/frame records.
No policy decisions, environment transitions or benchmark writes occur here.
"""
from html import escape
from io import BytesIO
import re
from threading import RLock

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

from .episodes import frame
from .visualization import draw

SELECTED_RUN = 'm4-tuned-q-learning-seed101'
ALGORITHM_LABELS = {'q-learning': 'Q-learning', 'sarsa': 'SARSA', 'dqn': 'DQN',
                    'ppo': 'PPO', 'voting': 'Policy voting'}
# Matplotlib shares process-wide state across Streamlit session threads.
_MAP_RENDER_LOCK = RLock()


def apply_theme():
    from .theme import CONTROL_ROOM_CSS
    st.markdown(CONTROL_ROOM_CSS, unsafe_allow_html=True)


def workspace_header(map_id, page):
    title = 'Live map' if page == 'Navigation simulator' else page
    st.markdown('<div class="lab-topbar"><div><div class="lab-breadcrumb">Autonomous navigation / '
                f'{escape(title)}</div><h1>Navigation Control</h1></div>'
                '<div class="lab-environment"><span class="lab-icon">map</span>'
                f'<div><small>ENVIRONMENT</small><b>{escape(map_id)}</b></div></div></div>',
                unsafe_allow_html=True)


def run_label(directory):
    name=directory.name
    neural = re.fullmatch(r'm9-(pilot|search|control|tuned)-(dqn|ppo)(?:-(c\d\d))?-seed(\d+)(?:-attempt(\d+))?',name)
    if neural:
        stage={'pilot':'Tuning pilot','search':'Tuning search','control':'Fresh baseline','tuned':'Selected settings'}[neural[1]]
        candidate=f' {neural[3]}' if neural[3] else ''
        attempt=f' · Attempt {neural[5]}' if neural[5] else ''
        return f'{ALGORITHM_LABELS[neural[2]]} · {stage}{candidate} · seed {neural[4]}{attempt}'
    demo = re.fullmatch(r'learning-replay-(q-learning|sarsa|dqn|ppo)-seed(\d+)', name)
    if demo:
        return f'{ALGORITHM_LABELS[demo[1]]} · Recorded training demo · seed {demo[2]}'
    if name=='m4-voting-q-learning':
        return 'Q-learning · Three-policy vote'
    if name=='smoke-seed7':
        return 'Q-learning · Initial pilot · seed 7'
    match=re.match(r'm(\d+)-(?:(control|relative|tuned|terminal|search)-)?(q-learning|sarsa|dqn|ppo)-(.*)',name)
    if not match:
        return name
    milestone,stage,algorithm,suffix=match.groups()
    family={'q-learning':'Q-learning','sarsa':'SARSA','dqn':'DQN','ppo':'PPO'}[algorithm]
    stage={'control':'Baseline','tuned':'Tuned','terminal':'Zero step cost','search':'Search'}.get(stage,
           'Pilot' if 'pilot' in suffix else 'Baseline' if milestone=='3' else 'Early baseline')
    if algorithm=='ppo':
        stage='Pilot' if 'pilot' in suffix else 'Additional baseline'
    if milestone=='8' and algorithm=='dqn':
        stage=('Absolute + relative' if stage=='Early baseline' else 'Absolute control')
        if 'relative' in name:
            stage='Absolute + relative'
        if 'pilot' in suffix:
            stage+=' pilot'
    seed=re.search(r'seed(\d+)',suffix)
    candidate=re.search(r'c\d\d',suffix)
    label=f'{family} · {stage}'
    if candidate:
        label+=f' {candidate.group()}'
    if seed:
        label+=f' · seed {seed.group(1)}'
    if name==SELECTED_RUN:
        label+=' · Selected system'
    if name.endswith('-repeat'):
        label+=' · Repeat'
    return label


def preferred_runs(runs):
    priorities=(SELECTED_RUN,'m7-ppo-seed7','m4-voting-q-learning','m4-control-q-learning-seed101',
                'm4-tuned-sarsa-seed101','smoke-seed7')
    return sorted(runs,key=lambda p:(priorities.index(p.name) if p.name in priorities else len(priorities),p.name))


def experiment_group(directory):
    """Human-readable provenance; never infer tuning from measured performance."""
    name = directory.name
    if name.startswith('learning-replay-'):
        return 'Learning replay demonstrations'
    if name.startswith('m9-'):
        return 'Neural tuning'
    if name.startswith('m8-'):
        return 'DQN input representation'
    if name.startswith('m7-'):
        return 'Additional PPO baseline'
    if name.startswith('m3-'):
        return 'Original four-input baselines'
    if name.startswith('m4-tuned-'):
        return 'Tuned tabular confirmation'
    if name.startswith('m4-control-'):
        return 'Matched tabular controls'
    if name.startswith('m4-search-'):
        return 'Hyperparameter search'
    if name.startswith('m4-terminal-'):
        return 'Reward ablation'
    if name.startswith('m4-voting-'):
        return 'Policy voting'
    return 'Early runs and pilots'


def run_selector(runs, key='saved_run', label='Trained model'):
    """Optional library filters; selecting a filter never picks a performance winner."""
    with st.container(key=f'{key}_picker'):
        return _run_selector(runs, key, label)


def _run_selector(runs, key, label):
    from .core import read_json
    selector, filters = st.columns([4, 1.8], vertical_alignment='bottom')
    selection_area = selector.empty()
    with filters.popover('Filter', icon=':material/tune:'):
        group = st.selectbox('Experiment', ['All experiments', *dict.fromkeys(experiment_group(p) for p in runs)],
                             key=f'{key}_experiment', help='Groups runs by the experiment that produced them.')
        family = st.selectbox('Algorithm', ['All algorithms', *ALGORITHM_LABELS.values()], key=f'{key}_algorithm')
    filtered = [p for p in runs if (group == 'All experiments' or experiment_group(p) == group)
                and (family == 'All algorithms' or ALGORITHM_LABELS.get(
                    read_json(p / 'manifest.json').get('algorithm', 'q-learning')) == family)]
    if not filtered:
        st.info('No saved runs match these filters. Choose another experiment or algorithm.')
        return None
    if st.session_state.get(key) not in filtered:
        st.session_state[key] = filtered[0]
    selected = selection_area.selectbox(label, filtered, format_func=run_label, key=key,
                            help='Algorithm, experiment and training seed. A seed identifies an independent training run.')
    return selected


def playback_status(index, playing, shared=False, training=False):
    label = 'Shared action index' if shared else 'Action index'
    st.markdown(f'<div class="lab-session"><span><b>{label}: {index}</b> · '
                f'{"Playing" if playing else "Paused"}</span>'
                f'<span>{"Recorded training actions · no live learning" if training else "Saved model · no learning during playback"}</span></div>', unsafe_allow_html=True)


def playback_toolbar(callback, prefix='', speed_key='speed', compact=False):
    """One stable native toolbar shared by all simulator layouts."""
    controls = st.columns([1, 1, 1.2, 1, 1 if compact else 2], gap='small', vertical_alignment='center')
    for col, label, command, icon in zip(controls, ['Play', 'Pause', 'Next step', 'Reset'],
            ['play', 'pause', 'next', 'reset'], ['play_arrow', 'pause', 'skip_next', 'restart_alt']):
        col.button(label, on_click=callback, args=(command,), key=f'{prefix}{command}',
                   icon=None if compact else f':material/{icon}:',
                   type='primary' if command == 'play' else 'secondary', width='stretch')
    if compact:
        with controls[4].popover('Speed', width='stretch'):
            return st.select_slider('Playback speed (steps / second)', options=[.5, 1., 2., 4.], value=2., key=speed_key)
    return controls[4].select_slider('Playback speed (steps / second)', options=[.5, 1., 2., 4.], value=2., key=speed_key)


def context_strip(episode):
    scenario=episode['scenario']
    values=(('SCENARIO',scenario['id']),('START',str(scenario['start'])),('GOAL',str(scenario['goal'])),
            ('SHORTEST ROUTE',f"{scenario['shortest_length']} actions"),('MODE',episode['mode']))
    st.markdown('<div class="lab-context">'+''.join(
        f'<div><small>{escape(label)}</small>{escape(value)}</div>' for label,value in values)+'</div>',unsafe_allow_html=True)


def panel_heading(directory, subtitle='', policy_id=None):
    label=run_label(directory).replace(' · Selected system','')
    reference = policy_id and policy_id.startswith(('bfs','random'))
    if reference:
        label='BFS · Known-map planner' if policy_id.startswith('bfs') else 'Random · Untrained reference'
    st.markdown(f'<div class="lab-panel"><strong>{escape(label)}</strong>'
                f'<div class="lab-subtle">{escape(subtitle)}</div></div>',
                unsafe_allow_html=True)


def status_badge(status,playing=False,held=False):
    label={'ready':'Ready to start','running':'In progress','success':'Goal reached',
           'collision':'Collision','timeout':'Step limit reached'}.get(status,status)
    suffix='Terminal frame held' if held else 'Playback playing' if playing else 'Playback paused'
    st.markdown(f'<span class="lab-status lab-{escape(status)}">{escape(label)}</span>'
                f'<span class="lab-subtle">{suffix}</span>',unsafe_allow_html=True)


def repeated_pattern(episode,index):
    """Recognize two observed repetitions, without inspecting any future action."""
    if not episode['policy_id'].startswith(('q-learning/','sarsa/','dqn/','ppo/','voting/')):
        return None
    transitions=episode['transitions'][:index]
    moves=[(tuple(t['observation']),t['action'],tuple(t['next_observation'])) for t in transitions]
    for period in range(1,min(12,len(moves)//2)+1):
        if moves[-period:]==moves[-2*period:-period]:
            return period
    return None


def telemetry(episode,index,max_steps,playing=False,shared_index=None,paired=True,compact=False,show_metrics=True):
    state=frame(episode,index)
    held=shared_index is not None and shared_index>=len(episode['transitions'])
    status_badge(state['status'],playing and state['status'] in ('ready','running'),held)
    if compact:
        st.caption(f"Steps {state['index']} / {max_steps} · reward {state['return']:.2f} · position {state['position']}")
    else:
        if show_metrics:
            a,b=st.columns(2) if paired else (st,st)
            a.metric('Actions',f"{state['index']} / {max_steps}")
            b.metric('Cumulative reward',f"{state['return']:.2f}")
        rows=(('Robot position',str(state['position'])),('Last action',state['action']),
              ('Step reward',f"{state['step_reward']:+.2f}"),('Revisited cells',str(state['revisited_cells'])))
        st.markdown(''.join(f'<div class="lab-kv"><span>{escape(k)}</span><b>{escape(v)}</b></div>' for k,v in rows),unsafe_allow_html=True)
    if state['status']=='collision':
        st.error(f"Collision attempt at {state['attempted_position']}. Robot remains at {state['position']}.")
    period=repeated_pattern(episode,state['index'])
    if period:
        message = ('These were the actual actions recorded during training.' if episode['mode']=='training'
                   else 'The trained model keeps making its own decisions; playback does not help it escape.')
        st.warning(f'Repeating {period}-action pattern. {message}')
    if state['index']==len(episode['transitions']) and 'metrics' in episode:
        st.caption(f"Completed SPL {episode['metrics']['spl']:.3f} · {episode['metrics']['outcome']}")
    elif state['index']==len(episode['transitions']) and episode.get('complete') is False:
        st.info('Recording ends here. This training episode was cut short; no completed outcome was recorded.')


def replay_statistics(episode, index):
    """Displayed-prefix readings in discrete grid units, with no future outcome."""
    state = frame(episode, index)
    traveled = sum(abs(a[0]-b[0]) + abs(a[1]-b[1])
                   for a, b in zip(state['trail'], state['trail'][1:]))
    goal = episode['scenario']['goal']
    distance = sum(abs(a-b) for a, b in zip(state['position'], goal))
    return {'actions':state['index'], 'return':state['return'],
            'traveled':traveled, 'goal_distance':distance}


def reward_history(episode, index):
    """Include reset and observed rewards only, even for incomplete recordings."""
    state = frame(episode, index)
    rewards = [0.0]
    for transition in episode['transitions'][:state['index']]:
        rewards.append(rewards[-1] + transition['reward'])
    return {'Step':list(range(len(rewards))), 'Reward':rewards}


def card_heading(title, subtitle, chip=None):
    badge = f'<span class="lab-chip">{escape(chip)}</span>' if chip else ''
    st.markdown(f'<div class="lab-card-heading"><div><strong>{escape(title)}</strong>'
                f'<small>{escape(subtitle)}</small></div>{badge}</div>', unsafe_allow_html=True)


def reward_chart(episode, index):
    history = reward_history(episode, index)
    spec = {
        'height':170, 'background':'transparent',
        'layer':[
            {'mark':{'type':'area', 'color':'#00bdf2', 'opacity':.12}},
            {'mark':{'type':'line', 'color':'#00bdf2', 'strokeWidth':2,
                     'point':{'filled':True, 'size':28}}}],
        'encoding':{
            'x':{'field':'Step', 'type':'quantitative',
                 'scale':{'domain':[0,max(1,history['Step'][-1])]},
                 'axis':{'title':'Recorded step', 'format':'d', 'tickMinStep':1, 'grid':False}},
            'y':{'field':'Reward', 'type':'quantitative',
                 'scale':{'domain':[min(-.1,min(history['Reward'])),max(.1,max(history['Reward']))]},
                 'axis':{'title':None, 'format':'.2f', 'tickCount':4}}},
        'config':{'view':{'stroke':None}, 'axis':{'labelColor':'#88a9bf', 'titleColor':'#88a9bf',
                   'domain':False, 'tickColor':'#1c394a', 'gridColor':'#1c394a'}}}
    st.vega_lite_chart(history, spec, width='stretch')


@st.cache_data(show_spinner=False,max_entries=160)
def map_png(grid,episode,index,trail=True,visits=False,bfs=False,landscape=False):
    """Cache actual replay-prefix rendering, independent of timer refreshes."""
    with _MAP_RENDER_LOCK:
        figure=draw(grid,episode,index,trail=trail,visits=visits,bfs=bfs)
        try:
            # Fixed geometry: never let text, legends or a second layout pass resize
            # one panel's grid independently of the other.
            figure.set_layout_engine(None)
            width, height = (7.6,5.3) if landscape else (6.4,6.0)
            figure.set_size_inches(width,height,forward=False)
            figure.set_facecolor('#091923')
            ax=figure.axes[0];ax.set_facecolor('#091923')
            ax.set_position([.08,.10,.89,.87])
            if landscape:
                # This is a coordinate schematic. Both axes retain grid-cell
                # units while the live workspace uses a wider viewing panel.
                ax.set_aspect('auto')
            ax.images[0].set_cmap(ListedColormap(['#0b1d29','#243e50']))
            ax.tick_params(colors='#88a9bf',labelsize=8)
            ax.grid(which='minor',color='#355368',linewidth=.65)
            for spine in ax.spines.values():
                spine.set_color('#3d556c')
            ax.xaxis.label.set_color('#9db6cc');ax.yaxis.label.set_color('#9db6cc')
            ax.set_title('',loc='left')
            for patch,color in zip(ax.patches,('#8ba6b8','#00dfb5')):
                patch.set_edgecolor(color)
            # The first collection is the real robot position from draw().
            # Keep collision markers and attempted-move arrows red.
            ax.collections[0].set_facecolor('#00bdf2')
            ax.collections[0].set_edgecolor('#a3eeff')
            ax.collections[0].set_linewidth(1.8)
            for text in ax.texts:
                label=text.get_text()
                if label in ('S','G'):
                    text.set_color('#8ba6b8' if label=='S' else '#00dfb5')
                elif label=='R':
                    text.set_color('#052333')
                elif label.isdigit():
                    text.set_color('#8de5dd')
                    text.set_bbox({'facecolor':'#102132','edgecolor':'none','pad':.5})
            for line in ax.lines:
                line.set_color('#ba9dff' if line.get_linestyle()=='--' else '#86ddff')
                line.set_alpha(.95)
            for text in figure.texts:
                text.set_color('#a8bdce')
            legend=ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor('#142637');legend.get_frame().set_edgecolor('#496078')
                for text in legend.get_texts():
                    text.set_color('#c1d2e1')
            buffer=BytesIO();figure.savefig(buffer,format='png',dpi=145,facecolor=figure.get_facecolor())
            return buffer.getvalue()
        finally:
            plt.close(figure)


def show_map(grid,episode,index,**options):
    st.image(map_png(grid,episode,index,**options),width='stretch',
             alt=f"{grid.map_id}; robot at {frame(episode,index)['position']}; action {frame(episode,index)['index']}")
