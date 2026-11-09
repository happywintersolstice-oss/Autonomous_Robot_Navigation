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
    st.markdown('''<style>
    .stApp {background:#0a111a;color:#e6eef7;}
    [data-testid="stHeader"] {background:#0a111a;height:2.5rem;}
    [data-testid="stMainBlockContainer"] {max-width:1680px;padding:3.5rem 2rem 2rem;}
    [data-testid="stSidebar"] {background:#0e1925;border-right:1px solid #243347;}
    [data-testid="stSidebar"] h2 {font-size:1rem;}
    h1 {font-size:1.9rem !important;letter-spacing:-.04em;padding-top:0 !important;}
    h2 {font-size:1.25rem !important;letter-spacing:-.02em;}
    h3 {font-size:1.05rem !important;}
    [data-testid="stVerticalBlock"] {gap:.7rem;}
    [data-testid="stMetric"] {background:#142232;border:1px solid #293b4d;border-radius:10px;padding:10px 12px;}
    [data-testid="stMetricLabel"] {color:#a3b7ca;font-size:.75rem;}
    [data-testid="stMetricValue"] {color:#ecf5ff;font-size:1.6rem;font-weight:600;}
    [data-testid="stCaptionContainer"] {color:#9ab0c4;}
    .stButton button {border-radius:8px;border-color:#304459;min-height:2.5rem;}
    .stButton button:focus-visible {outline:2px solid #8fe9d0;outline-offset:3px;}
    .stButton button[kind="primary"] {color:#061813;font-weight:650;background:#58d9b5;border-color:#58d9b5;}
    [data-testid="stExpander"] {border-color:#28394b;background:#0f1b28;border-radius:10px;}
    .lab-brand {padding:0 0 12px;border-bottom:1px solid #263647;margin-bottom:10px;}
    .lab-brand strong {font-size:1.1rem;letter-spacing:.1em;color:#e7f3ff;}
    .lab-eyebrow {font:600 .68rem monospace;letter-spacing:.15em;color:#72cfb8;margin-bottom:.2rem;}
    .lab-subtle {color:#9ab0c4;font-size:.77rem;line-height:1.5;}
    .lab-context {display:flex;gap:1.6rem;flex-wrap:wrap;padding:12px 16px;background:#101e2c;border:1px solid #263a4d;border-radius:10px;}
    .lab-context div {font-size:.84rem;color:#dfebf6;}
    .lab-context small {display:block;font:600 .6rem monospace;color:#8fa9bf;letter-spacing:.1em;margin-bottom:3px;}
    .st-key-single_playback_controls, .st-key-comparison_playback_controls {
      padding:14px 16px 8px;background:#101e2c;border:1px solid #263a4d;border-radius:12px;
    }
    .st-key-workspace_switch {padding:8px 0 4px;}
    .st-key-workspace_switch [role="radiogroup"] {gap:.6rem;}
    .st-key-workspace_switch [data-testid="stRadio"] label[data-baseweb="radio"] {
      padding:9px 16px;background:#111e2c;border:1px solid #304459;border-radius:8px;margin:0;
    }
    .lab-panel {min-height:78px;border-top:2px solid #58d9b5;padding:10px 2px 6px;}
    .lab-panel strong {display:block;line-height:1.4;font-size:1rem;color:#e6f2fd;}
    .lab-panel code {font-size:.69rem;color:#97afc6;background:none;padding:0;overflow-wrap:anywhere;}
    .lab-status {display:inline-block;border:1px solid #385069;border-radius:5px;padding:3px 8px;margin-right:8px;font:600 .68rem monospace;letter-spacing:.06em;color:#b9ccdc;}
    .lab-success {color:#72e2b6;border-color:#2a725b;background:#102e27;}
    .lab-collision {color:#ff9e9e;border-color:#8a454a;background:#351c25;}
    .lab-timeout {color:#f4c873;border-color:#806638;background:#302819;}
    .lab-kv {display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px solid #233448;font-size:.84rem;}
    .lab-kv:last-child {border-bottom:0;}
    .lab-kv span {color:#9cb1c6;}.lab-kv b {color:#e4effa;font-weight:500;}
    .lab-session {display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
      padding:8px 0;color:#a9bfd2;font-size:.8rem;}
    .lab-session b {color:#e5f1fc;font-weight:600;}
    .st-key-single_map, .st-key-pair_maps, .st-key-four_maps {background:#0f1b28;border:1px solid #263a4d;border-radius:12px;padding:12px;}
    .st-key-single_map [data-testid="stImage"] img {max-height:590px;object-fit:contain;}
    .st-key-pair_maps [data-testid="stImage"] img {max-height:480px;object-fit:contain;}
    .st-key-four_maps [data-testid="stImage"] img {max-height:480px;object-fit:contain;}
    .st-key-single_inspector {background:#101e2c;border:1px solid #263a4d;border-radius:12px;padding:16px;}
    .st-key-comparison_summary {padding-top:4px;}
    .st-key-saved_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
    .st-key-learning_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
      flex:0 0 100px !important;min-width:100px !important;width:100px !important;
    }
    .st-key-saved_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
    .st-key-learning_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
      flex:1 1 0% !important;min-width:0 !important;width:auto !important;
    }
    .st-key-single_decision_details,
    .st-key-comparison_decision_details_0,
    .st-key-comparison_decision_details_1 {padding-top:12px;}
    @media(max-width:1100px) {
      .st-key-single_playback_controls [data-testid="stHorizontalBlock"],
      .st-key-comparison_playback_controls [data-testid="stHorizontalBlock"] {
        display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem;
      }
      .st-key-single_playback_controls [data-testid="stColumn"],
      .st-key-comparison_playback_controls [data-testid="stColumn"] {
        width:100% !important;min-width:0 !important;
      }
      .st-key-single_playback_controls [data-testid="stColumn"]:last-child,
      .st-key-comparison_playback_controls [data-testid="stColumn"]:last-child {grid-column:1 / -1;}
      .st-key-single_replay_content [data-testid="stHorizontalBlock"]:has(.st-key-single_map),
      .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-saved_run_picker),
      .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-learning_run_picker),
      .st-key-pair_maps [data-testid="stHorizontalBlock"]:has(.lab-panel),
      .st-key-four_maps [data-testid="stHorizontalBlock"]:has(.lab-panel) {flex-direction:column;}
      .st-key-single_replay_content [data-testid="stHorizontalBlock"]:has(.st-key-single_map) > [data-testid="stColumn"],
      .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-saved_run_picker) > [data-testid="stColumn"],
      .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-learning_run_picker) > [data-testid="stColumn"],
      .st-key-pair_maps [data-testid="stHorizontalBlock"]:has(.lab-panel) > [data-testid="stColumn"],
      .st-key-four_maps [data-testid="stHorizontalBlock"]:has(.lab-panel) > [data-testid="stColumn"] {width:100% !important;flex:1 1 100% !important;}
    }
    @media(max-width:800px) {
      [data-testid="stMainBlockContainer"] {padding:3rem 1rem 1rem;}
      .lab-context {gap:.75rem;} h1 {font-size:1.55rem !important;}
      .lab-panel {min-height:0;}
    }
    </style>''', unsafe_allow_html=True)


def run_label(directory):
    name=directory.name
    neural = re.fullmatch(r'm9-(pilot|search|control|tuned)-(dqn|ppo)(?:-(c\d\d))?-seed(\d+)',name)
    if neural:
        stage={'pilot':'Tuning pilot','search':'Tuning search','control':'Fresh baseline','tuned':'Selected settings'}[neural[1]]
        candidate=f' {neural[3]}' if neural[3] else ''
        return f'{ALGORITHM_LABELS[neural[2]]} · {stage}{candidate} · seed {neural[4]}'
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


def playback_toolbar(callback, prefix='', speed_key='speed'):
    """One stable native toolbar shared by all simulator layouts."""
    controls = st.columns([1, 1, 1.2, 1, 2], vertical_alignment='center')
    for col, label, command, icon in zip(controls, ['Play', 'Pause', 'Next step', 'Reset'],
            ['play', 'pause', 'next', 'reset'], ['play_arrow', 'pause', 'skip_next', 'restart_alt']):
        col.button(label, on_click=callback, args=(command,), key=f'{prefix}{command}',
                   icon=f':material/{icon}:', type='primary' if command == 'play' else 'secondary', width='stretch')
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


def telemetry(episode,index,max_steps,playing=False,shared_index=None,paired=True,compact=False):
    state=frame(episode,index)
    held=shared_index is not None and shared_index>=len(episode['transitions'])
    status_badge(state['status'],playing and state['status'] in ('ready','running'),held)
    if compact:
        st.caption(f"Steps {state['index']} / {max_steps} · reward {state['return']:.2f} · position {state['position']}")
    else:
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


@st.cache_data(show_spinner=False,max_entries=160)
def map_png(grid,episode,index,trail=True,visits=False,bfs=False):
    """Cache actual replay-prefix rendering, independent of timer refreshes."""
    with _MAP_RENDER_LOCK:
        figure=draw(grid,episode,index,trail=trail,visits=visits,bfs=bfs)
        try:
            # Fixed geometry: never let text, legends or a second layout pass resize
            # one panel's grid independently of the other.
            figure.set_layout_engine(None)
            figure.set_size_inches(6.4,6.0,forward=False)
            figure.set_facecolor('#0f1b28')
            ax=figure.axes[0];ax.set_facecolor('#0f1b28')
            ax.set_position([.10,.12,.84,.82])
            ax.images[0].set_cmap(ListedColormap(['#152637','#3c5268']))
            ax.tick_params(colors='#9db6cc',labelsize=9)
            ax.grid(which='minor',color='#32495e',linewidth=.6)
            for spine in ax.spines.values():
                spine.set_color('#3d556c')
            ax.xaxis.label.set_color('#9db6cc');ax.yaxis.label.set_color('#9db6cc')
            ax.set_title(ax.get_title(loc='left'),loc='left',color='#e2edf7',fontsize=10)
            for patch,color in zip(ax.patches,('#78c5ff','#6ee2b7')):
                patch.set_edgecolor(color)
            for text in ax.texts:
                label=text.get_text()
                if label in ('S','G'):
                    text.set_color('#78c5ff' if label=='S' else '#6ee2b7')
                elif label=='R':
                    text.set_color('#231700')
                elif label.isdigit():
                    text.set_color('#8de5dd')
                    text.set_bbox({'facecolor':'#102132','edgecolor':'none','pad':.5})
            for line in ax.lines:
                line.set_color('#ba9dff' if line.get_linestyle()=='--' else '#58d9ca')
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
