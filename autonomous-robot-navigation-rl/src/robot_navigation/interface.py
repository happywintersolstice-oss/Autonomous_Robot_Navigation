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
# Matplotlib shares process-wide state across Streamlit session threads.
_MAP_RENDER_LOCK = RLock()


def apply_theme():
    st.markdown('''<style>
    .stApp {background:#0a111a;color:#e6eef7;}
    [data-testid="stHeader"] {background:#0a111a;height:2.5rem;}
    [data-testid="stMainBlockContainer"] {max-width:1560px;padding:3.5rem 2rem 2rem;}
    [data-testid="stSidebar"] {background:#0e1925;border-right:1px solid #243347;}
    [data-testid="stSidebar"] h2 {font-size:1rem;}
    h1 {font-size:1.9rem !important;letter-spacing:-.04em;padding-top:0 !important;}
    h2 {font-size:1.25rem !important;letter-spacing:-.02em;}
    h3 {font-size:1.05rem !important;}
    [data-testid="stVerticalBlock"] {gap:.5rem;}
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
    .lab-context {display:flex;gap:1.6rem;flex-wrap:wrap;padding:10px 14px;margin-bottom:8px;background:#101e2c;border:1px solid #263a4d;border-radius:10px;}
    .lab-context div {font-size:.84rem;color:#dfebf6;}
    .lab-context small {display:block;font:600 .6rem monospace;color:#8fa9bf;letter-spacing:.1em;margin-bottom:3px;}
    .st-key-single_playback_controls, .st-key-comparison_playback_controls {
      padding:18px 0 12px;
    }
    .lab-panel {height:106px;border-top:2px solid #58d9b5;padding:8px 2px 5px;}
    .lab-panel strong {display:block;min-height:2.7em;line-height:1.35;font-size:1rem;color:#e6f2fd;}
    .lab-panel code {font-size:.69rem;color:#97afc6;background:none;padding:0;overflow-wrap:anywhere;}
    .lab-status {display:inline-block;border:1px solid #385069;border-radius:5px;padding:3px 8px;margin-right:8px;font:600 .68rem monospace;letter-spacing:.06em;color:#b9ccdc;}
    .lab-success {color:#72e2b6;border-color:#2a725b;background:#102e27;}
    .lab-collision {color:#ff9e9e;border-color:#8a454a;background:#351c25;}
    .lab-timeout {color:#f4c873;border-color:#806638;background:#302819;}
    .lab-kv {display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px solid #233448;font-size:.84rem;}
    .lab-kv:last-child {border-bottom:0;}
    .lab-kv span {color:#9cb1c6;}.lab-kv b {color:#e4effa;font-weight:500;}
    .st-key-single_decision_details,
    .st-key-comparison_decision_details_0,
    .st-key-comparison_decision_details_1 {padding-top:12px;}
    @media(max-width:800px) {
      [data-testid="stMainBlockContainer"] {padding:3rem 1rem 1rem;}
      .lab-context {gap:.75rem;} h1 {font-size:1.55rem !important;}
    }
    </style>''', unsafe_allow_html=True)


def run_label(directory):
    name=directory.name
    if name=='m4-voting-q-learning':
        return 'Q-learning · Three-policy vote'
    if name=='smoke-seed7':
        return 'Q-learning · Initial pilot · seed 7'
    match=re.match(r'm(\d+)-(?:(control|tuned|terminal|search)-)?(q-learning|sarsa|dqn|ppo)-(.*)',name)
    if not match:
        return name
    milestone,stage,algorithm,suffix=match.groups()
    family={'q-learning':'Q-learning','sarsa':'SARSA','dqn':'DQN','ppo':'PPO'}[algorithm]
    stage={'control':'Baseline','tuned':'Refined','terminal':'Zero step cost','search':'Search'}.get(stage,
           'DQN pilot' if 'pilot' in suffix else 'Part A' if milestone=='3' else 'Early baseline')
    if algorithm=='ppo':
        stage='Pilot' if 'pilot' in suffix else 'Additional baseline'
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


def context_strip(episode):
    scenario=episode['scenario']
    values=(('SCENARIO',scenario['id']),('START',str(scenario['start'])),('GOAL',str(scenario['goal'])),
            ('SHORTEST ROUTE',f"{scenario['shortest_length']} actions"),('MODE',episode['mode']))
    st.markdown('<div class="lab-context">'+''.join(
        f'<div><small>{escape(label)}</small>{escape(value)}</div>' for label,value in values)+'</div>',unsafe_allow_html=True)


def panel_heading(directory, subtitle=''):
    label=run_label(directory).replace(' · Selected system','')
    st.markdown(f'<div class="lab-panel"><strong>{escape(label)}</strong>'
                f'<code>{escape(directory.name)}</code><div class="lab-subtle">{escape(subtitle)}</div></div>',
                unsafe_allow_html=True)


def status_badge(status,playing=False,held=False):
    label=status.upper()
    suffix='Terminal frame held' if held else 'Playing' if playing else 'Paused'
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


def telemetry(episode,index,max_steps,playing=False,shared_index=None,paired=True):
    state=frame(episode,index)
    held=shared_index is not None and shared_index>len(episode['transitions'])
    status_badge(state['status'],playing and not held,held)
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
        st.warning(f'Repeating {period}-action pattern. This frozen policy keeps its decisions until an episode ends.')
    if state['index']==len(episode['transitions']):
        st.caption(f"Completed SPL {episode['metrics']['spl']:.3f} · {episode['metrics']['outcome']}")


@st.cache_data(show_spinner=False,max_entries=160)
def map_png(grid,episode,index,trail=True,visits=False,bfs=False):
    """Cache actual replay-prefix rendering, independent of timer refreshes."""
    with _MAP_RENDER_LOCK:
        figure=draw(grid,episode,index,trail=trail,visits=visits,bfs=bfs)
        try:
            # Fixed geometry: never let text, legends or a second layout pass resize
            # one panel's grid independently of the other.
            figure.set_layout_engine(None)
            figure.set_size_inches(7.2,4.6,forward=False)
            figure.set_facecolor('#0f1b28')
            ax=figure.axes[0];ax.set_facecolor('#0f1b28')
            ax.set_position([.10,.14,.80,.76])
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
