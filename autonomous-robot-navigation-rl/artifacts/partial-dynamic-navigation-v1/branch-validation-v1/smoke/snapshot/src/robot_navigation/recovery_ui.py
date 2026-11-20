"""Verified development recovery results and paired recorded-action playback."""
import gzip
import json
import time

import pandas as pd
import streamlit as st

from .core import read_json
from .dashboard import seek_from_widget
from .episodes import ComparisonPlayback
from .evidence import EXPECTED_ERRORS,check_pinned_summary
from .interface import playback_toolbar,playback_status,show_map,telemetry
from .loop_intervention import STUDY
from .release import sha
from .unseen_benchmark import MapBenchmark


@st.cache_data
def recordings(path,checksum):
    if sha(path)!=checksum:
        raise ValueError(f'Recovery recording checksum mismatch: {path}')
    with gzip.open(path,'rt',encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def render_recovery(root):
    st.subheader('History-aware SARSA recovery: development only')
    study = root/'artifacts/studies'/STUDY
    try:
        if read_json(study/'status.json')['status']!='verified' or read_json(study/'verification.json')['status']!='passed':
            raise ValueError('Recovery study is unverified')
        check_pinned_summary(root,STUDY)
        summary = read_json(study/'summary.json')
        hashes = read_json(study/'outputs.json')
        if sha(study/'summary.json')!=hashes['summary.json']:
            raise ValueError('Recovery summary checksum mismatch')
    except EXPECTED_ERRORS as exc:
        st.info(f'{STUDY}: unavailable or unverified ({exc}). Restore the registered development study.')
        return
    st.caption('Four development layouts, 192 routes per controller. Three pretrained SARSA seeds, each with three recovery RNG streams; the deterministic baseline is reused. No new training or final-layout evaluation. History changes the controller information contract.')
    st.dataframe(pd.DataFrame([{'Training seed':r['training_seed'],'Recovery RNG':r['rng_seed'],
        'Base success':r['baseline']['success_rate'],'Recovery success':r['wrapper']['success_rate'],
        'Base SPL':r['baseline']['spl'],'Recovery SPL':r['wrapper']['spl'],
        'Recovery collision':r['wrapper']['collision_rate'],
        'Timeout to success':r['paired_changes'].get('timeout->success',0),
        'Timeout to collision':r['paired_changes'].get('timeout->collision',0)} for r in summary['rows']]),hide_index=True,width='stretch')
    st.write(f"Mean success {summary['baseline_success']:.2%} to {summary['wrapper_success']:.2%}; "
             f"SPL {summary['baseline_spl']:.4f} to {summary['wrapper_spl']:.4f}. "
             f"Promotion rule {'passed' if summary['promotion_passed'] else 'missed'}; original controller selection is unchanged.")
    with st.expander('Matched recovery recording'):
        try:
            row = st.selectbox('Recovery run',summary['rows'],
                format_func=lambda r:f"Training seed {r['training_seed']} / recovery RNG {r['rng_seed']}",key='recovery_run')
            left = f"baseline-{row['training_seed']}.jsonl.gz"
            right = f"wrapper-{row['training_seed']}-{row['rng_seed']}.jsonl.gz"
            baseline = recordings(study/left,hashes[left])
            wrapper = recordings(study/right,hashes[right])
            indices = list(range(len(wrapper)))
            # First show a timeout changed to success, when one exists; keep every case accessible.
            indices.sort(key=lambda i:not (baseline[i]['metrics']['outcome']=='timeout' and wrapper[i]['metrics']['outcome']=='success'))
            index = st.selectbox('Paired development route',indices,
                format_func=lambda i:f"{wrapper[i]['scenario']['id']} / {baseline[i]['metrics']['outcome']} to {wrapper[i]['metrics']['outcome']}",key=f'recovery_case:{left}:{right}')
            identity = (left,right,index)
            if st.session_state.get('recovery_identity')!=identity:
                previous = st.session_state.get('recovery_playback')
                if previous is not None:
                    previous.pause()
                st.session_state['recovery_identity'] = identity
                st.session_state['recovery_playback'] = ComparisonPlayback([baseline[index],wrapper[index]])
                st.session_state['recovery_tick'] = time.monotonic()
            benchmark = MapBenchmark(root)
            scenario = wrapper[index]['scenario']
            layout = next(m for m in benchmark.layouts('validation') if scenario in m['scenarios'])
            grid = benchmark.task(layout).grid
        except EXPECTED_ERRORS as exc:
            st.info(f'{STUDY}: paired recording unavailable ({exc}). Restore its recorded route files.')
            return
        def command(name):
            player = st.session_state['recovery_playback']
            getattr(player,name)()
            st.session_state['recovery_tick'] = time.monotonic()
        speed = playback_toolbar(command,prefix='recovery_',speed_key='recovery_speed')
        def seek(key):
            seek_from_widget(st.session_state,key,player_key='recovery_playback',
                             position_key='recovery_seek',active_key='recovery_timeline')
        player = st.session_state['recovery_playback']
        @st.fragment(run_every=.2 if player.playing else None)
        def panels():
            if st.session_state.get('dashboard_page')!='Optimization and robustness' or st.session_state.get('recovery_identity')!=identity:
                return
            player = st.session_state['recovery_playback']
            now = time.monotonic()
            if player.playing and now-st.session_state['recovery_tick']>=1/speed:
                player.tick()
                st.session_state['recovery_tick'] = now
                if not player.playing:
                    st.rerun()
            playback_status(player.index,player.playing,shared=True)
            st.caption(f'Development validation / {layout["map_id"]} / training seed {row["training_seed"]} / recovery RNG {row["rng_seed"]} / shared action {player.index}')
            for col,episode,label in zip(st.columns(2),player.episodes,('Base SARSA','History-aware recovery')):
                with col:
                    st.write(label)
                    show_map(grid,episode,player.index)
                    telemetry(episode,player.index,200,player.playing,shared_index=player.index)
            key = f'recovery-timeline:{identity}:{player.index}'
            st.session_state['recovery_timeline'] = key
            st.slider('Recovery replay position',0,player.length,value=player.index,key=key,on_change=seek,args=(key,))
        panels()
