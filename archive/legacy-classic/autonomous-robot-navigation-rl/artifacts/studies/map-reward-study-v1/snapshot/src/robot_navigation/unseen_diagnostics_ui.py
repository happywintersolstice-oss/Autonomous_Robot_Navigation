"""Saved diagnostics and event links into the existing recorded-action viewer."""
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from .core import read_json
from .interface import ALGORITHM_LABELS
from .release import sha
from .unseen_diagnostics import STUDY, COMPOSITION, COLORS, saved
from .unseen_final import STUDY as FINAL, FinalBenchmark
from .unseen_main import STUDY as MAIN
from .unseen_benchmark import MapBenchmark


def render_diagnostics(root,source_study,open_example):
    if source_study not in (MAIN,FINAL):
        st.info('Failure analysis covers the shared-feature main study and final evaluation.')
        return
    directory = root/'artifacts/studies'/STUDY
    try:
        summary = saved(root)
        verification = read_json(directory/'verification.json')
        if (summary['status']!='verified' or verification['status']!='passed'
                or verification['summary_sha256']!=sha(directory/'summary.json')):
            raise ValueError('Diagnostic verification is incomplete')
        rows = read_json(directory/'episodes.json')
    except (OSError,ValueError,KeyError) as exc:
        st.info(f'Saved failure analysis is unavailable. {exc}')
        return
    dataset = 'final' if source_study==FINAL else 'development'
    groups = [g for g in summary['groups'] if g['dataset']==dataset]
    st.caption('Two-cell cycles alternate between two positions until the action limit. '
               'Cycle labels describe saved behavior. These post hoc patterns guide further investigation.')
    chart_rows = [{'Algorithm':ALGORITHM_LABELS[g['algorithm']],'Pattern':name,
        'Order':index,
        'Cases':sum(g['counts'][k] for k in members),
        'Share':100*sum(g['counts'][k] for k in members)/g['episodes']}
        for g in groups for index,(name,members) in enumerate(COMPOSITION.items())]
    st.altair_chart(alt.Chart(pd.DataFrame(chart_rows)).mark_bar().encode(
        x=alt.X('Algorithm:N',sort=list(ALGORITHM_LABELS.values()),title=None,axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Share:Q',title='Cases (%)',scale=alt.Scale(domain=[0,100])),
        order=alt.Order('Order:Q',sort='ascending'),
        color=alt.Color('Pattern:N',scale=alt.Scale(domain=list(COMPOSITION),range=list(COLORS)),
                        legend=alt.Legend(orient='bottom',columns=3)),
        tooltip=['Algorithm:N','Pattern:N','Cases:Q',alt.Tooltip('Share:Q',format='.2f')]
    ).properties(height=340),width='stretch',height=340)
    st.dataframe(pd.DataFrame([{'Algorithm':ALGORITHM_LABELS[g['algorithm']],'Cases':g['episodes'],
        'Two-cell timeouts':g['counts']['two-cell timeout'],
        'Longer-cycle timeouts':g['counts']['longer-cycle timeout'],
        'Collisions':g['counts']['wall collision']+g['counts']['boundary collision'],
        'Other timeouts':g['counts']['stationary timeout']+g['counts']['non-periodic timeout'],
        'Mean cycle entry action':g['mean_cycle_entry_action'],
        'Actions after first return':g['actions_after_first_return']}
        for g in groups]),hide_index=True,width='stretch')
    with st.expander('Route difficulty and seed variation'):
        st.dataframe(pd.DataFrame([{'Algorithm':ALGORITHM_LABELS[g['algorithm']],'Route band':g['distance_band'],
            'Cases':g['episodes'],'Success (%)':100*g['success_mean'],
            'Success SD (points)':100*g['success_sd'] if g['success_sd'] is not None else None,
            'Two-cell timeouts':g['counts']['two-cell timeout'],
            'Longer-cycle timeouts':g['counts']['longer-cycle timeout'],
            'Collisions':g['counts']['wall collision']+g['counts']['boundary collision']}
            for g in summary['difficulty'] if g['dataset']==dataset]),hide_index=True,width='stretch')
        st.caption('Means and sample SD are across three training seeds. '
                   'Routes within a layout and policy are dependent observations.')
    algorithm = st.selectbox('Diagnostic algorithm',[g['algorithm'] for g in groups],
                            format_func=lambda a:ALGORITHM_LABELS[a],key='diagnostic_algorithm')
    selected = [r for r in rows if r['dataset']==dataset and r['algorithm']==algorithm]
    map_id = st.selectbox('Failure map',sorted({r['layout_id'] for r in selected}),key=f'diagnostic_map:{dataset}')
    map_rows = [r for r in selected if r['layout_id']==map_id]
    counts = Counter(tuple(r['failure_cell']) for r in map_rows if r['failure_cell'] is not None)
    benchmark = FinalBenchmark(root/'artifacts/studies'/FINAL) if dataset=='final' else MapBenchmark(root)
    layout = next(m for m in benchmark.layouts('test' if dataset=='final' else 'validation') if m['map_id']==map_id)
    cells = pd.DataFrame([{'Row':r,'Column':c,'Failures':counts[(r,c)],'Wall':cell=='#'}
                          for r,line in enumerate(layout['rows']) for c,cell in enumerate(line)])
    encoding = dict(x=alt.X('Column:O',sort=list(range(10))),y=alt.Y('Row:O',sort=list(range(10))),
                    tooltip=['Row:O','Column:O','Failures:Q','Wall:N'])
    base = alt.Chart(cells).mark_rect(stroke='#385467',strokeWidth=.5).encode(**encoding,
        color=alt.condition('datum.Wall',alt.value('#405b6d'),alt.value('#0b2230')))
    heat = alt.Chart(cells).transform_filter('datum.Failures > 0').mark_rect(stroke='#7b9bb0',strokeWidth=.5).encode(
        **encoding,color=alt.Color('Failures:Q',scale=alt.Scale(scheme='orangered'),legend=alt.Legend(title='Failed cases')))
    st.altair_chart((base+heat).resolve_scale(color='independent').properties(height=380),width='stretch')
    st.caption(f'{len(map_rows)} policy–route cases on this map. Each failed case contributes one location. '
               'Cycles use the entry cell, collisions the last valid robot cell, and other timeouts the final cell.')
    examples = [e for e in summary['examples'] if e['dataset']==dataset and e['algorithm']==algorithm]
    if examples:
        example_key = f'diagnostic_example:{dataset}:{algorithm}'
        example = st.selectbox('Illustrative recording',examples,
            format_func=lambda e:f"Seed {e['training_seed']} · {e['category']} · {e['layout_id']}",
            key=example_key)
        if example['period']:
            st.caption(f"This recording enters a {example['period']}-action cycle after action {example['cycle_start']}. "
                       f"Its first return is at action {example['first_return']}.")
        else:
            st.caption(f"The recorded collision occurs at action {example['event_action']}.")
        def open_selection():
            # A selection and click can arrive before the intervening rerun.
            # Resolve the submitted widget state instead of a captured example.
            current_study = st.session_state.get('unseen_study')
            if current_study not in (MAIN,FINAL):
                return
            current_dataset = 'final' if current_study==FINAL else 'development'
            current_algorithm = st.session_state.get('diagnostic_algorithm')
            key = f'diagnostic_example:{current_dataset}:{current_algorithm}'
            current = st.session_state.get(key)
            if current is not None and (current['dataset'],current['algorithm'])==(current_dataset,current_algorithm):
                open_example(current)

        st.caption('Illustrative recordings name their own source map.')
        st.button('Replay this event',on_click=open_selection,key='diagnostic_open_event',type='primary')
    st.caption('Examples are bounded saved cases. The action count after a cycle return describes observed repetition. '
               'Performance changes require a new training study and fresh independent evaluation.')
