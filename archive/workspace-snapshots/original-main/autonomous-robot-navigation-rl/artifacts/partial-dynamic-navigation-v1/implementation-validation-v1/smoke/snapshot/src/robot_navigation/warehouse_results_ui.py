"""Source-backed results for the registered warehouse comparison."""
import pandas as pd
import streamlit as st

from .core import read_json
from .evidence import EXPECTED_ERRORS
from .warehouse_study import FAMILIES,location,saved,checked_run


def study_progress(root):
    folder = location(root)
    completed = 0
    failures = []
    for path in (folder/'runs').glob('*/status.json'):
        status = read_json(path)
        completed += status.get('status')=='verified'
        if status.get('status')=='failed':
            failures.append(f"{path.parent.name}: {status.get('reason','unknown failure')}")
    st.info('The registered warehouse comparison is being prepared. Saved results appear after all twelve runs pass their recording audits.')
    st.progress(completed/12,text=f'{completed} / 12 verified training runs')
    for failure in failures:
        st.error(failure)


def render_results(root):
    st.title('Four-algorithm warehouse comparison')
    try:
        result = saved(root)
    except EXPECTED_ERRORS as exc:
        if (location(root)/'published.json').exists():
            st.error(f'Warehouse comparison could not be verified: {exc}')
            return
        study_progress(root)
        return
    st.write('Q-learning, SARSA, DQN and PPO navigate the same warehouse with the same reward rules and training routes.')
    a,b,c = st.columns(3)
    a.metric('Algorithms compared','4')
    b.metric('Independent seeds per algorithm','3')
    c.metric('Training moves per run','600,000')
    st.caption('Each final model is assessed on the same 188 development routes. Values below are means and sample SD across three training seeds. Higher completion and route efficiency are better; lower collision and timeout rates are better.')
    rows = []
    for algorithm in FAMILIES:
        system = result['systems'][algorithm]
        metrics = system['metrics']
        rows.append({'Algorithm':system['label'],
            'Completion (%)':100*metrics['success_rate']['mean'],
            'Completion SD (pp)':100*metrics['success_rate']['sample_sd'],
            'Collision (%)':100*metrics['collision_rate']['mean'],
            'Timeout (%)':100*metrics['timeout_rate']['mean'],
            'Route efficiency (SPL)':metrics['spl']['mean'],
            'SPL SD':metrics['spl']['sample_sd'],
            'Looping timeouts (all seeds)':system['looping_timeouts']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.bar_chart(pd.DataFrame(rows).set_index('Algorithm')[['Completion (%)','Collision (%)','Timeout (%)']])
    st.info('This is a controlled development comparison on one familiar warehouse. These routes were already inspected in earlier work; they are not a fresh final holdout or evidence of physical-robot safety.')
    with st.expander('Learning curves across the three seeds'):
        values = []
        for algorithm,system in result['systems'].items():
            for seed in system['seeds']:
                _,folder = checked_run(algorithm,seed,root)
                values.extend({'step':r['training_interactions'],'Algorithm':system['label'],
                               'completion':r['success_rate']} for r in read_json(folder/'monitoring.json'))
        frame = pd.DataFrame(values).groupby(['step','Algorithm'])['completion'].mean().unstack()
        st.line_chart(frame)
        st.caption('The same 35-route monitoring set was used for all models. Curves are descriptive; only fixed final-budget checkpoints enter the comparison.')
    with st.expander('Per-seed results and recorded costs'):
        values = []
        for algorithm,system in result['systems'].items():
            for row in system['seed_results']:
                values.append({'Algorithm':system['label'],'Seed':row['seed'],
                    'Completion (%)':100*row['assessment']['success_rate'],
                    'Collision (%)':100*row['assessment']['collision_rate'],
                    'Timeout (%)':100*row['assessment']['timeout_rate'],'SPL':row['assessment']['spl'],
                    'Recorded training seconds':row['training']['training_seconds'],
                    'Decision microseconds':row['assessment']['mean_decision_microseconds'],
                    'Checkpoint bytes':row['checkpoint_bytes']})
        st.dataframe(pd.DataFrame(values),hide_index=True,width='stretch')
        st.caption('Equal environment-action budgets do not imply equal compute. At most two CPU workers ran concurrently. Tabular timing excludes setup; neural recorded timing includes setup. Timings are descriptive, not isolated hardware benchmarks.')
    st.download_button('Download warehouse comparison CSV',(location(root)/'comparison.csv').read_bytes(),
                       file_name='warehouse-four-algorithm-comparison.csv',mime='text/csv')
