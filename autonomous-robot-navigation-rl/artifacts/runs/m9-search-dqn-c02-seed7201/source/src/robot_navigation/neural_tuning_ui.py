"""Read-only presentation of the separate preliminary neural tuning study."""
import pandas as pd
import streamlit as st
import yaml

from .core import read_json
from .neural_tuning import STUDY, saved_summary


def open_pair(root, system):
    from .simulator_ui import pause_players
    pause_players()
    control=system['conditions']['control']['seed_results'][0]
    selected=system['conditions']['tuned']['seed_results'][0]
    st.session_state.update(dashboard_page='Navigation simulator',replay_tab='Compare agents',
        comparison_size='Two agents',saved_run=root/'artifacts/runs'/selected['run_id'],
        comparison_second=root/'artifacts/runs'/control['run_id'],
        saved_run_experiment='All experiments',saved_run_algorithm='All algorithms',
        comparison_kind='Different models')
    st.session_state.pop('comparison_identity',None)


def render_tuning(task, root):
    st.write('Compare fresh baseline settings with settings chosen in a bounded DQN/PPO search.')
    st.caption('Preliminary study with three matched training seeds per algorithm. Screening uses one separate seed. '
               'Results use development-validation routes on the familiar fixed map.')
    study=root/'artifacts/studies'/STUDY
    try:
        summary=saved_summary(root)
        if summary:
            task.check(summary['signature'])
    except (OSError,ValueError,KeyError) as exc:
        st.error(f'Neural study evidence unavailable. {exc}')
        return
    if summary is None:
        st.info('Verified results will appear after training and integrity checks finish.')
        if (study/'manifest.json').is_file():
            status=read_json(study/'manifest.json')
            st.write(f"Study {status['status']} · {len(status['completed_runs'])} completed runs · {status['phase']}")
            if status.get('error'):
                st.warning(status['error'])
        with (root/'configs/neural_tuning.yaml').open(encoding='utf-8') as file:
            st.json(yaml.safe_load(file))
        return
    st.subheader('Matched confirmation results')
    rows=[]
    for family,system in summary['systems'].items():
        for condition,values in system['conditions'].items():
            metrics=values['metrics']
            label='Fresh baseline' if condition=='control' else 'Selected settings'
            if condition=='tuned' and system['baseline_retained']:
                label+=' (baseline retained)'
            rows.append({'Algorithm':family.upper(),'Settings':label,'Training seeds':system['seed_count'],
                'Success %':100*metrics['success_rate']['mean'],
                'Success seed SD (pp)':100*metrics['success_rate']['sample_sd'],
                'Collision %':100*metrics['collision_rate']['mean'],
                'Timeout %':100*metrics['timeout_rate']['mean'],'SPL':metrics['spl']['mean'],
                'Mean return':metrics['mean_return']['mean']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.caption('Each run trains for 200,000 steps and evaluates 928 validation routes. '
               'SD describes variation across three training seeds. The routes share the same map and training history within each run.')
    for family,system in summary['systems'].items():
        change=system['paired_change']['success_rate']
        st.write(f"**{family.upper()}** · selected {system['selected_candidate']} · mean paired success change "
                 f"{100*change['mean']:+.2f} percentage points · seed SD {100*change['sample_sd']:.2f} pp")
        if system['baseline_retained']:
            st.caption('The original settings won screening. Both rows refer to the same three control runs. '
                       'There is no separate selected-settings replicate.')
        else:
            st.button(f'Compare {family.upper()} models',key=f'neural_compare_{family}',
                help='Opens the first predeclared confirmation seed. Choose any validation route in the simulator.',
                on_click=open_pair,args=(root,system))
    st.image(str(study/'paired_neural_tuning.png'),width='stretch')
    st.image(str(study/'neural_tuning_learning.png'),width='stretch')
    with st.expander('Every confirmation seed and recorded cost'):
        seeds=[]
        for family,system in summary['systems'].items():
            for condition,values in system['conditions'].items():
                for row in values['seed_results']:
                    seeds.append({'Algorithm':family.upper(),'Settings':condition,'Training seed':row['seed'],
                        'Run':row['run_id'],'Selected step':row['selected_interactions'],
                        'Success %':100*row['validation']['success_rate'],**row['costs']})
        st.dataframe(pd.DataFrame(seeds),hide_index=True,width='stretch')
        st.caption('Two concurrent one-thread training processes were used. Timings are descriptive measurements on this machine. '
                   'Training excludes monitoring and export. Validation includes recording/export. Decision latency excludes rendering. '
                   'Checkpoint bytes measure saved model size.')
    with st.expander('Screening candidates and frozen choices'):
        st.dataframe(pd.DataFrame([{'Algorithm':r['algorithm'].upper(),'Candidate':r['candidate'],
            'Development seed':r['seed'],'Success %':100*r['validation']['success_rate'],
            'SPL':r['validation']['spl'],'Collision %':100*r['validation']['collision_rate']}
            for r in sorted(summary['search_results'],key=lambda r:(r['algorithm'],r['candidate']))]),
            hide_index=True,width='stretch')
        st.json(read_json(study/'plan.json')['candidates'])
        st.caption('Screening uses 100,000 steps per candidate. Confirmation uses fresh seeds at 200,000 steps. '
                   'Choices were frozen before confirmation and were kept regardless of the confirmation results.')
    st.caption('The original final-test selection remains Q-learning. This study adds development evidence. '
               'Independent final evaluation, unseen maps and hardware remain future work.')
