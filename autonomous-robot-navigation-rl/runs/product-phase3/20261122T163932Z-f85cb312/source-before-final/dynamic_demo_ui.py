"""The single partially observed dynamic-navigation product UI."""
import html,json,time
from pathlib import Path
import streamlit as st
from .core import ROOT,read_json
from .app_scenarios import NAMES,scenario as phase2_scenario
from .phase3_demo import NAMES as PHASE3_NAMES,scenario as phase3_scenario
from . import phase3_ui

def scenario(name,*args,**kwargs):
    return phase3_scenario(name,*args,**kwargs) if name in PHASE3_NAMES else phase2_scenario(name,*args,**kwargs)
from .app_registry import registry,newest_by_algorithm
from .app_service import generate,frame,cursor,compare,start_job,job_status,saved_comparison

LABELS={'q-learning':'Q-learning','sarsa':'SARSA','dqn':'DQN','ppo':'PPO'}

def grid_svg(snapshot,truth=False,trail=()):
    cells=[];size=38
    occupancy=snapshot['occupancy'] if truth else snapshot['memory']['occupancy'];seen=snapshot['memory']['currently_seen']
    for r in range(10):
        for c in range(10):
            v=int(occupancy[r][c]);color='#607585' if v==1 else '#dce7ee' if v==-1 else '#f7fafc'
            if truth and v==1 and snapshot['true_rows'][r][c]!='#':color='#f3a746'
            stroke='#238eaa' if not truth and seen[r][c] else '#b9c7d1'
            cells.append(f'<rect x="{c*size+24}" y="{r*size+24}" width="{size}" height="{size}" fill="{color}" stroke="{stroke}"/>')
    for r,c in trail:cells.append(f'<circle cx="{c*size+43}" cy="{r*size+43}" r="4" fill="#80aabb"/>')
    for position,label,color in ((snapshot['goal'],'G','#198754'),(snapshot['position'],'R','#1265b4')):
        r,c=position;cells.append(f'<circle cx="{c*size+43}" cy="{r*size+43}" r="14" fill="{color}"/><text x="{c*size+43}" y="{r*size+48}" text-anchor="middle" fill="white" font-size="15">{label}</text>')
    for i in range(10):
        cells.append(f'<text x="{i*size+43}" y="17" text-anchor="middle" font-size="12" fill="#758a99">{i}</text>')
        cells.append(f'<text x="14" y="{i*size+48}" text-anchor="middle" font-size="12" fill="#758a99">{i}</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 426 410">'+''.join(cells)+'</svg>'

def simulator():
    st.header('Robot Simulator');st.write('Choose a delivery route and a controller, then record one journey. Playback uses that saved journey.')
    left,right=st.columns([1,2])
    with left:
        name=st.selectbox('Map preset',NAMES+PHASE3_NAMES);mode=st.radio('Obstacle mode',['Static','Temporary'],horizontal=True,index=1)
        default=scenario(name,mode);custom=st.checkbox('Choose start and goal')
        start=default['start'];goal=default['goal']
        if custom:
            a,b=st.columns(2)
            start=[a.number_input('Start row',0,9,value=start[0]),b.number_input('Start column',0,9,value=start[1])]
            goal=[a.number_input('Goal row',0,9,value=goal[0]),b.number_input('Goal column',0,9,value=goal[1])]
        items=sorted(registry(),key=lambda i:not i.get('phase3_selected',False));options={'Observed-Map Replanner (non-RL reference)':'Observed-Map Replanner','Random (non-RL reference)':'Random reference'}
        options.update({item['label']:item for item in items})
        choice=st.selectbox('Controller / checkpoint',list(options))
        if not items:st.info('No native RL checkpoints yet. Train a quick starter in Train Models; the references are available now.')
        seed=st.number_input('Evaluation seed',0,2**31-1,value=0)
        if st.button('Run journey',type='primary',use_container_width=True):
            try:
                with st.spinner('Recording real observations and actions…'):
                    record,path=generate(scenario(name,mode,start,goal),options[choice],int(seed))
                st.session_state.journey=record;st.session_state.journey_file=str(path);st.session_state.frame_index=0;st.session_state.playing=False
            except Exception as exc:st.error('Unable to run: '+str(exc))
        st.caption('Blue R = robot · Green G = goal. A blocked attempted move ends in collision. The replanner is a separate reference; it never substitutes for an RL policy.')
    with right:
        if 'journey' not in st.session_state:
            st.image(grid_svg({'occupancy':[[int(ch=='#') for ch in row] for row in default['rows']],
                'memory':{'occupancy':[[-1]*10 for _ in range(10)],'currently_seen':[[False]*10 for _ in range(10)]},
                'true_rows':default['rows'],'position':start,'goal':goal},True),caption='Preset preview. Orange temporary blockers appear in the recorded scene.')
        else:playback()

def playback():
    @st.fragment(run_every=.2 if st.session_state.get('playing') else None)
    def player():
        record=st.session_state.journey;length=len(record['transitions']);index=st.session_state.get('frame_index',0)
        st.caption('Recorded route: '+record['scenario']['id']+' · Controller: '+str(record.get('policy_id','recorded controller')))
        controls=st.columns(6)
        for col,label,command in zip(controls,['Play','Pause','Back','Step','Reset','End'],['play','pause','back','step','reset','end']):
            if col.button(label,key='player_'+command,use_container_width=True):
                if command in ('play','pause'):st.session_state.playing=command=='play';st.session_state.last_tick=time.monotonic();st.rerun()
                st.session_state.playing=False;st.session_state.frame_index=length if command=='end' else cursor(index,command,length);st.rerun()
        speed=st.select_slider('Playback speed (frames/second)',options=[1,2,4,8],value=4)
        if st.session_state.get('playing') and time.monotonic()-st.session_state.get('last_tick',0)>=1/speed:
            index=cursor(index,'step',length);st.session_state.frame_index=index;st.session_state.last_tick=time.monotonic()
            if index==length:st.session_state.playing=False
        chosen=st.slider('Seek recorded step',0,length,value=int(st.session_state.get('frame_index',0)),key='seek_'+str(st.session_state.get('frame_index',0)))
        if chosen!=st.session_state.get('frame_index',0):st.session_state.frame_index=chosen;st.session_state.playing=False
        past_actions=[t['action'] for t in record['transitions'][:chosen]];wait_streak=0
        for action in reversed(past_actions):
            if action!=4:break
            wait_streak+=1
        st.caption('Recorded public history: last action '+(['Up','Down','Left','Right','Wait'][past_actions[-1]] if past_actions else 'none')+' | consecutive Waits '+str(wait_streak)+'. Phase3 uses only public past decisions, never blockage identity or clearing time.')
        index=chosen;snap=frame(record,index);truth=st.toggle('Show scene truth — viewer only',value=True)
        a,b=st.columns(2);trail=[record['initial_snapshot']['position']]+[t['snapshot']['position'] for t in record['transitions'][:index]]
        a.image(grid_svg(snap,truth,trail),caption='Scene truth (never supplied to the controller)' if truth else 'Robot memory only: grey occupied, pale unknown')
        b.image(grid_svg(snap,False),caption='Robot memory: blue borders are sensed now. Occupied cells do not reveal permanent vs temporary.')
        st.caption('Sensors N / S / E / W: '+ ' / '.join('clear within 2 cells' if v==0 else f'blocked at {int(v*2)} cell(s)' for v in snap['scan']['distances']))
        m=record['metrics'];stats=st.columns(5)
        stats[0].metric('Recorded outcome',m['outcome'].title());stats[1].metric('Steps',f'{index} / {length}');stats[2].metric('Total reward',f"{m['return']:.2f}");stats[3].metric('Wait actions',m['wait_count']);stats[4].metric('Moving revisits',m['repeated_movement_visits'])
        st.caption(f"Two-cell loops: {m['two_cell_oscillation_windows']} · Recoveries: {m['recoveries']} / {m['recovery_opportunities']} observed opportunities. Outcome and totals describe the full recorded journey.")
        path=Path(st.session_state.journey_file)
        st.download_button('Download this replay',path.read_bytes(),file_name=path.name,mime='application/gzip')
    player()

def training():
    st.header('Train Models');st.write('Create real five-action checkpoints. Quick training is a small CPU starter and may still navigate poorly.')
    preset=st.radio('Training preset',['quick','standard'],horizontal=True)
    feature_choice=st.selectbox('Learning representation',['Phase 3 public history','Phase 2 local goal features'])
    features='product-v3' if feature_choice.startswith('Phase 3') else 'product-v2'
    chosen=st.multiselect('RL algorithms',list(LABELS),default=list(LABELS),format_func=lambda v:LABELS[v])
    actions=st.number_input('Actions per model',min_value=32,max_value=50000,value=256 if preset=='quick' else 10000,step=32 if preset=='quick' else 200)
    shaping=st.selectbox('Training feedback',['none','liveness-v1'] if features=='product-v3' else ['none','progress-v1'],help='Feedback is train-only. Phase3 liveness uses public distance progress, recent revisits and repeated Waits with an observed unvisited adjacent alternative. It knows no obstacle identity or clear time. Official evaluation reward is unchanged.')
    st.caption('Quick is a small starter. Phase3 uses 8 past public decision positions, the last action and a Wait counter: 409 public inputs, 431 internal state values and 520 derived features. Phase2 uses its original 480 features. Q-learning uses stabilized targets. Training feedback and all measured failures are recorded.')
    seed=st.number_input('Training seed',min_value=0,max_value=2**31-1,value=74001)
    st.code(f'python -m robot_navigation.app_training --preset {preset} --models '+ ' '.join(chosen)+f' --actions {actions} --seed {seed} --shaping {shaping} --features {features}',language='powershell')
    if preset=='standard':st.warning('Standard is user-initiated development training. It may take minutes or longer on CPU; closing the browser does not stop it. No archived data is used.')
    status=job_status(st.session_state.job) if st.session_state.get('job') else None
    running=status and status['status'] in ('starting','running')
    if st.button('Start training',type='primary',disabled=bool(running) or not chosen):
        from .app_training import model_settings
        cfg=model_settings(preset)
        if ('ppo' in chosen and actions%cfg['n_steps']) or ('dqn' in chosen and actions%4):st.error('Use an exact PPO rollout multiple (32 quick / 200 standard), and a multiple of 4 for DQN.')
        else:
            try:st.session_state.job=start_job(preset,chosen,int(actions),int(seed),shaping,features)
            except OSError as exc:st.error('Unable to start training: '+str(exc))
            else:st.rerun()
    if status:
        st.progress(float(status.get('fraction',0)),text=f"{status['status'].title()} · {status.get('algorithm','')} · {status.get('completed_actions',0)} / {status.get('total_actions','?')} actions")
        if status.get('error'):st.error(status['error'])
        a,b=st.columns(2)
        if a.button('Refresh training status'):st.rerun()
        if b.button('Cancel training',disabled=not running):
            (Path(st.session_state.job)/'cancel.request').write_text('cancel');st.info('Cancellation requested. The worker preserves the partial run.')
        if status['status']=='complete':st.success('Checkpoints are ready in Simulator and Compare & Export. These are inference checkpoints, not exact-resume snapshots.')
    items=registry();st.subheader('Available native checkpoints')
    st.dataframe([{'Algorithm':LABELS[i['algorithm']],'Status':i['selection_scope'],'Features':i['feature_version'],'Agent state':i['agent_state_values'],'Actions':i['actions'],'Seed':i['seed'],'Run':i['id']} for i in items],use_container_width=True)

    if items:
        target=st.selectbox('Checkpoint integrity check',[i['label'] for i in items],key='integrity_checkpoint')
        if st.button('Verify selected checkpoint'):
            from .app_registry import load_model
            try:
                item=next(i for i in items if i['label']==target);load_model(item);st.success('Task, feature/history version, five actions and SHA-256 verified: '+item['checkpoint_sha256'])
            except ValueError as exc:st.error(str(exc))

def comparison():
    st.header('Compare & Export')
    st.write('Choose saved development evidence or the single published product held-out assessment. References are separately labeled non-RL.')
    available=phase3_ui.SCOPES;default=0 if (ROOT/'runs/phase3-publication.json').is_file() else 1 if (ROOT/'runs/phase3-selection.json').is_file() else 2
    scope=st.selectbox('Evaluation scope',available,index=default)
    if st.session_state.get('comparison_scope')!=scope:
        st.session_state.comparison_scope=scope;st.session_state.comparison=phase3_ui.saved(scope)
    if st.button('Load saved comparison'):st.session_state.comparison=phase3_ui.saved(scope)
    held=scope==phase3_ui.SCOPES[0]
    if held:
        st.info('Frozen product held-out results are read-only. This page never reruns them or uses them to tune checkpoints. Phase3 may retain a Phase2 checkpoint when development selection favored it.')
    else:
        items=registry();selected={}
        for algorithm in LABELS:
            candidates=[i for i in items if i['algorithm']==algorithm];candidates.sort(key=lambda i:not (i.get('phase3_selected') if scope==phase3_ui.SCOPES[1] else i.get('phase2_baseline')))
            if not candidates:st.info(LABELS[algorithm]+' has no checkpoint');continue
            label=st.selectbox(LABELS[algorithm]+' checkpoint',[i['label'] for i in candidates],key='compare_'+scope+algorithm);selected[algorithm]=next(i for i in candidates if i['label']==label)
        include_random=st.checkbox('Include Random as an additional non-RL reference')
        if st.button('Run fair comparison',type='primary',disabled=len(selected)!=4):
            try:
                with st.spinner('Explicit development evaluation; no model updates or held-out cases'):
                    st.session_state.comparison=phase3_ui.development_compare(selected,include_random) if scope==phase3_ui.SCOPES[1] else compare(selected,include_random=include_random)
            except Exception as exc:st.error('Comparison could not finish: '+str(exc))
    result=st.session_state.get('comparison')
    if not result:st.info('No saved result is available for this scope yet.');return
    manifest=result['manifest'];st.caption(manifest.get('purpose','Recorded product comparison'))
    if manifest.get('holdout_suite_hash'):st.code('Holdout: '+manifest['holdout_version']+' | '+manifest['holdout_suite_hash'])
    overview=[row for row in manifest['summary'] if row['Subset']=='all'];st.dataframe(overview,use_container_width=True,hide_index=True)
    import pandas as pd
    st.bar_chart(pd.DataFrame(overview).set_index('Policy')[['Success %']])
    st.caption('Counts keep their denominators. Successful-only Wait and step metrics exclude failed episodes. Empty values mean N/A. Observed alternatives are safe next cells, not a guaranteed whole detour or knowledge of obstacle permanence.')
    with st.expander('Condition, distance, topology and outcome breakdowns'):st.dataframe(manifest['summary'],use_container_width=True,hide_index=True)
    with st.expander('All case outcomes and checkpoint provenance'):
        st.dataframe(result['rows'],use_container_width=True,hide_index=True);st.json(manifest.get('training_provenance',{}))
    folder=Path(result['folder']);a,b,c,d=st.columns(4)
    a.download_button('Summary CSV',(folder/'summary.csv').read_bytes(),file_name='summary.csv',mime='text/csv')
    b.download_button('Case-level CSV',(folder/'case-results.csv').read_bytes(),file_name='case-results.csv',mime='text/csv')
    c.download_button('Comparison JSON',(folder/'manifest.json').read_bytes(),file_name='comparison.json',mime='application/json')
    d.download_button('Results + all replays ZIP',(folder/'results-and-replays.zip').read_bytes(),file_name='results-and-replays.zip',mime='application/zip')
    if (folder/'failure-traces.json').is_file():st.download_button('Failure trace JSON',(folder/'failure-traces.json').read_bytes(),file_name='failure-traces.json',mime='application/json')
    kind=st.radio('Recorded journeys',['Failures','Successes','All'],horizontal=True)
    rows=[r for r in result['rows'] if kind=='All' or (r['outcome']=='success')==(kind=='Successes')]
    if rows:
        labels=[f"{i}: {row['policy']} | {row['case_id']} | {row['outcome']} ({row['steps']} actions)" for i,row in enumerate(rows)];chosen=st.selectbox('Choose a recorded journey',labels);row=rows[labels.index(chosen)];path=(folder/row['replay']).resolve()
        if path.is_relative_to(folder.resolve()):st.button('Open recorded journey in Simulator',on_click=phase3_ui.open_replay,args=(str(path),))

def guide():
    st.header('Guide');st.markdown('''### A 3-5 minute walkthrough
1. Choose a **Phase 3 selected** checkpoint explicitly in Simulator. Member 1: Q-learning, *Temporary bridge - Wait*; member 2: SARSA, *New static maze*; member 3: DQN, *Temporary gate - detour*; member 4: PPO, *Permanent dead end with detour*. Keep Temporary mode and seed 0. These trajectories were checked before held-out evaluation.
2. Run once, then Play/Pause/Step/Seek saved frames. Compare human-only truth with robot memory. Wait is correct on the bridge; all five actions remain available.
3. Show Q-learning's honestly timed-out *New static maze*. Do not switch to the replanner and call it RL success.
4. In Compare & Export, view the saved **Phase 3 product held-out** assessment, compare Phase2/Phase3 on the same panel, download CSV/JSON/ZIP and open a failure replay. Held-out viewing never reruns evaluation.
5. Train Models shows real feature versions, actions and integrity checks. Start training only when explicitly intended; no live training is needed for this presentation. See the four-person script below for checked actions/Waits and exact checkpoint IDs.

### History is reference material
The old coordinate-only four-action application has been retired. Its data, source, failed trials and Phase 5A–5H history remain accessible through Archives and the repository archive guide. Older models can be incompatible with this task. Archived scores are never presented as results of new app training. There is no old application mode or model-transfer adapter.
''')

    team=ROOT.parent/'docs/team-demo-script.md'
    if team.is_file():
        with st.expander('Four-person teammate demonstration'):st.markdown(team.read_text(encoding='utf-8'))

def archives():
    st.header('Archives — read-only reference');root=ROOT.parent;receipt=root/'archive/manifests/integrity.json'
    if receipt.is_file():
        r=read_json(receipt);st.success(f"Preservation verified: {r['file_versions']:,} unique file versions, {r['missing_files']} missing files, {r['missing_lfs']} missing LFS payloads.")
    st.write('Historical runs, checkpoints, negative results, source versions and Phase 5A–5H reports are indexed in archive/INDEX.csv. Historical models and scientific scores are not loaded into this app registry.')
    st.code('python scripts/export_history_archives.py --output-dir history-exports\npython scripts/export_history_archives.py --output-dir history-exports --verify',language='powershell')
    st.caption('Run from the repository root. ZIP exports stream to disk, preserve originals, include inventories and fail if data is missing. Choose a fresh output folder.')
    guide=root/'archive/README.md'
    if guide.is_file():st.markdown(guide.read_text(encoding='utf-8'))
    docs=list((ROOT/'docs').glob('partial_phase5*_results.md'))
    if docs:
        selected=st.selectbox('Read an archived research report',[p.name for p in docs]);p=next(p for p in docs if p.name==selected)
        st.download_button('Download historical report',p.read_bytes(),file_name=p.name,mime='text/markdown')
        with st.expander('View historical text (not app results)'):st.markdown(p.read_text(encoding='utf-8'))

def render():
    st.set_page_config(page_title='Partial & Dynamic Robot Navigation',page_icon='🤖',layout='wide')
    st.title('Partial & Dynamic Robot Navigation');st.caption('Sense nearby. Remember. Move, detour or wait. Four native RL methods, one honest simulator.')
    page=st.sidebar.radio('Workspace',['Robot Simulator','Train Models','Compare & Export','Guide','Archives'],key='workspace_page')
    st.sidebar.caption('10×10 world · range-two sensing · five actions. Scene truth is display-only.')
    {'Robot Simulator':simulator,'Train Models':training,'Compare & Export':comparison,'Guide':guide,'Archives':archives}[page]()
