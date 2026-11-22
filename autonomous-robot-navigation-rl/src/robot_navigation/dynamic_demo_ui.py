"""The single partially observed dynamic-navigation product UI."""
import html,json,time
from pathlib import Path
import streamlit as st
from .core import ROOT,read_json
from .app_scenarios import NAMES,scenario
from .app_registry import registry,newest_by_algorithm
from .app_service import generate,frame,cursor,compare,start_job,job_status

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
        name=st.selectbox('Map preset',NAMES);mode=st.radio('Obstacle mode',['Static','Temporary'],horizontal=True,index=1)
        default=scenario(name,mode);custom=st.checkbox('Choose start and goal')
        start=default['start'];goal=default['goal']
        if custom:
            a,b=st.columns(2)
            start=[a.number_input('Start row',0,9,value=start[0]),b.number_input('Start column',0,9,value=start[1])]
            goal=[a.number_input('Goal row',0,9,value=goal[0]),b.number_input('Goal column',0,9,value=goal[1])]
        items=registry();options={'Observed-Map Replanner (non-RL reference)':'Observed-Map Replanner','Random (non-RL reference)':'Random reference'}
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
    chosen=st.multiselect('RL algorithms',list(LABELS),default=list(LABELS),format_func=lambda v:LABELS[v])
    actions=st.number_input('Actions per model',min_value=32,max_value=50000,value=256 if preset=='quick' else 10000,step=32 if preset=='quick' else 100)
    seed=st.number_input('Training seed',min_value=0,max_value=2**31-1,value=74001)
    st.code(f'python -m robot_navigation.app_training --preset {preset} --models '+ ' '.join(chosen)+f' --actions {actions} --seed {seed}',language='powershell')
    if preset=='standard':st.warning('Standard is user-initiated development training. It may take minutes or longer on CPU; closing the browser does not stop it. No archived data is used.')
    status=job_status(st.session_state.job) if st.session_state.get('job') else None
    running=status and status['status'] in ('starting','running')
    if st.button('Start training',type='primary',disabled=bool(running) or not chosen):
        from .app_training import model_settings
        cfg=model_settings(preset)
        if ('ppo' in chosen and actions%cfg['n_steps']) or ('dqn' in chosen and actions%4):st.error('Use an exact PPO rollout multiple (32 quick / 100 standard), and a multiple of 4 for DQN.')
        else:
            try:st.session_state.job=start_job(preset,chosen,int(actions),int(seed))
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
    st.dataframe([{'Algorithm':LABELS[i['algorithm']],'Actions':i['actions'],'Seed':i['seed'],'Run':i['id']} for i in items],use_container_width=True)

def comparison():
    st.header('Compare & Export');st.write('All four RL checkpoints face the identical four manual development scenarios and event seeds. References are explicitly non-RL.')
    items=registry();selected={}
    for algorithm in LABELS:
        candidates=[i for i in items if i['algorithm']==algorithm]
        if not candidates:st.info(f'{LABELS[algorithm]} has no checkpoint. Train it before a four-model comparison.');continue
        label=st.selectbox(LABELS[algorithm]+' checkpoint',[i['label'] for i in candidates],key='compare_'+algorithm)
        selected[algorithm]=next(i for i in candidates if i['label']==label)
    random=st.checkbox('Include Random as an additional non-RL reference')
    if st.button('Run fair comparison',type='primary',disabled=len(selected)!=4):
        st.session_state.pop('comparison',None)
        try:
            with st.spinner('Evaluating each checkpoint once per scenario, without updates…'):st.session_state.comparison=compare(selected,include_random=random)
        except Exception as exc:st.error('Comparison could not finish: '+str(exc))
    if st.session_state.get('comparison'):
        r=st.session_state.comparison;st.dataframe(r['manifest']['summary'],use_container_width=True,hide_index=True)
        st.caption('No recovery opportunity is shown as blank, not zero. Failure routes have no successful-path efficiency. Static/dynamic subsets retain their own denominators.')
        st.dataframe(r['rows'],use_container_width=True,hide_index=True)
        folder=Path(r['folder']);a,b,c=st.columns(3)
        a.download_button('Case-level CSV',(folder/'case-results.csv').read_bytes(),file_name='case-results.csv',mime='text/csv')
        b.download_button('Comparison JSON',(folder/'manifest.json').read_bytes(),file_name='comparison.json',mime='application/json')
        c.download_button('Results + all replays ZIP',(folder/'results-and-replays.zip').read_bytes(),file_name='results-and-replays.zip',mime='application/zip')

def guide():
    st.header('Guide');st.markdown('''### A robot that can only see nearby
The robot sees occupancy along four cardinal rays, at most **two cells** away. Sensor order is **North, South, East, West**. It remembers sensed occupancy, how old it is, visits and consecutive blocked observations. Unknown areas and stale walls remain uncertain. The robot knows its own position and the goal.

The unchanged observation has **409 values**. B1 builds **412 features**, including a bias and the relative goal scaled by four. Permanent walls and temporary pallets look the same while occupied. The robot never receives event schedules, unblock times, scene truth or a route oracle.

### Five actions and honest outcomes
**0 Up · 1 Down · 2 Left · 3 Right · 4 Wait**. Every action costs −0.01; reaching the goal adds +10; a blocked attempted move adds −5 and ends in collision. An episode ends after at most 200 steps; timeout has no extra penalty. Waiting can reveal that a pallet cleared, but a model can also get stuck waiting. That failure stays visible.

### Four real learning methods
- **Q-learning:** learns five action values from public features and bootstraps the largest next value.
- **SARSA:** uses the next action sampled by its current exploration policy for its update.
- **DQN:** a neural action-value model with replay and a target network.
- **PPO:** a neural action policy and value model trained on collected rollouts.

The **Observed-Map Replanner** is a separate non-RL reference using remembered occupancy only. It is never an invisible fallback for RL. Quick checkpoints prove that the workflow works; they are not evidence of strong navigation or deployment readiness.

### A 3–5 minute walkthrough
1. In **Robot Simulator**, choose *Blocked aisle — wait for clearing*, Temporary, and the replanner. Run once, Play, Pause and Step. Compare scene truth with robot memory.
2. Try *Storage rack detour*. Seek backward and forward; playback never makes new decisions.
3. In **Train Models**, explicitly start Quick for all four methods (256 actions each), then Refresh status. Existing starter checkpoints may already be available.
4. Select a native checkpoint in Simulator. Keep collisions, loops and timeouts visible.
5. In **Compare & Export**, select four checkpoints, run the fixed development panel and download CSV plus replay ZIP.

### History is reference material
The old coordinate-only four-action application has been retired. Its data, source, failed trials and Phase 5A–5H history remain accessible through Archives and the repository archive guide. Older models can be incompatible with this task. Archived scores are never presented as results of new app training. There is no old application mode or model-transfer adapter.
''')

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
    page=st.sidebar.radio('Workspace',['Robot Simulator','Train Models','Compare & Export','Guide','Archives'])
    st.sidebar.caption('10×10 world · range-two sensing · five actions. Scene truth is display-only.')
    {'Robot Simulator':simulator,'Train Models':training,'Compare & Export':comparison,'Guide':guide,'Archives':archives}[page]()
