"""Reference-inspired presentation tokens, scoped to the Streamlit workspace."""

CONTROL_ROOM_CSS = '''<style>
:root {
  --lab-bg:#06121c; --lab-panel:#091923; --lab-border:#1c394a;
  --lab-text:#dceaf3; --lab-muted:#88a9bf; --lab-cyan:#00bdf2; --lab-green:#00d6ae;
}
.stApp {background:var(--lab-bg);color:var(--lab-text);}
[data-testid="stHeader"] {background:transparent !important;height:2rem !important;min-height:0 !important;pointer-events:none;}
[data-testid="stHeader"] button {pointer-events:auto;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stMainBlockContainer"] {max-width:1680px;padding:.7rem 2rem 2rem;container-type:inline-size;container-name:workspace;}
[data-testid="stVerticalBlock"] {gap:.8rem;}
[data-testid="stSidebar"] {background:#081621;border-right:1px solid var(--lab-border);}
[data-testid="stSidebar"][data-collapsed="false"] {width:264px !important;min-width:264px !important;}
[data-testid="stSidebar"][data-collapsed="true"] {width:0 !important;min-width:0 !important;}
[data-testid="stSidebarContent"] {padding-top:1rem;}
[data-testid="stSidebarHeader"] {height:0;min-height:0;margin:0;padding:0;}
[data-testid="stSidebarCollapseButton"] {position:absolute;right:0;top:0;visibility:visible;}
[data-testid="stSidebarUserContent"] {padding:1rem .55rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.45rem;}
[data-testid="stSidebar"] .stButton button {justify-content:flex-start;background:transparent;border:0;color:var(--lab-muted);min-height:46px;padding:10px 14px;}
[data-testid="stSidebar"] .stButton button > div {justify-content:flex-start;}
[data-testid="stSidebar"] .stButton button[kind="primary"] {background:#0c3041;color:#6bdbff;border-left:3px solid var(--lab-cyan);border-radius:9px;}
.lab-brand {display:flex;align-items:center;gap:14px;padding:2px 8px 24px;margin-bottom:12px;border-bottom:1px solid var(--lab-border);}
.lab-brand strong {font-size:1.5rem;letter-spacing:.02em;}
.lab-brand strong span {color:var(--lab-cyan);}
.lab-icon {font-family:"Material Symbols Rounded";font-weight:400;font-size:24px;line-height:1;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;}
.lab-brand .lab-icon {background:#0b3548;color:#59d4ff;border:1px solid #1b5267;border-radius:12px;width:44px;height:44px;}
.lab-nav-label {font-size:.67rem;font-weight:600;letter-spacing:.15em;color:#83a5bd;margin:8px 14px 12px;}
.lab-sidebar-status {padding:16px 14px;margin-top:30px;border:1px solid var(--lab-border);border-radius:12px;background:#0a1c28;}
.lab-sidebar-status small {font-size:.7rem;line-height:1.6;color:var(--lab-muted);display:block;margin-top:12px;}
.lab-topbar {display:flex;justify-content:space-between;align-items:center;gap:20px;padding:0 0 12px;margin-bottom:4px;border-bottom:1px solid var(--lab-border);}
.lab-breadcrumb {font-size:.65rem;line-height:1.3;color:var(--lab-muted);margin-bottom:5px;}
.lab-topbar h1 {font-size:1.25rem !important;font-weight:600;letter-spacing:-.035em;margin:0;padding:0 !important;line-height:1.25;}
.lab-environment {display:flex;align-items:center;gap:12px;border:1px solid var(--lab-border);border-radius:10px;padding:6px 12px;min-width:188px;background:#0a1a25;}
.lab-environment .lab-icon {color:var(--lab-cyan);}
.lab-environment small {display:block;font-size:.56rem;line-height:1.2;letter-spacing:.1em;color:var(--lab-muted);}
.lab-environment b {display:block;font-size:.8rem;line-height:1.3;font-weight:500;}
h1 {font-size:1.7rem !important;letter-spacing:-.035em;}
h2 {font-size:1.2rem !important;letter-spacing:-.02em;}
h3 {font-size:1rem !important;font-weight:600;}
[data-testid="stCaptionContainer"] {color:var(--lab-muted);font-size:.75rem;}
.stTabs [data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid var(--lab-border);}
.stTabs [data-baseweb="tab"] {padding:10px 16px;height:42px;color:var(--lab-muted);font-size:.83rem;}
.stTabs [aria-selected="true"] {background:#0b2634;color:#9ee5ff;border-radius:8px 8px 0 0;}
.stTabs [data-baseweb="tab-highlight"] {background:var(--lab-cyan);height:2px;box-shadow:0 0 10px #00bdf270;}
.stButton button {border-radius:8px;border-color:#244658;background:#0c1f2b;min-height:40px;font-size:.8rem;color:#c3dbe9;}
.stButton button:hover {border-color:var(--lab-cyan);color:#dff7ff;}
.stButton button[kind="primary"] {color:#dff7ff;font-weight:600;background:#0c4057;border-color:#096485;}
.stButton button:focus-visible {outline:2px solid var(--lab-cyan);outline-offset:3px;}
[data-baseweb="select"] > div {background:#0c202d;border-color:#244658;border-radius:8px;}
[data-testid="stWidgetLabel"] p {font-size:.75rem;color:#a6c4d8;}
[data-testid="stExpander"] {background:#091a26;border-color:var(--lab-border);border-radius:10px;}
[data-testid="stExpander"] summary p {font-size:.8rem;}
.st-key-model_setup {background:#0a1d29;border:1px solid var(--lab-border);border-radius:12px;padding:8px 14px;}
.st-key-model_setup [data-testid="stVerticalBlock"] {gap:.3rem;}
.st-key-single_playback_controls, .st-key-comparison_playback_controls {padding:4px 0;}
.st-key-single_playback_controls [data-testid="stHorizontalBlock"] {gap:.35rem;}
.st-key-single_playback_controls button {padding:6px 8px;}
.lab-eyebrow {font-size:.6rem;letter-spacing:.14em;color:#71a5c2;font-weight:600;}
.lab-subtle {color:var(--lab-muted);font-size:.74rem;line-height:1.5;}
.lab-session {display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:var(--lab-muted);font-size:.7rem;padding:0 0 4px;}
.lab-session b {color:#b5d4e6;font-weight:500;}
[data-testid="stMetric"] {background:var(--lab-panel);border:1px solid var(--lab-border);border-radius:12px;padding:16px 18px;}
[data-testid="stMetricLabel"] {color:var(--lab-muted);font-size:.75rem;}
[data-testid="stMetricValue"] {font-size:1.6rem;font-weight:600;color:#e3f1fb;letter-spacing:-.03em;}
:is(.st-key-left_metrics,.st-key-right_metrics) [data-testid="stMetric"] {min-height:100px;padding:14px;}
:is(.st-key-left_metrics,.st-key-right_metrics) [data-testid="stMetricLabel"] {display:flex;gap:0;}
:is(.st-key-left_metrics,.st-key-right_metrics) [data-testid="stMetricLabel"] > span {margin-left:6px;}
:is(.st-key-left_metrics,.st-key-right_metrics) [data-testid="stMetricLabel"]::before {font-family:"Material Symbols Rounded";content:"route";font-size:18px;display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:8px;margin-right:8px;background:#073242;color:#00c7ff;flex-shrink:0;}
.st-key-left_metrics > div:nth-child(2) [data-testid="stMetricLabel"]::before {content:"monitoring";color:#b692ff;background:#1d2340;}
.st-key-right_metrics [data-testid="stMetricLabel"]::before {content:"conversion_path";color:#ffb454;background:#2b2b24;}
.st-key-right_metrics > div:nth-child(2) [data-testid="stMetricLabel"]::before {content:"flag";color:#00dfb5;background:#06332f;}
.lab-zone-label {font-size:.57rem;letter-spacing:.15em;color:#81a9c0;padding:0 0 8px;margin:0 0 3px;border-bottom:1px solid var(--lab-border);}
.st-key-single_map, .st-key-pair_maps, .st-key-four_maps,
.st-key-single_inspector, .st-key-reward_panel {background:var(--lab-panel);border:1px solid var(--lab-border);border-radius:12px;padding:18px;}
.st-key-single_map {border-color:#09607a;}
.st-key-single_inspector, .st-key-reward_panel {padding:12px;}
.st-key-single_inspector .lab-kv {gap:6px;padding:9px 0;font-size:.72rem;}
.lab-card-heading {display:flex;justify-content:space-between;align-items:center;gap:12px;border-bottom:1px solid var(--lab-border);padding:0 2px 16px;margin-bottom:6px;}
.lab-card-heading strong {font-size:.95rem;font-weight:600;display:block;}
.lab-card-heading small {display:block;color:var(--lab-muted);font-size:.7rem;margin-top:5px;}
.lab-chip {padding:5px 8px;border:1px solid #126453;background:#07392f;color:#23e2b6;border-radius:5px;font-size:.59rem;letter-spacing:.08em;white-space:nowrap;}
.lab-panel {padding:0 2px 12px;margin-bottom:6px;border-bottom:1px solid var(--lab-border);min-height:72px;}
.lab-panel strong {display:block;line-height:1.4;font-size:.9rem;}
.lab-panel code {font-size:.69rem;color:var(--lab-muted);background:none;padding:0;overflow-wrap:anywhere;}
.st-key-single_map [data-testid="stImage"] img {height:auto;object-fit:contain;}
.st-key-pair_maps [data-testid="stImage"] img, .st-key-four_maps [data-testid="stImage"] img {max-height:480px;object-fit:contain;}
.lab-kv {display:flex;justify-content:space-between;gap:12px;padding:10px 0;border-bottom:1px solid var(--lab-border);font-size:.75rem;}
.lab-kv:last-child {border-bottom:0;}
.lab-kv span {color:var(--lab-muted);}
.lab-kv b {color:#d5ebf8;font-weight:500;}
.lab-status {display:inline-block;border:1px solid #35576c;border-radius:5px;padding:4px 8px;margin:4px 8px 4px 0;font-size:.65rem;color:#add3e9;}
.lab-success {color:#28e7ba;border-color:#126453;background:#07392f;}
.lab-collision {color:#ffa6a6;border-color:#8a454a;background:#351c25;}
.lab-timeout {color:#f4c873;border-color:#806638;background:#302819;}
.lab-context {display:flex;gap:1.6rem;flex-wrap:wrap;padding:12px 16px;border:1px solid var(--lab-border);border-radius:10px;}
.lab-context div {font-size:.8rem;}
.lab-context small {display:block;font-size:.6rem;color:var(--lab-muted);letter-spacing:.1em;margin-bottom:3px;}
.st-key-saved_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
.st-key-learning_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {flex:0 0 116px !important;min-width:116px !important;width:116px !important;}
.st-key-saved_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
.st-key-learning_run_picker [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {flex:1 1 0% !important;min-width:0 !important;width:auto !important;}
.st-key-single_decision_details, .st-key-comparison_decision_details_0,
.st-key-comparison_decision_details_1 {padding-top:12px;}
@media(max-width:1250px) {
  [data-testid="stMainBlockContainer"] {padding:.7rem 1.25rem 1.5rem;}
  .st-key-single_replay_content > [data-testid="stHorizontalBlock"] {gap:1rem;}
  .st-key-single_inspector, .st-key-reward_panel {padding:14px;}
}
@container workspace (max-width:1050px) {
  .st-key-single_playback_controls [data-testid="stHorizontalBlock"],
  .st-key-comparison_playback_controls [data-testid="stHorizontalBlock"] {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.5rem;}
  .st-key-single_playback_controls [data-testid="stColumn"],
  .st-key-comparison_playback_controls [data-testid="stColumn"] {width:100% !important;min-width:0 !important;}
  .st-key-single_playback_controls [data-testid="stColumn"]:last-child,
  .st-key-comparison_playback_controls [data-testid="stColumn"]:last-child {grid-column:1 / -1;}
  .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-saved_run_picker),
  .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-learning_run_picker),
  .st-key-comparison_setup [data-testid="stHorizontalBlock"],
  .st-key-pair_maps [data-testid="stHorizontalBlock"]:has(.lab-panel),
  .st-key-four_maps [data-testid="stHorizontalBlock"]:has(.lab-panel) {flex-direction:column;gap:1rem;}
  .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-saved_run_picker) > [data-testid="stColumn"],
  .st-key-model_setup [data-testid="stHorizontalBlock"]:has(.st-key-learning_run_picker) > [data-testid="stColumn"],
  .st-key-comparison_setup [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  .st-key-pair_maps [data-testid="stHorizontalBlock"]:has(.lab-panel) > [data-testid="stColumn"],
  .st-key-four_maps [data-testid="stHorizontalBlock"]:has(.lab-panel) > [data-testid="stColumn"] {width:100% !important;flex:0 0 auto !important;}
  .st-key-single_replay_content [data-testid="stHorizontalBlock"]:has(.st-key-single_map) {display:flex;flex-direction:column;gap:1rem;}
  .st-key-single_replay_content [data-testid="stHorizontalBlock"]:has(.st-key-single_map) > [data-testid="stColumn"] {width:100% !important;flex:0 0 auto !important;}
  .st-key-single_replay_content [data-testid="stHorizontalBlock"]:has(.st-key-single_map) > [data-testid="stColumn"]:nth-child(2) {order:-1;}
  .st-key-left_metrics, .st-key-right_metrics {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;}
}
@media(max-width:600px) {
  [data-testid="stMainBlockContainer"] {padding:3rem .85rem 1rem;}
  .lab-topbar {align-items:flex-start;flex-direction:column;gap:12px;}
  .lab-topbar h1 {font-size:1.4rem !important;}
  .lab-environment {padding:8px 12px;}
  .stTabs [data-baseweb="tab"] {padding:10px 12px;}
  .st-key-model_setup {padding:12px;}
  .st-key-single_map {padding:10px;}
  :is(.st-key-left_metrics,.st-key-right_metrics) [data-testid="stMetric"] {padding:12px;min-height:100px;}
  :is(.st-key-left_metrics,.st-key-right_metrics) [data-testid="stMetricLabel"]::before {width:24px;height:24px;font-size:16px;margin-right:6px;}
  [data-testid="stMetricValue"] {font-size:1.3rem;}
  .lab-context {gap:.75rem;}
}
</style>'''
