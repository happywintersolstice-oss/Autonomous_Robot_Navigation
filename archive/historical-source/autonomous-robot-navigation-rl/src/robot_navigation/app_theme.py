"""Accessible presentation for the local robot navigation app."""

APP_CSS = '''<style>
:root {--lab-bg:#f4f7f8;--lab-panel:#ffffff;--lab-border:#d8e3e7;
 --lab-text:#17313c;--lab-muted:#526b77;--lab-cyan:#087e8b;--lab-green:#237453;}
.stApp {background:var(--lab-bg);color:var(--lab-text);}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stMainBlockContainer"] {max-width:1280px;padding:2.2rem 2rem 3rem;}
[data-testid="stVerticalBlock"] {gap:.9rem;}
h1 {font-size:2.2rem !important;letter-spacing:-.04em;line-height:1.2;}
h2 {font-size:1.45rem !important;letter-spacing:-.025em;}
h3 {font-size:1.1rem !important;}
[data-testid="stCaptionContainer"] {color:var(--lab-muted);}
[data-testid="stWidgetLabel"] p {font-size:.88rem;font-weight:600;}
.stButton button {border-radius:10px;min-height:44px;border-color:#b6ccd3;background:white;color:#17313c;}
.stButton button[kind="primary"] {background:#087e8b;color:white;border-color:#087e8b;}
.stButton button:hover {border-color:#087e8b;}
button:focus-visible, input:focus-visible {outline:3px solid #087e8b !important;outline-offset:3px;}
[data-baseweb="select"] > div {background:white;border-color:#b6ccd3;border-radius:10px;}
[data-testid="stExpander"] {border-color:var(--lab-border);background:white;border-radius:12px;}
[data-testid="stMetric"] {background:white;border:1px solid var(--lab-border);border-radius:12px;padding:16px;}
[data-testid="stMetricLabel"] {color:var(--lab-muted);}
[data-testid="stMetricValue"] {font-size:1.75rem;font-weight:650;color:#17313c;}
.stTabs [data-baseweb="tab-list"] {gap:1.25rem;border-bottom:1px solid var(--lab-border);}
.stTabs [data-baseweb="tab"] {height:48px;padding:8px 3px;color:#526b77;font-size:.95rem;}
.stTabs [aria-selected="true"] {color:#066773;font-weight:650;}
.stTabs [data-baseweb="tab-highlight"] {background:#087e8b;}
.lab-topbar {display:flex;align-items:center;justify-content:space-between;gap:16px;margin:1rem 0 .4rem;}
.lab-topbar h1 {margin:0;padding:0;}
.lab-breadcrumb,.lab-eyebrow {color:#526b77;font-size:.78rem;font-weight:650;letter-spacing:.06em;text-transform:uppercase;}
.lab-environment {padding:8px 14px;background:#e7f3f1;border-radius:100px;font-size:.85rem;color:#205951;}
.lab-environment small,.lab-icon {display:none;}
.lab-subtle {color:#526b77;font-size:.85rem;line-height:1.55;}
.lab-session {display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#526b77;font-size:.82rem;}
.lab-context {display:flex;flex-wrap:wrap;gap:1.5rem;}
.lab-context small {display:block;color:#526b77;font-size:.75rem;}
.lab-card-heading {display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:.7rem;}
.lab-card-heading strong {display:block;font-size:1.1rem;}
.lab-card-heading small {display:block;color:#526b77;font-size:.8rem;margin-top:4px;}
.lab-chip {padding:5px 10px;border-radius:100px;background:#e7f3f1;color:#205951;font-size:.72rem;}
.lab-status {display:inline-block;border:1px solid #aac4ce;border-radius:100px;padding:7px 12px;margin:4px 10px 8px 0;font-size:.86rem;color:#294957;background:#edf4f6;}
.lab-success {color:#16603d;background:#e5f5eb;border-color:#9bc5a9;}
.lab-collision {color:#922d2d;background:#fff0ee;border-color:#edb7ae;}
.lab-timeout {color:#795014;background:#fff5df;border-color:#dcc087;}
.lab-kv {display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--lab-border);font-size:.86rem;}
.lab-kv span {color:#526b77;}
.lab-panel {padding:.5rem 0;border-bottom:1px solid var(--lab-border);margin-bottom:1rem;}
.lab-panel strong {display:block;font-size:1rem;}
.st-key-single_map,.st-key-pair_maps,.st-key-four_maps {background:white;border:1px solid var(--lab-border);border-radius:16px;padding:18px;}
.st-key-app_navigation {margin-bottom:.5rem;}
.st-key-app_navigation button {font-size:1rem;font-weight:600;}
.st-key-model_setup {background:white;border:1px solid var(--lab-border);border-radius:12px;padding:14px;}
@media(max-width:800px) {
 [data-testid="stMainBlockContainer"] {padding:2.5rem 1rem 2rem;}
 h1 {font-size:1.8rem !important;}
 .lab-topbar {align-items:flex-start;flex-direction:column;}
 .st-key-model_setup [data-testid="stHorizontalBlock"],
 .st-key-single_replay_content > [data-testid="stHorizontalBlock"] {flex-direction:column;}
 .st-key-model_setup [data-testid="stColumn"],
 .st-key-single_replay_content > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {width:100% !important;flex:1 1 auto !important;}
 .stTabs [data-baseweb="tab-list"] {gap:.75rem;}
 .stTabs [data-baseweb="tab"] {font-size:.83rem;}
 .lab-session {font-size:.76rem;}
}
@media(prefers-reduced-motion:reduce) {* {scroll-behavior:auto !important;transition:none !important;}}
</style>'''
