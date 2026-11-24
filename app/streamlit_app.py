"""MediNav prototype: hospital medicine-delivery robot driven by reinforcement learning.

    streamlit run app/streamlit_app.py      (or double-click "Start App.cmd")

Tabs
    Delivery      pick a hospital floor and an agent, watch the delivery as an animated replay
    Compare       several agents on the same floor (and success rate over many floors)
    Results       the automated pipeline's tables and figures as a dashboard
    How it works  one-page explanation for the demo / Q&A

All heavy lifting lives in the `medinav` package; this file only arranges it on screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from medinav.agents.registry import baseline, find_runs, load_agent, read_meta  # noqa: E402
from medinav.config import load_config  # noqa: E402
from medinav.env import make_env, map_seed  # noqa: E402
from medinav.evaluation.rollout import run_episode  # noqa: E402
from medinav.viz import episode_animation, paths_figure, pick_figure  # noqa: E402

st.set_page_config(page_title="MediNav · hospital delivery robot", page_icon="\U0001F3E5", layout="wide",
                   initial_sidebar_state="collapsed")
RUNS = ROOT / "runs"
CURRENT_EXPERIMENTS = ("part_b", "part_a", "smoke")

# ====================================================================== styling
CROSS = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56'%3E"
         "%3Cpath d='M24 18h8v6h6v8h-6v6h-8v-6h-6v-8h6z' fill='%231f6feb' fill-opacity='0.035'/%3E%3C/svg%3E\")")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stTabs { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: CROSS_BG, linear-gradient(180deg, #eef4fb 0%, #f7f9fc 340px); }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1420px; }

/* ---- brand header */
.brand { position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between;
         gap: 18px; padding: 18px 24px; border-radius: 18px; color: #fff; margin-bottom: .8rem;
         background: radial-gradient(circle at 92% -20%, rgba(56,217,169,.45) 0, transparent 38%),
                     linear-gradient(105deg, #0a2f6e 0%, #1452c6 55%, #1f6feb 100%);
         box-shadow: 0 10px 30px -12px rgba(20,82,198,.55); }
.brand::after { content: ""; position: absolute; right: -40px; bottom: -60px; width: 220px; height: 220px;
                border-radius: 50%; border: 28px solid rgba(255,255,255,.06); }
.brand .logo { width: 52px; height: 52px; border-radius: 14px; background: rgba(255,255,255,.14);
               border: 1px solid rgba(255,255,255,.28); display: grid; place-items: center; font-size: 1.7rem; }
.brand .who { display: flex; align-items: center; gap: 14px; }
.brand h1 { font-size: 1.6rem; font-weight: 800; margin: 0; color: #fff; letter-spacing: -0.02em; padding: 0; }
.brand p { margin: 2px 0 0; opacity: .88; font-size: .93rem; }
.brand .stats { display: flex; gap: 10px; z-index: 1; flex-wrap: wrap; justify-content: flex-end; }
.brand .stat { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25); border-radius: 12px;
               padding: 6px 12px; text-align: right; min-width: 92px; }
.brand .stat b { display: block; font-size: 1.15rem; font-weight: 800; }
.brand .stat span { font-size: .72rem; opacity: .85; }
.brand .tag { font-size: .72rem; opacity: .8; margin-top: 6px; letter-spacing: .04em; text-transform: uppercase; }

/* ---- tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #ffffffcc; padding: 5px; border-radius: 12px;
                                    border: 1px solid #e3e8ee; width: fit-content; }
.stTabs [data-baseweb="tab"] { padding: 7px 16px; border-radius: 9px; font-weight: 600; height: auto; }
.stTabs [aria-selected="true"] { background: #1f6feb; }
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---- panels and section headers */
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 16px !important;
    border-color: #e3e8ee !important; box-shadow: 0 4px 18px -10px rgba(15,40,80,.18); }
.sec { font-size: .78rem; font-weight: 700; color: #0a2f6e; text-transform: uppercase; letter-spacing: .07em;
       margin: 6px 0 2px; padding-bottom: 6px; border-bottom: 1px dashed #d8e1ec; }
.sec span { text-transform: none; letter-spacing: 0; font-weight: 500; color: #7b8794; float: right; }
.stButton button[kind="primary"] { background: linear-gradient(100deg, #1452c6, #1f6feb); border: 0;
    font-weight: 700; padding: .6rem 1rem; box-shadow: 0 6px 16px -8px rgba(31,111,235,.8); }
.stButton button[kind="primary"]:hover { filter: brightness(1.08); }

/* ---- mission card */
.card { background: #fff; border: 1px solid #e3e8ee; border-radius: 14px; padding: 12px 16px;
        box-shadow: 0 4px 18px -12px rgba(15,40,80,.2); }
.mission { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.mission .route { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 1.02rem; }
.place { padding: 6px 12px; border-radius: 10px; font-weight: 700; }
.place.from { background: #e8f1ff; color: #0a2f6e; border: 1px solid #c9dcfb; }
.place.to { background: #fff1e6; color: #a6420a; border: 1px solid #ffd3b3; }
.arrow { color: #9aa6b2; font-size: 1.1rem; }
.mission .sub { color: #7b8794; font-weight: 400; font-size: .9rem; }
.chip { padding: 6px 13px; border-radius: 999px; font-weight: 700; font-size: .84rem; white-space: nowrap; }
.chip.ok { background: #e6f4ea; color: #1e7a3c; } .chip.bad { background: #fdecea; color: #b42318; }
.chip.warn { background: #fff4dc; color: #8a5a00; } .chip.idle { background: #eef2f6; color: #52606d; }
.hint { margin: 10px 0 2px; padding: 9px 14px; border-radius: 12px; font-size: .88rem; color: #0a2f6e;
        background: linear-gradient(90deg, #e8f1ff, #effaf6); border: 1px solid #d3e3fb; }

/* ---- KPI cards */
.kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 10px 0 6px; }
.kpi { background: #fff; border: 1px solid #e3e8ee; border-radius: 14px; padding: 10px 14px;
       border-top: 4px solid var(--tone); box-shadow: 0 4px 14px -10px rgba(15,40,80,.25); }
.kpi.blue { --tone: #1f6feb; } .kpi.teal { --tone: #12b886; } .kpi.green { --tone: #2f9e44; }
.kpi.pink { --tone: #d6336c; } .kpi.amber { --tone: #f59f00; } .kpi.grey { --tone: #adb5bd; }
.kpi .v { font-size: 1.35rem; font-weight: 800; color: #1f2933; line-height: 1.25; }
.kpi .l { font-size: .72rem; color: #7b8794; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
.kpi .ic { margin-right: 5px; }
.kpi .s { font-size: .78rem; color: #52606d; }

.legend { font-size: .8rem; color: #52606d; display: flex; gap: 14px; flex-wrap: wrap; margin-top: 4px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
.rooms { font-size: .85rem; color: #52606d; }
div[data-testid="stExpander"] { background: #fff; border-radius: 12px; }
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #e3e8ee; border-radius: 14px; padding: 4px;
                                overflow: hidden;
                                box-shadow: 0 4px 18px -12px rgba(15,40,80,.2); }
.foot { text-align: center; color: #9aa6b2; font-size: .78rem; margin-top: 2rem; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                            .brand { flex-direction: column; align-items: flex-start; } }
</style>
""".replace("CROSS_BG", CROSS), unsafe_allow_html=True)


def headline_stats() -> list:
    """(value, caption) pills for the header, from the latest results (empty if none yet)."""
    for exp in CURRENT_EXPERIMENTS:
        f = RUNS / exp / "results" / "summary.csv"
        if f.exists():
            try:
                d = pd.read_csv(f)
                learned = d[~d["agent"].isin(["astar", "random"])].sort_values("success", ascending=False)
                astar = d.loc[d["agent"] == "astar", "success"]
                out = [(f"{learned['success'].iloc[0]:.0%}", "best delivery rate")]
                if len(astar):
                    out.append((f"{astar.iloc[0]:.0%}", "A* baseline"))
                out.append((f"{len(d)}", "agents tested"))
                return out
            except Exception:  # noqa: BLE001
                return []
    return []


st.markdown(
    '<div class="brand"><div class="who"><div class="logo">\U0001F916</div><div>'
    '<h1>MediNav</h1><p>A reinforcement-learning robot that carries medicine from the pharmacy to ward beds</p>'
    '<div class="tag">BMDS2114 Machine Learning · hospital delivery robot</div></div></div>'
    '<div class="stats">' + "".join(f'<div class="stat"><b>{v}</b><span>{c}</span></div>'
                                    for v, c in headline_stats()) + '</div></div>',
    unsafe_allow_html=True)

# ====================================================================== naming
ALGO_NAMES = {"q_learning": "Q-learning", "sarsa": "SARSA", "dqn": "DQN", "double_dqn": "Double DQN",
              "dueling_dqn": "Dueling DQN", "dueling_double_dqn": "Dueling Double DQN", "ppo": "PPO",
              "a2c": "A2C", "recurrent_ppo": "Recurrent PPO", "sac": "SAC", "td3": "TD3",
              "astar": "A* planner (baseline)", "random": "Random (baseline)"}
SUFFIXES = [("_hybrid_shield", " + A* guidance + safety shield"), ("_hybrid", " + A* guidance"),
            ("_tuned", " (tuned)"), ("_noroute", " (no route guidance)"), ("_long", " (2× training)"), ("_curriculum", " + curriculum"),
            ("_euclidean", " (straight-line reward)"), ("_drop_none", " · 20% LiDAR loss, gaps kept"),
            ("_drop_maxrange", " · 20% LiDAR loss, gaps = clear"),
            ("_drop_lastvalid", " · 20% LiDAR loss, gaps = last reading"),
            ("_drop_stack3", " · 20% LiDAR loss, 3-frame memory"), ("_shield", " + safety shield")]
ENSEMBLE_NAMES = {"ens_dqn_family_q": "Ensemble: DQN family (Q-average)",
                  "ens_dueling_seeds": "Ensemble: Dueling DQN, 3 seeds",
                  "ens_vote_discrete": "Ensemble: majority vote (Q-learning, DQN, Dueling)",
                  "ens_sac_td3": "Ensemble: SAC + TD3 (averaged steering)",
                  "ens_continuous_all": "Ensemble: SAC + TD3 + PPO + A2C"}
LEVELS = {0: "Quiet night", 1: "Normal shift", 2: "Busy day", 3: "Visiting hours"}
LEVEL_HELP = ("Quiet night: no people, closest ward. Normal shift: 2 people. Busy day (training level): 4 people, "
              "any ward, 3 trolleys. Visiting hours: 7 people, far wards, 5 trolleys.")
OUTCOMES = {"success": ("Delivered", "ok"), "collision_wall": ("Hit a wall or furniture", "bad"),
            "collision_human": ("Bumped into a person", "bad"), "stuck_oscillating": ("Got stuck", "warn"),
            "too_slow": ("Ran out of time", "warn"), "timeout": ("Ran out of time", "warn")}


def pretty(label: str) -> str:
    if label in ENSEMBLE_NAMES:
        return ENSEMBLE_NAMES[label]
    for base in sorted(ALGO_NAMES, key=len, reverse=True):
        if label == base:
            return ALGO_NAMES[base]
        if label.startswith(base + "_"):
            rest = label[len(base):]
            for suf, text in SUFFIXES:
                if rest == suf:
                    return ALGO_NAMES[base] + text
    return label


# ====================================================================== agents
@st.cache_data(show_spinner=False, ttl=60)
def trained_runs() -> dict:
    """{label: {seed: run_dir}} for the current experiments (Part B overrides Part A labels)."""
    out: dict = {}
    for exp in reversed(CURRENT_EXPERIMENTS):          # later (higher priority) experiments overwrite
        for rd in find_runs(RUNS / exp) if (RUNS / exp).exists() else []:
            try:
                m = read_meta(rd)
            except Exception:  # noqa: BLE001 - half-written run
                continue
            out.setdefault(m["label"], {})[int(m["seed"])] = str(rd)
    return out


@st.cache_data(show_spinner=False)
def ensemble_specs() -> dict:
    try:
        return {e["name"]: e for e in load_config(ROOT / "configs" / "part_b.yaml").get("ensembles") or []}
    except Exception:  # noqa: BLE001
        return {}


def agent_options() -> list:
    runs = trained_runs()
    ens = [n for n, s in ensemble_specs().items() if all(m in runs for m in s["members"])]
    learned = sorted(runs, key=lambda l: pretty(l).lower())
    return ["astar", "random"] + learned + ens


@st.cache_data(show_spinner=False, ttl=60)
def best_measured() -> list:
    """Agents ranked by measured test success (latest experiment first), for the default pick."""
    for exp in CURRENT_EXPERIMENTS:
        f = RUNS / exp / "results" / "summary.csv"
        if f.exists():
            try:
                return pd.read_csv(f).sort_values("success", ascending=False)["agent"].tolist()
            except Exception:  # noqa: BLE001 - a broken summary just falls back to the fixed order
                pass
    return []


def default_agent(opts):
    for pick in best_measured() + ["sac", "td3", "dueling_dqn", "ppo"]:
        if pick in opts and pick not in ("astar", "random"):
            return pick
    return opts[0]


@st.cache_resource(show_spinner="Loading the trained model…", max_entries=24)
def get_agent(label: str, seed: int):
    if label in ("astar", "random"):
        return baseline(label), {"action_mode": "discrete", "env": {}, "wrappers": {}}
    runs = trained_runs()
    if label in runs:
        seeds = runs[label]
        return load_agent(seeds.get(seed, seeds[min(seeds)]))
    spec = ensemble_specs()[label]                      # ensemble: load just its members
    from medinav.evaluation.evaluate import build_ensembles

    trained = {}
    for m in spec["members"]:
        seeds = runs[m]
        trained[m] = [load_agent(seeds.get(seed, seeds[min(seeds)]))]
    built = build_ensembles([{**spec, "per_seed": True}], trained)
    if not built:
        raise ValueError(f"{label}: members do not fit together")
    return built[0]


def safe_agent(label, seed):
    try:
        return get_agent(label, seed)
    except Exception as exc:  # noqa: BLE001 - shown to the user, app keeps running
        st.error(f"Could not load **{pretty(label)}**: {type(exc).__name__}: {exc}")
        return None, None


def env_for(meta, cond):
    cfg = {**(meta.get("env") or {}), **cond, "action_mode": meta["action_mode"]}
    return make_env(cfg, meta.get("wrappers"))


def rooms_visited(frames) -> list:
    seq = []
    for f in frames:
        x, y = f["pos"]
        name = ""
        for room in f.get("rooms") or []:
            x0, y0, x1, y1 = room.rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                name = room.name
                break
        if not name:
            cx0, cy0, cx1, cy1 = f.get("corridor") or (0, 0, 0, 0)
            name = "Corridor" if cx0 <= x <= cx1 and cy0 <= y <= cy1 else "Doorway"
        if name != "Doorway" and (not seq or seq[-1] != name):
            seq.append(name)
    return seq


def floor_controls(key: str, cols=None, mission: bool = False):
    """Floor + busy-ness + stress options. Returns (map seed, env overrides)."""
    c = cols or st.columns(1)
    split = c[0].selectbox("Floors", ["test", "validation", "train"], key=key + "split",
                           format_func={"test": "Unseen test floors", "validation": "Validation floors",
                                        "train": "Training floors"}.get,
                           help="Test floors were never seen during training or tuning.")
    idx = c[min(1, len(c) - 1)].number_input("Floor number", 0, 9999, 0, key=key + "idx")
    level = c[min(2, len(c) - 1)].select_slider("How busy is the hospital?", options=list(LEVELS), value=2,
                                                format_func=LEVELS.get, key=key + "lvl", help=LEVEL_HELP)
    with st.expander("Stress test (make it harder than training)"):
        s1, s2 = st.columns(2)
        people = s1.slider("People on the floor", -1, 10, -1, key=key + "ppl", help="-1 = as the busy-ness level says")
        speed = s1.slider("Walking speed (m/s)", 0.0, 1.2, 0.5, 0.1, key=key + "spd")
        careless = s1.toggle("Careless people (walk into the robot)", key=key + "care")
        drop = s2.slider("Share of LiDAR rays lost", 0.0, 0.6, 0.0, 0.05, key=key + "drop")
        noise = s2.slider("LiDAR noise σ (m)", 0.0, 0.15, 0.0, 0.01, key=key + "noise")
        big = s2.toggle("Bigger hospital wing (32 × 20 m)", key=key + "big")
    cond = {"difficulty": level, "human_speed": speed, "humans_yield": not careless,
            "lidar_noise_std": noise, "dropout_prob": drop, "n_humans": None if people < 0 else people}
    if big:
        cond.update(map_width=32.0, map_height=20.0)
    mseed = map_seed(split, int(idx))
    if mission:
        cond.update(mission_controls(key, mseed, cond))
    return mseed, cond


@st.cache_data(show_spinner=False, max_entries=64)
def floor_places(mseed: int, cond_items: tuple) -> tuple:
    """(rooms, beds, default start room, default bed) of one floor, for the mission pickers."""
    env = make_env(dict(cond_items))
    env.reset(seed=0, options={"map_seed": mseed})
    lay = env.unwrapped.layout
    rooms = [r.name for r in lay.rooms]
    beds = [(r.name, b) for r in lay.rooms for b, _ in r.beds]
    return rooms, beds, lay.start_room, lay.goal_label


def pick_key() -> str:
    """Widget key of the clickable floor map. A new key after each click = a fresh, unselected chart."""
    return f"floor_pick_{st.session_state.get('pick_n', 0)}"


def apply_map_click(tag: str, rooms: list, beds: list):
    """Turn the last click on the floor map into picker values (must run before the pickers are drawn)."""
    event = st.session_state.get(pick_key())
    targets = st.session_state.get("pick_targets")
    points = (event or {}).get("selection", {}).get("points", []) if event else []
    if not points or not targets:
        return
    p = points[0]
    try:
        hit = targets[p["curve_number"]][p["point_index"]]
    except (KeyError, IndexError, TypeError):
        return
    if hit[0] == "bed" and (hit[1], hit[2]) in beds:
        st.session_state[tag + "ward"] = hit[1]
        st.session_state[tag + hit[1] + "bed"] = hit[2]
    elif hit[0] == "room" and hit[1] in rooms:
        st.session_state[tag + "start"] = hit[1]
    st.session_state["pick_n"] = st.session_state.get("pick_n", 0) + 1


def mission_controls(key: str, mseed: int, cond: dict) -> dict:
    """'Start from' + 'Deliver to' pickers for the chosen floor. Returns env overrides."""
    try:
        rooms, beds, start0, bed0 = floor_places(mseed, tuple(sorted(cond.items())))
    except Exception:  # noqa: BLE001 - an impossible floor is reported by the preview instead
        return {}
    tag = f"{key}{mseed}{cond.get('map_width')}"           # new floor -> fresh choices
    apply_map_click(tag, rooms, beds)
    st.markdown('<div class="sec">📍 Mission <span>or click the map</span></div>', unsafe_allow_html=True)
    others = sorted(r for r in rooms if r != start0)
    start = st.selectbox("Start from", [start0] + others, key=tag + "start",
                         format_func=lambda r: f"{r} (usual start)" if r == start0 else r,
                         help="Medicine normally leaves from the Pharmacy. The robots were trained only on "
                              "Pharmacy starts, so other rooms test how well they generalise.")
    wards = sorted({w for w, _ in beds})
    ward = st.selectbox("Deliver to ward", ["auto"] + wards, key=tag + "ward",
                        format_func=lambda w: f"Automatic (bed {bed0})" if w == "auto" else w,
                        help="Automatic = the bed this floor's mission normally uses (what the test results use).")
    bed = None
    if ward != "auto":
        ward_beds = [b for w, b in beds if w == ward]
        bed = st.selectbox("Bed", ward_beds, key=tag + ward + "bed")
    if bed is None and start != start0:
        bed = bed0          # "Automatic" = this floor's usual bed, even when starting from another room
    return {"start_room": None if start == start0 else start, "goal_bed": bed}


def kpi(label, value, sub="", icon="", tone="blue"):
    ic = f'<span class="ic">{icon}</span>' if icon else ""
    return (f'<div class="kpi {tone}"><div class="l">{ic}{label}</div><div class="v">{value}</div>'
            f'<div class="s">{sub}</div></div>')


ROOM_ICONS = {"Pharmacy": "💊", "Nurse station": "🩺", "Lobby": "🛋️",
              "Supply": "📦", "Corridor": "🚶"}


def place_chip(name: str, cls: str) -> str:
    icon = ROOM_ICONS.get(name.split(",")[0], "🛏️" if name.startswith("Ward") else "📍")
    return f'<span class="place {cls}">{icon} {name}</span>'


def mission_card(mission: str, sub: str, right_html: str) -> str:
    """'Pharmacy → Ward B, bed B-1' as From/To chips."""
    start, _, goal = mission.partition(" → ")
    goal = goal.replace(", bed ", " · bed ")
    return (f'<div class="card mission"><div class="route">{place_chip(start, "from")}'
            f'<span class="arrow">➜</span>{place_chip(goal, "to")}<span class="sub">{sub}</span></div>'
            f'{right_html}</div>')


# ====================================================================== tabs
tab_run, tab_cmp, tab_res, tab_about = st.tabs(["\U0001F69A  Delivery", "⚖️  Compare agents",
                                                 "\U0001F4CA  Results", "\U0001F4D8  How it works"])

# ---------------------------------------------------------------- Delivery
with tab_run:
    left, right = st.columns([1, 2.6], gap="large")
    with left, st.container(border=True):
        st.markdown('<div class="sec">🧠 Robot</div>', unsafe_allow_html=True)
        opts = agent_options()
        label = st.selectbox("Robot brain", opts, index=opts.index(default_agent(opts)), format_func=pretty,
                             help="Trained agents come from runs/part_a and runs/part_b. A* knows the map but "
                                  "cannot see people; the RL agents only see LiDAR + the goal direction.")
        n_seeds = len(trained_runs().get(label, {0: None}))
        seed = st.selectbox("Training run (seed)", sorted(trained_runs().get(label, {0: None})), disabled=n_seeds < 2,
                            help="Each algorithm was trained 3 times with different random seeds.")
        st.markdown('<div class="sec">🏥 Floor</div>', unsafe_allow_html=True)
        mseed, cond = floor_controls("run", mission=True)
        show_rays = st.toggle("Show LiDAR rays", value=True)
        go_btn = st.button("\U0001F69A  Start delivery", type="primary", width="stretch")
        if len(opts) <= 2:
            st.info("No trained models yet: only the baselines are available. Run `Run Smoke Test.cmd` or the "
                    "Part A pipeline first.")
    with right:
        if go_btn:
            agent, meta = safe_agent(label, seed)
            if agent is not None:
                try:
                    env = env_for(meta, cond)
                    with st.spinner("Driving…"):
                        row, frames = run_episode(agent, env, mseed, 0, record=True)
                    st.session_state["run"] = dict(row=row, frames=frames, label=label, seed=seed, setup=(mseed, cond))
                except Exception as exc:  # noqa: BLE001 - e.g. an impossible stress-test combination
                    st.error(f"The delivery could not run: {type(exc).__name__}: {exc}")
        res = st.session_state.get("run")
        if res is not None and res.get("setup") != (mseed, cond):
            res = None                       # settings changed since the last run: preview the new mission
        if res is None:
            # preview the selected floor so the page is never empty
            try:
                env = make_env(cond)
                env.reset(seed=0, options={"map_seed": mseed})
                snap = env.unwrapped.state_snapshot()
                st.markdown(mission_card(snap["mission"], "floor preview",
                                         '<div class="chip idle">Ready to go</div>'), unsafe_allow_html=True)
                st.markdown('<div class="hint">👆 <b>Click a bed</b> to deliver there &nbsp;·&nbsp; '
                            '<b>click inside a room</b> to start from it &nbsp;·&nbsp; then press '
                            '<b>Start delivery</b></div>', unsafe_allow_html=True)
                fig, targets = pick_figure(snap, height=560)
                st.session_state["pick_targets"] = targets
                st.plotly_chart(fig, width="stretch", key=pick_key(), on_select="rerun",
                                selection_mode="points", config={"displayModeBar": False, "scrollZoom": False})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not build this floor: {exc}")
        else:
            row, frames = res["row"], res["frames"]
            text, kind = OUTCOMES.get(row["failure_type"], (row["failure_type"], "warn"))
            st.markdown(mission_card(frames[0]["mission"], pretty(res["label"]),
                                     f'<div class="chip {kind}">{"✔ " if kind == "ok" else ""}{text}</div>'),
                        unsafe_allow_html=True)
            dt = frames[0].get("dt", 0.2)
            eff_ok = bool(row["success"])
            eff = min(1.0, row["optimal_length"] / max(row["path_length"], 1e-6)) if row["success"] else np.nan
            closest = row["min_human_dist"]
            st.markdown('<div class="kpis">' +
                        kpi("Time", f"{row['steps'] * dt:.1f} s", f"{row['steps']} decisions", "⏱️", "blue") +
                        kpi("Distance driven", f"{row['path_length']:.1f} m",
                            f"shortest route {row['optimal_length']:.1f} m", "📏", "teal") +
                        kpi("Route efficiency", "—" if np.isnan(eff) else f"{eff:.0%}", "vs A* route", "🧭",
                            "green" if eff_ok else "grey") +
                        kpi("Closest person", "—" if np.isnan(closest) else f"{closest:.2f} m", "body to body",
                            "🧍", "pink") +
                        kpi("In comfort zones", f"{row['social_intrusion_rate']:.0%}", "of the time, < 0.7 m",
                            "🛡️", "amber") +
                        "</div>", unsafe_allow_html=True)
            st.plotly_chart(episode_animation(frames, height=560, show_rays=show_rays), width="stretch",
                            config={"displayModeBar": False, "scrollZoom": False})
            st.markdown('<div class="legend"><span><span class="dot" style="background:#1f6feb"></span>robot + path'
                        '</span><span><span class="dot" style="background:#d6336c"></span>person (dotted = comfort '
                        'zone)</span><span><span class="dot" style="background:#e8590c"></span>target bed</span>'
                        '<span>Rx = start point</span></div>',
                        unsafe_allow_html=True)
            if st.button("🗺️  Pick another bed on the map"):
                st.session_state.pop("run", None)
                st.rerun()
            a, b = st.columns(2)
            with a.expander("Route through the building", expanded=True):
                st.markdown('<div class="rooms">' + " → ".join(rooms_visited(frames)) + "</div>",
                            unsafe_allow_html=True)
            with b.expander("Where the reward came from"):
                terms = {k[2:]: v for k, v in row.items() if k.startswith("r_")}
                st.bar_chart(pd.Series(terms, name="reward"), height=180, color="#1f6feb")

# ---------------------------------------------------------------- Compare
with tab_cmp:
    st.markdown("##### Same floor, different robot brains")
    opts = agent_options()
    defaults = [o for o in ("astar", default_agent(opts), "dueling_dqn", "ppo") if o in opts]
    picks = st.multiselect("Agents", opts, default=list(dict.fromkeys(defaults))[:4], format_func=pretty,
                           max_selections=6)
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1])
    mseed_c, cond_c = floor_controls("cmp", [c1, c2, c3])
    n_floors = c4.number_input("Also score over N floors", 1, 50, 10,
                               help="Success rate over this many consecutive floors (same for every agent).")
    if st.button("⚖️  Run comparison", type="primary") and picks:
        runs, table = [], []
        snap0 = None
        prog = st.progress(0.0, text="Driving…")
        for k, lab in enumerate(picks):
            agent, meta = safe_agent(lab, 0)
            if agent is None:
                continue
            succ, coll, times = [], [], []
            for j in range(int(n_floors)):
                env = env_for(meta, cond_c)
                row, frames = run_episode(agent, env, mseed_c + j, 0, record=(j == 0))
                if j == 0:
                    snap0 = snap0 or frames[0]
                    runs.append((pretty(lab), np.asarray(frames[-1]["path"]),
                                 OUTCOMES.get(row["failure_type"], (row["failure_type"], ""))[0], row["success"]))
                succ.append(row["success"])
                coll.append(row["collision"])
                if row["success"]:
                    times.append(row["steps"] * 0.2)
            table.append({"Agent": pretty(lab), "Success rate": float(np.mean(succ)),
                          "Collision rate": float(np.mean(coll)),
                          "Avg. delivery time (s)": float(np.mean(times)) if times else np.nan})
            prog.progress((k + 1) / len(picks), text=f"Driving… {pretty(lab)} done")
        prog.empty()
        st.session_state["cmp"] = (snap0, runs, table, int(n_floors))
    if "cmp" in st.session_state and st.session_state["cmp"][0] is not None:
        snap0, runs, table, n = st.session_state["cmp"]
        a, b = st.columns([2.2, 1], gap="large")
        with a:
            st.markdown(f'<div class="card mission"><div class="route">{snap0["mission"]}'
                        f' <span>· first floor of {n}</span></div></div>', unsafe_allow_html=True)
            st.plotly_chart(paths_figure(snap0, runs, height=540), width="stretch",
                            config={"displayModeBar": False, "scrollZoom": False})
        with b:
            st.markdown(f"**Over {n} floor(s)**")
            st.dataframe(pd.DataFrame(table).sort_values("Success rate", ascending=False), hide_index=True,
                         width="stretch", column_config={
                             "Success rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                             "Collision rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                             "Avg. delivery time (s)": st.column_config.NumberColumn(format="%.1f")})
            st.caption("● = delivered, ✕ = failed. For the official numbers (50 test floors × 3 seeds, "
                       "with confidence intervals) see the Results tab.")

# ---------------------------------------------------------------- Results
with tab_res:
    exps = [e for e in CURRENT_EXPERIMENTS if (RUNS / e / "results" / "summary.csv").exists()]
    archived = sorted(p.parent.relative_to(RUNS).as_posix() for p in RUNS.glob("env_*/*/results")
                      if (p / "summary.csv").exists())
    if not exps and not archived:
        st.info("No results yet. Run the Part A / Part B pipelines first (see README).")
    else:
        names = {"part_a": "Part A — shortlist models", "part_b": "Part B — refined models & ensembles",
                 "smoke": "Smoke test (installation check)"}
        exp = st.selectbox("Experiment", exps + archived,
                           format_func=lambda e: names.get(e, f"Archive: {e.replace('env_v1', 'env v1 (10 m room, careless people)').replace('env_v2_room', 'env v2 (10 m room)')}"),
                           help="Archives keep the results of earlier environment versions (the iteration story).")
        res = RUNS / exp / "results"
        summ = pd.read_csv(res / "summary.csv")
        learned = summ[~summ["agent"].isin(["astar", "random"])]
        best = learned.iloc[0] if len(learned) else summ.iloc[0]
        astar = summ[summ["agent"] == "astar"]
        # "safe" only counts if the agent actually delivers (an agent that never leaves the pharmacy never crashes)
        useful = learned[learned["success"] >= 0.5]
        safest = useful.sort_values("collision").iloc[0] if len(useful) else best
        n_eps = int(summ["n_episodes"].max()) if "n_episodes" in summ else 0
        st.markdown('<div class="kpis">' +
                    kpi("Best agent", pretty(best["agent"]), f"{best['success']:.0%} delivered") +
                    kpi("95% confidence", f"{best.get('success_ci_low', np.nan):.0%}–{best.get('success_ci_high', np.nan):.0%}",
                        "success of the best agent") +
                    kpi("A* baseline", f"{astar['success'].iloc[0]:.0%}" if len(astar) else "—",
                        f"collides {astar['collision'].iloc[0]:.0%}" if len(astar) else "") +
                    kpi("Safest agent", pretty(safest["agent"]),
                        f"collides {safest['collision']:.0%} · of agents delivering ≥ 50%") +
                    kpi("Agents compared", f"{len(summ)}", f"up to {n_eps} test deliveries each") +
                    "</div>", unsafe_allow_html=True)

        t_lb, t_ov, t_rb, t_tr, t_pb, t_int, t_fs = st.tabs(
            ["Leaderboard", "Overview", "Robustness", "Training & tuning", "Part B refinements",
             "Interpretability", "Member fact sheets"])
        with t_lb:
            show = summ.copy()
            show.insert(0, "Agent", show["agent"].map(pretty))
            cols = {"success": "Success", "collision": "Collision", "collision_human": "Hit a person",
                    "timeout": "Timeout", "path_efficiency": "Route efficiency", "time_to_goal_s": "Time (s)",
                    "social_intrusion_rate": "In comfort zones", "success_seed_std": "Seed std", "n_seeds": "Seeds"}
            show = show[["Agent"] + [c for c in cols if c in show]].rename(columns=cols)
            for c in ("Success", "Collision", "Hit a person", "Timeout", "Route efficiency", "In comfort zones"):
                if c in show:
                    show[c] = show[c].round(2)          # whole percents: 0.4867 -> 49%
            pct = dict(format="percent", min_value=0, max_value=1)
            st.dataframe(show, hide_index=True, width="stretch", height=min(38 * len(show) + 40, 720),
                         column_config={"Success": st.column_config.ProgressColumn(**pct),
                                        "Collision": st.column_config.ProgressColumn(**pct),
                                        "Hit a person": st.column_config.NumberColumn(format="percent"),
                                        "Timeout": st.column_config.NumberColumn(format="percent"),
                                        "Route efficiency": st.column_config.NumberColumn(format="percent"),
                                        "Time (s)": st.column_config.NumberColumn(format="%.1f"),
                                        "In comfort zones": st.column_config.NumberColumn(format="percent"),
                                        "Seed std": st.column_config.NumberColumn(format="%.3f")})
            if (res / "significance.md").exists():
                st.markdown("**Is the gap between the top agents real?** Permutation test on per-delivery success:")
                st.markdown((res / "significance.md").read_text())

        def show_png(name, caption=None):
            p = res / name
            if p.exists():
                st.image(str(p), caption=caption, width="stretch")
                return True
            return False

        with t_ov:
            show_png("comparison.png", "Success, collisions, route efficiency and comfort-zone time (95% CI)")
            show_png("failure_breakdown.png", "How each agent's test deliveries end")
            show_png("trajectories.png", "Top agents and A* on the same test floors")
        with t_rb:
            st.caption("Success rate when the hospital is harder than during training. Darker = better.")
            show_png("robustness_success.png")
            show_png("robustness_collision.png")
        with t_tr:
            show_png("learning_curves.png", "Validation success during training (mean ± std over seeds)")
            if (res / "sample_efficiency.csv").exists():
                st.dataframe(pd.read_csv(res / "sample_efficiency.csv").round(3), hide_index=True,
                             width="stretch")
            tuning = sorted((RUNS / "tuning").glob("tuning_*.png")) if exp in CURRENT_EXPERIMENTS else []
            if tuning:
                st.markdown("**Hyper-parameter search: TPE vs random search** (best score so far per trial)")
                cs = st.columns(2)
                for i, p in enumerate(tuning):
                    cs[i % 2].image(str(p), width="stretch")
        with t_pb:
            if not show_png("refinements.png"):
                st.info("Refinement table is produced for Part B.")
            if (res / "refinements.md").exists():
                st.markdown((res / "refinements.md").read_text())
        with t_int:
            agents = sorted({p.stem.split("_", 1)[1] for p in res.glob("importance_*.png")}, key=pretty)
            if agents:
                pick = st.selectbox("Agent", agents, format_func=pretty, key="int_agent")
                a, b = st.columns(2)
                with a:
                    show_png(f"importance_{pick}.png", "Success drop when one input group is scrambled")
                with b:
                    if not show_png(f"saliency_{pick}.png", "Which LiDAR directions drive the steering"):
                        st.caption("No saliency map (tabular, recurrent or frame-stacked agents).")
        with t_fs:
            sheets = sorted((res / "factsheets").glob("*.md")) if (res / "factsheets").exists() else []
            if sheets:
                pick = st.selectbox("Member", sheets, format_func=lambda p: p.stem.replace("_", " ").title(),
                                    key="fs_pick")
                st.markdown(pick.read_text(encoding="utf-8"))
            else:
                st.info("Run `python -m medinav factsheet --config ...` to create the fact sheets.")

# ---------------------------------------------------------------- How it works
with tab_about:
    doc = ROOT / "docs" / "how_it_works.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "See README.md")

st.markdown('<div class="foot">MediNav · simulated hospital wing, trained agents from runs/part_a and runs/part_b · '
            'BMDS2114 Machine Learning</div>', unsafe_allow_html=True)
