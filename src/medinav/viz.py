"""Interactive (Plotly) floor-plan views for the app.

Why Plotly and not matplotlib frames?
    Re-drawing a matplotlib figure per step is slow (~80 ms) and flickers in a browser.
    Here the static floor plan is rendered ONCE (render.py) and used as the background
    image; only the moving parts (robot, people, LiDAR, path) are animated, client-side,
    so playback is smooth, zoomable and needs no round-trip to Python.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from .render import COLORS, draw_world  # noqa: E402

PATH_COLORS = ["#1f6feb", "#e8590c", "#2f9e44", "#7048e8", "#d6336c", "#495057"]


def _circle(xy, r, n=24):
    t = np.linspace(0, 2 * np.pi, n)
    return xy[0] + r * np.cos(t), xy[1] + r * np.sin(t)


def floor_background(snap: dict, dpi: int = 160) -> str:
    """Static floor plan (rooms, walls, furniture, beds, labels, goal) as a base64 PNG data URI."""
    static = dict(snap)
    static.update(humans=[], path=None, lidar=None, waypoint=None)
    W = snap["grid"].shape[1] * snap["resolution"]
    H = snap["grid"].shape[0] * snap["resolution"]
    fig = plt.figure(figsize=(W / 2.5, H / 2.5), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    draw_world(ax, static, show_rays=False, show_social=False, show_robot=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _base_figure(snap, height):
    W = snap["grid"].shape[1] * snap["resolution"]
    H = snap["grid"].shape[0] * snap["resolution"]
    fig = go.Figure()
    fig.add_layout_image(dict(source=floor_background(snap), xref="x", yref="y", x=0, y=H, sizex=W, sizey=H,
                              sizing="stretch", layer="below"))
    fig.update_xaxes(range=[0, W], visible=False, constrain="domain", fixedrange=True)
    fig.update_yaxes(range=[0, H], visible=False, scaleanchor="x", scaleratio=1, fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False, dragmode=False)
    return fig


def _dynamic_traces(snap, path_upto, show_rays):
    """Traces that change every frame. Same count/order in every frame (Plotly requirement)."""
    traces = []
    # LiDAR rays as one polyline with gaps
    pos, hd = snap["pos"], snap["heading"]
    if show_rays and snap.get("lidar") is not None:
        xs, ys = [], []
        for ang, rng in zip(snap["ray_angles"], snap["lidar"]):
            xs += [pos[0], pos[0] + rng * np.cos(ang), None]
            ys += [pos[1], pos[1] + rng * np.sin(ang), None]
    else:
        xs, ys = [None], [None]
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=COLORS["ray"], width=1), opacity=0.22,
                             hoverinfo="skip"))
    # path so far
    p = path_upto
    traces.append(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=COLORS["path"], width=3),
                             hoverinfo="skip"))
    # people: comfort zone + body (one trace each, polygons with gaps)
    zx, zy, bx, by = [], [], [], []
    for h in snap["humans"]:
        x, y = _circle(h, snap["human_radius"] + 0.7)
        zx += list(x) + [None]
        zy += list(y) + [None]
        x, y = _circle(h, snap["human_radius"], 16)
        bx += list(x) + [None]
        by += list(y) + [None]
    traces.append(go.Scatter(x=zx or [None], y=zy or [None], mode="lines", fill="toself",
                             fillcolor="rgba(214,51,108,0.08)", line=dict(color="rgba(214,51,108,0.45)",
                                                                           width=1, dash="dot"),
                             hoverinfo="skip"))
    traces.append(go.Scatter(x=bx or [None], y=by or [None], mode="lines", fill="toself",
                             fillcolor=COLORS["human"], line=dict(color="white", width=1), hoverinfo="skip"))
    # robot body + heading triangle
    r = snap["robot_radius"]
    x, y = _circle(pos, r, 20)
    traces.append(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=COLORS["robot"],
                             line=dict(color=COLORS["robot_edge"], width=1.5), hoverinfo="skip"))
    tip = pos + 0.95 * r * np.array([np.cos(hd), np.sin(hd)])
    left = pos + 0.45 * r * np.array([np.cos(hd + 2.3), np.sin(hd + 2.3)])
    right = pos + 0.45 * r * np.array([np.cos(hd - 2.3), np.sin(hd - 2.3)])
    traces.append(go.Scatter(x=[tip[0], left[0], right[0], tip[0]], y=[tip[1], left[1], right[1], tip[1]],
                             mode="lines", fill="toself", fillcolor="white", line=dict(color="white", width=0),
                             hoverinfo="skip"))
    return traces


def episode_animation(frames: list, height: int = 560, show_rays: bool = True, max_frames: int = 160):
    """Animated replay of one recorded episode (frames = list of env.state_snapshot())."""
    fig = _base_figure(frames[0], height)
    full_path = np.asarray(frames[-1]["path"])
    stride = max(1, int(np.ceil(len(frames) / max_frames)))
    idx = list(range(0, len(frames), stride))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    dt = frames[0].get("dt", 0.2)
    for tr in _dynamic_traces(frames[idx[0]], full_path[:1], show_rays):
        fig.add_trace(tr)
    fig.frames = [go.Frame(data=_dynamic_traces(frames[i], full_path[: i + 1], show_rays), name=str(i))
                  for i in idx]
    steps = [dict(method="animate", label=f"{i * dt:.0f}s" if k % 10 == 0 else "",
                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for k, i in enumerate(idx)]
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top",
                          pad=dict(t=4, r=8), showactive=False, bgcolor="#ffffff", bordercolor="#d0d7de",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0),
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=len(idx) - 1, steps=steps, x=0.16, len=0.84, y=-0.02, yanchor="top",
                      pad=dict(t=4), currentvalue=dict(visible=False), ticklen=0,
                      bgcolor="#e9ecef", activebgcolor=COLORS["robot"], bordercolor="#d0d7de")],
        margin=dict(l=0, r=0, t=0, b=48),
    )
    # start the view on the LAST frame (the finished delivery), Play replays from the start
    for k, tr in enumerate(_dynamic_traces(frames[idx[-1]], full_path, show_rays)):
        fig.data[k].update(tr)
    return fig


def pick_figure(snap: dict, height: int = 560):
    """Floor preview you can click on. Returns (figure, targets).

    Plotly can only report clicks on data points, so every bed gets a column of invisible
    points ("deliver here") and every room floor a sparse grid of them ("start from here").
    targets[curve][point_index] = ("bed", ward, label) or ("room", name), so the app can map
    a click event (curve_number, point_index) back to a choice.
    """
    fig = _base_figure(snap, height)
    room_x, room_y, room_t, room_h = [], [], [], []
    bed_x, bed_y, bed_t, bed_h = [], [], [], []
    bed_boxes = []                                          # bed bodies (+ margin): no "start" targets there
    for room in snap.get("rooms") or []:
        for _, (bx, by) in room.beds:
            toward = 1 if by < (room.rect[1] + room.rect[3]) / 2 else -1
            ya, yb = by - toward * 0.3, by - toward * 2.8
            bed_boxes.append((bx - 0.75, min(ya, yb), bx + 0.75, max(ya, yb)))
    for room in snap.get("rooms") or []:
        x0, y0, x1, y1 = room.rect
        for x in np.arange(x0 + 0.4, x1 - 0.2, 0.7):
            for y in np.arange(y0 + 0.4, y1 - 0.2, 0.7):
                if any(a <= x <= c and b <= y <= d for a, b, c, d in bed_boxes):
                    continue
                room_x.append(x)
                room_y.append(y)
                room_t.append(("room", room.name))
                room_h.append(f"<b>{room.name}</b><br>click: start the robot here")
        for label, (bx, by) in room.beds:
            toward = 1 if by < (y0 + y1) / 2 else -1
            for d in np.linspace(0.75, 2.45, 5):            # along the bed body
                bed_x.append(bx)
                bed_y.append(by - toward * d)
                bed_t.append(("bed", room.name, label))
                bed_h.append(f"<b>{room.name} · bed {label}</b><br>click: deliver the medicine here")
    clear = dict(color="rgba(0,0,0,0)", size=22, line=dict(width=0))
    hover = dict(bgcolor="#ffffff", bordercolor="#d0d7de", font=dict(family="Inter, sans-serif", size=12,
                                                                     color="#1f2933"))
    fig.add_trace(go.Scatter(x=room_x, y=room_y, mode="markers", marker=clear, hovertext=room_h,
                             hoverinfo="text", hoverlabel=hover, name="rooms"))
    fig.add_trace(go.Scatter(x=bed_x, y=bed_y, mode="markers", marker=dict(clear, size=26), hovertext=bed_h,
                             hoverinfo="text", hoverlabel=hover, name="beds"))
    # visible "you are here" pin on the start point
    if snap.get("start") is not None:
        sx, sy = snap["start"]
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers+text", text=["START"], textposition="top center",
                                 textfont=dict(size=10, color="#0b3d91", family="Inter, sans-serif"),
                                 marker=dict(symbol="circle", size=13, color="#1f6feb",
                                             line=dict(color="white", width=2)),
                                 hoverinfo="skip"))
    # dragmode must not be False here: plotly then swallows clicks and never emits a selection.
    # "pan" with fixed axes = the map does not move, but a click selects the nearest target.
    fig.update_layout(clickmode="event+select", hovermode="closest", dragmode="pan")
    return fig, [room_t, bed_t]


def paths_figure(snap0: dict, runs: list, height: int = 560):
    """Several agents' finished paths on the same floor. runs = [(name, path array, outcome, success)]."""
    fig = _base_figure(snap0, height)
    for k, (name, path, outcome, success) in enumerate(runs):
        c = PATH_COLORS[k % len(PATH_COLORS)]
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color=c, width=3),
                                 name=f"{name} — {outcome}", hovertemplate=f"{name}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[path[-1, 0]], y=[path[-1, 1]], mode="markers",
                                 marker=dict(symbol="circle" if success else "x", size=12, color=c,
                                             line=dict(color="white", width=1.5)),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.02, yanchor="top", x=0,
                                                   font=dict(size=12), bgcolor="rgba(255,255,255,0.8)"))
    return fig
