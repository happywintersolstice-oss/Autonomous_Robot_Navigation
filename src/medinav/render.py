"""Floor-plan drawing of the hospital world (used by the app and the report figures).

The occupancy grid only knows "blocked / free". For a readable plan we draw it in layers:
room floors tinted by room type -> walls (dark) -> furniture (light grey, beds white) ->
room names and bed numbers -> people with their comfort zone -> robot, LiDAR and path.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

# Light, clinical palette. Room tints are pale so the robot, people and path stay the focus.
COLORS = {
    "background": "#ffffff",
    "corridor": "#f3f5f7",
    "wall": "#2a3441",
    "furniture": "#c9d1da",
    "bed": "#ffffff",
    "bed_edge": "#9aa6b2",
    "robot": "#1f6feb",
    "robot_edge": "#0b3d91",
    "goal": "#e8590c",
    "human": "#d6336c",
    "social": "#d6336c",
    "ray": "#1f6feb",
    "path": "#1f6feb",
    "waypoint": "#b58100",
    "label": "#3b4652",
    "muted": "#7b8794",
}
ROOM_TINT = {
    "pharmacy": "#e3effd",   # blue
    "ward": "#e6f4ea",       # green
    "nurses": "#fff4dc",     # amber
    "lobby": "#f1ebfb",      # lilac
    "supply": "#eeeeec",     # grey
}


def _grid_image(snap):
    """RGBA image: walls dark, furniture inside rooms/corridor light grey, free cells transparent."""
    g = snap["grid"]
    res = snap["resolution"]
    h, w = g.shape
    img = np.zeros((h, w, 4))
    inside = np.zeros_like(g)
    rects = [r.rect for r in snap.get("rooms") or []]
    if snap.get("corridor"):
        rects.append(snap["corridor"])
    for x0, y0, x1, y1 in rects:
        inside[int(round(y0 / res)):int(round(y1 / res)), int(round(x0 / res)):int(round(x1 / res))] = True
    wall_rgb = matplotlib.colors.to_rgb(COLORS["wall"])
    furn_rgb = matplotlib.colors.to_rgb(COLORS["furniture"])
    walls = g & ~inside
    furn = g & inside
    img[walls] = (*wall_rgb, 1.0)
    img[furn] = (*furn_rgb, 1.0)
    return img


def draw_world(ax, snap: dict, show_rays=True, show_social=True, trail_color=None, label=None,
               show_labels=True, show_robot=True):
    g = snap["grid"]
    res = snap["resolution"]
    W, H = g.shape[1] * res, g.shape[0] * res
    ax.set_facecolor(COLORS["background"])

    # 1. floors
    if snap.get("corridor"):
        x0, y0, x1, y1 = snap["corridor"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=COLORS["corridor"], lw=0, zorder=0))
    for room in snap.get("rooms") or []:
        x0, y0, x1, y1 = room.rect
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=ROOM_TINT.get(room.kind, "#f5f5f5"),
                               lw=0, zorder=0))
    # 2. walls + furniture from the grid
    ax.imshow(_grid_image(snap), origin="lower", extent=(0, W, 0, H), interpolation="nearest", zorder=1)
    # 3. beds drawn over their grid cells, numbered; target bed highlighted
    for room in snap.get("rooms") or []:
        for bed_label, pt in room.beds:
            bx, by = pt
            target = room.name == snap.get("goal_room") and bed_label == snap.get("goal_label")
            # bed body: 0.9 m wide, 2.0 m long, ending 0.6 m before the delivery point
            towards_wall = 1 if by < (room.rect[1] + room.rect[3]) / 2 else -1
            y_foot = by - towards_wall * 0.6
            y_head = y_foot - towards_wall * 2.0
            ax.add_patch(FancyBboxPatch((bx - 0.45, min(y_foot, y_head)), 0.9, 2.0,
                                        boxstyle="round,pad=0,rounding_size=0.12",
                                        fc="#fff3e8" if target else COLORS["bed"],
                                        ec=COLORS["goal"] if target else COLORS["bed_edge"],
                                        lw=1.6 if target else 0.8, zorder=2))
            if show_labels:
                ax.text(bx, (y_foot + y_head) / 2, bed_label, ha="center", va="center", fontsize=5.5,
                        color=COLORS["goal"] if target else COLORS["muted"], rotation=90, zorder=3,
                        fontweight="bold" if target else "normal")
    # 4. room names
    if show_labels:
        for room in snap.get("rooms") or []:
            x0, y0, x1, y1 = room.rect
            top_half = (y0 + y1) / 2 > H / 2
            ty = y1 - 0.35 if top_half else y0 + 0.35
            ax.text(x0 + 0.3, ty, room.name.upper(), ha="left", va="top" if top_half else "bottom",
                    fontsize=6.5, color=COLORS["label"], fontweight="bold", zorder=3, alpha=0.85)
    # 5. goal and start markers
    goal = snap["goal"]
    ax.add_patch(Circle(goal, 0.32, fc="none", ec=COLORS["goal"], lw=1.4, ls="--", zorder=4))
    ax.plot(goal[0], goal[1], marker="*", markersize=11, color=COLORS["goal"], zorder=5,
            markeredgecolor="white", markeredgewidth=0.6)
    if snap.get("start") is not None:
        sx, sy = snap["start"]
        ax.text(sx, sy, "Rx", ha="center", va="center", fontsize=6, color=COLORS["robot_edge"], zorder=3,
                fontweight="bold", alpha=0.6)
    # 6. people with comfort zone
    for hp in snap["humans"]:
        if show_social:
            ax.add_patch(Circle(hp, snap["human_radius"] + 0.7, fc=COLORS["social"], alpha=0.07, lw=0, zorder=4))
            ax.add_patch(Circle(hp, snap["human_radius"] + 0.7, fill=False, ls=":", lw=0.7,
                                color=COLORS["social"], alpha=0.5, zorder=4))
        ax.add_patch(Circle(hp, snap["human_radius"], fc=COLORS["human"], ec="white", lw=0.8, zorder=5))
    # 7. LiDAR, path, robot
    pos, hd = snap["pos"], snap["heading"]
    if show_robot and show_rays and snap.get("lidar") is not None:
        for ang, rng in zip(snap["ray_angles"], snap["lidar"]):
            ax.plot([pos[0], pos[0] + rng * np.cos(ang)], [pos[1], pos[1] + rng * np.sin(ang)],
                    color=COLORS["ray"], alpha=0.16, lw=0.7, zorder=5)
    path = snap.get("path")
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], color=trail_color or COLORS["path"], lw=2, label=label, zorder=6,
                solid_capstyle="round")
    if snap.get("waypoint") is not None:
        ax.plot(*snap["waypoint"], marker="D", color=COLORS["waypoint"], markersize=4, zorder=6)
    if show_robot:
        r = snap["robot_radius"]
        ax.add_patch(Circle(pos, r, fc=COLORS["robot"], ec=COLORS["robot_edge"], lw=1, zorder=7))
        tip = pos + 0.95 * r * np.array([np.cos(hd), np.sin(hd)])
        left = pos + 0.45 * r * np.array([np.cos(hd + 2.3), np.sin(hd + 2.3)])
        right = pos + 0.45 * r * np.array([np.cos(hd - 2.3), np.sin(hd - 2.3)])
        ax.add_patch(Polygon([tip, left, right], closed=True, color="white", zorder=8))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_frame_rgb(snap: dict, title: str | None = None) -> np.ndarray:
    W = snap["grid"].shape[1] * snap["resolution"]
    H = snap["grid"].shape[0] * snap["resolution"]
    fig, ax = plt.subplots(figsize=(7, 7 * H / W), dpi=110)
    draw_world(ax, snap)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
