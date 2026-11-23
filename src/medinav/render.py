"""Matplotlib drawing of the hospital world (used by the app and report figures)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrow  # noqa: E402

COLORS = {
    "wall": "#2b2f36",
    "floor": "#f4f1ea",
    "robot": "#1f6feb",
    "goal": "#2da44e",
    "human": "#d1495b",
    "social": "#d1495b",
    "ray": "#1f6feb",
    "path": "#1f6feb",
    "waypoint": "#bf8700",
}


def draw_world(ax, snap: dict, show_rays=True, show_social=True, trail_color=None, label=None):
    g = snap["grid"]
    res = snap["resolution"]
    size = g.shape[1] * res
    ax.imshow(
        g, cmap=matplotlib.colors.ListedColormap([COLORS["floor"], COLORS["wall"]]),
        origin="lower", extent=(0, size, 0, size), interpolation="nearest",
    )
    goal = snap["goal"]
    ax.plot(goal[0], goal[1], marker="*", markersize=16, color=COLORS["goal"], zorder=5)
    for hp in snap["humans"]:
        ax.add_patch(Circle(hp, snap["human_radius"], color=COLORS["human"], zorder=4))
        if show_social:
            ax.add_patch(Circle(hp, snap["human_radius"] + 0.7, fill=False, ls=":", color=COLORS["social"], alpha=0.6))
    pos, hd = snap["pos"], snap["heading"]
    if show_rays and snap.get("lidar") is not None:
        for ang, rng in zip(snap["ray_angles"], snap["lidar"]):
            ax.plot([pos[0], pos[0] + rng * np.cos(ang)], [pos[1], pos[1] + rng * np.sin(ang)],
                    color=COLORS["ray"], alpha=0.18, lw=0.8)
    path = snap.get("path")
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], color=trail_color or COLORS["path"], lw=2, label=label)
    if snap.get("waypoint") is not None:
        ax.plot(*snap["waypoint"], marker="D", color=COLORS["waypoint"], markersize=6)
    ax.add_patch(Circle(pos, snap["robot_radius"], color=COLORS["robot"], zorder=6))
    ax.add_patch(FancyArrow(pos[0], pos[1], 0.35 * np.cos(hd), 0.35 * np.sin(hd),
                            width=0.05, color="white", zorder=7))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_frame_rgb(snap: dict, title: str | None = None) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
    draw_world(ax, snap)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
