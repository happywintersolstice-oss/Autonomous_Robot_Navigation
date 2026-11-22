"""Shared Matplotlib renderer; only the displayed trajectory prefix is visible."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from .episodes import frame


def draw(grid, episode=None, index=0):
    fig, ax = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    fig.patch.set_facecolor("#f7fafc")
    blocked = np.array([[v == "#" for v in row] for row in grid.rows])
    ax.imshow(blocked, cmap=ListedColormap(["#f7fafc", "#334155"]), vmin=0, vmax=1)
    h, w = grid.shape
    ax.set(xticks=range(w), yticks=range(h), xlabel="Column", ylabel="Row")
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    title = grid.map_id
    if episode is not None:
        state = frame(episode, index)
        trail = np.array(state["trail"])
        ax.plot(trail[:, 1], trail[:, 0], color="#0891b2", linewidth=3, alpha=.65, marker=".")
        start, goal = episode["scenario"]["start"], episode["scenario"]["goal"]
        for position, label, color in ((start, "S", "#0369a1"), (goal, "G", "#15803d")):
            ax.add_patch(Rectangle((position[1] - .42, position[0] - .42), .84, .84,
                                   fill=False, edgecolor=color, linewidth=2.5))
            ax.text(position[1] - .25, position[0] - .23, label, color=color, weight="bold", fontsize=10)
        r, c = state["position"]
        ax.scatter([c], [r], s=180, color="#f59e0b", edgecolors="#78350f", zorder=5)
        ax.text(c, r + .08, "R", ha="center", fontsize=9, weight="bold", zorder=6)
        if state["status"] == "collision":
            ar, ac = state["attempted_position"]
            ax.annotate("", xy=(ac, ar), xytext=(c, r),
                        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 3})
            ax.scatter([ac], [ar], marker="x", s=180, color="#dc2626", zorder=7, clip_on=False)
        title += f" · action {state['index']} · {state['status']}"
    ax.set_xlim(-1, w)
    ax.set_ylim(h, -1)
    ax.set_title(title, fontsize=12, loc="left", pad=14)
    fig.supxlabel("S Start   G Goal   R Robot   ■ Wall   — Traveled trail   × Collision attempt",
                  fontsize=8, color="#334155")
    return fig


def save_figure(path, grid, episode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = draw(grid, episode, len(episode["transitions"]) if episode else 0)
    figure.savefig(path, dpi=140)
    plt.close(figure)
