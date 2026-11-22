"""Four active demo presets, constructed independently of the held-out file."""

from .partial_dynamic_env import make_scenario

NAMES = (
    "New static maze",
    "Permanent dead end with detour",
    "Temporary bridge - Wait",
    "Temporary gate - detour",
)


def scenario(name, mode="Temporary", start=None, goal=None):
    a = [list("#" * 10) for _ in range(10)]
    events = []
    if name == NAMES[0]:
        for row in (1, 3, 5, 7):
            for c in range(1, 9):
                a[row][c] = "."
        for r, c in ((2, 8), (4, 1), (6, 8)):
            a[r][c] = "."
        default_start = [1, 1]
        default_goal = [7, 1]
    elif name == NAMES[1]:
        for r in range(1, 9):
            for c in range(1, 9):
                a[r][c] = "."
        for row in range(1, 8):
            a[row][4] = "#"
        for c in range(2, 4):
            a[3][c] = "#"
        default_start = [2, 3]
        default_goal = [2, 7]
    elif name == NAMES[2]:
        for c in range(1, 9):
            a[5][c] = "."
        for r in (4, 6):
            for c in (7, 8):
                a[r][c] = "."
        default_start = [5, 1]
        default_goal = [5, 8]
        events = [{"id": "demo-pallet", "cell": [5, 2], "start": 0, "clear": 6}]
    elif name == NAMES[3]:
        for r in range(1, 9):
            for c in range(1, 9):
                a[r][c] = "."
        for r in (3, 4):
            for c in (4, 5):
                a[r][c] = "#"
        default_start = [5, 1]
        default_goal = [5, 8]
        events = [{"id": "demo-gate", "cell": [5, 3], "start": 0, "clear": 20}]
    else:
        raise ValueError("Unknown Phase3 demo preset")
    if mode not in ("Static", "Temporary"):
        raise ValueError("Static or Temporary required")
    case = make_scenario(
        ["".join(row) for row in a],
        list(start or default_start),
        list(goal or default_goal),
        events if mode == "Temporary" else [],
        scenario_id="phase3-demo-" + str(NAMES.index(name)) + "-" + mode.lower(),
        layout_id="phase3-demo-layout-" + str(NAMES.index(name)),
        event_seed=93600 + NAMES.index(name),
        split="manual",
        condition="phase3-demonstration",
    )
    case.update(
        product_scope="demo/development; never frozen heldout",
        topology_family="active-demo-" + str(NAMES.index(name)),
    )
    return case
