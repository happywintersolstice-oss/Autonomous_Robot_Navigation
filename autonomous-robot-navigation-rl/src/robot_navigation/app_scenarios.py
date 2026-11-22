"""Small, explicit manual/development and train-only app routes. No final data."""

from copy import deepcopy
from .partial_dynamic_env import make_scenario

NAMES = (
    "Open loading floor",
    "Storage rack detour",
    "Blocked aisle — wait for clearing",
    "Temporary pallet detour",
)


def geometry(name):
    cells = [list("." * 10) for _ in range(10)]
    if name == NAMES[1]:
        for r in range(1, 7):
            cells[r][3] = "#"
        for r in range(3, 9):
            cells[r][6] = "#"
    elif name == NAMES[2]:
        cells = [list("#" * 10) for _ in range(10)]
        for c in range(1, 9):
            cells[5][c] = "."
    elif name == NAMES[3]:
        for r in (2, 7):
            for c in (4, 5):
                cells[r][c] = "#"
    elif name != NAMES[0]:
        raise ValueError("Unknown preset")
    return ["".join(row) for row in cells]


def scenario(
    name, mode="Temporary", start=None, goal=None, split="manual", identifier=None
):
    defaults = {
        NAMES[0]: ([8, 1], [1, 8]),
        NAMES[1]: ([2, 1], [2, 8]),
        NAMES[2]: ([5, 1], [5, 8]),
        NAMES[3]: ([5, 1], [5, 8]),
    }
    default_start, default_goal = defaults[name]
    events = []
    if mode not in ("Static", "Temporary"):
        raise ValueError("Choose Static or Temporary")
    if mode == "Temporary" and name in NAMES[2:]:
        events = [
            {
                "id": "pallet-blockage",
                "cell": [5, 2] if name == NAMES[2] else [5, 3],
                "start": 0,
                "clear": 6 if name == NAMES[2] else 12,
            }
        ]
    c = make_scenario(
        geometry(name),
        list(start or default_start),
        list(goal or default_goal),
        events,
        scenario_id=identifier or f"app-development-{NAMES.index(name)}-{mode.lower()}",
        layout_id="app-layout-" + str(NAMES.index(name)),
        event_seed=73001 + NAMES.index(name),
        split=split,
    )
    c["app_preset"] = name
    return c


def development_cases():
    return [
        scenario(name, "Temporary" if i >= 2 else "Static")
        for i, name in enumerate(NAMES)
    ]


def training_cases():
    pairs = [
        ([8, 2], [1, 7]),
        ([1, 7], [8, 2]),
        ([8, 1], [1, 8]),
        ([1, 8], [8, 1]),
        ([5, 2], [5, 7]),
        ([5, 7], [5, 2]),
        ([6, 1], [4, 8]),
        ([4, 8], [6, 1]),
    ]
    result = []
    for i, (start, goal) in enumerate(pairs):
        name = NAMES[i // 2]
        base = scenario(
            name,
            "Static",
            start,
            goal,
            split="train",
            identifier=f"app-training-{i:02d}",
        )
        if i >= 4:
            base = make_scenario(
                base["rows"],
                start,
                goal,
                [
                    {
                        "id": "training-pallet",
                        "cell": [5, 4],
                        "start": 0,
                        "clear": 6 if i < 6 else 10,
                    }
                ],
                scenario_id=base["id"],
                layout_id=base["layout_id"],
                event_seed=73100 + i,
                split="train",
            )
        result.append(base)
    return deepcopy(result)
