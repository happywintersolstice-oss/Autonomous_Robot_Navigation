"""Hand-authored implementation cases, separate from generated research splits."""
from .partial_dynamic_env import make_scenario


def corridor(clear=4):
    rows = ['#'*10 for _ in range(10)]
    rows[5] = '#' + '.'*8 + '#'
    event = {'id': 'corridor-block', 'cell': [5, 3], 'start': 0, 'clear': clear}
    return make_scenario(rows, [5, 1], [5, 8], [event], scenario_id='only-corridor')


def detour(clear=8):
    rows = ['#'*10 for _ in range(10)]
    rows[4] = rows[5] = '#' + '.'*8 + '#'
    event = {'id': 'route-block', 'cell': [5, 3], 'start': 0, 'clear': clear}
    return make_scenario(rows, [5, 1], [5, 8], [event], scenario_id='detour')


def long_detour(clear=3):
    rows = ['#'*10 for _ in range(10)]
    rows[1] = rows[5] = '#' + '.'*8 + '#'
    for r in (2, 3, 4): rows[r] = '#.' + '#'*6 + '.#'
    event = {'id': 'short-block-long-detour', 'cell': [5, 2], 'start': 0, 'clear': clear}
    return make_scenario(rows, [5, 1], [5, 8], [event], scenario_id='long-detour')
