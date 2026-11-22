"""Hand-authored diagnostic cases, not the generated research benchmark splits."""
from collections import deque
from copy import deepcopy

from .core import GridMap
from .partial_contract import MOVES
from .partial_dynamic_env import make_scenario, validate_scenario

FIXTURES = 'phase5a-handwritten-fixtures-v1'


def _rotate(cell, turns):
    r, c = cell
    for _ in range(turns): r, c = c, 9-r
    return [r, c]


def _turn_action(action, turns):
    if action == 4: return 4
    dr, dc = MOVES[action]
    for _ in range(turns): dr, dc = dc, -dr
    return MOVES.index((dr, dc))


def _entry(category, index, split, rows, start, goal, events, script, turns=0, kind=None):
    rotated = [['#']*10 for _ in range(10)]
    for r in range(10):
        for c in range(10):
            rr, cc = _rotate((r, c), turns); rotated[rr][cc] = rows[r][c]
    events = [{**e, 'cell': _rotate(e['cell'], turns)} for e in events]
    case_id = f'{FIXTURES}-{category}-{split}-{index:02}'
    case = make_scenario([''.join(row) for row in rotated], _rotate(start, turns), _rotate(goal, turns),
        events, scenario_id=case_id, layout_id=f'{FIXTURES}-{category}-rotation{turns}',
        event_seed=24000+index, split='manual')
    case.update(fixture_version=FIXTURES, fixture_category=category, fixture_split=split,
                fixture_kind=kind or category, difficulty={'distance_band': 'handwritten-diagnostic'})
    grid = validate_scenario(case)
    if len(grid.distances(case['start'])) != len(grid.free_cells):
        raise ValueError('Diagnostic base layout must be connected')
    return {'case': case, 'script': [_turn_action(a, turns) for a in script]}


def suite():
    """Fixed order and paired versions are independent of learned outcomes."""
    entries = []
    open_rows = ['.'*10]*10
    train = [((2, 2), (1, 2), [0]), ((7, 7), (8, 7), [1]),
             ((7, 2), (7, 1), [2]), ((2, 7), (2, 8), [3]),
             ((5, 5), (3, 5), [0, 0]), ((4, 4), (6, 4), [1, 1]),
             ((3, 6), (3, 4), [2, 2]), ((6, 3), (6, 5), [3, 3])]
    evaluation = [((3, 3), (2, 3), [0]), ((6, 6), (7, 6), [1]),
                  ((6, 3), (6, 2), [2]), ((3, 6), (3, 7), [3])]
    for split, values in (('train', train), ('evaluation', evaluation)):
        for i, (start, goal, script) in enumerate(values):
            entries.append(_entry('A', i, split, open_rows, start, goal, [], script))
    for split in ('train', 'evaluation'):
        end = 7 if split == 'train' else 8
        for i in range(4):
            rows = ['#'*10 for _ in range(10)]
            rows[4] = rows[5] = '#'+'.'*8+'#'
            row = list(rows[5]); row[3] = '#'; rows[5] = ''.join(row)
            entries.append(_entry('B', i, split, rows, [5, 1], [5, end], [],
                                  [0]+[3]*(end-1)+[1], i, 'permanent-wall-detour'))
            rows = ['#'*10 for _ in range(10)]; rows[5] = '#'+'.'*8+'#'
            clear = (3 if i % 2 == 0 else 5) if split == 'train' else (4 if i % 2 == 0 else 7)
            event = {'id': 'necessary-block', 'cell': [5, 2], 'start': 0, 'clear': clear}
            entries.append(_entry('C', i, split, rows, [5, 1], [5, end], [event],
                                  [4]*clear+[3]*(end-1), i, 'mandatory-initial-wait'))
        for i in range(4):
            short = i % 2 == 0
            rows = ['#'*10 for _ in range(10)]
            if short:
                rows[4] = rows[5] = '#'+'.'*8+'#'
                clear = 12 if split == 'train' else 16
                script = [0]+[3]*(end-1)+[1]
            else:
                rows[1] = rows[5] = '#'+'.'*8+'#'
                for r in (2, 3, 4): rows[r] = '#.'+'#'*6+'.#'
                clear = 3 if split == 'train' else 4
                script = [4]*clear+[3]*(end-1)
            event = {'id': 'detour-choice', 'cell': [5, 2], 'start': 0, 'clear': clear}
            entries.append(_entry('D', i, split, rows, [5, 1], [5, end], [event], script, i,
                                  'short-detour-long-block' if short else 'long-detour-short-block'))
            rows = ['#'*10 for _ in range(10)]; rows[5] = '#'+'.'*8+'#'
            for r in (2, 3, 4): rows[r] = '#.'+'#'*8
            appeared = i % 2 == 1
            clear = (10 if split == 'train' else 12) if appeared else (3 if split == 'train' else 4)
            event = {'id': 'out-of-range-change', 'cell': [5, 3], 'start': 3 if appeared else 0, 'clear': clear}
            script = [0]*3+[4]+[1]*3+[4]*max(0, clear-7)+[3]*(end-1)
            entries.append(_entry('E', i, split, rows, [5, 1], [5, end], [event], script, i,
                                  'appears-out-of-range' if appeared else 'reopens-out-of-range'))
    return entries


def cases(category=None, split='evaluation'):
    if split not in ('train', 'evaluation'): raise ValueError('No final/test fixture access')
    return [deepcopy(e['case']) for e in suite()
            if e['case']['fixture_split'] == split and (category is None or e['case']['fixture_category'] in category)]


class ScriptedMechanics:
    """Privileged handwritten action list; environment verification only."""
    algorithm = policy_id = 'scripted-mechanics-not-a-learner-v1'
    def __init__(self, actions): self.actions = iter(actions)
    def act(self, observation): return next(self.actions)


def hindsight_actions(case):
    """Trusted future-aware time-expanded BFS, never an agent input/reference."""
    grid = validate_scenario(case)
    start, goal = tuple(case['start']), tuple(case['goal'])
    todo, visited = deque([(start, 0)]), {(start, 0)}
    while todo:
        position, t = todo.popleft()
        if position == goal: return t
        if t == 200: continue
        occupied = {tuple(e['cell']) for e in case['events'] if e['start'] <= t < e['clear']}
        occupied.discard(position)  # Same occupied-cell suppression as the environment.
        for dr, dc in MOVES:
            nxt = (position[0]+dr, position[1]+dc)
            state = (nxt, t+1)
            if grid.is_free(nxt) and nxt not in occupied and state not in visited:
                visited.add(state); todo.append(state)
    return None
