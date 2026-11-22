"""Predeclared development pairs on the unchanged open A geometry; no final data."""
from collections import Counter

import numpy as np

from .core import digest
from .partial_contract import MOVES, ACTION_NAMES
from .partial_dynamic_env import make_scenario, validate_scenario
from .partial_feasibility_fixtures import cases

VERSION = 'phase5b-open-development-pairs-v1'
FIXTURE_SEED = 23022


def pair(case):
    return tuple(case['start']), tuple(case['goal'])


def goal_action(observation):
    """Analytic mechanics/capacity reference only; never a learner fallback."""
    relative = np.asarray(observation)[2:4]-np.asarray(observation)[:2]
    scores = np.array([-relative[0], relative[0], -relative[1], relative[1], 0.])
    return int(np.argmax(scores))


def strata(case):
    delta = np.array(case['goal'])-case['start']
    action = MOVES.index(tuple(np.sign(delta)))
    return {'direction': ACTION_NAMES[action], 'distance': int(np.abs(delta).sum()),
            'quadrant': int(case['start'][0] >= 5)*2+int(case['start'][1] >= 5),
            'boundary_band': 'edge' if any(v in (0, 9) for v in case['start']) else 'interior'}


def expanded_cases():
    rng = np.random.default_rng(FIXTURE_SEED)
    used = {pair(c) for c in cases('A', 'train')+cases('A')}
    starts, result = set(), []
    for action, (dr, dc) in enumerate(MOVES[:4]):
        for distance in (1, 2):
            for replicate in (0, 1):
                quadrant = (action+2*(distance-1)+replicate) % 4
                candidates = []
                for r in range(10):
                    for c in range(10):
                        goal = r+dr*distance, c+dc*distance
                        key = ((r, c), goal)
                        if (int(r >= 5)*2+int(c >= 5) == quadrant and all(0 <= x < 10 for x in goal)
                                and key not in used and (r, c) not in starts):
                            candidates.append(key)
                start, goal = candidates[int(rng.integers(len(candidates)))]
                case = make_scenario(['.'*10]*10, start, goal,
                    scenario_id=f'{VERSION}-{len(result):02}', layout_id=cases('A')[0]['layout_id'])
                case.update(fixture_category='A', fixture_split='expanded-development',
                    fixture_version=VERSION, fixture_kind='open-grid-scaling-diagnostic',
                    difficulty={'distance_band': f'{distance}-step-development'})
                result.append(case); used.add(pair(case)); starts.add(start)
    audit_cases(result)
    return result


def audit_cases(expanded):
    original, training = cases('A'), cases('A', 'train')
    if len(expanded) != 16 or len({pair(c) for c in expanded+original+training}) != 28:
        raise ValueError('Duplicate or missing development pair')
    rows = ['.'*10]*10
    for case in expanded+original+training:
        validate_scenario(case)
        if case['rows'] != rows or case['events'] or case['fixture_category'] != 'A':
            raise ValueError('A-only immutable geometry required')
    counts = {key: dict(Counter(str(strata(c)[key]) for c in expanded))
              for key in ('direction', 'distance', 'quadrant', 'boundary_band')}
    if sorted(counts['direction'].values()) != [4]*4 or counts['distance'] != {'1': 8, '2': 8}:
        raise ValueError('Unbalanced directions/distances')
    if sorted(counts['quadrant'].values()) != [4]*4:
        raise ValueError('Unbalanced quadrants')
    return {'version': VERSION, 'generator_seed': FIXTURE_SEED, 'counts': counts,
            'original_hash': digest(original), 'training_hash': digest(training),
            'expanded_hash': digest(expanded), 'pair_overlap': 0,
            'information_scope': 'Previously inspected familiar open geometry; development pairs only'}


class GoalDirectionScript:
    algorithm = policy_id = 'analytic-goal-direction-mechanics-not-learned'
    def act(self, observation):
        return goal_action(observation)
