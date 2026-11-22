"""POST HOC analytic policy-class witness; zero training or executed env actions.

This never loads/initializes a learned policy, changes a model, or supplies a
controller to training/evaluation. It only multiplies legal observed features
by an explicitly labelled mathematical coefficient matrix.
"""
from pathlib import Path
import argparse

import numpy as np

from robot_navigation.core import ROOT, read_json, write_json
from robot_navigation.release import sha
from robot_navigation.partial_contract import MOVES
from robot_navigation.partial_dynamic_env import PartialDynamicEnv
from robot_navigation.partial_features import FEATURE_COUNT, encode_features
from robot_navigation.partial_feasibility import location, checked_protocol
from robot_navigation.partial_feasibility_fixtures import cases


def calculate():
    assert FEATURE_COUNT == 412
    coefficients = np.zeros((5, FEATURE_COUNT))
    # Existing final two features are goal-row minus robot-row, goal-column
    # minus robot-column. This is an analytic witness, NOT learner initialization.
    coefficients[0, -2] = -1; coefficients[1, -2] = 1
    coefficients[2, -1] = -1; coefficients[3, -1] = 1
    rows = []
    for case in cases('A'):
        observation, _ = PartialDynamicEnv([case]).reset(seed=0)
        scores = coefficients @ encode_features(observation)
        action = int(np.argmax(scores))
        r, c = case['start']; dr, dc = MOVES[action]
        matches = [r+dr, c+dc] == case['goal']
        assert matches
        rows.append({'case_id': case['id'], 'observation': observation.tolist(),
                     'analytic_scores': scores.tolist(), 'analytic_argmax': action,
                     'one_step_goal_direction_matches': matches})
    return {'analysis_version': 'phase5a-affine-policy-capacity-witness-v1', 'post_hoc': True,
        'training_interactions': 0, 'environment_step_calls': 0, 'observations_reset_only': 4,
        'trained_models_loaded_or_changed': 0, 'feature_count': FEATURE_COUNT,
        'analytic_coefficients': coefficients.tolist(), 'cases': rows,
        'conclusion': 'The existing affine policy class can express the correct initial action for all four '
            'one-step A diagnostic cases. This is not a learned controller or an empirical success rate. '
            'It does not prove representability of the true Bellman value function, TD convergence, '
            'or competence on B-E. An inability to express A policy decisions is therefore not established '
            'as the cause; optimization, normalization, state distribution and finite budgets remain hypotheses.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); checked_protocol()
    output = location()/'derived-analysis/affine-policy-capacity-v1'
    result = calculate()
    if args.verify:
        assert read_json(output/'analysis.json') == result
        inputs = read_json(output/'manifest.json')
        assert inputs['analysis_sha256'] == sha(output/'analysis.json')
        assert inputs['source_sha256'] == sha(Path(__file__))
        print('Analytic witness verified; zero learning/evaluation actions executed.')
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'analysis.json', result)
    write_json(output/'manifest.json', {'post_hoc': True, 'source_path': 'developer/commands/phase5a_affine_capacity_v1.py',
        'source_sha256': sha(Path(__file__)), 'analysis_sha256': sha(output/'analysis.json'),
        'measured_output_inventory_sha256': sha(location()/'outputs.json'),
        'protocol_sha256': sha(location()/'phase5a_pilot_protocol.json'), 'no_extra_learning': True})
    print('Analytic witness written: four legal observed-feature score calculations, zero env actions.')


if __name__ == '__main__': main()
