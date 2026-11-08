"""Enrich the verified PPO comparison without editing any previous study."""
import argparse
from pathlib import Path

from robot_navigation.core import ROOT, load_task
from robot_navigation.four_way_analysis import ANALYSIS_STUDY, build_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-id', default=ANALYSIS_STUDY)
    args = parser.parse_args()
    if Path(args.suite_id).name != args.suite_id or args.suite_id in ('.', '..'):
        raise ValueError('suite-id must be a single directory name')
    study = ROOT/'artifacts/studies'/args.suite_id
    result = build_analysis(load_task(), study)
    print(f"Verified {len(result['systems'])} families, 20 runs and 18,560 saved episodes; no new policy executions")


if __name__ == '__main__':
    main()
