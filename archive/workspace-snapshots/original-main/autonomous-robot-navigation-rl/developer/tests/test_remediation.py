"""Audit regressions: damaged optional evidence, ID joins and independent pages."""

from navigation import benchmark_app
from copy import deepcopy
import json
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from robot_navigation.core import ROOT, load_task, read_json
from robot_navigation.release import final_summary


def test_fixed_final_rejects_arbitrary_completed_body(tmp_path):
    study = tmp_path/'artifacts/studies/final-v1'
    study.mkdir(parents=True)
    (study/'status.json').write_text('{"final_evaluation_complete":true}')
    (study/'summary.json').write_text('{"invented_success":1}')
    with pytest.raises((ValueError, FileNotFoundError)):
        final_summary(tmp_path)


def test_fixed_final_rejects_changed_metric_with_same_signature(tmp_path):
    source = ROOT/'artifacts/studies/final-v1'
    target = tmp_path/'artifacts/studies/final-v1'
    target.mkdir(parents=True)
    for name in ('status.json','summary.json','seal.json','freeze.json'):
        shutil.copy2(source/name,target/name)
    shutil.copytree(ROOT/'docs/evidence',tmp_path/'docs/evidence')
    summary = read_json(target/'summary.json')
    summary['systems']['Q-learning Refined']['metrics']['success_rate']['mean'] = .123
    (target/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='checksum'):
        final_summary(tmp_path)


def test_comparison_joins_by_id_and_rejects_missing_extra_families():
    from robot_navigation.dashboard import comparison_rows
    fourth = read_json(ROOT/'artifacts/studies/four-rl-v1/summary.json')
    details = deepcopy(read_json(ROOT/'artifacts/studies/four-rl-analysis-v1/summary.json'))
    for index,system in enumerate(details['systems'].values(),1):
        system['metrics']['mean_successful_route_length'].update(mean=index*11,contributing_seeds=index)
        system['counts']['success'] = index*101
        system['metrics']['mean_return']['mean'] = -index
    expected = comparison_rows(fourth,details)
    details['systems'] = dict(reversed(list(details['systems'].items())))
    assert comparison_rows(fourth,details)==expected
    assert [r['Successful route actions'] for r in expected]==[11,22,33,44]
    assert [r['Route contributing seeds'] for r in expected]==[1,2,3,4]
    assert [r['Successes across five seeds'] for r in expected]==[101,202,303,404]
    assert [r['Mean return'] for r in expected]==[-1,-2,-3,-4]
    del details['systems']['ppo']
    details['systems']['unknown'] = deepcopy(details['systems']['dqn'])
    with pytest.raises(ValueError,match='missing.*ppo.*extra.*unknown'):
        comparison_rows(fourth,details)


def test_methodology_does_not_read_optional_studies(tmp_path):
    script = tmp_path/'methods.py'
    script.write_text('from unittest.mock import patch\n'
        'from robot_navigation.core import ROOT\n'
        'from robot_navigation.dashboard_ui import render_page\n'
        'with patch("robot_navigation.dashboard_ui.four_way_summary", side_effect=ValueError("unverified")):\n'
        '    render_page("Methodology",None,ROOT)\n')
    app = AppTest.from_file(str(script)).run()
    assert not app.exception
    assert any('BFS' in text.value for text in app.markdown)


@pytest.mark.parametrize('failure',[FileNotFoundError('missing'),ValueError('malformed JSON'),
    ValueError('unverified'),KeyError('systems'),ValueError('task signature mismatch')])
def test_optional_four_family_failure_preserves_overview(tmp_path,failure):
    script = tmp_path/'panel.py'
    script.write_text('from unittest.mock import patch\n'
        'from robot_navigation.core import ROOT,load_task\n'
        'from robot_navigation.dashboard_ui import render_page\n'
        f'with patch("robot_navigation.dashboard_ui.four_way_summary",side_effect={failure!r}):\n'
        '    render_page("Overview",load_task(),ROOT)\n')
    app = AppTest.from_file(str(script),default_timeout=30).run()
    assert not app.exception
    assert app.dataframe
    assert any('four-rl-v1' in text.value for text in app.info)


def test_unseen_discovery_skips_malformed_candidate(tmp_path,monkeypatch):
    from robot_navigation.unseen_ui import discover_studies
    source = ROOT/'artifacts/studies'
    target = tmp_path/'artifacts/studies'
    target.mkdir(parents=True)
    for name in ('unseen-map-pilot-v1','unseen-map-main-v1'):
        (target/name).mkdir()
        for item in ('summary.json','verification.json'):
            shutil.copy2(source/name/item,target/name/item)
    (target/'unseen-map-final-v1').mkdir()
    (target/'unseen-map-final-v1/summary.json').write_text('{broken')
    choices,problems = discover_studies(tmp_path)
    assert choices==['unseen-map-main-v1','unseen-map-pilot-v1']
    assert problems and problems[0][0]=='unseen-map-final-v1'


def test_unseen_page_needs_no_fixed_validation(monkeypatch):
    import robot_navigation.benchmark as benchmark
    def forbidden(*args,**kwargs):
        raise FileNotFoundError('fixed-layout validation absent')
    monkeypatch.setattr(benchmark,'load_scenarios',forbidden)
    app = benchmark_app(str(ROOT/'app/streamlit_app.py'),default_timeout=30)
    app.session_state['dashboard_page'] = 'Unseen maps'
    app.run()
    assert not app.exception and app.dataframe
    app.session_state['unseen_view'] = 'Replay'
    app.run()
    assert not app.exception and app.session_state['unseen_playback'].index==0
