import shutil

import pytest

from robot_navigation.core import ROOT, write_json
from robot_navigation.partial_commands import gate_path, require_gates, validation_inputs


def test_smoke_requires_successful_current_source_config_and_test_gates(tmp_path):
    (tmp_path/'configs').mkdir()
    shutil.copy2(ROOT/'configs/partial_dynamic.yaml', tmp_path/'configs/partial_dynamic.yaml')
    with pytest.raises(FileNotFoundError): require_gates(tmp_path)
    proof = {'status': 'failed', 'inputs': validation_inputs(tmp_path)}
    write_json(gate_path(tmp_path), proof)
    with pytest.raises(ValueError, match='failed/stale'): require_gates(tmp_path)
    proof['status'] = 'passed'; write_json(gate_path(tmp_path), proof)
    assert require_gates(tmp_path)['status'] == 'passed'
    with (tmp_path/'configs/partial_dynamic.yaml').open('a') as file: file.write('\n# changed\n')
    with pytest.raises(ValueError, match='failed/stale'): require_gates(tmp_path)


def test_validate_creates_scratch_parent_in_a_fresh_checkout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import robot_navigation.partial_commands as commands
    (tmp_path/'configs').mkdir()
    shutil.copy2(ROOT/'configs/partial_dynamic.yaml', tmp_path/'configs/partial_dynamic.yaml')
    tests = tmp_path/'developer/tests'; tests.mkdir(parents=True)
    for i in range(8): (tests/f'test_partial_fixture_{i}.py').write_text('')
    assert not (tmp_path/'.runtime').exists()
    def run(*args, **kwargs):
        assert (tmp_path/'.runtime').is_dir()
        return SimpleNamespace(returncode=0, stdout='fixture validation passed\n', stderr='')
    monkeypatch.setattr(commands.subprocess, 'run', run)
    assert commands.validate(tmp_path)['status'] == 'passed'
