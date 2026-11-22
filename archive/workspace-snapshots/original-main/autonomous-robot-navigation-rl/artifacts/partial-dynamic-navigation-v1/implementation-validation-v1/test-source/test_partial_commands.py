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
