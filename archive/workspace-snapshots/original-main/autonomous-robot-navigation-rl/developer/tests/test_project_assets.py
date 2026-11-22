"""The rehearsal inventory names damaged dependencies and never moves archives."""
import pytest

from robot_navigation.core import write_json
from robot_navigation.project_assets import verify,MANIFEST
from robot_navigation.release import sha


def test_demo_dependency_check_names_missing_and_changed_assets(tmp_path):
    asset = tmp_path/'model.npz'
    asset.write_bytes(b'fixture')
    write_json(tmp_path/MANIFEST,{'schema_version':1,'groups':{'demo':['model.npz']},
        'files':{'model.npz':{'sha256':sha(asset),'bytes':7}}})
    assert verify(tmp_path)['status']=='passed'
    asset.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed project asset: model.npz'):
        verify(tmp_path)
    asset.unlink()
    with pytest.raises(ValueError,match='Missing project asset: model.npz'):
        verify(tmp_path,groups=['demo'])


def test_demo_inventory_rejects_escape_path(tmp_path):
    write_json(tmp_path/MANIFEST,{'schema_version':1,'files':{'../outside':{'sha256':'x','bytes':0}}})
    with pytest.raises(ValueError,match='Missing project asset'):
        verify(tmp_path)
