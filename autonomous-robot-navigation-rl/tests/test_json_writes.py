"""Complete JSON records survive concurrent readers and Windows replace failures."""
import errno
import json
import os
import threading

import pytest

from robot_navigation import core


def test_atomic_json_preserves_byte_format_and_rejects_nonfinite_values(tmp_path):
    value={'name':'robot','steps':[1,2],'reward':-.01}
    expected=tmp_path/'expected.json'
    expected.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    target=tmp_path/'record.json'
    core.write_json(target,value)
    assert target.read_bytes()==expected.read_bytes()
    with pytest.raises(ValueError):
        core.write_json(target,{'reward':float('nan')})
    assert target.read_bytes()==expected.read_bytes()
    assert not list(tmp_path.glob('*.tmp'))


def test_transient_windows_replace_failure_retries_complete_file(tmp_path,monkeypatch):
    target=tmp_path/'record.json'
    core.write_json(target,{'version':1})
    replace=os.replace
    attempts=[]
    def briefly_locked(source,destination):
        attempts.append(1)
        assert core.read_json(source)=={'version':2}
        if len(attempts)<3:
            assert core.read_json(target)=={'version':1}
            raise OSError(errno.EINVAL,'Windows destination temporarily mapped')
        replace(source,destination)
    monkeypatch.setattr(core.os,'replace',briefly_locked)
    monkeypatch.setattr(core.time,'sleep',lambda seconds:None)
    core.write_json(target,{'version':2})
    assert len(attempts)==3 and core.read_json(target)=={'version':2}
    assert not list(tmp_path.glob('*.tmp'))


def test_permanent_replace_failure_keeps_old_record_and_cleans_temp(tmp_path,monkeypatch):
    target=tmp_path/'record.json'
    core.write_json(target,{'version':1})
    attempts=[]
    def locked(source,destination):
        attempts.append(1)
        raise PermissionError(errno.EACCES,'Locked destination')
    monkeypatch.setattr(core.os,'replace',locked)
    monkeypatch.setattr(core.time,'sleep',lambda seconds:None)
    with pytest.raises(PermissionError):
        core.write_json(target,{'version':2})
    assert len(attempts)==6
    assert core.read_json(target)=={'version':1}
    assert not list(tmp_path.glob('*.tmp'))


def test_concurrent_reader_never_observes_a_partial_json_record(tmp_path):
    target=tmp_path/'record.json'
    core.write_json(target,{'version':0,'values':list(range(100))})
    stop=threading.Event()
    ready=threading.Event()
    errors=[]
    def reader():
        while not stop.is_set():
            try:
                record=core.read_json(target)
                assert record['values']==list(range(100))
                ready.set()
            except BaseException as exc:
                errors.append(exc)
                ready.set()
                return
    thread=threading.Thread(target=reader,daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=5)
        for version in range(1,31):
            core.write_json(target,{'version':version,'values':list(range(100))})
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert core.read_json(target)['version']==30
