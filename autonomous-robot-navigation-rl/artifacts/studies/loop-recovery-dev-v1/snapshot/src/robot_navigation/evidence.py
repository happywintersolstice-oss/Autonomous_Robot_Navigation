"""Small evidence contracts shared by readers; no UI or policy execution."""
import math

from .core import read_json

EXPECTED_ERRORS = (OSError, ValueError, KeyError)


def object_record(value, required=()):
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError(f'Evidence schema requires object fields: {", ".join(required)}')
    return value


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Evidence contains a nonfinite number')
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def checked_summary(value, task=None):
    object_record(value, ('signature','systems'))
    object_record(value['signature'])
    if not isinstance(value['systems'], (dict,list)) or not value['systems']:
        raise ValueError('Evidence requires nonempty systems')
    if task is not None:
        task.check(value['signature'])
    finite_tree(value)
    return value


def check_pinned_summary(root, study):
    """The reviewed manifest is versioned separately from frozen artifacts."""
    from .release import sha
    pins = read_json(root/'docs/evidence/study-summaries-v1.json')
    expected = pins['summaries'][study]
    if sha(root/'artifacts/studies'/study/'summary.json') != expected:
        raise ValueError(f'{study} summary checksum mismatch; restore reviewed evidence')
