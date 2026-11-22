"""Post-analysis delivery-only correction; no learning or inference."""
import ast
import importlib.util

import pytest

from robot_navigation.core import ROOT

PATH=ROOT/'developer/commands/phase5d_delivery.py'
SPEC=importlib.util.spec_from_file_location('phase5d_delivery',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('raw,expected',[('True',True),('False',False),('true',True),('false',False),('0',0),('Right','Right')])
def test_boolean_decoder_preserves_other_CSV_fields(raw,expected):
    assert MODULE.decode(raw)==expected and type(MODULE.decode(raw)) is type(expected)


def test_boolean_aggregation_and_JSON_vector_unchanged():
    assert sum(MODULE.decode(v) for v in ('True','False','True'))==2
    assert MODULE.decode('[0.1,0.2,0.3,0.4,0]')==[.1,.2,.3,.4,0]
    assert MODULE.decode('') is None


def test_delivery_has_no_training_or_scoring_path():
    tree=ast.parse(PATH.read_text(encoding='utf-8'))
    forbidden={'learn','train','update','step','act','analyze','measured_score','get_distribution','predict_values'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in forbidden
