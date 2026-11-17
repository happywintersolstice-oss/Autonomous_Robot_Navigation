from pathlib import Path
import pytest
import yaml

from robot_navigation.core import ROOT
from robot_navigation.ppo_config import validate_config


def config():
    plan=yaml.safe_load((ROOT/'configs/ppo_comparison.yaml').read_text('utf-8'))
    return {**plan['training'],'seed':7,'run_id':'fixture-ppo'}


def test_registered_ppo_settings_have_exact_rollout_and_monitor_budgets():
    settings=config()
    validate_config(settings)
    assert settings['interactions']==200000
    assert settings['interactions']//settings['n_steps']==200
    assert settings['monitor_every']==10000


@pytest.mark.parametrize('changes',[{'interactions':200001},{'monitor_every':9999},{'batch_size':7},
    {'run_id':'../bad'},{'n_steps':1},{'learning_rate':float('nan')},{'discount':1.1}])
def test_invalid_ppo_configuration_is_rejected_before_training(changes):
    with pytest.raises(ValueError):
        validate_config({**config(),**changes})
