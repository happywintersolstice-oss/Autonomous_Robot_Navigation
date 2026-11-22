"""Run the complete unchanged parent regression suite with passive operation counts."""
import dis
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter

import gymnasium as gym
import pytest
import torch

from .core import write_json
from .partial_agents import LinearPolicy
from .partial_features_scaled import ScaledLinearPolicy


def run(plan_path):
    plan=json.loads(Path(plan_path).read_text());counts={'completed_environment_transitions':0,'completed_neural_optimizer_steps':0,
        'completed_linear_updates':0,'failed_environment_step_calls':0}
    code={inspect.unwrap(cls.step).__code__ for cls in (torch.optim.Adam,torch.optim.SGD)}
    linear={LinearPolicy.update.__code__,ScaledLinearPolicy.update.__code__}
    def profile(frame,event,arg):
        if event!='return':return
        obj=frame.f_locals.get('self')
        if frame.f_code.co_name=='step' and isinstance(obj,gym.Env) and not isinstance(obj,gym.Wrapper):
            if isinstance(arg,tuple) and len(arg) in (4,5):counts['completed_environment_transitions']+=1
            elif arg is None:counts['failed_environment_step_calls']+=1
        elif frame.f_code in code or frame.f_code in linear:
            opcode=dis.opname[frame.f_code.co_code[frame.f_lasti]]
            if opcode.startswith('RETURN_'):
                counts['completed_linear_updates' if frame.f_code in linear else 'completed_neural_optimizer_steps']+=1
    started=perf_counter();sys.setprofile(profile)
    try:result=pytest.main(['-q',*plan['test_files'],'--junitxml='+plan['junit']])
    finally:sys.setprofile(None)
    write_json(Path(plan['receipt']),{'classification':'legacy_regression_mechanics','exit_code':int(result),
        'operations':counts,'wall_seconds':perf_counter()-started,'scope':'Complete unchanged parent test list, separately accounted by explicit user authorization',
        'scientific_actions':0,'research_evaluation_episodes':0})
    return int(result)
