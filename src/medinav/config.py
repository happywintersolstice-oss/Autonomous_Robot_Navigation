"""YAML experiment configs -> python dicts. Every experiment is reproducible from its file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path) -> dict:
    """Load a YAML file. `base: other.yaml` inherits from another file in the same folder."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(parent, cfg)
    return cfg


def run_root(cfg: dict) -> Path:
    return ROOT / "runs" / cfg.get("experiment", "default")


def algorithm_specs(cfg: dict, only=None) -> dict:
    """{label: spec}. A label is a run name (e.g. 'ppo_lastvalid'); spec['algo'] the algorithm."""
    specs = {}
    for label, spec in (cfg.get("algorithms") or {}).items():
        spec = dict(spec or {})
        spec.setdefault("algo", label)
        if spec.get("enabled", True) and (not only or label in only or spec["algo"] in only):
            specs[label] = spec
    return specs


def run_roots(cfg: dict) -> list:
    """This experiment's runs + runs of other experiments listed in `include_runs`."""
    return [run_root(cfg)] + [ROOT / "runs" / name for name in cfg.get("include_runs") or []]
