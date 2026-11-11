"""Config loading: fab.json + demand.json + reward.json, optional scenario overrides."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_json(name: str) -> Dict[str, Any]:
    p = Path(name)
    if not p.exists():
        p = CONFIG_DIR / (name if name.endswith(".json") else name + ".json")
    return json.loads(p.read_text(encoding="utf-8"))


def scenario_names():
    return list(load_json("scenarios.json")["scenarios"])


def resolve_config(config: Union[None, str, Dict[str, Any]] = None, overrides: Optional[dict] = None) -> Dict[str, Any]:
    """None/'normal' = base fab; 'scenario:<name>' or '<name>' = scenario; dict = overrides."""
    if isinstance(config, dict) and config.get("_resolved"):
        cfg = copy.deepcopy(config)
    else:
        cfg = load_json("fab.json")
        cfg["demand"] = load_json("demand.json")
        cfg["reward"] = load_json("reward.json")
        cfg["blocks"] = []
        cfg["scenario"] = "normal"
        if isinstance(config, dict):
            cfg = deep_merge(cfg, config)
        elif config:
            name = str(config).split(":", 1)[-1]
            sc = load_json("scenarios.json")["scenarios"]
            if name not in sc:
                raise KeyError(f"unknown scenario '{name}', available: {list(sc)}")
            over = {k: v for k, v in sc[name].items() if k != "description"}
            cfg = deep_merge(cfg, over)
            cfg["scenario"] = name
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_resolved"] = True
    return cfg
