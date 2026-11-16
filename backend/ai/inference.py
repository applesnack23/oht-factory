"""Load trained OHT models (SB3 .zip or NumPy .npz) for inference only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from backend.sim.config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "backend" / "models"


def have_sb3() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def load_model(path):
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = MODELS_DIR / p
    if p.is_dir():
        for name in ("best_model.npz", "best_model.zip", "final_model.npz", "final_model.zip"):
            if (p / name).exists():
                p = p / name
                break
    for ext in (".npz", ".zip"):
        if p.suffix == "" and p.with_suffix(ext).exists():
            p = p.with_suffix(ext)
    if p.suffix == ".npz":
        from backend.ai.numpy_ppo import NumpyPPO
        return NumpyPPO.load(p)
    if p.suffix == ".zip":
        from stable_baselines3 import PPO  # type: ignore
        return PPO.load(str(p), device="cpu")
    raise FileNotFoundError(f"model not found: {path}")


class PPOPolicy:
    """Wraps a trained model with the common policy interface; masks invalid candidates."""

    name = "ppo"

    def __init__(self, model_or_path, deterministic: bool = True):
        self.model = load_model(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
        self.deterministic = deterministic
        self.env = None

    def reset(self, env=None) -> None:
        if env is not None:
            self.env = env

    def act(self, observation) -> np.ndarray:
        mask = self.env.unwrapped.action_masks() if self.env is not None else None
        try:
            a, _ = self.model.predict(observation, deterministic=self.deterministic, action_masks=mask)
        except TypeError:  # plain SB3 PPO (no masking; the env falls back on invalid choices)
            a, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(a, dtype=np.int64).reshape(-1)


def list_models():
    out = []
    if not MODELS_DIR.exists():
        return out
    for d in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
        files = {f.name for f in d.iterdir()}
        main = next((n for n in ("best_model.npz", "best_model.zip", "final_model.npz", "final_model.zip") if n in files), None)
        if main is None:
            continue
        meta = {}
        if "run_meta.json" in files:
            try:
                meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out.append({"name": f"{d.name}/{main}", "run": d.name, "kind": main.split(".")[-1], "scenario": meta.get("level"),
                    "algo": meta.get("algo"), "timesteps": meta.get("timesteps"), "bestEval": meta.get("best_eval"),
                    "hasLatest": any(n.startswith("latest_model.") and not n.endswith(".json") for n in files)})
    return out
