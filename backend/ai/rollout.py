"""Episode runner and aggregate evaluation for OHT policies."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

KEYS = ("throughputPerHour", "avgDeliveryTime", "p95DeliveryTime", "avgWaitForVehicle", "emptyTravelRatio",
        "utilization", "blockedRatio", "hotOnTimeRate", "pending", "delivered")


def run_episode(env, policy, seed: int) -> Dict:
    obs, _ = env.reset(seed=seed)
    policy.reset(env)
    total, steps, done, info = 0.0, 0, False, {}
    while not done:
        obs, r, te, tr, info = env.step(policy.act(obs))
        total += r
        steps += 1
        done = te or tr
    m = info["metrics"]
    return {"seed": seed, "reward": total, "steps": steps, "reward_info": info.get("episode_reward_info", {}),
            **{k: v for k, v in m.items() if not isinstance(v, dict)}}


def evaluate_policy(env, policy, seeds: List[int]) -> Dict:
    eps = [run_episode(env, policy, s) for s in seeds]
    out = {"episodes": eps, "mean_reward": float(np.mean([e["reward"] for e in eps])),
           "std_reward": float(np.std([e["reward"] for e in eps]))}
    for k in KEYS:
        vals = [e[k] for e in eps if e.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
