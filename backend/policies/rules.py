"""Baseline OHT controllers. They read only the env observation (same information as the AI).

basic    : industry default — nearest idle vehicle (NVF), shortest-time route, idle vehicles stay put
smart    : NVF with idle-surplus preference, congestion-aware routing, demand-driven repositioning
           (basic and smart only dispatch idle vehicles, like a classic OHT controller)
preassign: smart + may pre-assign a job to a vehicle that is about to finish its drop when that
           vehicle can reach the pickup sooner than any idle one
random: uniformly random valid choice
"""
from __future__ import annotations

import numpy as np

from backend.ai.oht_env import C, F, G, TYPES

DISPATCH_I, ROUTE_I, REPOSITION_I = 0, 1, 2


def _parse(obs, K):
    obs = np.asarray(obs)
    kind = int(np.argmax(obs[G:G + len(TYPES)]))
    cb = G + len(TYPES) + C
    cands = obs[cb:cb + K * F].reshape(K, F)
    valid = cands[:, 0] > 0.5
    return kind, cands, valid


BUSY = 10


def _idle_only(cands, valid):
    return valid & (cands[:, BUSY] < 0.5)


class BasicRulePolicy:
    name = "rule_basic"

    def __init__(self, env=None):
        self.K = env.unwrapped.K if env is not None else 8

    def reset(self, env=None):
        pass

    def act(self, obs):
        kind, cands, valid = _parse(obs, self.K)
        if kind == DISPATCH_I:
            valid = _idle_only(cands, valid)
        if kind == REPOSITION_I or not valid.any():
            return np.array([self.K])  # stay / defer
        return np.array([int(np.flatnonzero(valid)[0])])  # nearest idle vehicle / fastest route


class SmartRulePolicy:
    name = "rule_smart"

    use_busy = False
    busy_margin = 0.0

    def __init__(self, env=None, congestion_s=6.0, stopped_s=10.0, surplus_bonus=0.15, move_threshold=0.6):
        self.K = env.unwrapped.K if env is not None else 8
        self.congestion_s, self.stopped_s = congestion_s, stopped_s
        self.surplus_bonus, self.move_threshold = surplus_bonus, move_threshold

    def reset(self, env=None):
        pass

    def act(self, obs):
        kind, c, valid = _parse(obs, self.K)
        if kind == DISPATCH_I and not self.use_busy:
            valid = _idle_only(c, valid)
        if not valid.any():
            return np.array([self.K])
        idx = np.flatnonzero(valid)
        if kind == DISPATCH_I:
            # ETA (s) minus a bonus for taking a vehicle from an area with idle surplus
            eta = c[idx, 1] * 120.0
            surplus = np.clip(c[idx, 6] * 10.0 - c[idx, 5] * 10.0, 0, 4)
            score = eta * (1.0 - self.surplus_bonus * surplus / 4.0) + self.busy_margin * c[idx, BUSY]
            return np.array([int(idx[np.argmin(score)])])
        if kind == ROUTE_I:
            cost = c[idx, 1] * 120.0 + self.congestion_s * c[idx, 3] * 10.0 * 0.2 + self.stopped_s * c[idx, 4] * 5.0
            return np.array([int(idx[np.argmin(cost)])])
        # REPOSITION: go where demand exceeds idle supply, if clearly better than staying
        rate, idle, pend = c[idx, 5] * 5.0, c[idx, 4] * 10.0, c[idx, 3] * 10.0
        eta = c[idx, 1] * 120.0
        need = rate * 1.5 + pend - idle
        score = need / (1.0 + eta / 60.0)
        cur = np.flatnonzero(c[idx, 6] > 0.5)
        best = int(np.argmax(score))
        if cur.size and score[best] - score[cur[0]] < self.move_threshold:
            return np.array([self.K])  # stay
        return np.array([int(idx[best])])


class PreassignRulePolicy(SmartRulePolicy):
    name = "rule_preassign"
    use_busy = True
    busy_margin = 10.0  # a busy vehicle must be >10 s faster than the best idle one


class RandomPolicy:
    name = "random"

    def __init__(self, env=None, seed=0):
        self.K = env.unwrapped.K if env is not None else 8
        self.rng = np.random.default_rng(seed)

    def reset(self, env=None):
        pass

    def act(self, obs):
        _, _, valid = _parse(obs, self.K)
        opts = list(np.flatnonzero(valid)) + [self.K]
        return np.array([int(self.rng.choice(opts))])
