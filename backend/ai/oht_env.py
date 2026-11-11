"""Event-driven Gymnasium environment for OHT control.

One env step = one decision. After the action is applied the simulation
advances until the next decision is needed (ROUTE when a loaded/assigned
vehicle approaches a crossover diverge, DISPATCH when a job is pending and
idle vehicles exist, REPOSITION when a vehicle becomes idle). The reward is
everything that happened in between (event + time based).

Observation (fixed size, normalized):
  global (16) | decision type one-hot (3) | context (12) | K candidates x 10
Action: MultiDiscrete([K+1]) -> candidate index, or K = defer / default.
Candidates are pre-sorted by ETA, so index 0 is always "nearest/fastest".
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from backend.compat.gym_compat import gym, spaces
from backend.sim.config import resolve_config
from backend.sim.fab import DISPATCH, IDLE, REPOSITION, ROUTE, FabSimulation

TRAIN_SEED_RANGE = (0, 1_000_000)
EVAL_SEED_BASE = 1_000_000_000
TYPES = (DISPATCH, ROUTE, REPOSITION)
G, C, F = 16, 12, 10


def eval_seeds(n: int, offset: int = 0):
    return [EVAL_SEED_BASE + offset + i for i in range(n)]


class RewardCalc:
    def __init__(self, cfg: dict):
        self.c = cfg

    def step(self, r: dict, dt: float) -> Dict[str, float]:
        c = self.c
        hot = r["hotInSystem"]
        return {"delivery": len(r["delivered"]) * c["delivery"],
                "inSystem": ((r["inSystem"] - hot) * c["jobInSystemPerSecond"] + hot * c["hotInSystemPerSecond"]) * dt,
                "hotLate": r["hotLate"] * c["hotLate"],
                "emptyTravel": r["emptyMeters"] * c["emptyTravelPerMeter"],
                "blocked": r["blockedVehicles"] * dt * c["blockedPerVehicleSecond"],
                "deadlock": c["deadlock"] if r["deadlock"] else 0.0}


class OHTEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, overrides: Optional[dict] = None, seed_mode: str = "train"):
        super().__init__()
        # "mix" / "mix:a,b,c": every episode samples one scenario (training distribution)
        self.mix = None
        if isinstance(config, str) and config.startswith("mix"):
            from backend.sim.config import scenario_names
            names = config.split(":", 1)[1].split(",") if ":" in config else scenario_names()
            self.mix = [n.strip() for n in names if n.strip()]
            self._fabs = {n: FabSimulation(n, overrides) for n in self.mix}
            config = self.mix[0]
        self.cfg = resolve_config(config, overrides)
        self.fab = self._fabs[self.mix[0]] if self.mix else FabSimulation(self.cfg)
        self.K = self.fab.K
        self.seed_mode = seed_mode
        self.rc = RewardCalc(self.cfg["reward"])
        self.obs_dim = G + len(TYPES) + C + self.K * F
        self.observation_space = spaces.Box(low=-1.0, high=3.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.array([self.K + 1], dtype=np.int64))
        self.decision = None
        self.reward_info = {}

    # ------------------------------------------------------------- core
    def _advance_to_decision(self) -> Dict[str, float]:
        acc = {k: 0.0 for k in ("delivery", "inSystem", "hotLate", "emptyTravel", "blocked", "deadlock")}
        f = self.fab
        while True:
            ds = f.decisions()
            if ds:
                self.decision = ds[0]
                return acc
            if f.done:
                self.decision = None
                return acc
            r = f.step()
            for k, v in self.rc.step(r, f.dt).items():
                acc[k] += v

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is None:
            seed = (1_000_000_000 + int(self.np_random.integers(0, 1_000_000))) if self.seed_mode == "eval" \
                else int(self.np_random.integers(*TRAIN_SEED_RANGE))
        if self.seed_mode == "train" and seed >= EVAL_SEED_BASE:
            raise ValueError("evaluation seed used in training mode")
        if self.mix:
            name = self.mix[int(np.random.default_rng(int(seed) + 17).integers(len(self.mix)))]
            if options and "scenario" in options:
                name = options["scenario"]
            self.fab = self._fabs[name]
            self.cfg = self.fab.cfg
            self.rc = RewardCalc(self.cfg["reward"])
        self.fab.reset(int(seed))
        self.episode = {k: 0.0 for k in ("delivery", "inSystem", "hotLate", "emptyTravel", "blocked", "deadlock", "invalid", "end")}
        self._advance_to_decision()
        return self._obs(), {"seed": int(seed), "scenario": self.cfg.get("scenario")}

    def step(self, action):
        a = int(np.asarray(action).reshape(-1)[0])
        mask = self.action_masks()
        invalid = 0.0
        if not mask[a]:
            invalid = self.cfg["reward"]["invalidAction"]
            a = self.K  # fall back to defer/default
        choice = None if a >= self.K else a
        res = self.fab.apply_decision(self.decision, choice)
        acc = self._advance_to_decision()
        acc["invalid"] = invalid
        terminated = bool(self.fab.metrics["deadlock"])
        truncated = bool(self.decision is None and not terminated)
        if terminated or truncated:
            acc["end"] = len(self.fab.jobs) * self.cfg["reward"]["episodeEndPerUndelivered"]
        reward = float(sum(acc.values()))
        for k, v in acc.items():
            self.episode[k] = self.episode.get(k, 0.0) + v
        info = {"reward_info": acc, "applied": res, "kind": None if self.decision is None else self.decision.kind}
        if terminated or truncated:
            info["metrics"] = self.fab.summary()
            info["episode_reward_info"] = dict(self.episode)
        obs = self._obs() if self.decision is not None else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        m = np.zeros(self.K + 1, dtype=bool)
        m[self.K] = True  # defer / default is always valid
        if self.decision is not None:
            m[: len(self.decision.candidates)] = True
        return m

    # -------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        f = self.fab
        o = np.zeros(self.obs_dim, dtype=np.float32)
        lay = f.layout
        st = f.area_stats()
        pend = [j for j in f.jobs.values() if j.state == "PENDING"]
        V = len(f.vehicles)
        g = [f.time / f.episode_length, len(pend) / 50.0, len(f.jobs) / 80.0,
             sum(1 for v in f.vehicles if v.state == IDLE) / V,
             sum(1 for v in f.vehicles if v.state != IDLE and v.v < 0.05) / V,
             (np.mean([f.time - j.created for j in pend]) / 300.0) if pend else 0.0,
             sum(1 for j in pend if j.hot) / 10.0,
             sum(st[a]["rate"] for a in st) / 30.0]
        g += [st[b]["pending"] / 10.0 for b in lay.bay_ids]
        o[:G] = g[:G]
        d = self.decision
        if d is None:
            return np.clip(o, -1.0, 3.0)
        o[G + TYPES.index(d.kind)] = 1.0
        base = G + len(TYPES)
        sx, sz = 130.0, 60.0

        def pxz(port):
            p = lay.ports[port]
            return p.x / sx, p.z / sz

        ctx = [0.0] * C
        if d.kind == DISPATCH:
            j = f.jobs[d.subject]
            ax, az = pxz(j.src)
            bx, bz = pxz(j.dst)
            ps, pd = lay.ports[j.src], lay.ports[j.dst]
            dist = f.track.meters_from(f.track.edges[ps.edge].v, pd.edge, pd.offset)
            a = st[j.src_area]
            slack = (j.deadline - f.time) / 300.0 if j.hot else 1.0
            ctx = [ax, az, bx, bz, (f.time - j.created) / 300.0, float(j.hot), 0.0, dist / 300.0,
                   a["idle"] / 10.0, a["pending"] / 10.0, a["rate"] / 5.0, slack]
        else:
            v = f.vehicles[int(d.subject)]
            x, z, _ = f.vehicle_pose(v)
            j = f.jobs.get(v.job) if v.job else None
            if d.kind == ROUTE:
                te, to = v.target
                tx, tz = f.track.edges[te].piece.point(to)
                slack = ((j.deadline - f.time) / 300.0) if (j and j.hot) else 1.0
                ctx = [x / sx, z / sz, tx / sx, tz / sz, ((f.time - j.created) / 300.0) if j else 0.0,
                       float(bool(j and j.hot)), float(v.state == "TO_DROP"), d.candidates[0]["meters"] / 300.0,
                       0.0, 0.0, 0.0, slack]
            else:
                a = lay.areas[v.home]
                h = st[v.home]
                ctx = [x / sx, z / sz, a.cx / sx, a.cz / sz, 0.0, 0.0, 0.0, 0.0, h["idle"] / 10.0,
                       h["pending"] / 10.0, h["rate"] / 5.0, 0.0]
        o[base:base + C] = ctx
        cb = base + C
        for i, c in enumerate(d.candidates[: self.K]):
            if d.kind == DISPATCH:
                feat = [1.0, c["eta"] / 120.0, c["meters"] / 300.0, c["cong"] / 10.0, c["stopped"] / 5.0,
                        c["homePending"] / 10.0, c["homeIdle"] / 10.0, c["homeRate"] / 5.0, c["sameArea"], c["repositioning"]]
            elif d.kind == ROUTE:
                feat = [1.0, c["eta"] / 120.0, c["meters"] / 300.0, c["cong"] / 10.0, c["stopped"] / 5.0,
                        c["detour"] / 150.0, c["crossover"], 0.0, 0.0, 0.0]
            else:
                feat = [1.0, c["eta"] / 120.0, c["meters"] / 300.0, c["pending"] / 10.0, c["idle"] / 10.0,
                        c["rate"] / 5.0, c["current"], 0.0, 0.0, 0.0]
            o[cb + i * F: cb + (i + 1) * F] = feat
        return np.clip(o, -1.0, 3.0)

    def close(self):
        pass


def check_env_lite(env: OHTEnv, n: int = 200) -> str:
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs) and isinstance(info, dict)
    for _ in range(n):
        m = env.action_masks()
        a = np.array([np.random.choice(np.flatnonzero(m))])
        assert env.action_space.contains(a)
        obs, r, te, tr, info = env.step(a)
        assert env.observation_space.contains(obs) and np.isfinite(obs).all() and np.isfinite(r)
        assert isinstance(te, bool) and isinstance(tr, bool)
        if te or tr:
            env.reset()
    o1, _ = env.reset(seed=5)
    o2, _ = env.reset(seed=5)
    assert np.array_equal(o1, o2)
    return "lite"


def check_env(env: OHTEnv) -> str:
    try:
        from stable_baselines3.common.env_checker import check_env as sb3_check  # type: ignore
    except Exception:
        return check_env_lite(env)
    sb3_check(env, warn=True)
    return "sb3"
