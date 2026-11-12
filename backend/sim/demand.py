"""Transport job record (requests are created by the closed-loop lot flow in lotflow.py).

DemandGenerator below is the earlier open-loop model (random requests); it is kept for reference only."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Job:
    id: str
    src: str
    dst: str
    created: float
    hot: bool
    deadline: float
    src_area: str
    dst_area: str
    state: str = "PENDING"  # PENDING ASSIGNED LOADING DELIVERING UNLOADING DONE
    vehicle: Optional[int] = None
    assigned_t: Optional[float] = None
    picked_t: Optional[float] = None
    done_t: Optional[float] = None
    defer_until: float = 0.0
    late: bool = False
    defers: int = 0
    lot: Optional[object] = None  # the lot being moved (closed-loop flow)

    def to_dict(self, now: float) -> dict:
        return {"id": self.id, "src": self.src, "dst": self.dst, "srcArea": self.src_area, "dstArea": self.dst_area,
                "state": self.state, "vehicle": self.vehicle, "hot": self.hot, "age": round(now - self.created, 1),
                "late": self.late}


class DemandGenerator:
    def __init__(self, dcfg: dict, layout):
        self.cfg = dcfg
        self.layout = layout
        self.bays_by_process: Dict[str, List[str]] = {}
        for aid in layout.bay_ids:
            self.bays_by_process.setdefault(layout.areas[aid].process, []).append(aid)
        self.stockers = [a for a in layout.area_ids if layout.areas[a].kind == "STOCKER"]

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.seq = 0
        self.next_t = self._gap(0.0)

    def _phase(self, t: float) -> dict:
        ph = self.cfg["phases"][0]
        for p in self.cfg["phases"]:
            if t >= p["start"]:
                ph = p
        return ph

    def rate(self, t: float) -> float:
        return self.cfg["jobsPerMinute"] / 60.0 * self._phase(t)["multiplier"]

    def _gap(self, t: float) -> float:
        return float(self.rng.exponential(1.0 / max(self.rate(t), 1e-6)))

    def area_weights(self, t: float) -> Dict[str, float]:
        boost = self._phase(t).get("boost", {})
        w = {}
        for aid in self.layout.bay_ids:
            proc = self.layout.areas[aid].process
            w[aid] = self.cfg["bayWeights"].get(proc, 1.0) * boost.get(proc, 1.0)
        return w

    def _pick(self, weights: Dict[str, float], u: float) -> str:
        keys = list(weights)
        p = np.array([weights[k] for k in keys], dtype=float)
        c = np.cumsum(p / p.sum())
        return keys[min(int(np.searchsorted(c, u, side="right")), len(keys) - 1)]

    def generate(self, t: float) -> List[Job]:
        out = []
        while t + 1e-9 >= self.next_t:
            at = self.next_t
            u = self.rng.random(7)  # fixed number of draws per job: stream alignment
            w = self.area_weights(at)
            if u[0] < self.cfg["stockerShare"] * 0.5 and self.stockers:  # stocker -> bay
                src_area = self.stockers[int(u[1] * len(self.stockers))]
                dst_area = self._pick(w, u[2])
            else:
                src_area = self._pick(w, u[1])
                if u[0] > 1 - self.cfg["stockerShare"] * 0.5 and self.stockers:  # bay -> stocker
                    dst_area = self.stockers[int(u[2] * len(self.stockers))]
                else:
                    nxt = self.cfg["processFlow"].get(self.layout.areas[src_area].process, [])
                    cands = [b for p in nxt for b in self.bays_by_process.get(p, []) if b != src_area] or \
                        [b for b in self.layout.bay_ids if b != src_area]
                    dst_area = cands[int(u[2] * len(cands))]
            sp = self.layout.areas[src_area].ports
            dp = self.layout.areas[dst_area].ports
            hot = bool(u[5] < self.cfg["hotLotProbability"])
            self.seq += 1
            out.append(Job(f"J{self.seq:05d}", sp[int(u[3] * len(sp))], dp[int(u[4] * len(dp))], at, hot,
                           at + self.cfg["hotLotDeadline"] if hot else math.inf, src_area, dst_area))
            self.next_t = at + self._gap(at)
        return out
