"""Stocker interior: shelves + one stacker crane per stocker (visual model).

Ports are either inputs (IN: OHTs drop FOUPs) or outputs (OUT: OHTs pick FOUPs up). Each port
conveyor has two positions, the stage under the rail and the inner end (at the near rack's bottom
level), and each position holds one FOUP:
  IN : stage_in --(conveyor, when the inner end is free)--> in --> inner_in --(crane takes)
  OUT: (crane gives, only onto a free inner end) inner_out --(when the stage is free)--> out --> stage
An OHT lowers onto an IN stage only when it is free, and lifts from an OUT stage only when its FOUP
is there; otherwise it waits with its gripper up (fab metric portWaitTime).

Crane: travels along the aisle (z), lifts (y = fork top), turns its fork 180 degrees on a
turntable to serve the near or the far rack, and extends a narrow fork (u). Shelves and conveyors
have a groove in the middle: the fork enters below the FOUP, lifts it off the two shelf pads,
retracts; to put down it enters above, lowers until the FOUP rests on the pads, retracts.

Retrievals have priority. An OHT that reaches the port before the FOUP is on the stage waits
there with its gripper up (fab metric portWaitTime). The stocker uses its own random generator.

Local coordinates per stocker: u = metres from the OHT lane into the stocker (0 = under the
rail, where the I/O port shelves are), z = world z, y = height. worldX = laneX + dir * u.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

WALL_U, NEAR_U, AISLE_U, FAR_U, BACK_U = 0.6, 1.1, 2.3, 3.5, 4.0
LEVELS = [0.9, 1.75, 2.6, 3.45]  # shelf surface heights; the port conveyors' inner end is at LEVELS[0]
N_BAYS, BAY_PITCH = 12, 1.7
TRAVEL_V, LIFT_V, FORK_V = 3.5, 1.5, 2.0
FINE_V = 0.35  # slow lift under / onto the FOUP
DY = 0.07  # fork top below (pick) or above (place) the shelf surface when entering
TURN_T = 1.2  # 180 degree turn of the fork
CONV_T = 2.5  # port conveyor stage <-> inner position


class Stocker:
    def __init__(self, sid: str, lane_x: float, direction: int, ports: Dict[str, float], seed: int, fill: float = 0.6,
                 io: Optional[Dict[str, str]] = None):
        self.id, self.lane_x, self.dir = sid, lane_x, direction
        self.ports = ports  # port id -> z
        self.io = io or {p: ("IN" if i % 2 == 0 else "OUT") for i, p in enumerate(ports)}
        z0 = -(N_BAYS - 1) * BAY_PITCH / 2
        self.bay_z = [z0 + i * BAY_PITCH for i in range(N_BAYS)]
        # cells: (rack 0 = near/lane side, 1 = far), bay, level. Near rack level 0 holds the port conveyors.
        self.cells = [(r, b, l) for r in (0, 1) for b in range(N_BAYS) for l in range(len(LEVELS)) if not (r == 0 and l == 0)]
        self.rng = np.random.default_rng(seed)
        self.occ = (self.rng.random(len(self.cells)) < fill).astype(np.int8)
        self.tasks: deque = deque()
        self.conv: List[list] = []  # [port, kind, t0, t1, job]; kinds see module docstring
        self.ready: set = set()  # job ids whose FOUP is on the stage
        self.cancelled: set = set()
        self.z, self.y, self.fork = 0.0, LEVELS[1], AISLE_U
        self.side = -1  # fork faces the near rack (-1) or the far rack (+1)
        self.rot = 0.0  # 0 = facing near, 1 = facing far (animated during a turn)
        self.carry = False
        self.task: Optional[dict] = None
        self.legs: List[tuple] = []
        self.done = 0
        self.t = 0.0

    # ------------------------------------------------------------ geometry
    def cell_pos(self, i: int):
        r, b, l = self.cells[i]
        return (NEAR_U if r == 0 else FAR_U), self.bay_z[b], LEVELS[l]

    def to_dict(self) -> dict:
        return {"id": self.id, "laneX": self.lane_x, "dir": self.dir, "wallU": WALL_U, "nearU": NEAR_U, "aisleU": AISLE_U,
                "farU": FAR_U, "backU": BACK_U, "levels": LEVELS, "bayZ": self.bay_z, "bayPitch": BAY_PITCH,
                "cells": [list(self.cell_pos(i)) for i in range(len(self.cells))], "convT": CONV_T,
                "ports": [{"id": p, "z": z, "io": self.io[p]} for p, z in self.ports.items()]}

    # ------------------------------------------------------------ requests
    def _at(self, port: str, *kinds) -> bool:
        return any(c[0] == port and c[1] in kinds for c in self.conv)

    def can_drop(self, port: str) -> bool:
        """An OHT may lower a FOUP onto this IN port's stage."""
        return not self._at(port, "stage_in")

    def store(self, port: str) -> None:
        """An OHT has put a FOUP on the IN stage: it is conveyed in when the inner end is free."""
        self.conv.append([port, "stage_in", self.t, self.t, None])

    def retrieve(self, job_id: str, port: str) -> None:
        """Retrievals go before queued stores: an OHT may be waiting for this FOUP."""
        task = {"kind": "RETRIEVE", "port": port, "job": job_id}
        for i, t in enumerate(self.tasks):
            if t["kind"] == "STORE":
                self.tasks.insert(i, task)
                return
        self.tasks.append(task)

    def picked(self, job_id: str) -> None:
        """The OHT has lifted the FOUP of job_id from the stage."""
        if job_id in self.ready:
            self.ready.discard(job_id)
            self.conv = [c for c in self.conv if c[4] != job_id]
            return
        for t in list(self.tasks):  # vehicle came before the crane: drop the pending retrieval
            if t["kind"] == "RETRIEVE" and t["job"] == job_id:
                self.tasks.remove(t)
                return
        self.cancelled.add(job_id)  # retrieval in progress: its FOUP disappears when placed

    # ------------------------------------------------------------ crane
    def _pick_cell(self, empty: bool, near_z: float) -> int:
        idx = np.flatnonzero(self.occ == (0 if empty else 1))
        if idx.size == 0:  # full / empty stocker: recycle a cell (stock flows in and out of the fab)
            idx = np.arange(len(self.cells))
        if empty:  # store close to the port
            d = [abs(self.cell_pos(i)[1] - near_z) + 0.3 * self.cells[i][2] for i in idx]
            return int(idx[int(np.argmin(d))])
        return int(self.rng.choice(idx))

    @staticmethod
    def _pick_legs(u, z, y):
        side = -1 if u < AISLE_U else 1
        return [("face", side), ("move", z, y - DY), ("fork", u), ("lift", y), ("take",), ("lift", y + DY), ("fork", AISLE_U)]

    @staticmethod
    def _place_legs(u, z, y):
        side = -1 if u < AISLE_U else 1
        return [("face", side), ("move", z, y + DY), ("fork", u), ("lift", y), ("give",), ("lift", y - DY), ("fork", AISLE_U)]

    def _plan(self, task: dict) -> None:
        inner = (NEAR_U, self.ports[task["port"]], LEVELS[0])  # inner end of the port conveyor
        if task["kind"] == "STORE":
            c = self._pick_cell(True, inner[1])
            self.legs = self._pick_legs(*inner) + self._place_legs(*self.cell_pos(c))
        else:
            c = self._pick_cell(False, inner[1])
            self.legs = self._pick_legs(*self.cell_pos(c)) + self._place_legs(*inner)
        task["cell"] = c

    def _conveyors(self) -> None:
        for c in self.conv:
            if c[1] == "in" and self.t >= c[3]:
                c[1], c[2] = "inner_in", c[3]
                self.tasks.append({"kind": "STORE", "port": c[0], "conv": c})
            elif c[1] == "out" and self.t >= c[3]:
                c[1], c[2] = "stage", c[3]
                self.ready.add(c[4])
        for c in self.conv:  # start a move when the next position is free
            if c[1] == "stage_in" and not self._at(c[0], "in", "inner_in"):
                c[1], c[2], c[3] = "in", self.t, self.t + CONV_T
            elif c[1] == "inner_out" and not self._at(c[0], "out", "stage"):
                c[1], c[2], c[3] = "out", self.t, self.t + CONV_T

    def _next_task(self) -> Optional[dict]:
        """First executable task: a retrieval needs a free inner end on its OUT port."""
        for t in self.tasks:
            if t["kind"] == "STORE" or not self._at(t["port"], "inner_out"):
                self.tasks.remove(t)
                return t
        return None

    def update(self, dt: float) -> None:
        self.t += dt
        self._conveyors()
        budget = dt
        while budget > 1e-9:
            if not self.legs:
                if self.task is not None:
                    self.done += 1
                    self.task = None
                    self._conveyors()
                self.task = self._next_task()
                if self.task is None:
                    return
                self._plan(self.task)
            leg = self.legs[0]
            kind = leg[0]
            if kind == "face":
                if leg[1] == self.side:
                    self.legs.pop(0)
                    continue
                target = 0.0 if leg[1] < 0 else 1.0
                step = budget / TURN_T
                if abs(target - self.rot) <= step:
                    budget -= abs(target - self.rot) * TURN_T
                    self.rot, self.side = target, leg[1]
                    self.legs.pop(0)
                else:
                    self.rot += step if target > self.rot else -step
                    budget = 0
            elif kind in ("move", "lift"):
                tz = leg[1] if kind == "move" else self.z
                ty = leg[2] if kind == "move" else leg[1]
                dz, dy = tz - self.z, ty - self.y
                need = max(abs(dz) / TRAVEL_V, abs(dy) / (LIFT_V if kind == "move" else FINE_V))
                if need <= budget:
                    self.z, self.y = tz, ty
                    budget -= need
                    self.legs.pop(0)
                else:
                    f = budget / need
                    self.z += dz * f
                    self.y += dy * f
                    budget = 0
            elif kind == "fork":
                d = leg[1] - self.fork
                need = abs(d) / FORK_V
                if need <= budget:
                    self.fork = leg[1]
                    budget -= need
                    self.legs.pop(0)
                else:
                    self.fork += np.sign(d) * FORK_V * budget
                    budget = 0
            else:  # take / give: the fork top is flush with the shelf surface
                self._transfer(kind)
                self.legs.pop(0)

    def _transfer(self, kind: str) -> None:
        t = self.task
        if kind == "take":
            self.carry = True
            if t["kind"] == "STORE":
                self.conv = [c for c in self.conv if c is not t.get("conv")]
            else:
                self.occ[t["cell"]] = 0
        else:
            self.carry = False
            if t["kind"] == "STORE":
                self.occ[t["cell"]] = 1
            elif t["job"] in self.cancelled:
                self.cancelled.discard(t["job"])
            else:
                self.conv.append([t["port"], "inner_out", self.t, self.t, t["job"]])
                self._conveyors()

    def state(self) -> dict:
        return {"id": self.id, "z": round(self.z, 3), "y": round(self.y, 3), "fork": round(float(self.fork), 3),
                "rot": round(self.rot, 3), "carry": self.carry, "cells": "".join("1" if o else "0" for o in self.occ),
                "queue": len(self.tasks) + (1 if self.task else 0), "done": self.done,
                "conv": [[c[0], c[1], round(c[2], 2), round(c[3], 2)] for c in self.conv]}


def build_stockers(layout, seed: int) -> Dict[str, Stocker]:
    out = {}
    for k, a in enumerate(a for a in layout.areas.values() if a.kind == "STOCKER"):
        ps = [layout.ports[p] for p in a.ports]
        lane_x = ps[0].x
        direction = 1 if a.cx > lane_x else -1
        out[a.id] = Stocker(a.id, lane_x, direction, {p.id: p.z for p in ps}, seed * 7919 + 101 + k,
                            io={p.id: p.io for p in ps})
    return out
