"""Stocker interior: shelves + one stacker crane per stocker (visual model).

The crane stores FOUPs that OHTs drop on a stocker port and retrieves FOUPs to the port for
jobs that start at the stocker (retrievals have priority). An OHT that reaches the port before
the crane has delivered the FOUP waits there with its gripper up (fab metric stockerWaitTime).
It uses its own random generator so it does not change the demand random stream.

Local coordinates per stocker: u = metres from the OHT lane into the stocker (0 = under the
rail, where the I/O port shelves are), z = world z, y = height. worldX = laneX + dir * u.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

WALL_U, NEAR_U, AISLE_U, FAR_U, BACK_U = 0.6, 1.1, 2.3, 3.5, 4.0
LEVELS = [0.9, 1.75, 2.6, 3.45]  # shelf surface heights; port shelves are at LEVELS[0]
N_BAYS, BAY_PITCH = 12, 1.7
TRAVEL_V, LIFT_V, FORK_V, TRANSFER_T = 2.5, 1.0, 1.6, 0.6


class Stocker:
    def __init__(self, sid: str, lane_x: float, direction: int, ports: Dict[str, float], seed: int, fill: float = 0.6):
        self.id, self.lane_x, self.dir = sid, lane_x, direction
        self.ports = ports  # port id -> z
        z0 = -(N_BAYS - 1) * BAY_PITCH / 2
        self.bay_z = [z0 + i * BAY_PITCH for i in range(N_BAYS)]
        # cells: (rack 0 = near/lane side, 1 = far), bay, level. Near rack level 0 is the I/O opening.
        self.cells = [(r, b, l) for r in (0, 1) for b in range(N_BAYS) for l in range(len(LEVELS)) if not (r == 0 and l == 0)]
        self.rng = np.random.default_rng(seed)
        self.occ = (self.rng.random(len(self.cells)) < fill).astype(np.int8)
        self.tasks: deque = deque()
        self.port_items: Dict[str, int] = {p: 0 for p in ports}  # FOUPs waiting on a port to be stored
        self.ready: set = set()  # job ids whose FOUP the crane has put on the port
        self.cancelled: set = set()
        self.z, self.y, self.fork = 0.0, LEVELS[1], AISLE_U
        self.carry = False
        self.task: Optional[dict] = None
        self.legs: List[tuple] = []
        self.done = 0

    # ------------------------------------------------------------ geometry
    def cell_pos(self, i: int):
        r, b, l = self.cells[i]
        return (NEAR_U if r == 0 else FAR_U), self.bay_z[b], LEVELS[l]

    def to_dict(self) -> dict:
        return {"id": self.id, "laneX": self.lane_x, "dir": self.dir, "wallU": WALL_U, "nearU": NEAR_U, "aisleU": AISLE_U,
                "farU": FAR_U, "backU": BACK_U, "levels": LEVELS, "bayZ": self.bay_z, "bayPitch": BAY_PITCH,
                "cells": [list(self.cell_pos(i)) for i in range(len(self.cells))],
                "ports": [{"id": p, "z": z} for p, z in self.ports.items()]}

    # ------------------------------------------------------------ requests
    def store(self, port: str) -> None:
        self.port_items[port] += 1
        self.tasks.append({"kind": "STORE", "port": port})

    def retrieve(self, job_id: str, port: str) -> None:
        """Retrievals go before queued stores: an OHT may be waiting for this FOUP, a store can wait on its port."""
        task = {"kind": "RETRIEVE", "port": port, "job": job_id}
        for i, t in enumerate(self.tasks):
            if t["kind"] == "STORE":
                self.tasks.insert(i, task)
                return
        self.tasks.append(task)

    def picked(self, job_id: str) -> None:
        """The OHT has lifted the FOUP of job_id from the port."""
        if job_id in self.ready:
            self.ready.discard(job_id)
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

    def _plan(self, task: dict) -> None:
        pz = self.ports[task["port"]]
        if task["kind"] == "STORE":
            c = self._pick_cell(True, pz)
            u, z, y = self.cell_pos(c)
            src, dst = (0.0, pz, LEVELS[0]), (u, z, y)
        else:
            c = self._pick_cell(False, pz)
            u, z, y = self.cell_pos(c)
            src, dst = (u, z, y), (0.0, pz, LEVELS[0])
        task["cell"] = c
        # legs: (kind, target...)
        self.legs = [("move", src[1], src[2]), ("fork", src[0]), ("take",), ("fork", AISLE_U),
                     ("move", dst[1], dst[2]), ("fork", dst[0]), ("give",), ("fork", AISLE_U)]
        self.timer = 0.0

    def update(self, dt: float) -> None:
        budget = dt
        while budget > 1e-9:
            if not self.legs:
                if self.task is not None:
                    self._finish(self.task)
                    self.task = None
                if not self.tasks:
                    return
                self.task = self.tasks.popleft()
                self._plan(self.task)
            leg = self.legs[0]
            if leg[0] == "move":
                dz, dy = leg[1] - self.z, leg[2] - self.y
                need = max(abs(dz) / TRAVEL_V, abs(dy) / LIFT_V)
                if need <= budget:
                    self.z, self.y = leg[1], leg[2]
                    budget -= need
                    self.legs.pop(0)
                else:
                    f = budget / need
                    self.z += dz * f
                    self.y += dy * f
                    budget = 0
            elif leg[0] == "fork":
                d = leg[1] - self.fork
                need = abs(d) / FORK_V
                if need <= budget:
                    self.fork = leg[1]
                    budget -= need
                    self.legs.pop(0)
                else:
                    self.fork += np.sign(d) * FORK_V * budget
                    budget = 0
            else:  # take / give
                self.timer += budget
                if self.timer < TRANSFER_T:
                    budget = 0
                    continue
                budget = self.timer - TRANSFER_T
                self.timer = 0.0
                self._transfer(leg[0])
                self.legs.pop(0)

    def _transfer(self, kind: str) -> None:
        t = self.task
        if kind == "take":
            self.carry = True
            if t["kind"] == "STORE":
                self.port_items[t["port"]] = max(0, self.port_items[t["port"]] - 1)
            else:
                self.occ[t["cell"]] = 0
        else:
            self.carry = False
            if t["kind"] == "STORE":
                self.occ[t["cell"]] = 1
            elif t["job"] in self.cancelled:
                self.cancelled.discard(t["job"])
            else:
                self.ready.add(t["job"])

    def _finish(self, task: dict) -> None:
        self.done += 1

    def port_has_foup(self, port: str) -> bool:
        return self.port_items.get(port, 0) > 0

    def state(self) -> dict:
        return {"id": self.id, "z": round(self.z, 3), "y": round(self.y, 3), "fork": round(float(self.fork), 3),
                "carry": self.carry, "cells": "".join("1" if o else "0" for o in self.occ),
                "queue": len(self.tasks) + (1 if self.task else 0), "done": self.done}


def build_stockers(layout, seed: int) -> Dict[str, Stocker]:
    out = {}
    for k, a in enumerate(a for a in layout.areas.values() if a.kind == "STOCKER"):
        ps = [layout.ports[p] for p in a.ports]
        lane_x = ps[0].x
        direction = 1 if a.cx > lane_x else -1
        out[a.id] = Stocker(a.id, lane_x, direction, {p.id: p.z for p in ps}, seed * 7919 + 101 + k)
    return out
