"""Closed-loop lot flow: process tools with EFEM load ports, lots and stocker queues.

Every bay port is a load port (LP) with room for exactly one FOUP. Neighbouring ports on a lane
form one tool (an EFEM with 2 load ports, like most process tools); the tool processes one lot
at a time while the other LP can already receive and dock the next lot:

    EMPTY -> RESERVED (a FOUP is on its way) -> REST (just put down) -> DOCK (FOUP slides
    against the EFEM, door opens) -> QUEUE (docked, waiting for the process chamber) -> PROC
    (wafers processed, FOUP stays docked) -> UNDOCK (door closes, FOUP slides back) -> WAIT
    (transport request issued) -> EMPTY (OHT took it)

When a lot finishes at a tool it moves to its next process step (processFlow). Its destination
is a free LP of a tool for that process, reserved at once, so an OHT never delivers to an
occupied load port. If no such LP is free (or the lot must queue: stockerShare), it goes to a
stocker. Stocker lots wait per next process and are sent out when an LP of that process becomes
free. WIP is constant (lots re-enter the flow), like a loaded fab.

Transport requests therefore come from the fab itself: faster dispatching frees load ports
sooner, keeps tools busier and raises throughput.

Per-lot randomness (next process, processing time, stocker diversion, LP pick) is drawn from a
generator keyed by (seed, lot, step), so it does not depend on the order in which the policy
causes events. Two policies on the same seed face the same lots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .demand import Job

EMPTY, RESERVED, REST, DOCK, QUEUE, PROC, UNDOCK, WAIT = "E", "V", "R", "D", "Q", "P", "U", "W"


@dataclass
class Lot:
    id: int
    hot: bool
    step: int = 0
    proc: str = ""  # process of the current / next step


@dataclass
class LoadPort:
    port: str
    area: str
    process: str
    state: str = EMPTY
    lot: Optional[Lot] = None
    t0: float = 0.0  # current phase start
    t1: float = 0.0  # current phase end
    job: Optional[str] = None  # reserving (incoming) or outgoing job id
    busy_time: float = 0.0  # seconds spent processing (utilization)
    tool: str = ""


@dataclass
class Tool:
    id: str
    process: str
    area: str
    ports: List[str]
    active: Optional[str] = None  # LP being processed
    mode: str = "auto"  # auto | manual | alarm (tower lamp red; no new lot is started). Never set by the flow itself.
    busy_time: float = 0.0


class LotFlow:
    def __init__(self, dcfg: dict, layout):
        self.cfg = dcfg
        self.layout = layout
        self.lps: Dict[str, LoadPort] = {p.id: LoadPort(p.id, p.area, p.process) for p in layout.ports.values()
                                         if p.kind == "TOOL"}
        self.lps_by_proc: Dict[str, List[str]] = {}
        for lp in self.lps.values():
            self.lps_by_proc.setdefault(lp.process, []).append(lp.port)
        # tools: consecutive ports of a lane, `portsPerTool` each (B1-L01+L02 -> tool B1-T01 ...)
        per = int(dcfg.get("portsPerTool", 2))
        self.tools: Dict[str, Tool] = {}
        lanes: Dict[tuple, List[str]] = {}
        for pid, p in layout.ports.items():
            if p.kind == "TOOL":
                lanes.setdefault((p.area, p.edge), []).append(pid)
        n = 0
        for (area, _), pids in sorted(lanes.items(), key=lambda kv: kv[1][0]):
            pids.sort(key=lambda q: layout.ports[q].offset)
            for i in range(0, len(pids), per):
                n += 1
                group = pids[i:i + per]
                tid = f"{area}-T{len([t for t in self.tools.values() if t.area == area]) + 1:02d}"
                self.tools[tid] = Tool(tid, layout.ports[group[0]].process, area, group)
                for q in group:
                    self.lps[q].tool = tid
        self.stockers = [a for a in layout.area_ids if layout.areas[a].kind == "STOCKER"]
        self.rest_t = float(dcfg.get("restTime", 1.0))
        self.dock_t = float(dcfg.get("dockTime", 2.0))

    # ------------------------------------------------------------ randomness
    def _u(self, lot: Lot, k: int = 6) -> np.ndarray:
        return np.random.default_rng([self.seed, lot.id, lot.step]).random(k)

    def _phase(self, t: float) -> dict:
        ph = self.cfg["phases"][0]
        for p in self.cfg["phases"]:
            if t >= p["start"]:
                ph = p
        return ph

    def speed(self, proc: str, t: float) -> float:
        ph = self._phase(t)
        return float(self.cfg.get("speed", 1.0)) * ph.get("multiplier", 1.0) * ph.get("boost", {}).get(proc, 1.0)

    def process_time(self, lot: Lot, proc: str, t: float) -> float:
        base = self.cfg["processTime"].get(proc, 300.0)
        jit = float(self.cfg.get("processJitter", 0.2))
        return base * (1 - jit + 2 * jit * self._u(lot)[0]) / self.speed(proc, t)

    # ------------------------------------------------------------ setup
    def reset(self, seed: int) -> List[Job]:
        self.seed = int(seed)
        self.seq = 0
        self.lot_seq = 0
        self.stock: Dict[str, List[Tuple[Lot, str]]] = {}  # next process -> [(lot, stocker area)]
        self.new_jobs: List[Job] = []
        self.processed = 0
        rng = np.random.default_rng([self.seed, 999_983])
        occ = float(self.cfg.get("initialOccupancy", 0.85))
        hot_p = float(self.cfg["hotLotProbability"])
        for lp in self.lps.values():
            lp.state, lp.lot, lp.job, lp.busy_time = EMPTY, None, None, 0.0
        for tool in self.tools.values():
            tool.active, tool.busy_time, tool.mode = None, 0.0, "auto"
            for i, q in enumerate(tool.ports):  # already loaded: one lot in process, others docked in queue
                if rng.random() >= occ:
                    continue
                lp = self.lps[q]
                lot = self._new_lot(rng.random() < hot_p, lp.process)
                if tool.active is None:
                    dur = self.process_time(lot, lp.process, 0.0)
                    start = -float(rng.uniform(0, dur))
                    lp.state, lp.lot, lp.t0, lp.t1 = PROC, lot, start, start + dur
                    tool.active = q
                else:
                    lp.state, lp.lot, lp.t0, lp.t1 = QUEUE, lot, 0.0, 0.0
        per = int(self.cfg.get("stockerWipPerProcess", 2))
        for k, proc in enumerate(sorted(self.lps_by_proc)):
            for i in range(per):
                lot = self._new_lot(rng.random() < hot_p, proc)
                self.stock.setdefault(proc, []).append((lot, self.stockers[(k + i) % len(self.stockers)]))
        self._fill(0.0)
        out, self.new_jobs = self.new_jobs, []
        return out

    def _new_lot(self, hot: bool, proc: str) -> Lot:
        self.lot_seq += 1
        return Lot(self.lot_seq, bool(hot), 0, proc)

    def _job(self, src: str, dst: str, src_area: str, dst_area: str, lot: Lot, t: float) -> Job:
        self.seq += 1
        j = Job(f"J{self.seq:05d}", src, dst, t, lot.hot, t + self.cfg["hotLotDeadline"] if lot.hot else math.inf,
                src_area, dst_area)
        j.lot = lot
        self.new_jobs.append(j)
        return j

    # ------------------------------------------------------------ routing of a finished lot
    def _route(self, lot: Lot, src: str, src_area: str, t: float) -> Job:
        u = self._u(lot)
        flow = self.cfg["processFlow"].get(lot.proc, []) or sorted(self.lps_by_proc)
        lot.step += 1
        lot.proc = flow[min(int(u[1] * len(flow)), len(flow) - 1)]
        free = [p for p in self.lps_by_proc.get(lot.proc, []) if self.lps[p].state == EMPTY]
        if free and u[2] >= float(self.cfg.get("stockerShare", 0.2)):
            dst = free[min(int(u[3] * len(free)), len(free) - 1)]
            lp = self.lps[dst]
            j = self._job(src, dst, src_area, lp.area, lot, t)
            lp.state, lp.job = RESERVED, j.id
            return j
        sk = self.stockers[min(int(u[4] * len(self.stockers)), len(self.stockers) - 1)]
        ports = self.layout.areas[sk].ports
        return self._job(src, ports[min(int(u[5] * len(ports)), len(ports) - 1)], src_area, sk, lot, t)

    def _fill(self, t: float) -> None:
        """Send waiting stocker lots to free load ports of their next process (hot lots first)."""
        for proc, q in self.stock.items():
            if not q:
                continue
            free = [p for p in self.lps_by_proc.get(proc, []) if self.lps[p].state == EMPTY]
            q.sort(key=lambda x: (not x[0].hot, x[0].id))
            while q and free:
                lot, sk = q.pop(0)
                u = self._u(lot)
                dst = free.pop(min(int(u[3] * len(free)), len(free) - 1))
                ports = self.layout.areas[sk].ports
                src = ports[min(int(u[5] * len(ports)), len(ports) - 1)]
                lp = self.lps[dst]
                j = self._job(src, dst, sk, lp.area, lot, t)
                lp.state, lp.job = RESERVED, j.id

    # ------------------------------------------------------------ events from the transport system
    def delivered(self, job: Job, t: float) -> None:
        lot = job.lot
        if job.dst in self.lps:
            lp = self.lps[job.dst]
            lp.state, lp.lot, lp.job, lp.t0, lp.t1 = REST, lot, None, t, t + self.rest_t
        else:  # stocker: queue for the next process
            self.stock.setdefault(lot.proc, []).append((lot, job.dst_area))
            self._fill(t)

    def picked(self, job: Job, t: float) -> None:
        lp = self.lps.get(job.src)
        if lp is not None and lp.job == job.id:
            lp.state, lp.lot, lp.job = EMPTY, None, None
            self._fill(t)

    # ------------------------------------------------------------ time
    def update(self, t: float, dt: float) -> List[Job]:
        for tool in self.tools.values():
            if tool.active is not None:
                tool.busy_time += dt
            for q in tool.ports:
                lp = self.lps[q]
                while lp.state in (REST, DOCK, PROC, UNDOCK) and t + 1e-9 >= lp.t1:
                    if lp.state == REST:
                        lp.state, lp.t0, lp.t1 = DOCK, lp.t1, lp.t1 + self.dock_t
                    elif lp.state == DOCK:
                        lp.state, lp.t0 = QUEUE, lp.t1
                    elif lp.state == PROC:
                        self.processed += 1
                        tool.active = None
                        lp.state, lp.t0, lp.t1 = UNDOCK, lp.t1, lp.t1 + self.dock_t
                    else:  # UNDOCK finished: the lot asks for transport
                        j = self._route(lp.lot, lp.port, lp.area, t)
                        lp.state, lp.job, lp.t0 = WAIT, j.id, t
            if tool.active is None and tool.mode == "auto":  # chamber free: start the docked lot that has waited longest (hot first)
                queued = [self.lps[q] for q in tool.ports if self.lps[q].state == QUEUE]
                if queued:
                    lp = min(queued, key=lambda x: (not x.lot.hot, x.t0))
                    lp.state, lp.t0, lp.t1 = PROC, t, t + self.process_time(lp.lot, lp.process, t)
                    tool.active = lp.port
        out, self.new_jobs = self.new_jobs, []
        return out

    def drain(self) -> List[Job]:
        out, self.new_jobs = self.new_jobs, []
        return out

    # ------------------------------------------------------------ view / KPI
    def set_mode(self, tool_id: str, mode: str) -> dict:
        """Operator / equipment state: 'manual' or 'alarm' stops the tool from starting new lots."""
        if tool_id not in self.tools or mode not in ("auto", "manual", "alarm"):
            raise ValueError(f"unknown tool or mode: {tool_id} {mode}")
        self.tools[tool_id].mode = mode
        return {"tool": tool_id, "mode": mode}

    def modes(self) -> Dict[str, str]:
        return {t.id: t.mode for t in self.tools.values() if t.mode != "auto"}

    def view(self, skip: set) -> List[list]:
        """[port, state, t0, t1, hot] for load ports holding a FOUP (except ports in skip)."""
        return [[lp.port, lp.state, round(lp.t0, 2), round(lp.t1, 2), int(lp.lot.hot) if lp.lot else 0]
                for lp in self.lps.values() if lp.state not in (EMPTY, RESERVED) and lp.port not in skip] + \
               [[lp.port, RESERVED, 0, 0, 0] for lp in self.lps.values() if lp.state == RESERVED]

    def utilization(self, t: float) -> float:
        if t <= 0:
            return 0.0
        return sum(tl.busy_time for tl in self.tools.values()) / (len(self.tools) * t)

    def tool_layout(self) -> List[dict]:
        out = []
        for tl in self.tools.values():
            ps = [self.layout.ports[q] for q in tl.ports]
            out.append({"id": tl.id, "process": tl.process, "area": tl.area, "ports": tl.ports,
                        "x": ps[0].tool_x, "z": sum(p.z for p in ps) / len(ps),
                        "zMin": min(p.z for p in ps), "zMax": max(p.z for p in ps)})
        return out

    def stock_count(self) -> int:
        return sum(len(q) for q in self.stock.values())
