"""FabSimulation: OHT vehicles on a one-way rail network (single source of truth).

Physics: each vehicle is (edge, offset s, speed v). Per fixed step it obeys the
edge speed limit (curves are slow), braking-distance limits toward its stop
point, a minimum gap to the vehicle ahead along its own path, FIFO locks at
merge nodes, and blocked (maintenance) edges. Vehicles stop at ports for a
hoist cycle, which blocks the rail behind them.

Decisions (made by a controller, never by the physics):
  DISPATCH    which idle vehicle takes a pending transport job (or defer)
  ROUTE       which out-edge to take at a crossover diverge (shortcut vs around)
  REPOSITION  which bay an idle vehicle should wait (circulate) in
"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import resolve_config
from .core import EventManager, SeededRandom, SimulationClock
from .demand import DemandGenerator, Job
from .layout import FabLayout

IDLE, TO_PICKUP, LOADING, TO_DROP, UNLOADING = "IDLE", "TO_PICKUP", "LOADING", "TO_DROP", "UNLOADING"
DISPATCH, ROUTE, REPOSITION = "DISPATCH", "ROUTE", "REPOSITION"


@dataclass
class Vehicle:
    id: int
    edge: int
    s: float
    v: float = 0.0
    state: str = IDLE
    job: Optional[str] = None
    target: Optional[Tuple[int, float]] = None
    stop_at_target: bool = False
    home: Optional[str] = None
    forced: Dict[int, int] = field(default_factory=dict)  # node -> chosen out-edge (route decisions)
    decided_nodes: set = field(default_factory=set)
    timer: float = 0.0
    needs_reposition: bool = False
    blocked_time: float = 0.0
    moved: float = 0.0
    next_job: Optional[str] = None  # pre-assigned job, started right after the current drop


@dataclass
class Decision:
    kind: str
    subject: str  # job id (DISPATCH) or vehicle id (ROUTE/REPOSITION)
    node: int = -1
    candidates: List[dict] = field(default_factory=list)


class FabSimulation:
    def __init__(self, config=None, overrides: Optional[dict] = None):
        self.cfg = resolve_config(config, overrides)
        self.layout = FabLayout(self.cfg)
        self.track = self.layout.track
        sim = self.cfg["simulation"]
        self.dt = float(sim["dt"])
        self.episode_length = float(sim["episodeLength"])
        self.decision_interval = float(sim["decisionInterval"])
        self.defer_retry = float(sim["deferRetry"])
        self.deadlock_time = float(sim["deadlockTime"])
        vc = self.cfg["vehicles"]
        self.vlen, self.gap = float(vc["length"]), float(vc["minGap"])
        self.accel, self.decel = float(vc["accel"]), float(vc["decel"])
        self.hoist = float(vc["hoistTime"])
        self.idle_speed = float(vc["idleSpeed"])
        self.merge_req = float(self.cfg["merge"]["requestDistance"])
        self.stop_margin = float(self.cfg["merge"]["stopMargin"])
        self.max_detour = float(self.cfg["routing"]["maxDetour"])
        self.route_lookahead = float(self.cfg["routing"]["decisionLookahead"])
        self.K = int(self.cfg["agent"]["candidates"])
        dc = self.cfg.get("dispatch", {})
        self.preassign = bool(dc.get("preassign", True))
        self.preassign_horizon = float(dc.get("preassignHorizon", 90.0))
        self.min_busy_slots = int(dc.get("minBusySlots", 3))
        self.clock = SimulationClock(self.dt)
        self.events = EventManager(400)
        self.demand = DemandGenerator(self.cfg["demand"], self.layout)
        self.block_edges: Dict[str, List[int]] = {}
        for tag, edges in self.layout.crossover_edges.items():
            self.block_edges[tag] = edges
        self.reset(int(self.cfg["demand"].get("seed", 7)))

    # ================================================================= reset
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = SeededRandom(self.seed)
        self.clock.reset()
        self.events.reset()
        self.demand.reset(self.rng.orders)
        for e in self.track.edges:
            e.blocked = False
        self.track.build_routing()
        self.jobs: Dict[str, Job] = {}
        self.done_jobs: List[Job] = []
        self.merge_queue: Dict[int, deque] = {n.id: deque() for n in self.track.nodes if n.is_merge}
        self.blocks = [dict(b, active=False) for b in self.cfg.get("blocks", [])]
        self.metrics = {"delivered": 0, "deliveryTimes": [], "waitTimes": [], "emptyMeters": 0.0, "loadedMeters": 0.0,
                        "busyTime": 0.0, "blockedTime": 0.0, "hotDelivered": 0, "hotOnTime": 0, "hotLate": 0,
                        "created": 0, "deadlock": False, "decisions": {DISPATCH: 0, ROUTE: 0, REPOSITION: 0},
                        "defers": 0, "mergeWaitTime": 0.0, "preassigned": 0}
        self.last_progress_t = 0.0
        self.last_decision_t = -1e9
        self.recent_jobs: deque = deque()  # (t, src_area) for demand-rate features
        self._spawn_vehicles(int(self.cfg["vehicles"]["count"]))
        self.events.emit("RESET", 0.0, seed=self.seed, scenario=self.cfg.get("scenario"))

    def _spawn_vehicles(self, n: int) -> None:
        """Distribute vehicles evenly over the bays' intrabay loops (idle, circulating)."""
        self.vehicles: List[Vehicle] = []
        bays = self.layout.bay_ids
        per = [n // len(bays) + (1 if i < n % len(bays) else 0) for i in range(len(bays))]
        vid = 0
        for b, k in zip(bays, per):
            loop = self.layout.areas[b].loop_edges
            total = sum(self.track.edges[e].length for e in loop)
            for j in range(k):
                d = total * j / max(k, 1)
                for e in loop:
                    L = self.track.edges[e].length
                    if d < L:
                        break
                    d -= L
                veh = Vehicle(vid, e, min(d, self.track.edges[e].length - 1e-3), home=b)
                self._set_idle_target(veh)
                self.vehicles.append(veh)
                vid += 1

    @property
    def time(self) -> float:
        return self.clock.time

    @property
    def done(self) -> bool:
        return self.time + 1e-9 >= self.episode_length or self.metrics["deadlock"]

    # ============================================================== routing
    def _target_point(self, port_id: str) -> Tuple[int, float]:
        p = self.layout.ports[port_id]
        return p.edge, p.offset

    def _set_idle_target(self, veh: Vehicle) -> None:
        a = self.layout.areas[veh.home]
        veh.target = (a.park_edge, a.park_offset)
        veh.stop_at_target = False

    def next_edge(self, veh: Vehicle, node: int, peek: bool = True) -> int:
        te = veh.target[0]
        T = self.track
        if node == T.edges[te].u:
            return te
        if node in veh.forced:
            e = veh.forced[node]
            if not peek:
                veh.forced.pop(node, None)
            return e
        e = int(T.next_edge[node, T.edges[te].u])
        if e < 0:  # unreachable (all routes blocked): take any open out-edge
            outs = [o for o in T.nodes[node].out_edges if not T.edges[o].blocked] or T.nodes[node].out_edges
            e = outs[0]
        return e

    def path_ahead(self, veh: Vehicle, horizon: float) -> List[Tuple[int, float]]:
        """[(edge, distance from vehicle to the start of that edge)], starting with the current edge (dist <= 0)."""
        T = self.track
        out = [(veh.edge, -veh.s)]
        d = T.edges[veh.edge].length - veh.s
        e = veh.edge
        guard = 0
        while d < horizon and guard < 30:
            n = T.edges[e].v
            e = self.next_edge(veh, n, peek=True)
            out.append((e, d))
            d += T.edges[e].length
            guard += 1
        return out

    def dist_to_target(self, veh: Vehicle, path) -> Optional[float]:
        te, to = veh.target
        for e, d in path:
            if e == te:
                dd = d + to
                if dd >= -1e-6:
                    return dd
        return None

    # ================================================================ step
    def step(self) -> dict:
        dt = self.dt
        t = self.clock.tick()
        res = {"delivered": [], "hotLate": 0, "emptyMeters": 0.0, "blockedVehicles": 0, "inSystem": 0, "hotInSystem": 0,
               "deadlock": False}
        self._update_blocks(t)
        for j in self.demand.generate(t):
            self.jobs[j.id] = j
            self.metrics["created"] += 1
            self.recent_jobs.append((t, j.src_area))
            self.events.emit("JOB", t, jobId=j.id, src=j.src, dst=j.dst, hot=j.hot)
        while self.recent_jobs and self.recent_jobs[0][0] < t - 600:
            self.recent_jobs.popleft()
        T = self.track
        # occupancy index: edge -> sorted list of (s, vehicle)
        occ: Dict[int, List[Tuple[float, Vehicle]]] = {}
        for veh in self.vehicles:
            occ.setdefault(veh.edge, []).append((veh.s, veh))
        for lst in occ.values():
            lst.sort(key=lambda x: x[0])
        paths = {}
        for veh in self.vehicles:
            horizon = max(12.0, veh.v * veh.v / (2 * self.decel) + self.merge_req + 4.0)
            paths[veh.id] = self.path_ahead(veh, horizon)
        self._update_merge_locks(paths)
        moved_any = False
        new_pos = []
        for veh in self.vehicles:
            if veh.state in (LOADING, UNLOADING):
                veh.timer -= dt
                new_pos.append(None)
                continue
            path = paths[veh.id]
            vmax = T.edges[veh.edge].vmax
            if veh.state == IDLE:
                vmax = min(vmax, self.idle_speed)
            limit = vmax
            hard = math.inf  # max displacement this step
            # speed limits of upcoming edges (curves)
            for e, d in path[1:]:
                if d > 0:
                    ev = T.edges[e].vmax if veh.state != IDLE else min(T.edges[e].vmax, self.idle_speed)
                    limit = min(limit, math.sqrt(ev * ev + 2 * self.decel * d))
                    if T.edges[e].blocked:
                        stop = max(0.0, d - self.stop_margin)
                        limit = min(limit, math.sqrt(2 * self.decel * stop))
                        hard = min(hard, stop)
                        break
            # stop at the target port
            if veh.stop_at_target:
                dtg = self.dist_to_target(veh, path)
                if dtg is not None:
                    limit = min(limit, math.sqrt(2 * self.decel * max(0.0, dtg)))
                    hard = min(hard, dtg)
            # leader along own path
            lead = self._leader_gap(veh, path, occ)
            if lead is not None:
                free = max(0.0, lead - self.vlen - self.gap)
                limit = min(limit, math.sqrt(2 * self.decel * free))
                hard = min(hard, free)
            # merge locks
            for e, d in path[1:]:
                n = T.edges[e].u
                if d > self.merge_req:
                    break
                if n in self.merge_queue and d >= 0:
                    q = self.merge_queue[n]
                    if not q or q[0] != veh.id:
                        stop = max(0.0, d - self.stop_margin)
                        limit = min(limit, math.sqrt(2 * self.decel * stop))
                        hard = min(hard, stop)
                        if veh.v < 0.05 and stop < 0.5:
                            self.metrics["mergeWaitTime"] += dt
                        break
            vnew = min(veh.v + self.accel * dt, limit)
            vnew = max(0.0, vnew)
            disp = min(vnew * dt, hard)
            disp = max(0.0, disp)
            new_pos.append(disp)
        # apply moves (synchronous: constraints used previous positions -> no overlap)
        for veh, disp in zip(self.vehicles, new_pos):
            if disp is None:
                if veh.timer <= 0:
                    self._finish_hoist(veh, t)
                continue
            veh.v = disp / dt
            if disp > 1e-9:
                moved_any = True
                self._advance(veh, disp)
                if veh.state in (IDLE, TO_PICKUP):
                    self.metrics["emptyMeters"] += disp
                    res["emptyMeters"] += disp
                else:
                    self.metrics["loadedMeters"] += disp
            if veh.state in (TO_PICKUP, TO_DROP) and veh.v < 0.05:
                veh.blocked_time += dt
                self.metrics["blockedTime"] += dt
                res["blockedVehicles"] += 1
            if veh.state != IDLE:
                self.metrics["busyTime"] += dt
            # arrival at target port
            if veh.stop_at_target and veh.edge == veh.target[0] and abs(veh.s - veh.target[1]) < 0.02:
                self._arrive(veh, t)
            elif veh.state == IDLE and veh.edge == veh.target[0] and veh.s >= veh.target[1] - 1e-6:
                pass  # circulating: target stays, vehicle loops around its bay
        # job bookkeeping
        for j in self.jobs.values():
            if j.state != "DONE":
                res["inSystem"] += 1
                if j.hot:
                    res["hotInSystem"] += 1
                    if not j.late and t > j.deadline:
                        j.late = True
                        res["hotLate"] += 1
                        self.metrics["hotLate"] += 1
        for jid in [k for k, j in self.jobs.items() if j.state == "DONE"]:
            res["delivered"].append(self.jobs.pop(jid))
        busy = any(v.state != IDLE for v in self.vehicles)
        if moved_any or not busy:
            self.last_progress_t = t
        elif t - self.last_progress_t > self.deadlock_time:
            self.metrics["deadlock"] = True
            res["deadlock"] = True
            self.events.emit("DEADLOCK", t)
        return res

    def _leader_gap(self, veh, path, occ) -> Optional[float]:
        """Distance from this vehicle's center to the nearest vehicle center ahead on its path.

        Right after a diverge node the branches are still side by side, so vehicles
        that have just entered a *sibling* out-edge also count as being ahead.
        """
        T = self.track
        zone = self.vlen + self.gap + 1.5
        best = None
        for i, (e, d) in enumerate(path):
            lst = occ.get(e)
            if lst:
                for s, other in lst:
                    if other is veh:
                        continue
                    if i == 0:
                        if s > veh.s + 1e-9 or (s == veh.s and other.id > veh.id):
                            cand = s - veh.s
                            best = cand if best is None else min(best, cand)
                            break
                    else:
                        cand = d + s
                        best = cand if best is None else min(best, cand)
                        break
            # siblings of this edge at a diverge
            node = T.nodes[T.edges[e].u]
            if len(node.out_edges) > 1:
                base = d if i > 0 else -veh.s
                for sib in node.out_edges:
                    if sib == e:
                        continue
                    for s, other in occ.get(sib, ()):
                        if other is veh or s > zone:
                            continue
                        cand = base + s
                        if cand > 1e-9 or (i == 0 and s > veh.s):
                            best = cand if best is None else min(best, cand)
            if best is not None and best < d:
                break
        return best

    def _advance(self, veh: Vehicle, disp: float) -> None:
        T = self.track
        veh.s += disp
        veh.moved += disp
        while veh.s > T.edges[veh.edge].length - 1e-9:
            over = veh.s - T.edges[veh.edge].length
            n = T.edges[veh.edge].v
            veh.decided_nodes.discard(n)
            veh.edge = self.next_edge(veh, n, peek=False)
            veh.s = over
            if over <= 1e-9:
                break

    def _update_merge_locks(self, paths) -> None:
        T = self.track
        want: Dict[int, List[Tuple[float, int]]] = {}
        for veh in self.vehicles:
            path = paths[veh.id]
            stop_d = self.dist_to_target(veh, path) if veh.stop_at_target else None
            for e, d in path[1:]:
                if d > self.merge_req:
                    break
                n = T.edges[e].u
                if n in self.merge_queue and d >= 0:
                    if stop_d is not None and stop_d < d:
                        break  # stops at a port before the merge: does not claim it
                    want.setdefault(n, []).append((d, veh.id))
                    break
        for n, q in self.merge_queue.items():
            # release owner once it has cleared the node (it no longer appears as approaching)
            approaching = {vid for _, vid in want.get(n, [])}
            while q and q[0] not in approaching and self._cleared(n, q[0]):
                q.popleft()
            for vid in list(q):
                if vid not in approaching and vid != (q[0] if q else None):
                    q.remove(vid)
            for _, vid in sorted(want.get(n, [])):
                if vid not in q:
                    q.append(vid)

    def _cleared(self, node: int, vid: int) -> bool:
        veh = self.vehicles[vid]
        e = self.track.edges[veh.edge]
        if e.u == node:
            return veh.s >= self.vlen + self.gap
        return True

    # ============================================================ hoisting
    def _arrive(self, veh: Vehicle, t: float) -> None:
        veh.v = 0.0
        j = self.jobs.get(veh.job)
        if veh.state == TO_PICKUP and j:
            veh.state, j.state = LOADING, "LOADING"
            veh.timer = self.hoist
        elif veh.state == TO_DROP and j:
            veh.state, j.state = UNLOADING, "UNLOADING"
            veh.timer = self.hoist
        veh.stop_at_target = False

    def _finish_hoist(self, veh: Vehicle, t: float) -> None:
        j = self.jobs.get(veh.job)
        if veh.state == LOADING:
            veh.state, j.state, j.picked_t = TO_DROP, "DELIVERING", t
            veh.target = self._target_point(j.dst)
            veh.stop_at_target = True
            veh.forced.clear()
            self.events.emit("PICKUP", t, jobId=j.id, vehicle=veh.id)
        elif veh.state == UNLOADING:
            j.state, j.done_t = "DONE", t
            m = self.metrics
            m["delivered"] += 1
            m["deliveryTimes"].append(t - j.created)
            if j.hot:
                m["hotDelivered"] += 1
                if t <= j.deadline:
                    m["hotOnTime"] += 1
            self.done_jobs.append(j)
            if len(self.done_jobs) > 300:
                self.done_jobs = self.done_jobs[-300:]
            self.events.emit("DELIVERED", t, jobId=j.id, vehicle=veh.id, duration=round(t - j.created, 1), hot=j.hot)
            veh.state, veh.job = IDLE, None
            nj = self.jobs.get(veh.next_job) if veh.next_job else None
            veh.next_job = None
            if nj is not None:
                self._start_pickup(veh, nj, t)
                return
            for pj in self.jobs.values():  # a vehicle became free: deferred jobs may ask again
                if pj.state == "PENDING" and pj.defer_until > t:
                    pj.defer_until = t
            area = self.layout.areas[self.layout.ports[j.dst].area]
            veh.home = area.id if area.kind == "BAY" else self._nearest_bay(veh)
            self._set_idle_target(veh)
            veh.forced.clear()
            veh.needs_reposition = True

    def _start_pickup(self, v: Vehicle, j: Job, t: float) -> None:
        j.state, j.vehicle = "ASSIGNED", v.id
        self.metrics["waitTimes"].append(t - j.created)  # wait until a vehicle actually heads to the job
        v.state, v.job = TO_PICKUP, j.id
        v.target = self._target_point(j.src)
        v.stop_at_target = True
        v.needs_reposition = False
        v.forced.clear()
        v.decided_nodes.clear()

    def _nearest_bay(self, veh: Vehicle) -> str:
        n = self.track.edges[veh.edge].v
        return min(self.layout.bay_ids, key=lambda b: self.track.meters_from(n, self.layout.areas[b].park_edge, 0.0))

    # ============================================================== blocks
    def _update_blocks(self, t: float) -> None:
        changed = False
        for b in self.blocks:
            active = b["start"] <= t < b["end"]
            if active != b["active"]:
                b["active"] = active
                for eid in self.block_edges.get(b["edgeTag"], []):
                    # the entry edge of the crossover is closed; vehicles already inside drive out
                    self.track.edges[eid].blocked = active and eid == self.block_edges[b["edgeTag"]][0]
                changed = True
                self.events.emit("TRACK_BLOCK" if active else "TRACK_OPEN", t, tag=b["edgeTag"])
        if changed:
            self.track.build_routing()
            for veh in self.vehicles:
                veh.forced = {n: e for n, e in veh.forced.items() if not self.track.edges[e].blocked}

    # ============================================================ decisions
    def idle_vehicles(self) -> List[Vehicle]:
        return [v for v in self.vehicles if v.state == IDLE]

    def preassignable(self) -> List[Vehicle]:
        if not self.preassign:
            return []
        return [v for v in self.vehicles if v.state in (TO_DROP, UNLOADING) and v.next_job is None]

    def _time_to_free(self, v: Vehicle) -> Tuple[float, float]:
        """Seconds until v finishes its drop, and nothing else. Drop point (edge, s) is v.target."""
        if v.state == UNLOADING:
            return v.timer, 0.0
        te, to = v.target
        eta, _ = self._eta_to(v, te, to)
        return eta + self.hoist, 0.0

    def _eta_from_point(self, edge: int, s: float, pe: int, po: float) -> Tuple[float, float]:
        T = self.track
        e = T.edges[edge]
        if edge == pe and po >= s:
            return (po - s) / e.vmax, po - s
        rem = e.length - s
        return rem / e.vmax + T.time_from(e.v, pe, po), rem + T.meters_from(e.v, pe, po)

    def pending_jobs(self) -> List[Job]:
        js = [j for j in self.jobs.values() if j.state == "PENDING"]
        js.sort(key=lambda j: (not j.hot, j.created))
        return js

    def decisions(self) -> List[Decision]:
        """Decisions needed now. ROUTE first (time-critical), then DISPATCH, then REPOSITION."""
        out: List[Decision] = []
        T = self.track
        for veh in self.vehicles:
            if veh.state in (LOADING, UNLOADING) or veh.target is None:
                continue
            path = self.path_ahead(veh, self.route_lookahead)
            for e, d in path[1:]:
                n = T.edges[e].u
                if d > self.route_lookahead:
                    break
                if n in self.layout.decision_nodes and n not in veh.decided_nodes and veh.state != IDLE:
                    cands = self.route_candidates(veh, n)
                    if len(cands) >= 2:
                        out.append(Decision(ROUTE, str(veh.id), n, cands))
                    else:
                        veh.decided_nodes.add(n)
                    break
        due = self.time - self.last_decision_t >= self.decision_interval - 1e-9
        if due:
            n_before = len(out)
            idle = self.idle_vehicles()
            if idle or self.preassignable():
                for j in self.pending_jobs():
                    if j.defer_until <= self.time + 1e-9:
                        out.append(Decision(DISPATCH, j.id, -1, self.dispatch_candidates(j)))
                        break  # one dispatch at a time (state changes after each)
            for veh in self.vehicles:
                if veh.needs_reposition and veh.state == IDLE:
                    out.append(Decision(REPOSITION, str(veh.id), -1, self.reposition_candidates(veh)))
                    break
            if len(out) == n_before:  # nothing left to dispatch/reposition this round
                self.last_decision_t = self.time
        return out

    def mark_decision_round(self) -> None:
        """Kept for API compatibility; rounds close automatically in decisions()."""
        return None

    def _vehicle_node(self, veh: Vehicle) -> int:
        return self.track.edges[veh.edge].v

    def _eta_to(self, veh: Vehicle, edge: int, offset: float) -> Tuple[float, float]:
        T = self.track
        e = T.edges[veh.edge]
        if veh.edge == edge and offset >= veh.s:
            d = offset - veh.s
            return d / e.vmax, d
        rem = e.length - veh.s
        n = e.v
        return rem / e.vmax + T.time_from(n, edge, offset), rem + T.meters_from(n, edge, offset)

    def _congestion(self, edges_ahead: List[int], horizon: float = 60.0) -> Tuple[int, int]:
        """(vehicles, stopped vehicles) on the first `horizon` meters of an edge sequence."""
        T = self.track
        ids = set(edges_ahead)
        n = stopped = 0
        for v in self.vehicles:
            if v.edge in ids:
                n += 1
                if v.v < 0.05:
                    stopped += 1
        return n, stopped

    def _edges_on_route(self, node: int, edge: int, offset: float, first: Optional[int] = None, horizon=60.0) -> List[int]:
        T = self.track
        out = []
        d = 0.0
        e = first
        n = node
        if e is not None:
            out.append(e)
            d += T.edges[e].length
            n = T.edges[e].v
        while d < horizon:
            if n == T.edges[edge].u:
                out.append(edge)
                break
            ne = int(T.next_edge[n, T.edges[edge].u])
            if ne < 0:
                break
            out.append(ne)
            d += T.edges[ne].length
            n = T.edges[ne].v
        return out

    def area_stats(self) -> Dict[str, dict]:
        st = {a: {"pending": 0, "idle": 0, "rate": 0.0, "homed": 0} for a in self.layout.area_ids}
        for j in self.jobs.values():
            if j.state == "PENDING":
                st[j.src_area]["pending"] += 1
        for v in self.vehicles:
            if v.state == IDLE and v.home:
                st[v.home]["idle"] += 1
        for _, a in self.recent_jobs:
            st[a]["rate"] += 1.0 / 10.0  # jobs per minute over the last 10 min
        return st

    def dispatch_candidates(self, job: Job) -> List[dict]:
        pe, po = self._target_point(job.src)
        st = self.area_stats()
        cands = []
        for v in self.idle_vehicles():
            eta, meters = self._eta_to(v, pe, po)
            route = self._edges_on_route(self._vehicle_node(v), pe, po)
            nveh, nstop = self._congestion(route)
            home = st.get(v.home, {"pending": 0, "idle": 0, "rate": 0.0})
            cands.append({"vehicle": v.id, "eta": eta, "meters": meters, "cong": nveh, "stopped": nstop,
                          "homePending": home["pending"], "homeIdle": home["idle"], "homeRate": home["rate"],
                          "sameArea": float(v.home == job.src_area), "repositioning": float(v.needs_reposition),
                          "busy": 0.0, "toFree": 0.0})
        cands.sort(key=lambda c: c["eta"])
        busy = []
        for v in self.preassignable():
            tf, _ = self._time_to_free(v)
            if tf > self.preassign_horizon:
                continue
            de, do = v.target
            eta2, meters = self._eta_from_point(de, do, pe, po)
            route = self._edges_on_route(self.track.edges[de].v, pe, po)
            nveh, nstop = self._congestion(route)
            cj = self.jobs.get(v.job)
            darea = self.layout.ports[cj.dst].area if cj else None
            busy.append({"vehicle": v.id, "eta": tf + eta2, "meters": meters, "cong": nveh, "stopped": nstop,
                         "homePending": 0.0, "homeIdle": 0.0, "homeRate": 0.0,
                         "sameArea": float(darea == job.src_area), "repositioning": 0.0, "busy": 1.0, "toFree": tf})
        busy.sort(key=lambda c: c["eta"])
        n_busy = min(len(busy), max(self.min_busy_slots, self.K - len(cands)))
        out = cands[: self.K - n_busy] + busy[:n_busy]
        out.sort(key=lambda c: c["eta"])
        return out

    def route_candidates(self, veh: Vehicle, node: int) -> List[dict]:
        T = self.track
        te, to = veh.target
        opts = []
        for e in T.nodes[node].out_edges:
            if T.edges[e].blocked:
                continue
            ed = T.edges[e]
            if ed.v == T.edges[te].u or e == te:
                t_ = ed.length / ed.vmax + (to / T.edges[te].vmax if e != te else 0.0)
                m_ = ed.length + (to if e != te else 0.0)
                if e == te:
                    t_, m_ = to / ed.vmax, to
            else:
                t_ = ed.length / ed.vmax + T.time_from(ed.v, te, to)
                m_ = ed.length + T.meters_from(ed.v, te, to)
            if m_ > 1e11:
                continue
            route = self._edges_on_route(node, te, to, first=e)
            nveh, nstop = self._congestion(route)
            opts.append({"edge": e, "eta": t_, "meters": m_, "cong": nveh, "stopped": nstop,
                         "crossover": float("XO" in ed.tags)})
        if not opts:
            return []
        best = min(o["meters"] for o in opts)
        opts = [dict(o, detour=o["meters"] - best) for o in opts if o["meters"] - best <= self.max_detour]
        opts.sort(key=lambda o: o["eta"])
        return opts[: self.K]

    def reposition_candidates(self, veh: Vehicle) -> List[dict]:
        st = self.area_stats()
        n = self._vehicle_node(veh)
        cands = []
        for b in self.layout.bay_ids:
            a = self.layout.areas[b]
            eta, meters = self._eta_to(veh, a.park_edge, a.park_offset)
            cands.append({"area": b, "eta": eta, "meters": meters, "pending": st[b]["pending"], "idle": st[b]["idle"],
                          "rate": st[b]["rate"], "current": float(b == veh.home)})
        cands.sort(key=lambda c: (c["current"] < 0.5, c["eta"]))  # current home first, then nearest
        return cands[: self.K]

    def apply_decision(self, dec: Decision, choice: Optional[int]) -> dict:
        """choice: candidate index, or None = defer/default. Returns {"ok", "reason"}."""
        t = self.time
        self.metrics["decisions"][dec.kind] += 1
        if dec.kind == DISPATCH:
            j = self.jobs.get(dec.subject)
            if j is None or j.state != "PENDING":
                return {"ok": False, "reason": "STALE"}
            if choice is None or choice >= len(dec.candidates):
                j.defer_until = t + self.defer_retry
                j.defers += 1
                self.metrics["defers"] += 1
                return {"ok": True, "reason": "DEFERRED"}
            c = dec.candidates[choice]
            v = self.vehicles[c["vehicle"]]
            if c.get("busy"):
                if v.state not in (TO_DROP, UNLOADING) or v.next_job is not None:
                    return {"ok": False, "reason": "VEHICLE_BUSY"}
                j.state, j.vehicle, j.assigned_t = "ASSIGNED", v.id, t
                v.next_job = j.id
                self.metrics["preassigned"] += 1
                self.events.emit("PREASSIGN", t, jobId=j.id, vehicle=v.id, eta=round(c["eta"], 1))
                return {"ok": True, "reason": "PREASSIGNED"}
            if v.state != IDLE:
                return {"ok": False, "reason": "VEHICLE_BUSY"}
            j.assigned_t = t
            self._start_pickup(v, j, t)
            self.events.emit("DISPATCH", t, jobId=j.id, vehicle=v.id, eta=round(c["eta"], 1))
            return {"ok": True, "reason": "ASSIGNED"}
        veh = self.vehicles[int(dec.subject)]
        if dec.kind == ROUTE:
            if choice is None or choice >= len(dec.candidates):
                choice = 0  # default: fastest
            veh.forced[dec.node] = dec.candidates[choice]["edge"]
            veh.decided_nodes.add(dec.node)
            return {"ok": True, "reason": "ROUTED"}
        # REPOSITION
        veh.needs_reposition = False
        if veh.state != IDLE:
            return {"ok": False, "reason": "VEHICLE_BUSY"}
        if choice is None or choice >= len(dec.candidates):
            return {"ok": True, "reason": "STAY"}
        veh.home = dec.candidates[choice]["area"]
        self._set_idle_target(veh)
        veh.forced.clear()
        return {"ok": True, "reason": "REPOSITIONED"}

    # ============================================================== views
    def summary(self) -> dict:
        m = self.metrics
        t = max(self.time, 1e-9)
        dts = m["deliveryTimes"]
        V = len(self.vehicles)
        moved = m["emptyMeters"] + m["loadedMeters"]
        return {
            "time": round(self.time, 2), "created": m["created"], "delivered": m["delivered"],
            "throughputPerHour": round(m["delivered"] / t * 3600.0, 1),
            "avgDeliveryTime": round(float(np.mean(dts)), 2) if dts else None,
            "p95DeliveryTime": round(float(np.percentile(dts, 95)), 2) if dts else None,
            "avgWaitForVehicle": round(float(np.mean(m["waitTimes"])), 2) if m["waitTimes"] else None,
            "pending": sum(1 for j in self.jobs.values() if j.state == "PENDING"),
            "inSystem": len(self.jobs),
            "emptyTravelRatio": round(m["emptyMeters"] / moved, 4) if moved else None,
            "utilization": round(m["busyTime"] / (t * V), 4) if V else None,
            "blockedRatio": round(m["blockedTime"] / max(m["busyTime"], 1e-9), 4),
            "hotDelivered": m["hotDelivered"], "hotOnTime": m["hotOnTime"], "hotLate": m["hotLate"],
            "hotOnTimeRate": round(m["hotOnTime"] / m["hotDelivered"], 4) if m["hotDelivered"] else None,
            "deadlock": m["deadlock"], "decisions": dict(m["decisions"]), "defers": m["defers"],
            "mergeWaitTime": round(m["mergeWaitTime"], 1), "vehicles": V,
            "idle": sum(1 for v in self.vehicles if v.state == IDLE),
            "preassigned": m["preassigned"],
        }

    def vehicle_pose(self, veh: Vehicle):
        e = self.track.edges[veh.edge]
        x, z = e.piece.point(veh.s)
        return x, z, e.piece.heading(veh.s)

    def get_layout(self) -> dict:
        d = self.layout.to_dict()
        d.update({"name": self.cfg.get("name"), "floor": self.cfg["floor"], "railHeight": self.cfg["railHeight"],
                  "bays": self.cfg["bays"], "vehicleLength": self.vlen, "dt": self.dt, "candidates": self.K})
        return d

    def get_state(self, include_events: bool = True) -> dict:
        vs = []
        for v in self.vehicles:
            x, z, h = self.vehicle_pose(v)
            j = self.jobs.get(v.job) if v.job else None
            vs.append({"id": v.id, "x": round(x, 3), "z": round(z, 3), "heading": round(h, 4), "v": round(v.v, 2),
                       "state": v.state, "job": v.job, "hot": bool(j and j.hot), "home": v.home,
                       "loaded": v.state in (TO_DROP, UNLOADING) or (v.state == LOADING and v.timer < self.hoist / 2)})
        pend = sorted((j for j in self.jobs.values()), key=lambda j: (j.state != "PENDING", not j.hot, j.created))
        st = {
            "time": round(self.time, 2), "seed": self.seed, "scenario": self.cfg.get("scenario"),
            "vehicles": vs, "jobs": [j.to_dict(self.time) for j in pend[:40]], "areas": self.area_stats(),
            "blocked": [e.id for e in self.track.edges if e.blocked], "metrics": self.summary(),
        }
        if include_events:
            st["events"] = self.events.recent(25, exclude={"JOB"})
        return st

    def state_hash(self) -> str:
        parts = [f"{self.time:.3f}"] + [f"{v.id}:{v.edge}:{v.s:.5f}:{v.state}:{v.job}" for v in self.vehicles]
        parts += sorted(f"{j.id}:{j.state}" for j in self.jobs.values())
        return hashlib.sha256("|".join(parts).encode()).hexdigest()
