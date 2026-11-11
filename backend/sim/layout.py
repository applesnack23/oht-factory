"""Fab layout generator: interbay ring + crossovers + intrabay loops + stocker spurs.

Interbay ring (one-way, clockwise on screen): top edge (z0) heading +x, east side
heading +z, bottom edge (z1) heading -x, west side heading -z.
Each bay = entry spur from the ring -> closed intrabay loop (left lane, far end,
right lane, return connector) -> exit spur back to the ring. Tools (load ports)
line both lanes. Stockers sit on spurs off the west/east sides of the ring.
Two crossovers (XD: top->bottom, XU: bottom->top) create genuine routing choices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .track import Straight, Track


@dataclass
class Port:
    id: str
    edge: int
    offset: float
    area: str  # bay id or stocker id
    kind: str  # TOOL | STOCKER
    process: str
    x: float
    z: float
    tool_x: float
    tool_z: float

    def to_dict(self):
        return {"id": self.id, "edge": self.edge, "offset": round(self.offset, 3), "area": self.area, "kind": self.kind,
                "process": self.process, "x": self.x, "z": self.z, "toolX": self.tool_x, "toolZ": self.tool_z}


@dataclass
class Area:
    id: str
    kind: str  # BAY | STOCKER
    process: str
    side: str
    cx: float
    cz: float
    ports: List[str] = field(default_factory=list)
    loop_edges: List[int] = field(default_factory=list)
    park_edge: int = -1
    park_offset: float = 0.0

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "process": self.process, "side": self.side, "cx": self.cx, "cz": self.cz,
                "ports": self.ports, "parkEdge": self.park_edge}


class FabLayout:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        tr = cfg["track"]
        ib = cfg["interbay"]
        bg = cfg["bayGeometry"]
        self.track = Track()
        x0, x1, z0, z1 = ib["x0"], ib["x1"], ib["z0"], ib["z1"]
        off, r, depth, hg = bg["entryOffset"], bg["radius"], bg["depth"], bg["laneHalfGap"]
        north = [b for b in cfg["bays"] if b["side"] == "north"]
        south = [b for b in cfg["bays"] if b["side"] == "south"]
        xd = next((c for c in cfg.get("crossovers", []) if c["direction"] == "down"), None)
        xu = next((c for c in cfg.get("crossovers", []) if c["direction"] == "up"), None)
        top = sorted({b["center"] - off for b in north} | {b["center"] + off for b in north}
                     | ({xd["x"] - 2} if xd else set()) | ({xu["x"] + 2} if xu else set()))
        bot = sorted({b["center"] + off for b in south} | {b["center"] - off for b in south}
                     | ({xu["x"] + 2} if xu else set()) | ({xd["x"] - 2} if xd else set()), reverse=True)
        west = [11.0, -11.0]  # stocker spur attach points on the west side (heading -z)
        east = [-11.0, 11.0]
        ring = [(x0, z0)] + [(x, z0) for x in top] + [(x1, z0)] + [(x1, z) for z in east] + [(x1, z1)] \
            + [(x, z1) for x in bot] + [(x0, z1)] + [(x0, z) for z in west]
        self.ring_edges = self.track.add_path(ring, ib["cornerRadius"], tr["interbaySpeed"], tr["curveSpeed"],
                                              closed=True, tags=("IB",), zone="IB")
        self.crossover_edges: Dict[str, List[int]] = {}
        if xd:
            x = xd["x"]
            self.crossover_edges[xd["id"]] = self.track.add_path(
                [(x - 2, z0), (x, z0), (x, z1), (x - 2, z1)], 2.0, tr["straightSpeed"], tr["curveSpeed"],
                tags=(xd["id"], "XO"), zone=xd["id"])
        if xu:
            x = xu["x"]
            self.crossover_edges[xu["id"]] = self.track.add_path(
                [(x + 2, z1), (x, z1), (x, z0), (x + 2, z0)], 2.0, tr["straightSpeed"], tr["curveSpeed"],
                tags=(xu["id"], "XO"), zone=xu["id"])
        self.areas: Dict[str, Area] = {}
        self.ports: Dict[str, Port] = {}
        for b in cfg["bays"]:
            self._add_bay(b, z0 if b["side"] == "north" else z1, off, r, depth, hg, bg, tr)
        for s in cfg.get("stockers", []):
            self._add_stocker(s, x0 if s["side"] == "west" else x1, tr)
        self.track.build_routing()
        # decision diverges: nodes whose out-edges lead into a crossover (genuine route choice)
        self.decision_nodes = set()
        for n in self.track.nodes:
            if n.is_diverge and any("XO" in self.track.edges[e].tags for e in n.out_edges):
                self.decision_nodes.add(n.id)
        self.area_ids = list(self.areas)
        self.bay_ids = [a for a in self.area_ids if self.areas[a].kind == "BAY"]

    # ---------------------------------------------------------------- bays
    def _add_bay(self, b, zr, off, r, depth, hg, bg, tr):
        c = b["center"]
        sgn = -1.0 if b["side"] == "north" else 1.0  # north bays extend to -z
        far = zr + sgn * depth
        join = zr + sgn * 8.0  # where the entry spur joins / exit spur leaves the loop
        if b["side"] == "north":  # ring heads +x: enter on the left (c-off), leave on the right (c+off)
            lx, rx, ent, ext = c - hg, c + hg, c - off, c + off
        else:  # ring heads -x: enter at c+off, leave at c-off
            lx, rx, ent, ext = c + hg, c - hg, c + off, c - off
        T = self.track
        # entry spur + up-lane + far end + down-lane + exit spur (one polyline through the loop)
        main = [(ent, zr), (lx, zr), (lx, join), (lx, far), (rx, far), (rx, join), (rx, zr), (ext, zr)]
        e_main = T.add_path(main, r, tr["straightSpeed"], tr["curveSpeed"], tags=(b["id"],), zone=b["id"])
        # return connector: down-lane at `join` back to up-lane at `join` (closes the intrabay loop)
        k = join - sgn * 2.0  # 2 m toward the ring
        ret = [(rx, join), (rx, k), (lx, k), (lx, join)]
        e_ret = T.add_path(ret, r, tr["straightSpeed"], tr["curveSpeed"], tags=(b["id"], "RETURN"), zone=b["id"])
        area = Area(b["id"], "BAY", b["process"], b["side"], c, (zr + far) / 2)
        # lanes = the two long straight edges between `join` and `far`
        lanes = [e for e in e_main if isinstance(T.edges[e].piece, Straight) and T.edges[e].length > depth * 0.5]
        n_ports = bg["portsPerLane"]
        for li, eid in enumerate(lanes):
            e = T.edges[eid]
            out_dir = -1.0 if (e.piece.a[0] < c) else 1.0  # tools on the outer side of each lane
            for k_ in range(n_ports):
                o = bg["portStart"] + k_ * bg["portSpacing"]
                if o > e.length - 1.0:
                    break
                x, z = e.piece.point(o)
                pid = f"{b['id']}-{'L' if li == 0 else 'R'}{k_ + 1:02d}"
                self.ports[pid] = Port(pid, eid, o, b["id"], "TOOL", b["process"], x, z, x + out_dir * 2.4, z)
                area.ports.append(pid)
        loop = [e for e in e_main if self._between(T.edges[e], join, far)] + e_ret
        area.loop_edges = loop
        area.park_edge = lanes[0]
        area.park_offset = 1.0
        self.areas[b["id"]] = area

    @staticmethod
    def _between(e, za, zb):
        lo, hi = min(za, zb) - 1e-6, max(za, zb) + 1e-6
        p0 = e.piece.point(0.0)
        p1 = e.piece.point(e.length)
        return lo <= p0[1] <= hi and lo <= p1[1] <= hi

    # ------------------------------------------------------------ stockers
    def _add_stocker(self, s, xr, tr):
        T = self.track
        if s["side"] == "west":  # ring heads -z on the west side
            pts = [(xr, 11), (xr, 9), (xr - 5, 9), (xr - 5, -9), (xr, -9), (xr, -11)]
            bx = xr - 9
        else:  # east side heads +z
            pts = [(xr, -11), (xr, -9), (xr + 5, -9), (xr + 5, 9), (xr, 9), (xr, 11)]
            bx = xr + 9
        edges = T.add_path(pts, 2.0, tr["straightSpeed"], tr["curveSpeed"], tags=(s["id"],), zone=s["id"])
        lane = max(edges, key=lambda e: T.edges[e].length if isinstance(T.edges[e].piece, Straight) else 0)
        area = Area(s["id"], "STOCKER", "STOCKER", s["side"], bx, 0.0)
        e = T.edges[lane]
        for k in range(s["ports"]):
            o = 2.0 + 3.0 * k
            x, z = e.piece.point(o)
            pid = f"{s['id']}-P{k + 1}"
            self.ports[pid] = Port(pid, lane, o, s["id"], "STOCKER", "STOCKER", x, z, bx, z)
            area.ports.append(pid)
        area.loop_edges = edges
        area.park_edge = lane
        self.areas[s["id"]] = area

    def to_dict(self) -> dict:
        return {"track": self.track.to_dict(), "ports": [p.to_dict() for p in self.ports.values()],
                "areas": [a.to_dict() for a in self.areas.values()], "decisionNodes": sorted(self.decision_nodes),
                "crossovers": {k: v for k, v in self.crossover_edges.items()}}
