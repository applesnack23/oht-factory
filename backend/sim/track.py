"""OHT rail network: geometry pieces, directed graph, shortest paths.

Coordinates are in the X-Z plane (rails hang at `railHeight`). Heading is
atan2(dz, dx). Every directed edge is a single straight or circular-arc piece;
nodes sit at piece boundaries. Polylines are filleted automatically so every
turn is a smooth arc (tangent-continuous, required for vehicle motion).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Straight:
    kind = "STRAIGHT"

    def __init__(self, a, b):
        self.a = (float(a[0]), float(a[1]))
        self.b = (float(b[0]), float(b[1]))
        dx, dz = self.b[0] - self.a[0], self.b[1] - self.a[1]
        self.length = math.hypot(dx, dz)
        self.ux, self.uz = dx / self.length, dz / self.length
        self.h = math.atan2(self.uz, self.ux)

    def point(self, s):
        s = min(max(s, 0.0), self.length)
        return self.a[0] + self.ux * s, self.a[1] + self.uz * s

    def heading(self, s):
        return self.h

    def to_dict(self):
        return {"kind": "STRAIGHT", "start": list(self.a), "end": list(self.b), "length": self.length}


class Arc:
    kind = "ARC"

    def __init__(self, center, radius, start_angle, sweep):
        self.c = (float(center[0]), float(center[1]))
        self.r = float(radius)
        self.a0 = float(start_angle)  # radians
        self.sweep = float(sweep)  # radians, signed
        self.sign = 1.0 if sweep >= 0 else -1.0
        self.length = abs(self.sweep) * self.r

    def _th(self, s):
        s = min(max(s, 0.0), self.length)
        return self.a0 + self.sign * s / self.r

    def point(self, s):
        th = self._th(s)
        return self.c[0] + self.r * math.cos(th), self.c[1] + self.r * math.sin(th)

    def heading(self, s):
        th = self._th(s)
        return math.atan2(math.cos(th) * self.sign, -math.sin(th) * self.sign)

    def to_dict(self):
        return {"kind": "ARC", "center": list(self.c), "radius": self.r, "startAngle": math.degrees(self.a0),
                "sweep": math.degrees(self.sweep), "length": self.length}


class Node:
    __slots__ = ("id", "x", "z", "out_edges", "in_edges")

    def __init__(self, nid, x, z):
        self.id, self.x, self.z = nid, x, z
        self.out_edges: List[int] = []
        self.in_edges: List[int] = []

    @property
    def is_merge(self):
        return len(self.in_edges) > 1

    @property
    def is_diverge(self):
        return len(self.out_edges) > 1


class Edge:
    __slots__ = ("id", "u", "v", "piece", "length", "vmax", "tags", "blocked", "zone")

    def __init__(self, eid, u, v, piece, vmax, tags, zone):
        self.id, self.u, self.v, self.piece = eid, u, v, piece
        self.length = piece.length
        self.vmax = vmax
        self.tags = set(tags)
        self.blocked = False
        self.zone = zone

    def to_dict(self):
        return {"id": self.id, "u": self.u, "v": self.v, "length": round(self.length, 4), "vmax": self.vmax,
                "tags": sorted(self.tags), "zone": self.zone, "path": self.piece.to_dict()}


def _norm(dx, dz):
    n = math.hypot(dx, dz)
    return dx / n, dz / n


class Track:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self._node_index: Dict[Tuple[int, int], int] = {}

    # ------------------------------------------------------------ building
    def node_at(self, x, z) -> int:
        key = (round(x * 1000), round(z * 1000))
        if key not in self._node_index:
            n = Node(len(self.nodes), float(x), float(z))
            self.nodes.append(n)
            self._node_index[key] = n.id
        return self._node_index[key]

    def _add_edge(self, piece, vmax, tags, zone) -> int:
        u = self.node_at(*piece.point(0.0))
        v = self.node_at(*piece.point(piece.length))
        e = Edge(len(self.edges), u, v, piece, vmax, tags, zone)
        self.edges.append(e)
        self.nodes[u].out_edges.append(e.id)
        self.nodes[v].in_edges.append(e.id)
        return e.id

    def add_path(self, points: Sequence[Sequence[float]], radius: float, v_straight: float, v_curve: float,
                 closed: bool = False, tags=(), zone: str = "") -> List[int]:
        """Add a filleted polyline. Collinear vertices become plain nodes (branch points)."""
        pts = [(float(p[0]), float(p[1])) for p in points]
        if closed:
            pts = pts + [pts[0]]
        edges: List[int] = []
        n = len(pts)
        # tangent points per vertex: (enter point, exit point, arc or None)
        corners = []
        for i in range(n):
            p = pts[i]
            if (not closed and (i == 0 or i == n - 1)):
                corners.append((p, p, None))
                continue
            prev = pts[i - 1] if i > 0 else pts[-2]
            nxt = pts[i + 1] if i < n - 1 else pts[1]
            d1 = _norm(p[0] - prev[0], p[1] - prev[1])
            d2 = _norm(nxt[0] - p[0], nxt[1] - p[1])
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            dot = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
            theta = math.acos(dot)
            if theta < 1e-6:
                corners.append((p, p, None))
                continue
            t = radius * math.tan(theta / 2)
            t1 = (p[0] - d1[0] * t, p[1] - d1[1] * t)
            t2 = (p[0] + d2[0] * t, p[1] + d2[1] * t)
            nvec = (-d1[1], d1[0]) if cross > 0 else (d1[1], -d1[0])
            c = (t1[0] + nvec[0] * radius, t1[1] + nvec[1] * radius)
            a0 = math.atan2(t1[1] - c[1], t1[0] - c[0])
            corners.append((t1, t2, Arc(c, radius, a0, theta if cross > 0 else -theta)))
        if closed:  # first vertex == last vertex: use the computed corner of the last for the first
            corners[0] = corners[-1]
        cur = corners[0][1]
        for i in range(1, n):
            enter, exit_, arc = corners[i]
            if math.hypot(enter[0] - cur[0], enter[1] - cur[1]) > 1e-6:
                edges.append(self._add_edge(Straight(cur, enter), v_straight, tags, zone))
            if arc is not None:
                edges.append(self._add_edge(arc, v_curve, tags, zone))
            cur = exit_
        return edges

    # ------------------------------------------------------------ routing
    def build_routing(self) -> None:
        """All-pairs shortest travel *time* (s) and next-hop edges (Floyd-Warshall, numpy)."""
        N = len(self.nodes)
        INF = 1e12
        dist = np.full((N, N), INF)
        met = np.full((N, N), INF)
        nxt = np.full((N, N), -1, dtype=np.int64)
        np.fill_diagonal(dist, 0.0)
        np.fill_diagonal(met, 0.0)
        for e in self.edges:
            if e.blocked:
                continue
            w = e.length / e.vmax
            if w < dist[e.u, e.v]:
                dist[e.u, e.v] = w
                met[e.u, e.v] = e.length
                nxt[e.u, e.v] = e.id
        for k in range(N):
            via = dist[:, k:k + 1] + dist[k:k + 1, :]
            better = via < dist - 1e-12
            if better.any():
                dist = np.where(better, via, dist)
                met = np.where(better, met[:, k:k + 1] + met[k:k + 1, :], met)
                nxt = np.where(better, np.broadcast_to(nxt[:, k:k + 1], (N, N)), nxt)
        self.time = dist
        self.meters = met
        self.next_edge = nxt

    def _path_meters(self, u, v) -> float:
        m = 0.0
        guard = 0
        while u != v and guard < 10000:
            e = self.edges[int(self.next_edge[u, v])]
            m += e.length
            u = e.v
            guard += 1
        return m

    def path_edges(self, u: int, v: int) -> List[int]:
        out = []
        guard = 0
        while u != v and guard < 10000:
            eid = int(self.next_edge[u, v])
            if eid < 0:
                return []
            out.append(eid)
            u = self.edges[eid].v
            guard += 1
        return out

    def time_from(self, node: int, edge: int, offset: float) -> float:
        """Shortest time from a node to a point (edge, offset)."""
        e = self.edges[edge]
        return float(self.time[node, e.u]) + offset / e.vmax

    def meters_from(self, node: int, edge: int, offset: float) -> float:
        e = self.edges[edge]
        return float(self.meters[node, e.u]) + offset

    def to_dict(self):
        return {"nodes": [{"id": n.id, "x": n.x, "z": n.z, "merge": n.is_merge, "diverge": n.is_diverge} for n in self.nodes],
                "edges": [e.to_dict() for e in self.edges]}
