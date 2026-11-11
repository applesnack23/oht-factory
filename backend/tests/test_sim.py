import math
from collections import defaultdict

import numpy as np
import pytest

from backend.sim.fab import DISPATCH, IDLE, REPOSITION, ROUTE, FabSimulation
from backend.sim.layout import FabLayout
from backend.sim.config import resolve_config, scenario_names


def run_basic(f, seconds=None, check=None):
    end = f.time + seconds if seconds else f.episode_length
    while f.time < end - 1e-9 and not f.done:
        while True:
            ds = f.decisions()
            if not ds:
                break
            d = ds[0]
            f.apply_decision(d, 0 if d.kind != REPOSITION else None)
        f.step()
        if check:
            check(f)
    return f


def test_layout_connected_smooth_and_sized():
    L = FabLayout(resolve_config())
    T = L.track
    assert len(L.ports) == 120 and len(L.bay_ids) == 8
    assert (T.time > 1e11).sum() == 0  # every node reaches every node (one-way network)
    for n in T.nodes:
        for i in n.in_edges:
            for o in n.out_edges:
                hi, ho = T.edges[i].piece.heading(T.edges[i].length), T.edges[o].piece.heading(0)
                assert abs((hi - ho + math.pi) % (2 * math.pi) - math.pi) < 1e-6
                a, b = T.edges[i].piece.point(T.edges[i].length), T.edges[o].piece.point(0)
                assert math.dist(a, b) < 1e-6
    assert len(L.decision_nodes) == 2  # the two crossover diverges
    for p in L.ports.values():  # ports lie on their edge
        x, z = T.edges[p.edge].piece.point(p.offset)
        assert math.dist((x, z), (p.x, p.z)) < 1e-6


def test_shortest_path_tables_consistent():
    L = FabLayout(resolve_config())
    T = L.track
    rng = np.random.default_rng(0)
    for _ in range(50):
        u, v = rng.integers(0, len(T.nodes), 2)
        if u == v:
            continue
        edges = T.path_edges(int(u), int(v))
        assert edges and T.edges[edges[0]].u == u and T.edges[edges[-1]].v == v
        assert sum(T.edges[e].length for e in edges) == pytest.approx(T.meters[u, v])


def test_same_seed_same_result():
    a = run_basic(FabSimulation(None), 300)
    b = run_basic(FabSimulation(None), 300)
    assert a.state_hash() == b.state_hash()
    c = FabSimulation(None)
    c.reset(99)
    run_basic(c, 300)
    assert c.state_hash() != a.state_hash()


def test_no_overlap_anywhere_including_merges():
    worst = [1e9]

    def check(f):
        pts = [f.vehicle_pose(v)[:2] for v in f.vehicles]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                worst[0] = min(worst[0], math.dist(pts[i], pts[j]))

    run_basic(FabSimulation("high_load"), 400, check)
    assert worst[0] >= 1.0 - 1e-6  # vehicle length 1.0 m: centers never closer


def test_speed_limits_respected():
    f = FabSimulation(None)

    def check(f):
        for v in f.vehicles:
            assert v.v <= f.track.edges[v.edge].vmax + 1e-6 or v.v <= 3.5 + 1e-6

    run_basic(f, 200, check)
    f2 = FabSimulation(None)
    prev = {v.id: 0.0 for v in f2.vehicles}

    def acc(f):
        for v in f.vehicles:
            assert v.v - prev[v.id] <= f.accel * f.dt + 1e-6
            prev[v.id] = v.v

    run_basic(f2, 100, acc)


def test_full_job_lifecycle_and_hoist_time():
    f = FabSimulation(None)
    seen = {}

    def check(f):
        for v in f.vehicles:
            if v.state in ("LOADING", "UNLOADING"):
                seen.setdefault((v.id, v.job, v.state), f.time)

    run_basic(f, 600, check)
    s = f.summary()
    assert s["delivered"] > 100 and not s["deadlock"]
    done = f.done_jobs[0]
    assert done.created <= done.assigned_t <= done.picked_t <= done.done_t
    assert done.picked_t - done.assigned_t >= f.hoist - 1e-6  # hoist cycle took place
    assert s["created"] == s["delivered"] + len(f.jobs)  # job conservation


def test_dispatch_route_reposition_decisions():
    f = FabSimulation(None)
    kinds = set()
    for _ in range(3000):
        for d in f.decisions():
            kinds.add(d.kind)
            assert 1 <= len(d.candidates) <= f.K
            if d.kind == DISPATCH:
                etas = [c["eta"] for c in d.candidates]
                assert etas == sorted(etas)  # candidate 0 = nearest vehicle
                res = f.apply_decision(d, 0)
                assert res["ok"]
                v = f.vehicles[d.candidates[0]["vehicle"]]
                assert v.state == "TO_PICKUP" and v.job == d.subject
            elif d.kind == ROUTE:
                assert len(d.candidates) >= 2 and d.candidates[0]["detour"] == 0.0
                f.apply_decision(d, len(d.candidates) - 1)
                assert f.vehicles[int(d.subject)].forced[d.node] == d.candidates[-1]["edge"]
            else:
                f.apply_decision(d, 1)
            break
        f.step()
    assert kinds == {DISPATCH, ROUTE, REPOSITION}


def test_defer_postpones_job():
    f = FabSimulation(None)
    while True:
        ds = [d for d in f.decisions() if d.kind == DISPATCH]
        if ds:
            break
        f.step()
    d = ds[0]
    r = f.apply_decision(d, None)
    assert r["reason"] == "DEFERRED" and f.jobs[d.subject].state == "PENDING"
    assert all(x.subject != d.subject for x in f.decisions() if x.kind == DISPATCH)


def test_track_block_reroutes_and_reopens():
    f = FabSimulation("track_block")
    entry = f.block_edges["XD"][0]

    def check(f):
        if 300 < f.time < 1200:
            assert f.track.edges[entry].blocked
            for v in f.vehicles:  # nobody enters the closed crossover
                if v.edge == entry:
                    assert v.s > 0 or f.time < 305
        if f.time > 1201:
            assert not f.track.edges[entry].blocked

    run_basic(f, None, check)
    assert not f.summary()["deadlock"]


@pytest.mark.parametrize("name", scenario_names())
def test_scenarios_run_without_deadlock(name):
    f = run_basic(FabSimulation(name), 600)
    s = f.summary()
    assert not s["deadlock"] and s["delivered"] > 50


def test_reset_restores_initial_state():
    f = FabSimulation(None)
    h0 = f.state_hash()
    run_basic(f, 200)
    f.reset(f.seed)
    assert f.state_hash() == h0 and f.time == 0 and f.summary()["delivered"] == 0
