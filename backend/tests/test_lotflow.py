"""Closed-loop lot flow: EFEM load ports, one FOUP per load port, constant WIP."""
from collections import Counter

from backend.ai.oht_env import OHTEnv, eval_seeds
from backend.policies.rules import BasicRulePolicy, PreassignRulePolicy
from backend.sim.fab import FabSimulation
from backend.sim.lotflow import PROC, QUEUE, RESERVED, WAIT

HOLDING = ("R", "D", "Q", "P", "U", "W")


def run(f, steps, check=None):
    for _ in range(steps):
        for d in f.decisions():
            f.apply_decision(d, 0 if d.candidates else None)
            break
        f.step()
        if check:
            check(f)


def lots_in_fab(f):
    on_lp = sum(1 for lp in f.flow.lps.values() if lp.state in HOLDING)
    in_transit = sum(1 for j in f.jobs.values() if f.flow.lps.get(j.src) is None or f.flow.lps[j.src].job != j.id)
    return on_lp + f.flow.stock_count() + in_transit


def test_one_foup_per_load_port_and_constant_wip():
    f = FabSimulation(None)
    wip0 = lots_in_fab(f)
    assert wip0 > 60

    def check(f):
        for v in f.vehicles:
            if v.state == "UNLOADING":
                j = f.jobs[v.job]
                lp = f.flow.lps.get(j.dst)
                if lp is not None:  # delivering to a tool: its load port was reserved for exactly this job
                    assert lp.state == RESERVED and lp.job == j.id
        for tool in f.flow.tools.values():  # one lot in the process chamber at a time
            assert sum(1 for q in tool.ports if f.flow.lps[q].state == PROC) <= 1
        for j in f.jobs.values():  # a request from a tool comes from a finished FOUP waiting on its load port
            lp = f.flow.lps.get(j.src)
            if lp is not None and j.state in ("PENDING", "ASSIGNED"):
                assert lp.state == WAIT and lp.job == j.id
        assert lots_in_fab(f) == wip0

    run(f, 6000, check)
    s = f.summary()
    assert s["lotsProcessed"] > 50 and s["delivered"] > 150 and not s["deadlock"]
    assert 0.3 < s["toolUtilization"] < 1.0


def test_lot_phases_and_queueing():
    f = FabSimulation(None)
    seen = Counter()

    def check(f):
        for lp in f.flow.lps.values():
            seen[lp.state] += 1
            if lp.state == QUEUE:  # docked and waiting: the tool's chamber is busy with the other load port
                tool = f.flow.tools[lp.tool]
                assert tool.active is not None and tool.active != lp.port

    run(f, 4000, check)
    for st in ("E", "V", "R", "D", "Q", "P", "U", "W"):
        assert seen[st] > 0, st


def test_lots_are_policy_independent():
    """Each lot's next process is keyed by (seed, lot, step): policies face the same lots."""
    routes = []
    for P in (BasicRulePolicy, PreassignRulePolicy):
        env = OHTEnv("normal", seed_mode="eval")
        obs, _ = env.reset(seed=eval_seeds(1)[0])
        pol = P(env)
        seen = {}
        for _ in range(4000):
            obs, r, te, tr, _ = env.step(pol.act(obs))
            for j in env.unwrapped.fab.jobs.values():
                seen[(j.lot.id, j.lot.step)] = j.lot.proc
            if te or tr:
                break
        routes.append(seen)
    common = set(routes[0]) & set(routes[1])
    assert len(common) > 50
    assert all(routes[0][k] == routes[1][k] for k in common)


def test_view_lists_load_ports_for_the_client():
    f = FabSimulation(None)
    run(f, 1500)
    st = f.get_state(include_events=False)
    lps = st["tools"]["lps"]
    hoisting = {v["port"] for v in st["vehicles"] if v["port"]}
    assert lps and all(row[0] in f.flow.lps for row in lps)
    assert not {row[0] for row in lps if row[1] != RESERVED} & hoisting
    L = f.get_layout()
    assert len(L["tools"]) == 64 and sum(len(t["ports"]) for t in L["tools"]) == 112
