from backend.sim.fab import FabSimulation
from backend.sim.stocker import Stocker


def _sk():
    return Stocker("S", -5.0, -1, {"S-P1": 5.0, "S-P2": 2.0}, seed=3, fill=0.5)


def run(sk, seconds, dt=0.2):
    for _ in range(int(seconds / dt)):
        sk.update(dt)


def test_store_moves_foup_from_port_to_shelf():
    sk = _sk()
    n0 = int(sk.occ.sum())
    sk.store("S-P1")
    assert sk.port_has_foup("S-P1")
    run(sk, 40)
    assert not sk.port_has_foup("S-P1") and int(sk.occ.sum()) == n0 + 1 and sk.done == 1 and not sk.carry


def test_retrieve_puts_foup_on_port_and_vehicle_takes_it():
    sk = _sk()
    n0 = int(sk.occ.sum())
    sk.retrieve("J1", "S-P2")
    run(sk, 40)
    assert "J1" in sk.ready and int(sk.occ.sum()) == n0 - 1
    sk.picked("J1")
    assert "J1" not in sk.ready


def test_vehicle_before_crane_cancels_retrieval():
    sk = _sk()
    sk.retrieve("J1", "S-P1")
    sk.retrieve("J2", "S-P2")
    run(sk, 1.0)  # J1 in progress, J2 queued
    sk.picked("J2")
    assert all(t.get("job") != "J2" for t in sk.tasks)
    sk.picked("J1")
    run(sk, 60)
    assert not sk.ready  # J1's FOUP vanished when placed: the vehicle already has it


def test_crane_stays_inside_stocker():
    sk = _sk()
    for k in range(6):
        sk.store("S-P1") if k % 2 else sk.retrieve(f"J{k}", "S-P2")
    zs = sk.bay_z
    for _ in range(1500):
        sk.update(0.2)
        assert min(zs) - 1e-6 <= sk.z <= max(zs) + 1e-6 or sk.z in (5.0, 2.0)
        assert 0.9 - 1e-6 <= sk.y <= 3.45 + 1e-6 and -1e-6 <= sk.fork <= 3.5 + 1e-6


def test_state_exposes_hoist_port_foups_and_stockers():
    f = FabSimulation(None)
    seen_hoist = seen_crane = False
    for _ in range(3000):
        for d in f.decisions():
            f.apply_decision(d, 0 if d.candidates else None)
            break
        f.step()
        st = f.get_state(include_events=False)
        seen_hoist |= any(0 < v["hoist"] < 1 for v in st["vehicles"])
        seen_crane |= any(s["queue"] > 0 for s in st["stockers"])
        for p in st["portFoups"]:
            assert p in f.layout.ports
    assert seen_hoist and seen_crane
    assert sum(s.done for s in f.stockers.values()) > 5
    L = f.get_layout()
    assert len(L["stockers"]) == 2 and len(L["stockers"][0]["cells"]) == 84


def test_oht_waits_for_crane_and_foup_is_never_lost():
    f = FabSimulation(None)
    calls = []
    for sk in f.stockers.values():
        orig = sk.picked
        sk.picked = (lambda o, s: (lambda jid: (calls.append(jid in s.ready), o(jid))))(orig, sk)
    for _ in range(9000):
        for d in f.decisions():
            f.apply_decision(d, 0 if d.candidates else None)
            break
        f.step()
        for v in f.vehicles:  # no hoisting progress at a stocker port before the crane delivered the FOUP
            if v.state == "LOADING" and not f._foup_ready(v):
                assert v.timer == f.hoist
        st = f.get_state(include_events=False)
        hoist_ports = {v["port"] for v in st["vehicles"] if v["port"]}
        assert not hoist_ports & set(st["portFoups"])  # a hoisting OHT's port shows only its own FOUP
    assert calls and all(calls)  # every stocker FOUP was on the port when the OHT lifted it
    assert not any(sk.cancelled for sk in f.stockers.values())

