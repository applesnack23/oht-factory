from backend.sim.fab import FabSimulation
from backend.sim.stocker import AISLE_U, CONV_T, DY, FAR_U, NEAR_U, Stocker


def _sk():
    return Stocker("S", -5.0, -1, {"S-P1": 5.0, "S-P2": 2.0}, seed=3, fill=0.5)


def run(sk, seconds, dt=0.2):
    for _ in range(int(seconds / dt)):
        sk.update(dt)


def test_store_conveys_in_then_crane_shelves_it():
    sk = _sk()
    n0 = int(sk.occ.sum())
    sk.store("S-P1")
    assert [c[1] for c in sk.conv] == ["in"] and not sk.tasks  # the stage conveyor carries it in first
    run(sk, CONV_T + 0.2)
    assert [c[1] for c in sk.conv] == ["inner"] or sk.carry  # waiting at the inner end for the crane
    run(sk, 40)
    assert not sk.conv and int(sk.occ.sum()) == n0 + 1 and sk.done == 1 and not sk.carry


def test_retrieve_crane_then_conveyor_out_to_stage():
    sk = _sk()
    n0 = int(sk.occ.sum())
    sk.retrieve("J1", "S-P2")
    t_out = None
    for _ in range(400):
        sk.update(0.1)
        if t_out is None and any(c[1] == "out" for c in sk.conv):
            t_out = sk.t
            assert "J1" not in sk.ready  # not on the stage until the conveyor has carried it out
        if "J1" in sk.ready:
            break
    assert t_out is not None and sk.t - t_out >= CONV_T - 0.11
    assert int(sk.occ.sum()) == n0 - 1 and [c[1] for c in sk.conv] == ["stage"]
    sk.picked("J1")
    assert "J1" not in sk.ready and not sk.conv


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


def test_fork_enters_low_lifts_and_turns_for_the_far_rack():
    sk = _sk()
    far = [i for i, c in enumerate(sk.cells) if c[0] == 1]
    sk.occ[:] = 0
    sk.occ[far[5]] = 1  # the only stored FOUP is on the far rack
    u, z, y = sk.cell_pos(far[5])
    sk.retrieve("J1", "S-P1")
    seen_far = seen_near = False
    prev_carry = False
    for _ in range(4000):
        sk.update(0.02)
        ext = sk.fork - AISLE_U
        if abs(ext) > 1e-6:  # fork out: turned to that side, never while turning
            assert sk.rot == (1.0 if ext > 0 else 0.0)
            seen_far |= ext > 0
            seen_near |= ext < 0
        if 0 < sk.rot < 1:
            assert abs(ext) < 1e-9
        if not prev_carry and sk.carry:  # taken when the fork top reached the shelf surface (then keeps rising)
            assert y - 1e-6 <= sk.y <= y + 0.02 * 0.35 + 1e-6 and abs(sk.fork - u) < 1e-6
        if sk.carry and abs(sk.fork - u) < 1e-6 and abs(sk.z - z) < 1e-6:
            assert sk.y >= y - 1e-6  # carrying: never below the shelf it came from while still in it
        if not sk.carry and abs(sk.fork - u) < 1e-6 and abs(sk.z - z) < 1e-6 and not prev_carry:
            assert sk.y <= y + 1e-6  # entering empty: below the FOUP
        prev_carry = sk.carry
        if "J1" in sk.ready:
            break
    assert seen_far and seen_near and "J1" in sk.ready


def test_crane_stays_inside_stocker():
    sk = _sk()
    for k in range(6):
        sk.store("S-P1") if k % 2 else sk.retrieve(f"J{k}", "S-P2")
    zs = sk.bay_z
    for _ in range(1500):
        sk.update(0.2)
        assert min(zs) - 1e-6 <= sk.z <= max(zs) + 1e-6 or sk.z in (5.0, 2.0)
        assert 0.9 - DY - 1e-6 <= sk.y <= 3.45 + DY + 1e-6 and NEAR_U - 1e-6 <= sk.fork <= FAR_U + 1e-6
        assert 0.0 <= sk.rot <= 1.0


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

