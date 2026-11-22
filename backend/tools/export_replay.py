"""Record a headless run as a replay for the offline 3D viewer (no server needed).

    python -m backend.tools.export_replay --mode rule_preassign --minutes 15 --out offline/replay.json.gz

The replay is {layout, frames, frameInterval, status, meta}: frames are fab states sampled every
`interval` simulated seconds, played back in real time at x1 by the ReplayClient.
build_offline.py packs it into a single HTML file.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from backend.api.service import FabService


def slim(st: dict) -> dict:
    st = dict(st)
    st["jobs"] = st.get("jobs", [])[:18]  # the jobs table shows 18 rows
    st["events"] = st.get("events", [])[-12:]  # the event log shows 12 lines
    c = dict(st.get("control", {}))
    c["state"], c["watch"] = "replay", None
    st["control"] = c
    return st


def record(mode: str, scenario: str, minutes: float, interval: float, warmup: float, seed: int, model: str | None) -> dict:
    svc = FabService(scenario=scenario, seed=seed)
    svc.set_mode(mode, model)
    dt = svc.fab.dt
    per = max(1, int(round(interval / dt)))
    svc.advance(int(warmup / dt))
    frames = []
    for _ in range(int(minutes * 60 / (per * dt))):
        svc.advance(per)
        frames.append(slim(svc.get_state()))
    status = {k: v for k, v in svc.status().items() if k in ("mode", "scenario", "seed", "model", "speeds", "scenarios", "modes")}
    status.update(speed=5, modes=[svc.mode], scenarios=[scenario])  # the replay shows one recorded run
    return {"layout": svc.fab.get_layout(), "frames": frames, "frameInterval": per * dt, "status": status,
            "meta": {"mode": svc.mode, "model": model, "scenario": scenario, "seed": seed, "warmup": warmup, "minutes": minutes}}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="rule_preassign", help="rule_basic | rule_smart | rule_preassign | ppo")
    p.add_argument("--model", default=None, help="model name for --mode ppo (backend/models/<name>)")
    p.add_argument("--scenario", default="normal")
    p.add_argument("--minutes", type=float, default=15, help="simulated minutes to record")
    p.add_argument("--interval", type=float, default=0.4, help="simulated seconds between frames")
    p.add_argument("--warmup", type=float, default=300, help="simulated seconds before recording starts")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="offline/replay.json.gz")
    a = p.parse_args(argv)
    data = record(a.mode, a.scenario, a.minutes, a.interval, a.warmup, a.seed, a.model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    out.write_bytes(gzip.compress(raw, 9))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB gz, {len(raw) / 1e6:.1f} MB json, {len(data['frames'])} frames)")


if __name__ == "__main__":
    main()
