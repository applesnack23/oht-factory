import asyncio
import json
import shutil
import socket
import threading
import time
import urllib.request

import numpy as np

from backend.ai.inference import MODELS_DIR, PPOPolicy, load_model
from backend.ai.numpy_ppo import NumpyPPO
from backend.ai.oht_env import OHTEnv
from backend.ai.rollout import run_episode
from backend.api.devserver import DevServer
from backend.api.service import FabService


def test_train_smoke_save_load_and_inference():
    from backend.ai import train_ppo
    meta = train_ppo.main(["--level", "normal", "--smoke", "--run", "pytest_oht", "--algo", "numpy"])
    assert meta["timesteps"] >= 1024
    m = load_model(MODELS_DIR / "pytest_oht" / "best_model.npz")
    env = OHTEnv(None)
    ep = run_episode(env, PPOPolicy(m), 7)
    assert ep["delivered"] > 0
    assert (MODELS_DIR / "pytest_oht" / "latest_model.json").exists()
    for d in (train_ppo.MODELS_DIR, train_ppo.CKPT_DIR, train_ppo.LOGS_DIR):
        shutil.rmtree(d / "pytest_oht", ignore_errors=True)


def test_behavior_cloning_smoke():
    from backend.ai import pretrain_bc
    meta = pretrain_bc.main(["--level", "normal", "--run", "pytest_bc", "--episodes", "2", "--dagger", "0", "--epochs", "2",
                             "--eval-episodes", "1", "--hidden", "16", "16"])
    assert meta["samples"] > 500 and "teacher_eval" in meta
    shutil.rmtree(MODELS_DIR / "pytest_bc", ignore_errors=True)


def test_service_controls_and_watch(tmp_path):
    svc = FabService("normal")
    svc.set_mode("rule_basic")
    svc.advance(int(120 / svc.fab.dt))
    assert svc.metrics()["delivered"] > 0 and svc.decision_log
    e = OHTEnv(None)
    model = NumpyPPO(e.observation_space.shape[0], e.action_space.nvec, seed=0)
    run = "pytest_watch"
    d = MODELS_DIR / run
    model.save(d / "latest_model")
    (d / "latest_model.json").write_text(json.dumps({"step": 5000, "totalSteps": 9000, "snapshot": 2}))
    try:
        w = svc.start_watch(run)
        assert not w["waiting"] and w["step"] == 5000 and svc.mode == "ppo" and svc.fab.time == 0.0
        svc.advance(100)
        assert svc.decision_log and svc.decision_log[-1]["mode"] == "ppo"
        svc.set_mode("rule_smart")
        assert svc.watch is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_http_and_websocket():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    svc = FabService("normal")
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.start_server(DevServer(svc).handle, "127.0.0.1", port))
        svc.ensure_loop(); ready.set(); loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(5)
    base = f"http://127.0.0.1:{port}"

    def call(m, p, b=None):
        req = urllib.request.Request(base + p, data=json.dumps(b).encode() if b is not None else None, method=m)
        return json.loads(urllib.request.urlopen(req, timeout=10).read())

    st = call("POST", "/api/fab/start", {"speed": 50})
    assert st["state"] == "running"
    time.sleep(1.5)
    assert call("GET", "/api/fab/metrics")["time"] > 10
    assert call("POST", "/api/fab/reset", {"scenario": "track_block"})["scenario"] == "track_block"
    assert call("POST", "/api/control/mode", {"mode": "rule_basic"})["mode"] == "rule_basic"
    # websocket handshake + first frames
    import base64, os, struct
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall(f"GET /ws/fab HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(4096)
    head, buf = buf.split(b"\r\n\r\n", 1)
    assert b"101" in head
    msgs = []
    while len(msgs) < 2:
        while len(buf) < 4:
            buf += sock.recv(65536)
        n = buf[1] & 0x7F; off = 2
        if n == 126:
            n = struct.unpack("!H", buf[2:4])[0]; off = 4
        elif n == 127:
            while len(buf) < 10:
                buf += sock.recv(65536)
            n = struct.unpack("!Q", buf[2:10])[0]; off = 10
        while len(buf) < off + n:
            buf += sock.recv(65536)
        msgs.append(json.loads(buf[off:off + n])); buf = buf[off + n:]
    sock.close()
    assert msgs[0]["type"] == "layout" and len(msgs[0]["data"]["ports"]) == 120
    assert msgs[1]["type"] == "state" and len(msgs[1]["data"]["vehicles"]) == 50
    loop.call_soon_threadsafe(loop.stop)
