"""FabService: live OHT operation for the dashboard (framework-agnostic).

Runs the fab in real time (x1..x50), lets the selected controller (basic rule,
smart rule, PPO) take every decision, streams state over WebSocket, manages
the training subprocess and the "watch training" mode.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from backend.ai.oht_env import OHTEnv
from backend.policies.rules import BasicRulePolicy, PreassignRulePolicy, SmartRulePolicy
from backend.sim.config import PROJECT_ROOT, scenario_names

LOGS_DIR = PROJECT_ROOT / "backend" / "logs"
STATUS_FILE = LOGS_DIR / "training_status.json"
SPEEDS = (1, 2, 5, 10, 20, 50)
MODES = ("rule_basic", "rule_smart", "rule_preassign", "ppo")


class FabService:
    def __init__(self, scenario: str = "normal", seed: int = 42, broadcast_hz: float = 10.0):
        self.scenario = scenario
        self.seed = seed
        self.broadcast_interval = 1.0 / broadcast_hz
        self.speed = 5
        self.state = "stopped"
        self.mode = "rule_smart"
        self.model_name: Optional[str] = None
        self.ppo_policy = None
        self.subscribers: List[Callable[[str], Any]] = []
        self.train_proc: Optional[subprocess.Popen] = None
        self.train_run: Optional[str] = None
        self._task = None
        self.watch: Optional[Dict[str, Any]] = None
        self._last_watch_check = 0.0
        self.loop_stats = {"simStepsPerSec": 0.0, "lag": 0.0}
        self._build()

    # --------------------------------------------------------------- build
    def _build(self) -> None:
        self.env = OHTEnv(None if self.scenario == "normal" else self.scenario)
        self.env.fab.episode_length = 1e12  # live: run indefinitely
        self.fab = self.env.fab
        self.fab.reset(self.seed)
        self.rules = {"rule_basic": BasicRulePolicy(self.env), "rule_smart": SmartRulePolicy(self.env),
                      "rule_preassign": PreassignRulePolicy(self.env)}
        if self.ppo_policy is not None:
            self.ppo_policy.reset(self.env)
        self.episode_reward = 0.0
        self.decision_log: Deque[dict] = deque(maxlen=120)
        self.defer_count = 0

    def _policy(self):
        if self.mode == "ppo" and self.ppo_policy is not None:
            return self.ppo_policy
        return self.rules.get(self.mode, self.rules["rule_smart"])

    # ----------------------------------------------------------- stepping
    def _decide_all(self) -> None:
        f, env = self.fab, self.env
        pol = self._policy()
        guard = 0
        while guard < 500:
            ds = f.decisions()
            if not ds:
                return
            d = ds[0]
            env.decision = d
            obs = env._obs()
            a = int(np.asarray(pol.act(obs)).reshape(-1)[0])
            choice = a if a < len(d.candidates) else None
            res = f.apply_decision(d, choice)
            entry = {"t": round(f.time, 1), "kind": d.kind, "subject": d.subject, "choice": choice, "result": res["reason"],
                     "n": len(d.candidates), "mode": self.mode}
            if choice is not None and d.candidates:
                c = d.candidates[choice]
                entry["eta"] = round(c["eta"], 1)
                entry["target"] = c.get("vehicle", c.get("area", c.get("edge")))
                entry["bestEta"] = round(d.candidates[0]["eta"], 1)
                entry["busy"] = bool(c.get("busy"))
            if res["reason"] == "DEFERRED":
                self.defer_count += 1  # keep the monitor readable: defers are counted, not listed
            else:
                self.decision_log.append(entry)
            guard += 1

    def sim_step(self) -> None:
        self._decide_all()
        r = self.fab.step()
        self.episode_reward += sum(self.env.rc.step(r, self.fab.dt).values())

    def advance(self, n: int) -> None:
        for _ in range(n):
            self.sim_step()

    async def run_loop(self) -> None:
        dt = self.fab.dt
        last_b = 0.0
        next_t = time.perf_counter()
        window = deque(maxlen=30)
        while True:
            now = time.perf_counter()
            if self.state == "running":
                self.advance(self.speed)
                window.append((now, self.speed))
                if len(window) > 1:
                    span = window[-1][0] - window[0][0]
                    self.loop_stats["simStepsPerSec"] = sum(s for _, s in list(window)[1:]) / span if span > 0 else 0.0
            if self.watch is not None and now - self._last_watch_check >= 1.0:
                self._last_watch_check = now
                try:
                    self.watch_check()
                except Exception as e:
                    self.watch["error"] = repr(e)
            if now - last_b >= self.broadcast_interval:
                last_b = now
                await self.broadcast(self.state_message())
            next_t += dt
            delay = next_t - time.perf_counter()
            self.loop_stats["lag"] = max(0.0, -delay)
            if delay < -0.5:
                next_t = time.perf_counter()
                delay = 0
            await asyncio.sleep(max(0.0, delay))

    def ensure_loop(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_event_loop().create_task(self.run_loop())

    # ------------------------------------------------------------ control
    def start(self) -> dict:
        self.state = "running"
        return self.status()

    def pause(self) -> dict:
        if self.state == "running":
            self.state = "paused"
        return self.status()

    def stop(self) -> dict:
        self.state = "stopped"
        self._build()
        return self.status()

    def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> dict:
        if scenario:
            if scenario not in scenario_names():
                raise ValueError(f"unknown scenario {scenario}")
            self.scenario = scenario
        if seed is not None:
            self.seed = int(seed)
        was = self.state
        self._build()
        self.state = "paused" if was == "running" else was
        return self.status()

    def set_speed(self, speed: int) -> dict:
        if int(speed) not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        self.speed = int(speed)
        return self.status()

    def set_mode(self, mode: str, model: Optional[str] = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        self.watch = None
        if mode == "ppo":
            if model:
                self.load_model(model)
            if self.ppo_policy is None:
                raise ValueError("no PPO model loaded: choose a model in 'AI model' first")
        self.mode = mode
        return self.status()

    def load_model(self, name: str) -> dict:
        from backend.ai.inference import MODELS_DIR, PPOPolicy
        p = Path(name)
        if not p.is_absolute():
            p = MODELS_DIR / name
        pol = PPOPolicy(p)
        pol.reset(self.env)
        n = pol.model.obs_dim if hasattr(pol.model, "obs_dim") else self.env.obs_dim
        if n != self.env.obs_dim:
            raise ValueError("model observation size does not match this fab")
        self.ppo_policy = pol
        self.model_name = str(name)
        return {"loaded": str(name)}

    # ---------------------------------------------------------- reporting
    def status(self) -> dict:
        from backend.ai.inference import list_models
        return {"state": self.state, "mode": self.mode, "speed": self.speed, "scenario": self.scenario, "seed": self.seed,
                "model": self.model_name, "time": round(self.fab.time, 2), "speeds": list(SPEEDS),
                "scenarios": scenario_names(), "modes": list(MODES),
                "availableModes": {"rule_basic": True, "rule_smart": True, "rule_preassign": True, "ppo": bool(list_models())},
                "watch": self.watch_status(), "loop": self.loop_stats}

    def get_state(self) -> dict:
        st = self.fab.get_state()
        st["control"] = {"mode": self.mode, "state": self.state, "speed": self.speed, "model": self.model_name,
                         "episodeReward": round(self.episode_reward, 1), "watch": self.watch_status(),
                         "decisions": list(self.decision_log)[-12:], "defers": self.defer_count}
        return st

    def state_message(self) -> str:
        return json.dumps({"type": "state", "data": self.get_state()}, separators=(",", ":"))

    def layout_message(self) -> str:
        return json.dumps({"type": "layout", "data": self.fab.get_layout()}, separators=(",", ":"))

    def metrics(self) -> dict:
        return {**self.fab.summary(), "episodeReward": round(self.episode_reward, 1)}

    def subscribe(self, send) -> None:
        self.subscribers.append(send)

    def unsubscribe(self, send) -> None:
        if send in self.subscribers:
            self.subscribers.remove(send)

    async def broadcast(self, msg: str) -> None:
        dead = []
        for send in list(self.subscribers):
            try:
                r = send(msg)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                dead.append(send)
        for d in dead:
            self.unsubscribe(d)

    async def handle_ws_message(self, text: str) -> Optional[str]:
        try:
            msg = json.loads(text)
        except Exception:
            return json.dumps({"type": "error", "error": "invalid json"})
        if msg.get("type") == "ping":
            return json.dumps({"type": "pong", "time": self.fab.time})
        if msg.get("type") == "getLayout":
            return self.layout_message()
        return json.dumps({"type": "error", "error": "unknown message"})

    # ----------------------------------------------------------- training
    def training_status(self) -> dict:
        st = {}
        if STATUS_FILE.exists():
            try:
                st = json.loads(STATUS_FILE.read_text())
            except Exception:
                st = {}
        alive = self.train_proc is not None and self.train_proc.poll() is None
        st["processAlive"] = alive
        st.setdefault("state", "idle")
        if st["state"] == "running" and self.train_proc is not None and not alive:
            st["state"] = "exited"
        return st

    def start_training(self, scenario: str = "mix", timesteps: int = 200000, run: Optional[str] = None,
                       extra_args: Optional[List[str]] = None, init_from: Optional[str] = None) -> dict:
        if self.train_proc is not None and self.train_proc.poll() is None:
            raise RuntimeError("training already running")
        run = run or f"oht_{scenario}_ui_{time.strftime('%Y%m%d_%H%M%S')}"
        cmd = [sys.executable, "-m", "backend.ai.train_ppo", "--level", scenario, "--timesteps", str(int(timesteps)),
               "--run", run, "--status-file", str(STATUS_FILE)]
        if init_from:
            cmd += ["--init-from", init_from]
            from backend.ai.inference import load_model, MODELS_DIR
            m = load_model(MODELS_DIR / init_from)
            if hasattr(m, "cfg"):
                cmd += ["--hidden", *[str(h) for h in m.cfg["hidden"]]]
        cmd += list(extra_args or [])
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOGS_DIR / f"{run}.out", "w")
        self.train_proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.train_run = run
        STATUS_FILE.write_text(json.dumps({"run": run, "level": scenario, "state": "starting", "totalSteps": timesteps,
                                           "step": 0, "progress": 0.0, "log": []}))
        return {"run": run, "pid": self.train_proc.pid}

    def stop_training(self, timeout: float = 15.0) -> dict:
        if self.train_proc is None or self.train_proc.poll() is not None:
            return {"stopped": False, "reason": "not running"}
        stop = LOGS_DIR / self.train_run / "STOP"
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_text("stop")
        proc = self.train_proc
        threading.Timer(timeout, lambda: proc.poll() is None and proc.send_signal(signal.SIGTERM)).start()
        return {"stopping": True, "run": self.train_run}

    def list_models(self) -> List[dict]:
        from backend.ai.inference import list_models
        return list_models()

    # ---------------------------------------------------- watch training
    def start_watch(self, run: Optional[str] = None, speed: int = 10) -> dict:
        from backend.ai.inference import MODELS_DIR
        if not run:
            run = self.train_run
            if not run and STATUS_FILE.exists():
                run = json.loads(STATUS_FILE.read_text()).get("run")
        if not run:
            raise ValueError("no training run to watch")
        self.watch = {"run": run, "dir": str(MODELS_DIR / run), "mtime": None, "reloads": 0, "step": None,
                      "totalSteps": None, "snapshot": None, "lastEval": None, "waiting": True, "error": None}
        self.set_speed(speed)
        self.state = "running"
        self.watch_check()
        return self.watch_status()

    def stop_watch(self) -> dict:
        self.watch = None
        return self.status()

    def watch_check(self) -> bool:
        if self.watch is None:
            return False
        from backend.ai.inference import PPOPolicy
        d = Path(self.watch["dir"])
        f = next((d / n for n in ("latest_model.npz", "latest_model.zip") if (d / n).exists()), None)
        if f is None:
            return False
        mtime = f.stat().st_mtime
        if mtime == self.watch["mtime"]:
            return False
        meta = {}
        if (d / "latest_model.json").exists():
            try:
                meta = json.loads((d / "latest_model.json").read_text())
            except Exception:
                meta = {}
        self.ppo_policy = PPOPolicy(f)
        self.mode = "ppo"
        self.model_name = f"{self.watch['run']} @ {meta.get('step', '?')} steps"
        was = self.state
        self._build()  # fresh episode with the same seed for each snapshot
        self.state = was
        self.watch.update({"mtime": mtime, "reloads": self.watch["reloads"] + 1, "waiting": False, "step": meta.get("step"),
                           "totalSteps": meta.get("totalSteps"), "snapshot": meta.get("snapshot"), "lastEval": meta.get("lastEval")})
        return True

    def watch_status(self):
        return None if self.watch is None else {k: v for k, v in self.watch.items() if k not in ("dir", "mtime")}
