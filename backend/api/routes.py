"""Route table shared by the FastAPI app and the fallback dev server."""
from __future__ import annotations

from backend.api.service import FabService


def _b(b):
    return b or {}


def _start(s: FabService, b):
    if _b(b).get("speed"):
        s.set_speed(int(b["speed"]))
    return s.start()


ROUTES = {
    ("GET", "/api/fab/state"): lambda s, b: s.get_state(),
    ("GET", "/api/fab/layout"): lambda s, b: s.fab.get_layout(),
    ("GET", "/api/fab/metrics"): lambda s, b: s.metrics(),
    ("GET", "/api/fab/status"): lambda s, b: s.status(),
    ("POST", "/api/fab/reset"): lambda s, b: s.reset(_b(b).get("scenario"), _b(b).get("seed")),
    ("POST", "/api/fab/start"): _start,
    ("POST", "/api/fab/pause"): lambda s, b: s.pause(),
    ("POST", "/api/fab/stop"): lambda s, b: s.stop(),
    ("POST", "/api/fab/tool-mode"): lambda s, b: s.fab.flow.set_mode(_b(b)["tool"], _b(b).get("mode", "auto")),
    ("POST", "/api/fab/speed"): lambda s, b: s.set_speed(int(_b(b).get("speed", 5))),
    ("GET", "/api/models"): lambda s, b: {"models": s.list_models(), "loaded": s.model_name},
    ("POST", "/api/models/load"): lambda s, b: s.load_model(_b(b)["name"]),
    ("POST", "/api/control/mode"): lambda s, b: s.set_mode(_b(b).get("mode", "rule_smart"), _b(b).get("model")),
    ("GET", "/api/control/decisions"): lambda s, b: {"decisions": list(s.decision_log)},
    ("GET", "/api/training/status"): lambda s, b: s.training_status(),
    ("POST", "/api/training/start"): lambda s, b: s.start_training(_b(b).get("scenario", "mix"), int(_b(b).get("timesteps", 200000)),
                                                                _b(b).get("run"), _b(b).get("extra_args"), _b(b).get("init_from")),
    ("POST", "/api/training/stop"): lambda s, b: s.stop_training(),
    ("POST", "/api/training/watch"): lambda s, b: s.start_watch(_b(b).get("run"), int(_b(b).get("speed", 10))),
    ("POST", "/api/training/unwatch"): lambda s, b: s.stop_watch(),
}
