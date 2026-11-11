"""Core utilities: simulation clock, event manager, seeded random streams."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np


class SimulationClock:
    """Fixed-step simulation clock. Simulation time is independent of wall time."""

    def __init__(self, dt: float = 0.05):
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self._time = 0.0

    def tick(self) -> float:
        self.step_count += 1
        # Use step_count * dt to avoid floating point drift.
        self._time = self.step_count * self.dt
        return self._time

    @property
    def time(self) -> float:
        return self._time


@dataclass
class Event:
    eid: int
    type: str
    time: float
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.eid, "type": self.type, "time": round(self.time, 3), **self.data}


class EventManager:
    """Collects simulation events.

    `pending` holds events not yet consumed by the reward calculator; `drain()`
    returns them exactly once (prevents duplicate rewards). `history` keeps a
    bounded log for the UI.
    """

    def __init__(self, history_size: int = 500):
        self.history: Deque[Event] = deque(maxlen=history_size)
        self.pending: List[Event] = []
        self._next_id = 1
        self.counts: Dict[str, int] = {}

    def reset(self) -> None:
        self.history.clear()
        self.pending = []
        self._next_id = 1
        self.counts = {}

    def emit(self, etype: str, time: float, **data: Any) -> Event:
        ev = Event(self._next_id, etype, time, data)
        self._next_id += 1
        self.pending.append(ev)
        self.history.append(ev)
        self.counts[etype] = self.counts.get(etype, 0) + 1
        return ev

    def drain(self) -> List[Event]:
        out, self.pending = self.pending, []
        return out

    def recent(self, n: int = 50, exclude: Optional[set] = None) -> List[Dict[str, Any]]:
        items = [e for e in self.history if not exclude or e.type not in exclude]
        return [e.to_dict() for e in items[-n:]]


class SeededRandom:
    """Independent, reproducible random streams derived from one seed.

    Separate streams keep exogenous randomness (orders, material properties,
    sensor noise, fault schedules) independent of the control policy, so two
    policies evaluated on the same seed face exactly the same disturbances.
    """

    STREAMS = ("orders", "materials", "sensors", "faults", "policy", "misc")

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        ss = np.random.SeedSequence(self.seed)
        children = ss.spawn(len(self.STREAMS))
        self._streams = {name: np.random.default_rng(c) for name, c in zip(self.STREAMS, children)}

    def stream(self, name: str) -> np.random.Generator:
        return self._streams[name]

    def __getattr__(self, name: str) -> np.random.Generator:  # convenience: rng.orders
        streams = self.__dict__.get("_streams")
        if streams is not None and name in streams:
            return streams[name]
        raise AttributeError(name)
