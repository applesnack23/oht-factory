"""Process tools at the bay load ports (visual model, driven by the real transport jobs).

A FOUP dropped on a tool's load port rests for DWELL s, slides along the tray into the tool
(SLIDE s), waits in the tool's internal buffer and is processed one lot at a time
(processing time per process type, ±20%). For a job that starts at a tool port, the tool pushes
the FOUP out onto the port during the SLIDE s before the transport request is released (the
simulation draws requests SLIDE s ahead), so the FOUP is on the port when the request appears.
Processing never feeds back into demand. The tools use their own random generator, so the
demand random stream is unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

DWELL, SLIDE = 1.0, 3.0
# seconds per lot (visual; tools are kept busy like a loaded fab)
PROCESS_TIME = {"LITHO": 420, "ETCH": 360, "DEPO": 480, "CMP": 390, "DIFF": 600, "IMPLANT": 300, "METRO": 240, "CLEAN": 270}


class Tool:
    __slots__ = ("port", "process", "inbound", "buffer", "proc", "done", "outbound")

    def __init__(self, port: str, process: str):
        self.port, self.process = port, process
        self.inbound: List[float] = []  # time each delivered FOUP starts sliding in
        self.buffer = 0  # FOUPs inside, waiting for processing
        self.proc: Optional[tuple] = None  # (start, end)
        self.done = 0
        self.outbound: Dict[str, float] = {}  # job id -> time the FOUP starts sliding out


class ToolSet:
    def __init__(self, layout, seed: int):
        self.tools = {p.id: Tool(p.id, p.process) for p in layout.ports.values() if p.kind == "TOOL"}
        self.rng = np.random.default_rng(seed * 104729 + 7)
        self.ready_t: Dict[str, float] = {}  # job id -> time its FOUP is on the port
        for tool in self.tools.values():  # the fab is already running: WIP inside most tools
            if self.rng.random() < 0.8:
                dur = self._duration(tool)
                start = -float(self.rng.uniform(0, dur))
                tool.proc = (start, start + dur)
                tool.buffer = int(self.rng.integers(0, 3))

    def _duration(self, tool: Tool) -> float:
        return PROCESS_TIME.get(tool.process, 400) * float(self.rng.uniform(0.8, 1.2))

    # ------------------------------------------------------------ events
    def delivered(self, port: str, t: float) -> None:
        tool = self.tools.get(port)
        if tool is not None:
            tool.inbound.append(t + DWELL)

    def job_created(self, job_id: str, port: str, t: float) -> None:
        tool = self.tools.get(port)
        if tool is not None:
            tool.outbound[job_id] = t
            self.ready_t[job_id] = t + SLIDE

    def picked(self, job_id: str) -> None:
        self.ready_t.pop(job_id, None)

    def ready(self, job_id: str, t: float) -> bool:
        return self.ready_t.get(job_id, -1.0) <= t

    # ------------------------------------------------------------ time
    def update(self, t: float) -> None:
        for tool in self.tools.values():
            if tool.inbound and tool.inbound[0] + SLIDE <= t:
                tool.inbound = [s for s in tool.inbound if s + SLIDE > t]
                tool.buffer += 1
            for jid in [j for j, s in tool.outbound.items() if s + SLIDE <= t]:
                del tool.outbound[jid]
            if tool.proc is not None and tool.proc[1] <= t:
                tool.proc, tool.done = None, tool.done + 1
            if tool.proc is None and tool.buffer > 0:
                tool.buffer -= 1
                tool.proc = (t, t + self._duration(tool))

    # ------------------------------------------------------------ view
    def resting_on_port(self, t: float) -> List[str]:
        """Ports with a delivered FOUP that has not started sliding in yet."""
        return [tool.port for tool in self.tools.values() if any(s > t for s in tool.inbound)]

    def state(self) -> dict:
        """Compact view: active tools [port, procStart, procEnd, buffer, done] and moving FOUPs [port, start, dir]."""
        active, moves = [], []
        for tool in self.tools.values():
            if tool.proc or tool.buffer or tool.done:
                p = tool.proc or (0.0, 0.0)
                active.append([tool.port, round(p[0], 2), round(p[1], 2), tool.buffer, tool.done])
            for s in tool.inbound:
                moves.append([tool.port, round(s, 2), 1])
            for s in tool.outbound.values():
                moves.append([tool.port, round(s, 2), -1])
        return {"slide": SLIDE, "active": active, "moves": moves}
