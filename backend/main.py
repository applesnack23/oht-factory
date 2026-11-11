"""AI OHT Fab - FastAPI server.

    python -m backend.main [--port 8000] [--scenario normal]
REST routes: backend/api/routes.py (same table as the fallback dev server).
WebSocket /ws/fab: first message `layout`, then `state` at ~10 Hz.
Without FastAPI installed, falls back to backend/api/devserver.py.
"""
from __future__ import annotations

import argparse
import asyncio

from backend.api.routes import ROUTES
from backend.api.service import FabService

try:
    from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    HAVE_FASTAPI = False


def create_app(scenario: str = "normal"):  # pragma: no cover - requires fastapi
    service = FabService(scenario)
    app = FastAPI(title="AI OHT Fab", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.service = service

    @app.on_event("startup")
    async def _startup():
        service.ensure_loop()

    def make(fn, method):
        async def handler(request: Request):
            body = None
            if method == "POST":
                raw = await request.body()
                if raw:
                    import json
                    body = json.loads(raw)
            try:
                return fn(service, body)
            except (ValueError, KeyError, RuntimeError, FileNotFoundError) as e:
                raise HTTPException(status_code=400, detail=str(e))
        return handler

    for (method, path), fn in ROUTES.items():  # every route runs on the event-loop thread
        app.add_api_route(path, make(fn, method), methods=[method])

    @app.websocket("/ws/fab")
    async def ws_fab(ws: WebSocket):
        await ws.accept()
        await ws.send_text(service.layout_message())
        await ws.send_text(service.state_message())
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        def push(msg: str):
            if queue.full():
                try:
                    queue.get_nowait()
                except Exception:
                    pass
            queue.put_nowait(msg)

        service.subscribe(push)

        async def sender():
            while True:
                await ws.send_text(await queue.get())

        task = asyncio.create_task(sender())
        try:
            while True:
                reply = await service.handle_ws_message(await ws.receive_text())
                if reply:
                    push(reply)
        except WebSocketDisconnect:
            pass
        finally:
            service.unsubscribe(push)
            task.cancel()

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--scenario", default="normal")
    p.add_argument("--static", default=None)
    a = p.parse_args()
    if HAVE_FASTAPI:  # pragma: no cover
        import uvicorn
        uvicorn.run(create_app(a.scenario), host=a.host, port=a.port)
    else:
        print("FastAPI not installed -> dependency-free dev server (same API)")
        from backend.api.devserver import serve
        serve(a.host, a.port, a.scenario, a.static)


if __name__ == "__main__":
    main()
