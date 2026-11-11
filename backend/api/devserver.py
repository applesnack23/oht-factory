"""Dependency-free HTTP + WebSocket server (fallback for FastAPI).

Implements the same REST routes (backend/api/routes.py) and the /ws/factory
WebSocket (RFC 6455, text frames) on top of asyncio streams, so the system
can run and be integration-tested without FastAPI/uvicorn installed. Can also
serve a static frontend directory (e.g. frontend/dist).

    python -m backend.api.devserver --port 8000 [--static frontend/dist]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import struct
import traceback
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from backend.api.routes import ROUTES
from backend.api.service import FactoryService

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


class DevServer:
    def __init__(self, service: FactoryService, static_dir: Optional[str] = None):
        self.service = service
        self.static = Path(static_dir).resolve() if static_dir else None

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                method, target, _ = lines[0].split(" ", 2)
                headers = {}
                for l in lines[1:]:
                    if ":" in l:
                        k, v = l.split(":", 1)
                        headers[k.strip().lower()] = v.strip()
                body = b""
                n = int(headers.get("content-length", "0") or 0)
                if n:
                    body = await reader.readexactly(n)
                path = urlparse(target).path
                if path == "/ws/factory" and headers.get("upgrade", "").lower() == "websocket":
                    await self.websocket(reader, writer, headers)
                    return
                keep = await self.http(method, path, body, writer)
                if not keep or headers.get("connection", "").lower() == "close":
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            traceback.print_exc()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def http(self, method, path, body, writer) -> bool:
        if method == "OPTIONS":
            self._send(writer, 204, b"", "text/plain")
            return True
        fn = ROUTES.get((method, path))
        if fn is not None:
            try:
                data = json.loads(body.decode() or "{}") if body else None
                result = fn(self.service, data)
                self._send(writer, 200, json.dumps(result, default=_json_default).encode(), "application/json")
            except (ValueError, KeyError, RuntimeError, FileNotFoundError) as e:
                self._send(writer, 400, json.dumps({"detail": str(e)}).encode(), "application/json")
            except Exception as e:
                traceback.print_exc()
                self._send(writer, 500, json.dumps({"detail": repr(e)}).encode(), "application/json")
            await writer.drain()
            return True
        if path in ("/health", "/api/health"):
            self._send(writer, 200, b'{"ok": true}', "application/json")
        elif self.static and method == "GET":
            rel = path.lstrip("/") or "index.html"
            fp = (self.static / rel).resolve()
            if not str(fp).startswith(str(self.static)) or not fp.is_file():
                fp = self.static / "index.html"
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            if fp.suffix == ".js":
                ctype = "text/javascript"
            self._send(writer, 200, fp.read_bytes(), ctype)
        else:
            self._send(writer, 404, b'{"detail":"not found"}', "application/json")
        await writer.drain()
        return True

    @staticmethod
    def _send(writer, status, payload: bytes, ctype: str) -> None:
        reason = {200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}.get(status, "OK")
        hdr = (f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {len(payload)}\r\n"
               "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n"
               "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n")
        writer.write(hdr.encode() + payload)

    # ------------------------------------------------------------ websocket
    async def websocket(self, reader, writer, headers) -> None:
        key = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
        await writer.drain()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        def push(msg: str):
            if queue.full():
                try:
                    queue.get_nowait()  # drop stale frames for slow clients
                except Exception:
                    pass
            queue.put_nowait(msg)

        push(self.service.layout_message())
        push(self.service.state_message())
        self.service.subscribe(push)

        async def sender():
            while True:
                msg = await queue.get()
                writer.write(self._frame(msg.encode()))
                await writer.drain()

        task = asyncio.create_task(sender())
        try:
            while True:
                op, data = await self._read_frame(reader)
                if op == 0x8:  # close
                    writer.write(self._frame(b"", 0x8))
                    await writer.drain()
                    break
                if op == 0x9:  # ping
                    writer.write(self._frame(data, 0xA))
                    continue
                if op == 0x1:
                    reply = await self.service.handle_ws_message(data.decode("utf-8", "replace"))
                    if reply:
                        push(reply)
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.service.unsubscribe(push)
            task.cancel()

    @staticmethod
    def _frame(payload: bytes, opcode: int = 0x1) -> bytes:
        n = len(payload)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([n])
        elif n < 65536:
            head += bytes([126]) + struct.pack("!H", n)
        else:
            head += bytes([127]) + struct.pack("!Q", n)
        return head + payload

    @staticmethod
    async def _read_frame(reader):
        b1, b2 = await reader.readexactly(2)
        opcode = b1 & 0x0F
        masked = b2 & 0x80
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack("!H", await reader.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", await reader.readexactly(8))[0]
        mask = await reader.readexactly(4) if masked else b"\x00\x00\x00\x00"
        data = bytearray(await reader.readexactly(n))
        if masked:
            for i in range(n):
                data[i] ^= mask[i % 4]
        return opcode, bytes(data)


async def amain(host: str, port: int, level: str, static: Optional[str], autostart: bool = False):
    service = FactoryService(None if level == "full" else level)
    if autostart:
        service.start()
    server = DevServer(service, static)
    srv = await asyncio.start_server(server.handle, host, port)
    service.ensure_loop()
    print(f"AI Conveyor Factory dev server on http://{host}:{port}  (ws://{host}:{port}/ws/factory)", flush=True)
    async with srv:
        await srv.serve_forever()


def serve(host="127.0.0.1", port=8000, level="level6", static=None, autostart=False):
    asyncio.run(amain(host, port, level, static, autostart))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--level", default="level6")
    p.add_argument("--static", default=None)
    p.add_argument("--autostart", action="store_true")
    a = p.parse_args()
    serve(a.host, a.port, a.level, a.static, a.autostart)
