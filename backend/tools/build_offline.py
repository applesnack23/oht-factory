"""Pack the frontend + a recorded replay into ONE html file that runs without a server.

    python -m backend.tools.export_replay --out offline/replay.json.gz
    python -m backend.tools.build_offline --replay offline/replay.json.gz --out offline/viewer.html

Double-click the html (or open it from any folder / USB stick / mail attachment): no backend,
no npm, no internet. three.js comes from frontend/node_modules (run `npm install` once).

How: every ES module under frontend/src plus three.js becomes a data: URL behind an import map,
with relative imports rewritten to those names; the replay is stored gzip+base64 and inflated
in the browser (DecompressionStream) before main.js starts.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONT = ROOT / "frontend"
IMPORT_RE = re.compile(r"""(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])([^'"]+)\2""")


def data_url(code: str) -> str:
    return "data:text/javascript;charset=utf-8;base64," + base64.b64encode(code.encode("utf-8")).decode("ascii")


def three_files() -> dict:
    nm = FRONT / "node_modules" / "three"
    files = {"three": nm / "build" / "three.module.js"}
    for p in (FRONT / "src").rglob("*.js"):
        for _, _, spec in IMPORT_RE.findall(p.read_text(encoding="utf-8")):
            if spec.startswith("three/"):
                files[spec] = nm / spec[len("three/"):]
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise SystemExit("three.js not found - run `cd frontend && npm install` first:\n  " + "\n  ".join(missing))
    return files


def build(replay: Path, out: Path, title: str | None = None) -> Path:
    src = FRONT / "src"
    main_code = (src / "main.js").read_text(encoding="utf-8")
    m = re.search(r"window\.(__\w+_REPLAY__)", main_code)
    if not m:
        raise SystemExit("frontend/src/main.js has no window.__*_REPLAY__ hook")
    global_name = m.group(1)

    imports = {}
    for spec, path in three_files().items():
        imports[spec] = data_url(path.read_text(encoding="utf-8"))
    for p in sorted(src.rglob("*.js")):
        rel = p.relative_to(src).as_posix()

        def fix(mm, here=p.parent):
            spec = mm.group(3)
            if spec.startswith("."):
                spec = "app/" + (here / spec).resolve().relative_to(src.resolve()).as_posix()
            return f"{mm.group(1)}{mm.group(2)}{spec}{mm.group(2)}"

        imports["app/" + rel] = data_url(IMPORT_RE.sub(fix, p.read_text(encoding="utf-8")))

    html = (FRONT / "index.html").read_text(encoding="utf-8")
    css = "\n".join(p.read_text(encoding="utf-8") for p in sorted((src / "styles").glob("*.css")))
    html = re.sub(r'<link rel="stylesheet"[^>]*>', lambda _: f"<style>\n{css}\n</style>", html)
    if title:
        html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html)
    payload = base64.b64encode(replay.read_bytes()).decode("ascii")
    loader = f"""<div id="offline-loading" style="position:fixed;inset:0;display:grid;place-items:center;background:#0d1117;color:#cfd8e3;font:15px system-ui;z-index:99">재생 데이터를 불러오는 중…</div>
  <script type="importmap">{json.dumps({"imports": imports})}</script>
  <script id="replay-data" type="application/octet-stream">{payload}</script>
  <script type="module">
    const b64 = document.getElementById('replay-data').textContent.trim();
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    window.{global_name} = await new Response(stream).json();
    document.getElementById('offline-loading').remove();
    await import('app/main.js');
  </script>"""
    html = re.sub(r'<script type="module" src="[^"]*main\.js"></script>', lambda _: loader, html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--replay", default="offline/replay.json.gz")
    p.add_argument("--out", default="offline/viewer.html")
    p.add_argument("--title", default=None)
    a = p.parse_args()
    o = build(Path(a.replay), Path(a.out), a.title)
    print(f"wrote {o} ({o.stat().st_size / 1e6:.1f} MB)")
