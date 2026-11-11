// WebSocket client for /ws/fab with auto-reconnect, plus a REST helper.
// A ReplayClient with the same interface plays back recorded frames (demo mode).

export function apiBase() {
  const q = new URLSearchParams(location.search);
  return q.get('server') || '';
}

export async function api(method, path, body) {
  const res = await fetch(apiBase() + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export class WebSocketClient {
  constructor(url) {
    const q = new URLSearchParams(location.search);
    const server = q.get('server');
    if (url) this.url = url;
    else if (server) this.url = server.replace(/^http/, 'ws') + '/ws/fab';
    else this.url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/fab`;
    this.handlers = { layout: [], state: [], status: [], message: [] };
    this.retry = 500;
    this.connected = false;
    this.live = true;
  }

  on(type, fn) { (this.handlers[type] ||= []).push(fn); return this; }
  emit(type, data) { for (const fn of this.handlers[type] || []) fn(data); }

  connect() {
    this.emit('status', 'connecting');
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => { this.connected = true; this.retry = 500; this.emit('status', 'open'); };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      this.emit('message', msg);
      if (msg.type === 'layout') this.emit('layout', msg.data);
      else if (msg.type === 'state') this.emit('state', msg.data);
    };
    ws.onclose = () => {
      this.connected = false;
      this.emit('status', 'closed');
      setTimeout(() => this.connect(), this.retry);
      this.retry = Math.min(this.retry * 2, 8000);
    };
    ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
  }

  send(obj) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
  }
}

// Plays back {layout, frames:[state...], frameInterval} recorded by
// backend/tools/export_replay.py. Used for offline demos (no server).
export class ReplayClient {
  constructor(replay) {
    this.replay = replay;
    this.handlers = { layout: [], state: [], status: [], message: [] };
    this.index = 0;
    this.speed = 1;
    this.playing = true;
    this.live = false;
  }
  on(type, fn) { (this.handlers[type] ||= []).push(fn); return this; }
  emit(type, data) { for (const fn of this.handlers[type] || []) fn(data); }
  connect() {
    this.emit('status', 'replay');
    this.emit('layout', this.replay.layout);
    const dt = (this.replay.frameInterval || 0.2) * 1000;
    const tick = () => {
      if (this.playing && this.replay.frames.length) {
        this.emit('state', this.replay.frames[this.index]);
        this.index = (this.index + 1) % this.replay.frames.length;
      }
      setTimeout(tick, dt / this.speed);
    };
    tick();
  }
  send() {}
}
