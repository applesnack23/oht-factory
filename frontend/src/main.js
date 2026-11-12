// AI OHT Fab frontend: WebSocket state -> 3D fab + dashboard.
import { FabRenderer, STATE_COLOR } from './rendering/FabRenderer.js';
import { ReplayClient, WebSocketClient } from './network/WebSocketClient.js';
import { FleetPanel, KpiBar, RightPanel, TopBar } from './ui/panels.js';

const hex = (n) => '#' + n.toString(16).padStart(6, '0');

class App {
  constructor() {
    const replay = window.__FAB_REPLAY__;
    this.live = !replay;
    this.client = replay ? new ReplayClient(replay) : new WebSocketClient();
    this.renderer = new FabRenderer(document.getElementById('scene'));
    this.top = new TopBar(document.getElementById('topbar'), this);
    this.fleet = new FleetPanel(document.getElementById('left-panel'));
    this.right = new RightPanel(document.getElementById('right-panel'), this);
    this.kpi = new KpiBar(document.getElementById('kpi-bar'));
    const vp = document.getElementById('viewport');
    const views = document.createElement('div');
    views.className = 'views';
    views.innerHTML = '<button data-v="overview">Overview</button><button data-v="top">Top</button><button data-v="front">Front</button>'
      + '<button data-v="STK1">Stocker 1</button><button data-v="STK2">Stocker 2</button><button data-v="hoist">Hoist</button><button data-v="tool">Tool</button>';
    views.onclick = (e) => {
      const v = e.target.dataset.v;
      if (!v) return;
      if (v.startsWith('STK')) this.renderer.viewStocker(v);
      else if (v === 'tool') { if (!this.renderer.viewTool()) this.toast('no tool is active right now - try again in a second'); }
      else if (v === 'hoist') { if (!this.renderer.viewHoist()) this.toast('no vehicle is hoisting right now - try again in a second'); }
      else this.renderer.cameraCtl[v]();
    };
    vp.appendChild(views);
    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML = Object.entries(STATE_COLOR).filter(([k]) => k !== 'UNLOADING').map(([k, c]) => `<span><i style="background:${hex(c)}"></i>${k === 'LOADING' ? 'HOIST' : k}</span>`).join('')
      + '<span><i style="background:#f0f2f4"></i>FOUP</span><span><i style="background:#ff3b47;border-radius:50%"></i>hot lot</span>'
      + '<span><i style="background:#3ddc84"></i>tool processing</span><span><i style="background:#4aa3ff"></i>tool loading/unloading</span><span><i style="background:#f2a93b"></i>lots waiting in tool</span>'
      + '<span><i style="background:#8c6fd1"></i>shortcut (red = closed)</span><span><i style="background:#d94bd9"></i>route decision point</span>';
    vp.appendChild(legend);
    this.chips = document.getElementById('overlay');
    this.client.on('status', (s) => { this.top.setConnection(s); if (s === 'open') this.top.refresh(); });
    this.client.on('layout', (l) => this.renderer.buildLayout(l));
    this.client.on('state', (s) => this.onState(s));
    this.client.connect();
    if (this.live) setInterval(() => this.right.poll(), 1500);
    this._ui = 0;
    window.__app = this;
  }

  onState(s) {
    this.renderer.applyState(s);
    const now = performance.now();
    if (now - this._ui < 250) return;
    this._ui = now;
    this.fleet.update(s);
    this.right.update(s);
    this.kpi.update(s);
    const c = s.control || {};
    const w = c.watch;
    const watch = w ? `<div class="chip watch">● WATCHING TRAINING <b>${w.run}</b> · ${w.waiting ? 'waiting for first snapshot…' : `model @ <b>${Number(w.step).toLocaleString()}</b>${w.totalSteps ? ' / ' + Number(w.totalSteps).toLocaleString() : ''} steps${w.lastEval ? ` · last eval <b>${Number(w.lastEval.reward).toFixed(0)}</b>` : ''}`} · reload #${w.reloads}</div>` : '';
    this.chips.innerHTML = watch + `
      <div class="chip">sim t <b>${(s.time / 60).toFixed(1)} min</b> · ${c.state || 'replay'} · x${c.speed ?? 1} · scenario <b>${s.scenario}</b></div>
      <div class="chip">control <b>${c.mode || '–'}</b>${c.model ? ' · ' + c.model : ''} · reward <b>${c.episodeReward ?? '–'}</b></div>
      <div class="chip">render <b>${this.renderer.fps.toFixed(0)} fps</b></div>`;
  }

  toast(msg, persistent = false) {
    const t = document.createElement('div');
    t.className = 'chip';
    t.style.cssText = `position:fixed;right:16px;bottom:100px;z-index:9;pointer-events:auto;max-width:420px;${persistent ? 'border-color:#f2555a' : ''}`;
    t.textContent = msg;
    if (persistent) { t.title = 'click to close'; t.onclick = () => t.remove(); }
    document.body.appendChild(t);
    setTimeout(() => t.remove(), persistent ? 9000 : 3500);
  }
}

new App();
