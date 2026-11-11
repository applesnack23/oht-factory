// Dashboard UI: top bar, fleet panel (left), AI + jobs panel (right), KPI bar (bottom).
import { api } from '../network/WebSocketClient.js';
import { STATE_COLOR } from '../rendering/FabRenderer.js';

const MODE_LABEL = { rule_basic: 'Rule: nearest vehicle', rule_smart: 'Rule: smart', rule_preassign: 'Rule: smart + pre-assign', ppo: 'AI (PPO)' };
const hex = (n) => '#' + n.toString(16).padStart(6, '0');
const f1 = (v, d = 1) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));

export class TopBar {
  constructor(el, app) {
    this.app = app;
    el.innerHTML = `
      <div class="brand">AI OHT Fab<small>Semiconductor AMHS digital twin · RL dispatch / routing / repositioning</small></div>
      <div class="group"><button id="b-start" class="primary">▶ Start</button><button id="b-pause">❚❚ Pause</button>
        <button id="b-stop" class="danger">■ Stop</button><button id="b-reset">↺ Reset</button></div>
      <div class="group"><label>Speed</label><select id="s-speed"></select></div>
      <div class="group"><label>Control</label><select id="s-mode"></select></div>
      <div class="group"><label>Scenario</label><select id="s-scen"></select></div>
      <div class="conn"><span id="c-dot" class="dot"></span><span id="c-text">connecting…</span></div>`;
    this.$ = (id) => el.querySelector('#' + id);
    this.$('b-start').onclick = () => this.call('POST', '/api/fab/start', { speed: Number(this.$('s-speed').value) });
    this.$('b-pause').onclick = () => this.call('POST', '/api/fab/pause');
    this.$('b-stop').onclick = () => this.call('POST', '/api/fab/stop');
    this.$('b-reset').onclick = () => this.call('POST', '/api/fab/reset', { scenario: this.$('s-scen').value });
    this.$('s-speed').onchange = (e) => this.call('POST', '/api/fab/speed', { speed: Number(e.target.value) });
    this.$('s-scen').onchange = (e) => this.call('POST', '/api/fab/reset', { scenario: e.target.value });
    this.$('s-mode').onchange = (e) => this.setMode(e.target.value);
  }

  async call(m, p, b) {
    try { const r = await api(m, p, b); if (r && r.state) this.apply(r); return r; } catch (e) { this.app.toast(e.message, true); this.refresh(); return null; }
  }

  async setMode(mode) {
    const sel = this.app.right.selectedModel();
    if (mode === 'ppo' && sel) {
      try { await api('POST', '/api/models/load', { name: sel }); } catch (e) { this.app.toast(e.message, true); }
    }
    await this.call('POST', '/api/control/mode', { mode });
  }

  async refresh() { try { this.apply(await api('GET', '/api/fab/status')); } catch { /* offline */ } }

  apply(st) {
    const fill = (id, items, labelFn, cur) => {
      const s = this.$(id);
      if (s.options.length !== items.length) s.innerHTML = items.map((v) => `<option value="${v}">${labelFn(v)}</option>`).join('');
      s.value = String(cur);
    };
    fill('s-speed', st.speeds, (v) => `x${v}`, st.speed);
    fill('s-mode', st.modes, (v) => MODE_LABEL[v] || v, st.mode);
    fill('s-scen', st.scenarios, (v) => v, st.scenario);
    this.$('b-start').classList.toggle('active', st.state === 'running');
    this.$('b-pause').classList.toggle('active', st.state === 'paused');
  }

  setConnection(s) {
    this.$('c-dot').className = 'dot ' + (s === 'open' ? 'ok' : s === 'replay' ? 'warn' : 'bad');
    this.$('c-text').textContent = s === 'open' ? 'server connected' : s === 'replay' ? 'replay' : 'reconnecting…';
  }
}

export class FleetPanel {
  constructor(el) {
    el.innerHTML = `
      <div class="section"><h3>Fleet <span id="fl-n" class="muted"></span></h3><div class="statebar" id="fl-bar"></div><div class="kv" id="fl-kv"></div></div>
      <div class="section"><h3>Bays</h3><table><thead><tr><th>Area</th><th class="num">waiting</th><th class="num">idle OHT</th><th class="num">jobs/min</th></tr></thead><tbody id="fl-areas"></tbody></table></div>
      <div class="section"><h3>Vehicles</h3><table><thead><tr><th>#</th><th>State</th><th class="num">m/s</th><th>Job</th><th>Home</th></tr></thead><tbody id="fl-v"></tbody></table></div>`;
    this.$ = (id) => el.querySelector('#' + id);
  }

  update(s) {
    const vs = s.vehicles || [];
    const cnt = {};
    for (const v of vs) cnt[v.state] = (cnt[v.state] || 0) + 1;
    this.$('fl-n').textContent = `${vs.length} vehicles`;
    this.$('fl-bar').innerHTML = Object.keys(STATE_COLOR).map((k) => `<i style="width:${((cnt[k] || 0) / Math.max(1, vs.length)) * 100}%;background:${hex(STATE_COLOR[k])}"></i>`).join('');
    this.$('fl-kv').innerHTML = Object.keys(STATE_COLOR).map((k) => `<span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${hex(STATE_COLOR[k])}"></i> ${k}</span><span>${cnt[k] || 0}</span>`).join('');
    const areas = s.areas || {};
    this.$('fl-areas').innerHTML = Object.entries(areas).map(([id, a]) =>
      `<tr><td>${id}</td><td class="num ${a.pending >= 6 ? 'hot' : ''}">${a.pending}</td><td class="num">${a.idle}</td><td class="num">${f1(a.rate)}</td></tr>`).join('');
    this.$('fl-v').innerHTML = vs.slice(0, 60).map((v) =>
      `<tr><td>${v.id}</td><td><span class="badge b-${v.state}">${v.state}</span></td><td class="num">${f1(v.v)}</td><td>${v.job ? `${v.hot ? '<span class="hot">●</span> ' : ''}${v.job}` : ''}</td><td class="muted">${v.home || ''}</td></tr>`).join('');
  }
}

export class RightPanel {
  constructor(el, app) {
    this.app = app;
    el.innerHTML = `
      <div class="section"><h3>AI model</h3>
        <div class="row"><select id="m-sel"><option value="">— no model —</option></select><button id="m-load" style="flex:0 0 auto">Load</button></div>
        <div class="kv" id="m-kv"></div></div>
      <div class="section"><h3>AI training <span id="t-state" class="muted"></span></h3>
        <div class="row"><label>Scenario</label><select id="t-scen"></select></div>
        <div class="row"><label>Timesteps</label><input id="t-steps" type="number" value="200000" step="10000" min="1024"></div>
        <div class="row"><label>Start from</label><select id="t-init"><option value="">scratch (new model)</option></select></div>
        <div class="row"><button id="t-start" class="primary">Start training</button><button id="t-stop" class="danger">Stop</button></div>
        <div class="row"><button id="t-watch">Watch in 3D</button><button id="t-unwatch">Stop watching</button></div>
        <div class="bar" style="height:8px;margin:8px 0"><i id="t-bar" style="width:0%"></i></div>
        <div class="kv" id="t-kv"></div>
        <div class="log" id="t-log" style="margin-top:8px;max-height:110px"></div></div>
      <div class="section"><h3>Transport jobs <span id="j-n" class="muted"></span></h3>
        <table><thead><tr><th>Job</th><th>From → To</th><th>State</th><th class="num">age s</th></tr></thead><tbody id="j-body"></tbody></table></div>
      <div class="section"><h3>AI decision monitor</h3><div class="log" id="d-log" style="max-height:170px"></div></div>
      <div class="section"><h3>Events</h3><div class="log" id="e-log" style="max-height:120px"></div></div>`;
    this.$ = (id) => el.querySelector('#' + id);
    this.$('m-load').onclick = async () => {
      const name = this.selectedModel();
      if (!name) return;
      try { await api('POST', '/api/models/load', { name }); app.toast(`model loaded: ${name} — set Control to AI (PPO)`); } catch (e) { app.toast(e.message, true); }
    };
    this.$('t-start').onclick = async () => {
      const body = { scenario: this.$('t-scen').value, timesteps: Number(this.$('t-steps').value) };
      if (this.$('t-init').value) body.init_from = this.$('t-init').value;
      try { const r = await api('POST', '/api/training/start', body); app.toast(`training started: ${r.run}`); } catch (e) { app.toast(e.message, true); }
    };
    this.$('t-stop').onclick = () => api('POST', '/api/training/stop').catch((e) => app.toast(e.message, true));
    this.$('t-watch').onclick = async () => {
      try { const w = await api('POST', '/api/training/watch', { speed: 10 }); app.toast(`watching ${w.run}`); app.top.refresh(); } catch (e) { app.toast(e.message, true); }
    };
    this.$('t-unwatch').onclick = () => api('POST', '/api/training/unwatch').then(() => app.top.refresh());
  }

  selectedModel() { return this.$('m-sel').value; }

  async poll() {
    try {
      const [m, st, fs] = await Promise.all([api('GET', '/api/models'), api('GET', '/api/training/status'), api('GET', '/api/fab/status')]);
      const sel = this.$('m-sel');
      const names = m.models.map((x) => x.name);
      if (sel.dataset.sig !== names.join('|')) {
        sel.dataset.sig = names.join('|');
        const cur = sel.value || m.loaded || '';
        sel.innerHTML = '<option value="">— no model —</option>' + m.models.map((x) =>
          `<option value="${x.name}">${x.run}${x.algo ? ' · ' + x.algo : ''}${x.bestEval != null ? ' · ' + Number(x.bestEval).toFixed(0) : ''}</option>`).join('');
        sel.value = names.includes(cur) ? cur : '';
        const init = this.$('t-init');
        init.innerHTML = '<option value="">scratch (new model)</option>' + m.models.map((x) => `<option value="${x.run}">${x.run}</option>`).join('');
      }
      const ts = this.$('t-scen');
      if (ts.options.length !== fs.scenarios.length + 1) {
        ts.innerHTML = ['mix', ...fs.scenarios].map((s) => `<option value="${s}">${s === 'mix' ? 'mix (all scenarios)' : s}</option>`).join('');
      }
      this.$('m-kv').innerHTML = `<span>Loaded</span><span>${m.loaded || '–'}</span>`;
      this.$('t-state').textContent = st.state || 'idle';
      this.$('t-bar').style.width = `${Math.round((st.progress || 0) * 100)}%`;
      this.$('t-kv').innerHTML = `<span>Run</span><span>${st.run || '–'}</span><span>Step</span><span>${st.step ?? 0} / ${st.totalSteps ?? '–'}</span>
        <span>Mean reward</span><span>${f1(st.meanReward)}</span><span>Best eval</span><span>${f1(st.bestEvalReward)}</span>
        <span>Moves / h</span><span>${f1(st.throughput, 0)}</span><span>Avg delivery s</span><span>${f1(st.avgDeliveryTime)}</span>`;
      this.$('t-log').textContent = (st.log || []).slice(-10).join('\n');
      this.$('t-start').disabled = st.processAlive === true;
      this.$('t-stop').disabled = st.processAlive !== true;
    } catch { /* server offline */ }
  }

  update(s) {
    const jobs = s.jobs || [];
    this.$('j-n').textContent = `${s.metrics?.pending ?? 0} waiting · ${s.metrics?.inSystem ?? 0} in system`;
    this.$('j-body').innerHTML = jobs.slice(0, 18).map((j) =>
      `<tr><td>${j.hot ? '<span class="hot">●</span> ' : ''}${j.id}</td><td class="muted">${j.srcArea} → ${j.dstArea}</td><td><span class="badge b-${j.state}">${j.state}</span></td><td class="num">${f1(j.age, 0)}</td></tr>`).join('');
    const ds = s.control?.decisions || [];
    this.$('d-log').textContent = ds.slice().reverse().map((d) => {
      const what = d.kind === 'DISPATCH' ? `${d.subject} → OHT ${d.target ?? '—'}${d.busy ? ' (after its drop)' : ''}` : d.kind === 'ROUTE' ? `OHT ${d.subject} → edge ${d.target ?? 'default'}` : `OHT ${d.subject} → ${d.target ?? 'stay'}`;
      const eta = d.eta != null ? ` eta ${d.eta}s${d.bestEta != null && d.eta > d.bestEta + 0.05 ? ` (nearest ${d.bestEta}s)` : ''}` : '';
      return `${String(d.t).padStart(7)} ${d.kind.padEnd(10)} ${what}${eta} [${d.result}]`;
    }).join('\n') + (s.control?.defers ? `\n… ${s.control.defers} deferred dispatch decisions (not listed)` : '');
    this.$('e-log').textContent = (s.events || []).slice(-12).reverse().map((e) => `${e.time.toFixed(1).padStart(7)} ${e.type} ${e.jobId || e.tag || ''} ${e.vehicle != null ? 'OHT ' + e.vehicle : ''} ${e.duration ? e.duration + 's' : ''}`).join('\n');
  }
}

const KPIS = [
  ['throughputPerHour', 'Moves / h', (m) => f1(m.throughputPerHour, 0)],
  ['delivered', 'Delivered', (m) => m.delivered],
  ['avgDeliveryTime', 'Avg delivery s', (m) => f1(m.avgDeliveryTime)],
  ['p95DeliveryTime', 'P95 delivery s', (m) => f1(m.p95DeliveryTime)],
  ['avgWaitForVehicle', 'Wait for OHT s', (m) => f1(m.avgWaitForVehicle)],
  ['pending', 'Waiting jobs', (m) => m.pending],
  ['emptyTravelRatio', 'Empty travel', (m) => (m.emptyTravelRatio == null ? '–' : `${(m.emptyTravelRatio * 100).toFixed(0)}%`)],
  ['utilization', 'Utilization', (m) => (m.utilization == null ? '–' : `${(m.utilization * 100).toFixed(0)}%`)],
];

export class KpiBar {
  constructor(el) {
    el.innerHTML = KPIS.map(([k, l]) => `<div class="kpi"><div class="k">${l}</div><div class="v" id="k-${k}">–</div></div>`).join('');
    this.el = el;
  }

  update(s) {
    const m = s.metrics || {};
    for (const [k, , fn] of KPIS) this.el.querySelector('#k-' + k).textContent = fn(m);
  }
}
