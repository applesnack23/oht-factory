// 3D OHT fab: ceiling rails, hangers, tools, stockers, vehicles (instanced) with FOUPs.
// All logistics run on the Python server; the browser only interpolates vehicle poses.
import * as THREE from 'three';
import { CameraController } from './CameraController.js';
import { pointAt, samplePath, lerpAngle } from './pathUtils.js';

const PROCESS_COLOR = {
  LITHO: 0x7aa6ff, ETCH: 0xff9a6b, DEPO: 0x9be38a, CMP: 0xd9b3ff, DIFF: 0xffd36b,
  IMPLANT: 0x6bd6d6, METRO: 0xb8c4cf, CLEAN: 0xa0e0ff, STOCKER: 0x8892a0,
};
const PORT_TOP = 0.9, FOUP_H = 0.42;
const TOOL_IDLE = 0x5d6670, TOOL_BUSY = 0x3ddc84, TOOL_MOVE = 0x4aa3ff, TOOL_WAIT = 0xf2a93b;

// Hoist profile over the 0..1 hoist progress: lower (0-0.4), grip/release at the port (0.4-0.6), raise (0.6-1).
export function hoistDepth(p) {
  const ease = (x) => x * x * (3 - 2 * x);
  if (p <= 0) return 0;
  if (p < 0.4) return ease(p / 0.4);
  if (p <= 0.6) return 1;
  if (p < 1) return ease((1 - p) / 0.4);
  return 0;
}

export const STATE_COLOR = { IDLE: 0x9aa5b1, TO_PICKUP: 0x33b1ff, LOADING: 0xf5a524, TO_DROP: 0x3ecf8e, UNLOADING: 0xf5a524 };

function label(text, { bg = 'rgba(20,28,36,.85)', color = '#e6edf3', size = 1.0 } = {}) {
  const c = document.createElement('canvas');
  const g = c.getContext('2d');
  g.font = '600 44px -apple-system, Segoe UI, sans-serif';
  const w = Math.ceil(g.measureText(text).width) + 24;
  c.width = w; c.height = 62;
  g.font = '600 44px -apple-system, Segoe UI, sans-serif';
  g.fillStyle = bg; g.fillRect(0, 0, w, c.height);
  g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, 12, 33);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set((size * w) / c.height, size, 1);
  sp.renderOrder = 10;
  sp.userData.canvas = c;
  return sp;
}

function setLabel(sprite, text, bg) {
  const c = sprite.userData.canvas;
  const g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = bg; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = '#fff'; g.font = '600 44px -apple-system, Segoe UI, sans-serif'; g.textBaseline = 'middle';
  g.fillText(text, 12, 33);
  sprite.material.map.needsUpdate = true;
}

export class FabRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xc9ced3);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.5, 800);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x7d858c, 1.15));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(-40, 90, 50);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -90, right: 90, top: 80, bottom: -80, far: 250 });
    this.scene.add(sun);
    this.world = new THREE.Group();
    this.scene.add(this.world);
    this.cameraCtl = new CameraController(this.camera, canvas, new THREE.Vector3(52, 0, 0), 3.2);
    this.prev = new Map(); this.next = new Map(); this.tPrev = 0; this.tNext = 0;
    this.fps = 0; this._frames = 0; this._fpsT = performance.now();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.renderer.setAnimationLoop(() => this._loop());
  }

  resize() {
    const p = this.canvas.parentElement;
    if (!p.clientWidth || !p.clientHeight) return;
    this.renderer.setSize(p.clientWidth, p.clientHeight, false);
    this.camera.aspect = p.clientWidth / p.clientHeight;
    this.camera.updateProjectionMatrix();
  }

  buildLayout(L) {
    this.layout = L;
    this.world.clear();
    const H = L.railHeight || 4.5;
    this.H = H;
    const fx = L.floor?.sizeX || 130, fz = L.floor?.sizeZ || 110;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(fx, fz), new THREE.MeshStandardMaterial({ color: 0xc4c9ce, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(52, 0, 0);
    floor.receiveShadow = true;
    this.world.add(floor);
    const grid = new THREE.GridHelper(130, 26, 0xaab1b8, 0xb5bcc2);
    grid.position.set(52, 0.01, 0);
    this.world.add(grid);
    // rails: ribbon per edge + hangers
    const railMat = new THREE.MeshStandardMaterial({ color: 0x5b6672, metalness: 0.6, roughness: 0.35 });
    const xoMat = new THREE.MeshStandardMaterial({ color: 0x8c6fd1, metalness: 0.5, roughness: 0.4 });
    const hangerMat = new THREE.MeshStandardMaterial({ color: 0x7a838c });
    const box = new THREE.BoxGeometry(1, 1, 1);
    this.edgeMeshes = new Map();
    for (const e of L.track.edges) {
      const pts = samplePath(e.path, e.path.kind === 'ARC' ? 0.25 : e.length);
      const mat = e.tags.includes('XO') ? xoMat.clone() : railMat;
      const g = new THREE.Group();
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i];
        const len = Math.hypot(b.x - a.x, b.z - a.z);
        const m = new THREE.Mesh(box, mat);
        m.scale.set(len + 0.02, 0.12, 0.34);
        m.position.set((a.x + b.x) / 2, H + 0.45, (a.z + b.z) / 2);
        m.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x);
        g.add(m);
      }
      for (let s = 3; s < e.length; s += 6) {
        const p = pointAt(e.path, s);
        const h = new THREE.Mesh(box, hangerMat);
        h.scale.set(0.06, 1.6, 0.06);
        h.position.set(p.x, H + 1.3, p.z);
        g.add(h);
      }
      this.world.add(g);
      this.edgeMeshes.set(e.id, { group: g, mat, xo: e.tags.includes('XO') });
    }
    // tools + stockers
    const toolGeo = new THREE.BoxGeometry(1.8, 2.2, 2.2);
    const lpMat = new THREE.MeshStandardMaterial({ color: 0x39424b, roughness: 0.6 });
    const trayMat = new THREE.MeshStandardMaterial({ color: 0x59636d, roughness: 0.6 });
    this.portPos = new Map(L.ports.map((p) => [p.id, p]));
    this.toolViews = new Map();
    const doorMat = new THREE.MeshStandardMaterial({ color: 0x1c2126, roughness: 0.5 });
    const barBgMat = new THREE.MeshStandardMaterial({ color: 0x20252b });
    for (const p of L.ports) {
      if (p.kind === 'STOCKER') continue;
      const mat = new THREE.MeshStandardMaterial({ color: PROCESS_COLOR[p.process] || 0x999999, roughness: 0.7, emissive: 0x000000 });
      const m = new THREE.Mesh(toolGeo, mat);
      m.position.set(p.toolX, 1.1, p.toolZ);
      m.castShadow = true;
      this.world.add(m);
      const side = Math.sign(p.x - p.toolX);  // tool face points to the load port
      const faceX = p.toolX + side * 0.9;
      const door = new THREE.Mesh(box, doorMat);  // opening the FOUP slides through
      door.scale.set(0.03, 0.62, 0.62);
      door.position.set(faceX + side * 0.016, 1.15, p.z);
      this.world.add(door);
      const bg = new THREE.Mesh(box, barBgMat);  // progress bar background, above the door
      bg.scale.set(0.03, 0.16, 1.6);
      bg.position.set(faceX + side * 0.016, 1.85, p.z);
      this.world.add(bg);
      this.toolViews.set(p.id, { p, mat, faceX, side, idx: this.toolViews.size });
      const lp = new THREE.Mesh(box, lpMat);
      lp.scale.set(0.75, 0.9, 0.75);
      lp.position.set(p.x, 0.45, p.z);
      lp.castShadow = true;
      this.world.add(lp);
      const tray = new THREE.Mesh(box, trayMat);  // load port -> tool
      tray.scale.set(Math.abs(p.x - p.toolX), 0.08, 0.5);
      tray.position.set((p.x + p.toolX) / 2, 0.86, p.z);
      this.world.add(tray);
    }
    this.areaBadges = new Map();
    for (const a of L.areas) {
      const t = label(`${a.id} · ${a.process}`, { size: 1.6 });
      t.position.set(a.kind === 'STOCKER' ? a.cx + (a.side === 'west' ? -2 : 2) : a.cx, a.kind === 'STOCKER' ? 6.4 : 7.4, a.side === 'north' ? a.cz - 18 : a.side === 'south' ? a.cz + 18 : a.cz - 13);
      this.world.add(t);
      const badge = label('0 waiting', { size: 1.3, bg: 'rgba(60,70,80,.85)' });
      badge.position.set(t.position.x, t.position.y - 1.6, t.position.z);
      this.world.add(badge);
      this.areaBadges.set(a.id, badge);
    }
    this.stockerViews = new Map();
    for (const sk of L.stockers || []) this.stockerViews.set(sk.id, this._buildStocker(sk, box));
    for (const n of L.decisionNodes) {
      const node = L.track.nodes.find((x) => x.id === n);
      const m = new THREE.Mesh(new THREE.ConeGeometry(0.6, 1.2, 4), new THREE.MeshBasicMaterial({ color: 0xd94bd9 }));
      m.position.set(node.x, H + 1.7, node.z);
      m.rotation.x = Math.PI;
      this.world.add(m);
    }
    // vehicles (instanced): body + FOUP
    const N = 256;
    this.vBody = new THREE.InstancedMesh(new THREE.BoxGeometry(1.0, 0.75, 0.7), new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.2 }), N);
    this.vFoup = new THREE.InstancedMesh(new THREE.BoxGeometry(0.5, 0.42, 0.45), new THREE.MeshStandardMaterial({ color: 0xf0f2f4, roughness: 0.4 }), N);
    this.vHot = new THREE.InstancedMesh(new THREE.SphereGeometry(0.22, 10, 8), new THREE.MeshBasicMaterial({ color: 0xff3b47 }), N);
    // hoist: gripper + 4 belts per vehicle; FOUPs sitting on load ports
    this.vGrip = new THREE.InstancedMesh(new THREE.BoxGeometry(0.62, 0.12, 0.56), new THREE.MeshStandardMaterial({ color: 0x2b3138, metalness: 0.5, roughness: 0.4 }), N);
    this.vBelt = new THREE.InstancedMesh(new THREE.BoxGeometry(0.03, 1, 0.07), new THREE.MeshStandardMaterial({ color: 0x15191d }), N * 4);
    this.pFoup = new THREE.InstancedMesh(this.vFoup.geometry, this.vFoup.material, 256);
    this.sFoup = new THREE.InstancedMesh(this.vFoup.geometry, this.vFoup.material, 256);  // FOUPs sliding port <-> tool
    // dynamic instance sets: three.js computes an InstancedMesh's bounding sphere once (here, while it is
    // empty) and would then cull it forever, so these are never frustum-culled
    for (const m of [this.vBody, this.vFoup, this.vHot, this.vGrip, this.vBelt, this.pFoup, this.sFoup]) {
      m.count = 0; m.castShadow = true; m.frustumCulled = false; this.world.add(m);
    }
    const T = this.toolViews.size;
    this.tLamp = new THREE.InstancedMesh(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshBasicMaterial({ color: 0xffffff }), T);  // unlit: status light
    this.tBar = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x3ddc84 }), T);
    this.tBar.count = 0;
    this.tBar.frustumCulled = false;
    this.tLamp.frustumCulled = false;
    this.world.add(this.tLamp, this.tBar);
    const d0 = new THREE.Object3D();
    for (const tv of this.toolViews.values()) {
      d0.position.set(tv.p.toolX, 2.4, tv.p.toolZ); d0.updateMatrix();
      this.tLamp.setMatrixAt(tv.idx, d0.matrix);
      this.tLamp.setColorAt(tv.idx, new THREE.Color(TOOL_IDLE));
    }
    this.toolActive = new Map();
    this.toolMoves = [];
    this.dummy = new THREE.Object3D();
    this.cameraCtl.overview(true);
  }

  applyState(s) {
    const now = performance.now() / 1000;
    this.prev = this.next; this.tPrev = this.tNext;
    this.next = new Map((s.vehicles || []).map((v) => [v.id, v]));
    this.tNext = now;
    this.vehicles = s.vehicles || [];
    this.simPrev = this.simNext ?? s.time; this.simNext = s.time;
    if (s.tools && this.toolViews) {
      this.toolActive = new Map(s.tools.active.map((a) => [a[0], a]));
      this.toolMoves = s.tools.moves;
      this.toolSlide = s.tools.slide;
    }
    const blocked = new Set(s.blocked || []);
    if (this.pFoup && s.portFoups) {
      const d = this.dummy;
      let n = 0;
      for (const id of s.portFoups) {
        const p = this.portPos.get(id);
        if (!p || n >= 256) continue;
        d.position.set(p.x, PORT_TOP + FOUP_H / 2, p.z); d.rotation.set(0, 0, 0); d.scale.set(1, 1, 1); d.updateMatrix();
        this.pFoup.setMatrixAt(n++, d.matrix);
      }
      this.pFoup.count = n;
      this.pFoup.instanceMatrix.needsUpdate = true;
    }
    if (this.stockerViews && s.stockers) {
      this.skPrev = this.skNext || new Map();
      this.skNext = new Map(s.stockers.map((k) => [k.id, k]));
      for (const k of s.stockers) this._updateStockerCells(this.stockerViews.get(k.id), k.cells);
    }
    if (this.edgeMeshes) {
      for (const [id, em] of this.edgeMeshes) {
        if (!em.xo) continue;
        em.mat.color.setHex(blocked.has(id) ? 0xf2555a : 0x8c6fd1);
      }
    }
    if (this.areaBadges && s.areas) {
      for (const [id, b] of this.areaBadges) {
        const a = s.areas[id] || {};
        const p = a.pending || 0;
        setLabel(b, `${p} waiting · ${a.idle || 0} idle`, p >= 6 ? 'rgba(200,60,70,.9)' : p >= 3 ? 'rgba(210,140,30,.9)' : 'rgba(47,120,90,.88)');
      }
    }
  }

  _buildStocker(sk, box) {
    const g = new THREE.Group();
    const X = (u) => sk.laneX + sk.dir * u;
    const zEnd = sk.bayZ[sk.bayZ.length - 1] + sk.bayPitch / 2 + 0.3;
    const top = 4.0;
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x6d7782, metalness: 0.5, roughness: 0.45 });
    const shelfMat = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, metalness: 0.3, roughness: 0.6 });
    const glass = new THREE.MeshStandardMaterial({ color: 0xbfd6e6, transparent: true, opacity: 0.16, depthWrite: false, roughness: 0.1 });
    const add = (mat, sx, sy, sz, x, y, z, shadow = false) => {
      const m = new THREE.Mesh(box, mat);
      m.scale.set(sx, sy, sz); m.position.set(x, y, z); m.castShadow = shadow; g.add(m); return m;
    };
    const w = sk.backU - sk.wallU, cu = (sk.wallU + sk.backU) / 2;
    // see-through enclosure: walls + roof, with a frame
    add(glass, 0.04, top, 2 * zEnd, X(sk.wallU), top / 2, 0);
    add(glass, 0.04, top, 2 * zEnd, X(sk.backU), top / 2, 0);
    add(glass, w, top, 0.04, X(cu), top / 2, zEnd);
    add(glass, w, top, 0.04, X(cu), top / 2, -zEnd);
    add(glass, w, 0.04, 2 * zEnd, X(cu), top, 0);
    for (const u of [sk.wallU, sk.backU]) for (const z of [-zEnd, zEnd]) add(frameMat, 0.12, top, 0.12, X(u), top / 2, z, true);
    for (const u of [sk.wallU, sk.backU]) add(frameMat, 0.1, 0.1, 2 * zEnd, X(u), top, 0);
    add(new THREE.MeshStandardMaterial({ color: 0x4f5964, roughness: 0.9 }), w, 0.04, 2 * zEnd, X(cu), 0.02, 0);
    // racks: uprights at bay boundaries + shelf plates per level
    const rackLen = sk.bayZ.length * sk.bayPitch;
    for (const [ru, rack] of [[sk.nearU, 0], [sk.farU, 1]]) {
      for (let b = 0; b <= sk.bayZ.length; b++) {
        const z = sk.bayZ[0] - sk.bayPitch / 2 + b * sk.bayPitch;
        for (const du of [-0.42, 0.42]) add(frameMat, 0.06, top - 0.2, 0.06, X(ru + du), (top - 0.2) / 2, z);
      }
      sk.levels.forEach((y, l) => {
        if (rack === 0 && l === 0) return;  // I/O level on the lane side
        add(shelfMat, 0.9, 0.04, rackLen, X(ru), y - 0.02, 0);
      });
    }
    // I/O ports: a shelf under the OHT rail + a short conveyor into the stocker
    const lpMat = new THREE.MeshStandardMaterial({ color: 0x39424b, roughness: 0.6 });
    const convMat = new THREE.MeshStandardMaterial({ color: 0x3f8f6b, roughness: 0.5 });
    for (const p of sk.ports) {
      add(lpMat, 0.75, PORT_TOP, 0.75, X(0), PORT_TOP / 2, p.z, true);
      add(convMat, sk.nearU + 0.5, 0.06, 0.62, X((sk.nearU + 0.5) / 2), PORT_TOP - 0.03, p.z);
    }
    // stacker crane: floor + ceiling rail, mast, carriage, telescopic fork, carried FOUP
    add(frameMat, 0.16, 0.1, rackLen + 0.6, X(sk.aisleU), 0.05, 0);
    add(frameMat, 0.16, 0.1, rackLen + 0.6, X(sk.aisleU), top - 0.15, 0);
    const craneMat = new THREE.MeshStandardMaterial({ color: 0xffc21a, emissive: 0x3a2a00, metalness: 0.2, roughness: 0.45 });
    const crane = new THREE.Group();
    const mast = new THREE.Mesh(box, craneMat); mast.scale.set(0.3, top - 0.25, 0.36); mast.position.set(X(sk.aisleU), (top - 0.25) / 2, 0); mast.castShadow = true;
    const base = new THREE.Mesh(box, craneMat); base.scale.set(0.5, 0.25, 1.1); base.position.set(X(sk.aisleU), 0.2, 0);
    const head = new THREE.Mesh(box, craneMat); head.scale.set(0.4, 0.2, 1.0); head.position.set(X(sk.aisleU), top - 0.2, 0);
    crane.add(mast, base, head);
    const carriage = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0x2f3a45, metalness: 0.4 }));
    carriage.scale.set(0.9, 0.22, 0.8);
    const fork = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0xe0572b, roughness: 0.4 }));
    const cFoup = new THREE.Mesh(this.vFoup ? this.vFoup.geometry : new THREE.BoxGeometry(0.5, FOUP_H, 0.45), new THREE.MeshStandardMaterial({ color: 0xf0f2f4, roughness: 0.4 }));
    cFoup.castShadow = true;
    crane.add(carriage, fork, cFoup);
    g.add(crane);
    // stored FOUPs
    const stored = new THREE.InstancedMesh(new THREE.BoxGeometry(0.5, FOUP_H, 0.45), new THREE.MeshStandardMaterial({ color: 0xe9edf1, roughness: 0.45 }), sk.cells.length);
    stored.count = 0;
    stored.frustumCulled = false;
    g.add(stored);
    const t = label(`${sk.id} crane · 00 tasks`, { size: 0.8, bg: 'rgba(160,110,20,.85)' });
    t.position.set(X(sk.aisleU), top + 0.8, -zEnd - 1.2);
    g.add(t);
    this.world.add(g);
    return { sk, X, crane, carriage, fork, cFoup, stored, cells: '', label: t };
  }

  _simNow(now) {
    const span = this.tNext - this.tPrev;
    const t = span > 0 ? Math.min(1, Math.max(0, (now - this.tNext) / span)) : 1;
    return (this.simPrev ?? 0) + ((this.simNext ?? 0) - (this.simPrev ?? 0)) * t;
  }

  _animateTools(now) {
    const sim = this._simNow(now);
    const d = this.dummy, col = new THREE.Color();
    const pulse = 0.55 + 0.45 * Math.sin(now * 4);
    let nb = 0;
    for (const tv of this.toolViews.values()) {
      const a = this.toolActive.get(tv.p.id);
      const busy = a && a[2] > sim && a[1] <= sim;
      const moving = this.toolMoves.some((m) => m[0] === tv.p.id && sim >= m[1] && sim < m[1] + this.toolSlide);
      const c = busy ? TOOL_BUSY : moving ? TOOL_MOVE : a && a[3] > 0 ? TOOL_WAIT : TOOL_IDLE;
      if (tv._c !== c || busy) {
        tv._c = c;
        this.tLamp.setColorAt(tv.idx, col.setHex(c).multiplyScalar(busy ? pulse + 0.3 : 1));
        this.tLamp.instanceColor.needsUpdate = true;
      }
      tv.mat.emissive.setHex(busy ? PROCESS_COLOR[tv.p.process] || 0x333333 : 0x000000);
      tv.mat.emissiveIntensity = busy ? 0.18 + 0.12 * pulse : 0;
      if (busy) {
        const f = Math.min(1, (sim - a[1]) / (a[2] - a[1]));
        const len = 1.56 * f;
        d.position.set(tv.faceX + tv.side * 0.035, 1.85, tv.p.z - 0.78 + len / 2);
        d.rotation.set(0, 0, 0); d.scale.set(0.03, 0.12, Math.max(0.001, len)); d.updateMatrix();
        this.tBar.setMatrixAt(nb++, d.matrix);
      }
    }
    this.tBar.count = nb;
    this.tBar.instanceMatrix.needsUpdate = true;
    // FOUPs sliding along the tray: dir 1 = port -> tool (delivered lot), -1 = tool -> port (lot ready to ship)
    let ns = 0;
    d.scale.set(1, 1, 1); d.rotation.set(0, 0, 0);
    for (const [pid, start, dir] of this.toolMoves) {
      const tv = this.toolViews.get(pid);
      if (!tv || ns >= 256) continue;
      let f = Math.min(1, Math.max(0, (sim - start) / this.toolSlide));
      if (dir < 0) f = 1 - f;
      const ease = f * f * (3 - 2 * f);
      const inside = tv.faceX - tv.side * 0.45;  // just inside the tool body: hidden by it
      const x = tv.p.x + (inside - tv.p.x) * ease;
      d.position.set(x, PORT_TOP + FOUP_H / 2, tv.p.z); d.updateMatrix();
      this.sFoup.setMatrixAt(ns++, d.matrix);
    }
    this.sFoup.count = ns;
    this.sFoup.instanceMatrix.needsUpdate = true;
  }

  viewTool() {
    // a tool that is taking a FOUP in or pushing one out right now
    const sim = this._simNow(performance.now() / 1000);
    // prefer a FOUP that is about to start sliding, so the camera arrives in time to see it
    const moves = (this.toolMoves || []).filter((q) => sim < q[1] + this.toolSlide).sort((a, b) => b[1] - a[1]);
    const m = moves[0];
    const tv = m ? this.toolViews.get(m[0]) : [...this.toolViews.values()].find((t) => this.toolActive.get(t.p.id));
    if (!tv) return false;
    const { p, side } = tv;
    this.cameraCtl.look([p.x + side * 2.2, 2.9, p.z + 3.6], [(p.x + p.toolX) / 2, 1.1, p.z]);
    return true;
  }

  viewStocker(id) {
    const v = this.stockerViews?.get(id);
    if (!v) return;
    const { sk, X } = v;
    // from above the OHT lane side, looking down through the glass roof into the crane aisle
    const zEnd = sk.bayZ[sk.bayZ.length - 1] + sk.bayPitch;
    // from the end of the stocker, slightly above the roof, looking down the crane aisle
    this.cameraCtl.look([X(sk.aisleU - 0.6), 7.5, zEnd + 5], [X(sk.aisleU), 0.8, zEnd - 9]);
  }

  viewHoist() {
    const vs = (this.vehicles || []).filter((v) => (v.state === 'LOADING' || v.state === 'UNLOADING') && v.hoist < 0.3);
    if (!vs.length) return false;
    const v = vs[0];
    this.cameraCtl.look([v.x + 2.5, 3.0, v.z + 6], [v.x, 2.4, v.z]);
    return true;
  }

  _updateStockerCells(view, cells) {
    if (!view || view.cells === cells) return;
    view.cells = cells;
    const d = this.dummy || new THREE.Object3D();
    let n = 0;
    for (let i = 0; i < cells.length; i++) {
      if (cells[i] !== '1') continue;
      const [u, z, y] = view.sk.cells[i];
      d.position.set(view.X(u), y + FOUP_H / 2, z); d.rotation.set(0, 0, 0); d.scale.set(1, 1, 1); d.updateMatrix();
      view.stored.setMatrixAt(n++, d.matrix);
    }
    view.stored.count = n;
    view.stored.instanceMatrix.needsUpdate = true;
  }

  _animateStockers(now) {
    const span = this.tNext - this.tPrev;
    const t = span > 0 ? Math.min(1, Math.max(0, (now - this.tNext) / span)) : 1;
    for (const [id, view] of this.stockerViews) {
      const b = this.skNext.get(id);
      if (!b) continue;
      const a = this.skPrev.get(id) || b;
      const L = (k) => a[k] + (b[k] - a[k]) * t;
      const z = L('z'), y = L('y'), fu = L('fork');
      const { sk, X } = view;
      view.crane.position.z = z;
      view.carriage.position.set(X(sk.aisleU), y - 0.09, 0);
      const u0 = Math.min(sk.aisleU, fu), u1 = Math.max(sk.aisleU, fu);
      view.fork.scale.set(Math.max(0.5, u1 - u0 + 0.3), 0.05, 0.42);
      view.fork.position.set(X((u0 + u1) / 2), y - 0.02, 0);
      view.cFoup.visible = !!b.carry;
      view.cFoup.position.set(X(fu), y + FOUP_H / 2, 0);
      if (view._q !== b.queue) {
        view._q = b.queue;
        setLabel(view.label, `${id} crane · ${b.queue ? b.queue + ' task' + (b.queue > 1 ? 's' : '') : 'idle'}`, 'rgba(160,110,20,.85)');
      }
    }
  }

  _pose(v, now) {
    const a = this.prev.get(v.id);
    const span = this.tNext - this.tPrev;
    const t = span > 0 ? Math.min(1, Math.max(0, (now - this.tNext) / span)) : 1;
    const hp1 = v.hoist || 0;
    if (!a || Math.hypot(a.x - v.x, a.z - v.z) > 12) return { x: v.x, z: v.z, h: v.heading, hp: hp1 };
    const hp = a.state === v.state ? (a.hoist || 0) + (hp1 - (a.hoist || 0)) * t : hp1;
    return { x: a.x + (v.x - a.x) * t, z: a.z + (v.z - a.z) * t, h: lerpAngle(a.heading, v.heading, t), hp };
  }

  _loop() {
    const now = performance.now() / 1000;
    if (this.vBody && this.vehicles) {
      const vs = this.vehicles;
      const d = this.dummy;
      const col = new THREE.Color();
      let nf = 0, nh = 0, nb = 0;
      const full = this.H - 0.6 - (PORT_TOP + FOUP_H / 2);  // FOUP travel from the vehicle down to the port
      const hb = this.H - 0.375;  // vehicle body bottom
      for (let i = 0; i < vs.length; i++) {
        const v = vs[i];
        const p = this._pose(v, now);
        d.position.set(p.x, this.H, p.z); d.rotation.set(0, -p.h, 0); d.scale.set(1, 1, 1); d.updateMatrix();
        this.vBody.setMatrixAt(i, d.matrix);
        this.vBody.setColorAt(i, col.setHex(STATE_COLOR[v.state] ?? 0x999999));
        const hoisting = v.state === 'LOADING' || v.state === 'UNLOADING';
        const depth = hoisting ? full * hoistDepth(p.hp) : 0;
        const gy = this.H - 0.6 + FOUP_H / 2 + 0.06 - depth;  // gripper sits on top of the FOUP
        d.position.set(p.x, gy, p.z); d.updateMatrix();
        this.vGrip.setMatrixAt(i, d.matrix);
        const blen = hb - (gy + 0.06);
        if (blen > 0.03) {
          for (const [ox, oz] of [[-0.24, -0.2], [0.24, -0.2], [-0.24, 0.2], [0.24, 0.2]]) {
            const c = Math.cos(p.h), sn = Math.sin(p.h);
            d.position.set(p.x + ox * c - oz * sn, gy + 0.06 + blen / 2, p.z + ox * sn + oz * c);
            d.scale.set(1, blen, 1); d.updateMatrix();
            this.vBelt.setMatrixAt(nb++, d.matrix);
          }
          d.scale.set(1, 1, 1);
        }
        // The FOUP being hoisted is drawn here (the server leaves it out of portFoups), using the same
        // interpolated hoist progress as the gripper, so it never jumps between port and vehicle.
        if (v.state === 'TO_DROP') {
          d.position.set(p.x, this.H - 0.6, p.z); d.updateMatrix();
          this.vFoup.setMatrixAt(nf++, d.matrix);
        } else if (hoisting && (v.state === 'UNLOADING' || v.foupReady !== false)) {
          const onGripper = v.state === 'LOADING' ? p.hp >= 0.5 : p.hp < 0.5;
          // on the port: pinned to the port (the latest state's position), even while the drawn vehicle
          // is still finishing its last interpolated metres
          if (onGripper) d.position.set(p.x, this.H - 0.6 - depth, p.z);
          else d.position.set(v.x, PORT_TOP + FOUP_H / 2, v.z);
          d.updateMatrix();
          this.vFoup.setMatrixAt(nf++, d.matrix);
        }
        if (v.hot) {
          d.position.set(p.x, this.H + 0.6, p.z); d.updateMatrix();
          this.vHot.setMatrixAt(nh++, d.matrix);
        }
      }
      this.vBody.count = vs.length; this.vFoup.count = nf; this.vHot.count = nh;
      this.vGrip.count = vs.length; this.vBelt.count = nb;
      for (const m of [this.vBody, this.vFoup, this.vHot, this.vGrip, this.vBelt]) {
        m.instanceMatrix.needsUpdate = true;
        if (m.instanceColor) m.instanceColor.needsUpdate = true;
      }
    }
    if (this.stockerViews && this.skNext) this._animateStockers(now);
    if (this.toolViews && this.tLamp) this._animateTools(now);
    this.cameraCtl.update(1 / 60);
    this.renderer.render(this.scene, this.camera);
    this._frames++;
    const t = performance.now();
    if (t - this._fpsT > 1000) { this.fps = (this._frames * 1000) / (t - this._fpsT); this._frames = 0; this._fpsT = t; }
  }
}
