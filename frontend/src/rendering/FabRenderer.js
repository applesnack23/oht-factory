// 3D OHT fab: ceiling rails, hangers, tools, stockers, vehicles (instanced) with FOUPs.
// All logistics run on the Python server; the browser only interpolates vehicle poses.
import * as THREE from 'three';
import { CameraController } from './CameraController.js';
import { pointAt, samplePath, lerpAngle } from './pathUtils.js';

const PROCESS_COLOR = {
  LITHO: 0x7aa6ff, ETCH: 0xff9a6b, DEPO: 0x9be38a, CMP: 0xd9b3ff, DIFF: 0xffd36b,
  IMPLANT: 0x6bd6d6, METRO: 0xb8c4cf, CLEAN: 0xa0e0ff, STOCKER: 0x8892a0,
};
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
    for (const p of L.ports) {
      if (p.kind === 'STOCKER') continue;
      const m = new THREE.Mesh(toolGeo, new THREE.MeshStandardMaterial({ color: PROCESS_COLOR[p.process] || 0x999999, roughness: 0.7 }));
      m.position.set(p.toolX, 1.1, p.toolZ);
      m.castShadow = true;
      this.world.add(m);
      const lp = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0x39424b }));
      lp.scale.set(0.7, 0.9, 0.7);
      lp.position.set((p.x + p.toolX) / 2 + (p.x - p.toolX) * 0.15, 0.45, p.z);
      this.world.add(lp);
    }
    this.areaBadges = new Map();
    for (const a of L.areas) {
      if (a.kind === 'STOCKER') {
        const m = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: PROCESS_COLOR.STOCKER, roughness: 0.6 }));
        m.scale.set(5, 4.2, 14);
        m.position.set(a.cx, 2.1, a.cz);
        m.castShadow = true;
        this.world.add(m);
      }
      const t = label(`${a.id} · ${a.process}`, { size: 1.6 });
      t.position.set(a.cx, a.kind === 'STOCKER' ? 5.8 : 7.4, a.side === 'north' ? a.cz - 18 : a.side === 'south' ? a.cz + 18 : a.cz - 9);
      this.world.add(t);
      const badge = label('0 waiting', { size: 1.3, bg: 'rgba(60,70,80,.85)' });
      badge.position.set(t.position.x, t.position.y - 1.6, t.position.z);
      this.world.add(badge);
      this.areaBadges.set(a.id, badge);
    }
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
    for (const m of [this.vBody, this.vFoup, this.vHot]) { m.count = 0; m.castShadow = true; this.world.add(m); }
    this.dummy = new THREE.Object3D();
    this.cameraCtl.overview(true);
  }

  applyState(s) {
    const now = performance.now() / 1000;
    this.prev = this.next; this.tPrev = this.tNext;
    this.next = new Map((s.vehicles || []).map((v) => [v.id, v]));
    this.tNext = now;
    this.vehicles = s.vehicles || [];
    const blocked = new Set(s.blocked || []);
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

  _pose(v, now) {
    const a = this.prev.get(v.id);
    const span = this.tNext - this.tPrev;
    const t = span > 0 ? Math.min(1, Math.max(0, (now - this.tNext) / span)) : 1;
    if (!a || Math.hypot(a.x - v.x, a.z - v.z) > 12) return { x: v.x, z: v.z, h: v.heading };
    return { x: a.x + (v.x - a.x) * t, z: a.z + (v.z - a.z) * t, h: lerpAngle(a.heading, v.heading, t) };
  }

  _loop() {
    const now = performance.now() / 1000;
    if (this.vBody && this.vehicles) {
      const vs = this.vehicles;
      const d = this.dummy;
      const col = new THREE.Color();
      let nf = 0, nh = 0;
      for (let i = 0; i < vs.length; i++) {
        const v = vs[i];
        const p = this._pose(v, now);
        d.position.set(p.x, this.H, p.z); d.rotation.set(0, -p.h, 0); d.scale.set(1, 1, 1); d.updateMatrix();
        this.vBody.setMatrixAt(i, d.matrix);
        this.vBody.setColorAt(i, col.setHex(STATE_COLOR[v.state] ?? 0x999999));
        if (v.loaded) {
          d.position.set(p.x, this.H - 0.6, p.z); d.updateMatrix();
          this.vFoup.setMatrixAt(nf++, d.matrix);
        }
        if (v.hot) {
          d.position.set(p.x, this.H + 0.6, p.z); d.updateMatrix();
          this.vHot.setMatrixAt(nh++, d.matrix);
        }
      }
      this.vBody.count = vs.length; this.vFoup.count = nf; this.vHot.count = nh;
      for (const m of [this.vBody, this.vFoup, this.vHot]) {
        m.instanceMatrix.needsUpdate = true;
        if (m.instanceColor) m.instanceColor.needsUpdate = true;
      }
    }
    this.cameraCtl.update(1 / 60);
    this.renderer.render(this.scene, this.camera);
    this._frames++;
    const t = performance.now();
    if (t - this._fpsT > 1000) { this.fps = (this._frames * 1000) / (t - this._fpsT); this._frames = 0; this._fpsT = t; }
  }
}
