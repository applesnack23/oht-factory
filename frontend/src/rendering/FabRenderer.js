// 3D OHT fab: ceiling rails, hangers, tools, stockers, vehicles (instanced) with FOUPs.
// All logistics run on the Python server; the browser only interpolates vehicle poses.
import * as THREE from 'three';
import { TOWER, TOWER_OFF, towerSegments } from './towerLamp.js';
import { FOUP, GRIP, foupGeometry, plateGeometry, fingerGeometry, footGeometry, gripClosed } from './models.js';
import { CameraController } from './CameraController.js';
import { pointAt, samplePath, lerpAngle } from './pathUtils.js';

const PROCESS_COLOR = {
  LITHO: 0x7aa6ff, ETCH: 0xff9a6b, DEPO: 0x9be38a, CMP: 0xd9b3ff, DIFF: 0xffd36b,
  IMPLANT: 0x6bd6d6, METRO: 0xb8c4cf, CLEAN: 0xa0e0ff, STOCKER: 0x8892a0,
};
const PORT_TOP = 0.9, FOUP_H = 0.42;
// load-port status light: E empty, V lot on its way, R/D/U docking, Q docked waiting, P processing, W ready for OHT
const LP_COLOR = { E: 0x5d6670, V: 0xe8ecf0, R: 0x3fd0e0, D: 0x3fd0e0, U: 0x3fd0e0, Q: 0xff9a1f, P: 0x2f7bff, W: 0x29d65a };

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
    // tools: process module + EFEM with its load ports (one LP per bay port, under the OHT rail)
    this.portPos = new Map(L.ports.map((p) => [p.id, p]));
    this.dockOffset = L.dockOffset ?? 0.14;
    this._buildTools(L, box);
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
    this.foupGeo = foupGeometry();
    this.foupMat = new THREE.MeshStandardMaterial({ color: 0xf0f2f4, roughness: 0.4 });
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
    this.vFoup = new THREE.InstancedMesh(this.foupGeo, this.foupMat, N);
    this.vHot = new THREE.InstancedMesh(new THREE.SphereGeometry(0.22, 10, 8), new THREE.MeshBasicMaterial({ color: 0xff3b47 }), N);
    // hoist: gripper (plate + 2 fingers with feet that close under the FOUP flange) + 4 belts per vehicle
    const gripMat = new THREE.MeshStandardMaterial({ color: 0x2b3138, metalness: 0.5, roughness: 0.4 });
    this.vGrip = new THREE.InstancedMesh(plateGeometry(), gripMat, N);
    this.vFinger = new THREE.InstancedMesh(fingerGeometry(), new THREE.MeshStandardMaterial({ color: 0x59626b, metalness: 0.6, roughness: 0.35 }), N * 2);
    this.vFoot = new THREE.InstancedMesh(footGeometry(), new THREE.MeshStandardMaterial({ color: 0x59626b, metalness: 0.6, roughness: 0.35 }), N * 2);
    this.vBelt = new THREE.InstancedMesh(new THREE.BoxGeometry(0.03, 1, 0.07), new THREE.MeshStandardMaterial({ color: 0x15191d }), N * 4);
    this.pFoup = new THREE.InstancedMesh(this.vFoup.geometry, this.vFoup.material, 256);
    // dynamic instance sets: three.js computes an InstancedMesh's bounding sphere once (here, while it is
    // empty) and would then cull it forever, so these are never frustum-culled
    for (const m of [this.vBody, this.vFoup, this.vHot, this.vGrip, this.vFinger, this.vFoot, this.vBelt, this.pFoup]) {
      m.count = 0; m.castShadow = true; m.frustumCulled = false; this.world.add(m);
    }
    // load-port FOUPs, LP status lights, process progress bars (one per LP)
    const nLP = this.lpViews.size;
    this.lpFoup = new THREE.InstancedMesh(this.vFoup.geometry, this.vFoup.material, nLP);
    this.lpLamp = new THREE.InstancedMesh(new THREE.BoxGeometry(0.1, 0.1, 0.22), new THREE.MeshBasicMaterial({ color: 0xffffff }), nLP);
    this.lpBar = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x2f7bff }), nLP);
    this.tLamp = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.085, 0.085, 0.15, 14), new THREE.MeshBasicMaterial({ color: 0xffffff }), this.toolViews.size * 4);
    for (const m of [this.lpFoup, this.lpLamp, this.lpBar, this.tLamp]) { m.frustumCulled = false; this.world.add(m); }
    this.lpFoup.castShadow = true;
    this.lpFoup.count = 0; this.lpBar.count = 0;
    const d0 = new THREE.Object3D(), c0 = new THREE.Color();
    for (const lv of this.lpViews.values()) {
      d0.position.set(lv.plateX + lv.side * 0.05, 1.63, lv.p.z); d0.updateMatrix();
      this.lpLamp.setMatrixAt(lv.idx, d0.matrix);
      this.lpLamp.setColorAt(lv.idx, c0.setHex(LP_COLOR.E));
    }
    for (const tv of this.toolViews.values()) {
      for (let k = 0; k < 4; k++) {  // stacked segments on top of the tower pole
        d0.position.set(tv.towerX, tv.towerY + 0.155 * (3 - k) + 0.075, tv.towerZ); d0.updateMatrix();
        this.tLamp.setMatrixAt(tv.idx * 4 + k, d0.matrix);
        this.tLamp.setColorAt(tv.idx * 4 + k, c0.setHex(TOWER_OFF).lerp(new THREE.Color(TOWER[k]), 0.14));
      }
    }
    this.lpState = new Map();
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
    if (s.tools && this.lpViews) {
      this.lpState = new Map(s.tools.lps.map((a) => [a[0], a]));
      this.toolModes = s.tools.modes || {};
      this.dockT = s.tools.dock;
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
    // racks: uprights at bay boundaries, a back beam per level, and per cell two shelf pads with a groove
    // between them: the crane's fork enters the groove below the FOUP and lifts it off the pads
    const rackLen = sk.bayZ.length * sk.bayPitch;
    const padMat = new THREE.MeshStandardMaterial({ color: 0xa9b2bb, metalness: 0.35, roughness: 0.5 });
    const padGeo = new THREE.BoxGeometry(0.8, 0.03, 0.17);
    const nPads = sk.cells.length * 2;
    const pads = new THREE.InstancedMesh(padGeo, padMat, nPads);
    const dd = new THREE.Object3D();
    let np = 0;
    for (const [u, z, y] of sk.cells) {
      for (const dz of [-0.2, 0.2]) {
        dd.position.set(X(u), y - 0.015, z + dz); dd.updateMatrix();
        pads.setMatrixAt(np++, dd.matrix);
      }
    }
    pads.count = np;
    g.add(pads);
    for (const [ru, rack] of [[sk.nearU, 0], [sk.farU, 1]]) {
      const backU = rack === 0 ? ru - 0.42 : ru + 0.42;  // the rack's outer edge (away from the aisle)
      for (let b = 0; b <= sk.bayZ.length; b++) {
        const z = sk.bayZ[0] - sk.bayPitch / 2 + b * sk.bayPitch;
        for (const du of [-0.42, 0.42]) add(frameMat, 0.06, top - 0.2, 0.06, X(ru + du), (top - 0.2) / 2, z);
      }
      sk.levels.forEach((y, l) => {
        if (rack === 0 && l === 0) return;  // lane-side bottom level: port conveyors
        add(frameMat, 0.05, 0.05, rackLen, X(backU), y - 0.045, 0);
      });
    }
    // in/out ports: stage under the OHT rail + conveyor (two belts with a groove) into the stocker.
    // Input ports (OHT drops) have green belts, output ports (OHT picks up) blue belts, plus a sign.
    const beltIn = new THREE.MeshStandardMaterial({ color: 0x2f7a55, roughness: 0.6 });
    const beltOut = new THREE.MeshStandardMaterial({ color: 0x2f5f9a, roughness: 0.6 });
    const convFrame = new THREE.MeshStandardMaterial({ color: 0x4a535c, roughness: 0.5 });
    const u0 = -0.38, u1 = sk.nearU + 0.42, cl = u1 - u0, cm = (u0 + u1) / 2;
    for (const p of sk.ports) {
      const isIn = p.io !== 'OUT';
      const sign = label(isIn ? 'IN' : 'OUT', { size: 0.2, bg: isIn ? 'rgba(40,140,90,.92)' : 'rgba(40,100,180,.92)' });
      sign.position.set(X(u0 - 0.03), PORT_TOP + 0.42, p.z);
      g.add(sign);
      for (const dz of [-0.2, 0.2]) {
        add(isIn ? beltIn : beltOut, cl, 0.04, 0.15, X(cm), PORT_TOP - 0.02, p.z + dz);
        add(convFrame, cl, 0.08, 0.04, X(cm), PORT_TOP - 0.06, p.z + dz + Math.sign(dz) * 0.1);
      }
      for (const u of [u0 + 0.1, u1 - 0.1]) add(convFrame, 0.08, PORT_TOP - 0.1, 0.5, X(u), (PORT_TOP - 0.1) / 2, p.z, true);
      add(convFrame, 0.06, 0.25, 0.62, X(u0 - 0.03), PORT_TOP + 0.08, p.z);  // end stop at the stage
    }
    // stacker crane: floor + ceiling rail, mast, carriage with a turntable, single narrow fork
    add(frameMat, 0.16, 0.1, rackLen + 0.6, X(sk.aisleU), 0.05, 0);
    add(frameMat, 0.16, 0.1, rackLen + 0.6, X(sk.aisleU), top - 0.15, 0);
    const craneMat = new THREE.MeshStandardMaterial({ color: 0xffc21a, emissive: 0x3a2a00, metalness: 0.2, roughness: 0.45 });
    const crane = new THREE.Group();
    const mastZ = -0.45;  // mast beside the carriage so the fork can turn freely
    const mast = new THREE.Mesh(box, craneMat); mast.scale.set(0.3, top - 0.25, 0.22); mast.position.set(X(sk.aisleU), (top - 0.25) / 2, mastZ); mast.castShadow = true;
    const base = new THREE.Mesh(box, craneMat); base.scale.set(0.5, 0.25, 1.3); base.position.set(X(sk.aisleU), 0.2, 0);
    const head = new THREE.Mesh(box, craneMat); head.scale.set(0.4, 0.2, 1.1); head.position.set(X(sk.aisleU), top - 0.2, 0);
    crane.add(mast, base, head);
    const lift = new THREE.Group();  // moves up/down with the fork top at y = 0
    const carriage = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0x2f3a45, metalness: 0.4 }));
    carriage.scale.set(0.7, 0.12, 0.95); carriage.position.set(X(sk.aisleU), -0.17, -0.05);
    const table = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.05, 28), new THREE.MeshStandardMaterial({ color: 0x59626b, metalness: 0.6, roughness: 0.35 }));
    table.position.set(X(sk.aisleU), -0.085, 0);
    const turn = new THREE.Group();  // rotates on the turntable; local +x = fork direction
    turn.position.set(X(sk.aisleU), 0, 0);
    const forkBase = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0x3a434c, metalness: 0.5 }));
    forkBase.scale.set(0.62, 0.05, 0.3); forkBase.position.set(0, -0.045, 0);
    const tine = new THREE.Mesh(box, new THREE.MeshStandardMaterial({ color: 0xe0572b, roughness: 0.4 }));
    tine.castShadow = true;
    const cFoup = new THREE.Mesh(this.foupGeo, this.foupMat);
    cFoup.castShadow = true;
    turn.add(forkBase, tine, cFoup);
    lift.add(carriage, table, turn);
    crane.add(lift);
    g.add(crane);
    // stored FOUPs + FOUPs on the port conveyors
    const stored = new THREE.InstancedMesh(this.foupGeo, this.foupMat, sk.cells.length);
    stored.count = 0;
    stored.frustumCulled = false;
    stored.castShadow = true;
    g.add(stored);
    const conv = new THREE.InstancedMesh(this.foupGeo, this.foupMat, 32);
    conv.count = 0;
    conv.frustumCulled = false;
    conv.castShadow = true;
    g.add(conv);
    const t = label(`${sk.id} crane · 00 tasks`, { size: 0.8, bg: 'rgba(160,110,20,.85)' });
    t.position.set(X(sk.aisleU), top + 0.8, -zEnd - 1.2);
    g.add(t);
    this.world.add(g);
    return { sk, X, crane, lift, turn, tine, cFoup, stored, conv, cells: '', label: t };
  }

  _simNow(now) {
    const span = this.tNext - this.tPrev;
    const t = span > 0 ? Math.min(1, Math.max(0, (now - this.tNext) / span)) : 1;
    return (this.simPrev ?? 0) + ((this.simNext ?? 0) - (this.simPrev ?? 0)) * t;
  }

  _buildTools(L, box) {
    // Geometry per tool (x measured from the rail toward the tool, `side` = direction tool -> rail):
    //   0 .. 0.45  load port stage (FOUP sits here, under the OHT)
    //   0.45       EFEM front with one load-port plate + door per LP
    //   0.45..1.45 EFEM (wafer handling robot inside), light grey
    //   1.45..3.35 process module, process colour
    this.toolViews = new Map();
    this.lpViews = new Map();
    const efemMat = new THREE.MeshStandardMaterial({ color: 0xdfe3e7, roughness: 0.55 });
    const winMat = new THREE.MeshStandardMaterial({ color: 0x8fb4cc, roughness: 0.15, metalness: 0.2, transparent: true, opacity: 0.75 });
    const plateMat = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.5 });
    const doorMat = new THREE.MeshStandardMaterial({ color: 0x2a3036, roughness: 0.5 });
    const stageMat = new THREE.MeshStandardMaterial({ color: 0x4a535c, roughness: 0.5 });
    const barBgMat = new THREE.MeshStandardMaterial({ color: 0x20252b });
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x8a939c, metalness: 0.5, roughness: 0.4 });
    const capMat = new THREE.MeshStandardMaterial({ color: 0x2b3036, roughness: 0.5 });
    const add = (mat, sx, sy, sz, x, y, z, shadow = true) => {
      const m = new THREE.Mesh(box, mat);
      m.scale.set(sx, sy, sz); m.position.set(x, y, z); m.castShadow = shadow; this.world.add(m); return m;
    };
    for (const t of L.tools || []) {
      const p0 = this.portPos.get(t.ports[0]);
      const side = Math.sign(p0.x - p0.toolX) || 1;
      const X = (u) => p0.x - side * u;  // u = metres from the rail line toward the tool
      const z0 = t.zMin - 1.2, z1 = t.zMax + 1.2, w = z1 - z0, zc = (z0 + z1) / 2;
      const efemH = 2.05, pmH = 2.5;
      add(efemMat, 1.0, efemH, w, X(0.95), efemH / 2, zc);
      add(winMat, 0.03, 0.5, w - 0.6, X(0.45) + side * 0.016, 1.95 - 0.25, zc, false);  // EFEM window band
      const pmMat = new THREE.MeshStandardMaterial({ color: PROCESS_COLOR[t.process] || 0x999999, roughness: 0.7, emissive: 0x000000 });
      add(pmMat, 1.9, pmH, w, X(2.4), pmH / 2, zc);
      // tower lamp: on the EFEM roof, at the operator end of the tool
      const towerX = X(0.75), towerZ = z1 - 0.3, poleH = 0.32;
      add(poleMat, 0.05, poleH, 0.05, towerX, efemH + poleH / 2, towerZ, false);
      add(poleMat, 0.2, 0.04, 0.2, towerX, efemH + 0.02, towerZ, false);
      const cap = add(capMat, 0.19, 0.05, 0.19, towerX, efemH + poleH + 4 * 0.155 + 0.025, towerZ, false);
      cap.rotation.y = Math.PI / 4;
      const tv = { t, side, X, pmMat, pmX: X(2.4), pmH, zMax: z1, idx: this.toolViews.size, lps: t.ports,
                   towerX, towerZ, towerY: efemH + poleH, seg: [-1, -1, -1, -1] };
      this.toolViews.set(t.id, tv);
      for (const q of t.ports) {
        const p = this.portPos.get(q);
        add(plateMat, 0.06, 1.15, 0.78, X(0.42), 0.95 + 1.15 / 2 - 0.4, p.z);  // load-port plate on the EFEM
        add(doorMat, 0.03, 0.46, 0.5, X(0.42) + side * 0.035, PORT_TOP + 0.26, p.z, false);  // FOUP door opening
        add(stageMat, 0.78, 0.07, 0.72, X(0.0), PORT_TOP - 0.035, p.z);  // stage the FOUP sits on
        add(stageMat, 0.35, PORT_TOP - 0.07, 0.4, X(0.2), (PORT_TOP - 0.07) / 2, p.z);  // stage support
        add(barBgMat, 0.02, 0.1, 0.62, X(0.42) + side * 0.035, 1.5, p.z, false);  // progress bar slot on the plate
        this.lpViews.set(q, { p, side, tool: t.id, plateX: X(0.42), barX: X(0.42) + side * 0.05, X, idx: this.lpViews.size });
      }
    }
  }

  _animateTools(now) {
    const sim = this._simNow(now);
    const d = this.dummy, col = new THREE.Color();
    const pulse = 0.55 + 0.45 * Math.sin(now * 4);
    const ease = (f) => { f = Math.min(1, Math.max(0, f)); return f * f * (3 - 2 * f); };
    const busyTools = new Set();
    let nf = 0, nb = 0;
    d.rotation.set(0, 0, 0);
    for (const lv of this.lpViews.values()) {
      const a = this.lpState.get(lv.p.id);
      const st = a ? a[1] : 'E';
      // FOUP on the stage: slides against the EFEM to dock (door opens), back when undocking
      if (a && st !== 'V') {
        const f = a[3] > a[2] ? (sim - a[2]) / (a[3] - a[2]) : 1;
        const dock = st === 'D' ? ease(f) : st === 'U' ? 1 - ease(f) : (st === 'Q' || st === 'P') ? 1 : 0;
        d.scale.set(1, 1, 1);
        d.position.set(lv.X(this.dockOffset * dock), PORT_TOP + FOUP_H / 2, lv.p.z); d.updateMatrix();
        this.lpFoup.setMatrixAt(nf++, d.matrix);
      }
      if (st === 'P') {
        busyTools.add(lv.tool);
        const f = Math.min(1, Math.max(0, (sim - a[2]) / Math.max(1e-6, a[3] - a[2])));
        const len = 0.58 * f;
        d.position.set(lv.barX, 1.5, lv.p.z - 0.29 + len / 2);
        d.scale.set(0.02, 0.07, Math.max(0.001, len)); d.updateMatrix();
        this.lpBar.setMatrixAt(nb++, d.matrix);
      }
      const c = LP_COLOR[st] ?? LP_COLOR.E;
      if (lv._c !== c || st === 'P') {
        lv._c = c;
        this.lpLamp.setColorAt(lv.idx, col.setHex(c).multiplyScalar(st === 'P' ? pulse + 0.3 : 1));
        this.lpLamp.instanceColor.needsUpdate = true;
      }
    }
    this.lpFoup.count = nf; this.lpFoup.instanceMatrix.needsUpdate = true;
    this.lpBar.count = nb; this.lpBar.instanceMatrix.needsUpdate = true;
    const blinkOn = Math.floor(now * 2.5) % 2 === 0;
    const c1 = new THREE.Color();
    const modes = this.toolModes || {};
    for (const tv of this.toolViews.values()) {
      const busy = busyTools.has(tv.t.id);
      const seg = towerSegments(tv.lps.map((q) => (this.lpState.get(q) || [])[1] || 'E'), modes[tv.t.id]);
      for (let k = 0; k < 4; k++) {
        const on = seg[k] === 1 || (seg[k] === 2 && blinkOn);
        if (tv.seg[k] !== on) {
          tv.seg[k] = on;
          this.tLamp.setColorAt(tv.idx * 4 + k, on ? col.setHex(TOWER[k]) : col.setHex(TOWER_OFF).lerp(c1.setHex(TOWER[k]), 0.14));
          this.tLamp.instanceColor.needsUpdate = true;
        }
      }
      tv.pmMat.emissive.setHex(busy ? PROCESS_COLOR[tv.t.process] || 0x333333 : 0x000000);
      tv.pmMat.emissiveIntensity = busy ? 0.15 + 0.1 * pulse : 0;
    }
  }

  viewTool() {
    // a load port where a FOUP is docking/undocking right now, else one that is processing
    const pick = (sts) => [...this.lpState.values()].filter((a) => sts.includes(a[1])).sort((a, b) => b[2] - a[2])[0];
    const a = pick(['R', 'D']) || pick(['U']) || pick(['P']);
    const lv = a && this.lpViews.get(a[0]);
    if (!lv) return false;
    const { p, side } = lv;
    this.cameraCtl.look([p.x + side * 3.2, 2.8, p.z + 3.6], [p.x - side * 0.8, 1.1, p.z]);
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
    const sim = this._simNow(now);
    const d = this.dummy;
    for (const [id, view] of this.stockerViews) {
      const b = this.skNext.get(id);
      if (!b) continue;
      const a = this.skPrev.get(id) || b;
      const L = (k) => (a[k] ?? b[k]) + (b[k] - (a[k] ?? b[k])) * t;
      const z = L('z'), y = L('y'), fu = L('fork'), rot = L('rot');
      const { sk, X } = view;
      view.crane.position.z = z;
      view.lift.position.y = y;  // fork top
      // turntable: 0 = fork towards the near rack (smaller u), 1 = towards the far rack
      const nearDir = -sk.dir;  // world x direction of decreasing u
      const ang = rot * Math.PI;
      const dx = nearDir * Math.cos(ang), dz = Math.sin(ang);
      view.turn.rotation.y = -Math.atan2(dz, dx);
      const ext = Math.abs(fu - sk.aisleU);  // telescopic extension beyond the turntable
      const len = 0.5 + ext;
      view.tine.scale.set(len, 0.035, 0.15);
      view.tine.position.set(ext + 0.25 - len / 2, -0.0175, 0);
      view.cFoup.visible = !!b.carry;
      view.cFoup.position.set(ext, FOUP_H / 2, 0);
      // conveyors: IN  stage_in -> in (moving) -> inner_in (crane takes it)
      //            OUT inner_out (put by the crane) -> out (moving) -> stage (ready for an OHT)
      let n = 0;
      d.rotation.set(0, 0, 0); d.scale.set(1, 1, 1);
      for (const [pid, kind, t0, t1] of b.conv || []) {
        const p = sk.ports.find((q) => q.id === pid);
        if (!p || n >= 32) continue;
        const f = Math.min(1, Math.max(0, (sim - t0) / Math.max(1e-6, t1 - t0)));
        const e = f * f * (3 - 2 * f);
        const u = kind === 'in' ? sk.nearU * e : kind === 'out' ? sk.nearU * (1 - e)
          : (kind === 'inner_in' || kind === 'inner_out' || kind === 'inner') ? sk.nearU : 0;
        d.position.set(X(u), PORT_TOP + FOUP_H / 2, p.z); d.updateMatrix();
        view.conv.setMatrixAt(n++, d.matrix);
      }
      view.conv.count = n;
      view.conv.instanceMatrix.needsUpdate = true;
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
      let nf = 0, nh = 0, nb = 0, nfi = 0;
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
        const fc = this.H - 0.6 - depth;  // centre of the FOUP held (or about to be held) by the gripper
        const cs = Math.cos(p.h), sn = Math.sin(p.h);
        const at = (ox, y, oz) => d.position.set(p.x + ox * cs - oz * sn, y, p.z + ox * sn + oz * cs);
        at(0, fc + GRIP.plateY, 0); d.updateMatrix();
        this.vGrip.setMatrixAt(i, d.matrix);
        const g = gripClosed(v.state, p.hp);
        const fx = GRIP.openX + (GRIP.closedX - GRIP.openX) * g;
        for (const sgn of [-1, 1]) {  // fingers hang beside the flange; their feet slide under it when closed
          at(sgn * fx, fc + GRIP.fingerY, 0); d.updateMatrix();
          this.vFinger.setMatrixAt(nfi, d.matrix);
          at(sgn * (fx - 0.035), fc + GRIP.footY, 0); d.updateMatrix();
          this.vFoot.setMatrixAt(nfi++, d.matrix);
        }
        const btop = fc + GRIP.plateY + 0.025;
        const blen = hb - btop;
        if (blen > 0.03) {
          for (const [ox, oz] of [[-0.17, -0.14], [0.17, -0.14], [-0.17, 0.14], [0.17, 0.14]]) {
            at(ox, btop + blen / 2, oz);
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
      this.vGrip.count = vs.length; this.vBelt.count = nb; this.vFinger.count = nfi; this.vFoot.count = nfi;
      for (const m of [this.vBody, this.vFoup, this.vHot, this.vGrip, this.vFinger, this.vFoot, this.vBelt]) {
        m.instanceMatrix.needsUpdate = true;
        if (m.instanceColor) m.instanceColor.needsUpdate = true;
      }
    }
    if (this.stockerViews && this.skNext) this._animateStockers(now);
    if (this.lpViews && this.lpFoup) this._animateTools(now);
    this.cameraCtl.update(1 / 60);
    this.renderer.render(this.scene, this.camera);
    this._frames++;
    const t = performance.now();
    if (t - this._fpsT > 1000) { this.fps = (this._frames * 1000) / (t - this._fpsT); this._frames = 0; this._fpsT = t; }
  }
}
