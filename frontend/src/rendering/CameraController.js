// OrbitControls camera with presets (overview / top / front) and fly-to-equipment.
//
// Wheel / trackpad:
//   two-finger swipe (trackpad)  -> move the view over the floor (pan), like a map
//   pinch (trackpad, ctrlKey)    -> zoom
//   mouse wheel (line steps)     -> zoom;  Shift + wheel -> pan
//   left drag rotate, right drag pan (OrbitControls)
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { wheelAction } from './wheel.js';

export class CameraController {
  constructor(camera, dom, center = new THREE.Vector3(3, 0, 0), scale = 1) {
    this.scale = scale;
    this.camera = camera;
    this.center = center;
    this.controls = new OrbitControls(camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.03;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 90 * scale;
    this.controls.screenSpacePanning = true;
    this.controls.enableZoom = false;  // wheel handled below: swipe = pan, pinch = zoom
    this.anim = null;
    this._wheel = (e) => this.onWheel(e);
    dom.addEventListener('wheel', this._wheel, { passive: false });
    this.overview(true);
  }

  _fly(pos, target, instant = false) {
    if (instant) {
      this.camera.position.copy(pos);
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    this.anim = { p0: this.camera.position.clone(), t0: this.controls.target.clone(), p1: pos, t1: target, k: 0 };
  }

  overview(instant) { const k = this.scale; this._fly(new THREE.Vector3(this.center.x - 4 * k, 24 * k, this.center.z + 26 * k), this.center.clone(), instant); }
  top() { this._fly(new THREE.Vector3(this.center.x, 46 * this.scale, this.center.z + 0.01), this.center.clone()); }
  front() { this._fly(new THREE.Vector3(this.center.x, 6 * this.scale, this.center.z + 30 * this.scale), new THREE.Vector3(this.center.x, 1, this.center.z)); }
  look(pos, target) { this._fly(new THREE.Vector3(...pos), new THREE.Vector3(...target)); }
  focus(point) {
    const t = point.clone();
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this._fly(t.clone().add(dir.multiplyScalar(9)).setY(Math.max(4, t.y + 5)), t);
  }

  onWheel(e) {
    e.preventDefault();
    this.anim = null;
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    const dx = e.deltaX * unit, dy = e.deltaY * unit;
    if (wheelAction(e) === 'zoom') this.zoom(Math.exp(dy * (e.ctrlKey ? 0.01 : 0.0015)));
    else if (e.shiftKey && dx === 0) this.pan(dy, 0);  // Shift + mouse wheel: sideways
    else this.pan(dx, dy);
  }

  zoom(factor) {
    const c = this.controls;
    const off = this.camera.position.clone().sub(c.target);
    const len = THREE.MathUtils.clamp(off.length() * factor, c.minDistance, c.maxDistance);
    this.camera.position.copy(c.target).add(off.setLength(len));
  }

  pan(dx, dy) {
    // move camera and target together over the floor plane; speed scales with the viewing distance
    const dist = this.camera.position.distanceTo(this.controls.target);
    const k = dist * 0.0016;
    const fwd = new THREE.Vector3();
    this.camera.getWorldDirection(fwd);
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1).applyQuaternion(this.camera.quaternion).setY(0);  // looking straight down
    fwd.normalize();
    const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
    const move = right.multiplyScalar(dx * k).add(fwd.multiplyScalar(-dy * k));
    this.camera.position.add(move);
    this.controls.target.add(move);
  }

  update(dt) {
    if (this.anim) {
      const a = this.anim;
      a.k = Math.min(1, a.k + dt * 1.8);
      const e = a.k < 0.5 ? 2 * a.k * a.k : 1 - Math.pow(-2 * a.k + 2, 2) / 2;
      this.camera.position.lerpVectors(a.p0, a.p1, e);
      this.controls.target.lerpVectors(a.t0, a.t1, e);
      if (a.k >= 1) this.anim = null;
    }
    this.controls.update();
  }
}
