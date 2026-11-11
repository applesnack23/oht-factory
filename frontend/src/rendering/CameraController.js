// OrbitControls camera with presets (overview / top / front) and fly-to-equipment.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

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
    this.anim = null;
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
  focus(point) {
    const t = point.clone();
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this._fly(t.clone().add(dir.multiplyScalar(9)).setY(Math.max(4, t.y + 5)), t);
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
