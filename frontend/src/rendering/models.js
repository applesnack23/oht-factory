// Small procedural models built from boxes (merged into one geometry so they can be instanced).
import * as THREE from 'three';

// Merge boxes [sx, sy, sz, x, y, z] into one non-indexed BufferGeometry.
export function mergeBoxes(boxes) {
  const pos = [], nor = [];
  for (const [sx, sy, sz, x, y, z] of boxes) {
    const g = new THREE.BoxGeometry(sx, sy, sz).toNonIndexed();
    g.translate(x, y, z);
    pos.push(...g.attributes.position.array);
    nor.push(...g.attributes.normal.array);
    g.dispose();
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  out.computeBoundingSphere();
  return out;
}

// FOUP, origin at its centre, 0.42 m tall: shell + top handle (neck + flange) that the OHT gripper holds.
export const FOUP = { h: 0.42, flangeTop: 0.195, flangeBottom: 0.16, flangeHalf: 0.15 };
export function foupGeometry() {
  return mergeBoxes([
    [0.5, 0.33, 0.45, 0, -0.045, 0],       // shell (-0.21 .. 0.12)
    [0.46, 0.02, 0.41, 0, 0.13, 0],        // rounded-top step
    [0.16, 0.03, 0.16, 0, 0.145, 0],       // neck
    [0.3, 0.035, 0.26, 0, 0.1775, 0],      // flange (0.16 .. 0.195): the robotic handle
    [0.04, 0.12, 0.22, 0.255, -0.06, 0],   // side handles
    [0.04, 0.12, 0.22, -0.255, -0.06, 0],
  ]);
}

// OHT gripper pieces, relative to the centre of the FOUP it holds.
export const GRIP = { plateY: 0.22, fingerY: 0.175, footY: 0.145, closedX: 0.168, openX: 0.235 };
export const plateGeometry = () => mergeBoxes([[0.42, 0.05, 0.36, 0, 0, 0], [0.12, 0.05, 0.12, 0, 0.05, 0]]);
export const fingerGeometry = () => mergeBoxes([[0.03, 0.09, 0.2, 0, 0, 0]]);
export const footGeometry = () => mergeBoxes([[0.07, 0.02, 0.2, 0, 0, 0]]);

// 0 = open, 1 = closed, from the hoist state and progress.
export function gripClosed(state, hp) {
  const ramp = (a, b, x) => Math.min(1, Math.max(0, (x - a) / (b - a)));
  if (state === 'LOADING') return ramp(0.44, 0.5, hp);          // close around the flange at the port
  if (state === 'UNLOADING') return 1 - ramp(0.5, 0.56, hp);    // open after setting the FOUP down
  return state === 'TO_DROP' ? 1 : 0;
}
