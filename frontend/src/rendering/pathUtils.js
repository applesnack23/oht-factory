// Path geometry mirrored from backend/simulation/routing.py (pure functions,
// no Three.js dependency -> unit-testable with `node --test`).
// Coordinates: (x, z) plane, heading = atan2(dz, dx).

export function pathLength(p) {
  if (p.kind === 'STRAIGHT') return Math.hypot(p.end[0] - p.start[0], p.end[1] - p.start[1]);
  return Math.abs(p.sweep * Math.PI / 180) * p.radius;
}

export function pointAt(p, s) {
  const L = pathLength(p);
  s = Math.min(Math.max(s, 0), L);
  if (p.kind === 'STRAIGHT') {
    const ux = (p.end[0] - p.start[0]) / L, uz = (p.end[1] - p.start[1]) / L;
    return { x: p.start[0] + ux * s, z: p.start[1] + uz * s, h: Math.atan2(uz, ux) };
  }
  const sign = p.sweep >= 0 ? 1 : -1;
  const th = p.startAngle * Math.PI / 180 + sign * s / p.radius;
  return {
    x: p.center[0] + p.radius * Math.cos(th),
    z: p.center[1] + p.radius * Math.sin(th),
    h: Math.atan2(Math.cos(th) * sign, -Math.sin(th) * sign),
  };
}

export function samplePath(p, step = 0.25) {
  const L = pathLength(p);
  const n = Math.max(1, Math.ceil(L / step));
  const out = [];
  for (let i = 0; i <= n; i++) out.push({ s: (L * i) / n, ...pointAt(p, (L * i) / n) });
  return out;
}

export function lerpAngle(a, b, t) {
  let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

// Interpolates material poses between the two most recent server frames.
export class PoseInterpolator {
  constructor() { this.prev = new Map(); this.next = new Map(); this.tPrev = 0; this.tNext = 0; }
  push(materials, recvTime) {
    this.prev = this.next;
    this.tPrev = this.tNext;
    this.next = new Map(materials.map((m) => [m.id, m]));
    this.tNext = recvTime;
  }
  alpha(now) {
    const span = this.tNext - this.tPrev;
    if (span <= 0) return 1;
    return Math.min(1, Math.max(0, (now - this.tNext) / span));
  }
  pose(id, now) {
    const b = this.next.get(id);
    if (!b) return null;
    const a = this.prev.get(id);
    if (!a || a.x === undefined) return { x: b.x, y: b.y, z: b.z, h: b.heading };
    const t = this.alpha(now);
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t, h: lerpAngle(a.heading, b.heading, t) };
  }
}
