// node --test frontend/tests  (pure JS: path math must match the Python server)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathLength, pointAt, lerpAngle, PoseInterpolator } from '../src/rendering/pathUtils.js';

const fixture = JSON.parse(readFileSync(new URL('./layout_fixture.json', import.meta.url)));

test('path length and points match server geometry', () => {
  for (const c of fixture.conveyors) {
    assert.ok(Math.abs(pathLength(c.path) - c.length) < 1e-3, c.id);
    for (const s of c.samples) {
      const p = pointAt(c.path, s.s);
      assert.ok(Math.abs(p.x - s.x) < 1e-6 && Math.abs(p.z - s.z) < 1e-6, `${c.id}@${s.s}`);
      assert.ok(Math.abs(Math.atan2(Math.sin(p.h - s.h), Math.cos(p.h - s.h))) < 1e-6, `${c.id} heading`);
    }
  }
});

test('connected conveyors meet at the same point', () => {
  const byId = Object.fromEntries(fixture.conveyors.map((c) => [c.id, c]));
  for (const c of fixture.conveyors) for (const n of c.next) {
    const a = pointAt(c.path, c.length), b = pointAt(byId[n].path, 0);
    assert.ok(Math.hypot(a.x - b.x, a.z - b.z) < 1e-6, `${c.id}->${n}`);
  }
});

test('angle interpolation takes the short way', () => {
  assert.ok(Math.abs(lerpAngle(3.1, -3.1, 0.5) - Math.PI) < 0.05);
});

test('pose interpolation between frames', () => {
  const it = new PoseInterpolator();
  it.push([{ id: 'A', x: 0, y: 1, z: 0, heading: 0 }], 10.0);
  it.push([{ id: 'A', x: 1, y: 1, z: 0, heading: 0 }], 10.1);
  assert.ok(Math.abs(it.pose("A", 10.15).x - 0.5) < 1e-9);
  assert.equal(it.pose('A', 10.5).x, 1);
  assert.equal(it.pose('B', 10.5), null);
});
