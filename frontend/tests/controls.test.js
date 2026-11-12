import test from 'node:test';
import assert from 'node:assert/strict';
import { towerSegments, TOWER } from '../src/rendering/towerLamp.js';
import { isMouseWheel, wheelAction } from '../src/rendering/wheel.js';

test('tower lamp: blue processing, green finished, orange idle, red alarm/manual', () => {
  assert.equal(TOWER.length, 4);
  assert.deepEqual(towerSegments(['P', 'E']), [0, 0, 0, 1]);
  assert.deepEqual(towerSegments(['P', 'W']), [0, 0, 1, 1]);
  assert.deepEqual(towerSegments(['W', 'E']), [0, 1, 1, 0]);
  assert.deepEqual(towerSegments(['E', 'V']), [0, 1, 0, 0]);
  assert.deepEqual(towerSegments(['Q', 'D']), [0, 1, 0, 0]);
  assert.deepEqual(towerSegments(['P'], 'alarm'), [2, 0, 0, 1]);
  assert.deepEqual(towerSegments(['E'], 'manual'), [1, 1, 0, 0]);
});

test('wheel: trackpad swipe pans, pinch and mouse wheel zoom', () => {
  assert.equal(wheelAction({ deltaMode: 0, deltaX: 3.5, deltaY: -12.25, wheelDeltaY: 36.75, ctrlKey: false }), 'pan');
  assert.equal(wheelAction({ deltaMode: 0, deltaX: 0, deltaY: -60, wheelDeltaY: 180 * 1 + 0.5, ctrlKey: false }), 'pan');
  assert.equal(wheelAction({ deltaMode: 0, deltaX: 0, deltaY: -4.2, wheelDeltaY: 12.6, ctrlKey: true }), 'zoom');
  assert.equal(wheelAction({ deltaMode: 0, deltaX: 0, deltaY: 100, wheelDeltaY: -120, ctrlKey: false }), 'zoom');
  assert.equal(wheelAction({ deltaMode: 1, deltaX: 0, deltaY: 3, ctrlKey: false }), 'zoom');
  assert.equal(wheelAction({ deltaMode: 1, deltaX: 0, deltaY: 3, ctrlKey: false, shiftKey: true }), 'pan');
  assert.equal(isMouseWheel({ deltaMode: 0, deltaX: 0, deltaY: 40, wheelDeltaY: -120 }), true);
});
