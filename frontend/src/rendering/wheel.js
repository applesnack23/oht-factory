// Tell a mouse wheel from a trackpad gesture.
// Mouse wheels report line/page steps, or (Chrome/Safari) a legacy wheelDelta in whole notches of 120;
// trackpad swipes report small free-form pixel deltas. Pinch-zoom arrives as a wheel event with ctrlKey.
export function isMouseWheel(e) {
  if (e.deltaMode === 1 || e.deltaMode === 2) return true;
  const w = e.wheelDeltaY;
  return e.deltaX === 0 && typeof w === 'number' && w !== 0 && w % 120 === 0 && Math.abs(e.deltaY) >= 4;
}

// What a wheel event should do with the camera.
export function wheelAction(e) {
  if (e.ctrlKey) return 'zoom';  // trackpad pinch
  if (e.shiftKey) return 'pan';
  return isMouseWheel(e) ? 'zoom' : 'pan';
}
