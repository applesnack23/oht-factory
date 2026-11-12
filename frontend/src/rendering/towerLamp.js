// Tower lamp (signal tower) on every tool, top to bottom: red, orange, green, blue.
//   red    = alarm (blinking) / manual mode (steady)   - wired, but the simulation never raises it
//   orange = not processing
//   green  = processing finished, transport requested (waiting for an OHT)
//   blue   = processing
// Several segments can be lit at once (e.g. blue + green: one lot running, another waiting for pickup).
export const TOWER = [0xff2a2a, 0xff9a1f, 0x29d65a, 0x2f7bff];
export const TOWER_OFF = 0x2a2e33;  // unlit segment: dark, with a hint of its colour

// Segment states (red, orange, green, blue) for one tool: 0 off, 1 on, 2 blink.
// lpStates: the tool's load-port states (P processing, W finished + transport requested, ...)
// mode: undefined/'auto' | 'manual' | 'alarm'
export function towerSegments(lpStates, mode) {
  const processing = lpStates.includes('P');
  return [mode === 'alarm' ? 2 : mode === 'manual' ? 1 : 0, processing ? 0 : 1, lpStates.includes('W') ? 1 : 0, processing ? 1 : 0];
}
