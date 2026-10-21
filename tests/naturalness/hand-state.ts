/**
 * The hand state the fusion works in (src/mediapipe/hand-prior.ts): finger
 * angles relative to the rig's rest, degrees — [index..pinky × (MCP, PIP,
 * DIP)], [index..pinky] abduction — then the thumb's three segment
 * directions in the palm frame (unit vectors × 180/π, so small changes read
 * in degrees like the angles).
 */

import { fingerAngles, thumbDirections } from "../../src/mediapipe/hand-geometry.js";
import type { Vec3 } from "../../src/mediapipe/geom.js";
import type { RigInfo } from "../../src/mediapipe/rig.js";
import type { Side } from "../roundtrip/metrics.js";

export const STATE_DIMS = 25;
export const FINGER_DIMS = 16;
const R2D = 180 / Math.PI;

/** `pts`: 21 hand points in Anny coordinates (MP_HAND order). */
export function handState(pts: Vec3[], side: Side, rig: RigInfo): number[] {
  const a = fingerAngles(pts, side);
  const hr = rig.hands[side];
  return [
    ...a.flex.flatMap((f, i) => f.map((x, j) => (x - hr.fingers[i].restFlex[j]) * R2D)),
    ...a.abd.map((x, i) => (x - hr.fingers[i].restAbd) * R2D),
    ...thumbDirections(pts).flatMap(d => d.map(x => x * R2D)),
  ];
}
