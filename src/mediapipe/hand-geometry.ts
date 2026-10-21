/**
 * Hand-local geometry shared by the driver's two inputs — MediaPipe hand
 * landmarks and the rig's own rest joints — so finger angles can be
 * transferred as *differences* (MP angle − rig rest angle) rather than as
 * world directions. That makes finger articulation independent of the palm
 * orientation estimate and of the subject's knuckle layout.
 *
 * Inputs are 21 points in MP_HAND order, Anny world coordinates.
 *
 * Chirality: with fwd = wrist→middle-MCP and across = pinky→index (⊥ fwd),
 * the palmar normal is +(across × fwd) for a RIGHT hand and −(across × fwd)
 * for a LEFT hand. (Measured: MediaPipe's labelled test photos and the rig's
 * own +X finger flexion agree on this; see docs/hand-tracking.md.)
 */

import { MP_HAND } from "./constants.js";
import {
  DEG, frameXY, signedAngle, vcross, vdot, vneg, vnorm, vsub,
  type Mat3, type Vec3,
} from "./geom.js";
import type { WorldLandmark } from "./types.js";

export type HandSide = "L" | "R";

/** Fingers 2–5 (index..pinky) as [MCP, PIP, DIP, TIP] landmark indices. */
export const FINGER_CHAINS: ReadonlyArray<readonly [number, number, number, number]> = [
  [MP_HAND.INDEX_MCP, MP_HAND.INDEX_PIP, MP_HAND.INDEX_DIP, MP_HAND.INDEX_TIP],
  [MP_HAND.MIDDLE_MCP, MP_HAND.MIDDLE_PIP, MP_HAND.MIDDLE_DIP, MP_HAND.MIDDLE_TIP],
  [MP_HAND.RING_MCP, MP_HAND.RING_PIP, MP_HAND.RING_DIP, MP_HAND.RING_TIP],
  [MP_HAND.PINKY_MCP, MP_HAND.PINKY_PIP, MP_HAND.PINKY_DIP, MP_HAND.PINKY_TIP],
];

/** Palm frame, columns (X across toward index, Y wrist→middle MCP, Z = X×Y). */
export function palmFrame(p: readonly Vec3[]): Mat3 {
  return frameXY(
    vsub(p[MP_HAND.INDEX_MCP], p[MP_HAND.PINKY_MCP]),
    vsub(p[MP_HAND.MIDDLE_MCP], p[MP_HAND.WRIST]),
  );
}

/** Unit palmar normal (out of the palm) for a hand of the given side. */
export function palmarNormal(frame: Mat3, side: HandSide): Vec3 {
  const z: Vec3 = [frame[2], frame[5], frame[8]];
  return side === "R" ? z : vneg(z);
}

export interface FingerAngles {
  /** Per finger (index..pinky): [MCP, PIP, DIP] flexion, radians, palmar +. */
  flex: number[][];
  /**
   * Per finger: MCP abduction, radians, side-normalised — the proximal
   * phalanx's tilt out of its finger's flexion plane. Measured out-of-plane
   * (not as an in-palm angle) so it stays well-defined on a curled finger.
   */
  abd: number[];
}

/**
 * Each finger flexes about its own hinge, ⊥ its metacarpal (wrist → MCP) and
 * the palm normal. Flexion angles are measured about that hinge; abduction
 * is the elevation out of the plane it spans — the same decomposition as a
 * rotation Rx(flex)·Rz(abd) in a phalanx frame whose X is the hinge.
 */
export function fingerAngles(p: readonly Vec3[], side: HandSide): FingerAngles {
  const palmar = palmarNormal(palmFrame(p), side);
  const sideSign = side === "R" ? 1 : -1;
  const flex: number[][] = [];
  const abd: number[] = [];
  for (const [mcp, pip, dip, tip] of FINGER_CHAINS) {
    const seg = [
      vsub(p[mcp], p[MP_HAND.WRIST]), vsub(p[pip], p[mcp]),
      vsub(p[dip], p[pip]), vsub(p[tip], p[dip]),
    ];
    // Rotating the metacarpal about `hinge` by +θ moves it toward the palm.
    const hinge = vnorm(vcross(seg[0], palmar));
    flex.push([0, 1, 2].map(j => signedAngle(seg[j], seg[j + 1], hinge)));
    abd.push(sideSign * Math.asin(Math.max(-1, Math.min(1, vdot(vnorm(seg[1]), hinge)))));
  }
  return { flex, abd };
}

/** Thumb segment directions (CMC→MCP, MCP→IP, IP→TIP) in palm-frame coordinates. */
export function thumbDirections(p: readonly Vec3[]): Vec3[] {
  const F = palmFrame(p);
  const seg = (a: number, b: number): Vec3 => {
    const d = vnorm(vsub(p[b], p[a]));
    return [F[0] * d[0] + F[3] * d[1] + F[6] * d[2], F[1] * d[0] + F[4] * d[1] + F[7] * d[2], F[2] * d[0] + F[5] * d[1] + F[8] * d[2]];
  };
  return [
    seg(MP_HAND.THUMB_CMC, MP_HAND.THUMB_MCP),
    seg(MP_HAND.THUMB_MCP, MP_HAND.THUMB_IP),
    seg(MP_HAND.THUMB_IP, MP_HAND.THUMB_TIP),
  ];
}

/** Mean PIP+DIP flexion over fingers 2–5 (radians, palmar positive). */
function meanCurl(pts: readonly Vec3[], side: HandSide): number {
  const a = fingerAngles(pts, side);
  return a.flex.reduce((s, f) => s + f[1] + f[2], 0) / a.flex.length;
}

/** Fingers curling this far *away* from the palm on average mean mirrored chirality. */
const CHIRALITY_CURL = 25 * DEG;

/**
 * Hand landmarks (MediaPipe camera axes) → Anny coordinates, depth-flipped
 * if their chirality contradicts `side`.
 *
 * Fingers curling strongly *away* from the palm mean MediaPipe picked the
 * mirror-image hand (its classic depth ambiguity — same 2D silhouette,
 * opposite chirality). Reflecting depth about the wrist restores a hand of
 * the right chirality with the same image projection.
 */
export function handPointsAnny(hand: readonly WorldLandmark[], side: HandSide, mirrorX = false): Vec3[] {
  const sx = mirrorX ? -1 : 1;
  const pts = hand.map((lm): Vec3 => [sx * lm.x, lm.z, -lm.y]);
  if (meanCurl(pts, side) > -CHIRALITY_CURL) return pts;
  const w = pts[MP_HAND.WRIST];
  const flipped = pts.map((p): Vec3 => [p[0], 2 * w[1] - p[1], p[2]]);
  return meanCurl(flipped, side) > meanCurl(pts, side) ? flipped : pts;
}
