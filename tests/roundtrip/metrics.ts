/**
 * View-invariant pose descriptors + comparisons for the round-trip validator.
 *
 * Every source (MediaPipe on a photo, MediaPipe on a render, the rig's own
 * joints) is reduced to the same descriptors, computed by the same code, in
 * MediaPipe world axes (+x right, +y down, +z into the scene — camera-aligned
 * for BOTH pose and hand world landmarks; see docs/hand-tracking.md).
 *
 * Hand convention (verified on MediaPipe's own labelled test photos and on
 * renders of the rig): with fwd = wrist→middle-MCP and across = pinky-MCP→
 * index-MCP, the palmar normal is −cross(fwd, across) for a RIGHT hand and
 * +cross(fwd, across) for a LEFT hand. Finger flexion is signed: positive =
 * toward the palm (natural curl), negative = hyperextension ("bent backward").
 */

import { MP, MP_HAND } from "../../src/mediapipe/index.js";
import {
  angleDeg, cross, dot, mid, neg, norm, RAD2DEG, signedAngleDeg, sub, type V3,
} from "./vec.js";

export type Side = "L" | "R";
type P3 = { x: number; y: number; z: number };
const v = (p: P3): V3 => [p.x, p.y, p.z];

// ── Hands ──────────────────────────────────────────────────────────────────

/** Fingers 2–5 (index..pinky): [MCP, PIP, DIP, TIP] landmark indices. */
const FINGERS: ReadonlyArray<readonly [number, number, number, number]> = [
  [MP_HAND.INDEX_MCP, MP_HAND.INDEX_PIP, MP_HAND.INDEX_DIP, MP_HAND.INDEX_TIP],
  [MP_HAND.MIDDLE_MCP, MP_HAND.MIDDLE_PIP, MP_HAND.MIDDLE_DIP, MP_HAND.MIDDLE_TIP],
  [MP_HAND.RING_MCP, MP_HAND.RING_PIP, MP_HAND.RING_DIP, MP_HAND.RING_TIP],
  [MP_HAND.PINKY_MCP, MP_HAND.PINKY_PIP, MP_HAND.PINKY_DIP, MP_HAND.PINKY_TIP],
];
export const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"] as const;

export interface HandFrame {
  fwd: V3;
  across: V3;
  /** Unit normal pointing out of the palm. */
  palmar: V3;
}

/** Palm frame from wrist + index/middle/pinky MCPs, chirality from `side`. */
export function handFrame(p: V3[], side: Side): HandFrame {
  const fwd = norm(sub(p[MP_HAND.MIDDLE_MCP], p[MP_HAND.WRIST]));
  const acrossRaw = sub(p[MP_HAND.INDEX_MCP], p[MP_HAND.PINKY_MCP]);
  const across = norm(sub(acrossRaw, [fwd[0] * dot(acrossRaw, fwd), fwd[1] * dot(acrossRaw, fwd), fwd[2] * dot(acrossRaw, fwd)]));
  const c = norm(cross(fwd, across));
  return { fwd, across, palmar: side === "R" ? neg(c) : c };
}

export interface HandAngles {
  /**
   * Signed flexion per joint, degrees, palmar positive.
   * Rows: thumb [CMC→MCP vs palm-plane, MCP, IP], index..pinky [MCP, PIP, DIP].
   */
  flex: number[][];
  /** Index..pinky MCP abduction (out of each finger's flexion plane), degrees. */
  abd: number[];
  frame: HandFrame;
}

export function handAngles(world: P3[], side: Side): HandAngles {
  const p = world.map(v);
  const frame = handFrame(p, side);
  // Hinge axis: rotating `fwd` about `hinge` by +θ moves it toward the palm.
  const hinge = norm(cross(frame.fwd, frame.palmar));
  const flex: number[][] = [];

  // Thumb: its pad faces the index finger rather than the palm normal, so
  // MCP/IP flexion is signed toward the index MCP (positive = curling in).
  const t = [MP_HAND.THUMB_CMC, MP_HAND.THUMB_MCP, MP_HAND.THUMB_IP, MP_HAND.THUMB_TIP].map(i => p[i]);
  const tSeg = [sub(t[1], t[0]), sub(t[2], t[1]), sub(t[3], t[2])];
  const thumbJoint = (k: 1 | 2): number => {
    const pad = sub(p[MP_HAND.INDEX_MCP], t[k]);
    return signedAngleDeg(tSeg[k - 1], tSeg[k], norm(cross(tSeg[k - 1], pad)));
  };
  flex.push([
    // Thumb metacarpal elevation out of the palm plane (positive = palmar, i.e. opposition).
    90 - angleDeg(tSeg[0], frame.palmar),
    thumbJoint(1),
    thumbJoint(2),
  ]);

  const abd: number[] = [];
  for (const [mcp, pip, dip, tip] of FINGERS) {
    const seg = [sub(p[mcp], p[MP_HAND.WRIST]), sub(p[pip], p[mcp]), sub(p[dip], p[pip]), sub(p[tip], p[dip])];
    flex.push([
      signedAngleDeg(seg[0], seg[1], hinge),
      signedAngleDeg(seg[1], seg[2], hinge),
      signedAngleDeg(seg[2], seg[3], hinge),
    ]);
    // Abduction = proximal phalanx elevation out of the finger's flexion
    // plane (stable on curled fingers, unlike an in-palm-plane angle).
    // Sign-normalised so mirrored L/R hands report the same value.
    const own = norm(cross(seg[0], frame.palmar));
    abd.push((side === "R" ? 1 : -1) * Math.asin(Math.max(-1, Math.min(1, dot(norm(seg[1]), own)))) * RAD2DEG);
  }
  return { flex, abd, frame };
}

/**
 * Chirality read off the 3D geometry alone: fingers curl toward the palm, so
 * the fingertips sit on the palmar side of the knuckles. Returns the side the
 * landmarks look like, or undefined when the hand is too flat to tell.
 */
export function geometricSide(world: P3[], minCurl = 0.15): Side | undefined {
  const p = world.map(v);
  const fwd = norm(sub(p[MP_HAND.MIDDLE_MCP], p[MP_HAND.WRIST]));
  const across = norm(sub(p[MP_HAND.INDEX_MCP], p[MP_HAND.PINKY_MCP]));
  const c = norm(cross(fwd, across));
  const palmLen = Math.hypot(...sub(p[MP_HAND.MIDDLE_MCP], p[MP_HAND.WRIST]));
  let s = 0;
  for (const [mcp, , , tip] of FINGERS) s += dot(sub(p[tip], p[mcp]), c);
  const curl = s / 4 / palmLen;
  if (Math.abs(curl) < minCurl) return undefined;
  return curl > 0 ? "L" : "R";
}

export interface HandDiff {
  /** Mean |Δflex| over fingers 2–5 (12 joints), degrees. */
  fingerFlex: number;
  /** Mean |Δflex| over the thumb's 3 measures, degrees. */
  thumbFlex: number;
  /** Mean |Δabd|, degrees. */
  abduction: number;
  /** Angle between palm normals (camera frame), degrees. */
  palmNormal: number;
  /** Angle between hand directions wrist→middle-MCP (camera frame), degrees. */
  handDir: number;
  /** Joints where both sources bend ≥ 20° but in OPPOSITE directions. */
  bendSignFlips: number;
  perJoint: number[][];
}

export function compareHands(a: HandAngles, b: HandAngles): HandDiff {
  const perJoint = a.flex.map((row, f) => row.map((x, j) => Math.abs(x - b.flex[f][j])));
  const fingers = perJoint.slice(1).flat();
  let flips = 0;
  a.flex.forEach((row, f) => row.forEach((x, j) => {
    const y = b.flex[f][j];
    if (Math.abs(x) >= 20 && Math.abs(y) >= 20 && Math.sign(x) !== Math.sign(y)) flips++;
  }));
  return {
    fingerFlex: mean(fingers),
    thumbFlex: mean(perJoint[0]),
    abduction: mean(a.abd.map((x, i) => Math.abs(x - b.abd[i]))),
    palmNormal: angleDeg(a.frame.palmar, b.frame.palmar),
    handDir: angleDeg(a.frame.fwd, b.frame.fwd),
    bendSignFlips: flips,
    perJoint,
  };
}

// ── 2D (image-plane) agreement ─────────────────────────────────────────────

export type P2 = [number, number];

/**
 * 2D landmark agreement after removing translation and uniform scale (NOT
 * rotation — both images share the camera orientation). Returns the mean
 * residual distance as a fraction of `b`'s palm length (wrist → middle MCP),
 * so 0.1 means "landmarks are off by 10% of a palm on average".
 *
 * MediaPipe's 2D landmarks are far more reliable than its depth, so this is
 * the sharpest round-trip judge of "does the render look like the source".
 */
export function compare2D(a: P2[], b: P2[]): { err: number; perPoint: number[] } {
  const n = a.length;
  const ca: P2 = [a.reduce((s, p) => s + p[0], 0) / n, a.reduce((s, p) => s + p[1], 0) / n];
  const cb: P2 = [b.reduce((s, p) => s + p[0], 0) / n, b.reduce((s, p) => s + p[1], 0) / n];
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const ax = a[i][0] - ca[0], ay = a[i][1] - ca[1];
    num += ax * (b[i][0] - cb[0]) + ay * (b[i][1] - cb[1]);
    den += ax * ax + ay * ay;
  }
  const s = den > 0 ? num / den : 1;
  const palm = Math.hypot(b[MP_HAND.MIDDLE_MCP][0] - b[MP_HAND.WRIST][0], b[MP_HAND.MIDDLE_MCP][1] - b[MP_HAND.WRIST][1]) || 1;
  const perPoint = a.map((p, i) => Math.hypot(
    s * (p[0] - ca[0]) - (b[i][0] - cb[0]),
    s * (p[1] - ca[1]) - (b[i][1] - cb[1]),
  ) / palm);
  return { err: mean(perPoint), perPoint };
}

/** Normalised MP image landmarks → pixels. */
export const toPx = (lms: { x: number; y: number }[], W: number, H: number): P2[] =>
  lms.map(l => [l.x * W, l.y * H]);

// ── Body ───────────────────────────────────────────────────────────────────

/** Named limb segments as (from, to) MP pose indices. */
export const BODY_SEGMENTS: Readonly<Record<string, readonly [number, number]>> = {
  "upperarm.L": [MP.LEFT_SHOULDER, MP.LEFT_ELBOW],
  "upperarm.R": [MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW],
  "forearm.L": [MP.LEFT_ELBOW, MP.LEFT_WRIST],
  "forearm.R": [MP.RIGHT_ELBOW, MP.RIGHT_WRIST],
  "thigh.L": [MP.LEFT_HIP, MP.LEFT_KNEE],
  "thigh.R": [MP.RIGHT_HIP, MP.RIGHT_KNEE],
  "shin.L": [MP.LEFT_KNEE, MP.LEFT_ANKLE],
  "shin.R": [MP.RIGHT_KNEE, MP.RIGHT_ANKLE],
  "shoulders": [MP.RIGHT_SHOULDER, MP.LEFT_SHOULDER],
  "hips": [MP.RIGHT_HIP, MP.LEFT_HIP],
};

export function bodyDirections(world: (P3 | undefined)[]): Record<string, V3> {
  const out: Record<string, V3> = {};
  for (const [name, [a, b]] of Object.entries(BODY_SEGMENTS)) {
    const pa = world[a], pb = world[b];
    if (pa && pb) out[name] = norm(sub(v(pb), v(pa)));
  }
  const ls = world[MP.LEFT_SHOULDER], rs = world[MP.RIGHT_SHOULDER];
  const lh = world[MP.LEFT_HIP], rh = world[MP.RIGHT_HIP];
  if (ls && rs && lh && rh) out["torso"] = norm(sub(mid(v(ls), v(rs)), mid(v(lh), v(rh))));
  return out;
}

/** Per-segment angle (deg) between two direction sets, over shared keys. */
export function compareBody(a: Record<string, V3>, b: Record<string, V3>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const k of Object.keys(a)) if (b[k]) out[k] = angleDeg(a[k], b[k]);
  return out;
}

export function mean(xs: number[]): number {
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : NaN;
}

export function percentile(xs: number[], q: number): number {
  if (!xs.length) return NaN;
  const s = [...xs].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}
