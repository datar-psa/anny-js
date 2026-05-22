/**
 * Numeric pose verifier: compare each driven bone's *posed local Y axis in
 * world* (= what the driver actually sets) against the direction implied by
 * the MediaPipe landmarks. Small angular error per segment = driver did its
 * job.
 *
 * Important: this compares bone **axes**, not bone-head positions. The driver
 * controls rotations; bone-head positions also depend on rig geometry (twist
 * bones, joint-center conventions) the driver can't touch. Comparing
 * positions leaks ~5-10° of rest-pose bias into every limb row and hides the
 * real per-bone error.
 *
 * Extracted from the original demo/video.html so we can run it under bun:test
 * against captured landmark fixtures.
 */

import type { AnnyModel } from "../../src/anny/index.js";
import type { WorldLandmark } from "../../src/mediapipe/index.js";

export interface SegmentError {
  seg: string;
  errDeg: number;
  mpDir: [number, number, number];
  annyDir: [number, number, number];
}

export interface VerifyResult {
  meanErrDeg: number;
  segments: SegmentError[];
}

/** Extract a posed bone axis (local column `colIdx`) in world coordinates. */
function boneAxisWorld(
  model: AnnyModel,
  boneIndex: Map<string, number>,
  boneXforms: Float32Array,
  name: string,
  colIdx: 0 | 1 | 2,
): [number, number, number] | null {
  const i = boneIndex.get(name);
  if (i === undefined) return null;

  const x = boneXforms, o = i * 16;
  const r = model.restBonePoses, p = i * 16;
  // Rest is row-major: col 0 = (r[0], r[4], r[8]); col 1 = (r[1], r[5], r[9]); col 2 = (r[2], r[6], r[10]).
  const c0 = colIdx, c1 = colIdx + 4, c2 = colIdx + 8;
  const ax = r[p + c0], ay = r[p + c1], az = r[p + c2];
  // boneXforms = pose @ inv(rest), so boneXforms_R @ rest_R_col = pose_R @ e_i.
  const ox = x[o + 0]*ax + x[o + 1]*ay + x[o + 2]*az;
  const oy = x[o + 4]*ax + x[o + 5]*ay + x[o + 6]*az;
  const oz = x[o + 8]*ax + x[o + 9]*ay + x[o +10]*az;
  const L = Math.hypot(ox, oy, oz) || 1;
  return [ox/L, oy/L, oz/L];
}

function mpToAnny(lm: WorldLandmark): [number, number, number] {
  return [lm.x, lm.z, -lm.y];
}

function normSub(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  const v: [number, number, number] = [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const L = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/L, v[1]/L, v[2]/L];
}

function midpt(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2];
}

function angle(
  u: [number, number, number],
  v: [number, number, number],
): number {
  const d = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
  return Math.acos(Math.max(-1, Math.min(1, d))) * 180 / Math.PI;
}

const M = {
  LSH: 11, RSH: 12, LEL: 13, REL: 14, LWR: 15, RWR: 16,
  LHIP: 23, RHIP: 24, LKN: 25, RKN: 26, LAN: 27, RAN: 28,
} as const;

/**
 * Compare the posed bone axes against MediaPipe-derived expected directions.
 * Returns mean angular error (degrees) and a per-segment breakdown.
 */
export function verifyPoseAlignment(
  model: AnnyModel,
  boneIndex: Map<string, number>,
  boneXforms: Float32Array,
  worldLandmarks: WorldLandmark[],
): VerifyResult {
  const mpA: Record<keyof typeof M, [number, number, number]> = Object.fromEntries(
    (Object.entries(M) as [keyof typeof M, number][]).map(([k, i]) => [k, mpToAnny(worldLandmarks[i])])
  ) as Record<keyof typeof M, [number, number, number]>;

  const boneY = (name: string) => boneAxisWorld(model, boneIndex, boneXforms, name, 1);
  const boneX = (name: string) => boneAxisWorld(model, boneIndex, boneXforms, name, 0);

  const limbs: Array<[string, [number, number, number], [number, number, number] | null]> = [
    ["L_upperarm", normSub(mpA.LEL,  mpA.LSH),  boneY("upperarm01.L")],
    ["L_lowerarm", normSub(mpA.LWR,  mpA.LEL),  boneY("lowerarm01.L")],
    ["R_upperarm", normSub(mpA.REL,  mpA.RSH),  boneY("upperarm01.R")],
    ["R_lowerarm", normSub(mpA.RWR,  mpA.REL),  boneY("lowerarm01.R")],
    ["L_upperleg", normSub(mpA.LKN,  mpA.LHIP), boneY("upperleg01.L")],
    ["L_lowerleg", normSub(mpA.LAN,  mpA.LKN),  boneY("lowerleg01.L")],
    ["R_upperleg", normSub(mpA.RKN,  mpA.RHIP), boneY("upperleg01.R")],
    ["R_lowerleg", normSub(mpA.RAN,  mpA.RKN),  boneY("lowerleg01.R")],
  ];

  // hip axis: root is driven with a full rot target. Its local X should point
  // along the dancer's anatomical-left direction.
  const rootX = boneX("root");
  if (rootX) limbs.push(["hip_axis", normSub(mpA.LHIP, mpA.RHIP), rootX]);

  // spine: spine03's local Y should align with hip-mid → shoulder-mid.
  const spineY = boneY("spine03");
  if (spineY) {
    const spineDir = normSub(midpt(mpA.LSH, mpA.RSH), midpt(mpA.LHIP, mpA.RHIP));
    limbs.push(["spine", spineDir, spineY]);
  }

  const segments: SegmentError[] = [];
  let sum = 0;
  let counted = 0;
  for (const [name, mpDir, annyDir] of limbs) {
    if (!annyDir) continue;
    const errDeg = angle(mpDir, annyDir);
    segments.push({ seg: name, errDeg, mpDir, annyDir });
    sum += errDeg;
    counted++;
  }

  return {
    meanErrDeg: counted > 0 ? sum / counted : 0,
    segments,
  };
}
