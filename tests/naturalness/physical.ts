/**
 * Physical plausibility of a posed hand, from the skinned rig itself:
 *
 *   • finger interpenetration — each phalanx as a capsule (radius measured
 *     from the mesh); overlap between fingers beyond what the rest pose
 *     already has means fingers pass through each other.
 *   • skinning damage — triangles of the forearm+hand that collapse (area
 *     < 25 % of rest) or flip (normal opposite to where their bone turned
 *     them): the candy-wrapper / crushed-knuckle look of over-twisted joints.
 *
 * Neither uses MediaPipe or any joint-angle convention, so they're
 * independent of both the driver and the rule-based anatomy score.
 */

import { forwardKinematics, allocBoneTransforms, lbs, type AnnyModel, type PoseDeltas } from "../../src/anny/index.js";
import type { Side } from "../roundtrip/metrics.js";

type V = [number, number, number];

interface Capsule { finger: number; bone: number; child: number | null; tipLen: number; radius: number }

export interface PhysicalRig {
  model: AnnyModel;
  capsules: Record<Side, Capsule[]>;
  /** Rest overlap (mm) per capsule pair key "i-j", to subtract. */
  restOverlap: Record<Side, Map<string, number>>;
  /** Faces (indices into model.faces / 3) of each side's forearm + hand. */
  faces: Record<Side, number[]>;
  restArea: Float32Array;
  dominantBone: Int32Array;
  restArm: Record<Side, { twist: number; flex: number; deviation: number }>;
}

const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Closest distance between segments p0p1 and q0q1. */
function segSegDistance(p0: V, p1: V, q0: V, q1: V): number {
  const d1 = sub(p1, p0), d2 = sub(q1, q0), r = sub(p0, q0);
  const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
  let s: number, t: number;
  const c = dot(d1, r), b = dot(d1, d2), den = a * e - b * b;
  s = den > 1e-12 ? Math.min(1, Math.max(0, (b * f - c * e) / den)) : 0;
  t = e > 1e-12 ? (b * s + f) / e : 0;
  if (t < 0) { t = 0; s = a > 1e-12 ? Math.min(1, Math.max(0, -c / a)) : 0; }
  else if (t > 1) { t = 1; s = a > 1e-12 ? Math.min(1, Math.max(0, (b - c) / a)) : 0; }
  const cp = [p0[0] + d1[0] * s, p0[1] + d1[1] * s, p0[2] + d1[2] * s] as V;
  const cq = [q0[0] + d2[0] * t, q0[1] + d2[1] * t, q0[2] + d2[2] * t] as V;
  return Math.hypot(...sub(cp, cq));
}

function restHead(model: AnnyModel, b: number): V {
  const r = model.restBonePoses, o = b * 16;
  return [r[o + 3], r[o + 7], r[o + 11]];
}

function posedPoint(bx: Float32Array, b: number, p: V): V {
  const o = b * 16;
  return [
    bx[o] * p[0] + bx[o + 1] * p[1] + bx[o + 2] * p[2] + bx[o + 3],
    bx[o + 4] * p[0] + bx[o + 5] * p[1] + bx[o + 6] * p[2] + bx[o + 7],
    bx[o + 8] * p[0] + bx[o + 9] * p[1] + bx[o + 10] * p[2] + bx[o + 11],
  ];
}

function segments(rig: PhysicalRig, side: Side, bx: Float32Array): [V, V][] {
  const m = rig.model;
  return rig.capsules[side].map(c => {
    const h = restHead(m, c.bone);
    const t: V = c.child !== null
      ? restHead(m, c.child)
      : (() => {
        const r = m.restBonePoses, o = c.bone * 16;
        return [h[0] + r[o + 1] * c.tipLen, h[1] + r[o + 5] * c.tipLen, h[2] + r[o + 9] * c.tipLen] as V;
      })();
    return [posedPoint(bx, c.bone, h), posedPoint(bx, c.bone, t)];
  });
}

function pairOverlaps(rig: PhysicalRig, side: Side, bx: Float32Array): Map<string, number> {
  const caps = rig.capsules[side], segs = segments(rig, side, bx), out = new Map<string, number>();
  for (let i = 0; i < caps.length; i++) for (let j = i + 1; j < caps.length; j++) {
    if (caps[i].finger === caps[j].finger) continue;
    const d = segSegDistance(segs[i][0], segs[i][1], segs[j][0], segs[j][1]);
    out.set(`${i}-${j}`, Math.max(0, caps[i].radius + caps[j].radius - d) * 1000);
  }
  return out;
}

export function buildPhysicalRig(model: AnnyModel): PhysicalRig {
  const K = model.maxBonesPerVert, idx = (n: string) => model.boneLabels.indexOf(n);
  const dominant = new Int32Array(model.vertCount);
  for (let v = 0; v < model.vertCount; v++) {
    let bw = -1;
    for (let k = 0; k < K; k++) if (model.boneWeights[v * K + k] > bw) { bw = model.boneWeights[v * K + k]; dominant[v] = model.boneIndices[v * K + k]; }
  }
  const capsules = {} as Record<Side, Capsule[]>, faces = {} as Record<Side, number[]>;
  for (const s of ["L", "R"] as const) {
    const caps: Capsule[] = [];
    for (let f = 1; f <= 5; f++) for (let j = 1; j <= 3; j++) {
      const bone = idx(`finger${f}-${j}.${s}`);
      const child = j < 3 ? idx(`finger${f}-${j + 1}.${s}`) : null;
      // Radius: median distance from the bone axis of the vertices it dominates.
      const h = restHead(model, bone), r = model.restBonePoses, o = bone * 16;
      const y: V = [r[o + 1], r[o + 5], r[o + 9]];
      const ds: number[] = [];
      let tip = 0;
      for (let v = 0; v < model.vertCount; v++) {
        if (dominant[v] !== bone) continue;
        const p = sub([model.restVertices[v * 3], model.restVertices[v * 3 + 1], model.restVertices[v * 3 + 2]], h);
        const along = dot(p, y);
        tip = Math.max(tip, along);
        ds.push(Math.hypot(...cross(p, y)));
      }
      ds.sort((a, b) => a - b);
      // Shrink a little: the median surface distance overestimates the
      // collision core at the flat palm-side pads.
      caps.push({ finger: f, bone, child, tipLen: Math.max(0.005, tip - 0.004), radius: 0.8 * (ds[ds.length >> 1] ?? 0.006) });
    }
    capsules[s] = caps;
    const forearm = idx(`lowerarm01.${s}`);
    const inArm = (b: number) => { let p = b; while (p >= 0 && p !== forearm) p = model.boneParents[p]; return p === forearm; };
    const fs: number[] = [];
    for (let t = 0; t < model.faceCount; t++) {
      if ([0, 1, 2].every(k => inArm(dominant[model.faces[t * 3 + k]]))) fs.push(t);
    }
    faces[s] = fs;
  }
  const restArea = new Float32Array(model.faceCount);
  for (let t = 0; t < model.faceCount; t++) restArea[t] = triArea(model.restVertices, model.faces, t);
  const zero = { twist: 0, flex: 0, deviation: 0 };
  const rig: PhysicalRig = {
    model, capsules, faces, restArea, dominantBone: dominant,
    restOverlap: { L: new Map(), R: new Map() }, restArm: { L: zero, R: zero },
  };
  const bx = allocBoneTransforms(model.boneCount);
  forwardKinematics(model, new Array(model.boneCount).fill(null), bx);
  for (const s of ["L", "R"] as const) {
    rig.restOverlap[s] = pairOverlaps(rig, s, bx);
    rig.restArm[s] = armAngles(rig, bx, s);   // zero by construction; kept as a sanity anchor
  }
  return rig;
}

function triArea(v: Float32Array, f: Int32Array, t: number): number {
  const a = f[t * 3] * 3, b = f[t * 3 + 1] * 3, c = f[t * 3 + 2] * 3;
  const e1: V = [v[b] - v[a], v[b + 1] - v[a + 1], v[b + 2] - v[a + 2]];
  const e2: V = [v[c] - v[a], v[c + 1] - v[a + 1], v[c + 2] - v[a + 2]];
  return Math.hypot(...cross(e1, e2)) / 2;
}

function triNormal(v: Float32Array, f: Int32Array, t: number): V {
  const a = f[t * 3] * 3, b = f[t * 3 + 1] * 3, c = f[t * 3 + 2] * 3;
  const n = cross([v[b] - v[a], v[b + 1] - v[a + 1], v[b + 2] - v[a + 2]], [v[c] - v[a], v[c + 1] - v[a + 1], v[c + 2] - v[a + 2]]);
  const l = Math.hypot(...n) || 1;
  return [n[0] / l, n[1] / l, n[2] / l];
}

/**
 * Range of motion from anatomical neutral, degrees — the passive limits
 * (what a joint can be put in at all, not what people usually do):
 * pronation / supination ~95° (85° textbook active plus slack), wrist
 * extension and flexion 85° (a "stop" hand or a push-up reaches 80–90°),
 * ulnar deviation 40° / radial 25°.
 */
export const ARM_ROM = {
  twist: 95,
  flex: [-85, 85] as const,
  deviation: [-40, 25] as const,
};

export interface PhysicalScore {
  /** Summed finger-finger overlap beyond rest, mm. */
  penetrationMm: number;
  /** The part of `penetrationMm` involving the thumb. */
  thumbPenetrationMm: number;
  maxPenetrationMm: number;
  /** Fraction of forearm+hand triangles collapsed (< 25 % rest area). */
  collapsed: number;
  /** Fraction of forearm+hand triangles flipped against their bone's rotation. */
  flipped: number;
  /** Fraction of forearm+hand triangles whose area changed by more than ×1.5 / ÷1.5. */
  distorted: number;
  /** Forearm pronation(+)/supination(−) relative to rest, degrees. */
  forearmTwist: number;
  /** Wrist palmar flexion(+)/extension(−) and radial(+)/ulnar(−) deviation, degrees from rest. */
  wristFlex: number;
  wristDeviation: number;
  /** Degrees beyond forearm / wrist range of motion. */
  armRom: number;
}

export function physicalScore(rig: PhysicalRig, deltas: PoseDeltas, side: Side): PhysicalScore {
  const m = rig.model, bx = allocBoneTransforms(m.boneCount);
  forwardKinematics(m, deltas, bx);
  const over = pairOverlaps(rig, side, bx);
  let pen = 0, maxPen = 0, thumbPen = 0;
  for (const [k, o] of over) {
    const e = Math.max(0, o - (rig.restOverlap[side].get(k) ?? 0));
    pen += e; maxPen = Math.max(maxPen, e);
    const [i, j] = k.split("-").map(Number);
    if (rig.capsules[side][i].finger === 1 || rig.capsules[side][j].finger === 1) thumbPen += e;
  }
  const { vertices } = lbs(m, bx);
  let collapsed = 0, flipped = 0, distorted = 0;
  for (const t of rig.faces[side]) {
    const ratio = triArea(vertices, m.faces, t) / (rig.restArea[t] || 1e-12);
    if (ratio < 0.25) collapsed++;
    if (ratio < 1 / 1.5 || ratio > 1.5) distorted++;
    const n0 = triNormal(m.restVertices, m.faces, t);
    const b = rig.dominantBone[m.faces[t * 3]], o = b * 16;
    const expected: V = [
      bx[o] * n0[0] + bx[o + 1] * n0[1] + bx[o + 2] * n0[2],
      bx[o + 4] * n0[0] + bx[o + 5] * n0[1] + bx[o + 6] * n0[2],
      bx[o + 8] * n0[0] + bx[o + 9] * n0[1] + bx[o + 10] * n0[2],
    ];
    if (dot(triNormal(vertices, m.faces, t), expected) < 0) flipped++;
  }
  const n = rig.faces[side].length || 1;
  const arm = armAngles(rig, bx, side);
  const armRom = Math.max(0, Math.abs(arm.twist) - ARM_ROM.twist)
    + Math.max(0, arm.flex - ARM_ROM.flex[1], ARM_ROM.flex[0] - arm.flex)
    + Math.max(0, arm.deviation - ARM_ROM.deviation[1], ARM_ROM.deviation[0] - arm.deviation);
  return {
    penetrationMm: pen, thumbPenetrationMm: thumbPen, maxPenetrationMm: maxPen,
    collapsed: collapsed / n, flipped: flipped / n, distorted: distorted / n,
    forearmTwist: arm.twist, wristFlex: arm.flex, wristDeviation: arm.deviation, armRom,
  };
}

/**
 * Forearm twist and wrist angles in anatomical terms, from the posed bone
 * rotations — summed over whichever bones carry them, so they don't depend
 * on how a driver splits rotation between bones. Rest = 0 for all three; the
 * rig's rest forearm is mid-prone (palm facing the elbow's hinge axis), i.e.
 * anatomical neutral.
 *
 *   twist      rotation about the forearm axis of the forearm relative to the
 *              humerus plus of the hand relative to the forearm (each a
 *              swing–twist split; the elbow's flexion is a pure swing about
 *              that axis, so it can't leak in)
 *   flex       the hand's swing toward the palm (+) / back of the hand (−)
 *   deviation  the hand's swing toward the thumb (+) / little finger (−)
 */
function armAngles(rig: PhysicalRig, bx: Float32Array, side: Side): { twist: number; flex: number; deviation: number } {
  const m = rig.model, idx = (n: string) => m.boneLabels.indexOf(n);
  const humerus = idx(`upperarm02.${side}`), forearm = idx(`lowerarm02.${side}`), wrist = idx(`wrist.${side}`);
  const rest = (b: number): V => restHead(m, b);
  const axis = norm(sub(rest(wrist), rest(idx(`lowerarm01.${side}`))));
  // Rest hand frame (world): fingers forward, radial, palmar.
  const fwd = norm(sub(rest(idx(`finger3-1.${side}`)), rest(wrist)));
  const acrossRaw = sub(rest(idx(`finger2-1.${side}`)), rest(idx(`finger5-1.${side}`)));
  const radial = norm(sub(acrossRaw, fwd.map(x => x * dot(acrossRaw, fwd)) as V));
  const c = cross(fwd, radial);
  const palmar: V = side === "R" ? [-c[0], -c[1], -c[2]] : c;
  const forearmRel = mul(transpose(fromRest(bx, humerus)), fromRest(bx, forearm));
  const handRel = mul(transpose(fromRest(bx, forearm)), fromRest(bx, wrist));
  const handTwist = twistAbout(handRel, axis);
  const swing = rotVector(mul(handRel, axisRotation(axis, -handTwist)));
  const deg = 180 / Math.PI;
  return {
    twist: wrapDeg((twistAbout(forearmRel, axis) + handTwist) * deg),
    flex: dot(swing, norm(cross(fwd, palmar))) * deg,
    deviation: dot(swing, norm(cross(fwd, radial))) * deg,
  };
}

type M3 = number[];   // row-major 3×3

/** World rotation bone `b` underwent from rest: the rotation part of its skinning matrix. */
function fromRest(bx: Float32Array, b: number): M3 {
  const o = b * 16;
  return [bx[o], bx[o + 1], bx[o + 2], bx[o + 4], bx[o + 5], bx[o + 6], bx[o + 8], bx[o + 9], bx[o + 10]];
}

function mul(a: M3, b: M3): M3 {
  const o = new Array<number>(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return o;
}
const transpose = (a: M3): M3 => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];

function quat(m: M3): [number, number, number, number] {
  const w = Math.sqrt(Math.max(0, 1 + m[0] + m[4] + m[8])) / 2;
  const x = Math.sqrt(Math.max(0, 1 + m[0] - m[4] - m[8])) / 2 * Math.sign(m[7] - m[5] || 1);
  const y = Math.sqrt(Math.max(0, 1 - m[0] + m[4] - m[8])) / 2 * Math.sign(m[2] - m[6] || 1);
  const z = Math.sqrt(Math.max(0, 1 - m[0] - m[4] + m[8])) / 2 * Math.sign(m[3] - m[1] || 1);
  return [x, y, z, w];
}

/** Twist angle (rad) of rotation `m` about unit `axis` (swing–twist). */
function twistAbout(m: M3, axis: V): number {
  const [x, y, z, w] = quat(m);
  return 2 * Math.atan2(x * axis[0] + y * axis[1] + z * axis[2], w);
}

function rotVector(m: M3): V {
  const [x, y, z, w] = quat(m);
  const s = Math.hypot(x, y, z);
  if (s < 1e-12) return [0, 0, 0];
  const a = 2 * Math.atan2(s, w);
  return [x / s * a, y / s * a, z / s * a];
}

function axisRotation(a: V, t: number): M3 {
  const c = Math.cos(t), s = Math.sin(t), C = 1 - c;
  return [
    c + a[0] * a[0] * C, a[0] * a[1] * C - a[2] * s, a[0] * a[2] * C + a[1] * s,
    a[1] * a[0] * C + a[2] * s, c + a[1] * a[1] * C, a[1] * a[2] * C - a[0] * s,
    a[2] * a[0] * C - a[1] * s, a[2] * a[1] * C + a[0] * s, c + a[2] * a[2] * C,
  ];
}

const wrapDeg = (d: number): number => ((d + 540) % 360) - 180;

const norm = (a: V): V => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
