/**
 * Posed-skeleton queries + "virtual MediaPipe landmarks" read off the Anny rig.
 *
 * The round-trip validator compares landmarks from three sources with ONE
 * metric implementation: MediaPipe on a real image, MediaPipe on a render of
 * the driven model, and the driven model's own joints. This module provides
 * the third: it reads joint positions from the FK output and lays them out in
 * MediaPipe's 33-pose / 21-hand index order and MediaPipe world axes.
 */

import type { AnnyModel } from "../../src/anny/index.js";
import { MP, MP_HAND } from "../../src/mediapipe/index.js";
import { add, dot, norm, scale, sub, type V3 } from "./vec.js";

/** Anny world (+x left, +y back, +z up) → MP world (+x left, +y down, +z away). */
export const annyToMp = (p: V3): V3 => [p[0], -p[2], p[1]];
/** MP world → Anny world. */
export const mpToAnnyV = (p: V3): V3 => [p[0], p[2], -p[1]];

export type Side = "L" | "R";

/** Rest-pose geometry that FK doesn't carry: distal bone lengths. */
export interface RigExtras {
  boneIndex: Map<string, number>;
  /** Rest length along local +Y for leaf-ish bones (finger tips, toes, head). */
  tipLength: Map<string, number>;
}

/**
 * Estimate where a bone "ends" from the mesh: the furthest projection along
 * the bone's rest +Y of vertices that are mostly skinned to it. Used for
 * finger tips — the rig has no tail joints.
 */
function estimateTipLength(model: AnnyModel, b: number): number {
  const p = b * 16, r = model.restBonePoses;
  const head: V3 = [r[p+3], r[p+7], r[p+11]];
  const y: V3 = [r[p+1], r[p+5], r[p+9]];
  const K = model.maxBonesPerVert;
  let best = 0;
  for (let v = 0; v < model.vertCount; v++) {
    let w = 0;
    for (let k = 0; k < K; k++) {
      if (model.boneIndices[v*K+k] === b) w += model.boneWeights[v*K+k];
    }
    if (w < 0.5) continue;
    const pv: V3 = [model.restVertices[v*3], model.restVertices[v*3+1], model.restVertices[v*3+2]];
    best = Math.max(best, dot(sub(pv, head), y));
  }
  return best;
}

const TIP_BONES = [1, 2, 3, 4, 5].flatMap(f => [`finger${f}-3.L`, `finger${f}-3.R`]);

export function buildRigExtras(model: AnnyModel): RigExtras {
  const boneIndex = new Map<string, number>();
  model.boneLabels.forEach((n, i) => boneIndex.set(n, i));
  const tipLength = new Map<string, number>();
  for (const name of TIP_BONES) {
    const b = boneIndex.get(name);
    if (b !== undefined) tipLength.set(name, estimateTipLength(model, b));
  }
  return { boneIndex, tipLength };
}

/** Read posed joint positions / axes from forwardKinematics output. */
export class PosedSkeleton {
  constructor(
    private model: AnnyModel,
    private extras: RigExtras,
    private boneXforms: Float32Array,
  ) {}

  private idx(name: string): number {
    const i = this.extras.boneIndex.get(name);
    if (i === undefined) throw new Error(`unknown bone ${name}`);
    return i;
  }

  /** Posed bone head (joint) position, Anny world. */
  head(name: string): V3 {
    const i = this.idx(name), x = this.boneXforms, o = i * 16;
    const r = this.model.restBonePoses;
    const rx = r[o+3], ry = r[o+7], rz = r[o+11];
    return [
      x[o]*rx   + x[o+1]*ry + x[o+2]*rz  + x[o+3],
      x[o+4]*rx + x[o+5]*ry + x[o+6]*rz  + x[o+7],
      x[o+8]*rx + x[o+9]*ry + x[o+10]*rz + x[o+11],
    ];
  }

  /** Posed bone local axis `col` (0=X, 1=Y along bone, 2=Z) in Anny world. */
  axis(name: string, col: 0 | 1 | 2): V3 {
    const i = this.idx(name), x = this.boneXforms, o = i * 16;
    const r = this.model.restBonePoses;
    const a: V3 = [r[o+col], r[o+4+col], r[o+8+col]];
    return norm([
      x[o]*a[0]   + x[o+1]*a[1] + x[o+2]*a[2],
      x[o+4]*a[0] + x[o+5]*a[1] + x[o+6]*a[2],
      x[o+8]*a[0] + x[o+9]*a[1] + x[o+10]*a[2],
    ]);
  }

  /** Rest-pose local axis `col` of a bone in Anny world. */
  restAxis(name: string, col: 0 | 1 | 2): V3 {
    const o = this.idx(name) * 16, r = this.model.restBonePoses;
    return [r[o + col], r[o + 4 + col], r[o + 8 + col]];
  }

  /** Posed tip of a distal bone (head + Y · restLength). */
  tip(name: string): V3 {
    const L = this.extras.tipLength.get(name);
    if (L === undefined) throw new Error(`no tip length for ${name}`);
    return add(this.head(name), scale(this.axis(name, 1), L));
  }
}

/** 21 hand joints in MP_HAND order, Anny world. */
export function handJointsAnny(sk: PosedSkeleton, side: Side): V3[] {
  const s = `.${side}`;
  const out: V3[] = new Array(21);
  out[MP_HAND.WRIST] = sk.head(`wrist${s}`);
  // Thumb: finger1-1 is the thumb metacarpal (CMC→MCP).
  out[MP_HAND.THUMB_CMC] = sk.head(`finger1-1${s}`);
  out[MP_HAND.THUMB_MCP] = sk.head(`finger1-2${s}`);
  out[MP_HAND.THUMB_IP]  = sk.head(`finger1-3${s}`);
  out[MP_HAND.THUMB_TIP] = sk.tip(`finger1-3${s}`);
  const chains: ReadonlyArray<[number, number]> = [
    [2, MP_HAND.INDEX_MCP], [3, MP_HAND.MIDDLE_MCP], [4, MP_HAND.RING_MCP], [5, MP_HAND.PINKY_MCP],
  ];
  for (const [f, base] of chains) {
    out[base]     = sk.head(`finger${f}-1${s}`);
    out[base + 1] = sk.head(`finger${f}-2${s}`);
    out[base + 2] = sk.head(`finger${f}-3${s}`);
    out[base + 3] = sk.tip(`finger${f}-3${s}`);
  }
  return out;
}

/**
 * The subset of the 33 MP pose landmarks that has a clean rig analogue
 * (limbs, torso, coarse hand). Face landmarks are left undefined.
 */
export function poseJointsAnny(sk: PosedSkeleton): (V3 | undefined)[] {
  const out: (V3 | undefined)[] = new Array(33).fill(undefined);
  for (const side of ["L", "R"] as const) {
    const L = side === "L";
    const s = `.${side}`;
    out[L ? MP.LEFT_SHOULDER : MP.RIGHT_SHOULDER] = sk.head(`upperarm01${s}`);
    out[L ? MP.LEFT_ELBOW : MP.RIGHT_ELBOW]       = sk.head(`lowerarm01${s}`);
    out[L ? MP.LEFT_WRIST : MP.RIGHT_WRIST]       = sk.head(`wrist${s}`);
    out[L ? MP.LEFT_INDEX : MP.RIGHT_INDEX]       = sk.head(`finger2-1${s}`);
    out[L ? MP.LEFT_PINKY : MP.RIGHT_PINKY]       = sk.head(`finger5-1${s}`);
    out[L ? MP.LEFT_THUMB : MP.RIGHT_THUMB]       = sk.head(`finger1-2${s}`);
    out[L ? MP.LEFT_HIP : MP.RIGHT_HIP]           = sk.head(`upperleg01${s}`);
    out[L ? MP.LEFT_KNEE : MP.RIGHT_KNEE]         = sk.head(`lowerleg01${s}`);
    out[L ? MP.LEFT_ANKLE : MP.RIGHT_ANKLE]       = sk.head(`foot${s}`);
    out[L ? MP.LEFT_FOOT_INDEX : MP.RIGHT_FOOT_INDEX] = sk.head(`toe2-1${s}`);
  }
  return out;
}

/** Convert a list of Anny-world points to MP-world landmark objects. */
export function toMpWorld(points: V3[]): { x: number; y: number; z: number }[] {
  return points.map(p => {
    const [x, y, z] = annyToMp(p);
    return { x, y, z };
  });
}
