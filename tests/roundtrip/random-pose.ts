/** Seeded random but anatomically plausible full-body poses, built directly on the rig. */

import { identityDeltas, rodriguesToMat3, type AnnyModel, type PoseDeltas } from "../../src/anny/index.js";
import { handPoseDeltas, rng, type HandPoseSpec } from "./synthetic.js";

const DEG = Math.PI / 180;
type Axis = [number, number, number];

function rot(axis: Axis, deg: number): Float32Array {
  const l = Math.hypot(...axis);
  return rodriguesToMat3(new Float32Array(axis.map(x => x / l)), deg * DEG);
}

function mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}

/**
 * A random pose: body yaw, mild spine/neck bends, free shoulders and hips,
 * elbows/knees flexed about their hinge (+local X for both in this rig),
 * forearm twist, wrist flex/deviation and natural random hands.
 */
export function randomPose(model: AnnyModel, boneIndex: Map<string, number>, seed: number): PoseDeltas {
  const r = rng(seed);
  const u = (lo: number, hi: number) => lo + (hi - lo) * r();
  const axis = (): Axis => [u(-1, 1), u(-1, 1), u(-1, 1)];
  const hand = (side: "L" | "R"): HandPoseSpec => {
    const base = r();
    return {
      name: "", side,
      curl: [0, 1, 2, 3].map(i => Math.min(1, Math.max(0, base + u(-0.2, 0.2) + i * 0.03))),
      spread: [u(-6, 0), 0, u(0, 5), u(0, 9)],
      thumb: [u(0, 30), u(-10, 10), u(0, 50) * base, u(0, 60) * base],
      wrist: [0, 0],
      view: { axis: [0, 0, 1], deg: 0 },
    };
  };
  // Start from the two hands (they also set wrist + root; overwritten below).
  const d = identityDeltas(model.boneCount);
  for (const side of ["L", "R"] as const) {
    const h = handPoseDeltas(hand(side), boneIndex, model.boneCount);
    h.forEach((m, i) => { if (m) d[i] = m; });
  }
  const set = (n: string, m: Float32Array) => {
    const i = boneIndex.get(n);
    if (i === undefined) throw new Error(`no bone ${n}`);
    d[i] = m;
  };
  set("root", mul(rot([0, 0, 1], u(-180, 180)), rot(axis(), u(0, 12))));
  for (const b of ["spine05", "spine04", "spine03", "spine02", "spine01"]) set(b, rot(axis(), u(0, 8)));
  for (const b of ["neck01", "neck02", "neck03", "head"]) set(b, rot(axis(), u(0, 10)));
  for (const s of ["L", "R"] as const) {
    set(`upperarm01.${s}`, rot(axis(), u(10, 100)));
    set(`lowerarm01.${s}`, mul(rot([1, 0, 0], u(0, 110)), rot([0, 1, 0], u(-40, 40))));
    set(`lowerarm02.${s}`, rot([0, 1, 0], u(-40, 40)));
    set(`wrist.${s}`, mul(rot([1, 0, 0], u(-45, 45)), rot([0, 0, 1], u(-15, 15))));
    set(`upperleg01.${s}`, rot(axis(), u(0, 45)));
    set(`lowerleg01.${s}`, rot([1, 0, 0], u(0, 90)));
    set(`foot.${s}`, rot([1, 0, 0], u(-20, 20)));
  }
  return d;
}
