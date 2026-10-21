/**
 * Labelled hand poses for validating the naturalness metrics: plausible
 * hands (coupled, synergistic curls; named gestures) and several families of
 * implausible ones. Built directly as rig rotations (each phalanx flexes
 * about its local X), never through the driver.
 */

import { identityDeltas, rodriguesToMat3, type PoseDeltas } from "../../src/anny/index.js";
import { rng } from "../roundtrip/synthetic.js";
import type { Side } from "../roundtrip/metrics.js";

export interface JointPose {
  /** Optional forearm twist (about the forearm, degrees) and wrist [flex, deviation]. */
  forearmTwist?: number;
  wrist?: [number, number];
  /** Rotation about local X per [finger 2..5][MCP, PIP, DIP], degrees. */
  flex: number[][];
  /** Rotation about local Z per finger 2..5 (degrees, + = mirrored per side by caller). */
  abd: number[];
  /** Thumb [metacarpal X, metacarpal Z, MCP X, IP X], degrees. */
  thumb: number[];
}

export type PoseClass = "natural" | "independent" | "noisy" | "hook" | "hyperextended" | "crossed" | "twisted";

export interface LabelledPose { cls: PoseClass; name: string; side: Side; pose: JointPose }

const DEG = Math.PI / 180;
const rot = (axis: [number, number, number], deg: number) => rodriguesToMat3(new Float32Array(axis), deg * DEG);
function mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}

/** Hand pose deltas; the whole body is turned so the hand points up, palm to camera. */
export function jointPoseDeltas(p: JointPose, side: Side, boneIndex: Map<string, number>, boneCount: number): PoseDeltas {
  const d = identityDeltas(boneCount);
  const set = (n: string, m: Float32Array) => { d[boneIndex.get(n)!] = m; };
  const zSign = side === "R" ? 1 : -1;   // abduction axis is mirrored between hands
  [2, 3, 4, 5].forEach((f, i) => {
    set(`finger${f}-1.${side}`, mul(rot([1, 0, 0], p.flex[i][0]), rot([0, 0, 1], zSign * p.abd[i])));
    set(`finger${f}-2.${side}`, rot([1, 0, 0], p.flex[i][1]));
    set(`finger${f}-3.${side}`, rot([1, 0, 0], p.flex[i][2]));
  });
  set(`finger1-1.${side}`, mul(rot([1, 0, 0], p.thumb[0]), rot([0, 0, 1], zSign * p.thumb[1])));
  set(`finger1-2.${side}`, rot([1, 0, 0], p.thumb[2]));
  set(`finger1-3.${side}`, rot([1, 0, 0], p.thumb[3]));
  if (p.forearmTwist) {
    set(`lowerarm01.${side}`, rot([0, 1, 0], p.forearmTwist / 2));
    set(`lowerarm02.${side}`, rot([0, 1, 0], p.forearmTwist / 2));
  }
  if (p.wrist) set(`wrist.${side}`, mul(rot([1, 0, 0], p.wrist[0]), rot([0, 0, 1], p.wrist[1])));
  set("root", rot([0, 1, 0], 180));
  return d;
}

/** Plausible: one curl level with mild finger cascade, ⅔ DIP coupling, spread shrinking with curl. */
function natural(r: () => number): JointPose {
  const base = r();
  const flex = [0, 1, 2, 3].map(i => {
    const c = Math.min(1, Math.max(0, base + (r() - 0.5) * 0.25 + i * 0.04));
    const pip = 100 * c;
    return [80 * c + (r() - 0.5) * 8, pip, (2 / 3) * pip + (r() - 0.5) * 10];
  });
  const fan = (1 - base) * (r() * 8);
  return { flex, abd: [-fan, 0, fan * 0.6, fan * 1.2], thumb: [10 + 25 * base, (r() - 0.5) * 10, 35 * base, 45 * base] };
}

const GESTURES: Record<string, JointPose> = {
  open: { flex: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], abd: [-6, 0, 4, 8], thumb: [0, 0, 0, 0] },
  relaxed: { flex: [[15, 25, 15], [20, 30, 20], [25, 35, 22], [30, 40, 25]], abd: [-3, 0, 2, 4], thumb: [15, 0, 10, 15] },
  fist: { flex: [[80, 100, 65], [85, 105, 70], [85, 105, 70], [85, 100, 65]], abd: [0, 0, 0, 0], thumb: [25, 0, 40, 50] },
  point: { flex: [[0, 0, 0], [85, 105, 70], [85, 105, 70], [85, 100, 65]], abd: [0, 0, 0, 0], thumb: [25, 0, 40, 40] },
  victory: { flex: [[0, 0, 0], [0, 0, 0], [85, 105, 70], [85, 100, 65]], abd: [-8, 8, 0, 0], thumb: [25, 0, 40, 40] },
};

export function labelledPoses(seed = 3, nPerClass = 16): LabelledPose[] {
  const r = rng(seed);
  const out: LabelledPose[] = [];
  const side = (): Side => (r() < 0.5 ? "L" : "R");
  for (const [name, pose] of Object.entries(GESTURES)) for (const s of ["L", "R"] as const) out.push({ cls: "natural", name, side: s, pose });
  for (let k = 0; k < nPerClass; k++) out.push({ cls: "natural", name: `nat${k}`, side: side(), pose: natural(r) });
  for (let k = 0; k < nPerClass; k++) {
    out.push({
      cls: "independent", name: `ind${k}`, side: side(),
      pose: { flex: [0, 1, 2, 3].map(() => [r() * 90, r() * 110, r() * 80]), abd: [0, 0, 0, 0].map(() => (r() - 0.5) * 20), thumb: [r() * 40, (r() - 0.5) * 20, r() * 60, r() * 70] },
    });
    const n = natural(r);
    out.push({
      cls: "noisy", name: `noisy${k}`, side: side(),
      pose: { flex: n.flex.map(f => f.map(a => a + (r() - 0.5) * 70)), abd: n.abd.map(a => a + (r() - 0.5) * 30), thumb: n.thumb.map(a => a + (r() - 0.5) * 40) },
    });
    out.push({
      cls: "hook", name: `hook${k}`, side: side(),
      pose: { flex: [0, 1, 2, 3].map(() => [r() * 35, 90 + r() * 25, (r() - 0.5) * 20]), abd: [0, 0, 0, 0], thumb: n.thumb },
    });
    const h = natural(r);
    const bad = Math.floor(r() * 4);
    h.flex[bad] = [-(20 + r() * 25), -(15 + r() * 20), -(15 + r() * 20)];
    out.push({ cls: "hyperextended", name: `hyper${k}`, side: side(), pose: h });
    const c = natural(r);
    out.push({ cls: "crossed", name: `cross${k}`, side: side(), pose: { ...c, abd: [25 + r() * 10, -(20 + r() * 10), 25 + r() * 10, -(20 + r() * 10)] } });
    const t = natural(r);
    out.push({
      cls: "twisted", name: `twist${k}`, side: side(),
      pose: { ...t, forearmTwist: (r() < 0.5 ? -1 : 1) * (150 + r() * 60), wrist: [(r() < 0.5 ? -1 : 1) * (70 + r() * 30), (r() - 0.5) * 60] },
    });
  }
  return out;
}
