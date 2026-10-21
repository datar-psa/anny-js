/**
 * Ground-truth hands for measuring MediaPipe's error as a function of hand
 * size, and for scoring fusion of its readings with a natural-hand prior.
 *
 * Broader than the synthetic report's set on purpose: every common gesture
 * family (fist, point, V, claw, flat-hook, pinch, grasp, rock, call-me,
 * thumbs-up, spread) plus random natural hands, each seen from several sides.
 * Built directly as rig rotations — never through the driver.
 */

import { identityDeltas, rodriguesToMat3, type PoseDeltas } from "../../src/anny/index.js";
import type { Side } from "../roundtrip/metrics.js";
import { rng } from "../roundtrip/synthetic.js";
import type { JointPose } from "./poses.js";

type Axis = [number, number, number];

export interface GtHand {
  name: string;
  /** Gesture family ("random" for the random natural hands). */
  family: string;
  side: Side;
  pose: JointPose;
  view: { axis: Axis; deg: number };
}

const O = [0, 0, 0], F = (k = 1) => [80 * k, 100 * k, 67 * k];
const R = [15, 25, 15];
/** Flat: straighter than the rig's rest hand, whose knuckles are ~15° curled. */
const FLAT = [-15, -5, -5];
/** Curled fingers stay side by side: without a little spread the rig's curl axes converge them into each other. */
const SIDE_BY_SIDE = [-3, 0, 2, 4];

export const GESTURES: Record<string, JointPose> = {
  flat: { flex: [FLAT, FLAT, FLAT, FLAT], abd: [-4, 0, 3, 6], thumb: [-10, 0, 0, 0] },
  open: { flex: [O, O, O, O], abd: [-6, 0, 4, 8], thumb: [0, 0, 0, 0] },
  spread: { flex: [FLAT, O, O, FLAT], abd: [-14, 0, 9, 18], thumb: [-15, 0, 0, 0] },
  relaxed: { flex: [[15, 25, 15], [20, 30, 20], [25, 35, 22], [30, 40, 25]], abd: [-3, 0, 2, 4], thumb: [15, 0, 10, 15] },
  fist: { flex: [F(), F(1.05), F(1.05), F()], abd: SIDE_BY_SIDE, thumb: [20, 0, 40, 50] },
  point: { flex: [O, F(), F(), F()], abd: SIDE_BY_SIDE, thumb: [25, 0, 40, 40] },
  victory: { flex: [O, O, F(), F()], abd: [-8, 8, 2, 4], thumb: [25, 0, 40, 40] },
  three: { flex: [O, O, O, F()], abd: [-5, 0, 4, 4], thumb: [25, 0, 40, 40] },
  rock: { flex: [O, F(), F(), O], abd: [-5, 0, 2, 10], thumb: [20, 0, 30, 30] },
  callMe: { flex: [F(), F(), F(), O], abd: [-3, 0, 2, 12], thumb: [-15, 0, 0, 0] },
  thumbsUp: { flex: [F(), F(1.05), F(1.05), F()], abd: SIDE_BY_SIDE, thumb: [-10, 0, 0, 0] },
  claw: { flex: [[5, 90, 60], [5, 95, 63], [5, 95, 63], [5, 90, 60]], abd: [-4, 0, 3, 6], thumb: [20, 0, 30, 30] },
  flatHook: { flex: [[80, 5, 3], [83, 5, 3], [83, 5, 3], [80, 5, 3]], abd: SIDE_BY_SIDE, thumb: [10, 0, 10, 10] },
  pinch: { flex: [[45, 55, 35], R, R, R], abd: [0, 0, 2, 4], thumb: [0, 0, 40, 50] },
  grasp: { flex: [[45, 60, 40], [50, 65, 43], [55, 65, 43], [55, 60, 40]], abd: SIDE_BY_SIDE, thumb: [20, 0, 20, 20] },
};

/**
 * Rough everyday frequency of each family, for weighting the natural-hand
 * prior (tools/hand_trust.ts): hands mostly hang relaxed or open, hold
 * things, and only now and then make signs. A modelling choice, not a
 * measurement — it sets the prior's mean, i.e. what a hand too small to read
 * is assumed to be doing.
 */
export const FAMILY_FREQUENCY: Record<string, number> = {
  flat: 2, open: 2, relaxed: 4, spread: 0.5, random: 6,
  fist: 1.5, grasp: 1.5, pinch: 1, claw: 0.5, flatHook: 0.5,
  point: 0.5, victory: 0.5, three: 0.3, rock: 0.3, callMe: 0.3, thumbsUp: 0.5,
};

/** Views of a hanging hand: thumb side, back/palm (±90° yaw), fingers up. */
const GT_VIEWS: ReadonlyArray<{ axis: Axis; deg: number }> = [
  { axis: [0, 0, 1], deg: 0 },
  { axis: [0, 0, 1], deg: 90 },
  { axis: [0, 0, 1], deg: -90 },
  { axis: [0, 1, 0], deg: 180 },
];

/**
 * A random natural hand: a shared curl level (from slightly hyperextended to
 * a full fist), per-finger departures that are sometimes large (a finger
 * left out, as in gestures), MCP:PIP balance from claw to flat-hook, DIP
 * coupled to PIP with noise, spread closing as the fingers curl.
 */
function randomNatural(r: () => number): JointPose {
  const base = -0.15 + 1.15 * r();
  const balance = 0.6 + 0.8 * r();          // MCP share relative to PIP (claw ← 1 → hook)
  const flex = [0, 1, 2, 3].map(() => {
    const c = r() < 0.25 ? -0.15 + 1.15 * r() : Math.min(1, Math.max(-0.15, base + (r() - 0.5) * 0.3));
    if (c < 0) return [100 * c, 30 * c, 30 * c];            // straighter than rest: mostly at the knuckle
    const pip = 100 * c * Math.min(1, 2 - balance);
    return [Math.min(95, 80 * c * balance), pip, (2 / 3) * pip + (r() - 0.5) * 16];
  });
  const fan = (1 - Math.max(0, base)) * (r() * 12 - 2) + 3 * Math.max(0, base);
  return {
    flex,
    abd: [-fan, 0, 0.6 * fan, 1.2 * fan].map(a => a + (r() - 0.5) * 4),
    thumb: [r() * 35 - 5, (r() - 0.5) * 20, 45 * Math.max(0, base) * r(), 55 * Math.max(0, base) * r()],
    wrist: [(r() - 0.5) * 50, (r() - 0.5) * 30],
  };
}

export function gtHands(nRandom = 64, seed = 11): GtHand[] {
  const out: GtHand[] = [];
  for (const [family, pose] of Object.entries(GESTURES)) {
    GT_VIEWS.forEach((view, v) => {
      for (const side of ["L", "R"] as const) out.push({ name: `${family}-v${v}-${side}`, family, side, pose, view });
    });
  }
  const r = rng(seed);
  for (let k = 0; k < nRandom; k++) {
    const v = GT_VIEWS[k % GT_VIEWS.length];
    out.push({
      name: `rand${k}`, family: "random", side: r() < 0.5 ? "L" : "R",
      pose: randomNatural(r), view: { axis: v.axis, deg: v.deg + (r() - 0.5) * 60 },
    });
  }
  return out;
}

const DEG = Math.PI / 180;
const rot = (axis: Axis, deg: number) => {
  const l = Math.hypot(...axis);
  return rodriguesToMat3(new Float32Array(axis.map(x => x / l)), deg * DEG);
};
function mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return o;
}

/** Rig deltas for a ground-truth hand (fingers about local X, abduction about mirrored local Z). */
export function gtHandDeltas(h: GtHand, boneIndex: Map<string, number>, boneCount: number): PoseDeltas {
  const d = identityDeltas(boneCount);
  const set = (n: string, m: Float32Array) => {
    const i = boneIndex.get(n);
    if (i === undefined) throw new Error(`no bone ${n}`);
    d[i] = m;
  };
  const { pose: p, side } = h;
  const z = side === "R" ? 1 : -1;
  [2, 3, 4, 5].forEach((f, i) => {
    set(`finger${f}-1.${side}`, mul(rot([1, 0, 0], p.flex[i][0]), rot([0, 0, 1], z * p.abd[i])));
    set(`finger${f}-2.${side}`, rot([1, 0, 0], p.flex[i][1]));
    set(`finger${f}-3.${side}`, rot([1, 0, 0], p.flex[i][2]));
  });
  set(`finger1-1.${side}`, mul(rot([1, 0, 0], p.thumb[0]), rot([0, 0, 1], z * p.thumb[1])));
  set(`finger1-2.${side}`, rot([1, 0, 0], p.thumb[2]));
  set(`finger1-3.${side}`, rot([1, 0, 0], p.thumb[3]));
  if (p.wrist) set(`wrist.${side}`, mul(rot([1, 0, 0], p.wrist[0]), rot([0, 0, 1], p.wrist[1])));
  set("root", rot(h.view.axis, h.view.deg));
  return d;
}
