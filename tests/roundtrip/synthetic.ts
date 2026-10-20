/**
 * Synthetic ground truth: pose the rig directly with known joint angles,
 * render it, run MediaPipe, drive the rig from MediaPipe's reading, and
 * compare everything against the known pose.
 *
 *   gt      the rig posed by the generator (ground truth by construction)
 *   mp      MediaPipe on the render of gt         → mp↔gt     = MediaPipe's own error on renders
 *   driven  the driver's output from mp           → driven↔gt = whole-pipeline error
 *
 * Poses are built WITHOUT the driver (plain axis-angle deltas on the rig's
 * hinge axes) so the generator can't share a bug with the code under test.
 */

import { identityDeltas, rodriguesToMat3, type PoseDeltas } from "../../src/anny/index.js";
import { project } from "./camera.js";
import {
  compare2D, compareHands, handAngles, toPx, type HandAngles, type HandDiff, type P2, type Side,
} from "./metrics.js";
import { handCamera, handPoints, type Driver, type RunContext } from "./run.js";
import { handJointsAnny, PosedSkeleton, toMpWorld } from "./skeleton.js";
import { allocBoneTransforms, forwardKinematics } from "../../src/anny/index.js";

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const DEG = Math.PI / 180;
const rot = (axis: [number, number, number], deg: number) => rodriguesToMat3(new Float32Array(axis), deg * DEG);

function mul3(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return o;
}

export interface HandPoseSpec {
  name: string;
  side: Side;
  /** Per finger (index..pinky) curl in [0, 1]: 0 = open, 1 = fist. */
  curl: number[];
  /** Per finger MCP abduction (deg, rig local Z). */
  spread: number[];
  /** Thumb: [metacarpal about X, metacarpal about Z, MCP, IP] (deg). */
  thumb: number[];
  /** Wrist [flex about X, deviation about Z] (deg). */
  wrist: [number, number];
  /** Whole-body rotation (axis, deg) — sets the hand's view direction. */
  view: { axis: [number, number, number]; deg: number };
}

type Axis = [number, number, number];
/** Root rotations that present the hanging hand to the camera from different sides. */
const VIEWS: ReadonlyArray<{ axis: Axis; deg: number }> = [
  { axis: [0, 0, 1], deg: 0 },     // hand hanging, thumb side toward camera
  { axis: [0, 0, 1], deg: 90 },    // back / palm toward camera
  { axis: [0, 0, 1], deg: -90 },
  { axis: [0, 1, 0], deg: 180 },   // upside down: fingers point up
  { axis: [1, 1, 0], deg: 180 },
  { axis: [1, 0, 0], deg: -60 },   // leaning toward camera
];

/** Named gestures × views × sides, plus seeded random natural hands. */
export function handPoseSpecs(nRandom: number, seed = 7): HandPoseSpec[] {
  const named: Record<string, Omit<HandPoseSpec, "side" | "view" | "name">> = {
    open:    { curl: [0, 0, 0, 0], spread: [-6, 0, 4, 8], thumb: [0, 0, 0, 0], wrist: [0, 0] },
    fist:    { curl: [1, 1, 1, 1], spread: [0, 0, 0, 0], thumb: [20, 0, 40, 50], wrist: [0, 0] },
    point:   { curl: [0, 1, 1, 1], spread: [0, 0, 0, 0], thumb: [25, 0, 40, 40], wrist: [0, 0] },
    victory: { curl: [0, 0, 1, 1], spread: [-8, 8, 0, 0], thumb: [25, 0, 40, 40], wrist: [0, 0] },
    half:    { curl: [0.5, 0.5, 0.5, 0.5], spread: [0, 0, 0, 0], thumb: [10, 0, 20, 20], wrist: [30, 0] },
  };
  const out: HandPoseSpec[] = [];
  for (const [name, n] of Object.entries(named)) {
    VIEWS.forEach((view, j) => {
      for (const side of ["L", "R"] as const) out.push({ ...n, name: `${name}-v${j}-${side}`, side, view });
    });
  }
  const r = rng(seed);
  for (let k = 0; k < nRandom; k++) {
    // Natural hands: one overall curl with mild per-finger variation that
    // grows toward the pinky, small correlated spread.
    const base = r();
    const curl = [0, 1, 2, 3].map(i => Math.min(1, Math.max(0, base + (r() - 0.5) * 0.4 + i * 0.03)));
    const fan = (r() - 0.3) * 8;
    out.push({
      name: `rand${k}`,
      side: r() < 0.5 ? "L" : "R",
      curl,
      spread: [-fan, 0, fan * 0.6, fan * 1.2],
      thumb: [r() * 30, (r() - 0.5) * 20, r() * 50 * base, r() * 60 * base],
      wrist: [(r() - 0.5) * 60, (r() - 0.5) * 30],
      view: { ...VIEWS[k % VIEWS.length], deg: VIEWS[k % VIEWS.length].deg + (r() - 0.5) * 60 },
    });
  }
  return out;
}

/** Ground-truth deltas for a hand spec (fingers flex about their local X — the rig's hinge). */
export function handPoseDeltas(spec: HandPoseSpec, boneIndex: Map<string, number>, boneCount: number): PoseDeltas {
  const d = identityDeltas(boneCount);
  const set = (n: string, m: Float32Array) => {
    const i = boneIndex.get(n);
    if (i === undefined) throw new Error(`no bone ${n}`);
    d[i] = m;
  };
  const s = spec.side;
  // Natural curl split MCP : PIP : DIP for a fully closed finger.
  const FULL = [80, 100, 70];
  [2, 3, 4, 5].forEach((f, i) => {
    set(`finger${f}-1.${s}`, mul3(rot([1, 0, 0], FULL[0] * spec.curl[i]), rot([0, 0, 1], spec.spread[i])));
    set(`finger${f}-2.${s}`, rot([1, 0, 0], FULL[1] * spec.curl[i]));
    set(`finger${f}-3.${s}`, rot([1, 0, 0], FULL[2] * spec.curl[i]));
  });
  const [tmx, tmz, tmcp, tip] = spec.thumb;
  set(`finger1-1.${s}`, mul3(rot([1, 0, 0], tmx), rot([0, 0, 1], tmz)));
  set(`finger1-2.${s}`, rot([1, 0, 0], tmcp));
  set(`finger1-3.${s}`, rot([1, 0, 0], tip));
  set(`wrist.${s}`, mul3(rot([1, 0, 0], spec.wrist[0]), rot([0, 0, 1], spec.wrist[1])));
  set("root", rodriguesToMat3(new Float32Array(spec.view.axis.map(x => x / Math.hypot(...spec.view.axis))), spec.view.deg * DEG));
  return d;
}

export interface SyntheticHandResult {
  spec: HandPoseSpec;
  gt: HandAngles;
  mp?: HandAngles;
  driven?: HandAngles;
  mpVsGt?: HandDiff;
  drivenVsGt?: HandDiff;
  /** 2D residuals vs the projected ground truth (fraction of palm length). */
  mpVsGt2D?: number;
  drivenVsGt2D?: number;
  /** MediaPipe's handedness label on the render (gt side is known). */
  mpLabel?: Side;
  /** MediaPipe image landmarks on the render (for overlays). */
  mpImage?: { x: number; y: number }[];
  png: string;
}

function handOf(ctx: RunContext, deltas: PoseDeltas, side: Side) {
  const bx = allocBoneTransforms(ctx.model.boneCount);
  forwardKinematics(ctx.model, deltas, bx);
  return handJointsAnny(new PosedSkeleton(ctx.model, ctx.extras, bx), side);
}

export async function runSyntheticHand(ctx: RunContext, spec: HandPoseSpec, driver: Driver): Promise<SyntheticHandResult> {
  const deltas = handPoseDeltas(spec, ctx.boneIndex, ctx.model.boneCount);
  const joints = handOf(ctx, deltas, spec.side);
  const gt = handAngles(toMpWorld(joints), spec.side);
  const cam = handCamera(ctx, deltas, spec.side);
  const { detection, png } = await ctx.client.renderAndDetect(deltas, cam, { handOnly: spec.side }, { pose: false });
  const res: SyntheticHandResult = { spec, gt, png };
  const h = [...detection.hands].sort((a, b) => b.labelScore - a.labelScore)[0];
  if (!h) return res;
  res.mpLabel = h.labelSide;
  res.mpImage = h.image;
  const gt2D: P2[] = joints.map(p => project(cam, p).slice(0, 2) as P2);
  res.mpVsGt2D = compare2D(toPx(h.image, cam.W, cam.H), gt2D).err;
  const world = handPoints(h, cam.W, cam.H, ctx.handSource, spec.side);
  res.mp = handAngles(world, spec.side);
  res.mpVsGt = compareHands(gt, res.mp);
  const driven = driver(spec.side === "L" ? { leftHand: world } : { rightHand: world }, ctx.model, ctx.boneIndex);
  const drivenJoints = handOf(ctx, driven, spec.side);
  res.driven = handAngles(toMpWorld(drivenJoints), spec.side);
  // The driver doesn't place the hand, only orients it: compare shapes after
  // moving the driven hand onto the ground-truth wrist.
  const shift = joints[0].map((x, i) => x - drivenJoints[0][i]);
  const moved = drivenJoints.map(p => p.map((x, i) => x + shift[i]) as [number, number, number]);
  res.drivenVsGt2D = compare2D(moved.map(p => project(cam, p).slice(0, 2) as P2), gt2D).err;
  res.drivenVsGt = compareHands(gt, res.driven);
  return res;
}
