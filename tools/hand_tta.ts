#!/usr/bin/env bun
/**
 * How noisy is MediaPipe on REAL images? There's no ground truth for them,
 * but test-time augmentation gives a proxy: shift / scale / rotate the image
 * slightly and re-detect. Joint angles are invariant to all three, so the
 * spread of the readings is MediaPipe's instability on that image.
 *
 * On ground-truth renders the same spread can be compared with the true
 * error. If error / spread is stable across sizes, the spread measured on
 * real photos and video frames calibrates the render-fitted noise model
 * (tools/hand_trust.ts) for real footage.
 *
 *   bun tools/hand_tta.ts [--k 8] [--hands 48]
 */

import { parseArgs } from "node:util";

import { allocBoneTransforms, forwardKinematics } from "../src/anny/index.js";
import { handLandmarks3D, matchHandChirality, palmLengthPx } from "../src/mediapipe/hands.js";
import { handPointsAnny } from "../src/mediapipe/hand-geometry.js";
import type { Vec3 } from "../src/mediapipe/geom.js";
import { rigInfo } from "../src/mediapipe/rig.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { gtHandDeltas, gtHands } from "../tests/naturalness/hand-gt.js";
import { FINGER_DIMS, handState } from "../tests/naturalness/hand-state.js";
import { PHOTOS, photoPath, videoFrames } from "../tests/roundtrip/cases.js";
import { HarnessClient } from "../tests/roundtrip/client.js";
import type { Detection, DetectOptions, HandDetection } from "../tests/roundtrip/protocol.js";
import { handCamera, makeContext } from "../tests/roundtrip/run.js";
import { rng } from "../tests/roundtrip/synthetic.js";
import { handJointsAnny, PosedSkeleton } from "../tests/roundtrip/skeleton.js";

const { values: args } = parseArgs({
  options: {
    k: { type: "string", default: "8" },
    hands: { type: "string", default: "48" },
  },
});
const K = Number(args.k);

const model = await loadFixtureModel();
const rig = rigInfo(model);
const client = await HarnessClient.start({ poseModel: "lite" });
const ctx = makeContext(client, model);
const r = rng(5);
const augmentations: NonNullable<DetectOptions["augment"]>[] = Array.from({ length: K }, () => ({
  dx: (r() - 0.5) * 0.08, dy: (r() - 0.5) * 0.08, scale: 0.9 + 0.2 * r(), rotDeg: (r() - 0.5) * 24,
}));

function state(h: HandDetection, W: number, H: number, side: "L" | "R"): number[] {
  const world = matchHandChirality(handLandmarks3D(h.image, W, H), side, h.labelSide === "L" ? "Left" : "Right");
  return handState(handPointsAnny(world, side), side, rig);
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);

/** Per-dimension variance of a set of state vectors. */
function variance(vs: number[][]): number[] {
  return vs[0].map((_, j) => {
    const m = mean(vs.map(v => v[j]));
    return mean(vs.map(v => (v[j] - m) ** 2)) * vs.length / Math.max(1, vs.length - 1);
  });
}

interface Spread { px: number; fingerVar: number; thumbVar: number; fingerMse?: number; thumbMse?: number }

const fingerPart = (v: number[]) => mean(v.slice(0, FINGER_DIMS));
const thumbPart = (v: number[]) => mean(v.slice(FINGER_DIMS));

/** Real image: one spread per detected hand side. */
async function realSpreads(path: string): Promise<Spread[]> {
  const base: Detection = await client.detectFile(path);
  const out: Spread[] = [];
  const reads: Record<string, number[][]> = {};
  const px: Record<string, number> = {};
  for (const h of base.hands) {
    (reads[h.side] ??= []).push(state(h, base.width, base.height, h.side));
    px[h.side] = palmLengthPx(h.image, base.width, base.height);
  }
  for (const a of augmentations) {
    const det = await client.detectFile(path, { augment: a });
    for (const h of det.hands) if (reads[h.side]) reads[h.side].push(state(h, det.width, det.height, h.side));
  }
  for (const [side, vs] of Object.entries(reads)) {
    if (vs.length < 4) continue;
    const v = variance(vs);
    out.push({ px: px[side], fingerVar: fingerPart(v), thumbVar: thumbPart(v) });
  }
  return out;
}

/** Ground-truth render at `size`: spread across augmentations and the true error. */
async function syntheticSpread(hand: ReturnType<typeof gtHands>[number], size: number): Promise<Spread | undefined> {
  const deltas = gtHandDeltas(hand, ctx.boneIndex, model.boneCount);
  const bx = allocBoneTransforms(model.boneCount);
  forwardKinematics(model, deltas, bx);
  const gt = handState(handJointsAnny(new PosedSkeleton(model, ctx.extras, bx), hand.side) as Vec3[], hand.side, rig);
  const cam = handCamera(ctx, deltas, hand.side, size);
  const upscaleTo = size < 256 ? 256 : undefined;
  const vs: number[][] = [];
  let px = 0;
  for (const augment of [undefined, ...augmentations]) {
    const { detection } = await client.renderAndDetect(deltas, cam, { handOnly: hand.side }, { pose: false, upscaleTo, augment });
    const best = [...detection.hands].sort((a, b) => b.labelScore - a.labelScore)[0];
    if (!best) continue;
    if (!augment) px = palmLengthPx(best.image, cam.W, cam.H);
    vs.push(state(best, cam.W, cam.H, hand.side));
  }
  if (vs.length < 4 || !px) return undefined;
  const v = variance(vs);
  const mse = gt.map((g, j) => mean(vs.map(x => (x[j] - g) ** 2)));
  return { px, fingerVar: fingerPart(v), thumbVar: thumbPart(v), fingerMse: fingerPart(mse), thumbMse: thumbPart(mse) };
}

function report(label: string, ss: Spread[]): void {
  const sd = (x: number) => Math.sqrt(x).toFixed(1).padStart(6);
  const line = [
    label.padEnd(26), String(ss.length).padStart(4), mean(ss.map(s => s.px)).toFixed(0).padStart(7),
    sd(mean(ss.map(s => s.fingerVar))), sd(mean(ss.map(s => s.thumbVar))),
  ];
  if (ss[0]?.fingerMse !== undefined) {
    line.push(sd(mean(ss.map(s => s.fingerMse!))), sd(mean(ss.map(s => s.thumbMse!))),
      (mean(ss.map(s => s.fingerMse!)) / mean(ss.map(s => s.fingerVar))).toFixed(1).padStart(8));
  }
  console.log(line.join(""));
}

console.log(`MediaPipe test-time-augmentation spread (${K} augmentations: ±4 % shift, ±10 % scale, ±12° rotation)`);
console.log("set".padEnd(26) + "   n palm px  spread SD: fingers thumb   true RMS: fingers thumb   MSE/var");

const photos: Spread[] = [];
for (const f of PHOTOS) photos.push(...await realSpreads(photoPath(f)));
const clips: Spread[] = [];
for (const clip of ["dancer", "boxer"]) for (const p of await videoFrames(clip, 16)) clips.push(...await realSpreads(p));

const hands = gtHands().filter((_, i) => i % Math.ceil(gtHands().length / Number(args.hands)) === 0);
const synthetic: Record<number, Spread[]> = {};
for (const size of [32, 64, 160, 384]) {
  synthetic[size] = [];
  for (const h of hands) {
    const s = await syntheticSpread(h, size);
    if (s) synthetic[size].push(s);
  }
}
await client.close();

report("real photos", photos);
report("real clip frames (ROI)", clips);
for (const [size, ss] of Object.entries(synthetic)) report(`renders @${size} (upscaled)`, ss);
await Bun.write(".cache/hand-tta.json", JSON.stringify({ photos, clips, synthetic }));
