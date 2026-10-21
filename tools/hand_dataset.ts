#!/usr/bin/env bun
/**
 * Capture MediaPipe readings of ground-truth hands at several sizes, the way
 * the demo sees them: rendered at N px, bilinearly upscaled to 256 px (the
 * ROI crop), then detected. Writes .cache/hand-dataset.json for offline
 * analysis (`bun tools/hand_trust.ts`) — fusion variants then score in
 * milliseconds without a browser.
 *
 *   bun tools/hand_dataset.ts [--random 64] [--sizes 24,32,48,64,96,160,384]
 */

import { parseArgs } from "node:util";

import { allocBoneTransforms, forwardKinematics } from "../src/anny/index.js";
import { project } from "../tests/roundtrip/camera.js";
import { HarnessClient } from "../tests/roundtrip/client.js";
import type { HandDetection } from "../tests/roundtrip/protocol.js";
import { handCamera, makeContext } from "../tests/roundtrip/run.js";
import { handJointsAnny, PosedSkeleton } from "../tests/roundtrip/skeleton.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { gtHandDeltas, gtHands, type GtHand } from "../tests/naturalness/hand-gt.js";
import { buildPhysicalRig, physicalScore } from "../tests/naturalness/physical.js";

const { values: args } = parseArgs({
  options: {
    random: { type: "string", default: "96" },
    sizes: { type: "string", default: "24,32,48,64,96,160,384" },
    out: { type: "string", default: ".cache/hand-dataset.json" },
  },
});

export interface HandSample {
  hand: GtHand;
  /** Render size (px, square) before the 256 px upscale. */
  size: number;
  /** Ground-truth palm length (wrist → middle MCP) in render pixels. */
  palmPx: number;
  /** Ground-truth hand joints, Anny world (MP_HAND order). */
  gtJoints: [number, number, number][];
  /** Best MediaPipe hand on the render, if any. */
  mp?: Pick<HandDetection, "image" | "labelSide" | "labelScore">;
  W: number;
  H: number;
}

const model = await loadFixtureModel();
const client = await HarnessClient.start({ poseModel: "lite" });
const ctx = makeContext(client, model);
// Only physically possible hands: no finger passing through another.
const physical = buildPhysicalRig(model);
const hands = gtHands(Number(args.random)).filter(h =>
  physicalScore(physical, gtHandDeltas(h, ctx.boneIndex, model.boneCount), h.side).penetrationMm < 1);
console.log(`${hands.length} ground-truth hands`);
const sizes = args.sizes.split(",").map(Number);
const samples: HandSample[] = [];
const t0 = performance.now();
for (const hand of hands) {
  const deltas = gtHandDeltas(hand, ctx.boneIndex, model.boneCount);
  const bx = allocBoneTransforms(model.boneCount);
  forwardKinematics(model, deltas, bx);
  const joints = handJointsAnny(new PosedSkeleton(model, ctx.extras, bx), hand.side);
  for (const size of sizes) {
    const cam = handCamera(ctx, deltas, hand.side, size);
    const p0 = project(cam, joints[0]), p9 = project(cam, joints[9]);
    const { detection } = await client.renderAndDetect(
      deltas, cam, { handOnly: hand.side }, { pose: false, upscaleTo: size < 256 ? 256 : undefined },
    );
    const best = [...detection.hands].sort((a, b) => b.labelScore - a.labelScore)[0];
    samples.push({
      hand, size, W: cam.W, H: cam.H,
      palmPx: Math.hypot(p9[0] - p0[0], p9[1] - p0[1]),
      gtJoints: joints.map(p => [p[0], p[1], p[2]]),
      mp: best && { image: best.image, labelSide: best.labelSide, labelScore: best.labelScore },
    });
  }
  if (samples.length % (sizes.length * 20) === 0) {
    console.log(`${samples.length}/${hands.length * sizes.length} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
  }
}
await client.close();
await Bun.write(args.out, JSON.stringify(samples));
console.log(`wrote ${samples.length} samples → ${args.out}`);
