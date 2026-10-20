/**
 * Inverse-kinematics recovery — the driver's ground-truth unit test.
 *
 * Pose the rig with KNOWN random deltas (built directly on its hinge axes,
 * not by the driver), read off the landmarks MediaPipe would ideally report,
 * drive the rig from those, and require the driver to reproduce what is
 * observable: limb directions, hip/shoulder lines, head orientation, palm
 * orientation and every finger joint angle. No browser, no MediaPipe — any
 * error here is the driver's own.
 */

import { beforeAll, describe, expect, test } from "bun:test";

import { buildBoneIndex, type AnnyModel } from "../../src/anny/index.js";
import { landmarksToPoseDeltas, type WorldLandmark } from "../../src/mediapipe/index.js";
import { loadFixtureModel } from "../_helpers/model.js";
import {
  bodyDirections, compareBody, compareHands, handAngles, mean, type Side,
} from "../roundtrip/metrics.js";
import { randomPose } from "../roundtrip/random-pose.js";
import { buildRigExtras, handJointsAnny, poseJointsAnny, toMpWorld, type PosedSkeleton, type RigExtras } from "../roundtrip/skeleton.js";
import { landmarksOf, posedSkeleton } from "../roundtrip/synthetic-landmarks.js";
import { angleDeg } from "../roundtrip/vec.js";

let model: AnnyModel;
let boneIndex: Map<string, number>;
let extras: RigExtras;

beforeAll(async () => {
  model = await loadFixtureModel();
  boneIndex = buildBoneIndex(model);
  extras = buildRigExtras(model);
});

const SEEDS = Array.from({ length: 60 }, (_, i) => i + 1);

const bodyDirs = (sk: PosedSkeleton) =>
  bodyDirections(toMpWorld(poseJointsAnny(sk).map(p => p ?? [0, 0, 0])));
const hand = (sk: PosedSkeleton, s: Side) => handAngles(toMpWorld(handJointsAnny(sk, s)), s);

describe("driver recovers known poses from ideal landmarks", () => {
  test("body: limbs, hip & shoulder lines, head", () => {
    const err: Record<string, number[]> = {};
    for (const seed of SEEDS) {
      const gt = posedSkeleton(model, extras, randomPose(model, boneIndex, seed));
      const lm = landmarksOf(gt);
      const got = posedSkeleton(model, extras, landmarksToPoseDeltas({ pose: lm.pose }, model));
      for (const [k, v] of Object.entries(compareBody(bodyDirs(gt), bodyDirs(got)))) (err[k] ??= []).push(v);
      (err.headFacing ??= []).push(angleDeg(gt.axis("head", 2), got.axis("head", 2)));
      (err.headUp ??= []).push(angleDeg(gt.axis("head", 1), got.axis("head", 1)));
    }
    for (const k of ["upperarm.L", "upperarm.R", "forearm.L", "forearm.R", "thigh.L", "thigh.R",
      "shin.L", "shin.R", "shoulders", "hips", "headFacing", "headUp"]) {
      expect(Math.max(...err[k])).toBeLessThan(0.5);
    }
    // "Up" (hips → shoulders) is the one soft target: the spine model spreads
    // a twist evenly, a real spine may not.
    expect(mean(err.torso)).toBeLessThan(3);
  });

  test("hands: palm orientation, every finger joint, thumb", () => {
    const fingers: number[] = [], thumbs: number[] = [], palms: number[] = [], flips: number[] = [];
    for (const seed of SEEDS) {
      const gt = posedSkeleton(model, extras, randomPose(model, boneIndex, seed));
      const lm = landmarksOf(gt);
      const got = posedSkeleton(model, extras, landmarksToPoseDeltas(lm, model));
      for (const s of ["L", "R"] as const) {
        const d = compareHands(hand(gt, s), hand(got, s));
        fingers.push(d.fingerFlex);
        thumbs.push(d.thumbFlex);
        palms.push(d.palmNormal);
        flips.push(d.bendSignFlips);
      }
    }
    expect(Math.max(...palms)).toBeLessThan(0.5);
    expect(Math.max(...thumbs)).toBeLessThan(0.5);
    expect(mean(fingers)).toBeLessThan(2);
    expect(Math.max(...fingers)).toBeLessThan(4);
    expect(Math.max(...flips)).toBe(0);
  });

  test("hands without a pose: hand still oriented and articulated in world", () => {
    for (const seed of SEEDS.slice(0, 20)) {
      const gt = posedSkeleton(model, extras, randomPose(model, boneIndex, seed));
      const lm = landmarksOf(gt);
      const got = posedSkeleton(model, extras, landmarksToPoseDeltas({ rightHand: lm.rightHand }, model));
      const d = compareHands(hand(gt, "R"), hand(got, "R"));
      expect(d.palmNormal).toBeLessThan(0.5);
      expect(d.fingerFlex).toBeLessThan(4);
    }
  });

  test("a depth-mirrored hand (MediaPipe chirality slip) still curls toward the palm", () => {
    let checked = 0;
    for (const seed of SEEDS) {
      const gt = posedSkeleton(model, extras, randomPose(model, boneIndex, seed));
      const g = hand(gt, "L");
      const curl = mean(g.flex.slice(1).map(f => f[1] + f[2]));
      if (curl < 60) continue;   // only clearly curled hands carry chirality
      const lm = landmarksOf(gt);
      const wz = lm.leftHand[0].z;
      const mirrored: WorldLandmark[] = lm.leftHand.map(p => ({ ...p, z: 2 * wz - p.z }));
      const got = posedSkeleton(model, extras, landmarksToPoseDeltas({ leftHand: mirrored }, model));
      const h = hand(got, "L");
      for (const f of h.flex.slice(1)) for (const a of f) expect(a).toBeGreaterThan(-25);
      expect(mean(h.flex.slice(1).map(f => f[1] + f[2]))).toBeGreaterThan(40);
      checked++;
    }
    expect(checked).toBeGreaterThan(5);
  });

  test("garbage hand landmarks never hyperextend fingers past the anatomical limits", () => {
    let seed = 99;
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
    for (let k = 0; k < 50; k++) {
      const noisy: WorldLandmark[] = Array.from({ length: 21 }, () => ({ x: rand() - 0.5, y: rand() - 0.5, z: rand() - 0.5 }));
      const got = posedSkeleton(model, extras, landmarksToPoseDeltas({ leftHand: noisy, rightHand: noisy }, model));
      for (const s of ["L", "R"] as const) {
        for (const f of hand(got, s).flex.slice(1)) {
          expect(f[0]).toBeGreaterThan(-45);
          expect(f[1]).toBeGreaterThan(-25);
          expect(f[2]).toBeGreaterThan(-30);
        }
      }
    }
  });

  test("runs well under a millisecond per frame", () => {
    const lms = SEEDS.slice(0, 20).map(seed => landmarksOf(posedSkeleton(model, extras, randomPose(model, boneIndex, seed))));
    for (const lm of lms) landmarksToPoseDeltas(lm, model);   // warm-up / JIT
    const t0 = performance.now();
    for (let rep = 0; rep < 5; rep++) for (const lm of lms) landmarksToPoseDeltas(lm, model);
    expect((performance.now() - t0) / (5 * lms.length)).toBeLessThan(1);
  });
});
