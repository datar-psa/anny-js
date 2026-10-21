/**
 * The fusion layer: Gaussian updates against closed forms, the OU process
 * keeping the prior stationary, and the driver's use of hand quality —
 * small hands lean on the prior, large ones on the reading.
 */

import { beforeAll, describe, expect, test } from "bun:test";

import { buildBoneIndex, type AnnyModel } from "../../src/anny/index.js";
import { diagonal, gaussianUpdate, mahalanobis2, ouPredict, posteriorMean } from "../../src/mediapipe/fusion.js";
import { handPointsAnny } from "../../src/mediapipe/hand-geometry.js";
import { HAND_PRIOR } from "../../src/mediapipe/hand-prior.js";
import { robustUpdate } from "../../src/mediapipe/hand-fusion.js";
import { landmarksToPoseDeltas, type WorldLandmark } from "../../src/mediapipe/index.js";
import { rigInfo } from "../../src/mediapipe/rig.js";
import { loadFixtureModel } from "../_helpers/model.js";
import { gtHandDeltas, GESTURES } from "../naturalness/hand-gt.js";
import { handState } from "../naturalness/hand-state.js";
import { buildRigExtras, handJointsAnny, toMpWorld, type RigExtras } from "../roundtrip/skeleton.js";
import { posedSkeleton } from "../roundtrip/synthetic-landmarks.js";

describe("Gaussian fusion", () => {
  test("scalar update is the textbook Kalman update", () => {
    const post = gaussianUpdate({ mean: [1], cov: [4] }, [3], [1]);
    // K = 4 / (4 + 1) = 0.8
    expect(post.mean[0]).toBeCloseTo(1 + 0.8 * 2, 12);
    expect(post.cov[0]).toBeCloseTo(0.2 * 4, 12);
    expect(posteriorMean({ mean: [1], cov: [4] }, [3], [1])[0]).toBeCloseTo(post.mean[0], 12);
  });

  test("correlated prior carries a measured dimension over to an unmeasured one", () => {
    // Two strongly correlated dims (fingers curl together); measure only the first
    // (huge noise on the second): the second follows.
    const prior = { mean: [0, 0], cov: [1, 0.9, 0.9, 1] };
    const post = gaussianUpdate(prior, [2, 0], [0.01, 0, 0, 1e6]);
    expect(post.mean[0]).toBeCloseTo(2 * (1 / 1.01), 3);
    expect(post.mean[1]).toBeGreaterThan(1.7);
  });

  test("OU prediction relaxes toward the prior and keeps it stationary", () => {
    const prior = { mean: [5, -1], cov: diagonal([4, 9]) };
    const stationary = ouPredict(prior, prior, 0.1, 0.3);
    stationary.mean.forEach((m, i) => expect(m).toBeCloseTo(prior.mean[i], 12));
    stationary.cov.forEach((c, i) => expect(c).toBeCloseTo(prior.cov[i], 12));
    const far = ouPredict({ mean: [50, 50], cov: diagonal([0.1, 0.1]) }, prior, 10, 0.3);
    expect(far.mean[0]).toBeCloseTo(5, 6);
    expect(far.cov[3]).toBeCloseTo(9, 4);
  });

  test("a blunder is discounted by how surprising it is", () => {
    const belief = { mean: [0, 0], cov: diagonal([1, 1]) };
    const R = diagonal([1, 1]);
    expect(mahalanobis2(belief, [10, 0], R)).toBeCloseTo(50, 9);
    const plain = gaussianUpdate(belief, [10, 0], R).mean[0];
    const robust = robustUpdate(belief, [10, 0], R).mean[0];
    expect(plain).toBeCloseTo(5, 9);
    expect(robust).toBeLessThan(plain / 2);
    // …while an ordinary reading is taken as usual.
    expect(robustUpdate(belief, [1, 0], R).mean[0]).toBeCloseTo(gaussianUpdate(belief, [1, 0], R).mean[0], 9);
  });
});

describe("driver: hand quality", () => {
  let model: AnnyModel;
  let extras: RigExtras;
  let fist: WorldLandmark[];

  beforeAll(async () => {
    model = await loadFixtureModel();
    extras = buildRigExtras(model);
    const hand = { name: "fist", family: "fist", side: "R" as const, pose: GESTURES.fist, view: { axis: [0, 0, 1] as [number, number, number], deg: 0 } };
    fist = toMpWorld(handJointsAnny(posedSkeleton(model, extras, gtHandDeltas(hand, buildBoneIndex(model), model.boneCount)), "R"));
  });

  /** Distance of the driven right hand's state from a target state (RMS over the 16 finger dims). */
  const driven = (q?: { palmPx: number; labelAgrees: boolean }) => {
    const d = landmarksToPoseDeltas({ rightHand: fist, rightHandQuality: q }, model);
    const pts = handJointsAnny(posedSkeleton(model, extras, d), "R");
    return handState(pts as never, "R", rigInfo(model));
  };
  const rms = (a: number[], b: readonly number[]) => Math.sqrt(a.slice(0, 16).reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / 16);

  test("a large clear hand is driven as read; a tiny one leans on the natural-hand prior", () => {
    const truth = handState(handPointsAnny(fist, "R"), "R", rigInfo(model));
    const asRead = driven();
    const large = driven({ palmPx: 200, labelAgrees: true });
    const tiny = driven({ palmPx: 3, labelAgrees: true });
    expect(rms(asRead, truth)).toBeLessThan(3);
    expect(rms(large, truth)).toBeLessThan(15);
    expect(rms(tiny, HAND_PRIOR.mean)).toBeLessThan(rms(tiny, truth));
    expect(rms(tiny, HAND_PRIOR.mean)).toBeLessThan(10);
  });
});
