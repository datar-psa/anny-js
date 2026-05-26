/**
 * Real-fixture hand tests — the NON-CIRCULAR ones.
 *
 * `hands-synth.test.ts` checks the driver against targets built the same way
 * the driver builds them, so it can only catch "the bone didn't go where we
 * asked". It is blind to a *shared* systematic error (wrong axis, mirrored
 * palm, depth flip) because the same wrong math is on both sides.
 *
 * These tests instead ground every assertion in the POSE landmarks, which
 * come from a different model and a different coordinate pipeline than the
 * hand landmarks the driver consumes — and they're already in real world
 * coordinates. So:
 *   • metacarpal bones must point toward pose's own knuckle landmarks
 *   • the wrist must point along pose's wrist→knuckle direction
 *   • every finger must curl toward the palm (palm side derived from pose's
 *     thumb, NOT from the hand-landmark targets)
 *
 * Fixtures are captured by `tools/capture_landmarks.ts` and committed under
 * `fixtures/`. Each frame stores the RAW MP pose + ROI hand worldLandmarks,
 * so re-running the driver here exercises the exact production code path.
 */

import { expect, test, describe, beforeAll } from "bun:test";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildBoneIndex, allocBoneTransforms, forwardKinematics, type AnnyModel,
} from "../../src/anny/index.js";
import { landmarksToPoseDeltas, type WorldLandmark } from "../../src/mediapipe/index.js";
import { loadFixtureModel } from "../_helpers/model.js";
import { verifyHandVsPose } from "./verify_hands.js";

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(here, "fixtures");

interface Frame {
  worldLandmarks: WorldLandmark[];
  imageLandmarks?: WorldLandmark[];
  leftHand?:  { world: WorldLandmark[]; image: WorldLandmark[] };
  rightHand?: { world: WorldLandmark[]; image: WorldLandmark[] };
}

describe("MediaPipe hands — pose-grounded (non-circular) parity", () => {
  let model: AnnyModel;
  let boneIndex: Map<string, number>;
  beforeAll(async () => {
    model = await loadFixtureModel();
    boneIndex = buildBoneIndex(model);
  });

  const fixtures = ["boxer", "dancer", "poser"];

  for (const name of fixtures) {
    const path = resolve(FIXTURES, `${name}_landmarks.json`);
    const exists = Bun.file(path).size > 0;
    const it = exists ? test : test.skip;

    it(`${name}: hands track pose + fingers curl palmward (never backward)`, async () => {
      const data: { frames: Frame[] } = JSON.parse(await Bun.file(path).text());
      const handFrames = data.frames.filter(f => f.leftHand || f.rightHand);
      // The fixture must actually contain hand detections, else this test is
      // silently vacuous — fail loudly so a bad capture can't hide a regression.
      expect(handFrames.length, `${name} has no hand frames — re-capture fixture`).toBeGreaterThan(0);

      const xforms = allocBoneTransforms(model.boneCount);
      let metaErrSum = 0, metaErrN = 0, metaErrMax = 0;
      let wristErrMax = 0;
      let backwardFrames = 0, curlChecks = 0;

      for (const f of handFrames) {
        const deltas = landmarksToPoseDeltas({
          pose: f.worldLandmarks,
          leftHand:  f.leftHand?.world,
          rightHand: f.rightHand?.world,
          visibilityMin: 0.2,
        }, model, boneIndex);
        forwardKinematics(model, deltas, xforms);

        for (const side of ["L", "R"] as const) {
          const h = side === "L" ? f.leftHand?.world : f.rightHand?.world;
          if (!h) continue;
          const r = verifyHandVsPose(model, boneIndex, xforms, f.worldLandmarks, side);
          if (!r.ran) continue;

          for (const e of [r.indexMetacarpalErrDeg, r.pinkyMetacarpalErrDeg]) {
            if (!Number.isNaN(e)) { metaErrSum += e; metaErrN++; metaErrMax = Math.max(metaErrMax, e); }
          }
          if (!Number.isNaN(r.wristErrDeg)) wristErrMax = Math.max(wristErrMax, r.wristErrDeg);

          // Skip the thumb (index 0): it opposes the palm rather than curling
          // into it, so it legitimately sits near/below the palm plane and the
          // palmar metric isn't meaningful for it. Check fingers 2-5 only.
          r.fingerCurlPalmward.forEach((c, fi) => {
            if (fi === 0 || Number.isNaN(c)) return;
            curlChecks++;
            // Clearly-backward curl (tip tilts toward the back of the hand by
            // a meaningful margin). Tiny negatives are straight-finger noise;
            // -0.15 is a real backward bend.
            if (c < -0.15) backwardFrames++;
          });
        }
      }

      const metaErrMean = metaErrN > 0 ? metaErrSum / metaErrN : 0;

      // Wrist is driven straight from pose, so it must match almost exactly.
      expect(wristErrMax, `wrist max err ${wristErrMax.toFixed(1)}°`).toBeLessThan(5);
      // Metacarpals come from aligned hand data; pose knuckles are coarse, so
      // allow more slack but still demand they point the right general way.
      expect(metaErrMean, `metacarpal mean err ${metaErrMean.toFixed(1)}°`).toBeLessThan(25);
      expect(metaErrMax,  `metacarpal max err ${metaErrMax.toFixed(1)}°`).toBeLessThan(45);
      // The headline assertion: fingers must not bend backward. A handful of
      // noisy frames are tolerable; a systematic palm-flip is not.
      const backwardFrac = curlChecks > 0 ? backwardFrames / curlChecks : 0;
      expect(backwardFrac, `${backwardFrames}/${curlChecks} finger-curls bend backward`).toBeLessThan(0.1);
    });
  }
});
