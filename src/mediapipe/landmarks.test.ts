import { expect, test, describe, beforeAll } from "bun:test";
import { landmarksToPoseDeltas } from "./landmarks.js";
import {
  buildBoneIndex, allocBoneTransforms, forwardKinematics,
  type AnnyModel,
} from "../anny/index.js";
import { loadFixtureModel } from "../../tests/_helpers/model.js";
import type { WorldLandmark } from "./types.js";
import { MP } from "./constants.js";

/**
 * Build a synthetic A-pose landmark set in MediaPipe world coordinates.
 *   MP world: +x = anatomical left, +y = down, +z = away from camera
 * The subject stands upright facing the camera with arms slightly out.
 */
function aPoseLandmarks(): WorldLandmark[] {
  // Hip-centered (MP world is hip-centered). y grows downward, z = depth.
  const lm: WorldLandmark[] = new Array(33).fill(0).map(() => ({ x: 0, y: 0, z: 0, visibility: 1 }));

  const set = (i: number, x: number, y: number, z: number) => {
    lm[i] = { x, y, z, visibility: 1 };
  };

  // Hips (origin in MP world is between them)
  set(MP.LEFT_HIP,  +0.10, 0.00, 0.00);
  set(MP.RIGHT_HIP, -0.10, 0.00, 0.00);

  // Shoulders ~ 50 cm above hips
  set(MP.LEFT_SHOULDER,  +0.18, -0.50, 0.00);
  set(MP.RIGHT_SHOULDER, -0.18, -0.50, 0.00);

  // Arms in A-pose: elbow down-and-out, wrist further out and down
  set(MP.LEFT_ELBOW,  +0.40, -0.30, 0.00);
  set(MP.RIGHT_ELBOW, -0.40, -0.30, 0.00);
  set(MP.LEFT_WRIST,  +0.55, -0.05, 0.00);
  set(MP.RIGHT_WRIST, -0.55, -0.05, 0.00);
  set(MP.LEFT_INDEX,  +0.60, +0.05, 0.00);
  set(MP.RIGHT_INDEX, -0.60, +0.05, 0.00);

  // Legs straight down
  set(MP.LEFT_KNEE,  +0.10, +0.45, 0.00);
  set(MP.RIGHT_KNEE, -0.10, +0.45, 0.00);
  set(MP.LEFT_ANKLE,  +0.10, +0.90, 0.00);
  set(MP.RIGHT_ANKLE, -0.10, +0.90, 0.00);
  set(MP.LEFT_FOOT_INDEX,  +0.10, +0.93, -0.10);
  set(MP.RIGHT_FOOT_INDEX, -0.10, +0.93, -0.10);

  // Head landmarks
  set(MP.LEFT_EAR,   +0.08, -0.70, 0.00);
  set(MP.RIGHT_EAR,  -0.08, -0.70, 0.00);
  set(MP.LEFT_EYE,   +0.04, -0.73, -0.05);
  set(MP.RIGHT_EYE,  -0.04, -0.73, -0.05);
  set(MP.MOUTH_LEFT, +0.03, -0.65, -0.05);
  set(MP.MOUTH_RIGHT,-0.03, -0.65, -0.05);

  return lm;
}

describe("landmarksToPoseDeltas", () => {
  let model: AnnyModel;
  let boneIndex: Map<string, number>;

  beforeAll(async () => {
    model = await loadFixtureModel();
    boneIndex = buildBoneIndex(model);
  });

  test("returns an array of length boneCount", () => {
    const deltas = landmarksToPoseDeltas({ pose: aPoseLandmarks() }, model, boneIndex);
    expect(deltas.length).toBe(model.boneCount);
  });

  test("drives a sensible subset of bones (root, spine03, limbs, head)", () => {
    const deltas = landmarksToPoseDeltas({ pose: aPoseLandmarks() }, model, boneIndex);
    const named = ["root", "spine03", "head", "upperarm01.L", "upperarm01.R",
                   "lowerarm01.L", "upperleg01.L", "upperleg01.R"];
    for (const name of named) {
      const idx = boneIndex.get(name)!;
      expect(deltas[idx], `expected ${name} (idx ${idx}) to have a delta`).not.toBeNull();
    }
  });

  test("rotation matrices are orthonormal (det ≈ +1)", () => {
    const deltas = landmarksToPoseDeltas({ pose: aPoseLandmarks() }, model, boneIndex);
    for (let b = 0; b < deltas.length; b++) {
      const d = deltas[b];
      if (d === null) continue;
      // det of 3×3 row-major
      const det =
        d[0]*(d[4]*d[8] - d[5]*d[7]) -
        d[1]*(d[3]*d[8] - d[5]*d[6]) +
        d[2]*(d[3]*d[7] - d[4]*d[6]);
      expect(det).toBeCloseTo(1, 3);
    }
  });

  test("low-visibility landmarks are skipped", () => {
    const lm = aPoseLandmarks();
    // Drop left arm visibility below threshold
    lm[MP.LEFT_SHOULDER].visibility = 0.1;
    lm[MP.LEFT_ELBOW].visibility = 0.1;
    const deltas = landmarksToPoseDeltas({ pose: lm }, model, boneIndex);
    expect(deltas[boneIndex.get("upperarm01.L")!]).toBeNull();
    // Right arm should still be driven.
    expect(deltas[boneIndex.get("upperarm01.R")!]).not.toBeNull();
  });

  describe("previousDeltas stickiness", () => {
    test("inherits previous delta when current frame can't drive the bone", () => {
      // Frame 1: everyone visible — produces a non-null upperarm01.L delta.
      const goodFrame = aPoseLandmarks();
      const first = landmarksToPoseDeltas({ pose: goodFrame }, model, boneIndex);
      const L_UA = boneIndex.get("upperarm01.L")!;
      const firstDelta = first[L_UA];
      expect(firstDelta, "frame 1 should have driven upperarm01.L").not.toBeNull();

      // Frame 2: same landmarks but left arm goes invisible. Without
      // previousDeltas, the bone falls back to null (parent chain pose).
      // With previousDeltas, it should reuse frame 1's delta.
      const occludedFrame = aPoseLandmarks();
      occludedFrame[MP.LEFT_SHOULDER].visibility = 0.1;
      occludedFrame[MP.LEFT_ELBOW].visibility = 0.1;

      const withoutPrev = landmarksToPoseDeltas({ pose: occludedFrame }, model, boneIndex);
      expect(withoutPrev[L_UA], "no stickiness: occluded bone is null").toBeNull();

      const withPrev = landmarksToPoseDeltas(
        { pose: occludedFrame, previousDeltas: first },
        model, boneIndex,
      );
      expect(withPrev[L_UA], "stickiness: occluded bone inherits prev").toBe(firstDelta);
    });

    test("current frame's delta wins over previous when the bone is visible", () => {
      const lm1 = aPoseLandmarks();
      const first = landmarksToPoseDeltas({ pose: lm1 }, model, boneIndex);

      // Frame 2 moves the left elbow — must produce a *different* delta even
      // though previousDeltas has one. Stickiness only fills nulls.
      const lm2 = aPoseLandmarks();
      lm2[MP.LEFT_ELBOW] = { x: +0.40, y: -0.50, z: 0.00, visibility: 1 }; // moved up

      const second = landmarksToPoseDeltas(
        { pose: lm2, previousDeltas: first },
        model, boneIndex,
      );
      const L_UA = boneIndex.get("upperarm01.L")!;
      expect(second[L_UA]).not.toBe(first[L_UA]);
      expect(second[L_UA]).not.toBeNull();
    });

    test("mismatched previousDeltas length is silently ignored", () => {
      const lm = aPoseLandmarks();
      lm[MP.LEFT_SHOULDER].visibility = 0.1;
      lm[MP.LEFT_ELBOW].visibility = 0.1;
      // Wrong length array (different rig) — must not throw or be applied.
      const bogusPrev = Array(7).fill(new Float32Array([1,0,0,0,1,0,0,0,1]));
      const deltas = landmarksToPoseDeltas(
        { pose: lm, previousDeltas: bogusPrev as never },
        model, boneIndex,
      );
      expect(deltas[boneIndex.get("upperarm01.L")!]).toBeNull();
    });

    test("null entries in previousDeltas don't overwrite a fresh delta with null", () => {
      const lm = aPoseLandmarks();
      const allNull = Array(model.boneCount).fill(null);
      const deltas = landmarksToPoseDeltas(
        { pose: lm, previousDeltas: allNull },
        model, boneIndex,
      );
      // Right arm was set this frame, prev[idx] is null → must NOT be nulled.
      expect(deltas[boneIndex.get("upperarm01.R")!]).not.toBeNull();
    });
  });

  /**
   * Anatomical-side invariants — keep these green and the left/right swap
   * bug stays dead. The convention is documented in three places and any of
   * them silently flipping reintroduces the same visual artefact (the model
   * mirrors the video's anatomical sides), so we pin each layer down here:
   *
   *   1. Anny rest rig: *.L bones live on the world +x side (Anny: +x =
   *      anatomical left), *.R bones on the -x side.
   *   2. The driver routes MP.LEFT_* → *.L bones (anatomical mapping).
   *   3. mpToAnny preserves MP's anatomical-left convention (+x → +x).
   *
   * Together (1)+(2)+(3) mean: an MP pose with the subject's anatomical
   * LEFT arm raised drives the model's anatomical-LEFT arm up.
   */
  describe("anatomical-side conventions (L/R swap guards)", () => {
    test("rest rig: upperarm01.L sits on +x, upperarm01.R on -x (Anny world)", () => {
      // Translation column of each bone's rest 4×4 (row-major, idx 3/7/11).
      const restX = (name: string) => {
        const i = boneIndex.get(name)!;
        return model.restBonePoses[i * 16 + 3];
      };
      expect(restX("upperarm01.L"), "L upper arm on +x (anatomical left)").toBeGreaterThan(0.05);
      expect(restX("upperarm01.R"), "R upper arm on -x (anatomical right)").toBeLessThan(-0.05);
      expect(restX("upperleg01.L"), "L upper leg on +x").toBeGreaterThan(0.05);
      expect(restX("upperleg01.R"), "R upper leg on -x").toBeLessThan(-0.05);
    });

    test("driver routes MP.LEFT_* to *.L and MP.RIGHT_* to *.R end-to-end", () => {
      // Asymmetric pose: subject's LEFT arm raised straight up, RIGHT arm at
      // rest. In MP world: +x = anatomical left, +y = down, +z = depth.
      // "Raised up" means y is much smaller (less down) than the shoulder.
      const lm = aPoseLandmarks();
      // Push left wrist way above shoulder, leave right arm in A-pose.
      lm[MP.LEFT_ELBOW] = { x: +0.20, y: -1.00, z: 0.0, visibility: 1 };
      lm[MP.LEFT_WRIST] = { x: +0.20, y: -1.40, z: 0.0, visibility: 1 };
      lm[MP.LEFT_INDEX] = { x: +0.20, y: -1.50, z: 0.0, visibility: 1 };

      const deltas = landmarksToPoseDeltas({ pose: lm }, model, boneIndex);
      const xforms = allocBoneTransforms(model.boneCount);
      forwardKinematics(model, deltas, xforms);

      const posedZ = (name: string): number => {
        const i = boneIndex.get(name)!;
        // boneXforms[b] = pose[b] @ inv(rest[b]). Applying it to the bone's
        // rest-world translation recovers the bone's posed world translation.
        const rx = model.restBonePoses[i * 16 + 3];
        const ry = model.restBonePoses[i * 16 + 7];
        const rz = model.restBonePoses[i * 16 + 11];
        // z row of the 4×4 starts at offset 8; +11 is the translation.
        return xforms[i * 16 + 8]*rx + xforms[i * 16 + 9]*ry + xforms[i * 16 +10]*rz + xforms[i * 16 +11];
      };

      const zL = posedZ("wrist.L");
      const zR = posedZ("wrist.R");
      // Anny world: +z = up. With MP.LEFT raised overhead, the model's *.L
      // wrist should sit substantially higher than the *.R wrist. If anything
      // in the chain silently swaps L/R (driver routing, mpToAnny x sign,
      // rest rig labels), this inequality flips and the test fails.
      expect(zL, `wrist.L z=${zL.toFixed(2)}, wrist.R z=${zR.toFixed(2)} — L should be higher`)
        .toBeGreaterThan(zR + 0.3);
    });

    test("mpToAnny preserves the anatomical-left convention (+x_mp → +x_anny)", () => {
      // Drive a pose where MP.LEFT_HIP is on the +x side (matches real MP
      // anatomical-left convention) and check that the resulting upperleg01.L
      // delta exists and that anatomical sides aren't being silently swapped
      // by the mpToAnny x-axis conversion.
      const lm = aPoseLandmarks();
      const deltas = landmarksToPoseDeltas({ pose: lm }, model, boneIndex);
      expect(deltas[boneIndex.get("upperleg01.L")!]).not.toBeNull();
      expect(deltas[boneIndex.get("upperleg01.R")!]).not.toBeNull();
      // Now flip the synthetic landmarks' x sign and check the *opposite*
      // bone gets the delta this would normally produce. (mirrorX=true is
      // documented as "your input is already mirrored" — flipping the input
      // landmarks and asserting symmetric output is the cleanest check that
      // the x-axis path is wired straight through.)
      const flipped = aPoseLandmarks().map(l => ({ ...l, x: -l.x }));
      const flippedDeltas = landmarksToPoseDeltas({ pose: flipped, mirrorX: true }, model, boneIndex);
      const L_UL = boneIndex.get("upperleg01.L")!;
      const R_UL = boneIndex.get("upperleg01.R")!;
      // With mirrorX=true correcting the flipped input, results should match
      // the un-flipped run within floating-point noise.
      const close = (a: Float32Array, b: Float32Array): boolean => {
        for (let i = 0; i < 9; i++) if (Math.abs(a[i] - b[i]) > 1e-5) return false;
        return true;
      };
      expect(close(deltas[L_UL]!, flippedDeltas[L_UL]!),
        "mirrorX should round-trip an x-flipped input to identical L delta").toBe(true);
      expect(close(deltas[R_UL]!, flippedDeltas[R_UL]!),
        "mirrorX should round-trip an x-flipped input to identical R delta").toBe(true);
    });
  });
});
