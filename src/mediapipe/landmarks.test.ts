import { expect, test, describe, beforeAll } from "bun:test";
import { landmarksToPoseDeltas } from "./landmarks.js";
import { buildBoneIndex, type AnnyModel } from "../anny/index.js";
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
});
