import { MP, MP_HAND } from "./constants.js";
import type { Landmark, WorldLandmark } from "./types.js";

/**
 * Pick which detected hand is left vs right, using proximity of each hand's
 * 2D wrist landmark to the body's LEFT_WRIST / RIGHT_WRIST in image space.
 *
 * MediaPipe Hand Landmarker also returns a `handedness` label per hand, but
 * the label is computed in raw camera space and can be flipped vs. user
 * expectation when the display is mirrored. Proximity-to-body-wrist is more
 * robust and doesn't depend on which side of the camera the user is on.
 *
 * @param handsImageLandmarks  HandLandmarker `result.landmarks` (image-space).
 * @param handsWorldLandmarks  HandLandmarker `result.worldLandmarks`.
 * @param poseImageLandmarks   PoseLandmarker `result.landmarks[0]` (image-space).
 * @returns                    Assigned world-space landmark arrays for the API.
 */
export function assignHands(
  handsImageLandmarks: Landmark[][],
  handsWorldLandmarks: WorldLandmark[][],
  poseImageLandmarks: Landmark[],
): { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } {
  const out: { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } = {};
  if (handsImageLandmarks.length === 0) return out;

  const bodyL = poseImageLandmarks[MP.LEFT_WRIST];
  const bodyR = poseImageLandmarks[MP.RIGHT_WRIST];
  if (!bodyL || !bodyR) return out;

  const sq = (a: Landmark, b: Landmark) =>
    (a.x - b.x)**2 + (a.y - b.y)**2;

  if (handsImageLandmarks.length === 1) {
    const w = handsImageLandmarks[0][MP_HAND.WRIST];
    if (sq(w, bodyL) < sq(w, bodyR)) out.leftHand = handsWorldLandmarks[0];
    else                              out.rightHand = handsWorldLandmarks[0];
    return out;
  }

  // ≥2 detected: pick the assignment with minimum total mismatch.
  const [a, b] = handsImageLandmarks;
  const wa = a[MP_HAND.WRIST];
  const wb = b[MP_HAND.WRIST];
  const costAB = sq(wa, bodyL) + sq(wb, bodyR);
  const costBA = sq(wb, bodyL) + sq(wa, bodyR);
  if (costAB <= costBA) {
    out.leftHand  = handsWorldLandmarks[0];
    out.rightHand = handsWorldLandmarks[1];
  } else {
    out.leftHand  = handsWorldLandmarks[1];
    out.rightHand = handsWorldLandmarks[0];
  }
  return out;
}
