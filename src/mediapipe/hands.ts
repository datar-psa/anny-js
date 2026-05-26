import { MP, MP_HAND } from "./constants.js";
import type { Landmark, WorldLandmark } from "./types.js";

/**
 * Assign detected MediaPipe hands to anatomical left/right slots using the
 * **pose** landmarker's LEFT_WRIST / RIGHT_WRIST as ground truth.
 *
 * Why not MediaPipe Hand Landmarker's own `handedness` label: per the MP
 * docs, "handedness is determined assuming the input image is mirrored…If
 * it is not the case, please swap." For raw camera frames (third-person
 * video, or webcam where the mirror is only applied in CSS for display),
 * MP's labels are inverted from the subject's anatomy. Pose's left/right
 * wrists, in contrast, are anatomical by definition — pose's
 * `MP.LEFT_WRIST` (idx 15) IS the subject's anatomical left wrist, with no
 * mirror caveat.
 *
 * So: take each detected hand's image-space wrist, measure its distance to
 * pose.LEFT_WRIST and pose.RIGHT_WRIST, and route to the closer side.
 *
 * @param handsImageLandmarks  HandLandmarker `result.landmarks` (image-space).
 * @param handsWorldLandmarks  HandLandmarker `result.worldLandmarks`.
 * @param poseImageLandmarks   PoseLandmarker `result.landmarks[0]` (image-space).
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

  // ≥2 detected: bipartite match — pick the (det0→L, det1→R) vs (det0→R,
  // det1→L) assignment that minimises total distance to pose's anatomical
  // wrists. Handles the boxing-guard case where both detections sit close
  // together; pose's anatomical labels still pull each to its correct side.
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
