import { MP, MP_HAND } from "./constants.js";
import type { Landmark, WorldLandmark } from "./types.js";

export type HandSide = "L" | "R";

/**
 * Anatomical side of each detected hand.
 *
 * With a pose: bipartite match of each hand's image-space wrist to the pose's
 * LEFT_WRIST / RIGHT_WRIST (anatomical by definition, robust to hands that
 * cross the midline). Without one: MediaPipe's handedness label — which,
 * despite the docs' "assumes a mirrored image" caveat, is anatomical for
 * ordinary non-mirrored footage (verified on MediaPipe's own labelled test
 * photos).
 *
 * @param handsImage  `HandLandmarkerResult.landmarks` (normalised).
 * @param labels      `handedness[i][0].categoryName` per hand ("Left"/"Right").
 * @param poseImage   `PoseLandmarkerResult.landmarks[0]`, if a pose was found.
 * @param width       Image width in px (distances are measured in pixels).
 * @param height      Image height in px.
 */
export function handSides(
  handsImage: Landmark[][],
  labels: string[],
  poseImage: Landmark[] | undefined,
  width = 1,
  height = 1,
): HandSide[] {
  const byLabel = labels.map((l): HandSide => (l === "Left" ? "L" : "R"));
  const bodyL = poseImage?.[MP.LEFT_WRIST];
  const bodyR = poseImage?.[MP.RIGHT_WRIST];
  if (!bodyL || !bodyR || handsImage.length === 0) return byLabel;
  const d = (a: Landmark, b: Landmark) => Math.hypot((a.x - b.x) * width, (a.y - b.y) * height);
  const wrist = (i: number) => handsImage[i][MP_HAND.WRIST];
  if (handsImage.length === 1) return [d(wrist(0), bodyL) <= d(wrist(0), bodyR) ? "L" : "R"];
  // Two (or more — extras keep their label) detections: pick the pairing
  // with the smaller total distance.
  const ab = d(wrist(0), bodyL) + d(wrist(1), bodyR);
  const ba = d(wrist(0), bodyR) + d(wrist(1), bodyL);
  return [ab <= ba ? "L" : "R", ab <= ba ? "R" : "L", ...byLabel.slice(2)];
}

/**
 * Assign detected hands' world landmarks to anatomical slots by pose-wrist
 * proximity. Prefer `handsForDriver`, which also lifts the more accurate
 * image landmarks to 3D and fixes chirality slips.
 */
export function assignHands(
  handsImageLandmarks: Landmark[][],
  handsWorldLandmarks: WorldLandmark[][],
  poseImageLandmarks: Landmark[],
): { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } {
  const out: { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } = {};
  if (!poseImageLandmarks[MP.LEFT_WRIST] || !poseImageLandmarks[MP.RIGHT_WRIST]) return out;
  const sides = handSides(handsImageLandmarks, [], poseImageLandmarks);
  sides.slice(0, 2).forEach((s, i) => {
    if (s === "L" && !out.leftHand) out.leftHand = handsWorldLandmarks[i];
    if (s === "R" && !out.rightHand) out.rightHand = handsWorldLandmarks[i];
  });
  return out;
}

/**
 * Make a hand's 3D landmarks match the anatomical side they'll drive.
 *
 * MediaPipe's hand landmarks have the chirality of its own handedness label
 * (anatomical for non-mirrored images). When that label disagrees with the
 * side you assigned the hand to (e.g. via `assignHands`), the model has
 * almost always fallen for the depth ambiguity — a left hand palm-forward
 * and a right hand palm-back project to the same 2D silhouette. Reflecting
 * depth about the wrist yields the other interpretation, with the same
 * image projection and the requested chirality.
 *
 * @param world   21 hand worldLandmarks.
 * @param side    Anatomical side the hand will drive.
 * @param label   MediaPipe's `handedness[0].categoryName` ("Left" | "Right").
 */
export function matchHandChirality(
  world: WorldLandmark[],
  side: HandSide,
  label: string,
): WorldLandmark[] {
  const labelSide = label === "Left" ? "L" : label === "Right" ? "R" : side;
  if (labelSide === side) return world;
  const wz = world[MP_HAND.WRIST].z;
  return world.map(lm => ({ ...lm, z: 2 * wz - lm.z }));
}

/**
 * Camera-axis 3D hand points from MediaPipe's IMAGE landmarks: x, y scaled
 * to pixels and the relative depth z (documented as "roughly the same scale
 * as x") scaled by the image width. Same axes as `worldLandmarks` (+x right,
 * +y down, +z away from the camera), arbitrary scale — fine for the driver,
 * which only uses directions and angles.
 *
 * Measured on renders of known hand poses, these give markedly better joint
 * angles than `worldLandmarks` (≈23° vs ≈36° mean finger-joint error, palm
 * normal ≈18° vs ≈33°) — MediaPipe's image head is the one its 2D accuracy
 * comes from. Use them for `leftHand` / `rightHand`.
 *
 * @param image   21 hand landmarks (`HandLandmarkerResult.landmarks[i]`), normalised.
 * @param width   Pixel width of the image the landmarks refer to.
 * @param height  Pixel height of that image.
 */
export function handLandmarks3D(image: Landmark[], width: number, height: number): WorldLandmark[] {
  return image.map(l => ({ x: l.x * width, y: l.y * height, z: (l.z ?? 0) * width }));
}

/** The parts of a `HandLandmarkerResult` the driver path needs. */
export interface HandDetections {
  /** `result.landmarks` — normalised image coordinates. */
  landmarks: Landmark[][];
  /** `result.handedness[i][0].categoryName` per hand ("Left" | "Right"). */
  handedness: string[];
}

/**
 * MediaPipe Hand results → `leftHand` / `rightHand` for `landmarksToPoseDeltas`.
 *
 * The recommended path: anatomical side via `handSides`, 3D from the image
 * landmarks (`handLandmarks3D`), chirality matched to the side
 * (`matchHandChirality`). The first detection per side wins.
 *
 * @param det        Hand detections for this frame.
 * @param width      Image width in px.
 * @param height     Image height in px.
 * @param poseImage  Pose image landmarks, if available (better side assignment).
 */
export function handsForDriver(
  det: HandDetections,
  width: number,
  height: number,
  poseImage?: Landmark[],
): { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } {
  const out: { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } = {};
  const sides = handSides(det.landmarks, det.handedness, poseImage, width, height);
  sides.forEach((side, i) => {
    const key = side === "L" ? "leftHand" : "rightHand";
    if (out[key]) return;
    out[key] = matchHandChirality(handLandmarks3D(det.landmarks[i], width, height), side, det.handedness[i] ?? "");
  });
  return out;
}
