/** MediaPipe landmark types shared across the driver. */

import type { PoseDeltas } from "../anny/types.js";

/** A MediaPipe image-space landmark (normalized 0..1 with depth). */
export interface Landmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

/** A MediaPipe world-space landmark (metric, hip-centered). */
export interface WorldLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/**
 * How much a hand reading can be trusted. `handsForDriver` fills it in.
 *
 * MediaPipe's finger angles and palm orientation degrade steeply as the hand
 * shrinks in the image (measured: tools/hand_trust.ts) — on a full-body
 * video frame a hand is often 10–30 px long, where its finger angles are
 * worse than simply assuming a relaxed hand.
 */
export interface HandQuality {
  /** Palm length (wrist → middle-finger knuckle) in image pixels. */
  palmPx: number;
  /**
   * Whether MediaPipe's handedness label matches the side the hand was
   * assigned to. A mismatch means its 3D reading was mirrored (and repaired
   * by a depth flip), which leaves the palm orientation unreliable.
   */
  labelAgrees: boolean;
}

/** Input bundle for the driver. */
export interface PoseInput {
  /**
   * 33 MediaPipe Pose worldLandmarks (metric, hip-centered). Drives trunk,
   * head and limbs. Optional: hands alone still pose the hand in world.
   */
  pose?: WorldLandmark[];
  /**
   * 21 MediaPipe Hand worldLandmarks for the subject's anatomical LEFT hand.
   * Route detections by anatomy (e.g. `assignHands`), not by position.
   */
  leftHand?: WorldLandmark[];
  /** 21 MediaPipe Hand worldLandmarks for the anatomical RIGHT hand. */
  rightHand?: WorldLandmark[];
  /**
   * Reliability of each hand reading. When given, the reading is fused with
   * a prior over natural hands and wrists — the smaller the hand in the
   * image, the more the prior decides. Without it, readings are taken at
   * face value (within anatomical limits). `handsForDriver` provides both.
   */
  leftHandQuality?: HandQuality;
  rightHandQuality?: HandQuality;
  /**
   * Flip Anny X axis. Only set true if your upstream is already mirrored.
   * Standard MediaPipe output is subject-relative — leave false. Default: false.
   */
  mirrorX?: boolean;
  /** Minimum landmark visibility to drive a bone (default 0.5). */
  visibilityMin?: number;
  /**
   * Maximum delta rotation angle per limb bone, radians. Bounds the damage a
   * single bad landmark can do. Default π (no limit beyond the anatomical
   * joint limits the driver always applies).
   */
  maxAngleRad?: number;
  /**
   * Per-bone deltas from the previous frame, used as a fallback for bones
   * the current frame can't drive (low landmark visibility, missing hand
   * detection). Without this, an under-threshold bone falls back to its
   * parent's chain pose every frame — making the limb snap to the spine
   * whenever MP's confidence dips below the threshold, which looks like a
   * jerk in the rendered body. With stickiness, the limb holds its last
   * driven pose until fresh landmarks come back.
   *
   * Pass the array returned by the previous `landmarksToPoseDeltas` call.
   * Length must equal the model's bone count; otherwise ignored.
   */
  previousDeltas?: PoseDeltas;
}
