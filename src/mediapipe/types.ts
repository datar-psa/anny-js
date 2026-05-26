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

/** Input bundle for the driver. */
export interface PoseInput {
  /** 33 MediaPipe Pose worldLandmarks (metric, hip-centered). Required. */
  pose: WorldLandmark[];
  /** 21 MediaPipe Hand worldLandmarks for the left hand. Optional. */
  leftHand?: WorldLandmark[];
  /** 21 MediaPipe Hand worldLandmarks for the right hand. Optional. */
  rightHand?: WorldLandmark[];
  /**
   * Flip Anny X axis. Only set true if your upstream is already mirrored.
   * Standard MediaPipe output is subject-relative — leave false. Default: false.
   */
  mirrorX?: boolean;
  /** Minimum landmark visibility to drive a bone (default 0.5). */
  visibilityMin?: number;
  /**
   * Maximum delta rotation angle per bone in radians, applied to `dir`
   * targets only. Bounds the damage a single bad landmark can do.
   * Default 2.5 rad (~143°).
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
