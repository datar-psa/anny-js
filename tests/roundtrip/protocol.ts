/**
 * Types shared between the Node-side validator and the browser harness page
 * (`tests/browser/harness.ts`). Everything here crosses `page.evaluate`, so it
 * must stay plain JSON (no typed arrays, no classes).
 */

import type { Landmark, WorldLandmark } from "../../src/mediapipe/index.js";

export type HandSide = "L" | "R";

export interface HandDetection {
  /** Anatomical side as assigned by the harness (pose proximity, else label). */
  side: HandSide;
  /** MediaPipe's handedness label (anatomical for non-mirrored images). */
  labelSide: HandSide;
  labelScore: number;
  image: Landmark[];
  world: WorldLandmark[];
  /** "roi" = detected on a pose-wrist crop, "full" = full-image detector. */
  source: "roi" | "full";
}

export interface Detection {
  width: number;
  height: number;
  pose?: { image: Landmark[]; world: WorldLandmark[] };
  hands: HandDetection[];
}

/**
 * Pinhole camera. `R` maps Anny world → camera (row-major 3×3); camera axes
 * are MediaPipe's: +x right, +y down, +z forward (into the scene). `C` is the
 * camera centre in Anny world.
 */
export interface CameraSpec {
  W: number;
  H: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  R: number[];
  C: [number, number, number];
}

export interface RenderOptions {
  /** Only draw this side's forearm + hand (hand close-ups). */
  handOnly?: HandSide;
  /** Background RGB 0..1. */
  background?: [number, number, number];
}

export interface DetectOptions {
  /** Run the pose landmarker (default true); false for hand close-ups. */
  pose?: boolean;
  /**
   * Bilinearly upscale the image to this width before detection — mimics the
   * demo's ROI path, which feeds MediaPipe a 256 px crop of a few-pixel hand.
   * Landmarks stay normalised to the original image.
   */
  upscaleTo?: number;
  /**
   * Test-time augmentation: shift (fractions of width/height), scale and
   * rotate the image about its centre before detection. Joint angles are
   * invariant to all three, so the spread of readings across augmentations
   * measures MediaPipe's instability on an image — no ground truth needed.
   * Landmarks are NOT mapped back (they refer to the augmented image).
   */
  augment?: { dx: number; dy: number; scale: number; rotDeg: number };
}

export type PoseModel = "lite" | "full" | "heavy";

export interface HarnessInit {
  poseModel: PoseModel;
}
