import { PoseLandmarker, HandLandmarker, FilesetResolver }
  from "@mediapipe/tasks-vision";

const VISION_VERSION = "0.10.14";
const VISION_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/wasm`;
const POSE_MODELS = {
  lite:  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  full:  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
  heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
} as const;
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export type PoseModelSize = keyof typeof POSE_MODELS;

export interface MediaPipeBundle {
  poseLandmarker: PoseLandmarker;
  handLandmarker: HandLandmarker;
  /**
   * IMAGE-mode hand landmarker, only present when `handMode: "image"` was
   * requested. Used by the ROI hand detector to run on each pose-wrist crop
   * as an independent still image (no inter-frame tracking state to confuse).
   */
  handLandmarkerImage?: HandLandmarker;
}

export interface MediaPipeOptions {
  /** Pose model variant. `lite` for live webcam (default), `heavy` for offline video. */
  poseModel?: PoseModelSize;
  /**
   * Hand detector mode.
   *   `video` (default): single VIDEO-mode HandLandmarker, runs on the full
   *     frame each call. Works when hands are large in the image (live demo).
   *   `image`: adds an extra IMAGE-mode HandLandmarker for the ROI detector
   *     to run on per-wrist crops. Use this when subjects are far from camera
   *     (offline-video demo) so the hands are too small for the full-frame
   *     detector to find.
   */
  handMode?: "video" | "image";
}

/**
 * Create a Pose + Hand landmarker pair sharing one Vision WASM fileset.
 * Both run on GPU when available.
 */
export async function createMediaPipe(opts: MediaPipeOptions = {}): Promise<MediaPipeBundle> {
  const poseUrl = POSE_MODELS[opts.poseModel ?? "lite"];
  const vision = await FilesetResolver.forVisionTasks(VISION_WASM);
  const tasks: Promise<unknown>[] = [
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: poseUrl, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    }),
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    }),
  ];
  // Each ROI crop is a separate small image, not a continuation of a video
  // stream — VIDEO mode would try to track between left-crop and right-crop
  // calls and get confused.
  //
  // 0.4 confidence is a deliberate compromise. Lower (e.g. 0.2) lets noisy
  // detections through — they pass to the smoother and then "hold-last"
  // keeps the noise visible. Higher (e.g. 0.6) drops too many partly-closed
  // fists/gloves that still have valid finger orientation. 0.4 keeps the
  // detector firing on real hands while rejecting the worst outliers.
  if (opts.handMode === "image") {
    tasks.push(HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
      runningMode: "IMAGE",
      numHands: 1,                 // one hand per ROI crop
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence:  0.4,
    }));
  }
  const results = await Promise.all(tasks);
  return {
    poseLandmarker: results[0] as PoseLandmarker,
    handLandmarker: results[1] as HandLandmarker,
    handLandmarkerImage: results[2] as HandLandmarker | undefined,
  };
}
