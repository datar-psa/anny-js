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
}

export interface MediaPipeOptions {
  /** Pose model variant. `lite` for live webcam (default), `heavy` for offline video. */
  poseModel?: PoseModelSize;
}

/**
 * Create a Pose + Hand landmarker pair sharing one Vision WASM fileset.
 * Both run on GPU when available.
 */
export async function createMediaPipe(opts: MediaPipeOptions = {}): Promise<MediaPipeBundle> {
  const poseUrl = POSE_MODELS[opts.poseModel ?? "lite"];
  const vision = await FilesetResolver.forVisionTasks(VISION_WASM);
  const [poseLandmarker, handLandmarker] = await Promise.all([
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
  ]);
  return { poseLandmarker, handLandmarker };
}
