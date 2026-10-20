/**
 * MediaPipe detection for the harness: Pose (IMAGE mode) + hands.
 *
 * Hands are found two ways and merged:
 *   • ROI: when a pose is present, crop a square around each pose wrist and
 *     run an IMAGE-mode HandLandmarker on the crop (hands in full-body shots
 *     are too small for the full-frame detector).
 *   • full: a full-image HandLandmarker, for close-up hand photos with no body.
 *
 * Anatomical side: pose-wrist proximity when a pose exists; otherwise
 * MediaPipe's handedness label. NOTE: despite the MediaPipe docs ("assumes a
 * mirrored image — swap if not"), the Tasks-API label is ANATOMICAL for
 * ordinary non-mirrored photos, and the 3D world landmarks have the
 * chirality of that label. Verified on MediaPipe's own left_hands.jpg /
 * right_hands.jpg / victory.jpg and on renders of the rig's known L/R hands.
 */

import {
  FilesetResolver, HandLandmarker, PoseLandmarker,
  type HandLandmarkerResult, type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

import { handSides, MP } from "../../src/mediapipe/index.js";
import type { Landmark } from "../../src/mediapipe/index.js";
import type { Detection, HandDetection, HandSide, PoseModel } from "../roundtrip/protocol.js";

type Source = HTMLCanvasElement | HTMLImageElement;

export interface DetectOptions {
  /** Run the pose landmarker (default true). */
  pose?: boolean;
}

export class Detector {
  private constructor(
    private pose: PoseLandmarker,
    private handFull: HandLandmarker,
    private handRoi: HandLandmarker,
    private crop: HTMLCanvasElement,
  ) {}

  static async create(poseModel: PoseModel): Promise<Detector> {
    const vision = await FilesetResolver.forVisionTasks("/mp/wasm");
    const handOpts = (numHands: number) => ({
      baseOptions: { modelAssetPath: "/cache/mediapipe/hand_landmarker.task", delegate: "CPU" as const },
      runningMode: "IMAGE" as const,
      numHands,
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence: 0.4,
    });
    const [pose, handFull, handRoi] = await Promise.all([
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `/cache/mediapipe/pose_landmarker_${poseModel}.task`, delegate: "CPU" },
        runningMode: "IMAGE",
        numPoses: 1,
      }),
      HandLandmarker.createFromOptions(vision, handOpts(2)),
      HandLandmarker.createFromOptions(vision, handOpts(1)),
    ]);
    const crop = document.createElement("canvas");
    crop.width = crop.height = 256;
    return new Detector(pose, handFull, handRoi, crop);
  }

  /**
   * @param opts.pose  false for hand close-ups: the pose model happily
   *   hallucinates a body around a lone forearm, and ROI crops around its
   *   fake wrists then miss the real hand.
   */
  detect(src: Source, opts: DetectOptions = {}): Detection {
    const width = src.width, height = src.height;
    const poseRes = opts.pose === false ? { landmarks: [], worldLandmarks: [] } : this.pose.detect(src);
    const pose = poseRes.landmarks.length > 0
      ? { image: plain(poseRes.landmarks[0]), world: plain(poseRes.worldLandmarks[0]) }
      : undefined;

    const hands = pose ? this.roiHands(src, pose.image) : [];
    if (hands.length === 0) hands.push(...this.fullHands(src, pose?.image));
    return { width, height, pose, hands };
  }

  private fullHands(src: Source, poseImg?: Landmark[]): HandDetection[] {
    const res = this.handFull.detect(src);
    const out = res.landmarks.map((_, i) => toHand(res, i, "full", lm => lm));
    if (poseImg) assignByPose(out, poseImg, src.width, src.height);
    return out;
  }

  private roiHands(src: Source, poseImg: Landmark[]): HandDetection[] {
    const W = src.width, H = src.height;
    const sh = Math.hypot(
      (poseImg[MP.LEFT_SHOULDER].x - poseImg[MP.RIGHT_SHOULDER].x) * W,
      (poseImg[MP.LEFT_SHOULDER].y - poseImg[MP.RIGHT_SHOULDER].y) * H,
    );
    const found: HandDetection[] = [];
    for (const side of ["L", "R"] as const) {
      const w = poseImg[side === "L" ? MP.LEFT_WRIST : MP.RIGHT_WRIST];
      const knuckles = side === "L"
        ? [MP.LEFT_INDEX, MP.LEFT_PINKY, MP.LEFT_THUMB]
        : [MP.RIGHT_INDEX, MP.RIGHT_PINKY, MP.RIGHT_THUMB];
      const kx = knuckles.reduce((s, k) => s + poseImg[k].x, 0) / 3;
      const ky = knuckles.reduce((s, k) => s + poseImg[k].y, 0) / 3;
      const cx = (w.x + kx) / 2 * W, cy = (w.y + ky) / 2 * H;
      const handPx = Math.hypot((kx - w.x) * W, (ky - w.y) * H);
      const size = Math.max(64, 4 * handPx, 0.9 * sh);
      const hit = this.detectCrop(src, cx - size / 2, cy - size / 2, size);
      if (hit) found.push(hit);
    }
    // Both crops can see the same physical hand (hands together): keep the
    // pair only if their wrists are distinct.
    if (found.length === 2) {
      const a = found[0].image[0], b = found[1].image[0];
      if (Math.hypot((a.x - b.x) * W, (a.y - b.y) * H) < 0.15 * sh) found.pop();
    }
    assignByPose(found, poseImg, W, H);
    return found;
  }

  private detectCrop(src: Source, sx: number, sy: number, size: number): HandDetection | null {
    const ctx = this.crop.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    const S = this.crop.width;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, S, S);
    ctx.drawImage(src, sx, sy, size, size, 0, 0, S, S);
    const res = this.handRoi.detect(this.crop);
    if (res.landmarks.length === 0) return null;
    // z is relative depth in units of the input width — the crop's, here.
    // Re-express it in units of the full image width like x.
    return toHand(res, 0, "roi", lm => ({
      x: (sx + lm.x * size) / src.width,
      y: (sy + lm.y * size) / src.height,
      z: lm.z * size / src.width,
    }));
  }
}

function plain<T extends { x: number; y: number; z: number; visibility?: number }>(a: T[]): T[] {
  return a.map(l => ({ ...l }));
}

function toHand(
  res: HandLandmarkerResult, i: number, source: "roi" | "full",
  mapImage: (lm: NormalizedLandmark) => Landmark,
): HandDetection {
  const cat = res.handedness[i][0];
  const labelSide: HandSide = cat.categoryName === "Left" ? "L" : "R";
  return {
    side: labelSide,
    labelSide,
    labelScore: cat.score,
    image: res.landmarks[i].map(mapImage),
    world: plain(res.worldLandmarks[i]),
    source,
  };
}

/** Anatomical sides via the library's own `handSides` (pose proximity, else label). */
function assignByPose(hands: HandDetection[], poseImg: Landmark[], W: number, H: number): void {
  const sides = handSides(
    hands.map(h => h.image), hands.map(h => (h.labelSide === "L" ? "Left" : "Right")), poseImg, W, H,
  );
  hands.forEach((h, i) => { h.side = sides[i]; });
}
