/**
 * ROI-cropped hand detection.
 *
 * `HandLandmarker` is trained on close-up hand images (~20-40% of the frame).
 * On a full-body shot the hand is often <5% of the frame and the detector
 * returns nothing — verified empirically on the boxer/dancer fixtures.
 *
 * Workaround: use the *pose* landmark for each wrist (which works fine on
 * full-body shots) to crop a tight region around each hand, then run an
 * IMAGE-mode `HandLandmarker` on the crop. The hand now fills the input,
 * detection succeeds, and we map the image-space landmarks back to full-frame
 * coords. World-space hand landmarks pass through unchanged — they're metric
 * vectors relative to the wrist in the same world frame as the pose, and the
 * crop is a 2D image transformation that doesn't affect 3D directions.
 *
 * Why a separate IMAGE-mode HandLandmarker: VIDEO mode keeps inter-frame
 * tracking state. If we feed it left-crop then right-crop in the same frame
 * the "palm jumps" between calls confuse the tracker. IMAGE mode treats each
 * crop independently, which is exactly what we want.
 */

import type { HandLandmarker, HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

import type { Landmark, WorldLandmark } from "../../src/mediapipe/index.js";

/**
 * ROI detection result, indexed by anatomical side.
 *
 * Side comes from MediaPipe's own `result.handedness` label, not from which
 * crop the detection came from. The HandLandmarker model classifies a hand
 * as Left or Right by anatomy (thumb side, finger geometry), which is
 * robust to the boxing-guard case where the two crops overlap and either
 * crop can capture either hand. Crop-source labelling was wrong whenever
 * the right-wrist crop happened to contain more of the left hand.
 */
export interface RoiHandResult {
  leftHand?:  { image: Landmark[]; world: WorldLandmark[] };
  rightHand?: { image: Landmark[]; world: WorldLandmark[] };
}

export interface HandRoiOptions {
  /**
   * Crop side length as a fraction of shoulder width. The hand fits in a box
   * roughly 1.0–1.5× the shoulder width when fingers spread. Default 1.6 for
   * a little safety margin.
   */
  cropSizeRatio?: number;
  /** Minimum wrist visibility from pose to attempt a hand crop. Default 0.2. */
  minWristVisibility?: number;
  /** Size of the offscreen canvas the crop is rendered into. Default 256. */
  offscreenSize?: number;
}

interface CropSpec {
  /** Pose's wrist landmark index. */
  wrist: number;
  /** Pose's knuckle landmark indices for the same hand — used to centre the crop on
   *  the actual hand region (the wrist alone biases the crop into the forearm). */
  knuckles: readonly number[];
  /** Which side of the body this crop is positioned on (informational, used only
   *  for debug/dedup tie-breaking; NOT used to assign anatomical handedness). */
  side: "L" | "R";
}

const CROPS: ReadonlyArray<CropSpec> = [
  // Pose's LEFT_PINKY / LEFT_INDEX / LEFT_THUMB are at 17/19/21.
  { wrist: 15, knuckles: [17, 19, 21], side: "L" },
  // Pose's RIGHT_PINKY / RIGHT_INDEX / RIGHT_THUMB at 18/20/22.
  { wrist: 16, knuckles: [18, 20, 22], side: "R" },
];

/** Component-wise average of a small set of landmarks. */
function avg(lms: ReadonlyArray<Landmark>, k: "x" | "y"): number {
  let s = 0;
  for (const lm of lms) s += lm[k];
  return s / lms.length;
}

// Five wrist→knuckle directions that characterise the hand's overall pose.
// Used by `isHandOutlier` to spot noisy detections — they tend to splay one
// finger wildly while the others stay still, so a large angular delta on any
// of these signals "this is noise, not real motion".
//   thumb_CMC (1), index_MCP (5), middle_MCP (9), ring_MCP (13), pinky_MCP (17)
const HAND_DIR_TARGETS = [1, 5, 9, 13, 17] as const;

function angleDeg(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
  const aLen = Math.hypot(ax, ay, az) || 1;
  const bLen = Math.hypot(bx, by, bz) || 1;
  const dot  = (ax*bx + ay*by + az*bz) / (aLen * bLen);
  return Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI;
}

/**
 * Decide whether `fresh` is too far from `prev` to be a plausible next frame.
 *
 * Compares the wrist→knuckle direction for five fingers; if the *max* angular
 * delta across them exceeds `maxAngleDeg`, treat `fresh` as a noisy detection
 * and reject it (the caller should hold the previous smoothed pose instead).
 *
 * `prev` null means "no history yet" — always accept the first detection.
 *
 * Why max rather than mean: real hand motion moves fingers together (all
 * fingers swing through similar deltas during a wave or punch). Noise typically
 * jerks one or two fingers far while the others stay put — the max delta
 * captures that asymmetry, while a mean would average it away.
 */
export function isHandOutlier(
  prev: ReadonlyArray<WorldLandmark> | null,
  fresh: ReadonlyArray<WorldLandmark>,
  maxAngleDeg = 60,
): boolean {
  if (!prev || prev.length === 0) return false;
  const wF = fresh[0], wP = prev[0];   // MP_HAND.WRIST = 0
  if (!wF || !wP) return false;

  let maxDelta = 0;
  for (const k of HAND_DIR_TARGETS) {
    const f = fresh[k], p = prev[k];
    if (!f || !p) continue;
    const d = angleDeg(
      f.x - wF.x, f.y - wF.y, f.z - wF.z,
      p.x - wP.x, p.y - wP.y, p.z - wP.z,
    );
    if (d > maxDelta) maxDelta = d;
    if (maxDelta > maxAngleDeg) return true;   // early out
  }
  return false;
}

/** Lightweight debug record so callers (Playwright/devtools) can see what the ROI did. */
export interface HandRoiDebug {
  /** Per-wrist: did we attempt a crop, did the detector find a hand. */
  attempts: Array<{ side: "L" | "R"; cropPx: number; wristVis: number; found: boolean }>;
  /** Last offscreen canvas (the crop sent to the detector) for visual inspection. */
  offscreen: HTMLCanvasElement;
}

/**
 * ROI hand detector. Construct once, call `detect()` per frame.
 *
 * Returns hands keyed by their anatomical side, derived from the source crop
 * — the L-crop is centred on pose's `LEFT_WRIST` so any hand it produces is
 * the anatomical left hand. This bypasses the image-space proximity
 * disambiguation that `assignHands` would otherwise perform, which is
 * critical when the two crops overlap (boxing guard, hands clasped, etc.):
 * proximity would gladly pair the same physical hand with both sides.
 */
export class HandRoiDetector {
  private off: HTMLCanvasElement;
  private offCtx: CanvasRenderingContext2D;
  private cropSizeRatio: number;
  private minWristVis: number;
  /** Last call's per-wrist breakdown; for debugging from devtools/Playwright. */
  lastDebug: HandRoiDebug;

  constructor(private landmarker: HandLandmarker, opts: HandRoiOptions = {}) {
    this.cropSizeRatio = opts.cropSizeRatio ?? 1.6;
    this.minWristVis   = opts.minWristVisibility ?? 0.2;
    const size         = opts.offscreenSize ?? 256;
    this.off           = document.createElement("canvas");
    this.off.width = size; this.off.height = size;
    this.offCtx        = this.off.getContext("2d")!;
    this.lastDebug     = { attempts: [], offscreen: this.off };
  }

  /**
   * Run hand detection on per-wrist crops of the current video frame.
   *
   * @param video      Source `<video>` or `<canvas>` to crop from.
   * @param poseImgLm  Pose Landmarker's image-space landmarks (33). Used to
   *                   centre each crop on the hand and size it from the
   *                   wrist→knuckle distance. Pass the *smoothed* landmarks
   *                   so the crop is stable across frames.
   */
  detect(video: HTMLVideoElement, poseImgLm: Landmark[]): RoiHandResult {
    const out: RoiHandResult = {};

    const vW = video.videoWidth  || 0;
    const vH = video.videoHeight || 0;
    if (vW === 0 || vH === 0) return out;

    // Fallback hand size for cases where knuckles aren't detected. Sized off
    // shoulder width, which gives a generous box that the detector can still
    // pick a hand out of when finger-knuckles are momentarily lost.
    const lSh = poseImgLm[11], rSh = poseImgLm[12];
    const shoulderWidthNorm = (lSh && rSh)
      ? Math.hypot(lSh.x - rSh.x, lSh.y - rSh.y)
      : 0.3;
    const fallbackPx = Math.max(vW, vH) * shoulderWidthNorm * this.cropSizeRatio;

    // Run both crops, then assign each detection to anatomical L or R by
    // comparing the detected hand's image-space wrist to pose's LEFT_WRIST
    // (15) and RIGHT_WRIST (16). Pose labels are anatomical and unambiguous
    // — far more reliable than the hand-landmarker's handedness, which
    // assumes a mirrored input image (see MP docs) and silently inverts on
    // most real footage.
    interface Detection {
      image: Landmark[];
      world: WorldLandmark[];
      dL: number;           // distance from detected wrist to pose LEFT_WRIST
      dR: number;           // distance from detected wrist to pose RIGHT_WRIST
      cropSide: "L" | "R";  // debug only
    }
    const detections: Detection[] = [];
    const poseL = poseImgLm[15];
    const poseR = poseImgLm[16];

    this.lastDebug.attempts = [];
    for (const c of CROPS) {
      const wrist = poseImgLm[c.wrist];
      const wristVis = wrist?.visibility ?? 0;
      if (!wrist || wristVis < this.minWristVis) {
        this.lastDebug.attempts.push({ side: c.side, cropPx: 0, wristVis, found: false });
        continue;
      }

      const knuckles = c.knuckles
        .map(i => poseImgLm[i])
        .filter((lm): lm is Landmark => !!lm && (lm.visibility ?? 0) >= this.minWristVis);
      const kCenter = knuckles.length > 0
        ? { x: avg(knuckles, "x"), y: avg(knuckles, "y") }
        : null;
      const cxNorm = kCenter ? (wrist.x + kCenter.x) * 0.5 : wrist.x;
      const cyNorm = kCenter ? (wrist.y + kCenter.y) * 0.5 : wrist.y;

      const kDistNorm = kCenter
        ? Math.hypot(kCenter.x - wrist.x, kCenter.y - wrist.y)
        : 0;
      const knucklePx = kDistNorm * Math.max(vW, vH) * 4;
      const cropPx    = Math.max(96, knucklePx, fallbackPx);

      const result = this.detectOnCrop(video, cxNorm, cyNorm, cropPx, vW, vH);
      this.lastDebug.attempts.push({
        side: c.side, cropPx, wristVis, found: !!result,
      });
      if (!result) continue;

      const hw = result.imageLm[0];
      const dL = poseL ? Math.hypot(hw.x - poseL.x, hw.y - poseL.y) : Infinity;
      const dR = poseR ? Math.hypot(hw.x - poseR.x, hw.y - poseR.y) : Infinity;
      detections.push({
        image: result.imageLm, world: result.worldLm, dL, dR, cropSide: c.side,
      });
    }

    // Two-hand case: bipartite match — pick the assignment (detection 0→L,
    // detection 1→R) vs (0→R, 1→L) that minimises the sum of distances.
    // This handles the boxing-guard case where both crops capture either
    // hand: pose's anatomical wrists pull each detection to its real side.
    if (detections.length >= 2) {
      const a = detections[0], b = detections[1];
      const costAR = a.dL + b.dR;
      const costAL = a.dR + b.dL;
      if (costAR <= costAL) {
        out.leftHand  = { image: a.image, world: a.world };
        out.rightHand = { image: b.image, world: b.world };
      } else {
        out.rightHand = { image: a.image, world: a.world };
        out.leftHand  = { image: b.image, world: b.world };
      }
    } else if (detections.length === 1) {
      const d = detections[0];
      const slot = { image: d.image, world: d.world };
      if (d.dL <= d.dR) out.leftHand  = slot;
      else              out.rightHand = slot;
    }
    return out;
  }

  /** Crop a square region around (cxNorm, cyNorm), run the detector, map results back. */
  private detectOnCrop(
    video: HTMLVideoElement,
    cxNorm: number,
    cyNorm: number,
    cropPx: number,
    vW: number,
    vH: number,
  ): { imageLm: Landmark[]; worldLm: WorldLandmark[] } | null {
    const cx = cxNorm * vW;
    const cy = cyNorm * vH;
    const half = cropPx / 2;
    const sx = cx - half;
    const sy = cy - half;

    // Render the crop into the offscreen canvas at full canvas size.
    // drawImage handles source rects partly outside the video — those pixels
    // come back as transparent, which is fine for hand detection.
    const dst = this.off.width;
    this.offCtx.fillStyle = "#000";
    this.offCtx.fillRect(0, 0, dst, dst);
    this.offCtx.drawImage(video, sx, sy, cropPx, cropPx, 0, 0, dst, dst);

    let res: HandLandmarkerResult;
    try {
      res = this.landmarker.detect(this.off);
    } catch {
      return null;
    }

    if (!res.landmarks || res.landmarks.length === 0) return null;
    // numHands: 1 in this detector, so res.landmarks[0] is the only hand.
    const handImg   = res.landmarks[0];
    const handWorld = res.worldLandmarks[0];

    // Map image-space landmarks (which are normalised to the 256×256 crop)
    // back to the full video frame's normalised coords.
    const mapped = handImg.map((lm: NormalizedLandmark): Landmark => ({
      x: (sx + lm.x * cropPx) / vW,
      y: (sy + lm.y * cropPx) / vH,
      z: lm.z,
      visibility: lm.visibility,
    }));

    // worldLandmarks are metric, wrist-origin, in the camera frame — no remap.
    return { imageLm: mapped, worldLm: handWorld };
  }
}
