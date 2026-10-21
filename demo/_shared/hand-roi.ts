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
 * crop is a 2D image transformation that doesn't affect 3D directions. Image
 * landmarks are mapped back to full-frame coordinates — including z, which
 * MediaPipe expresses in units of the *input* width and must be rescaled.
 *
 * Why a separate IMAGE-mode HandLandmarker: VIDEO mode keeps inter-frame
 * tracking state. If we feed it left-crop then right-crop in the same frame
 * the "palm jumps" between calls confuse the tracker. IMAGE mode treats each
 * crop independently, which is exactly what we want.
 */

import type { HandLandmarker, HandLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

import type { HandDetections, Landmark } from "../../src/mediapipe/index.js";

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
  /** Which pose wrist this crop is centred on (debug only — anatomical sides
   *  are assigned by `handsForDriver`). */
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

/** Lightweight debug record so callers (Playwright/devtools) can see what the ROI did. */
export interface HandRoiDebug {
  /** Per-wrist: did we attempt a crop, did the detector find a hand. */
  attempts: Array<{ side: "L" | "R"; cropPx: number; wristVis: number; found: boolean }>;
  /** Last offscreen canvas (the crop sent to the detector) for visual inspection. */
  offscreen: HTMLCanvasElement;
}

/**
 * ROI hand detector. Construct once, call `detect()` per frame; feed the
 * result to `handsForDriver` together with the pose's image landmarks.
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
  detect(video: HTMLVideoElement, poseImgLm: Landmark[]): HandDetections {
    const out: HandDetections = { landmarks: [], handedness: [] };

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
      this.lastDebug.attempts.push({ side: c.side, cropPx, wristVis, found: !!result });
      if (!result) continue;

      // Overlapping crops (hands together) can both find the same hand: keep
      // a second detection only if its wrist is clearly elsewhere.
      const dup = out.landmarks.some(prev =>
        Math.hypot((prev[0].x - result.imageLm[0].x) * vW, (prev[0].y - result.imageLm[0].y) * vH)
          < 0.15 * shoulderWidthNorm * Math.max(vW, vH));
      if (dup) continue;
      out.landmarks.push(result.imageLm);
      out.handedness.push(result.label);
    }
    // Anatomical sides are assigned downstream by `handsForDriver`, from the
    // pose's own wrists — not from which crop a hand came out of.
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
  ): { imageLm: Landmark[]; label: string } | null {
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
    const handImg = res.landmarks[0];

    // Map crop-normalised landmarks back to the full frame. z is relative
    // depth in units of the crop width; re-express it in frame-width units
    // like x so `handLandmarks3D` scales it correctly.
    const mapped = handImg.map((lm: NormalizedLandmark): Landmark => ({
      x: (sx + lm.x * cropPx) / vW,
      y: (sy + lm.y * cropPx) / vH,
      z: lm.z * cropPx / vW,
      visibility: lm.visibility,
    }));
    return { imageLm: mapped, label: res.handedness[0]?.[0]?.categoryName ?? "" };
  }
}
