/**
 * Offline replay of the video demo's per-frame pipeline on cached MediaPipe
 * detections (tools/capture_sequences.ts): pose EMA, per-wrist hands through
 * `handsForDriver`, hand EMA + outlier reject + hold-last, driver with
 * `previousDeltas`. Pure TypeScript — runs in milliseconds, so driver and
 * filtering variants can be compared on identical input.
 */

import {
  allocBoneTransforms, forwardKinematics, type AnnyModel, type PoseDeltas,
} from "../../src/anny/index.js";
import {
  handsForDriver, PoseTracker, type DriverHands, type PoseInput, type WorldLandmark,
} from "../../src/mediapipe/index.js";
import { smoothLandmarks } from "../../demo/_shared/smoothing.js";
import type { Detection } from "../roundtrip/protocol.js";
import { PosedSkeleton, type RigExtras } from "../roundtrip/skeleton.js";
import type { Side } from "../roundtrip/metrics.js";

export interface Sequence {
  clip: string;
  fps: number;
  detections: Detection[];
}

export type Driver = (input: PoseInput, model: AnnyModel) => PoseDeltas;

export interface ReplayOptions {
  /** Demo constants (video.ts). */
  poseAlpha?: number;
  handAlpha?: number;
  /** Run hands every other frame like the demo. */
  handsEveryOther?: boolean;
  /** Pass hand quality to the driver (default true). */
  quality?: boolean;
  /**
   * "tracker": raw hand detections into `PoseTracker` (no landmark EMA,
   * outlier gate or hold for hands). Default "demo": the demo's filtering
   * into the stateless driver.
   */
  hands?: "demo" | "tracker";
}

export interface FrameResult {
  deltas: PoseDeltas;
  skeleton: PosedSkeleton;
  /** Whether each side had hand landmarks driving it this frame. */
  driven: Record<Side, boolean>;
  /** Palm length (wrist → middle MCP) of the raw detection in pixels, per side. */
  palmPx: Partial<Record<Side, number>>;
}

export function replay(
  seq: Sequence, model: AnnyModel, extras: RigExtras, driver: Driver, opts: ReplayOptions = {},
): FrameResult[] {
  if (opts.hands === "tracker") return replayTracked(seq, model, extras, opts);
  const poseA = opts.poseAlpha ?? 0.5, handA = opts.handAlpha ?? 0.35;
  let poseImg: WorldLandmark[] | null = null, poseWorld: WorldLandmark[] | null = null;
  let left: WorldLandmark[] | null = null, right: WorldLandmark[] | null = null;
  let held: DriverHands = {};
  let leftQ: DriverHands["leftHandQuality"], rightQ: DriverHands["rightHandQuality"];
  let prev: PoseDeltas | undefined;
  const out: FrameResult[] = [];
  seq.detections.forEach((det, i) => {
    const palmPx: Partial<Record<Side, number>> = {};
    if (det.pose) {
      poseImg = smoothLandmarks(poseImg, det.pose.image.map(l => ({ ...l, z: l.z ?? 0 })) as WorldLandmark[], poseA);
      poseWorld = smoothLandmarks(poseWorld, det.pose.world as WorldLandmark[], poseA);
      if (!opts.handsEveryOther || i % 2 === 0) {
        held = handsForDriver(
          { landmarks: det.hands.map(h => h.image), handedness: det.hands.map(h => (h.labelSide === "L" ? "Left" : "Right")) },
          det.width, det.height, poseImg,
        );
      }
      for (const h of det.hands) {
        palmPx[h.side] = Math.hypot((h.image[9].x - h.image[0].x) * det.width, (h.image[9].y - h.image[0].y) * det.height);
      }
      if (held.leftHand && !isHandOutlier(left, held.leftHand)) {
        left = smoothLandmarks(left, held.leftHand, handA);
        leftQ = held.leftHandQuality;
      }
      if (held.rightHand && !isHandOutlier(right, held.rightHand)) {
        right = smoothLandmarks(right, held.rightHand, handA);
        rightQ = held.rightHandQuality;
      }
    }
    const deltas = poseWorld
      ? driver({
        pose: poseWorld, leftHand: left ?? undefined, rightHand: right ?? undefined,
        ...(opts.quality === false ? {} : { leftHandQuality: leftQ, rightHandQuality: rightQ }),
        visibilityMin: 0.2, previousDeltas: prev,
      }, model)
      : (prev ?? new Array(model.boneCount).fill(null));
    prev = deltas;
    const bx = allocBoneTransforms(model.boneCount);
    forwardKinematics(model, deltas, bx);
    out.push({ deltas, skeleton: new PosedSkeleton(model, extras, bx), driven: { L: !!left, R: !!right }, palmPx });
  });
  return out;
}

/** The tracker path: pose EMA as in the demo, raw hands into `PoseTracker`. */
function replayTracked(seq: Sequence, model: AnnyModel, extras: RigExtras, opts: ReplayOptions): FrameResult[] {
  const poseA = opts.poseAlpha ?? 0.5;
  const tracker = new PoseTracker(model);
  let poseImg: WorldLandmark[] | null = null, poseWorld: WorldLandmark[] | null = null;
  const driven: Record<Side, boolean> = { L: false, R: false };
  let prev: PoseDeltas | undefined;
  return seq.detections.map((det, i) => {
    const palmPx: Partial<Record<Side, number>> = {};
    let hands: DriverHands = {};
    if (det.pose) {
      poseImg = smoothLandmarks(poseImg, det.pose.image.map(l => ({ ...l, z: l.z ?? 0 })) as WorldLandmark[], poseA);
      poseWorld = smoothLandmarks(poseWorld, det.pose.world as WorldLandmark[], poseA);
      if (!opts.handsEveryOther || i % 2 === 0) {
        hands = handsForDriver(
          { landmarks: det.hands.map(h => h.image), handedness: det.hands.map(h => (h.labelSide === "L" ? "Left" : "Right")) },
          det.width, det.height, poseImg,
        );
      }
      for (const h of det.hands) {
        palmPx[h.side] = Math.hypot((h.image[9].x - h.image[0].x) * det.width, (h.image[9].y - h.image[0].y) * det.height);
      }
    }
    if (hands.leftHand) driven.L = true;
    if (hands.rightHand) driven.R = true;
    const deltas = poseWorld
      ? tracker.update({ pose: poseWorld, ...hands, visibilityMin: 0.2 }, (i / seq.fps) * 1000)
      : (prev ?? new Array(model.boneCount).fill(null));
    prev = deltas;
    const bx = allocBoneTransforms(model.boneCount);
    forwardKinematics(model, deltas, bx);
    return { deltas, skeleton: new PosedSkeleton(model, extras, bx), driven: { ...driven }, palmPx };
  });
}

// ── The demo's previous hand filtering, kept as the comparison baseline ──

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
function isHandOutlier(
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
