/**
 * Video: the driver with memory.
 *
 * Each hand's estimate (shape + wrist) is carried from frame to frame as a
 * Gaussian belief. Between frames it relaxes toward the natural-hand prior
 * (an Ornstein–Uhlenbeck process whose stationary distribution is exactly
 * that prior); each new MediaPipe reading updates it with the noise its
 * size warrants. That is a Kalman filter in which:
 *
 *   • a tiny, noisy hand is smoothed heavily and a large one barely at all —
 *     no hand-tuned smoothing factor;
 *   • a hand MediaPipe loses keeps moving plausibly and relaxes into a
 *     natural pose instead of freezing;
 *   • a blunder (a mirrored or garbled reading) is down-weighted by how
 *     surprising it is (Student-t update) instead of snapping the hand.
 *
 * Feed it raw per-frame detections (no landmark smoothing for hands).
 */

import type { AnnyModel, PoseDeltas } from "../anny/types.js";
import { predictBelief, type RelaxTimes } from "./hand-fusion.js";
import { solvePose, type HandBeliefs } from "./landmarks.js";
import type { PoseInput } from "./types.js";

export interface PoseTrackerOptions {
  /**
   * Correlation times (seconds) of hand shape and wrist: how long a pose
   * typically persists, which sets both how fast an unobserved hand relaxes
   * toward the natural prior and how far the filter lets it move between
   * frames. Shorter follows faster changes but lets single-frame blunders
   * through: at 1 s a one-frame garbled reading moves a hand ~40 % of the
   * way, a real change is followed within ~3 frames at 30 fps (see
   * tests/mediapipe/tracker.test.ts). Defaults: 1 / 0.8.
   */
  relax?: Partial<RelaxTimes>;
  /**
   * Forget a hand not seen for this long (seconds): it is no longer driven
   * and holds its last pose. Default 2.
   */
  forgetAfter?: number;
}

export class PoseTracker {
  private beliefs: HandBeliefs = {};
  private lastSeen: Partial<Record<"L" | "R", number>> = {};
  private previous?: PoseDeltas;
  private lastTime?: number;
  private readonly relax: RelaxTimes;
  private readonly forgetAfter: number;

  constructor(private readonly model: AnnyModel, opts: PoseTrackerOptions = {}) {
    this.relax = { shape: opts.relax?.shape ?? 1, wrist: opts.relax?.wrist ?? 0.8 };
    this.forgetAfter = opts.forgetAfter ?? 2;
  }

  /**
   * Drive one frame.
   *
   * @param input   This frame's landmarks (as for `landmarksToPoseDeltas`,
   *                ideally with hand quality from `handsForDriver`).
   * @param timeMs  Frame timestamp in milliseconds (video time or
   *                `performance.now()`); only differences matter.
   */
  update(input: PoseInput, timeMs: number): PoseDeltas {
    const t = timeMs / 1000;
    const dt = this.lastTime === undefined ? 0 : Math.max(0, t - this.lastTime);
    this.lastTime = t;
    const predicted: HandBeliefs = {};
    for (const side of ["L", "R"] as const) {
      const seen = side === "L" ? input.leftHand : input.rightHand;
      if (seen) this.lastSeen[side] = t;
      const b = this.beliefs[side];
      const stale = (t - (this.lastSeen[side] ?? -Infinity)) > this.forgetAfter;
      if (b && !stale) predicted[side] = predictBelief(b, dt, this.relax);
    }
    const out = solvePose({ ...input, previousDeltas: input.previousDeltas ?? this.previous }, this.model, predicted);
    this.beliefs = out.beliefs;
    this.previous = out.deltas;
    return out.deltas;
  }

  /** Forget everything (e.g. on a seek or a scene cut). */
  reset(): void {
    this.beliefs = {};
    this.lastSeen = {};
    this.previous = undefined;
    this.lastTime = undefined;
  }
}
