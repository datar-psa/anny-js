/**
 * Naturalness of a driven video: replay a clip's cached MediaPipe detections
 * through the demo pipeline and score every driven hand — anatomy, physics
 * (finger penetration, arm range of motion) and temporal behaviour.
 */

import type { AnnyModel } from "../../src/anny/index.js";
import { mean } from "../roundtrip/metrics.js";
import type { RigExtras } from "../roundtrip/skeleton.js";
import { replay, type Driver, type ReplayOptions, type Sequence } from "./replay.js";
import type { NaturalnessScorer } from "./score.js";
import { temporalScore } from "./temporal.js";

export interface SequenceScore {
  clip: string;
  side: string;
  frames: number;
  /** Mean / p90 rule-based anatomy excess (degrees beyond what hands do). */
  anatomy: number;
  anatomyP90: number;
  /** Fraction of driven frames with any implausibility: anatomy > 20, penetration > 3 mm, arm out of range. */
  implausible: number;
  penetration: number;
  armRom: number;
  twistP90: number;
  jitter: number;
  /** Frames where a joint jumps > 30° (fingers, or the forearm's roll). */
  snaps: number;
  longestFreeze: number;
}

const pct = (sorted: number[], q: number) => sorted[Math.floor(sorted.length * q)];

export function scoreSequence(
  seq: Sequence, ctx: { model: AnnyModel; extras: RigExtras; scorer: NaturalnessScorer }, driver: Driver, opts: ReplayOptions = {},
): SequenceScore[] {
  const res = replay(seq, ctx.model, ctx.extras, driver, opts);
  const out: SequenceScore[] = [];
  for (const side of ["L", "R"] as const) {
    const reps = res.map(r => (r.driven[side] ? ctx.scorer.score(r.deltas, side) : null));
    const ok = reps.filter((x): x is NonNullable<typeof x> => x !== null);
    if (!ok.length) continue;
    const anatomy = ok.map(r => r.anatomy.total).sort((a, b) => a - b);
    const twist = ok.map(r => Math.abs(r.physical.forearmTwist)).sort((a, b) => a - b);
    const fingers = temporalScore(reps.map(r => (r ? [...r.flex.flat(), ...r.abd] : null)));
    const roll = temporalScore(reps.map(r => (r ? [r.physical.forearmTwist] : null)));
    out.push({
      clip: seq.clip, side, frames: ok.length,
      anatomy: mean(anatomy), anatomyP90: pct(anatomy, 0.9),
      implausible: ok.filter(r => r.anatomy.total > 20 || r.physical.penetrationMm > 3 || r.physical.armRom > 0).length / ok.length,
      penetration: mean(ok.map(r => r.physical.penetrationMm)), armRom: mean(ok.map(r => r.physical.armRom)),
      twistP90: pct(twist, 0.9),
      jitter: fingers.jitter, snaps: fingers.snaps + roll.snaps, longestFreeze: fingers.longestFreeze,
    });
  }
  return out;
}

export function printSequences(rows: SequenceScore[], label = ""): void {
  if (label) console.log(`\n== ${label}`);
  console.log("clip    side frames  anatomy  p90  implaus%  pen(mm)  armRom  |twist|p90  jitter  snaps  freeze");
  for (const r of rows) {
    console.log(`${r.clip.padEnd(8)}${r.side.padEnd(5)}${String(r.frames).padStart(6)}${r.anatomy.toFixed(1).padStart(9)}${r.anatomyP90.toFixed(0).padStart(5)}${(r.implausible * 100).toFixed(0).padStart(10)}${r.penetration.toFixed(1).padStart(9)}${r.armRom.toFixed(1).padStart(8)}${r.twistP90.toFixed(0).padStart(12)}${r.jitter.toFixed(1).padStart(8)}${String(r.snaps).padStart(7)}${String(r.longestFreeze).padStart(8)}`);
  }
}
