/** One naturalness report per posed hand: anatomy + physical (+ critic when a browser is available). */

import { identityDeltas, type AnnyModel, type PoseDeltas } from "../../src/anny/index.js";
import { handAngles, type Side } from "../roundtrip/metrics.js";
import { handJointsAnny, toMpWorld, type RigExtras } from "../roundtrip/skeleton.js";
import { posedSkeleton } from "../roundtrip/synthetic-landmarks.js";
import { anatomyScore, type AnatomyScore } from "./anatomy.js";
import { physicalScore, type PhysicalRig, type PhysicalScore } from "./physical.js";

export interface HandReport { anatomy: AnatomyScore; physical: PhysicalScore; flex: number[][]; abd: number[] }

export class NaturalnessScorer {
  private restAbd: Record<Side, number[]>;
  constructor(private model: AnnyModel, private extras: RigExtras, private physical: PhysicalRig) {
    const rest = posedSkeleton(model, extras, identityDeltas(model.boneCount));
    this.restAbd = {
      L: handAngles(toMpWorld(handJointsAnny(rest, "L")), "L").abd,
      R: handAngles(toMpWorld(handJointsAnny(rest, "R")), "R").abd,
    };
  }

  score(deltas: PoseDeltas, side: Side): HandReport {
    const sk = posedSkeleton(this.model, this.extras, deltas);
    const a = handAngles(toMpWorld(handJointsAnny(sk, side)), side);
    return {
      anatomy: anatomyScore(a.flex, a.abd, this.restAbd[side]),
      physical: physicalScore(this.physical, deltas, side),
      flex: a.flex, abd: a.abd,
    };
  }
}

/** ROC AUC of "higher score = implausible" for positives vs negatives (Mann–Whitney). */
export function auc(negatives: number[], positives: number[]): number {
  let wins = 0;
  for (const p of positives) for (const n of negatives) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (positives.length * negatives.length || 1);
}
