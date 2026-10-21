/**
 * MediaPipe as a learned plausibility critic.
 *
 * MediaPipe's hand landmark model has learned what real hands look like. Show
 * it a render of a natural pose and it puts the landmarks where the joints
 * are; show it an impossible pose (backward knuckles, fingers through each
 * other, decoupled joints) and it "corrects" toward the nearest plausible
 * hand — so its 2D landmarks drift from the true projected joints. The mean
 * drift over several viewpoints, as a fraction of palm length, is an
 * implausibility score that shares nothing with the rule-based metrics.
 */

import type { PoseDeltas } from "../../src/anny/index.js";
import { fitCamera, R_FRONT, project } from "../roundtrip/camera.js";
import { compare2D, toPx, type P2, type Side } from "../roundtrip/metrics.js";
import { handCamera, type RunContext } from "../roundtrip/run.js";
import { handJointsAnny, PosedSkeleton } from "../roundtrip/skeleton.js";
import { allocBoneTransforms, forwardKinematics, lbs } from "../../src/anny/index.js";
import { axisAngle, mat3Mul, type M3, type V3 } from "../roundtrip/vec.js";

/** View rotations applied in camera space after the source-camera orientation. */
const VIEWS: ReadonlyArray<{ axis: V3; deg: number }> = [
  { axis: [0, 1, 0], deg: 0 },
  { axis: [0, 1, 0], deg: 60 },
  { axis: [0, 1, 0], deg: -60 },
  { axis: [0, 1, 0], deg: 180 },
  { axis: [1, 0, 0], deg: 50 },
  { axis: [1, 0, 0], deg: -50 },
];

export interface CriticScore {
  /** Mean 2D drift (fraction of palm length) over views where a hand was found. */
  drift: number;
  /** Views where MediaPipe found no hand at all. */
  missed: number;
  /** Views where MediaPipe labelled the hand as the other side. */
  wrongSide: number;
  perView: (number | null)[];
}

export async function criticScore(ctx: RunContext, deltas: PoseDeltas, side: Side): Promise<CriticScore> {
  const bx = allocBoneTransforms(ctx.model.boneCount);
  forwardKinematics(ctx.model, deltas, bx);
  const sk = new PosedSkeleton(ctx.model, ctx.extras, bx);
  const joints = handJointsAnny(sk, side);
  const { vertices } = lbs(ctx.model, bx);
  const verts: V3[] = ctx.handVerts[side].map(v => [vertices[v * 3], vertices[v * 3 + 1], vertices[v * 3 + 2]]);
  const perView: (number | null)[] = [];
  let missed = 0, wrongSide = 0;
  for (const view of VIEWS) {
    const R = mat3Mul(axisAngle(view.axis, view.deg * Math.PI / 180), R_FRONT as M3) as M3;
    const cam = view.deg === 0 ? handCamera(ctx, deltas, side) : fitCamera(verts, 384, 384, R, 30, 0.15);
    const { detection } = await ctx.client.renderAndDetect(deltas, cam, { handOnly: side }, { pose: false });
    const h = [...detection.hands].sort((a, b) => b.labelScore - a.labelScore)[0];
    if (!h) { missed++; perView.push(null); continue; }
    if (h.labelSide !== side) wrongSide++;
    const truth: P2[] = joints.map(p => project(cam, p).slice(0, 2) as P2);
    perView.push(compare2D(toPx(h.image, cam.W, cam.H), truth).err);
  }
  const found = perView.filter((x): x is number => x !== null);
  return {
    drift: found.length ? found.reduce((s, x) => s + x, 0) / found.length : 1,
    missed, wrongSide, perView,
  };
}
