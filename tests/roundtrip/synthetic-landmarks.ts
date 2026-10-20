/**
 * MediaPipe-format landmarks read off a posed rig — the input side of the
 * pure-TypeScript recovery tests (no browser, no MediaPipe): pose the rig
 * with known deltas, emit the 33 pose + 2×21 hand landmarks MediaPipe would
 * ideally report, drive the rig from them, and check the driver recovers
 * the pose.
 */

import type { AnnyModel, PoseDeltas } from "../../src/anny/index.js";
import { allocBoneTransforms, forwardKinematics } from "../../src/anny/index.js";
import { MP, type WorldLandmark } from "../../src/mediapipe/index.js";
import { handJointsAnny, PosedSkeleton, poseJointsAnny, toMpWorld, type RigExtras } from "./skeleton.js";
import { add, mat3MulVec, sub, transpose, mat3Mul, type M3, type V3 } from "./vec.js";

export function posedSkeleton(model: AnnyModel, extras: RigExtras, deltas: PoseDeltas): PosedSkeleton {
  const bx = allocBoneTransforms(model.boneCount);
  forwardKinematics(model, deltas, bx);
  return new PosedSkeleton(model, extras, bx);
}

/** Face points relative to the head joint for a level, forward-facing head (Anny world). */
const FACE_REST: ReadonlyArray<[number, V3]> = [
  [MP.LEFT_EAR, [0.075, 0.0, 0.06]],
  [MP.RIGHT_EAR, [-0.075, 0.0, 0.06]],
  [MP.LEFT_EYE, [0.032, -0.09, 0.09]],
  [MP.RIGHT_EYE, [-0.032, -0.09, 0.09]],
  [MP.MOUTH_LEFT, [0.025, -0.09, 0.02]],
  [MP.MOUTH_RIGHT, [-0.025, -0.09, 0.02]],
  [MP.NOSE, [0.0, -0.11, 0.06]],
];

function headWorldDelta(sk: PosedSkeleton): M3 {
  // World rotation of the head relative to rest = posed axes · rest axesᵀ.
  const posed: M3 = [0, 1, 2].flatMap(c => sk.axis("head", c as 0 | 1 | 2)) as unknown as M3;
  const rest: M3 = [0, 1, 2].flatMap(c => sk.restAxis("head", c as 0 | 1 | 2)) as unknown as M3;
  // `posed`/`rest` hold axes as ROWS; R = posedᵀ · rest.
  return mat3Mul(transpose(posed), rest);
}

export interface SyntheticLandmarks {
  pose: WorldLandmark[];
  leftHand: WorldLandmark[];
  rightHand: WorldLandmark[];
}

/** Ideal MediaPipe output for a posed rig (MP world axes, visibility 1). */
export function landmarksOf(sk: PosedSkeleton): SyntheticLandmarks {
  const joints = poseJointsAnny(sk);
  const R = headWorldDelta(sk);
  const head = sk.head("head");
  for (const [i, off] of FACE_REST) joints[i] = add(head, mat3MulVec(R, off));
  // Heels: just behind the ankle along the foot.
  joints[MP.LEFT_HEEL] = add(joints[MP.LEFT_ANKLE]!, sub(joints[MP.LEFT_ANKLE]!, joints[MP.LEFT_FOOT_INDEX]!).map(x => x * 0.25) as V3);
  joints[MP.RIGHT_HEEL] = add(joints[MP.RIGHT_ANKLE]!, sub(joints[MP.RIGHT_ANKLE]!, joints[MP.RIGHT_FOOT_INDEX]!).map(x => x * 0.25) as V3);
  const filled = joints.map(p => p ?? head);
  const pose = toMpWorld(filled).map(p => ({ ...p, visibility: 1 }));
  return {
    pose,
    leftHand: toMpWorld(handJointsAnny(sk, "L")),
    rightHand: toMpWorld(handJointsAnny(sk, "R")),
  };
}
