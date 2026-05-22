/**
 * Forward kinematics for Anny skeleton.
 *
 * Given rest bone poses (B×4×4 world-space) and local delta rotations
 * (one 3×3 per bone, or null for identity), computes world-space bone
 * transforms to feed into LBS.
 *
 * Matches Anny's parallel_forward_kinematic logic:
 *   T[b] = restPose[b] @ delta[b]
 *   pose[b] = transform[parent] @ T[b]
 *   transform[b] = pose[b] @ restPose[b]^-1
 */

import { mat4Mul, mat4InvertRigid, mat4FromRot3Translation, mat4Slice } from "./math.js";
import type { AnnyModel, Mat4, PoseDeltas } from "./types.js";

/** Allocate bone transform buffer: (B×16) Float32Array. */
export function allocBoneTransforms(boneCount: number): Float32Array {
  return new Float32Array(boneCount * 16);
}

// Scratch buffers reused across forwardKinematics calls (no allocation per frame
// after the first call). Module-local — safe because forwardKinematics is sync.
const _scratchD4      = new Float32Array(16);  // delta embedded in 4×4
const _scratchT       = new Float32Array(16);  // rest @ delta
const _scratchPose    = new Float32Array(16);  // parent @ T
const _scratchInvRest = new Float32Array(16);  // inverse of rest

/**
 * Run FK and write world-space bone transforms into `out` (B×16).
 *
 * Zero-allocation — call with the same `out` buffer each frame.
 *
 * @param model    Loaded AnnyModel (restBonePoses, boneParents).
 * @param deltas   PoseDeltas: length-B array, each entry is a 3×3 row-major
 *                 rotation matrix (Float32Array[9]) or null for identity.
 *                 Typically only a subset of bones need non-null deltas.
 * @param out      Pre-allocated (B×16) buffer — reuse each frame.
 */
export function forwardKinematics(
  model: AnnyModel,
  deltas: PoseDeltas,
  out: Float32Array
): void {
  const { boneCount, boneParents, restBonePoses } = model;

  for (let b = 0; b < boneCount; b++) {
    const rest  = mat4Slice(restBonePoses, b);
    const delta = deltas[b];

    // T = rest @ delta  (T points at _scratchT if delta non-null, else at rest)
    let T: Mat4;
    if (delta !== null) {
      mat4FromRot3Translation(delta, 0, 0, 0, _scratchD4);
      mat4Mul(rest, _scratchD4, _scratchT);
      T = _scratchT;
    } else {
      T = rest;
    }

    // pose = parentTransform @ T   (or = T for root)
    const parentId = boneParents[b];
    let pose: Mat4;
    if (parentId < 0) {
      pose = T;
    } else {
      const parentTransform = mat4Slice(out, parentId);
      mat4Mul(parentTransform, T, _scratchPose);
      pose = _scratchPose;
    }

    // transform[b] = pose @ inv(rest) — written directly into the output slot.
    // Safe: out[b*16..] is not aliased by pose, T, or parentTransform (parentId < b).
    mat4InvertRigid(rest, _scratchInvRest);
    mat4Mul(pose, _scratchInvRest, out.subarray(b * 16, b * 16 + 16));
  }
}

/** Build identity PoseDeltas array (length = boneCount, all null). */
export function identityDeltas(boneCount: number): PoseDeltas {
  return new Array<null>(boneCount).fill(null);
}

/**
 * Set a delta rotation on a named bone.
 * No-op if the bone name is not found (avoids throws in hot loops).
 */
export function setDelta(
  deltas: PoseDeltas,
  model: AnnyModel,
  boneName: string,
  rot3x3: Float32Array
): void {
  const idx = model.boneLabels.indexOf(boneName);
  if (idx >= 0) deltas[idx] = rot3x3;
}

/** Convenience: set the same rotation on multiple bone names. */
export function setDeltas(
  deltas: PoseDeltas,
  model: AnnyModel,
  entries: { bone: string; rot: Float32Array }[]
): void {
  for (const { bone, rot } of entries) setDelta(deltas, model, bone, rot);
}

/** Return a dict mapping bone label → index for fast lookup. */
export function buildBoneIndex(model: AnnyModel): Map<string, number> {
  const map = new Map<string, number>();
  for (let i = 0; i < model.boneLabels.length; i++)
    map.set(model.boneLabels[i], i);
  return map;
}
