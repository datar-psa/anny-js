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

/**
 * Run FK and write world-space bone transforms into `out` (B×16).
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

  // poses[b]     = world-space pose of bone b (rest @ delta composed up the chain)
  // transforms[b]= bone transform used for LBS: pose[b] @ inv(restPose[b])
  const poses      = new Float32Array(boneCount * 16);
  const transforms = out;

  const tmp  = new Float32Array(16);
  const tmp2 = new Float32Array(16);

  for (let b = 0; b < boneCount; b++) {
    const rest  = mat4Slice(restBonePoses, b);  // (4×4) rest pose for bone b
    const delta = deltas[b];

    // Build T = rest @ delta (delta is pure rotation applied in local space)
    let T: Mat4;
    if (delta !== null) {
      // Embed 3×3 rotation into 4×4 with zero translation
      const d4 = mat4FromRot3Translation(delta, 0, 0, 0, tmp2);
      T = mat4Mul(rest, d4, tmp);
    } else {
      T = rest;
    }

    const parentId = boneParents[b];
    let pose: Mat4;
    if (parentId < 0) {
      // Root bone — pose = T
      pose = new Float32Array(T);
    } else {
      // pose = transform[parent] @ T
      const parentTransform = mat4Slice(transforms, parentId);
      pose = mat4Mul(parentTransform, T, new Float32Array(16));
    }

    // Store pose (needed so children can access it via transforms[b] later)
    poses.set(pose, b * 16);

    // transform[b] = pose[b] @ inv(restPose[b])
    const invRest = mat4InvertRigid(rest);
    const boneTransform = mat4Mul(pose, invRest);
    transforms.set(boneTransform, b * 16);
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
