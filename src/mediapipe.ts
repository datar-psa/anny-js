/**
 * MediaPipe Pose → Anny bone delta rotations.
 *
 * Converts 33 normalized MediaPipe landmarks to local delta rotations for
 * Anny's FK.  Anny FK applies delta as:
 *
 *   T = rest_bone_pose @ delta
 *   transform = parent_transform @ T @ inv(rest_bone_pose)
 *
 * So `delta` must be in the BONE-LOCAL frame, not world space.
 * For each limb bone whose Blender-convention Y-axis points along the bone:
 *
 *   local_target = inv(rest_bone_R) @ observed_world_dir
 *   delta = rotFromTo([0,1,0], local_target)
 *
 * where rest_bone_R is the 3×3 rotation block of rest_bone_poses[b].
 */

import { normalize3, rotFromTo, mulMat3, dot3, cross } from "./math.js";
import { identityDeltas } from "./fk.js";
import type { AnnyModel, PoseDeltas } from "./types.js";

/** MediaPipe Pose landmark indices. */
export const MP = {
  NOSE:             0,
  LEFT_EAR:         7,
  RIGHT_EAR:        8,
  LEFT_SHOULDER:   11,
  RIGHT_SHOULDER:  12,
  LEFT_ELBOW:      13,
  RIGHT_ELBOW:     14,
  LEFT_WRIST:      15,
  RIGHT_WRIST:     16,
  LEFT_PINKY:      17,
  RIGHT_PINKY:     18,
  LEFT_INDEX:      19,
  RIGHT_INDEX:     20,
  LEFT_THUMB:      21,
  RIGHT_THUMB:     22,
  LEFT_HIP:        23,
  RIGHT_HIP:       24,
  LEFT_KNEE:       25,
  RIGHT_KNEE:      26,
  LEFT_ANKLE:      27,
  RIGHT_ANKLE:     28,
} as const;

export interface Landmark { x: number; y: number; z?: number; visibility?: number }

// ── helpers ────────────────────────────────────────────────────────────────

/** Extract 3×3 rotation from flat (B×16) rest-bone-pose buffer. Row-major. */
function restRot3(restBonePoses: Float32Array, boneIdx: number): Float32Array {
  const b = boneIdx * 16;
  return new Float32Array([
    restBonePoses[b+ 0], restBonePoses[b+ 1], restBonePoses[b+ 2],
    restBonePoses[b+ 4], restBonePoses[b+ 5], restBonePoses[b+ 6],
    restBonePoses[b+ 8], restBonePoses[b+ 9], restBonePoses[b+10],
  ]);
}

/** Transpose a 3×3 (= inverse for pure rotation matrices). */
function transpose3(m: Float32Array): Float32Array {
  return new Float32Array([
    m[0], m[3], m[6],
    m[1], m[4], m[7],
    m[2], m[5], m[8],
  ]);
}

/** Multiply 3×3 row-major matrix by column 3-vector. */
function mulR3v3(m: Float32Array, v: Float32Array): Float32Array {
  return new Float32Array([
    m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
    m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
    m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
  ]);
}

const Y_AXIS = new Float32Array([0, 1, 0]);

/**
 * For a bone whose Y-axis (Blender convention) points along the limb,
 * compute the local delta rotation that swings the bone to align its world
 * Y-axis with `worldDir`.
 */
function limbDelta(
  worldDir: Float32Array,
  boneIdx: number,
  restBonePoses: Float32Array,
): Float32Array {
  const R    = restRot3(restBonePoses, boneIdx);
  const invR = transpose3(R);
  const localDir = normalize3(mulR3v3(invR, worldDir));
  return rotFromTo(Y_AXIS, localDir);
}

/**
 * Elbow/knee relative bend delta.
 *
 * Computes the rotation from the parent bone direction to the child bone
 * direction, both projected into the child bone's local rest frame.
 * Depth (Y) is zeroed before projection — MediaPipe depth is noisy and
 * causes spurious forward/backward bending at elbow and knee joints.
 *
 * Straight limb: both project to same local direction → identity delta. ✓
 */
/**
 * Parent-corrected child bone delta.
 *
 * `limbDelta` alone is wrong for child bones (forearm, shin) because the
 * parent's FK transform is NOT identity — it has already rotated the bone
 * chain.  Without correction, `limbDelta(forearmWorldDir)` over-rotates by
 * the parent's rotation.
 *
 * Fix: pre-multiply the child world direction by inv(R_transform_parent),
 * which "undoes" the parent's world rotation before mapping to local space.
 *
 *   inv(R_transform_parent) = R_par @ inv(R_delta_par) @ inv(R_par)
 *   R_delta_par              = rotFromTo(e_y, inv(R_par) @ parentWorldDir)
 *
 * For a straight limb (childWorldDir ≈ parentWorldDir):
 *   adjusted = inv(R_transform_par) @ parentWorldDir
 *   limbDelta(adjusted, ...) → arm is straight in Anny.  ✓
 */
function childLimbDelta(
  parentWorldDir: Float32Array,
  childWorldDir:  Float32Array,
  parentBoneIdx: number,
  childBoneIdx:  number,
  restBonePoses: Float32Array,
): Float32Array {
  // Parent rest rotation and its inverse
  const R_par    = restRot3(restBonePoses, parentBoneIdx);
  const invR_par = transpose3(R_par);

  // Delta applied to parent bone (from limbDelta logic)
  const parentLocal = normalize3(mulR3v3(invR_par, parentWorldDir));
  const R_delta_par = rotFromTo(Y_AXIS, parentLocal);         // 3×3
  const invDelta    = transpose3(R_delta_par);               // inv(R_delta_par)

  // inv(R_transform_parent) = R_par @ invDelta @ invR_par
  const inv_R_transform_par = mulMat3(R_par, mulMat3(invDelta, invR_par));

  // Adjust child world direction to cancel parent's accumulated transform
  const adjustedDir = normalize3(mulR3v3(inv_R_transform_par, childWorldDir));

  return limbDelta(adjustedDir, childBoneIdx, restBonePoses);
}

// ── public API ─────────────────────────────────────────────────────────────

/** Minimum visibility score for upper-body landmarks. */
const VIS_MIN = 0.5;
/** Lower threshold for legs — still track in partial occlusion / low light. */
const VIS_LEG = 0.3;

/**
 * Convert MediaPipe pose landmarks to Anny PoseDeltas.
 *
 * @param landmarks  33-element array from MediaPipe (normalized 0-1, z = depth).
 * @param model      Loaded AnnyModel.
 * @param boneIndex  Map from bone label → index (build once with buildBoneIndex).
 * @param mirrorX    Flip X axis. False for a standard front-facing webcam (MediaPipe
 *                   operates on raw camera data, unaffected by CSS scaleX(-1)).
 */
export function landmarksToPoseDeltas(
  landmarks: Landmark[],
  model: AnnyModel,
  boneIndex: Map<string, number>,
  mirrorX = true
): PoseDeltas {
  const deltas = identityDeltas(model.boneCount);
  const { restBonePoses } = model;

  // Anny native space is Z-up: X=anatomical-left, Y=depth(forward), Z=up.
  // MediaPipe: x right (flip for mirrorX), y increases downward, z = depth (positive=further).
  //   Anny X = -lm.x (mirrorX) or lm.x            → lateral (anatomical left = +)
  //   Anny Y = +(lm.z ?? 0)                          → depth (Anny front=-Y, lm.z positive=further=+Y back)
  //   Anny Z = -lm.y                                → vertical (up = + since image y is down)
  const mpVec = (idx: number): Float32Array => {
    const lm = landmarks[idx];
    const sx = mirrorX ? -lm.x : lm.x;
    return new Float32Array([sx, lm.z ?? 0, -lm.y]);
  };

  // True if both landmarks are visible enough to trust
  const vis = (a: number, b: number) =>
    (landmarks[a].visibility ?? 1) >= VIS_MIN &&
    (landmarks[b].visibility ?? 1) >= VIS_MIN;

  // Normalised direction A → B in Anny world space
  const boneDir = (a: number, b: number): Float32Array => {
    const va = mpVec(a), vb = mpVec(b);
    return normalize3(new Float32Array([vb[0]-va[0], vb[1]-va[1], vb[2]-va[2]]));
  };

  // Helper: look up bone index (returns -1 if missing)
  const bi = (name: string) => boneIndex.get(name) ?? -1;

  // Project direction onto XZ plane (zero out Y/depth).
  // Used for arms and legs — MediaPipe depth is noisy and the rest bone poses
  // have baked-in Y components that cause phantom forward rotation.
  const xzOnly = (v: Float32Array) => normalize3(new Float32Array([v[0], 0, v[2]]));

  // ── Hip and shoulder rotation frames ─────────────────────────────────────
  // Build full 3×3 rotation matrices for hips and shoulders.
  // Y = lean direction (XZ only, no depth pitch).
  // X = left–right vector with full 3D depth → encodes the twist/yaw.
  // Z = cross(X, Y) completes the right-handed frame.
  const lh = mpVec(MP.LEFT_HIP),      rh = mpVec(MP.RIGHT_HIP);
  const ls = mpVec(MP.LEFT_SHOULDER), rs = mpVec(MP.RIGHT_SHOULDER);
  const hipMidX = (lh[0]+rh[0])/2, hipMidZ = (lh[2]+rh[2])/2;
  const shdMidX = (ls[0]+rs[0])/2, shdMidZ = (ls[2]+rs[2])/2;

  const yW = normalize3(new Float32Array([shdMidX-hipMidX, 0, shdMidZ-hipMidZ]));

  const buildFrame = (xRaw: Float32Array): Float32Array => {
    const d = dot3(xRaw, yW);
    const x = normalize3(new Float32Array([xRaw[0]-d*yW[0], xRaw[1]-d*yW[1], xRaw[2]-d*yW[2]]));
    const z = normalize3(cross(x, yW));
    return new Float32Array([x[0],yW[0],z[0], x[1],yW[1],z[1], x[2],yW[2],z[2]]);
  };
  const R_shl = buildFrame(new Float32Array([ls[0]-rs[0], ls[1]-rs[1], ls[2]-rs[2]]));
  void buildFrame(new Float32Array([lh[0]-rh[0], lh[1]-rh[1], lh[2]-rh[2]]));  // R_hip reserved for future root-bone drive

  // ── Spine — shoulder twist only (spine01) ─────────────────────────────
  // Driving pelvis/spine05 with a rotation frame destroys the figure because
  // pelvis.L/R point LATERALLY (not up) — their rest orientation is incompatible
  // with a vertical rotation frame.  Hips remain stationary; upper body twists.
  const spineIdx = bi("spine01");
  if (spineIdx >= 0)
    deltas[spineIdx] = mulMat3(transpose3(restRot3(restBonePoses, spineIdx)), R_shl);

  // ── Arms — XZ only (zero Y/depth): MediaPipe arm depth is noisy and the
  //    lowerarm rest has a large baked-in Y component that causes phantom bend.
  //    Arms move in the XZ plane; depth is irrelevant for most poses.
  const uDirL = xzOnly(boneDir(MP.LEFT_SHOULDER,  MP.LEFT_ELBOW));
  const uDirR = xzOnly(boneDir(MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW));
  const idxUL = bi("upperarm01.L"), idxUR = bi("upperarm01.R");
  if (idxUL >= 0 && vis(MP.LEFT_SHOULDER,  MP.LEFT_ELBOW))  deltas[idxUL] = limbDelta(uDirL, idxUL, restBonePoses);
  if (idxUR >= 0 && vis(MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW)) deltas[idxUR] = limbDelta(uDirR, idxUR, restBonePoses);

  const fDirL = xzOnly(boneDir(MP.LEFT_ELBOW,  MP.LEFT_WRIST));
  const fDirR = xzOnly(boneDir(MP.RIGHT_ELBOW, MP.RIGHT_WRIST));
  const idxFL = bi("lowerarm01.L"), idxFR = bi("lowerarm01.R");
  if (idxFL >= 0 && vis(MP.LEFT_ELBOW,  MP.LEFT_WRIST))  deltas[idxFL] = childLimbDelta(uDirL, fDirL, idxUL, idxFL, restBonePoses);
  if (idxFR >= 0 && vis(MP.RIGHT_ELBOW, MP.RIGHT_WRIST)) deltas[idxFR] = childLimbDelta(uDirR, fDirR, idxUR, idxFR, restBonePoses);

  // Wrist: wrist→index encodes wrist roll; falls back to forearm dir.
  const wLIdx = bi("wrist.L"), wRIdx = bi("wrist.R");
  if (wLIdx >= 0) {
    const wDirL = vis(MP.LEFT_WRIST, MP.LEFT_INDEX) ? boneDir(MP.LEFT_WRIST, MP.LEFT_INDEX) : fDirL;
    deltas[wLIdx] = childLimbDelta(fDirL, wDirL, idxFL, wLIdx, restBonePoses);
  }
  if (wRIdx >= 0) {
    const wDirR = vis(MP.RIGHT_WRIST, MP.RIGHT_INDEX) ? boneDir(MP.RIGHT_WRIST, MP.RIGHT_INDEX) : fDirR;
    deltas[wRIdx] = childLimbDelta(fDirR, wDirR, idxFR, wRIdx, restBonePoses);
  }

  // ── Legs ──────────────────────────────────────────────────────────────
  const thDirL = boneDir(MP.LEFT_HIP,   MP.LEFT_KNEE);
  const thDirR = boneDir(MP.RIGHT_HIP,  MP.RIGHT_KNEE);
  const idxTL = bi("upperleg01.L"), idxTR = bi("upperleg01.R");
  // Legs use lower visibility threshold (VIS_LEG) to track in low light / partial occlusion.
  const visLeg = (a: number, b: number) =>
    (landmarks[a].visibility ?? 1) >= VIS_LEG && (landmarks[b].visibility ?? 1) >= VIS_LEG;
  const thDirLxz = xzOnly(thDirL), thDirRxz = xzOnly(thDirR);

  if (idxTL >= 0 && visLeg(MP.LEFT_HIP,  MP.LEFT_KNEE))  deltas[idxTL] = limbDelta(thDirLxz, idxTL, restBonePoses);
  if (idxTR >= 0 && visLeg(MP.RIGHT_HIP, MP.RIGHT_KNEE)) deltas[idxTR] = limbDelta(thDirRxz, idxTR, restBonePoses);

  const shDirL = boneDir(MP.LEFT_KNEE,  MP.LEFT_ANKLE);
  const shDirR = boneDir(MP.RIGHT_KNEE, MP.RIGHT_ANKLE);
  const shDirLxz = xzOnly(shDirL), shDirRxz = xzOnly(shDirR);
  const idxSL = bi("lowerleg01.L"), idxSR = bi("lowerleg01.R");
  if (idxSL >= 0 && visLeg(MP.LEFT_KNEE,  MP.LEFT_ANKLE))  deltas[idxSL] = childLimbDelta(thDirLxz, shDirLxz, idxTL, idxSL, restBonePoses);
  if (idxSR >= 0 && visLeg(MP.RIGHT_KNEE, MP.RIGHT_ANKLE)) deltas[idxSR] = childLimbDelta(thDirRxz, shDirRxz, idxTR, idxSR, restBonePoses);

  // ── Head / neck ────────────────────────────────────────────────────────
  // Shoulder-mid → ear-mid; XZ-only; parent-corrected through spine01.
  {
    const le = mpVec(MP.LEFT_EAR), re = mpVec(MP.RIGHT_EAR);
    const earMidX = (le[0]+re[0])/2, earMidZ = (le[2]+re[2])/2;
    const neckDir = normalize3(new Float32Array([earMidX-shdMidX, 0, earMidZ-shdMidZ]));
    const neckIdx = bi("neck01");
    if (neckIdx >= 0 && spineIdx >= 0)
      deltas[neckIdx] = childLimbDelta(yW, neckDir, spineIdx, neckIdx, restBonePoses);
  }

  return deltas;
}
