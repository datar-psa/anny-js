/** Core Anny: model loading, forward kinematics, linear blend skinning, 2D renderer.
 *
 * No MediaPipe dependency — consumers that only need FK + LBS (custom rigs,
 * pre-recorded motion, procedural animation) can import from here directly. */

export type { AnnyModel, Mat4, PoseDeltas, SkinnedMesh } from "./types.js";
export { loadAnnyModel, parseAnnyModel } from "./loader.js";
export type { AnnyManifest } from "./loader.js";
export {
  forwardKinematics, allocBoneTransforms,
  identityDeltas, setDelta, setDeltas, buildBoneIndex,
} from "./fk.js";
export { lbs, allocVertexBuffer } from "./lbs.js";
export { renderAnny } from "./render2d.js";
export type { RenderOptions } from "./render2d.js";
// Rotation builders + the two 4×4 utilities anyone is likely to want.
// Lower-level helpers (mat3 multiply, vec3 ops, rigid invert, mat4 slicing)
// live in ./math.ts and are intentionally not re-exported — they were
// internal implementation details, and exposing them invited misuse
// (e.g. allocating-per-frame `mulMat3` in hot loops when the `*Into`
// variants are what you actually want).
export {
  rodriguesToMat3, rotFromTo,
  mat4Identity, mat4Mul,
} from "./math.js";

export { Pose } from "./pose.js";
