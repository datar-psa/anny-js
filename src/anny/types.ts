/** Loaded Anny model ready for LBS + FK. */
export interface AnnyModel {
  /** (V×3) T-pose rest vertices, shaped by baked phenotype params. */
  restVertices: Float32Array;
  /** (V×3) Smooth per-vertex normals in rest pose (area-weighted face
   * normals, unit length). Derived once at load time so renderers can do
   * proper smooth shading without per-frame normal recomputation. */
  restNormals: Float32Array;
  /** (F×3) triangle indices. */
  faces: Int32Array;
  /** (V×K) LBS weights per vertex. Rows sum to 1. */
  boneWeights: Float32Array;
  /** (V×K) Bone indices parallel to boneWeights. */
  boneIndices: Int32Array;
  /** (B×16) Rest-pose world transforms per bone, row-major 4×4. */
  restBonePoses: Float32Array;
  /** Parent bone index per bone. -1 = root. Length B. */
  boneParents: Int32Array;
  /** Human-readable bone names. Length B. */
  boneLabels: string[];

  vertCount: number;  // V
  faceCount: number;  // F
  boneCount: number;  // B
  maxBonesPerVert: number;  // K (typically 8)
}

/** One 4×4 transform, row-major (index [r*4+c]). */
export type Mat4 = Float32Array;

/** Pose input: one local delta rotation (3×3 row-major) per bone, or null for identity. */
export type PoseDeltas = (Float32Array | null)[];

/** Skinned mesh output from lbs(). */
export interface SkinnedMesh {
  /** (V×3) deformed vertex positions. */
  vertices: Float32Array;
  /** (V×3) deformed per-vertex normals, unit length. Skinned with the
   * rotation part of the bone transforms (no translation). Renderers should
   * prefer these over screen-space-derivative normals for smooth shading. */
  normals: Float32Array;
  faces: Int32Array;
}
