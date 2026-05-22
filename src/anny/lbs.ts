/**
 * Linear Blend Skinning for Anny body model.
 *
 * skinned[v] = Σ_k weight[v,k] * boneTransform[idx[v,k]] * restVert[v]
 *
 * All data is Float32Array / Int32Array — no GC allocations in the hot path
 * beyond the output buffer (which callers can pre-allocate and reuse).
 */

import type { AnnyModel, SkinnedMesh } from "./types.js";

/**
 * Apply LBS and return a SkinnedMesh.
 *
 * @param model      Loaded AnnyModel.
 * @param boneTransforms  (B×16) flat row-major 4×4 transforms per bone,
 *                        as produced by forwardKinematics().
 * @param outVertices (optional) pre-allocated (V×3) Float32Array to write into.
 *                    Pass the same buffer each frame to avoid allocation.
 */
export function lbs(
  model: AnnyModel,
  boneTransforms: Float32Array,
  outVertices?: Float32Array
): SkinnedMesh {
  const { vertCount, maxBonesPerVert, restVertices, boneWeights, boneIndices, faces } = model;
  const out = outVertices ?? new Float32Array(vertCount * 3);

  for (let v = 0; v < vertCount; v++) {
    const rx = restVertices[v * 3    ];
    const ry = restVertices[v * 3 + 1];
    const rz = restVertices[v * 3 + 2];

    let ox = 0, oy = 0, oz = 0;

    const wBase = v * maxBonesPerVert;
    for (let k = 0; k < maxBonesPerVert; k++) {
      const w = boneWeights[wBase + k];
      if (w < 1e-7) continue;

      const bi = boneIndices[wBase + k];
      const t  = bi * 16;  // offset into boneTransforms

      // Apply 4×4 transform (row-major) to homogeneous point (rx,ry,rz,1)
      const tx = boneTransforms[t    ]*rx + boneTransforms[t + 1]*ry + boneTransforms[t + 2]*rz + boneTransforms[t + 3];
      const ty = boneTransforms[t + 4]*rx + boneTransforms[t + 5]*ry + boneTransforms[t + 6]*rz + boneTransforms[t + 7];
      const tz = boneTransforms[t + 8]*rx + boneTransforms[t + 9]*ry + boneTransforms[t +10]*rz + boneTransforms[t +11];

      ox += w * tx;
      oy += w * ty;
      oz += w * tz;
    }

    out[v * 3    ] = ox;
    out[v * 3 + 1] = oy;
    out[v * 3 + 2] = oz;
  }

  return { vertices: out, faces };
}

/** Pre-allocate a vertex output buffer for reuse. */
export function allocVertexBuffer(model: AnnyModel): Float32Array {
  return new Float32Array(model.vertCount * 3);
}
