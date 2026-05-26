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
 * Skins both positions and normals. Normals use only the rotation part of
 * each bone transform — translation has no meaning for direction vectors,
 * and the rotations are orthonormal by construction (rigid skeletons), so
 * we don't need an inverse-transpose. The final per-vertex blended vector
 * is renormalised because the weighted sum of unit vectors is not unit.
 *
 * @param model      Loaded AnnyModel.
 * @param boneTransforms  (B×16) flat row-major 4×4 transforms per bone,
 *                        as produced by forwardKinematics().
 * @param outVertices (optional) pre-allocated (V×3) Float32Array to write into.
 *                    Pass the same buffer each frame to avoid allocation.
 * @param outNormals  (optional) pre-allocated (V×3) Float32Array for normals.
 *                    Pass the same buffer each frame to avoid allocation.
 */
export function lbs(
  model: AnnyModel,
  boneTransforms: Float32Array,
  outVertices?: Float32Array,
  outNormals?: Float32Array,
): SkinnedMesh {
  const {
    vertCount, maxBonesPerVert,
    restVertices, restNormals,
    boneWeights, boneIndices, faces,
  } = model;
  const outV = outVertices ?? new Float32Array(vertCount * 3);
  const outN = outNormals  ?? new Float32Array(vertCount * 3);

  for (let v = 0; v < vertCount; v++) {
    const rx = restVertices[v * 3    ];
    const ry = restVertices[v * 3 + 1];
    const rz = restVertices[v * 3 + 2];
    const nrx = restNormals[v * 3    ];
    const nry = restNormals[v * 3 + 1];
    const nrz = restNormals[v * 3 + 2];

    let ox = 0, oy = 0, oz = 0;
    let nx = 0, ny = 0, nz = 0;

    const wBase = v * maxBonesPerVert;
    for (let k = 0; k < maxBonesPerVert; k++) {
      const w = boneWeights[wBase + k];
      if (w < 1e-7) continue;

      const bi = boneIndices[wBase + k];
      const t  = bi * 16;  // offset into boneTransforms

      // Position: apply full 4×4 transform (row-major) to homogeneous point.
      const tx = boneTransforms[t    ]*rx + boneTransforms[t + 1]*ry + boneTransforms[t + 2]*rz + boneTransforms[t + 3];
      const ty = boneTransforms[t + 4]*rx + boneTransforms[t + 5]*ry + boneTransforms[t + 6]*rz + boneTransforms[t + 7];
      const tz = boneTransforms[t + 8]*rx + boneTransforms[t + 9]*ry + boneTransforms[t +10]*rz + boneTransforms[t +11];

      // Normal: apply only the 3×3 rotation block (cols 0..2 of rows 0..2).
      const mx = boneTransforms[t    ]*nrx + boneTransforms[t + 1]*nry + boneTransforms[t + 2]*nrz;
      const my = boneTransforms[t + 4]*nrx + boneTransforms[t + 5]*nry + boneTransforms[t + 6]*nrz;
      const mz = boneTransforms[t + 8]*nrx + boneTransforms[t + 9]*nry + boneTransforms[t +10]*nrz;

      ox += w * tx; oy += w * ty; oz += w * tz;
      nx += w * mx; ny += w * my; nz += w * mz;
    }

    outV[v * 3    ] = ox;
    outV[v * 3 + 1] = oy;
    outV[v * 3 + 2] = oz;

    // Renormalise — weighted sum of unit vectors isn't unit length, and
    // skin-deformed meshes need unit normals for correct lighting.
    const nLen = Math.hypot(nx, ny, nz) || 1;
    outN[v * 3    ] = nx / nLen;
    outN[v * 3 + 1] = ny / nLen;
    outN[v * 3 + 2] = nz / nLen;
  }

  return { vertices: outV, normals: outN, faces };
}

/** Pre-allocate a vertex output buffer for reuse. */
export function allocVertexBuffer(model: AnnyModel): Float32Array {
  return new Float32Array(model.vertCount * 3);
}

/** Pre-allocate a normal output buffer for reuse. */
export function allocNormalBuffer(model: AnnyModel): Float32Array {
  return new Float32Array(model.vertCount * 3);
}
