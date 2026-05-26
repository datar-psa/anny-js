import type { AnnyModel } from "./types.js";

interface ArrayEntry {
  dtype: string;
  shape: number[];
  offset: number;
  bytes: number;
}

export interface AnnyManifest {
  version: number;
  bone_count: number;
  vert_count: number;
  face_count: number;
  max_bones_per_vert: number;
  arrays: Record<string, ArrayEntry>;
  bone_parents: number[];
  bone_labels: string[];
}

function viewOf(buf: ArrayBuffer, entry: ArrayEntry): Float32Array | Int32Array {
  const isFloat = entry.dtype.includes("f");
  return isFloat
    ? new Float32Array(buf, entry.offset, entry.bytes / 4)
    : new Int32Array(buf, entry.offset, entry.bytes / 4);
}

/**
 * Compute smooth per-vertex normals from the rest mesh: for each triangle,
 * accumulate its (un-normalised) face normal onto each of its three vertices,
 * then normalise. This area-weights the average automatically — large faces
 * contribute more, which is the conventional/correct behaviour. Run once at
 * load time; the LBS path then skins these per frame.
 */
function computeRestNormals(
  vertices: Float32Array,
  faces: Int32Array,
  vertCount: number,
): Float32Array {
  const normals = new Float32Array(vertCount * 3);
  const fCount = faces.length / 3;

  for (let f = 0; f < fCount; f++) {
    const ia = faces[f * 3    ] * 3;
    const ib = faces[f * 3 + 1] * 3;
    const ic = faces[f * 3 + 2] * 3;

    const ax = vertices[ia], ay = vertices[ia + 1], az = vertices[ia + 2];
    const bx = vertices[ib], by = vertices[ib + 1], bz = vertices[ib + 2];
    const cx = vertices[ic], cy = vertices[ic + 1], cz = vertices[ic + 2];

    // Un-normalised face normal = (b - a) × (c - a). Length encodes 2× area,
    // so accumulating it gives area-weighted averaging — desirable on meshes
    // with mixed triangle sizes.
    const ex = bx - ax, ey = by - ay, ez = bz - az;
    const fx = cx - ax, fy = cy - ay, fz = cz - az;
    const nx = ey * fz - ez * fy;
    const ny = ez * fx - ex * fz;
    const nz = ex * fy - ey * fx;

    normals[ia    ] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib    ] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic    ] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }

  for (let v = 0; v < vertCount; v++) {
    const i = v * 3;
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i    ] /= len;
    normals[i + 1] /= len;
    normals[i + 2] /= len;
  }

  return normals;
}

/**
 * Assemble an AnnyModel from an already-parsed manifest + binary buffer.
 * Exposed for environments that read the data directly (Node/Bun tests,
 * bundled inline assets) rather than via fetch.
 */
export function parseAnnyModel(manifest: AnnyManifest, buf: ArrayBuffer): AnnyModel {
  const a = manifest.arrays;
  const restVertices = viewOf(buf, a.rest_vertices) as Float32Array;
  const faces = viewOf(buf, a.faces) as Int32Array;
  return {
    restVertices,
    restNormals:    computeRestNormals(restVertices, faces, manifest.vert_count),
    faces,
    boneWeights:    viewOf(buf, a.bone_weights) as Float32Array,
    boneIndices:    viewOf(buf, a.bone_indices) as Int32Array,
    restBonePoses:  viewOf(buf, a.rest_bone_poses) as Float32Array,
    boneParents:    new Int32Array(manifest.bone_parents),
    boneLabels:     manifest.bone_labels,
    vertCount:      manifest.vert_count,
    faceCount:      manifest.face_count,
    boneCount:      manifest.bone_count,
    maxBonesPerVert: manifest.max_bones_per_vert,
  };
}

/**
 * Load Anny model from a manifest JSON URL and a binary data URL.
 * Both paths may be relative (browser fetch) or absolute (Node file://).
 */
export async function loadAnnyModel(
  manifestUrl: string,
  binUrl: string
): Promise<AnnyModel> {
  const [manifestRes, binRes] = await Promise.all([
    fetch(manifestUrl),
    fetch(binUrl),
  ]);
  if (!manifestRes.ok) throw new Error(`Failed to fetch manifest: ${manifestUrl}`);
  if (!binRes.ok) throw new Error(`Failed to fetch binary: ${binUrl}`);

  const manifest: AnnyManifest = await manifestRes.json();
  const buf = await binRes.arrayBuffer();
  return parseAnnyModel(manifest, buf);
}
