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
 * Assemble an AnnyModel from an already-parsed manifest + binary buffer.
 * Exposed for environments that read the data directly (Node/Bun tests,
 * bundled inline assets) rather than via fetch.
 */
export function parseAnnyModel(manifest: AnnyManifest, buf: ArrayBuffer): AnnyModel {
  const a = manifest.arrays;
  return {
    restVertices:   viewOf(buf, a.rest_vertices) as Float32Array,
    faces:          viewOf(buf, a.faces) as Int32Array,
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
