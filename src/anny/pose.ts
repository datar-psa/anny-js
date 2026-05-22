/**
 * Ergonomic wrapper around `PoseDeltas` for building poses by bone name.
 *
 * The underlying functional API (`identityDeltas` + `setDelta` + `setDeltas`)
 * is the zero-alloc path; `Pose` is a thin façade that holds the same array
 * and a cached `boneIndex`. Reach for the functional API in hot loops or when
 * you already manage the bone-index Map; reach for `Pose` everywhere else.
 *
 * Both share the same `PoseDeltas` representation, so you can pass
 * `pose.deltas` straight into `forwardKinematics`.
 */

import type { AnnyModel, PoseDeltas } from "./types.js";
import { buildBoneIndex, identityDeltas } from "./fk.js";

export class Pose {
  /** Underlying per-bone delta rotations (null = identity). Pass into `forwardKinematics`. */
  readonly deltas: PoseDeltas;

  private readonly boneIndex: Map<string, number>;

  constructor(model: AnnyModel, boneIndex?: Map<string, number>) {
    this.boneIndex = boneIndex ?? buildBoneIndex(model);
    this.deltas = identityDeltas(model.boneCount);
  }

  /**
   * Set the delta rotation for one bone by name. Silently ignores unknown
   * bone names — convenient in hot loops and procedural rigs that may target
   * bones present on some Anny variants but not others.
   */
  set(boneName: string, rot3x3: Float32Array): this {
    const idx = this.boneIndex.get(boneName);
    if (idx !== undefined) this.deltas[idx] = rot3x3;
    return this;
  }

  /** Like `set`, but throws if the bone name is not in the rig. */
  setStrict(boneName: string, rot3x3: Float32Array): this {
    const idx = this.boneIndex.get(boneName);
    if (idx === undefined) {
      throw new Error(`Pose.setStrict: unknown bone "${boneName}"`);
    }
    this.deltas[idx] = rot3x3;
    return this;
  }

  /** Set many bones at once. Returns `this` for chaining. */
  setMany(entries: Iterable<{ bone: string; rot: Float32Array }>): this {
    for (const { bone, rot } of entries) this.set(bone, rot);
    return this;
  }

  /** Reset every bone to its rest pose (identity delta). */
  reset(): this {
    for (let i = 0; i < this.deltas.length; i++) this.deltas[i] = null;
    return this;
  }

  /** Read the current delta for a bone (null = identity). */
  get(boneName: string): Float32Array | null | undefined {
    const idx = this.boneIndex.get(boneName);
    return idx === undefined ? undefined : this.deltas[idx];
  }

  /** Adopt foreign deltas (e.g. the output of `landmarksToPoseDeltas`). */
  adopt(deltas: PoseDeltas): this {
    if (deltas.length !== this.deltas.length) {
      throw new Error(`Pose.adopt: length mismatch (got ${deltas.length}, expected ${this.deltas.length})`);
    }
    for (let i = 0; i < deltas.length; i++) this.deltas[i] = deltas[i];
    return this;
  }
}
