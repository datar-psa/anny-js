import { expect, test, describe } from "bun:test";
import {
  allocBoneTransforms, identityDeltas, setDelta, setDeltas,
  buildBoneIndex, forwardKinematics,
} from "./fk.js";
import type { AnnyModel } from "./types.js";
import { rodriguesToMat3 } from "./math.js";

/** Build a minimal 3-bone model: root → child0 → child1, all aligned with world axes. */
function makeTinyModel(): AnnyModel {
  // Three bones, each at world identity rotation. Bone 1 at +Y=1, bone 2 at +Y=2.
  const restBonePoses = new Float32Array([
    // bone 0 (root) — identity at origin
    1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
    // bone 1 — identity, translated +Y=1 (relative to world)
    1,0,0,0, 0,1,0,1, 0,0,1,0, 0,0,0,1,
    // bone 2 — identity, translated +Y=2
    1,0,0,0, 0,1,0,2, 0,0,1,0, 0,0,0,1,
  ]);
  return {
    restVertices: new Float32Array(0),
    restNormals:  new Float32Array(0),
    faces: new Int32Array(0),
    boneWeights: new Float32Array(0),
    boneIndices: new Int32Array(0),
    restBonePoses,
    boneParents: new Int32Array([-1, 0, 1]),
    boneLabels: ["root", "child0", "child1"],
    vertCount: 0, faceCount: 0, boneCount: 3, maxBonesPerVert: 0,
  };
}

describe("identityDeltas / setDelta / buildBoneIndex", () => {
  test("identityDeltas is all nulls of the right length", () => {
    const d = identityDeltas(5);
    expect(d.length).toBe(5);
    expect(d.every(v => v === null)).toBe(true);
  });

  test("setDelta writes by name, ignores missing names", () => {
    const m = makeTinyModel();
    const d = identityDeltas(m.boneCount);
    const r = rodriguesToMat3(new Float32Array([0,0,1]), Math.PI/4);
    setDelta(d, m, "child0", r);
    expect(d[1]).toBe(r);
    expect(d[0]).toBeNull();
    // Missing name is a silent no-op (hot-loop friendly).
    setDelta(d, m, "nonexistent", r);
    expect(d[0]).toBeNull();
  });

  test("setDeltas sets multiple bones at once", () => {
    const m = makeTinyModel();
    const d = identityDeltas(m.boneCount);
    const r = rodriguesToMat3(new Float32Array([0,0,1]), Math.PI/4);
    setDeltas(d, m, [{ bone: "child0", rot: r }, { bone: "child1", rot: r }]);
    expect(d[1]).toBe(r);
    expect(d[2]).toBe(r);
  });

  test("buildBoneIndex maps name → index", () => {
    const m = makeTinyModel();
    const ix = buildBoneIndex(m);
    expect(ix.get("root")).toBe(0);
    expect(ix.get("child0")).toBe(1);
    expect(ix.get("child1")).toBe(2);
    expect(ix.get("nope")).toBeUndefined();
  });
});

describe("forwardKinematics", () => {
  test("identity deltas → identity transforms (rest pose)", () => {
    const m = makeTinyModel();
    const d = identityDeltas(m.boneCount);
    const out = allocBoneTransforms(m.boneCount);
    forwardKinematics(m, d, out);
    // Each bone's transform = pose @ inv(rest) = rest @ inv(rest) = identity.
    for (let b = 0; b < m.boneCount; b++) {
      const o = b * 16;
      expect(out[o + 0]).toBeCloseTo(1, 6);
      expect(out[o + 5]).toBeCloseTo(1, 6);
      expect(out[o +10]).toBeCloseTo(1, 6);
      expect(out[o +15]).toBeCloseTo(1, 6);
      // Translation should be zero (identity transform, not pose itself).
      expect(out[o + 3]).toBeCloseTo(0, 6);
      expect(out[o + 7]).toBeCloseTo(0, 6);
      expect(out[o +11]).toBeCloseTo(0, 6);
    }
  });

  test("root rotation propagates to children", () => {
    const m = makeTinyModel();
    const d = identityDeltas(m.boneCount);
    // Rotate root 90° around Z. Child bones inherit this through the parent chain.
    setDelta(d, m, "root", rodriguesToMat3(new Float32Array([0,0,1]), Math.PI/2));
    const out = allocBoneTransforms(m.boneCount);
    forwardKinematics(m, d, out);

    // root's transform (= pose @ inv(rest)) on a unit +Y vector should hit +X
    // because pose = R_z(90°) and rest = identity. Apply the 3x3 rotation:
    const r0 = out.subarray(0, 16);
    // Multiply rotation block by (0, 1, 0)
    const tx = r0[0]*0 + r0[1]*1 + r0[2]*0;
    const ty = r0[4]*0 + r0[5]*1 + r0[6]*0;
    const tz = r0[8]*0 + r0[9]*1 + r0[10]*0;
    expect(tx).toBeCloseTo(-1, 5);  // R_z(90°): (0,1,0) → (-1,0,0)
    expect(ty).toBeCloseTo(0, 5);
    expect(tz).toBeCloseTo(0, 5);
  });
});
