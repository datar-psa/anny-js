import { expect, test, describe } from "bun:test";
import { Pose } from "./pose.js";
import type { AnnyModel } from "./types.js";
import { rodriguesToMat3 } from "./math.js";

function makeTinyModel(): AnnyModel {
  return {
    restVertices: new Float32Array(0),
    faces: new Int32Array(0),
    boneWeights: new Float32Array(0),
    boneIndices: new Int32Array(0),
    restBonePoses: new Float32Array(48), // 3 bones × 16
    boneParents: new Int32Array([-1, 0, 1]),
    boneLabels: ["root", "child0", "child1"],
    vertCount: 0, faceCount: 0, boneCount: 3, maxBonesPerVert: 0,
  };
}

describe("Pose", () => {
  test("starts with all identity deltas", () => {
    const p = new Pose(makeTinyModel());
    expect(p.deltas.length).toBe(3);
    expect(p.deltas.every(d => d === null)).toBe(true);
  });

  test("set + get + chaining", () => {
    const m = makeTinyModel();
    const r = rodriguesToMat3(new Float32Array([0,0,1]), Math.PI/4);
    const p = new Pose(m)
      .set("child0", r)
      .set("child1", r);
    expect(p.get("child0")).toBe(r);
    expect(p.get("root")).toBeNull();
    // unknown bone reads as undefined (not null)
    expect(p.get("nope")).toBeUndefined();
  });

  test("set ignores unknown bones silently", () => {
    const p = new Pose(makeTinyModel());
    const r = rodriguesToMat3(new Float32Array([0,0,1]), 0.1);
    expect(() => p.set("nonexistent", r)).not.toThrow();
  });

  test("setStrict throws on unknown bones", () => {
    const p = new Pose(makeTinyModel());
    const r = rodriguesToMat3(new Float32Array([0,0,1]), 0.1);
    expect(() => p.setStrict("nonexistent", r)).toThrow(/nonexistent/);
  });

  test("setMany applies multiple bones in one call", () => {
    const m = makeTinyModel();
    const r = rodriguesToMat3(new Float32Array([0,0,1]), 0.1);
    const p = new Pose(m).setMany([
      { bone: "child0", rot: r },
      { bone: "child1", rot: r },
    ]);
    expect(p.get("child0")).toBe(r);
    expect(p.get("child1")).toBe(r);
  });

  test("reset clears every bone back to identity", () => {
    const m = makeTinyModel();
    const r = rodriguesToMat3(new Float32Array([0,0,1]), 0.5);
    const p = new Pose(m).set("child0", r).set("child1", r);
    p.reset();
    expect(p.deltas.every(d => d === null)).toBe(true);
  });

  test("adopt copies foreign deltas into the pose buffer", () => {
    const m = makeTinyModel();
    const r = rodriguesToMat3(new Float32Array([0,0,1]), 0.5);
    const foreign = [null, r, null] as (Float32Array | null)[];
    const p = new Pose(m).adopt(foreign);
    expect(p.get("child0")).toBe(r);
    expect(p.get("root")).toBeNull();
    expect(p.get("child1")).toBeNull();
  });

  test("adopt rejects length mismatch", () => {
    const p = new Pose(makeTinyModel());
    expect(() => p.adopt([null, null])).toThrow(/length mismatch/);
  });

  test("constructor reuses provided boneIndex (no rebuild)", () => {
    const m = makeTinyModel();
    const ix = new Map([["root", 0], ["child0", 1], ["child1", 2]]);
    const r = rodriguesToMat3(new Float32Array([0,0,1]), 0.1);
    const p = new Pose(m, ix).set("child1", r);
    expect(p.get("child1")).toBe(r);
  });
});
