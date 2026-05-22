import { expect, test, describe } from "bun:test";
import {
  rodriguesToMat3, rotFromTo,
  mat4Identity, mat4Mul, mat4InvertRigid, mat4FromRot3Translation,
  cross, normalize3, dot3,
} from "./math.js";

const EPS = 1e-6;

const approxArr = (a: ArrayLike<number>, b: ArrayLike<number>, eps = EPS) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThan(eps);
  }
};

describe("rodriguesToMat3", () => {
  test("identity for zero angle", () => {
    const r = rodriguesToMat3(new Float32Array([0, 0, 1]), 0);
    approxArr(r, [1,0,0, 0,1,0, 0,0,1]);
  });

  test("90° around Z rotates +X to +Y", () => {
    const r = rodriguesToMat3(new Float32Array([0, 0, 1]), Math.PI / 2);
    const x = new Float32Array([1, 0, 0]);
    const out = new Float32Array([
      r[0]*x[0] + r[1]*x[1] + r[2]*x[2],
      r[3]*x[0] + r[4]*x[1] + r[5]*x[2],
      r[6]*x[0] + r[7]*x[1] + r[8]*x[2],
    ]);
    approxArr(out, [0, 1, 0]);
  });

  test("180° around X negates Y and Z", () => {
    const r = rodriguesToMat3(new Float32Array([1, 0, 0]), Math.PI);
    const v = new Float32Array([0, 1, 2]);
    const out = new Float32Array([
      r[0]*v[0] + r[1]*v[1] + r[2]*v[2],
      r[3]*v[0] + r[4]*v[1] + r[5]*v[2],
      r[6]*v[0] + r[7]*v[1] + r[8]*v[2],
    ]);
    approxArr(out, [0, -1, -2]);
  });
});

describe("rotFromTo", () => {
  test("identity when vectors equal", () => {
    const r = rotFromTo(new Float32Array([1, 0, 0]), new Float32Array([1, 0, 0]));
    approxArr(r, [1,0,0, 0,1,0, 0,0,1]);
  });

  test("rotates +X to +Y", () => {
    const r = rotFromTo(new Float32Array([1, 0, 0]), new Float32Array([0, 1, 0]));
    const v = new Float32Array([
      r[0]*1 + r[1]*0 + r[2]*0,
      r[3]*1 + r[4]*0 + r[5]*0,
      r[6]*1 + r[7]*0 + r[8]*0,
    ]);
    approxArr(v, [0, 1, 0]);
  });

  test("180° antiparallel case stays unit", () => {
    const r = rotFromTo(new Float32Array([1, 0, 0]), new Float32Array([-1, 0, 0]));
    const v = new Float32Array([
      r[0]*1, r[3]*1, r[6]*1,
    ]);
    // length 1, direction = -X
    expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(1, 5);
    expect(v[0]).toBeCloseTo(-1, 5);
  });
});

describe("mat4 helpers", () => {
  test("identity is identity", () => {
    const I = mat4Identity();
    approxArr(I, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  });

  test("multiply by identity is no-op", () => {
    const I = mat4Identity();
    const m = new Float32Array([
      1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
    ]);
    const out = mat4Mul(m, I);
    approxArr(out, m);
  });

  test("invertRigid is right inverse", () => {
    // Build a rigid transform: rotate 90° around Z, translate (1, 2, 3)
    const r = rodriguesToMat3(new Float32Array([0, 0, 1]), Math.PI / 2);
    const m = mat4FromRot3Translation(r, 1, 2, 3);
    const inv = mat4InvertRigid(m);
    const id = mat4Mul(m, inv);
    approxArr(id, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], 1e-5);
  });
});

describe("vec3 helpers", () => {
  test("cross product follows right-hand rule", () => {
    const c = cross(new Float32Array([1,0,0]), new Float32Array([0,1,0]));
    approxArr(c, [0, 0, 1]);
  });

  test("normalize gives unit length", () => {
    const v = new Float32Array([3, 4, 0]);
    normalize3(v);
    expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(1, 6);
  });

  test("normalize zero vector stays zero (no NaN)", () => {
    const v = new Float32Array([0, 0, 0]);
    normalize3(v);
    expect(Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2])).toBe(true);
  });

  test("dot of perpendicular vectors is zero", () => {
    expect(dot3(new Float32Array([1,0,0]), new Float32Array([0,1,0]))).toBeCloseTo(0, 6);
  });
});
