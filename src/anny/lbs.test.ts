import { expect, test, describe, beforeAll } from "bun:test";
import {
  allocBoneTransforms, allocVertexBuffer, allocNormalBuffer,
  buildBoneIndex, forwardKinematics, identityDeltas, lbs,
  rodriguesToMat3,
  type AnnyModel,
} from "./index.js";
import { loadFixtureModel } from "../../tests/_helpers/model.js";

/**
 * Unit tests for the LBS normal pipeline. The position path already has
 * Python parity coverage under tests/parity; these tests pin down properties
 * specific to the new skinned-normal output.
 */
describe("lbs — skinned normals", () => {
  let model: AnnyModel;

  beforeAll(async () => {
    model = await loadFixtureModel();
  });

  test("rest normals are unit length and finite", () => {
    expect(model.restNormals.length).toBe(model.vertCount * 3);
    let worstDev = 0, worstVid = -1;
    for (let v = 0; v < model.vertCount; v++) {
      const i = v * 3;
      const nx = model.restNormals[i    ];
      const ny = model.restNormals[i + 1];
      const nz = model.restNormals[i + 2];
      expect(Number.isFinite(nx + ny + nz), `vert ${v} normal not finite`).toBe(true);
      const len = Math.hypot(nx, ny, nz);
      const dev = Math.abs(len - 1);
      if (dev > worstDev) { worstDev = dev; worstVid = v; }
    }
    // Allow a hair of float drift; 1e-5 leaves room without hiding real bugs.
    expect(worstDev, `worst vid ${worstVid} dev ${worstDev}`).toBeLessThan(1e-5);
  });

  test("rest-pose LBS reproduces rest normals (identity skinning is no-op)", () => {
    const xforms = allocBoneTransforms(model.boneCount);
    const vBuf   = allocVertexBuffer(model);
    const nBuf   = allocNormalBuffer(model);
    forwardKinematics(model, identityDeltas(model.boneCount), xforms);
    const mesh = lbs(model, xforms, vBuf, nBuf);

    let worst = 0;
    for (let v = 0; v < model.vertCount; v++) {
      const i = v * 3;
      const dx = mesh.normals[i    ] - model.restNormals[i    ];
      const dy = mesh.normals[i + 1] - model.restNormals[i + 1];
      const dz = mesh.normals[i + 2] - model.restNormals[i + 2];
      const d  = Math.hypot(dx, dy, dz);
      if (d > worst) worst = d;
    }
    // Identity bone xforms = rest, so skinned normals must equal rest exactly
    // up to f32 noise from the (1.0)*(unit) renormalisation step.
    expect(worst).toBeLessThan(1e-5);
  });

  test("skinned normals are unit length after a non-trivial pose", () => {
    const boneIndex = buildBoneIndex(model);
    const deltas = identityDeltas(model.boneCount);
    // Rotate the right upper arm 60° around its local +X axis — drives a
    // chunk of the mesh through a non-identity rotation chain.
    const upArm = boneIndex.get("upperarm01.R");
    expect(upArm, "upperarm01.R bone must exist in the rig").not.toBeUndefined();
    deltas[upArm!] = rodriguesToMat3(new Float32Array([1, 0, 0]), Math.PI / 3);

    const xforms = allocBoneTransforms(model.boneCount);
    const vBuf   = allocVertexBuffer(model);
    const nBuf   = allocNormalBuffer(model);
    forwardKinematics(model, deltas, xforms);
    const mesh = lbs(model, xforms, vBuf, nBuf);

    let worstDev = 0;
    for (let v = 0; v < model.vertCount; v++) {
      const i = v * 3;
      const len = Math.hypot(mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]);
      const dev = Math.abs(len - 1);
      if (dev > worstDev) worstDev = dev;
    }
    // After LBS blending the raw weighted sum isn't unit; the renormalise
    // step inside lbs() must bring every output back within f32 epsilon.
    expect(worstDev).toBeLessThan(1e-5);
  });

  test("vertex normals correlate with anatomical front/back direction", () => {
    // Sanity check on the rest mesh: vertices on the anatomical front
    // (rest_y < 0 in Anny world) should, on average, have normals pointing
    // forward (-y). This catches a face-normal sign flip — if we accidentally
    // computed (c-a)×(b-a) instead of (b-a)×(c-a), every vertex on the front
    // would have a normal pointing backward.
    let frontDotSum = 0, frontCount = 0;
    let backDotSum  = 0, backCount  = 0;
    for (let v = 0; v < model.vertCount; v++) {
      const y  = model.restVertices[v * 3 + 1];
      const ny = model.restNormals [v * 3 + 1];
      // ignore vertices near y≈0 (sides) where the sign is noisy
      if (y < -0.05) { frontDotSum += -ny; frontCount++; }
      if (y > +0.05) { backDotSum  += +ny; backCount++;  }
    }
    expect(frontCount).toBeGreaterThan(100);
    expect(backCount).toBeGreaterThan(100);
    // Average alignment with the expected outward normal: should be clearly
    // positive on both halves. Body mesh has rounded sides so the average
    // doesn't approach 1.0 — 0.15 is enough to catch a sign flip (which
    // would land around -0.28) without overfitting the body geometry.
    expect(frontDotSum / frontCount, "anatomical front normals point -y").toBeGreaterThan(0.15);
    expect(backDotSum  / backCount,  "anatomical back normals point +y").toBeGreaterThan(0.15);
  });
});
