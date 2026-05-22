import { expect, test, describe, beforeAll } from "bun:test";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import {
  allocBoneTransforms, forwardKinematics, type AnnyModel, type PoseDeltas,
} from "../../src/anny/index.js";
import { loadFixtureModel } from "../_helpers/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(here, "fixtures");

interface ParityFixture {
  pose: string;
  boneCount: number;
  vertCount: number;
  deltas: (number[] | null)[];
  expected_bone_xforms: number[][];
  expected_vertices: number[];
}

async function loadFixture(name: string): Promise<ParityFixture> {
  const path = resolve(FIXTURES, `${name}.json`);
  return JSON.parse(await Bun.file(path).text());
}

function deltasFromFixture(fx: ParityFixture): PoseDeltas {
  return fx.deltas.map(d => d === null ? null : new Float32Array(d));
}

function maxAbsDiff(a: Float32Array | number[], b: Float32Array | number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

const POSES = ["rest", "arms_up", "knee_bend_R", "torso_twist", "compound"];

describe("forwardKinematics ↔ Python anny parity", () => {
  let model: AnnyModel;

  beforeAll(async () => {
    model = await loadFixtureModel();
  });

  for (const pose of POSES) {
    test(`pose '${pose}': bone transforms match Python within 1e-5`, async () => {
      const fx = await loadFixture(pose);
      expect(fx.boneCount).toBe(model.boneCount);

      const deltas = deltasFromFixture(fx);
      const out = allocBoneTransforms(model.boneCount);
      forwardKinematics(model, deltas, out);

      // Compare each bone's 4×4 transform to the Python reference.
      let worstBone = -1;
      let worstErr = 0;
      for (let b = 0; b < model.boneCount; b++) {
        const js = out.subarray(b * 16, b * 16 + 16);
        const py = fx.expected_bone_xforms[b];
        const err = maxAbsDiff(js, py);
        if (err > worstErr) { worstErr = err; worstBone = b; }
      }
      expect(worstErr, `worst bone idx=${worstBone} ‖js-py‖∞=${worstErr.toExponential(2)}`)
        .toBeLessThan(1e-5);
    });
  }
});
