import { expect, test, describe, beforeAll } from "bun:test";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import {
  allocBoneTransforms, allocVertexBuffer,
  forwardKinematics, lbs, type AnnyModel, type PoseDeltas,
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
  return JSON.parse(await Bun.file(resolve(FIXTURES, `${name}.json`)).text());
}

function deltasFromFixture(fx: ParityFixture): PoseDeltas {
  return fx.deltas.map(d => d === null ? null : new Float32Array(d));
}

const POSES = ["rest", "arms_up", "knee_bend_R", "torso_twist", "compound"];

describe("lbs ↔ Python anny parity", () => {
  let model: AnnyModel;

  beforeAll(async () => {
    model = await loadFixtureModel();
  });

  for (const pose of POSES) {
    test(`pose '${pose}': vertices match Python within 1e-4 m`, async () => {
      const fx = await loadFixture(pose);
      expect(fx.vertCount).toBe(model.vertCount);

      const deltas = deltasFromFixture(fx);
      const boneXforms = allocBoneTransforms(model.boneCount);
      const vertBuf = allocVertexBuffer(model);
      forwardKinematics(model, deltas, boneXforms);
      lbs(model, boneXforms, vertBuf);

      // Compute per-vertex Euclidean distance vs Python reference.
      let worstVert = -1;
      let worstErr = 0;
      const py = fx.expected_vertices;
      for (let v = 0; v < model.vertCount; v++) {
        const o = v * 3;
        const dx = vertBuf[o    ] - py[o    ];
        const dy = vertBuf[o + 1] - py[o + 1];
        const dz = vertBuf[o + 2] - py[o + 2];
        const d  = Math.hypot(dx, dy, dz);
        if (d > worstErr) { worstErr = d; worstVert = v; }
      }
      // 1e-4 m = 0.1 mm — well within numeric noise for f32 matmul chains.
      expect(worstErr, `worst vert idx=${worstVert} dist=${worstErr.toExponential(2)}m`)
        .toBeLessThan(1e-4);
    });
  }
});
