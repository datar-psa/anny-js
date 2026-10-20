/**
 * The sign conventions the driver and the validators rely on, pinned against
 * the rig itself so a re-baked model or a refactor can't silently flip them.
 */

import { beforeAll, describe, expect, test } from "bun:test";

import { buildBoneIndex, identityDeltas, rodriguesToMat3, type AnnyModel } from "../../src/anny/index.js";
import {
  assignHands, handLandmarks3D, matchHandChirality, MP, type WorldLandmark,
} from "../../src/mediapipe/index.js";
import { rigInfo } from "../../src/mediapipe/rig.js";
import { loadFixtureModel } from "../_helpers/model.js";
import { geometricSide, handAngles, mean } from "../roundtrip/metrics.js";
import { buildRigExtras, handJointsAnny, toMpWorld, type RigExtras } from "../roundtrip/skeleton.js";
import { posedSkeleton } from "../roundtrip/synthetic-landmarks.js";
import { angleDeg, cross, norm, sub } from "../roundtrip/vec.js";

let model: AnnyModel;
let boneIndex: Map<string, number>;
let extras: RigExtras;

beforeAll(async () => {
  model = await loadFixtureModel();
  boneIndex = buildBoneIndex(model);
  extras = buildRigExtras(model);
});

const rotX = (deg: number) => rodriguesToMat3(new Float32Array([1, 0, 0]), deg * Math.PI / 180);

describe("rig conventions", () => {
  test("+X on every phalanx flexes toward the palm, on both hands", () => {
    for (const side of ["L", "R"] as const) {
      const rest = handAngles(toMpWorld(handJointsAnny(posedSkeleton(model, extras, identityDeltas(model.boneCount)), side)), side);
      const d = identityDeltas(model.boneCount);
      for (const f of [2, 3, 4, 5]) for (const j of [1, 2, 3]) d[boneIndex.get(`finger${f}-${j}.${side}`)!] = rotX(30);
      const curled = handAngles(toMpWorld(handJointsAnny(posedSkeleton(model, extras, d), side)), side);
      for (let f = 1; f <= 4; f++) for (let j = 0; j < 3; j++) {
        expect(curled.flex[f][j] - rest.flex[f][j]).toBeGreaterThan(20);
      }
    }
  });

  test("driver's probed signs: flexion +X on both hands, abduction mirrored", () => {
    const rig = rigInfo(model);
    expect(rig.hands.L.flexSign).toBe(1);
    expect(rig.hands.R.flexSign).toBe(1);
    expect(rig.hands.L.abdSign).toBe(-rig.hands.R.abdSign);
  });

  test("+X on the elbow and the knee increases their bend", () => {
    for (const side of ["L", "R"] as const) {
      for (const [upper, lower, end] of [["upperarm01", "lowerarm01", "wrist"], ["upperleg01", "lowerleg01", "foot"]]) {
        const bend = (deg: number) => {
          const d = identityDeltas(model.boneCount);
          d[boneIndex.get(`${lower}.${side}`)!] = rotX(deg);
          const sk = posedSkeleton(model, extras, d);
          const a = sub(sk.head(`${lower}.${side}`), sk.head(`${upper}.${side}`));
          const b = sub(sk.head(`${end}.${side}`), sk.head(`${lower}.${side}`));
          return angleDeg(a, b);
        };
        expect(bend(30)).toBeGreaterThan(bend(0) + 20);
      }
    }
  });

  test("palmar normal = −(fwd × across) on a curled RIGHT hand, +(…) on a LEFT hand", () => {
    for (const side of ["L", "R"] as const) {
      const d = identityDeltas(model.boneCount);
      for (const f of [2, 3, 4, 5]) for (const j of [1, 2, 3]) d[boneIndex.get(`finger${f}-${j}.${side}`)!] = rotX(40);
      const p = toMpWorld(handJointsAnny(posedSkeleton(model, extras, d), side));
      expect(geometricSide(p)).toBe(side);
      const v = (i: number): [number, number, number] => [p[i].x, p[i].y, p[i].z];
      const c = norm(cross(sub(v(9), v(0)), sub(v(5), v(17))));
      const tips = mean([8, 12, 16, 20].map(i => {
        const t = sub(v(i), v(i - 3));
        return t[0] * c[0] + t[1] * c[1] + t[2] * c[2];
      }));
      expect(Math.sign(tips)).toBe(side === "L" ? 1 : -1);
    }
  });
});

describe("hand helpers", () => {
  const hand: WorldLandmark[] = Array.from({ length: 21 }, (_, i) => ({ x: i * 0.01, y: -i * 0.002, z: 0.03 + i * 0.001 }));

  test("matchHandChirality is a no-op when the label agrees", () => {
    expect(matchHandChirality(hand, "L", "Left")).toBe(hand);
  });

  test("matchHandChirality reflects depth about the wrist when the label disagrees", () => {
    const out = matchHandChirality(hand, "R", "Left");
    out.forEach((p, i) => {
      expect(p.x).toBe(hand[i].x);
      expect(p.y).toBe(hand[i].y);
      expect(p.z).toBeCloseTo(2 * hand[0].z - hand[i].z, 12);
    });
  });

  test("handLandmarks3D scales x/z by width and y by height", () => {
    const out = handLandmarks3D([{ x: 0.5, y: 0.25, z: -0.1 }], 640, 480);
    expect(out[0]).toEqual({ x: 320, y: 120, z: -64 });
  });

  test("assignHands routes by distance to the pose's anatomical wrists", () => {
    const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
    pose[MP.LEFT_WRIST] = { x: 0.8, y: 0.5 };   // subject's left = image right
    pose[MP.RIGHT_WRIST] = { x: 0.2, y: 0.5 };
    const near = (x: number) => Array.from({ length: 21 }, () => ({ x, y: 0.5 }));
    const worldA: WorldLandmark[] = [{ x: 1, y: 0, z: 0 }];
    const worldB: WorldLandmark[] = [{ x: 2, y: 0, z: 0 }];
    const out = assignHands([near(0.21), near(0.79)], [worldA, worldB], pose);
    expect(out.rightHand).toBe(worldA);
    expect(out.leftHand).toBe(worldB);
  });
});
