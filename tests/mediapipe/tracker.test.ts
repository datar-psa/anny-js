/**
 * PoseTracker on synthetic hand sequences: what a video filter must and must
 * not do, measured on the hand-shape state.
 */

import { beforeAll, describe, expect, test } from "bun:test";

import { buildBoneIndex, type AnnyModel, type PoseDeltas } from "../../src/anny/index.js";
import { HAND_PRIOR } from "../../src/mediapipe/hand-prior.js";
import { landmarksToPoseDeltas, PoseTracker, type HandQuality, type WorldLandmark } from "../../src/mediapipe/index.js";
import { rigInfo } from "../../src/mediapipe/rig.js";
import { loadFixtureModel } from "../_helpers/model.js";
import { GESTURES, gtHandDeltas } from "../naturalness/hand-gt.js";
import { handState } from "../naturalness/hand-state.js";
import { buildRigExtras, handJointsAnny, toMpWorld, type RigExtras } from "../roundtrip/skeleton.js";
import { posedSkeleton } from "../roundtrip/synthetic-landmarks.js";

let model: AnnyModel;
let extras: RigExtras;
const hands: Record<string, WorldLandmark[]> = {};
const CLEAR: HandQuality = { palmPx: 150, labelAgrees: true };
const FRAME_MS = 1000 / 30;

beforeAll(async () => {
  model = await loadFixtureModel();
  extras = buildRigExtras(model);
  const index = buildBoneIndex(model);
  for (const g of ["open", "fist"]) {
    const hand = { name: g, family: g, side: "R" as const, pose: GESTURES[g], view: { axis: [0, 0, 1] as [number, number, number], deg: 0 } };
    hands[g] = toMpWorld(handJointsAnny(posedSkeleton(model, extras, gtHandDeltas(hand, index, model.boneCount)), "R"));
  }
});

/** Finger part of the hand-shape state of the driven right hand. */
const fingers = (d: PoseDeltas) =>
  handState(handJointsAnny(posedSkeleton(model, extras, d), "R") as never, "R", rigInfo(model)).slice(0, 16);
const dist = (a: readonly number[], b: readonly number[]) =>
  Math.sqrt(a.slice(0, 16).reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / 16);

/** Feed `frames` (a landmark set or null = MediaPipe missed the hand) and record the finger state. */
function run(tracker: PoseTracker, frames: (WorldLandmark[] | null)[], start = 0): number[][] {
  return frames.map((h, i) => fingers(tracker.update(h ? { rightHand: h, rightHandQuality: CLEAR } : {}, (start + i) * FRAME_MS)));
}

describe("PoseTracker", () => {
  test("a lost hand relaxes toward a natural pose instead of freezing", () => {
    const tracker = new PoseTracker(model);
    const seen = run(tracker, Array(10).fill(hands.fist));
    const lost = run(tracker, Array(30).fill(null), 10);
    const prior = HAND_PRIOR.mean;
    const before = dist(seen[seen.length - 1], prior);
    expect(dist(lost[3], prior)).toBeLessThan(before);               // moving, not frozen…
    expect(dist(lost[29], prior)).toBeLessThan(0.5 * before);        // …into a relaxed hand within a second
    for (let i = 1; i < lost.length; i++) expect(dist(lost[i], lost[i - 1])).toBeLessThan(8);   // smoothly
  });

  test("a single garbled reading moves the hand only part of the way, and it recovers", () => {
    const tracker = new PoseTracker(model);
    const open = run(tracker, Array(15).fill(hands.open));
    const [blunder] = run(tracker, [hands.fist], 15);
    const after = run(tracker, Array(10).fill(hands.open), 16);
    const jump = dist(fingers(landmarksToPoseDeltas({ rightHand: hands.fist, rightHandQuality: CLEAR }, model)), open[14]);
    // A one-frame blunder and the first frame of a real fast change look
    // alike; the filter can only hedge between them.
    expect(dist(blunder, open[14])).toBeLessThan(0.5 * jump);
    expect(dist(after[9], open[14])).toBeLessThan(2);
  });

  test("a clear hand that really changes is followed within a few frames", () => {
    const tracker = new PoseTracker(model);
    const open = run(tracker, Array(15).fill(hands.open));
    const fist = run(tracker, Array(10).fill(hands.fist), 15);
    const target = fingers(landmarksToPoseDeltas({ rightHand: hands.fist, rightHandQuality: CLEAR }, model));
    const span = dist(open[14], target);
    expect(dist(fist[3], target)).toBeLessThan(0.25 * span);        // ~0.1 s at 30 fps
    expect(dist(fist[9], target)).toBeLessThan(0.2 * span);
  });

  test("reset forgets the hand", () => {
    const tracker = new PoseTracker(model);
    run(tracker, Array(5).fill(hands.fist));
    tracker.reset();
    const d = tracker.update({}, 0);
    expect(d.every(m => m === null)).toBe(true);
  });
});
