/**
 * Naturalness gate on the demo clips: real MediaPipe (pose + per-wrist ROI
 * hands, as the video demo runs it) on the first seconds of both clips,
 * replayed through `PoseTracker`, every driven hand scored for anatomy,
 * finger penetration, arm range of motion and temporal snaps.
 *
 * Opt-in like the rest of tests/browser: ANNY_BROWSER=1 bun test tests/browser
 */

import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import type { AnnyModel } from "../../src/anny/index.js";
import { landmarksToPoseDeltas } from "../../src/mediapipe/index.js";
import { fetchAssets } from "../../tools/fetch_assets.js";
import { loadFixtureModel } from "../_helpers/model.js";
import { buildPhysicalRig } from "../naturalness/physical.js";
import type { Sequence } from "../naturalness/replay.js";
import { NaturalnessScorer } from "../naturalness/score.js";
import { scoreSequence, type SequenceScore } from "../naturalness/sequences.js";
import { videoFrames } from "../roundtrip/cases.js";
import { HarnessClient } from "../roundtrip/client.js";
import type { Detection } from "../roundtrip/protocol.js";
import { buildRigExtras } from "../roundtrip/skeleton.js";
import { chromiumAvailable } from "./browser.js";

const ENABLED = process.env.ANNY_BROWSER === "1";
const SLOW = 600_000;
/** 3 s of each clip at 24 fps. */
const FRAMES = 72;

let client: HarnessClient | undefined;
let model: AnnyModel;

describe.skipIf(!ENABLED)("naturalness of driven hands on the demo clips", () => {
  beforeAll(async () => {
    if (!chromiumAvailable()) throw new Error("no Chromium for Playwright (set ANNY_CHROMIUM)");
    await fetchAssets(() => {});
    model = await loadFixtureModel();
    client = await HarnessClient.start({ poseModel: "heavy" });
  }, SLOW);

  afterAll(async () => {
    await client?.close();
  });

  test("PoseTracker keeps every hand anatomically and physically plausible, without snaps", async () => {
    const extras = buildRigExtras(model);
    const scorer = new NaturalnessScorer(model, extras, buildPhysicalRig(model));
    const scores: SequenceScore[] = [];
    for (const clip of ["dancer", "boxer"]) {
      const detections: Detection[] = [];
      for (const f of await videoFrames(clip, FRAMES)) detections.push(await client!.detectFile(f));
      const seq: Sequence = { clip, fps: 24, detections };
      scores.push(...scoreSequence(seq, { model, extras, scorer }, landmarksToPoseDeltas, { hands: "tracker", handsEveryOther: true }));
    }
    expect(scores.length).toBeGreaterThanOrEqual(3);   // both clips, most hands found
    for (const s of scores) {
      const what = `${s.clip} ${s.side}`;
      expect(s.implausible, what).toBeLessThanOrEqual(0.05);
      expect(s.anatomy, what).toBeLessThan(2);
      expect(s.penetration, what).toBeLessThan(1);
      expect(s.armRom, what).toBeLessThan(1);
      expect(s.snaps, what).toBeLessThanOrEqual(8);
    }
  }, SLOW);
});
