/**
 * Headless browser regression gates for the MediaPipe → Anny driver.
 *
 * Runs the real MediaPipe models (wasm, CPU) in headless Chromium against:
 *   • synthetic hands with known poses   (ground truth: rig angles)
 *   • MediaPipe's own labelled hand photos (round trip: photo → MP → rig → render → MP)
 *   • frames of the demo clips            (round trip on full-body video)
 *
 * Slow-ish (~1 min) and needs the model cache, so it is opt-in:
 *   ANNY_BROWSER=1 bun test tests/browser      (or `task test:browser`)
 *
 * Thresholds sit ~20–25 % above the measured values (see docs/hand-tracking.md)
 * so they catch regressions, not noise. MediaPipe and SwiftShader are both
 * deterministic on CPU, so runs repeat exactly on one machine.
 */

import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import type { AnnyModel } from "../../src/anny/index.js";
import { landmarksToPoseDeltas } from "../../src/mediapipe/index.js";
import { fetchAssets } from "../../tools/fetch_assets.js";
import { loadFixtureModel } from "../_helpers/model.js";
import { PHOTOS, photoPath, videoFrames } from "../roundtrip/cases.js";
import { HarnessClient } from "../roundtrip/client.js";
import { mean, percentile } from "../roundtrip/metrics.js";
import { makeContext, runImageCase, type CaseResult, type RunContext } from "../roundtrip/run.js";
import { summarize } from "../roundtrip/report.js";
import { handPoseSpecs, runSyntheticHand, type SyntheticHandResult } from "../roundtrip/synthetic.js";
import { chromiumAvailable } from "./browser.js";

const ENABLED = process.env.ANNY_BROWSER === "1";
const SLOW = 600_000;

let client: HarnessClient | undefined;
let ctx: RunContext;
let model: AnnyModel;

describe.skipIf(!ENABLED)("headless MediaPipe round trip", () => {
  beforeAll(async () => {
    if (!chromiumAvailable()) throw new Error("no Chromium for Playwright (set ANNY_CHROMIUM)");
    await fetchAssets(() => {});
    model = await loadFixtureModel();
    client = await HarnessClient.start({ poseModel: "heavy" });
    ctx = makeContext(client, model);
  }, SLOW);

  afterAll(async () => {
    await client?.close();
  });

  test("synthetic hands: MediaPipe reads our renders, the driver adds no error", async () => {
    const results: SyntheticHandResult[] = [];
    for (const spec of handPoseSpecs(30, 7)) results.push(await runSyntheticHand(ctx, spec, landmarksToPoseDeltas));
    const ok = results.filter(r => r.mpVsGt && r.drivenVsGt);
    const m = (f: (r: SyntheticHandResult) => number) => mean(ok.map(f));

    expect(ok.length / results.length).toBeGreaterThan(0.75);
    // Absolute accuracy of the whole pipeline against the known pose.
    expect(m(r => r.drivenVsGt!.fingerFlex)).toBeLessThan(26);
    expect(m(r => r.drivenVsGt!.palmNormal)).toBeLessThan(22);
    expect(m(r => r.drivenVsGt2D!)).toBeLessThan(0.2);
    // The driver must not be worse than its input: its anatomical model may
    // only remove MediaPipe noise, never add to it.
    expect(m(r => r.drivenVsGt!.fingerFlex)).toBeLessThan(m(r => r.mpVsGt!.fingerFlex) + 1);
    expect(m(r => r.drivenVsGt!.palmNormal)).toBeLessThan(m(r => r.mpVsGt!.palmNormal) + 1);
    const flips = (k: "mpVsGt" | "drivenVsGt") => ok.reduce((s, r) => s + r[k]!.bendSignFlips, 0);
    expect(flips("drivenVsGt")).toBeLessThanOrEqual(flips("mpVsGt"));
  }, SLOW);

  test("labelled hand photos: the render reads back like the photo", async () => {
    const results: CaseResult[] = [];
    for (const f of PHOTOS) results.push(await runImageCase(ctx, f, photoPath(f), landmarksToPoseDeltas));
    const s = summarize(results);

    // Driver fidelity: the rig does what MediaPipe saw (fingers only differ
    // where MediaPipe reports anatomically impossible flexion).
    expect(s.hands.palmNormal.srcVsModel).toBeLessThan(1);
    expect(s.hands.fingerFlex.srcVsModel).toBeLessThan(8);
    expect(s.hands.thumbFlex.srcVsModel).toBeLessThan(5);
    expect(s.hands.bendSignFlips.srcVsModel).toBe(0);
    // Round trip, judged by MediaPipe on the render.
    expect(s.hands.rtDetected / s.hands.n).toBeGreaterThan(0.85);
    expect(s.hands.shape2D.srcVsRt).toBeLessThan(0.13);
    expect(s.hands.fingerFlex.srcVsRt).toBeLessThan(18);
    expect(s.hands.bendSignFlips.srcVsRt).toBeLessThanOrEqual(2);
    expect(s.body.srcVsRt).toBeLessThan(9);
  }, SLOW);

  test("demo clips: full-body round trip", async () => {
    const results: CaseResult[] = [];
    for (const clip of ["dancer", "boxer"]) {
      for (const p of await videoFrames(clip, 8)) results.push(await runImageCase(ctx, p, p, landmarksToPoseDeltas));
    }
    const s = summarize(results);
    const limbErr = results.flatMap(r => (r.body?.srcVsRt ? Object.values(r.body.srcVsRt) : []));
    expect(s.body.srcVsModel).toBeLessThan(1);
    expect(s.body.srcVsRt).toBeLessThan(14);
    expect(percentile(limbErr, 0.5)).toBeLessThan(10);
    expect(s.hands.bendSignFlips.srcVsModel).toBeLessThanOrEqual(3);
  }, SLOW);
});
