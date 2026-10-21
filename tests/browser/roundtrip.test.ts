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
import { driverInput, makeContext, runImageCase, type CaseResult, type Driver, type RunContext } from "../roundtrip/run.js";
import { buildPhysicalRig } from "../naturalness/physical.js";
import { NaturalnessScorer } from "../naturalness/score.js";
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

  test("synthetic hands: the driver is closer to the truth than MediaPipe's reading", async () => {
    const results: SyntheticHandResult[] = [];
    for (const spec of handPoseSpecs(30, 7)) results.push(await runSyntheticHand(ctx, spec, landmarksToPoseDeltas));
    const ok = results.filter(r => r.mpVsGt && r.drivenVsGt);
    const m = (f: (r: SyntheticHandResult) => number) => mean(ok.map(f));

    expect(ok.length / results.length).toBeGreaterThan(0.75);
    // Absolute accuracy of the whole pipeline against the known pose.
    expect(m(r => r.drivenVsGt!.fingerFlex)).toBeLessThan(18);
    expect(m(r => r.drivenVsGt!.palmNormal)).toBeLessThan(22);
    expect(m(r => r.drivenVsGt2D!)).toBeLessThan(0.2);
    // Fused with the natural-hand prior, the hand is closer to the truth
    // than MediaPipe's own reading — and bends the wrong way far less often.
    expect(m(r => r.drivenVsGt!.fingerFlex)).toBeLessThan(0.8 * m(r => r.mpVsGt!.fingerFlex));
    expect(m(r => r.drivenVsGt!.palmNormal)).toBeLessThan(m(r => r.mpVsGt!.palmNormal) + 1);
    const flips = (k: "mpVsGt" | "drivenVsGt") => ok.reduce((s, r) => s + r[k]!.bendSignFlips, 0);
    expect(flips("drivenVsGt")).toBeLessThanOrEqual(0.25 * flips("mpVsGt"));
  }, SLOW);

  test("labelled hand photos: geometry reproduces MediaPipe's reading", async () => {
    // Without hand quality the reading is taken at face value: the rig must
    // do what MediaPipe saw, except where that leaves the joints' range of
    // motion (fingers past their limits, a wrist on a forearm whose depth the
    // pose got wrong).
    const asRead: Driver = (input, m) => landmarksToPoseDeltas({ ...input, leftHandQuality: undefined, rightHandQuality: undefined }, m);
    const results: CaseResult[] = [];
    for (const f of PHOTOS) results.push(await runImageCase(ctx, f, photoPath(f), asRead, { renderHands: false, renderBody: false }));
    const s = summarize(results);
    expect(s.hands.palmNormal.srcVsModel).toBeLessThan(1);
    expect(s.hands.fingerFlex.srcVsModel).toBeLessThan(6);
    expect(s.hands.thumbFlex.srcVsModel).toBeLessThan(7);
    // pose.jpg's left thumb is read inside the index finger; clearance moves it out.
    expect(s.hands.bendSignFlips.srcVsModel).toBeLessThanOrEqual(1);
  }, SLOW);

  test("labelled hand photos: the render reads back like the photo", async () => {
    const results: CaseResult[] = [];
    for (const f of PHOTOS) results.push(await runImageCase(ctx, f, photoPath(f), landmarksToPoseDeltas));
    const s = summarize(results);
    // With quality the reading is fused with the natural-hand prior, so the
    // rig departs from it on purpose — most of all where MediaPipe reports
    // anatomically impossible fingers (fist.jpg: a 147° PIP with the
    // fingertip hyperextended −31°). The round trip bounds how far…
    expect(s.hands.rtDetected / s.hands.n).toBeGreaterThan(0.85);
    expect(s.hands.shape2D.srcVsRt).toBeLessThan(0.16);
    expect(s.hands.fingerFlex.srcVsRt).toBeLessThan(20);
    expect(s.hands.bendSignFlips.srcVsRt).toBeLessThanOrEqual(3);
    expect(s.body.srcVsRt).toBeLessThan(9);
    // …and every driven hand is one a person could make.
    const scorer = new NaturalnessScorer(model, ctx.extras, buildPhysicalRig(model));
    for (const r of results) {
      const input = driverInput(r.detection, ctx.handSource);
      if (!input) continue;
      const deltas = landmarksToPoseDeltas(input, model);
      for (const h of r.hands) {
        const n = scorer.score(deltas, h.side);
        expect(n.anatomy.total, `${r.name} ${h.side}`).toBeLessThan(5);
        expect(n.physical.penetrationMm, `${r.name} ${h.side}`).toBeLessThan(3);
        expect(n.physical.armRom, `${r.name} ${h.side}`).toBeLessThan(1);
      }
    }
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
