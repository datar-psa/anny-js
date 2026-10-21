#!/usr/bin/env bun
/**
 * Round-trip validation CLI:  image → MediaPipe → driver → render → MediaPipe.
 *
 *   bun tools/roundtrip.ts [--set photos|video|all] [--pose lite|full|heavy]
 *                          [--frames N] [--report DIR]
 *
 * Prints a per-case metric table and writes an HTML report (source image,
 * renders, metrics) to `.cache/roundtrip/` for eyeballing.
 */

import { basename, resolve } from "node:path";
import { parseArgs } from "node:util";

import { landmarksToPoseDeltas } from "../src/mediapipe/index.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { CACHE_DIR, HarnessClient } from "../tests/roundtrip/client.js";
import { makeContext, runImageCase, type CaseResult, type HandSource } from "../tests/roundtrip/run.js";
import { PHOTOS, photoPath, VIDEOS, videoFrames } from "../tests/roundtrip/cases.js";
import { writeReport, summarize, printTable } from "../tests/roundtrip/report.js";
import type { PoseModel } from "../tests/roundtrip/protocol.js";
import { fetchAssets } from "./fetch_assets.js";

const { values: args } = parseArgs({
  options: {
    set: { type: "string", default: "all" },
    pose: { type: "string", default: "heavy" },
    frames: { type: "string", default: "12" },
    report: { type: "string", default: resolve(CACHE_DIR, "roundtrip") },
    only: { type: "string" },
    source: { type: "string", default: "image" },
  },
});

async function main(): Promise<void> {
  await fetchAssets();
  const cases: { name: string; path: string }[] = [];
  if (args.set === "photos" || args.set === "all") {
    for (const f of PHOTOS) cases.push({ name: f.replace(/\.jpg$/, ""), path: photoPath(f) });
  }
  if (args.set === "video" || args.set === "all") {
    for (const clip of VIDEOS) {
      for (const p of await videoFrames(clip, Number(args.frames))) {
        cases.push({ name: `${clip}/${basename(p, ".jpg")}`, path: p });
      }
    }
  }
  const selected = args.only ? cases.filter(c => c.name.includes(args.only!)) : cases;

  const model = await loadFixtureModel();
  const client = await HarnessClient.start({ poseModel: args.pose as PoseModel });
  const ctx = makeContext(client, model, args.source as HandSource);
  const results: CaseResult[] = [];
  try {
    for (const c of selected) {
      const t0 = performance.now();
      const r = await runImageCase(ctx, c.name, c.path, landmarksToPoseDeltas);
      results.push(r);
      console.error(`  ${c.name.padEnd(28)} ${(performance.now() - t0).toFixed(0)} ms ${r.notes.join("; ")}`);
    }
  } finally {
    await client.close();
  }
  printTable(results);
  const dir = args.report!;
  await writeReport(dir, results, new Map(selected.map(c => [c.name, c.path])));
  console.log(JSON.stringify(summarize(results), null, 2));
  console.log(`report → ${resolve(dir, "index.html")}`);
}

await main();
