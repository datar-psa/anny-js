#!/usr/bin/env bun
/**
 * Synthetic ground-truth hands: rig pose → render → MediaPipe → driver → compare with the pose.
 *
 *   bun tools/synthetic.ts [--random N] [--seed S] [--source image|world] [--only NAME]
 */

import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { landmarksToPoseDeltas } from "../src/mediapipe/index.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { CACHE_DIR, HarnessClient } from "../tests/roundtrip/client.js";
import { mean } from "../tests/roundtrip/metrics.js";
import { writeSyntheticReport } from "../tests/roundtrip/report.js";
import { makeContext, type HandSource } from "../tests/roundtrip/run.js";
import { handPoseSpecs, runSyntheticHand, type SyntheticHandResult } from "../tests/roundtrip/synthetic.js";
import { fetchAssets } from "./fetch_assets.js";

const { values: args } = parseArgs({
  options: {
    random: { type: "string", default: "20" },
    seed: { type: "string", default: "7" },
    only: { type: "string" },
    source: { type: "string", default: "image" },
    out: { type: "string", default: resolve(CACHE_DIR, "synthetic") },
  },
});

await fetchAssets();
const model = await loadFixtureModel();
const client = await HarnessClient.start({ poseModel: "lite" });
const ctx = makeContext(client, model, args.source as HandSource);
const specs = handPoseSpecs(Number(args.random), Number(args.seed)).filter(s => !args.only || s.name.includes(args.only));
const results: SyntheticHandResult[] = [];
try {
  for (const s of specs) {
    const r = await runSyntheticHand(ctx, s, landmarksToPoseDeltas);
    results.push(r);
    const f = (x?: number) => (x === undefined ? "  -  " : x.toFixed(1).padStart(5));
    console.log(`${s.name.padEnd(16)} label ${r.mpLabel ?? "-"}${r.mpLabel && r.mpLabel !== s.side ? "!" : " "} ` +
      `fingers mp~gt ${f(r.mpVsGt?.fingerFlex)} drv~gt ${f(r.drivenVsGt?.fingerFlex)}  ` +
      `thumb ${f(r.mpVsGt?.thumbFlex)} ${f(r.drivenVsGt?.thumbFlex)}  ` +
      `palm ${f(r.mpVsGt?.palmNormal)} ${f(r.drivenVsGt?.palmNormal)}  ` +
      `flips ${r.mpVsGt?.bendSignFlips ?? "-"} ${r.drivenVsGt?.bendSignFlips ?? "-"}  ` +
      `2D ${f(r.mpVsGt2D && r.mpVsGt2D * 100)}% ${f(r.drivenVsGt2D && r.drivenVsGt2D * 100)}%`);
  }
} finally {
  await client.close();
}
const ok = results.filter(r => r.mpVsGt);
const m = (k: "fingerFlex" | "thumbFlex" | "palmNormal" | "handDir", which: "mpVsGt" | "drivenVsGt") =>
  mean(ok.map(r => r[which]![k])).toFixed(1);
console.log(`\n${ok.length}/${results.length} detected`);
for (const k of ["fingerFlex", "thumbFlex", "palmNormal", "handDir"] as const) {
  console.log(`${k.padEnd(12)} mp~gt ${m(k, "mpVsGt")}  driven~gt ${m(k, "drivenVsGt")}`);
}
console.log(`2D residual  mp~gt ${(mean(ok.map(r => r.mpVsGt2D!)) * 100).toFixed(1)}%  driven~gt ${(mean(ok.map(r => r.drivenVsGt2D!)) * 100).toFixed(1)}% of palm length`);
console.log(`sign flips   mp~gt ${ok.reduce((s, r) => s + r.mpVsGt!.bendSignFlips, 0)}  driven~gt ${ok.reduce((s, r) => s + r.drivenVsGt!.bendSignFlips, 0)}`);
console.log(`label≠side   ${ok.filter(r => r.mpLabel !== r.spec.side).length}`);
await writeSyntheticReport(args.out!, results.map(r => ({
  name: r.spec.name,
  png: r.png,
  image: r.mpImage,
  caption: r.mpVsGt
    ? `mp~gt ${r.mpVsGt.fingerFlex.toFixed(0)}° palm ${r.mpVsGt.palmNormal.toFixed(0)}° | drv~gt ${r.drivenVsGt!.fingerFlex.toFixed(0)}°`
    : "not detected",
})));
console.log(`report → ${resolve(args.out!, "index.html")}`);
