#!/usr/bin/env bun
/**
 * Run MediaPipe (pose heavy + per-wrist ROI hands, as the video demo does) on
 * every frame of the demo clips and cache the raw detections, so driver /
 * smoothing variants can be replayed offline in milliseconds:
 *
 *   bun tools/capture_sequences.ts [--frames 96]   → .cache/sequences/<clip>.json
 */

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { videoFrames, VIDEOS } from "../tests/roundtrip/cases.js";
import { CACHE_DIR, HarnessClient } from "../tests/roundtrip/client.js";
import type { Detection } from "../tests/roundtrip/protocol.js";
import { fetchAssets } from "./fetch_assets.js";

const { values: args } = parseArgs({ options: { frames: { type: "string", default: "96" } } });

await fetchAssets();
const outDir = resolve(CACHE_DIR, "sequences");
mkdirSync(outDir, { recursive: true });
const client = await HarnessClient.start({ poseModel: "heavy" });
try {
  for (const clip of VIDEOS) {
    const frames = await videoFrames(clip, Number(args.frames));
    const detections: Detection[] = [];
    for (const f of frames) detections.push(await client.detectFile(f));
    const hands = detections.reduce((s, d) => s + d.hands.length, 0);
    await Bun.write(resolve(outDir, `${clip}.json`), JSON.stringify({ clip, fps: 24, detections }));
    console.log(`${clip}: ${frames.length} frames, ${hands} hand detections`);
  }
} finally {
  await client.close();
}
