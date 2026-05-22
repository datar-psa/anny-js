#!/usr/bin/env bun
/**
 * One-shot: run the driver over each captured fixture and report mean / p50 /
 * p95 / max angular error (both overall and per-segment). Use the numbers to
 * pick a tight threshold for the captured-fixture parity test.
 */
import {
  buildBoneIndex, allocBoneTransforms, forwardKinematics,
} from "../src/anny/index.js";
import { landmarksToPoseDeltas, type WorldLandmark } from "../src/mediapipe/index.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { verifyPoseAlignment } from "../tests/mediapipe/verify.js";
import { resolve } from "node:path";

const model = await loadFixtureModel();
const boneIndex = buildBoneIndex(model);

function pct(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

for (const name of ["boxer", "dancer"]) {
  const path = resolve("tests/mediapipe/fixtures", `${name}_landmarks.json`);
  const data: { frames: { worldLandmarks: WorldLandmark[] }[] } = JSON.parse(
    await Bun.file(path).text(),
  );

  const meanPerFrame: number[] = [];
  const perSegment: Record<string, number[]> = {};

  for (const frame of data.frames) {
    const lm = frame.worldLandmarks;
    // Match the test's runDriver: default visibilityMin (0.5) — anything else
    // and the threshold won't reflect what the test actually measures.
    const deltas = landmarksToPoseDeltas({ pose: lm }, model, boneIndex);
    const boneXforms = allocBoneTransforms(model.boneCount);
    forwardKinematics(model, deltas, boneXforms);
    const { meanErrDeg, segments } = verifyPoseAlignment(model, boneIndex, boneXforms, lm);
    meanPerFrame.push(meanErrDeg);
    for (const s of segments) {
      (perSegment[s.seg] ??= []).push(s.errDeg);
    }
  }

  const overall = meanPerFrame.reduce((a, b) => a + b, 0) / meanPerFrame.length;
  console.log(`\n${name}  (${data.frames.length} frames)`);
  console.log(`  overall mean:  ${overall.toFixed(2)}°`);
  console.log(`  per-frame mean p50 / p95 / max: ${pct(meanPerFrame,50).toFixed(2)}° / ${pct(meanPerFrame,95).toFixed(2)}° / ${Math.max(...meanPerFrame).toFixed(2)}°`);
  console.log(`  per-segment mean (over frames):`);
  for (const [seg, errs] of Object.entries(perSegment)) {
    const m = errs.reduce((a, b) => a + b, 0) / errs.length;
    console.log(`    ${seg.padEnd(12)}  mean ${m.toFixed(1)}°   p95 ${pct(errs, 95).toFixed(1)}°   max ${Math.max(...errs).toFixed(1)}°`);
  }
}
