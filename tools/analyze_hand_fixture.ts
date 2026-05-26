#!/usr/bin/env bun
/**
 * One-off diagnostic: run the hand verifier across every frame of a real
 * captured fixture (with hand landmarks) and print per-bone error stats,
 * naturality flags, and a few raw landmark dumps for inspection.
 *
 * Usage: `bun tools/analyze_hand_fixture.ts boxer`
 */

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildBoneIndex, allocBoneTransforms, forwardKinematics,
} from "../src/anny/index.js";
import { landmarksToPoseDeltas, type WorldLandmark } from "../src/mediapipe/index.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { verifyHand, formatVerifyResult, verifyHandVsPose, handNaturality } from "../tests/mediapipe/verify_hands.js";

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(here, "../tests/mediapipe/fixtures");

interface Frame {
  worldLandmarks: WorldLandmark[];
  imageLandmarks: WorldLandmark[];
  leftHand?:  { world: WorldLandmark[]; image: WorldLandmark[] };
  rightHand?: { world: WorldLandmark[]; image: WorldLandmark[] };
}

const name = process.argv[2] ?? "boxer";
const path = resolve(FIXTURES, `${name}_landmarks.json`);
const data: { frames: Frame[] } = JSON.parse(await Bun.file(path).text());
const handFrames = data.frames.filter(f => f.leftHand || f.rightHand);

console.log(`fixture=${name}  total=${data.frames.length}  withHands=${handFrames.length}`);
if (handFrames.length === 0) { console.log("no hand frames; bail"); process.exit(0); }

const model = await loadFixtureModel();
const boneIndex = buildBoneIndex(model);
const xforms = allocBoneTransforms(model.boneCount);

// Aggregate per-bone error stats across all frames.
const perBone: Record<string, { sum: number; max: number; n: number }> = {};
const curlConsistency: Record<string, { ok: number; bad: number }> = {};
let printedSample = 0;

for (const [idx, f] of handFrames.entries()) {
  const deltas = landmarksToPoseDeltas({
    pose: f.worldLandmarks,
    leftHand:  f.leftHand?.world,
    rightHand: f.rightHand?.world,
    visibilityMin: 0.2,
  }, model, boneIndex);
  forwardKinematics(model, deltas, xforms);

  for (const side of ["L", "R"] as const) {
    const h = side === "L" ? f.leftHand?.world : f.rightHand?.world;
    if (!h) continue;
    const r = verifyHand(model, boneIndex, xforms, h, side);
    for (const b of r.bones) {
      const key = b.bone;
      (perBone[key] ??= { sum: 0, max: 0, n: 0 });
      perBone[key].sum += b.errDeg;
      perBone[key].max = Math.max(perBone[key].max, b.errDeg);
      perBone[key].n++;
    }
    for (const fc of r.naturality.fingerCurls) {
      const key = `${fc.finger}.${side}`;
      (curlConsistency[key] ??= { ok: 0, bad: 0 });
      if (fc.signsConsistent) curlConsistency[key].ok++;
      else                    curlConsistency[key].bad++;
    }
    // Print full breakdown for the first 2 hand frames so a human can read.
    if (printedSample < 2) {
      console.log(`\n── FRAME ${idx} ${side} ──`);
      console.log(formatVerifyResult(r));
      printedSample++;
    }
  }
}

console.log("\n══ Per-bone error stats across all frames ══");
console.log("bone               mean°    max°   n");
const sorted = Object.entries(perBone).sort(([a],[b]) => a.localeCompare(b));
for (const [k, s] of sorted) {
  console.log(`  ${k.padEnd(18)} ${(s.sum/s.n).toFixed(1).padStart(6)} ${s.max.toFixed(1).padStart(7)} ${s.n.toString().padStart(3)}`);
}

console.log("\n══ Finger-curl naturality (sign consistency across joints) ══");
console.log("finger             OK   bad");
for (const [k, c] of Object.entries(curlConsistency).sort()) {
  console.log(`  ${k.padEnd(15)} ${c.ok.toString().padStart(4)}  ${c.bad.toString().padStart(4)}`);
}

// ── INDEPENDENT pose-grounded check (the non-circular one) ──────────────────
console.log("\n══ INDEPENDENT: posed bones vs POSE landmarks (deg) ══");
console.log("Compares avatar metacarpal/wrist bones to pose's own WRIST/INDEX/PINKY.");
console.log("This shares NO math with the driver targets — it's the true error.\n");
const poseAgg: Record<string, { idxSum: number; pnkSum: number; wrSum: number; curl: number[]; curlBackward: number[]; n: number }> = {};
for (const f of handFrames) {
  const deltas = landmarksToPoseDeltas({
    pose: f.worldLandmarks,
    leftHand:  f.leftHand?.world,
    rightHand: f.rightHand?.world,
    visibilityMin: 0.2,
  }, model, boneIndex);
  forwardKinematics(model, deltas, xforms);
  for (const side of ["L", "R"] as const) {
    const h = side === "L" ? f.leftHand?.world : f.rightHand?.world;
    if (!h) continue;
    const r = verifyHandVsPose(model, boneIndex, xforms, f.worldLandmarks, side);
    if (!r.ran) continue;
    const a = (poseAgg[side] ??= { idxSum: 0, pnkSum: 0, wrSum: 0, curl: [0,0,0,0,0], curlBackward: [0,0,0,0,0], n: 0 });
    a.idxSum += r.indexMetacarpalErrDeg;
    a.pnkSum += r.pinkyMetacarpalErrDeg;
    a.wrSum  += r.wristErrDeg;
    r.fingerCurlPalmward.forEach((c, i) => {
      a.curl[i] += c;
      if (c < -0.05) a.curlBackward[i]++;   // counts clearly-backward frames
    });
    a.n++;
  }
}
console.log("side  index-meta°  pinky-meta°  wrist°   n");
for (const [side, a] of Object.entries(poseAgg)) {
  console.log(`  ${side}    ${(a.idxSum/a.n).toFixed(1).padStart(8)}   ${(a.pnkSum/a.n).toFixed(1).padStart(8)}   ${(a.wrSum/a.n).toFixed(1).padStart(6)}  ${a.n}`);
}
console.log("\nPer-finger curl palmward (mean; POSITIVE=toward palm/good, NEGATIVE=backward/bad):");
console.log("side  thumb   index   middle  ring    pinky    [backward-frame counts]");
for (const [side, a] of Object.entries(poseAgg)) {
  const means = a.curl.map(c => (c/a.n).toFixed(2).padStart(6)).join("  ");
  const bw = a.curlBackward.join(",");
  console.log(`  ${side}   ${means}    [${bw}] of ${a.n}`);
}

// ── Naturality vs human norms ───────────────────────────────────────────────
console.log("\n══ NATURALITY (posed avatar vs human norms) ══");
console.log("Human norms: MCP 0-90°, PIP 0-110°, DIP 0-80°, DIP/PIP ≈ 0.6-0.8,");
console.log("adjacent fingertips near-parallel (<25°), cross-finger curl std small-but-nonzero.\n");
const natAgg: Record<string, { mcp: number[]; pip: number[]; dip: number[]; ratio: number[]; spread: number; tipPar: number[]; n: number }> = {};
for (const f of handFrames) {
  const deltas = landmarksToPoseDeltas({
    pose: f.worldLandmarks, leftHand: f.leftHand?.world, rightHand: f.rightHand?.world, visibilityMin: 0.2,
  }, model, boneIndex);
  forwardKinematics(model, deltas, xforms);
  for (const side of ["L", "R"] as const) {
    if (!(side === "L" ? f.leftHand : f.rightHand)) continue;
    const nat = handNaturality(model, boneIndex, xforms, side);
    const a = (natAgg[side] ??= { mcp: [0,0,0,0], pip: [0,0,0,0], dip: [0,0,0,0], ratio: [0,0,0,0], spread: 0, tipPar: [0,0,0], n: 0 });
    nat.jointDeg.forEach((j, i) => { a.mcp[i]+=j.mcp; a.pip[i]+=j.pip; a.dip[i]+=j.dip; });
    nat.dipPipRatio.forEach((r, i) => a.ratio[i]+=r);
    a.spread += nat.crossFingerCurlStdDeg;
    nat.adjacentTipParallelDeg.forEach((t, i) => a.tipPar[i]+=t);
    a.n++;
  }
}
for (const [side, a] of Object.entries(natAgg)) {
  console.log(`-- ${side} hand (mean over ${a.n} frames) --`);
  console.log("finger   MCP°   PIP°   DIP°   DIP/PIP");
  ["index","middle","ring","pinky"].forEach((fn, i) => {
    console.log(`  ${fn.padEnd(7)} ${(a.mcp[i]/a.n).toFixed(0).padStart(4)}  ${(a.pip[i]/a.n).toFixed(0).padStart(5)}  ${(a.dip[i]/a.n).toFixed(0).padStart(5)}   ${(a.ratio[i]/a.n).toFixed(2)}`);
  });
  console.log(`  cross-finger curl std: ${(a.spread/a.n).toFixed(1)}°   adjacent-tip angles: ${a.tipPar.map(t=>(t/a.n).toFixed(0)+"°").join(" ")}`);
}

// Raw landmark inspection for the first frame with both hands — gives a
// quick sanity check on the MP world frame orientation.
const both = handFrames.find(f => f.leftHand && f.rightHand);
if (both) {
  const fmt = (lm: WorldLandmark) => `(${lm.x.toFixed(3)}, ${lm.y.toFixed(3)}, ${lm.z.toFixed(3)})`;
  const inspect = (label: string, h: { world: WorldLandmark[] }) => {
    const w = h.world;
    console.log(`\n── ${label} raw MP world landmarks (origin = hand centroid) ──`);
    console.log(`  WRIST       ${fmt(w[0])}`);
    console.log(`  THUMB_CMC   ${fmt(w[1])}`);
    console.log(`  INDEX_MCP   ${fmt(w[5])}`);
    console.log(`  MIDDLE_MCP  ${fmt(w[9])}`);
    console.log(`  PINKY_MCP   ${fmt(w[17])}`);
    console.log(`  → thumb_CMC.x - wrist.x = ${(w[1].x - w[0].x).toFixed(3)}  ` +
      `(LEFT hand expects ~negative for anatomical thumb-medial)`);
    console.log(`  → middle_MCP - wrist = ${fmt({x: w[9].x-w[0].x, y: w[9].y-w[0].y, z: w[9].z-w[0].z} as WorldLandmark)}`);
  };
  inspect("LEFT hand",  both.leftHand!);
  inspect("RIGHT hand", both.rightHand!);
}
