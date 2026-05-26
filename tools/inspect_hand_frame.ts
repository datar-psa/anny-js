#!/usr/bin/env bun
/**
 * Dig into the captured MP hand worldLandmarks to figure out which frame
 * they're in.
 *
 *   Hypothesis A (camera-aligned, like Pose):  WRIST → MIDDLE_TIP varies
 *       frame-to-frame as the boxer moves their hand around the scene.
 *   Hypothesis B (hand-local):                 WRIST → MIDDLE_TIP stays
 *       roughly constant (hand orientation absorbed into the frame itself).
 *
 * Also dump per-frame:
 *   • thumb_CMC.x - wrist.x for each hand (chirality probe — negative for
 *     anatomical L, positive for anatomical R when frame is world-aligned)
 *   • angle between wrist→middle_MCP (hand-frame "forward") and
 *     pose forearm direction (in matching Anny frame) — if they roughly
 *     agree, hand IS world-aligned.
 */

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { WorldLandmark } from "../src/mediapipe/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURES = resolve(here, "../tests/mediapipe/fixtures");
const name = process.argv[2] ?? "boxer";

interface Frame {
  worldLandmarks: WorldLandmark[];
  leftHand?:  { world: WorldLandmark[] };
  rightHand?: { world: WorldLandmark[] };
}
const data: { frames: Frame[] } = JSON.parse(await Bun.file(resolve(FIXTURES, `${name}_landmarks.json`)).text());

const handFrames = data.frames.filter(f => f.leftHand || f.rightHand);

const v3 = (a: WorldLandmark, b: WorldLandmark) => ({
  x: a.x - b.x, y: a.y - b.y, z: a.z - b.z,
});
const len = (v: { x: number; y: number; z: number }) => Math.hypot(v.x, v.y, v.z);
const norm = (v: { x: number; y: number; z: number }) => {
  const L = len(v) || 1;
  return { x: v.x/L, y: v.y/L, z: v.z/L };
};
const ang = (a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }) => {
  const an = norm(a), bn = norm(b);
  const d = Math.max(-1, Math.min(1, an.x*bn.x + an.y*bn.y + an.z*bn.z));
  return Math.acos(d) * 180 / Math.PI;
};

console.log("frame | side | thumb_x-w | mid-tip vec (raw MP)              | wrist→midMCP (MP)               | forearm dir (MP)                | hand-forearm°");
console.log("-".repeat(160));
for (const [idx, f] of handFrames.entries()) {
  for (const side of ["L", "R"] as const) {
    const h = side === "L" ? f.leftHand?.world : f.rightHand?.world;
    if (!h) continue;
    const w = h[0];        // hand WRIST
    const mt = h[12];      // MIDDLE_TIP
    const mcp = h[9];      // MIDDLE_MCP
    const thumbCmc = h[1]; // THUMB_CMC
    const fingerDir = v3(mt, w);
    const wristMidMCP = v3(mcp, w);
    // Pose's elbow→wrist direction in MP world (forearm direction).
    const elbow = f.worldLandmarks[side === "L" ? 13 : 14];
    const poseWrist = f.worldLandmarks[side === "L" ? 15 : 16];
    const forearmDir = v3(poseWrist, elbow);
    const handForearm = ang(wristMidMCP, forearmDir);
    const dx = thumbCmc.x - w.x;
    const fmt = (v: { x: number; y: number; z: number }) =>
      `(${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)})`;
    console.log(
      `  ${idx.toString().padStart(3)} |  ${side}   | ${dx.toFixed(3).padStart(6)} | ${fmt(fingerDir)} L=${len(fingerDir).toFixed(2)} | ${fmt(wristMidMCP)} L=${len(wristMidMCP).toFixed(2)} | ${fmt(forearmDir)} L=${len(forearmDir).toFixed(2)} | ${handForearm.toFixed(1).padStart(5)}°`,
    );
  }
}
