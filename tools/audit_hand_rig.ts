#!/usr/bin/env bun
/**
 * Print the rest configuration of every hand bone in the Anny model so we
 * know the ground truth before writing a hand verifier:
 *   - bone label + parent
 *   - rest translation (origin in world)
 *   - rest local-Y axis in world (= column-1 of the 3×3 rotation), which is
 *     what the driver targets via `dir` BoneTarget
 *   - rest local-X axis in world (useful for checking thumb-side / pinky-side
 *     for chirality)
 *
 * Run: `bun tools/audit_hand_rig.ts`
 */

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseAnnyModel, type AnnyManifest } from "../src/anny/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const ASSETS = resolve(here, "../assets");

const manifest: AnnyManifest = JSON.parse(
  await Bun.file(resolve(ASSETS, "anny_model.json")).text(),
);
const buf = await Bun.file(resolve(ASSETS, "anny_model.bin")).arrayBuffer();
const model = parseAnnyModel(manifest, buf);

// Anny world axes per CLAUDE notes:
//   +x = anatomical left, +y = depth (back), +z = up
const axisLabel = (v: [number, number, number]): string => {
  const [x, y, z] = v;
  const max = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
  if (max < 1e-3) return "≈0";
  if (Math.abs(x) === max) return x > 0 ? "+x (anat-left)" : "-x (anat-right)";
  if (Math.abs(y) === max) return y > 0 ? "+y (back)" : "-y (forward)";
  return z > 0 ? "+z (up)" : "-z (down)";
};

const fmt = (n: number) => n.toFixed(3).padStart(7);
const fmtV = (v: [number, number, number]) => `[${fmt(v[0])}, ${fmt(v[1])}, ${fmt(v[2])}]`;

function restCol(b: number, col: 0 | 1 | 2): [number, number, number] {
  const o = b * 16;
  return [model.restBonePoses[o + col], model.restBonePoses[o + 4 + col], model.restBonePoses[o + 8 + col]];
}
function restT(b: number): [number, number, number] {
  const o = b * 16;
  return [model.restBonePoses[o + 3], model.restBonePoses[o + 7], model.restBonePoses[o + 11]];
}

const handBones = model.boneLabels
  .map((name, i) => ({ name, i }))
  .filter(({ name }) =>
    name.startsWith("wrist") ||
    name.startsWith("metacarpal") ||
    name.startsWith("finger") ||
    name.startsWith("hand") ||
    name === "lowerarm01.L" || name === "lowerarm01.R" ||
    name === "lowerarm02.L" || name === "lowerarm02.R"
  );

console.log(`Found ${handBones.length} hand-related bones.\n`);

for (const side of ["L", "R"] as const) {
  console.log(`\n══════════ ${side} HAND ══════════`);
  const sided = handBones.filter(b => b.name.endsWith(`.${side}`) || b.name === "lowerarm01" || b.name === "lowerarm02");
  // Stable sort: forearm → wrist → metacarpals → fingers
  const order = (n: string): number => {
    if (n.startsWith("lowerarm01")) return 0;
    if (n.startsWith("lowerarm02")) return 1;
    if (n.startsWith("wrist"))      return 2;
    if (n.startsWith("metacarpal")) return 3;
    if (n.startsWith("finger1"))    return 4;  // thumb
    if (n.startsWith("finger2"))    return 5;
    if (n.startsWith("finger3"))    return 6;
    if (n.startsWith("finger4"))    return 7;
    if (n.startsWith("finger5"))    return 8;
    return 9;
  };
  sided.sort((a, b) => order(a.name) - order(b.name) || a.name.localeCompare(b.name));

  for (const { name, i } of sided) {
    const p = model.boneParents[i];
    const pName = p >= 0 ? model.boneLabels[p] : "(root)";
    const t = restT(i);
    const x = restCol(i, 0);
    const y = restCol(i, 1);
    const z = restCol(i, 2);
    console.log(
      `${name.padEnd(18)} idx=${i.toString().padStart(3)}  parent=${pName.padEnd(18)}  ` +
      `t=${fmtV(t)}\n` +
      `${" ".repeat(20)}  +X=${fmtV(x)} (${axisLabel(x)})\n` +
      `${" ".repeat(20)}  +Y=${fmtV(y)} (${axisLabel(y)})\n` +
      `${" ".repeat(20)}  +Z=${fmtV(z)} (${axisLabel(z)})`
    );
  }
}
