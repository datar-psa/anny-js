#!/usr/bin/env bun
/**
 * Determine, empirically, what "palmar curl" vs "backward (dorsal) bend"
 * means in the Anny finger rig — independent of MediaPipe entirely.
 *
 * Method: pose the middle finger of the left hand with a known palmar curl
 * (rotate each phalange around its joint hinge toward the palm) and a known
 * dorsal bend, run FK, and report each phalange's posed-Y axis. This tells
 * us which world direction the fingertip travels when curling naturally.
 *
 * Run: `bun tools/probe_curl_direction.ts`
 */

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseAnnyModel, buildBoneIndex, allocBoneTransforms, forwardKinematics,
  identityDeltas, type AnnyManifest,
} from "../src/anny/index.js";
import { rodriguesToMat3 } from "../src/anny/math.js";

const here = dirname(fileURLToPath(import.meta.url));
const ASSETS = resolve(here, "../assets");
const manifest: AnnyManifest = JSON.parse(await Bun.file(resolve(ASSETS, "anny_model.json")).text());
const buf = await Bun.file(resolve(ASSETS, "anny_model.bin")).arrayBuffer();
const model = parseAnnyModel(manifest, buf);
const boneIndex = buildBoneIndex(model);

const fmt = (n: number) => n.toFixed(3).padStart(7);
const fmtV = (v: number[]) => `[${v.map(fmt).join(", ")}]`;

function restCol(b: number, c: 0|1|2): [number,number,number] {
  const o = b*16; return [model.restBonePoses[o+c], model.restBonePoses[o+4+c], model.restBonePoses[o+8+c]];
}
function restT(b: number): [number,number,number] {
  const o = b*16; return [model.restBonePoses[o+3], model.restBonePoses[o+7], model.restBonePoses[o+11]];
}
function posedY(xf: Float32Array, b: number): [number,number,number] {
  const ry = restCol(b, 1), o = b*16;
  const x = xf[o]*ry[0]+xf[o+1]*ry[1]+xf[o+2]*ry[2];
  const y = xf[o+4]*ry[0]+xf[o+5]*ry[1]+xf[o+6]*ry[2];
  const z = xf[o+8]*ry[0]+xf[o+9]*ry[1]+xf[o+10]*ry[2];
  const L = Math.hypot(x,y,z)||1; return [x/L,y/L,z/L];
}
function posedPos(xf: Float32Array, b: number): [number,number,number] {
  const t = restT(b), o = b*16;
  return [
    xf[o]*t[0]+xf[o+1]*t[1]+xf[o+2]*t[2]+xf[o+3],
    xf[o+4]*t[0]+xf[o+5]*t[1]+xf[o+6]*t[2]+xf[o+7],
    xf[o+8]*t[0]+xf[o+9]*t[1]+xf[o+10]*t[2]+xf[o+11],
  ];
}

// Rest palm normal of the left hand, from the metacarpal fan.
const m2 = restCol(boneIndex.get("metacarpal2.L")!, 1);  // index meta Y
const m4 = restCol(boneIndex.get("metacarpal4.L")!, 1);  // pinky meta Y
const palmN = ((): [number,number,number] => {
  const c: [number,number,number] = [
    m2[1]*m4[2]-m2[2]*m4[1], m2[2]*m4[0]-m2[0]*m4[2], m2[0]*m4[1]-m2[1]*m4[0],
  ];
  const L = Math.hypot(...c)||1; return [c[0]/L,c[1]/L,c[2]/L];
})();
console.log("Anny world: +x=anat-left, +y=back, +z=up");
console.log(`Rest palm-fan normal (cross of index/pinky meta Y) = ${fmtV(palmN)}\n`);

// The middle finger chain on the left hand and each bone's rest X axis (the
// joint hinge). Flexion rotates the phalange around its X axis.
const chain = ["metacarpal2.L", "finger3-1.L", "finger3-2.L", "finger3-3.L"];
for (const name of chain) {
  const b = boneIndex.get(name)!;
  console.log(`${name.padEnd(14)} restY=${fmtV(restCol(b,1))}  restX(hinge)=${fmtV(restCol(b,0))}`);
}

// Apply a +30° flex around each phalange's local X and see where the tip goes.
function poseAndReport(label: string, angle: number) {
  const deltas = identityDeltas(model.boneCount);
  for (const name of ["finger3-1.L","finger3-2.L","finger3-3.L"]) {
    deltas[boneIndex.get(name)!] = rodriguesToMat3(new Float32Array([1,0,0]), angle); // local-X rotation
  }
  const xf = allocBoneTransforms(model.boneCount);
  forwardKinematics(model, deltas, xf);
  const tip = posedPos(xf, boneIndex.get("finger3-3.L")!);
  const restTip = (() => { const id = identityDeltas(model.boneCount); const x = allocBoneTransforms(model.boneCount); forwardKinematics(model, id, x); return posedPos(x, boneIndex.get("finger3-3.L")!); })();
  const move: [number,number,number] = [tip[0]-restTip[0], tip[1]-restTip[1], tip[2]-restTip[2]];
  const Lm = Math.hypot(...move)||1;
  const moveN = [move[0]/Lm, move[1]/Lm, move[2]/Lm];
  const dotPalm = moveN[0]*palmN[0]+moveN[1]*palmN[1]+moveN[2]*palmN[2];
  console.log(`\n${label} (local-X ${(angle*180/Math.PI).toFixed(0)}°): tip moves ${fmtV(moveN)}`);
  console.log(`   tip-move · palmNormal = ${dotPalm.toFixed(3)}  → ${dotPalm < 0 ? "toward PALM (palmar/natural)" : "toward BACK (dorsal/backward)"}`);
}
poseAndReport("flex +X", +0.5);
poseAndReport("flex -X", -0.5);
