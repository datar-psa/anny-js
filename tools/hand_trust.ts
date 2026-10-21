#!/usr/bin/env bun
/**
 * How far to trust MediaPipe's hand reading, as a function of hand size.
 *
 * From .cache/hand-dataset.json (tools/hand_dataset.ts: ground-truth hands
 * rendered at several sizes, read back by MediaPipe) fit
 *
 *   • the natural-hand prior: mean + covariance of finger angles (relative
 *     to the rig's rest) over the ground-truth hands — the finger synergies
 *     live in its off-diagonal terms;
 *   • MediaPipe's finger noise: one correlation shape × a variance curve
 *     s²(px) = a + b / px in the palm length px;
 *   • MediaPipe's palm-orientation noise per camera axis (roll in the image
 *     plane is read far better than tilt toward / away from the camera),
 *     separately for hands whose handedness label agrees with the side they
 *     were assigned to and hands where it doesn't;
 *
 * then score the fusion with leave-one-gesture-family-out cross-validation
 * (the family being scored never informs the prior or the noise model).
 *
 *   bun tools/hand_trust.ts            report only
 *   bun tools/hand_trust.ts --emit     also write src/mediapipe/hand-prior.ts
 */

import { parseArgs } from "node:util";

import { handPointsAnny, palmFrame } from "../src/mediapipe/hand-geometry.js";
import { gaussianUpdate, type Gaussian } from "../src/mediapipe/fusion.js";
import { handLandmarks3D, matchHandChirality } from "../src/mediapipe/hands.js";
import { mT, mmul, quatFromMat3, type Vec3 } from "../src/mediapipe/geom.js";
import { rigInfo } from "../src/mediapipe/rig.js";
import { buildBoneIndex } from "../src/anny/index.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { FAMILY_FREQUENCY, gtHandDeltas } from "../tests/naturalness/hand-gt.js";
import { FINGER_DIMS, handState, STATE_DIMS } from "../tests/naturalness/hand-state.js";
import { buildPhysicalRig, physicalScore } from "../tests/naturalness/physical.js";
import type { HandSample } from "./hand_dataset.js";

const { values: args } = parseArgs({
  options: {
    data: { type: "string", default: ".cache/hand-dataset.json" },
    emit: { type: "boolean", default: false },
    /**
     * Floor on the prior's abduction SD (deg). The ground-truth set spreads
     * fingers less than people do; without a floor the prior would pin
     * abduction even on large, clear hands.
     */
    "abd-floor": { type: "string", default: "8" },
  },
});

const R2D = 180 / Math.PI;
const N = STATE_DIMS;
const model = await loadFixtureModel();
const rig = rigInfo(model);
const samples: HandSample[] = await Bun.file(args.data).json();

/**
 * Ground-truth hands whose fingers pass through each other (a thumb buried
 * in the index finger, say) are no "natural hand": they inform the noise
 * model — MediaPipe's error doesn't care — but not the prior.
 */
const MAX_PRIOR_PENETRATION_MM = 1.5;
const physical = buildPhysicalRig(model), boneIndex = buildBoneIndex(model);
const penetrating = new Set(
  [...new Map(samples.map(s => [s.hand.name, s.hand])).values()]
    .filter(h => physicalScore(physical, gtHandDeltas(h, boneIndex, model.boneCount), h.side).penetrationMm > MAX_PRIOR_PENETRATION_MM)
    .map(h => h.name),
);

const handVec = (pts: Vec3[], side: "L" | "R"): number[] => handState(pts, side, rig);

/**
 * Rotation vector (deg) taking the ground-truth palm to MediaPipe's, in
 * world axes — which are the camera's (Anny x = image right, y = view
 * direction, z = image up): the error is anything but isotropic there.
 */
function palmError(gt: Vec3[], mp: Vec3[]): Vec3 {
  const q = quatFromMat3(mmul(palmFrame(mp), mT(palmFrame(gt))));
  const s = Math.hypot(q[0], q[1], q[2]);
  const angle = 2 * Math.atan2(s, Math.abs(q[3])) * R2D * Math.sign(q[3] || 1);
  return s < 1e-9 ? [0, 0, 0] : [q[0] / s * angle, q[1] / s * angle, q[2] / s * angle];
}

interface Row { family: string; key: string; size: number; px: number; agree: boolean; g: number[]; m: number[]; palm: Vec3 }

const rows: Row[] = samples.filter(s => s.mp).map(s => {
  const side = s.hand.side;
  const world = matchHandChirality(handLandmarks3D(s.mp!.image, s.W, s.H), side, s.mp!.labelSide === "L" ? "Left" : "Right");
  const mp = handPointsAnny(world, side);
  const gt = s.gtJoints as Vec3[];
  return {
    family: s.hand.family, key: s.hand.name, size: s.size, px: s.palmPx,
    agree: s.mp!.labelSide === side, g: handVec(gt, side), m: handVec(mp, side), palm: palmError(gt, mp),
  };
});
const sizes = [...new Set(rows.map(r => r.size))].sort((a, b) => a - b);
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
const rms = (xs: number[]) => Math.sqrt(mean(xs.map(x => x * x)));

// ── Fitting ────────────────────────────────────────────────────────────────

/**
 * Weighted mean + covariance of the ground-truth hands: each family weighs
 * by its everyday frequency (FAMILY_FREQUENCY), shared among its hands.
 */
function fitPrior(rs: Row[], abdFloor: number): Gaussian {
  const hands = [...new Map(rs.filter(r => !penetrating.has(r.key)).map(r => [r.key, r])).values()];
  const perFamily = new Map<string, number>();
  for (const h of hands) perFamily.set(h.family, (perFamily.get(h.family) ?? 0) + 1);
  const raw = hands.map(h => (FAMILY_FREQUENCY[h.family] ?? 1) / perFamily.get(h.family)!);
  const total = raw.reduce((a, b) => a + b, 0);
  const w = raw.map(x => x / total);
  const mu = Array.from({ length: N }, (_, j) => hands.reduce((s, h, k) => s + w[k] * h.g[j], 0));
  const cov = new Array<number>(N * N).fill(0);
  hands.forEach((h, k) => {
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) cov[i * N + j] += w[k] * (h.g[i] - mu[i]) * (h.g[j] - mu[j]);
  });
  const unbias = 1 / (1 - w.reduce((s, x) => s + x * x, 0));
  for (let k = 0; k < N * N; k++) cov[k] *= unbias;
  // Abduction floor: rescale those rows/columns so correlations are kept.
  const sd = Array.from({ length: N }, (_, i) => Math.sqrt(cov[i * N + i]));
  const target = sd.map((s, i) => (i >= 12 && i < FINGER_DIMS ? Math.max(s, abdFloor) : s));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) cov[i * N + j] *= (target[i] / sd[i]) * (target[j] / sd[j]);
  return { mean: mu, cov };
}

/** a + b/px least squares over per-size bins of mean squared error. */
function fitCurve(px: number[], v: number[]): { a: number; b: number } {
  const x = px.map(p => 1 / p), mx = mean(x), my = mean(v);
  const b = x.reduce((s, xi, i) => s + (xi - mx) * (v[i] - my), 0) / x.reduce((s, xi) => s + (xi - mx) ** 2, 0);
  return { a: Math.max(1, my - b * mx), b: Math.max(0, b) };
}

/** Noise covariance R(px)_ij = corr_ij · √(v_i(px) v_j(px)),  v_i(px) = a_i + b_i / px. */
interface HandNoise { corr: number[]; a: number[]; b: number[] }

/** Shrinkage toward the diagonal for the per-size noise matrices (few samples per bin). */
const SHRINK = 0.3;

function fitHandNoise(rs: Row[]): HandNoise {
  const corr = new Array<number>(N * N).fill(0);
  const bins = sizes.map(size => rs.filter(r => r.size === size)).filter(b => b.length > N);
  const px = bins.map(b => mean(b.map(r => r.px)));
  const vars: number[][] = [];
  for (const bin of bins) {
    const res = bin.map(r => r.m.map((x, j) => x - r.g[j]));
    const M = Array.from({ length: N * N }, (_, k) => mean(res.map(e => e[Math.floor(k / N)] * e[k % N])));
    const v = Array.from({ length: N }, (_, i) => M[i * N + i]);
    for (let k = 0; k < N * N; k++) {
      const i = Math.floor(k / N), j = k % N;
      corr[k] += (i === j ? 1 : (1 - SHRINK) * M[k] / Math.sqrt(v[i] * v[j])) / bins.length;
    }
    vars.push(v);
  }
  const curves = Array.from({ length: N }, (_, i) => fitCurve(px, vars.map(v => v[i])));
  return { corr, a: curves.map(c => c.a), b: curves.map(c => c.b) };
}

function noiseAt(n: HandNoise, px: number): number[] {
  const sd = n.a.map((a, i) => Math.sqrt(a + n.b[i] / px));
  return n.corr.map((c, k) => c * sd[Math.floor(k / N)] * sd[k % N]);
}

interface PalmNoise { a: number[]; b: number[] }

function fitPalmNoise(rs: Row[]): PalmNoise {
  const bins = sizes.map(size => rs.filter(r => r.size === size)).filter(b => b.length >= 8);
  const fits = [0, 1, 2].map(axis => fitCurve(
    bins.map(b => mean(b.map(r => r.px))),
    bins.map(b => mean(b.map(r => r.palm[axis] ** 2))),
  ));
  return { a: fits.map(f => f.a), b: fits.map(f => f.b) };
}

// ── Cross-validated report ─────────────────────────────────────────────────

function fuse(prior: Gaussian, noise: HandNoise, m: number[], px: number): number[] {
  return gaussianUpdate(prior, m, noiseAt(noise, px)).mean;
}

/** Mean angle (deg) between estimated and true thumb segment directions. */
function thumbError(x: number[], g: number[]): number {
  return mean([0, 1, 2].map(k => {
    const a = x.slice(FINGER_DIMS + 3 * k, FINGER_DIMS + 3 * k + 3), b = g.slice(FINGER_DIMS + 3 * k, FINGER_DIMS + 3 * k + 3);
    const c = a.reduce((s, v, i) => s + v * b[i], 0) / (Math.hypot(...a) * Math.hypot(...b));
    return Math.acos(Math.max(-1, Math.min(1, c))) * R2D;
  }));
}

const families = [...new Set(rows.map(r => r.family))];
const err: Record<string, Record<number, { joint: number[]; abd: number[]; curl: number[]; thumb: number[] }>> = {};
for (const method of ["mediapipe", "prior", "fused"]) {
  err[method] = {};
  for (const s of sizes) err[method][s] = { joint: [], abd: [], curl: [], thumb: [] };
}
const abdFloor = Number(args["abd-floor"]);
for (const family of families) {
  const train = rows.filter(r => r.family !== family);
  const prior = fitPrior(train, abdFloor), noise = fitHandNoise(train);
  for (const r of rows.filter(x => x.family === family)) {
    const est: Record<string, number[]> = { mediapipe: r.m, prior: prior.mean, fused: fuse(prior, noise, r.m, r.px) };
    for (const [method, x] of Object.entries(est)) {
      const e = err[method][r.size];
      for (let j = 0; j < 12; j++) e.joint.push(x[j] - r.g[j]);
      for (let j = 12; j < 16; j++) e.abd.push(x[j] - r.g[j]);
      for (let f = 0; f < 4; f++) e.curl.push(x[3 * f] + x[3 * f + 1] + x[3 * f + 2] - (r.g[3 * f] + r.g[3 * f + 1] + r.g[3 * f + 2]));
      e.thumb.push(thumbError(x, r.g));
    }
  }
}
console.log(`Finger RMS error vs ground truth (deg), leave-one-gesture-family-out over ${families.length} families`);
console.log("palm px  detected   joint: MP  prior  fused   abduction: MP  prior  fused   finger curl: MP  prior  fused   thumb (mean deg): MP  prior  fused");
for (const s of sizes) {
  const px = mean(samples.filter(x => x.size === s).map(x => x.palmPx));
  const det = rows.filter(r => r.size === s).length / samples.filter(x => x.size === s).length;
  const col = (k: "joint" | "abd" | "curl") => ["mediapipe", "prior", "fused"].map(m => rms(err[m][s][k]).toFixed(1).padStart(6)).join(" ");
  const thumb = ["mediapipe", "prior", "fused"].map(m => mean(err[m][s].thumb).toFixed(1).padStart(6)).join(" ");
  console.log(`${px.toFixed(0).padStart(7)} ${(det * 100).toFixed(0).padStart(8)}%   ${col("joint")}   ${col("abd")}       ${col("curl")}             ${thumb}`);
}

// ── Full fit ───────────────────────────────────────────────────────────────

const prior = fitPrior(rows, abdFloor);
const handNoise = fitHandNoise(rows);
const palmAgree = fitPalmNoise(rows.filter(r => r.agree));
const palmDisagree = fitPalmNoise(rows.filter(r => !r.agree));
const sdAt = (n: { a: number; b: number }, px: number) => Math.sqrt(n.a + n.b / px).toFixed(0);
console.log(`\nprior: ${new Set(rows.map(r => r.key)).size - penetrating.size} hands (${penetrating.size} excluded for interpenetration)`);
console.log(`prior SD (deg): ${prior.mean.map((_, i) => Math.sqrt(prior.cov[i * N + i]).toFixed(0)).join(" ")}`);
console.log(`noise SD (deg) at 10/30/100 px: ${[10, 30, 100].map(p => handNoise.a.map((a, i) => Math.sqrt(a + handNoise.b[i] / p).toFixed(0)).join(" ")).join("  /  ")}`);
for (const [name, n] of [["agree", palmAgree], ["disagree", palmDisagree]] as const) {
  console.log(`palm noise SD about camera [x, view, z] (${name}) at 10/30/100 px: ` +
    [10, 30, 100].map(p => [0, 1, 2].map(k => sdAt({ a: n.a[k], b: n.b[k] }, p)).join(",")).join(" / "));
}

// ── Real-footage calibration (tools/hand_tta.ts) ───────────────────────────
//
// Test-time-augmentation spread of MediaPipe's readings tracks its true error
// on renders (error/spread is stable across sizes), so the spread on real
// photos and video frames, relative to renders of the same palm size, scales
// the render-fitted noise to real footage: κ(px) = k0 · (px0 / px)^exponent.

interface TtaSpread { px: number; fingerVar: number }
interface Footage { px0: number; k0: number; exponent: number }

function fitFootage(): Footage | undefined {
  const file = Bun.file(".cache/hand-tta.json");
  if (!file.size) return undefined;
  return file.json().then((t: { photos: TtaSpread[]; clips: TtaSpread[]; synthetic: Record<string, TtaSpread[]> }) => {
    const renders = Object.values(t.synthetic).map(ss => ({ px: mean(ss.map(s => s.px)), v: mean(ss.map(s => s.fingerVar)) }))
      .sort((a, b) => a.px - b.px);
    /** Render spread at `px`, log-log interpolated. */
    const renderVar = (px: number): number => {
      const above = renders.findIndex(r => r.px > px);
      const i = above < 0 ? renders.length - 2 : Math.max(0, above - 1);
      const [a, b] = [renders[i], renders[i + 1]];
      const f = (Math.log(px) - Math.log(a.px)) / (Math.log(b.px) - Math.log(a.px));
      return Math.exp(Math.log(a.v) + f * (Math.log(b.v) - Math.log(a.v)));
    };
    const point = (ss: TtaSpread[]) => {
      const px = mean(ss.map(s => s.px));
      return { px, k: mean(ss.map(s => s.fingerVar)) / renderVar(px) };
    };
    const big = point(t.photos), small = point(t.clips);
    const exponent = Math.log(small.k / big.k) / Math.log(big.px / small.px);
    console.log(`\nreal footage: noise × ${big.k.toFixed(2)} at ${big.px.toFixed(0)} px (photos), × ${small.k.toFixed(2)} at ${small.px.toFixed(0)} px (video ROI) → κ(px) = ${big.k.toFixed(2)}·(${big.px.toFixed(0)}/px)^${exponent.toFixed(2)}`);
    return { px0: big.px, k0: big.k, exponent };
  }) as unknown as Footage;
}
const footage = await fitFootage();

if (args.emit) {
  const f = (xs: number[], d = 1) => `[${xs.map(x => Number(x.toFixed(d))).join(", ")}]`;
  const axes = (n: PalmNoise) => `{ a: ${f(n.a)}, b: ${f(n.b)} }`;
  const out = `// Generated by \`bun tools/hand_trust.ts --emit\` from tools/hand_dataset.ts output. Do not edit.
//
// ${rows.length} MediaPipe readings of ${new Set(rows.map(r => r.key)).size} ground-truth hands (${families.length} gesture
// families + random natural hands) rendered at palm lengths ${Math.round(Math.min(...rows.map(r => r.px)))}–${Math.round(Math.max(...rows.map(r => r.px)))} px;
// the prior uses the ${new Set(rows.map(r => r.key)).size - penetrating.size} whose fingers don't interpenetrate.
// Angles in degrees, relative to the rig's rest; covariances in deg², row-major.

/**
 * Natural hands. State (25): [index..pinky × (MCP, PIP, DIP) flexion],
 * [index..pinky] abduction, then the thumb's CMC→MCP, MCP→IP and IP→TIP
 * directions in the palm frame (unit vectors × 180/π).
 */
export const HAND_PRIOR = {
  mean: ${f(prior.mean)},
  cov: ${f(prior.cov)},
};

/** MediaPipe's noise on that state: R_ij = corr_ij · √(v_i v_j), v_i = a_i + b_i / palmPx. */
export const HAND_NOISE = {
  corr: ${f(handNoise.corr, 3)},
  a: ${f(handNoise.a)},
  b: ${f(handNoise.b, 0)},
};

/**
 * MediaPipe palm-orientation noise about the world axes, which are the
 * camera's — [x: image right, y: view direction, z: image up] — so y is roll
 * in the image plane, read far better than the two tilts out of it.
 * Variance = a + b / palmPx, for a handedness label that agrees / disagrees
 * with the side the hand drives.
 */
export const PALM_NOISE = {
  agree: ${axes(palmAgree)},
  disagree: ${axes(palmDisagree)},
};

/**
 * The noise above is MediaPipe on renders. On real footage it differs —
 * smaller on crisp photos, larger on blurry, compressed video — measured by
 * test-time-augmentation spread (tools/hand_tta.ts): variance × κ(palmPx),
 * κ = k0 · (px0 / palmPx)^exponent.
 */
export const REAL_FOOTAGE = ${footage ? `{ px0: ${footage.px0.toFixed(0)}, k0: ${footage.k0.toFixed(2)}, exponent: ${footage.exponent.toFixed(2)} }` : "{ px0: 100, k0: 1, exponent: 0 }"};
`;
  await Bun.write("src/mediapipe/hand-prior.ts", out);
  console.log("\nwrote src/mediapipe/hand-prior.ts");
}
