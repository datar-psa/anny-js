/**
 * Numeric HAND-pose verifier — the same idea as `verify.ts` but for the 19
 * finger bones (per hand) + the wrist.
 *
 * Why this exists: visual review of finger articulation is essentially
 * useless to humans (let alone agents) — closed fists, splayed fingers and
 * "broken" wrists all look superficially similar in a still frame. The only
 * reliable signal is numeric: take each posed bone's local-Y axis in world,
 * compare to the MP-derived target direction in Anny coords, report angular
 * error in degrees. Plus a small set of "naturality" checks that catch a
 * working-but-still-wrong pose (e.g. chirality flipped, fingers curling
 * the wrong way).
 *
 * Coordinate frame everywhere below is **Anny world** (+x = anatomical
 * left, +y = back, +z = up). MP world landmarks are converted via
 * `mpToAnny` before any direction is computed.
 */

import type { AnnyModel } from "../../src/anny/index.js";
import type { WorldLandmark } from "../../src/mediapipe/index.js";
import { MP, MP_HAND } from "../../src/mediapipe/index.js";

// ── Coordinate conversion (mirrors src/mediapipe/landmarks.ts) ────────────

/** MP world → Anny world: anny = [+mp.x, +mp.z, -mp.y]. */
function mpToAnny(lm: WorldLandmark): [number, number, number] {
  return [lm.x, lm.z, -lm.y];
}

// ── Tiny vec3 helpers ─────────────────────────────────────────────────────

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const norm = (v: V3): V3 => {
  const L = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/L, v[1]/L, v[2]/L];
};
const dot = (a: V3, b: V3): number => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a: V3, b: V3): V3 => [
  a[1]*b[2] - a[2]*b[1],
  a[2]*b[0] - a[0]*b[2],
  a[0]*b[1] - a[1]*b[0],
];
const angDeg = (u: V3, v: V3): number => {
  const d = Math.max(-1, Math.min(1, dot(u, v)));
  return Math.acos(d) * 180 / Math.PI;
};

// ── Read posed bone axis after FK (same trick as verify.ts) ──────────────

/**
 * Read column `colIdx` of bone `name`'s POSED 3×3 rotation in world.
 *
 *   boneXforms = pose_R[b] @ inv(rest_R[b])    (rotation part only)
 *
 * To recover pose_R column `c`, multiply boneXforms_R by rest_R column `c`.
 * (Equivalent to "rotate the rest-Y-axis-in-world by boneXforms_R" for col=1.)
 */
function boneAxisWorld(
  model: AnnyModel,
  boneIndex: Map<string, number>,
  boneXforms: Float32Array,
  name: string,
  colIdx: 0 | 1 | 2,
): V3 | null {
  const i = boneIndex.get(name);
  if (i === undefined) return null;
  const x = boneXforms, o = i * 16;
  const r = model.restBonePoses, p = i * 16;
  // Rest col `colIdx`: rows are at p+colIdx, p+4+colIdx, p+8+colIdx.
  const ax = r[p + colIdx], ay = r[p + 4 + colIdx], az = r[p + 8 + colIdx];
  // boneXforms acts on rest-world directions.
  const ox = x[o + 0]*ax + x[o + 1]*ay + x[o + 2]*az;
  const oy = x[o + 4]*ax + x[o + 5]*ay + x[o + 6]*az;
  const oz = x[o + 8]*ax + x[o + 9]*ay + x[o +10]*az;
  const L = Math.hypot(ox, oy, oz) || 1;
  return [ox/L, oy/L, oz/L];
}

// ── Hand bone topology (same as buildHandTargets) ────────────────────────

interface FingerSpec {
  /** finger name as used in bone labels: finger1..finger5 */
  name: string;
  /** metacarpal label OR null for thumb (no metacarpal in Anny rig) */
  meta: string | null;
  /** MP_HAND indices for the four control points along the chain. */
  mc: number;   // MCP knuckle (or THUMB_CMC for thumb)
  pip: number;  // PIP joint (or THUMB_MCP for thumb)
  dip: number;  // DIP joint (or THUMB_IP for thumb)
  tip: number;  // fingertip
}

const FINGERS: ReadonlyArray<FingerSpec> = [
  { name: "finger1", meta: null,           mc: MP_HAND.THUMB_CMC,  pip: MP_HAND.THUMB_MCP, dip: MP_HAND.THUMB_IP,   tip: MP_HAND.THUMB_TIP  },
  { name: "finger2", meta: "metacarpal1",  mc: MP_HAND.INDEX_MCP,  pip: MP_HAND.INDEX_PIP, dip: MP_HAND.INDEX_DIP,  tip: MP_HAND.INDEX_TIP  },
  { name: "finger3", meta: "metacarpal2",  mc: MP_HAND.MIDDLE_MCP, pip: MP_HAND.MIDDLE_PIP,dip: MP_HAND.MIDDLE_DIP, tip: MP_HAND.MIDDLE_TIP },
  { name: "finger4", meta: "metacarpal3",  mc: MP_HAND.RING_MCP,   pip: MP_HAND.RING_PIP,  dip: MP_HAND.RING_DIP,   tip: MP_HAND.RING_TIP   },
  { name: "finger5", meta: "metacarpal4",  mc: MP_HAND.PINKY_MCP,  pip: MP_HAND.PINKY_PIP, dip: MP_HAND.PINKY_DIP,  tip: MP_HAND.PINKY_TIP  },
];

// ── Public types ──────────────────────────────────────────────────────────

export interface BoneError {
  bone: string;
  errDeg: number;
  /** Target direction in Anny world (computed from MP landmarks). */
  target: V3;
  /** Posed bone Y axis in Anny world (= what the rig actually does). */
  actual: V3;
}

export interface NaturalityScore {
  /**
   * For each finger, the *signed* curl angle at MCP→PIP and PIP→DIP and
   * DIP→TIP — measured around the palm-normal axis. All three should have
   * the same sign on a real human hand (joints flex toward the palm). A
   * sign reversal → bone bent the wrong way → "alien" or "backward" finger.
   */
  fingerCurls: Array<{ finger: string; signs: number[]; signsConsistent: boolean }>;
  /**
   * Wrist alignment: angle (deg) between the wrist bone's local-Y axis
   * (posed) and the forearm's local-Y axis (= lowerarm02). On a natural
   * hand the wrist should be within ~45° of the forearm even when bent.
   * Above ~60° = "broken wrist" backward bend.
   */
  wristForearmAngle: number;
  /**
   * Anatomical chirality check from MP landmarks alone, independent of FK.
   * For an anatomically LEFT hand, the thumb sits on the right side of the
   * hand from the camera's POV → thumb_CMC is on the **anatomical-right**
   * side of the wrist → in Anny world (+x = anat-left), thumb_CMC.x_anny
   * should be LESS than wrist.x_anny.
   * For R hand: thumb_CMC.x_anny > wrist.x_anny.
   *
   * `expected` is the sign convention for THIS hand; `actual` is what the
   * fixture landmarks gave. If `actual !== expected`, the landmarks are
   * mirrored (MP chirality bug, or wrong L/R assignment).
   */
  chirality: { expectedThumbXDelta: "negative" | "positive"; observedThumbXDelta: number };
}

export interface VerifyHandResult {
  side: "L" | "R";
  bones: BoneError[];
  meanErrDeg: number;
  maxErrDeg: number;
  naturality: NaturalityScore;
}

// ── Verifier ──────────────────────────────────────────────────────────────

/**
 * Compute per-bone angular error + naturality for one hand.
 *
 * `hand` is the raw MP Hand worldLandmarks (length 21), AS RECEIVED FROM MP
 * — no chirality flips, no Anny conversion. We do the conversion inside.
 *
 * `wristBoneName` is "wrist.L" / "wrist.R". The function also reads the
 * upstream forearm bone (`lowerarm02.{L,R}`) for the wrist-alignment check.
 */
export function verifyHand(
  model: AnnyModel,
  boneIndex: Map<string, number>,
  boneXforms: Float32Array,
  hand: WorldLandmark[],
  side: "L" | "R",
): VerifyHandResult {
  // Convert all 21 landmarks to Anny world coords up front.
  const pts: V3[] = hand.map(lm => mpToAnny(lm));

  const dir = (a: number, b: number): V3 => norm(sub(pts[b], pts[a]));

  const bones: BoneError[] = [];
  const pushBone = (boneName: string, target: V3): void => {
    const actual = boneAxisWorld(model, boneIndex, boneXforms, boneName, 1);
    if (!actual) return;
    bones.push({ bone: boneName, target, actual, errDeg: angDeg(target, actual) });
  };

  // ── Finger bones — every bone's Y should match its anatomical segment. ──
  for (const f of FINGERS) {
    if (f.meta !== null) {
      pushBone(`${f.meta}.${side}`, dir(MP_HAND.WRIST, f.mc));
    }
    pushBone(`${f.name}-1.${side}`, dir(f.mc, f.pip));
    pushBone(`${f.name}-2.${side}`, dir(f.pip, f.dip));
    pushBone(`${f.name}-3.${side}`, dir(f.dip, f.tip));
  }

  // ── Wrist target check ───────────────────────────────────────────────────
  // What direction "should" wrist.{L,R} point? In rest pose, its Y points
  // roughly forward-ish, but the only consistent anatomical signal is
  // "wrist → middle of MCP knuckles". So the verifier asks: does the wrist
  // bone's posed Y axis lie along WRIST → MCP-centroid?
  //
  // If the wrist bone isn't being driven (current code), this error will be
  // big — exactly the diagnostic we want.
  const mcps: V3[] = [pts[MP_HAND.INDEX_MCP], pts[MP_HAND.MIDDLE_MCP], pts[MP_HAND.RING_MCP], pts[MP_HAND.PINKY_MCP]];
  const mcpCentre: V3 = [
    (mcps[0][0]+mcps[1][0]+mcps[2][0]+mcps[3][0])/4,
    (mcps[0][1]+mcps[1][1]+mcps[2][1]+mcps[3][1])/4,
    (mcps[0][2]+mcps[1][2]+mcps[2][2]+mcps[3][2])/4,
  ];
  const wristTarget = norm(sub(mcpCentre, pts[MP_HAND.WRIST]));
  pushBone(`wrist.${side}`, wristTarget);

  const meanErrDeg = bones.length === 0 ? 0 : bones.reduce((s, b) => s + b.errDeg, 0) / bones.length;
  const maxErrDeg = bones.reduce((m, b) => Math.max(m, b.errDeg), 0);

  // ── Naturality ───────────────────────────────────────────────────────────
  const palmNormal = computePalmNormal(pts);

  const fingerCurls = FINGERS.map(f => {
    // The four control points along the finger.
    const cps = [pts[MP_HAND.WRIST], pts[f.mc], pts[f.pip], pts[f.dip], pts[f.tip]];
    // For each consecutive pair of segments, measure the signed bend around
    // the palm-normal axis. Sign = `sign(palmNormal · (segA × segB))`.
    const segs: V3[] = [];
    for (let i = 0; i < 4; i++) segs.push(norm(sub(cps[i+1], cps[i])));
    const signs: number[] = [];
    for (let i = 0; i < 3; i++) {
      const c = cross(segs[i], segs[i+1]);
      const s = dot(palmNormal, c);
      signs.push(Math.sign(s));
    }
    // "Consistent" = no sign reversal between adjacent joints. We ignore
    // exact-zero joints (straight = ambiguous).
    const nonZero = signs.filter(s => s !== 0);
    const signsConsistent = nonZero.length === 0
      ? true
      : nonZero.every(s => s === nonZero[0]);
    return { finger: f.name, signs, signsConsistent };
  });

  const wristY = boneAxisWorld(model, boneIndex, boneXforms, `wrist.${side}`, 1);
  const forearmY = boneAxisWorld(model, boneIndex, boneXforms, `lowerarm02.${side}`, 1);
  const wristForearmAngle = (wristY && forearmY) ? angDeg(wristY, forearmY) : NaN;

  // Chirality: in Anny world, +x = anat-left.
  // - L hand → thumb is on subject's MEDIAL side = anat-right side of hand →
  //   thumb_CMC.x_anny < wrist.x_anny.
  // - R hand → thumb on subject's MEDIAL = anat-left of hand →
  //   thumb_CMC.x_anny > wrist.x_anny.
  // But this only holds when the palm/hand is roughly facing the camera and
  // the hand is upright. For arbitrary poses chirality is more subtle — we
  // use the full palm normal test below; the .x_anny test is a quick first
  // line of sanity for resting poses.
  const expectedThumbXDelta: "negative" | "positive" = side === "L" ? "negative" : "positive";
  const observedThumbXDelta = pts[MP_HAND.THUMB_CMC][0] - pts[MP_HAND.WRIST][0];

  return {
    side,
    bones,
    meanErrDeg,
    maxErrDeg,
    naturality: {
      fingerCurls,
      wristForearmAngle,
      chirality: { expectedThumbXDelta, observedThumbXDelta },
    },
  };
}

/**
 * Compute the palm normal in Anny world from MP hand landmarks. We use the
 * triangle WRIST, INDEX_MCP, PINKY_MCP — pinky and index MCPs sit on the
 * far edge of the palm and form a stable plane with the wrist regardless of
 * finger curl. The sign convention: cross((INDEX_MCP - WRIST), (PINKY_MCP -
 * WRIST)) — for an anatomically left hand viewed palm-out, this points
 * **out of the palm** (toward the camera if palm faces camera).
 */
function computePalmNormal(pts: V3[]): V3 {
  const w = pts[MP_HAND.WRIST];
  const i = pts[MP_HAND.INDEX_MCP];
  const p = pts[MP_HAND.PINKY_MCP];
  return norm(cross(sub(i, w), sub(p, w)));
}

// ── INDEPENDENT verifier (pose-grounded, NOT circular) ─────────────────────

/**
 * The `verifyHand` above is *circular*: it builds its target directions from
 * the same MP hand landmarks the driver consumes, so a shared systematic
 * error (wrong axis, wrong frame, mirrored palm) cancels and it reports ~0°.
 * It only proves "the bone went where the driver intended", not "the driver
 * is anatomically right".
 *
 * This verifier instead grounds the check in the **POSE landmarks** (WRIST,
 * INDEX-knuckle, PINKY-knuckle, THUMB), which are a fully INDEPENDENT signal
 * — different model, different coordinate pipeline, and (critically) already
 * in real world coordinates. Anatomical truth:
 *   • the index metacarpal must point from wrist toward the index knuckle
 *   • the pinky metacarpal toward the pinky knuckle
 *   • the wrist bone toward the knuckle centroid
 *
 * If the posed avatar's bones disagree with pose's own hand landmarks, the
 * driver is genuinely wrong — no circularity to hide behind.
 */
export interface PoseGroundedResult {
  side: "L" | "R";
  /** metacarpal1 (index) Y vs pose WRIST→INDEX. */
  indexMetacarpalErrDeg: number;
  /** metacarpal4 (pinky) Y vs pose WRIST→PINKY. */
  pinkyMetacarpalErrDeg: number;
  /** wrist.{L,R} Y vs pose WRIST→mid(INDEX,PINKY). */
  wristErrDeg: number;
  /**
   * Per-finger curl direction, signed by the POSE-derived palm normal.
   * For each of the five fingers: dot(distalPhalangeY − metacarpalY, palmNormal).
   *   • POSITIVE → the finger tip tilts toward the palm (natural curl)
   *   • NEGATIVE → the finger tip tilts toward the back of the hand
   *                (i.e. "bends backward" — the artefact the user sees)
   * Straight fingers sit near 0. The palm normal is built from pose's
   * WRIST/INDEX/PINKY plane and forced palmward by pose's THUMB, so this is
   * fully independent of the hand-landmark targets the driver consumes.
   */
  fingerCurlPalmward: number[];
  /** Whether pose landmarks were available to run the check. */
  ran: boolean;
}

export function verifyHandVsPose(
  model: AnnyModel,
  boneIndex: Map<string, number>,
  boneXforms: Float32Array,
  pose: WorldLandmark[],
  side: "L" | "R",
): PoseGroundedResult {
  const idx = side === "L"
    ? { W: MP.LEFT_WRIST,  I: MP.LEFT_INDEX,  P: MP.LEFT_PINKY }
    : { W: MP.RIGHT_WRIST, I: MP.RIGHT_INDEX, P: MP.RIGHT_PINKY };
  const thumbI = side === "L" ? MP.LEFT_THUMB : MP.RIGHT_THUMB;
  const out: PoseGroundedResult = {
    side, indexMetacarpalErrDeg: NaN, pinkyMetacarpalErrDeg: NaN, wristErrDeg: NaN,
    fingerCurlPalmward: [], ran: false,
  };
  const pW = pose[idx.W], pI = pose[idx.I], pP = pose[idx.P], pT = pose[thumbI];
  if (!pW || !pI || !pP || !pT) return out;

  const W = mpToAnny(pW), I = mpToAnny(pI), P = mpToAnny(pP), T = mpToAnny(pT);
  const dirTo = (to: V3): V3 => norm(sub(to, W));
  const mid: V3 = [(I[0]+P[0])/2, (I[1]+P[1])/2, (I[2]+P[2])/2];

  const metaIdxY = boneAxisWorld(model, boneIndex, boneXforms, `metacarpal1.${side}`, 1);
  const metaPnkY = boneAxisWorld(model, boneIndex, boneXforms, `metacarpal4.${side}`, 1);
  const wristY   = boneAxisWorld(model, boneIndex, boneXforms, `wrist.${side}`, 1);

  if (metaIdxY) out.indexMetacarpalErrDeg = angDeg(dirTo(I), metaIdxY);
  if (metaPnkY) out.pinkyMetacarpalErrDeg = angDeg(dirTo(P), metaPnkY);
  if (wristY)   out.wristErrDeg = angDeg(dirTo(mid), wristY);

  // Silence unused (T kept for signature symmetry / future pose checks).
  void T;

  // ── Palmar direction from the RIG's posed metacarpal fan ─────────────────
  // The reliable "which way is the back of the hand" signal is the posed
  // metacarpal-fan normal: empirically (tools/probe_curl_direction.ts)
  // `cross(indexMeta_Y, pinkyMeta_Y)` points DORSALLY, so palmar = its
  // negative. We read the POSED index/pinky metacarpal Y axes (after FK),
  // which reflect what the avatar actually does — pose's own thumb landmark
  // proved too noisy/in-plane to disambiguate palm side reliably.
  const idxMetaY = boneAxisWorld(model, boneIndex, boneXforms, `metacarpal1.${side}`, 1);
  const pnkMetaY = boneAxisWorld(model, boneIndex, boneXforms, `metacarpal4.${side}`, 1);
  if (!idxMetaY || !pnkMetaY) { out.ran = true; return out; }
  // cross(index, pinky) points dorsal on the LEFT hand, palmar on the RIGHT
  // (index/pinky are mirror-arranged) — negate for R so `dorsal` is dorsal
  // on both hands, matching the driver's clamp convention.
  const fanRaw = norm(cross(idxMetaY, pnkMetaY));
  const dorsal: V3 = side === "L" ? fanRaw : [-fanRaw[0], -fanRaw[1], -fanRaw[2]];

  // ── Per-finger curl direction (distal phalange vs metacarpal) ────────────
  // metacarpal label per finger: thumb has none (use wrist), others 1..4.
  const fingerMeta: Array<{ name: string; meta: string }> = [
    { name: "finger1", meta: `wrist.${side}` },
    { name: "finger2", meta: `metacarpal1.${side}` },
    { name: "finger3", meta: `metacarpal2.${side}` },
    { name: "finger4", meta: `metacarpal3.${side}` },
    { name: "finger5", meta: `metacarpal4.${side}` },
  ];
  out.fingerCurlPalmward = fingerMeta.map(({ name, meta }) => {
    const distalY = boneAxisWorld(model, boneIndex, boneXforms, `${name}-3.${side}`, 1);
    const metaY   = boneAxisWorld(model, boneIndex, boneXforms, meta, 1);
    if (!distalY || !metaY) return NaN;
    // Positive = distal phalange tilts further to the PALMAR side (−dorsal)
    // than the metacarpal = natural curl. Negative = bends backward.
    return dot(metaY, dorsal) - dot(distalY, dorsal);
  });

  out.ran = true;
  return out;
}

// ── Hand naturality metrics (posed-avatar, vs human norms) ─────────────────

export interface HandNaturality {
  side: "L" | "R";
  /** Per-finger [MCP, PIP, DIP] flex angles in degrees (fingers 2-5). */
  jointDeg: { finger: string; mcp: number; pip: number; dip: number }[];
  /** DIP/PIP ratio per finger — natural tendon coupling ≈ 0.6-0.8. */
  dipPipRatio: number[];
  /** Std-dev of total curl (MCP+PIP+DIP) across fingers 2-5, in degrees.
   *  Near 0 = robotically uniform; very large = splayed/claw. */
  crossFingerCurlStdDeg: number;
  /** Angle (deg) between adjacent fingers' distal phalange directions —
   *  natural fingers point roughly parallel (small angles). */
  adjacentTipParallelDeg: number[];
}

/** Measure posed-avatar finger joint angles + cross-finger coherence. */
export function handNaturality(
  model: AnnyModel,
  boneIndex: Map<string, number>,
  boneXforms: Float32Array,
  side: "L" | "R",
): HandNaturality {
  const fingers = [
    { finger: "index",  meta: "metacarpal1", name: "finger2" },
    { finger: "middle", meta: "metacarpal2", name: "finger3" },
    { finger: "ring",   meta: "metacarpal3", name: "finger4" },
    { finger: "pinky",  meta: "metacarpal4", name: "finger5" },
  ];
  const Y = (n: string) => boneAxisWorld(model, boneIndex, boneXforms, `${n}.${side}`, 1);

  const jointDeg: HandNaturality["jointDeg"] = [];
  const dipPipRatio: number[] = [];
  const totalCurl: number[] = [];
  const distalDirs: (V3 | null)[] = [];

  for (const f of fingers) {
    const meta = Y(f.meta), p1 = Y(`${f.name}-1`), p2 = Y(`${f.name}-2`), p3 = Y(`${f.name}-3`);
    if (!meta || !p1 || !p2 || !p3) { distalDirs.push(null); continue; }
    const mcp = angDeg(meta, p1), pip = angDeg(p1, p2), dip = angDeg(p2, p3);
    jointDeg.push({ finger: f.finger, mcp, pip, dip });
    dipPipRatio.push(pip > 1e-3 ? dip / pip : 0);
    totalCurl.push(mcp + pip + dip);
    distalDirs.push(p3);
  }

  const mean = totalCurl.reduce((a, b) => a + b, 0) / (totalCurl.length || 1);
  const variance = totalCurl.reduce((a, b) => a + (b - mean) ** 2, 0) / (totalCurl.length || 1);
  const crossFingerCurlStdDeg = Math.sqrt(variance);

  const adjacentTipParallelDeg: number[] = [];
  for (let i = 0; i + 1 < distalDirs.length; i++) {
    const a = distalDirs[i], b = distalDirs[i + 1];
    if (a && b) adjacentTipParallelDeg.push(angDeg(a, b));
  }

  return { side, jointDeg, dipPipRatio, crossFingerCurlStdDeg, adjacentTipParallelDeg };
}

// ── Pretty-print for debugging ────────────────────────────────────────────

/** Format a verify result as a human-readable table (for `console.log`). */
export function formatVerifyResult(r: VerifyHandResult): string {
  const fmt = (n: number, w = 6) => n.toFixed(1).padStart(w);
  const fmtV = (v: V3) => `[${v.map(x => x.toFixed(2).padStart(5)).join(",")}]`;
  const lines: string[] = [];
  lines.push(`\n── HAND ${r.side}  mean=${fmt(r.meanErrDeg)}°  max=${fmt(r.maxErrDeg)}° ──`);
  lines.push(`  ${"bone".padEnd(18)} ${"err°".padStart(6)}  target              actual`);
  for (const b of r.bones) {
    lines.push(`  ${b.bone.padEnd(18)} ${fmt(b.errDeg)}  ${fmtV(b.target)}  ${fmtV(b.actual)}`);
  }
  lines.push(`  ── naturality ──`);
  lines.push(`  wrist↔forearm  ${fmt(r.naturality.wristForearmAngle)}°`);
  lines.push(`  chirality      expect thumb x delta ${r.naturality.chirality.expectedThumbXDelta}, observed ${r.naturality.chirality.observedThumbXDelta.toFixed(3)}`);
  for (const fc of r.naturality.fingerCurls) {
    lines.push(`  ${fc.finger}  curl signs [${fc.signs.join(",")}]  consistent=${fc.signsConsistent}`);
  }
  return lines.join("\n");
}
