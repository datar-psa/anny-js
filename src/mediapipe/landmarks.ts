/**
 * MediaPipe Pose + Hand → Anny bone delta rotations.
 *
 * Drives the full body chain (spine, neck, head, arms, legs, feet) plus
 * optional hands (38 finger bones) from MediaPipe Tasks Vision output.
 *
 * Design notes
 * ────────────
 * • Uses **worldLandmarks** (3D metric, hip-centered) rather than image-space
 *   landmarks. MediaPipe's image-space `z` is noisy and depth-only; world
 *   landmarks are properly 3D and don't fight aspect ratios.
 *
 * • Deltas are computed in **topological bone order** while a running
 *   accumulated rotation (`R_acc`) is maintained per bone. Child bones see
 *   the TRUE parent rotation, not just the parent's delta in isolation.
 *
 * • For each driven bone we solve:
 *
 *       pose_R[b] = R_acc[parent] @ inv(rest_R[parent]) @ rest_R[b] @ delta_R[b]
 *       (= rest_R[b] @ delta_R[b]   when parent = root)
 *
 *   For a Y-axis-along-bone target `y_world` (limbs, fingers):
 *       delta_R[b] = rotFromTo([0,1,0], inv(chain) @ y_world)
 *
 *   For a full-rotation target `world_R` (spine, head, root):
 *       delta_R[b] = inv(chain) @ world_R
 *
 * • Angle clamping for direction targets is a *MediaPipe-specific* robustness
 *   measure (lives here, not in core math), bounding bad-landmark damage to a
 *   single bone — adjacent bones aren't dragged along thanks to the per-bone
 *   chain accumulation.
 *
 * • Coordinate conversion (MP world ↔ Anny world):
 *       MP world:  +x = subject's anatomical left, +y = down,        +z = away from camera
 *       Anny:      +x = subject's anatomical left, +y = depth (back), +z = up
 *     ⇒  anny = [+mp.x, +mp.z, -mp.y]
 */

import { cross, normalize3, rodriguesToMat3 } from "../anny/math.js";
import { buildBoneIndex, identityDeltas } from "../anny/fk.js";
import type { AnnyModel, PoseDeltas } from "../anny/types.js";
import { MP, MP_HAND } from "./constants.js";
import type { PoseInput, WorldLandmark } from "./types.js";

// ── Local mat3 helpers (row-major, in-place variants for hot path) ─────────

const Y_AXIS = new Float32Array([0, 1, 0]);
const IDENTITY3 = new Float32Array([1,0,0, 0,1,0, 0,0,1]);

/** out = a @ b   (3×3 row-major). May alias `a` and `b` because we use a temp. */
function mulMat3Into(a: Float32Array, b: Float32Array, out: Float32Array): void {
  const a0=a[0],a1=a[1],a2=a[2],a3=a[3],a4=a[4],a5=a[5],a6=a[6],a7=a[7],a8=a[8];
  const b0=b[0],b1=b[1],b2=b[2],b3=b[3],b4=b[4],b5=b[5],b6=b[6],b7=b[7],b8=b[8];
  out[0] = a0*b0 + a1*b3 + a2*b6;
  out[1] = a0*b1 + a1*b4 + a2*b7;
  out[2] = a0*b2 + a1*b5 + a2*b8;
  out[3] = a3*b0 + a4*b3 + a5*b6;
  out[4] = a3*b1 + a4*b4 + a5*b7;
  out[5] = a3*b2 + a4*b5 + a5*b8;
  out[6] = a6*b0 + a7*b3 + a8*b6;
  out[7] = a6*b1 + a7*b4 + a8*b7;
  out[8] = a6*b2 + a7*b5 + a8*b8;
}

/** out = mᵀ. May alias `m`. */
function transpose3Into(m: Float32Array, out: Float32Array): void {
  const m0=m[0],m1=m[1],m2=m[2],m3=m[3],m5=m[5],m6=m[6],m7=m[7];
  out[0]=m0; out[1]=m3; out[2]=m6;
  out[3]=m1; out[4]=m[4]; out[5]=m7;
  out[6]=m2; out[7]=m5; out[8]=m[8];
}

/** out = m @ v   (3×3 row-major × column-vec). */
function mulMat3VecInto(m: Float32Array, v: Float32Array, out: Float32Array): void {
  const v0=v[0],v1=v[1],v2=v[2];
  out[0] = m[0]*v0 + m[1]*v1 + m[2]*v2;
  out[1] = m[3]*v0 + m[4]*v1 + m[5]*v2;
  out[2] = m[6]*v0 + m[7]*v1 + m[8]*v2;
}

/** Extract a bone's 3×3 rest rotation from the (B×16) restBonePoses buffer. */
function extractRestRot3(restBonePoses: Float32Array, boneIdx: number, out: Float32Array, outOffset: number): void {
  const b = boneIdx * 16;
  out[outOffset+0] = restBonePoses[b+ 0];
  out[outOffset+1] = restBonePoses[b+ 1];
  out[outOffset+2] = restBonePoses[b+ 2];
  out[outOffset+3] = restBonePoses[b+ 4];
  out[outOffset+4] = restBonePoses[b+ 5];
  out[outOffset+5] = restBonePoses[b+ 6];
  out[outOffset+6] = restBonePoses[b+ 8];
  out[outOffset+7] = restBonePoses[b+ 9];
  out[outOffset+8] = restBonePoses[b+10];
}

/**
 * MediaPipe-specific clamped rotation. Returns the smallest-angle 3×3
 * rotation mapping `from`→`to`, but caps the angle at `maxAngle` to bound
 * the damage a single bad landmark can do.
 */
function clampedRotFromTo(from: Float32Array, to: Float32Array, maxAngle: number): Float32Array {
  const ax = from[1]*to[2] - from[2]*to[1];
  const ay = from[2]*to[0] - from[0]*to[2];
  const az = from[0]*to[1] - from[1]*to[0];
  const sinA = Math.hypot(ax, ay, az);
  const cosA = from[0]*to[0] + from[1]*to[1] + from[2]*to[2];

  if (sinA < 1e-8) {
    if (cosA > 0) return new Float32Array(IDENTITY3);
    const perp = Math.abs(from[0]) < 0.9
      ? new Float32Array([1, 0, 0])
      : new Float32Array([0, 1, 0]);
    return rodriguesToMat3(normalize3(cross(from, perp)), Math.min(Math.PI, maxAngle));
  }
  const inv = 1 / sinA;
  const axis = new Float32Array([ax*inv, ay*inv, az*inv]);
  return rodriguesToMat3(axis, Math.min(Math.atan2(sinA, cosA), maxAngle));
}

// ── MediaPipe → Anny world coordinate conversion ───────────────────────────

/**
 * Convert a MediaPipe world landmark to Anny world coordinates.
 *
 *   MP world:  +x = anatomical left, +y = down, +z = away from camera (subject's back)
 *   Anny:      +x = anatomical left, +y = depth (back), +z = up
 *
 * `mirrorX` flips x after conversion — only set true if upstream already
 * mirrored the landmarks. Default false (standard MediaPipe output).
 */
export function mpToAnny(lm: WorldLandmark, mirrorX: boolean, out: Float32Array): void {
  out[0] = (mirrorX ? -1 : +1) * lm.x;
  out[1] = lm.z;
  out[2] = -lm.y;
}

// ── Bone target builders ───────────────────────────────────────────────────

/**
 * A target orientation for one bone, used by the topological FK below.
 * `dir` aligns the bone's local Y axis to `worldY` and leaves twist
 * unconstrained. `rot` pins the bone's full world rotation (used for spine
 * to preserve torso twist around vertical).
 */
export type BoneTarget =
  | { kind: "dir"; worldY: Float32Array }
  | { kind: "rot"; worldR: Float32Array };

interface TargetCtx {
  pose: WorldLandmark[];
  mirrorX: boolean;
  visMin: number;
  /** Reusable scratch — `mp(i, out)` writes into out. */
  scratch: Float32Array[];
}

/** Get a Pose landmark in Anny coords. Uses ctx.scratch[i] — do not store across calls. */
function poseAt(ctx: TargetCtx, i: number): Float32Array {
  mpToAnny(ctx.pose[i], ctx.mirrorX, ctx.scratch[i]);
  return ctx.scratch[i];
}

function visible(lm: WorldLandmark | undefined, threshold: number): boolean {
  return lm !== undefined && (lm.visibility ?? 1) >= threshold;
}

function pairVisible(ctx: TargetCtx, a: number, b: number): boolean {
  return visible(ctx.pose[a], ctx.visMin) && visible(ctx.pose[b], ctx.visMin);
}

function normSub(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array([a[0]-b[0], a[1]-b[1], a[2]-b[2]]);
  return normalize3(out);
}

function midpoint(a: Float32Array, b: Float32Array): Float32Array {
  return new Float32Array([(a[0]+b[0])*0.5, (a[1]+b[1])*0.5, (a[2]+b[2])*0.5]);
}

/**
 * Drive the `root` bone from the dancer's hip frame, rotating the WHOLE rig
 * (pelvis + legs + spine chain) to match the subject's body orientation.
 *
 * Why this is needed: driving only spine03 rotates the upper torso but leaves
 * pelvis.L/R (parents of the leg chains) at rest, so a sideways-facing dancer
 * gets a 90° kink at the lumbar with hips still facing forward.
 *
 * Why this is subtle: `root` has its OWN rest-frame convention, different
 * from spine03's. Inspecting `restBonePoses["root"]` (Anny v1):
 *   local_X → world (+1, 0, 0)             — anatomical-left
 *   local_Y → world (0, -0.93, +0.37)      — forward (slightly up)
 *   local_Z → world (0, -0.37, -0.93)      — down (slightly forward)
 *
 * Correct math: we want root's world rotation to equal `rest_R[root]` when
 * the dancer is in the canonical rest orientation, and to rotate together
 * with the dancer's hip frame otherwise:
 *
 *   worldR = dancer_anatomical_frame · inv(rest_anatomical_frame) · rest_R[root]
 */
function buildRootTarget(
  ctx: TargetCtx,
  out: Map<string, BoneTarget>,
  model: AnnyModel,
  boneIndex: Map<string, number>,
): void {
  if (!pairVisible(ctx, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER) ||
      !pairVisible(ctx, MP.LEFT_HIP, MP.RIGHT_HIP)) return;

  const rootIdx = boneIndex.get("root");
  if (rootIdx === undefined) return;

  const ls = new Float32Array(poseAt(ctx, MP.LEFT_SHOULDER));
  const rs = new Float32Array(poseAt(ctx, MP.RIGHT_SHOULDER));
  const lh = new Float32Array(poseAt(ctx, MP.LEFT_HIP));
  const rh = new Float32Array(poseAt(ctx, MP.RIGHT_HIP));

  const shdMid = midpoint(ls, rs);
  const hipMid = midpoint(lh, rh);
  const yUp = normSub(shdMid, hipMid);
  const xRaw = normSub(lh, rh);
  const dxy = xRaw[0]*yUp[0] + xRaw[1]*yUp[1] + xRaw[2]*yUp[2];
  const xL = normalize3(new Float32Array([
    xRaw[0] - dxy*yUp[0], xRaw[1] - dxy*yUp[1], xRaw[2] - dxy*yUp[2],
  ]));
  const zF = normalize3(cross(xL, yUp));

  const D = new Float32Array(9);
  D[0]=xL[0]; D[1]=yUp[0]; D[2]=zF[0];
  D[3]=xL[1]; D[4]=yUp[1]; D[5]=zF[1];
  D[6]=xL[2]; D[7]=yUp[2]; D[8]=zF[2];

  // inv(rest_anatomical_frame): for Anny rest, anatomical-left = world +x,
  // up = world +z, out-of-chest = world -y. Inverse (= transpose) is constant.
  const INV_REST_ANAT = new Float32Array([
    1, 0, 0,
    0, 0, 1,
    0, -1, 0,
  ]);

  const restRroot = new Float32Array(9);
  extractRestRot3(model.restBonePoses, rootIdx, restRroot, 0);

  const tmp = new Float32Array(9);
  mulMat3Into(D, INV_REST_ANAT, tmp);
  const worldR = new Float32Array(9);
  mulMat3Into(tmp, restRroot, worldR);

  out.set("root", { kind: "rot", worldR });
}

function buildBodyTargets(ctx: TargetCtx, out: Map<string, BoneTarget>): void {
  // ── spine03: full rotation frame (torso bend + twist around vertical) ──
  if (pairVisible(ctx, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER) &&
      pairVisible(ctx, MP.LEFT_HIP, MP.RIGHT_HIP)) {
    const ls = new Float32Array(poseAt(ctx, MP.LEFT_SHOULDER));
    const rs = new Float32Array(poseAt(ctx, MP.RIGHT_SHOULDER));
    const lh = new Float32Array(poseAt(ctx, MP.LEFT_HIP));
    const rh = new Float32Array(poseAt(ctx, MP.RIGHT_HIP));

    const shdMid = midpoint(ls, rs);
    const hipMid = midpoint(lh, rh);
    const y = normSub(shdMid, hipMid);
    const xRaw = normSub(ls, rs);
    const dxy = xRaw[0]*y[0] + xRaw[1]*y[1] + xRaw[2]*y[2];
    const x = normalize3(new Float32Array([
      xRaw[0] - dxy*y[0], xRaw[1] - dxy*y[1], xRaw[2] - dxy*y[2],
    ]));
    const z = normalize3(cross(x, y));
    const worldR = new Float32Array([
      x[0], y[0], z[0],
      x[1], y[1], z[1],
      x[2], y[2], z[2],
    ]);
    out.set("spine03", { kind: "rot", worldR });
  }

  // ── neck01: shoulder-mid → ear-mid ──
  if (pairVisible(ctx, MP.LEFT_EAR, MP.RIGHT_EAR) &&
      pairVisible(ctx, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER)) {
    const le = new Float32Array(poseAt(ctx, MP.LEFT_EAR));
    const re = new Float32Array(poseAt(ctx, MP.RIGHT_EAR));
    const ls = new Float32Array(poseAt(ctx, MP.LEFT_SHOULDER));
    const rs = new Float32Array(poseAt(ctx, MP.RIGHT_SHOULDER));
    out.set("neck01", { kind: "dir", worldY: normSub(midpoint(le, re), midpoint(ls, rs)) });
  }

  // ── head: full rotation frame from ears + eyes + mouth ──
  // The eyes (above) and mouth (below) give a clean face-vertical axis,
  // independent of skull anatomy — building Y from "ear-to-nose" produces a
  // chin-to-chest tilt because noses sit below ear level on a real face.
  if (pairVisible(ctx, MP.LEFT_EAR, MP.RIGHT_EAR) &&
      pairVisible(ctx, MP.LEFT_EYE, MP.RIGHT_EYE) &&
      pairVisible(ctx, MP.MOUTH_LEFT, MP.MOUTH_RIGHT)) {
    const le = new Float32Array(poseAt(ctx, MP.LEFT_EAR));
    const re = new Float32Array(poseAt(ctx, MP.RIGHT_EAR));
    const lEye = new Float32Array(poseAt(ctx, MP.LEFT_EYE));
    const rEye = new Float32Array(poseAt(ctx, MP.RIGHT_EYE));
    const lMth = new Float32Array(poseAt(ctx, MP.MOUTH_LEFT));
    const rMth = new Float32Array(poseAt(ctx, MP.MOUTH_RIGHT));

    const x = normSub(le, re);
    const eyeMid = midpoint(lEye, rEye);
    const mouthMid = midpoint(lMth, rMth);
    const yRaw = new Float32Array([
      eyeMid[0] - mouthMid[0],
      eyeMid[1] - mouthMid[1],
      eyeMid[2] - mouthMid[2],
    ]);
    const dyx = yRaw[0]*x[0] + yRaw[1]*x[1] + yRaw[2]*x[2];
    const y = normalize3(new Float32Array([
      yRaw[0] - dyx*x[0], yRaw[1] - dyx*x[1], yRaw[2] - dyx*x[2],
    ]));
    const z = normalize3(cross(x, y));
    const worldR = new Float32Array([
      x[0], y[0], z[0],
      x[1], y[1], z[1],
      x[2], y[2], z[2],
    ]);
    out.set("head", { kind: "rot", worldR });
  }

  arm(ctx, out, "L", MP.LEFT_SHOULDER, MP.LEFT_ELBOW, MP.LEFT_WRIST, MP.LEFT_INDEX);
  arm(ctx, out, "R", MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, MP.RIGHT_WRIST, MP.RIGHT_INDEX);

  leg(ctx, out, "L", MP.LEFT_HIP,  MP.LEFT_KNEE,  MP.LEFT_ANKLE,  MP.LEFT_FOOT_INDEX);
  leg(ctx, out, "R", MP.RIGHT_HIP, MP.RIGHT_KNEE, MP.RIGHT_ANKLE, MP.RIGHT_FOOT_INDEX);
}

function arm(
  ctx: TargetCtx, out: Map<string, BoneTarget>, side: "L"|"R",
  SH: number, EL: number, WR: number, IX: number,
): void {
  if (pairVisible(ctx, SH, EL)) {
    const a = new Float32Array(poseAt(ctx, SH)), b = new Float32Array(poseAt(ctx, EL));
    out.set(`upperarm01.${side}`, { kind: "dir", worldY: normSub(b, a) });
  }
  if (pairVisible(ctx, EL, WR)) {
    const a = new Float32Array(poseAt(ctx, EL)), b = new Float32Array(poseAt(ctx, WR));
    out.set(`lowerarm01.${side}`, { kind: "dir", worldY: normSub(b, a) });
  }
  if (pairVisible(ctx, WR, IX)) {
    const a = new Float32Array(poseAt(ctx, WR)), b = new Float32Array(poseAt(ctx, IX));
    out.set(`wrist.${side}`, { kind: "dir", worldY: normSub(b, a) });
  }
}

function leg(
  ctx: TargetCtx, out: Map<string, BoneTarget>, side: "L"|"R",
  HP: number, KN: number, AN: number, FI: number,
): void {
  if (pairVisible(ctx, HP, KN)) {
    const a = new Float32Array(poseAt(ctx, HP)), b = new Float32Array(poseAt(ctx, KN));
    out.set(`upperleg01.${side}`, { kind: "dir", worldY: normSub(b, a) });
  }
  if (pairVisible(ctx, KN, AN)) {
    const a = new Float32Array(poseAt(ctx, KN)), b = new Float32Array(poseAt(ctx, AN));
    out.set(`lowerleg01.${side}`, { kind: "dir", worldY: normSub(b, a) });
  }
  if (pairVisible(ctx, AN, FI)) {
    const a = new Float32Array(poseAt(ctx, AN)), b = new Float32Array(poseAt(ctx, FI));
    out.set(`foot.${side}`, { kind: "dir", worldY: normSub(b, a) });
  }
}

// ── Hand finger bone target builder ────────────────────────────────────────

/**
 * Map MP Hand landmarks (21) → finger bone targets (19 per hand).
 *
 * Anny rig:
 *   • Thumb (3 phalanges, no metacarpal): finger1-1, finger1-2, finger1-3
 *   • Index/Middle/Ring/Pinky (metacarpal + 3 phalanges):
 *       metacarpal{1,2,3,4} → finger{2,3,4,5}-{1,2,3}
 *
 * Also drives `wrist.{L,R}` from MP hand's wrist→middle-MCP, overriding the
 * coarse pose-only direction.
 */
function buildHandTargets(
  hand: WorldLandmark[],
  side: "L"|"R",
  mirrorX: boolean,
  out: Map<string, BoneTarget>,
): void {
  const pts: Float32Array[] = new Array(21);
  for (let i = 0; i < 21; i++) {
    pts[i] = new Float32Array(3);
    mpToAnny(hand[i], mirrorX, pts[i]);
  }
  const dir = (a: number, b: number) => normSub(pts[b], pts[a]);

  out.set(`wrist.${side}`, { kind: "dir", worldY: dir(MP_HAND.WRIST, MP_HAND.MIDDLE_MCP) });

  // Thumb
  out.set(`finger1-1.${side}`, { kind: "dir", worldY: dir(MP_HAND.THUMB_CMC, MP_HAND.THUMB_MCP) });
  out.set(`finger1-2.${side}`, { kind: "dir", worldY: dir(MP_HAND.THUMB_MCP, MP_HAND.THUMB_IP) });
  out.set(`finger1-3.${side}`, { kind: "dir", worldY: dir(MP_HAND.THUMB_IP, MP_HAND.THUMB_TIP) });

  // Fingers 2-5 (index, middle, ring, pinky)
  const fingers: ReadonlyArray<{ name: string; mc: number; pip: number; dip: number; tip: number; meta: string }> = [
    { name: "finger2", meta: "metacarpal1", mc: MP_HAND.INDEX_MCP,  pip: MP_HAND.INDEX_PIP,  dip: MP_HAND.INDEX_DIP,  tip: MP_HAND.INDEX_TIP  },
    { name: "finger3", meta: "metacarpal2", mc: MP_HAND.MIDDLE_MCP, pip: MP_HAND.MIDDLE_PIP, dip: MP_HAND.MIDDLE_DIP, tip: MP_HAND.MIDDLE_TIP },
    { name: "finger4", meta: "metacarpal3", mc: MP_HAND.RING_MCP,   pip: MP_HAND.RING_PIP,   dip: MP_HAND.RING_DIP,   tip: MP_HAND.RING_TIP   },
    { name: "finger5", meta: "metacarpal4", mc: MP_HAND.PINKY_MCP,  pip: MP_HAND.PINKY_PIP,  dip: MP_HAND.PINKY_DIP,  tip: MP_HAND.PINKY_TIP  },
  ];
  for (const f of fingers) {
    out.set(`${f.meta}.${side}`,   { kind: "dir", worldY: dir(MP_HAND.WRIST, f.mc) });
    out.set(`${f.name}-1.${side}`, { kind: "dir", worldY: dir(f.mc, f.pip) });
    out.set(`${f.name}-2.${side}`, { kind: "dir", worldY: dir(f.pip, f.dip) });
    out.set(`${f.name}-3.${side}`, { kind: "dir", worldY: dir(f.dip, f.tip) });
  }
}

// ── Topological delta computation (the heart of the driver) ────────────────

/**
 * Walk bones in topological order (parents before children — Anny stores
 * them in this order natively), and for each driven bone compute the local
 * delta from the accumulated parent world rotation. Maintains a per-bone
 * 3×3 `R_acc` so child bones see their true parent orientation.
 */
function computeDeltas(
  model: AnnyModel,
  targets: Map<number, BoneTarget>,
  maxAngle: number,
): PoseDeltas {
  const N = model.boneCount;
  const deltas = identityDeltas(N);

  const restR = new Float32Array(N * 9);
  for (let b = 0; b < N; b++) {
    extractRestRot3(model.restBonePoses, b, restR, b * 9);
  }

  const R_acc = new Float32Array(N * 9);
  const chain    = new Float32Array(9);
  const invChain = new Float32Array(9);
  const tmp9a    = new Float32Array(9);
  const tmp9b    = new Float32Array(9);
  const tmp3     = new Float32Array(3);

  for (let b = 0; b < N; b++) {
    const parent = model.boneParents[b];

    if (parent < 0) {
      chain.set(restR.subarray(b*9, b*9+9));
    } else {
      const restPar = restR.subarray(parent*9, parent*9+9);
      const restB   = restR.subarray(b*9, b*9+9);
      const RaccPar = R_acc.subarray(parent*9, parent*9+9);
      transpose3Into(restPar, tmp9a);
      mulMat3Into(tmp9a, restB, tmp9b);
      mulMat3Into(RaccPar, tmp9b, chain);
    }

    const target = targets.get(b);
    let delta: Float32Array | null = null;

    if (target !== undefined) {
      transpose3Into(chain, invChain);

      if (target.kind === "dir") {
        mulMat3VecInto(invChain, target.worldY, tmp3);
        normalize3(tmp3);
        delta = clampedRotFromTo(Y_AXIS, tmp3, maxAngle);
      } else {
        delta = new Float32Array(9);
        mulMat3Into(invChain, target.worldR, delta);
      }
      deltas[b] = delta;

      mulMat3Into(chain, delta, tmp9a);
      R_acc.set(tmp9a, b * 9);
    } else {
      R_acc.set(chain, b * 9);
    }
  }

  return deltas;
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Convert MediaPipe Pose + Hand worldLandmarks to Anny PoseDeltas, ready to
 * feed into `forwardKinematics()`.
 *
 * Drives the full body chain (spine, neck, head, arms, legs, feet) and, when
 * hand landmarks are supplied, all 38 finger bones plus a precise wrist
 * orientation.
 *
 * @param input      Landmark inputs + options.
 * @param model      Loaded AnnyModel.
 * @param boneIndex  Optional bone label→index map. If omitted, one is built
 *                   on demand (~163 Map writes — cheap, but cache it across
 *                   frames if you're calling this 30+ fps).
 */
export function landmarksToPoseDeltas(
  input: PoseInput,
  model: AnnyModel,
  boneIndex: Map<string, number> = buildBoneIndex(model),
): PoseDeltas {
  const {
    pose,
    leftHand,
    rightHand,
    mirrorX = false,
    visibilityMin = 0.5,
    maxAngleRad = 2.5,
  } = input;

  const ctx: TargetCtx = {
    pose,
    mirrorX,
    visMin: visibilityMin,
    scratch: Array.from({ length: pose.length }, () => new Float32Array(3)),
  };

  const named = new Map<string, BoneTarget>();
  buildBodyTargets(ctx, named);
  buildRootTarget(ctx, named, model, boneIndex);
  if (leftHand)  buildHandTargets(leftHand,  "L", mirrorX, named);
  if (rightHand) buildHandTargets(rightHand, "R", mirrorX, named);

  const byIdx = new Map<number, BoneTarget>();
  for (const [name, t] of named) {
    const idx = boneIndex.get(name);
    if (idx !== undefined) byIdx.set(idx, t);
  }

  return computeDeltas(model, byIdx, maxAngleRad);
}
