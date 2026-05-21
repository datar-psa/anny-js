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
 *   This fixes the grand-parent error that affected wrist/finger bones in v0.
 *
 * • For each driven bone we solve:
 *
 *       pose_R[b] = R_acc[parent] @ inv(rest_R[parent]) @ rest_R[b] @ delta_R[b]
 *       (= rest_R[b] @ delta_R[b]   when parent = root)
 *
 *   For a Y-axis-along-bone target `y_world` (limbs, fingers):
 *       delta_R[b] = rotFromTo([0,1,0], inv(chain) @ y_world)
 *       where chain = R_acc[parent] @ inv(rest_R[parent]) @ rest_R[b].
 *
 *   For a full-rotation target `world_R` (spine — preserves torso yaw):
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

import { cross, normalize3, rodriguesToMat3 } from "./math.js";
import { identityDeltas } from "./fk.js";
import type { AnnyModel, PoseDeltas } from "./types.js";

// ── MediaPipe landmark types & indices ─────────────────────────────────────

/** A MediaPipe image-space landmark (normalized 0..1 with depth). */
export interface Landmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

/** A MediaPipe world-space landmark (metric, hip-centered). */
export interface WorldLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/** MediaPipe Pose Landmarker — 33 landmark indices. */
export const MP = {
  NOSE: 0,
  LEFT_EYE_INNER: 1, LEFT_EYE: 2, LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4, RIGHT_EYE: 5, RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7, RIGHT_EAR: 8,
  MOUTH_LEFT: 9, MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_PINKY: 17, RIGHT_PINKY: 18,
  LEFT_INDEX: 19, RIGHT_INDEX: 20,
  LEFT_THUMB: 21, RIGHT_THUMB: 22,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
  LEFT_HEEL: 29, RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31, RIGHT_FOOT_INDEX: 32,
} as const;

/** MediaPipe Hand Landmarker — 21 landmark indices (per hand). */
export const MP_HAND = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
} as const;

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
    // antiparallel — 180° around any axis perpendicular to `from`
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
 * The MP worldLandmarks z follows the same convention as image landmarks
 * (smaller z = closer to camera, i.e. positive z = away from camera =
 * subject's back direction in world). So Anny.y = +mp.z aligns naturally.
 *
 * `mirrorX` flips x after conversion — only set true if upstream already
 * mirrored the landmarks. Default false (standard MediaPipe output).
 */
function mpToAnny(lm: WorldLandmark, mirrorX: boolean, out: Float32Array): void {
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
type BoneTarget =
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
 *   local_X → world (+1, 0, 0)             — anatomical-left ✓
 *   local_Y → world (0, -0.93, +0.37)      — forward (slightly up)
 *   local_Z → world (0, -0.37, -0.93)      — down (slightly forward)
 *
 * i.e. for the root bone, local-Y is "forward of body" (not "up along spine"
 * like the spine bones), and there's a built-in ~22° pitch in the rest pose.
 * If we naïvely set worldR = (x_left, y_up, z_chest) — the spine convention —
 * the body gets a 90° pitch backward at rest, manifesting as Anny crumpled
 * forward. This was the visible regression.
 *
 * Correct math: we want root's world rotation to equal `rest_R[root]` when
 * the dancer is in the canonical rest orientation, and to rotate together
 * with the dancer's hip frame otherwise:
 *
 *   worldR = dancer_anatomical_frame · inv(rest_anatomical_frame) · rest_R[root]
 *
 * where `dancer_anatomical_frame` and `rest_anatomical_frame` are 3×3
 * matrices with columns [anatomical-left, up-along-spine, out-of-chest], in
 * Anny world coords. The middle factor is a constant: the rest anatomical
 * frame is fixed by Anny world convention ((+x, +z, -y) for those three
 * axes), so its inverse is a known matrix.
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

  // Hip frame (NOT shoulder frame) — this is what makes the root track hips
  // independently from any upper-body twist that spine03 will handle.
  const shdMid = midpoint(ls, rs);
  const hipMid = midpoint(lh, rh);
  const yUp = normSub(shdMid, hipMid);   // up along spine
  const xRaw = normSub(lh, rh);          // anatomical-left across HIPS
  const dxy = xRaw[0]*yUp[0] + xRaw[1]*yUp[1] + xRaw[2]*yUp[2];
  const xL = normalize3(new Float32Array([
    xRaw[0] - dxy*yUp[0], xRaw[1] - dxy*yUp[1], xRaw[2] - dxy*yUp[2],
  ]));
  const zF = normalize3(cross(xL, yUp));  // out-of-chest

  // dancer_anatomical_frame (row-major 3×3, columns = [xL, yUp, zF]).
  // Indices labelled [r,c] for clarity.
  const D = new Float32Array(9);
  D[0]=xL[0]; D[1]=yUp[0]; D[2]=zF[0];
  D[3]=xL[1]; D[4]=yUp[1]; D[5]=zF[1];
  D[6]=xL[2]; D[7]=yUp[2]; D[8]=zF[2];

  // inv(rest_anatomical_frame): Anny rest pose has anatomical-left = world +x,
  // up = world +z, out-of-chest = world -y. So rest_anatomical_frame as cols
  // is [(1,0,0), (0,0,1), (0,-1,0)]; its inverse (transpose, since orthogonal)
  // is row-major [1,0,0; 0,0,1; 0,-1,0].
  const INV_REST_ANAT = new Float32Array([
    1, 0, 0,
    0, 0, 1,
    0, -1, 0,
  ]);

  // rest_R[root]: pull the 3×3 from model.restBonePoses[root].
  const restRroot = new Float32Array(9);
  extractRestRot3(model.restBonePoses, rootIdx, restRroot, 0);

  // worldR = D · INV_REST_ANAT · restRroot
  const tmp = new Float32Array(9);
  mul3(D, INV_REST_ANAT, tmp);
  const worldR = new Float32Array(9);
  mul3(tmp, restRroot, worldR);

  out.set("root", { kind: "rot", worldR });
}

/** Multiply two row-major 3×3 matrices: out = A · B. */
function mul3(A: Float32Array, B: Float32Array, out: Float32Array): void {
  const a0=A[0],a1=A[1],a2=A[2],a3=A[3],a4=A[4],a5=A[5],a6=A[6],a7=A[7],a8=A[8];
  const b0=B[0],b1=B[1],b2=B[2],b3=B[3],b4=B[4],b5=B[5],b6=B[6],b7=B[7],b8=B[8];
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

function buildBodyTargets(ctx: TargetCtx, out: Map<string, BoneTarget>): void {
  // ── spine03: full rotation frame (torso bend + twist around vertical) ──
  // Driving the topmost spine bone (spine01) creates a hinge just below the
  // shoulders. spine03 sits roughly at the lumbar/lower-thoracic transition —
  // a much more natural pivot. Everything above (spine02, spine01, neck,
  // head, shoulders, arms) inherits the rotation through the chain, so the
  // upper torso stays a rigid block that rotates with the shoulders.
  if (pairVisible(ctx, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER) &&
      pairVisible(ctx, MP.LEFT_HIP, MP.RIGHT_HIP)) {
    const ls = new Float32Array(poseAt(ctx, MP.LEFT_SHOULDER));
    const rs = new Float32Array(poseAt(ctx, MP.RIGHT_SHOULDER));
    const lh = new Float32Array(poseAt(ctx, MP.LEFT_HIP));
    const rh = new Float32Array(poseAt(ctx, MP.RIGHT_HIP));

    const shdMid = midpoint(ls, rs);
    const hipMid = midpoint(lh, rh);
    // Y along the spine (hip → shoulder midpoint)
    const y = normSub(shdMid, hipMid);
    // X across shoulders (anatomical-left direction)
    const xRaw = normSub(ls, rs);
    // Orthogonalise X against Y → pure left-right; renormalise
    const dxy = xRaw[0]*y[0] + xRaw[1]*y[1] + xRaw[2]*y[2];
    const x = normalize3(new Float32Array([
      xRaw[0] - dxy*y[0], xRaw[1] - dxy*y[1], xRaw[2] - dxy*y[2],
    ]));
    // Z = X × Y completes the right-handed frame (points "out of chest")
    const z = normalize3(cross(x, y));
    // Row-major 3×3 where columns are the world basis vectors that local
    // [1,0,0], [0,1,0], [0,0,1] map to. Matches Anny's spine rest convention:
    // local X = anatomical-left, local Y = along bone, local Z = out of chest.
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
  // Anny's `head` bone has local X = anatomical-left, Y = up through skull,
  // Z = forward out of face.
  //
  // Building Y from "ear-to-nose" (the obvious-sounding choice) is wrong:
  // the nose sits BELOW ear level on a real face, so that vector points
  // down-and-forward and the head ends up tilted forward (chin to chest)
  // even when the subject looks straight ahead.
  //
  // The eyes (above) and mouth (below) give a clean face-vertical axis,
  // independent of skull anatomy. Eye→mouth points down in face frame, so
  // Y_face = eyeMid - mouthMid points up. Then Z = cross(X, Y) gives the
  // face-forward direction, free of the nose's vertical offset.
  if (pairVisible(ctx, MP.LEFT_EAR, MP.RIGHT_EAR) &&
      pairVisible(ctx, MP.LEFT_EYE, MP.RIGHT_EYE) &&
      pairVisible(ctx, MP.MOUTH_LEFT, MP.MOUTH_RIGHT)) {
    const le = new Float32Array(poseAt(ctx, MP.LEFT_EAR));
    const re = new Float32Array(poseAt(ctx, MP.RIGHT_EAR));
    const lEye = new Float32Array(poseAt(ctx, MP.LEFT_EYE));
    const rEye = new Float32Array(poseAt(ctx, MP.RIGHT_EYE));
    const lMth = new Float32Array(poseAt(ctx, MP.MOUTH_LEFT));
    const rMth = new Float32Array(poseAt(ctx, MP.MOUTH_RIGHT));

    // X = anatomical-left across the head
    const x = normSub(le, re);
    // yRaw = up through face (mouth → eyes)
    const eyeMid = midpoint(lEye, rEye);
    const mouthMid = midpoint(lMth, rMth);
    const yRaw = new Float32Array([
      eyeMid[0] - mouthMid[0],
      eyeMid[1] - mouthMid[1],
      eyeMid[2] - mouthMid[2],
    ]);
    // Orthogonalise Y against X (head-roll is encoded in X already)
    const dyx = yRaw[0]*x[0] + yRaw[1]*x[1] + yRaw[2]*x[2];
    const y = normalize3(new Float32Array([
      yRaw[0] - dyx*x[0], yRaw[1] - dyx*x[1], yRaw[2] - dyx*x[2],
    ]));
    // Z = X × Y completes the right-handed frame; encodes chin-up/down + yaw
    const z = normalize3(cross(x, y));
    const worldR = new Float32Array([
      x[0], y[0], z[0],
      x[1], y[1], z[1],
      x[2], y[2], z[2],
    ]);
    out.set("head", { kind: "rot", worldR });
  }

  // ── Arms ──
  arm(ctx, out, "L", MP.LEFT_SHOULDER, MP.LEFT_ELBOW, MP.LEFT_WRIST, MP.LEFT_INDEX);
  arm(ctx, out, "R", MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, MP.RIGHT_WRIST, MP.RIGHT_INDEX);

  // ── Legs ──
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
  // Pose-only wrist: hand driver will override with a better signal if present.
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
  // Pre-convert all 21 landmarks once.
  const pts: Float32Array[] = new Array(21);
  for (let i = 0; i < 21; i++) {
    pts[i] = new Float32Array(3);
    mpToAnny(hand[i], mirrorX, pts[i]);
  }
  const dir = (a: number, b: number) => normSub(pts[b], pts[a]);

  // Wrist orientation (overrides body's wrist→index proxy)
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

  // Pre-extract rest rotations into a packed (N×9) buffer.
  const restR = new Float32Array(N * 9);
  for (let b = 0; b < N; b++) {
    extractRestRot3(model.restBonePoses, b, restR, b * 9);
  }

  const R_acc = new Float32Array(N * 9);  // pose_R[b] per bone

  // Scratch buffers
  const chain    = new Float32Array(9);
  const invChain = new Float32Array(9);
  const tmp9a    = new Float32Array(9);
  const tmp9b    = new Float32Array(9);
  const tmp3     = new Float32Array(3);

  for (let b = 0; b < N; b++) {
    const parent = model.boneParents[b];

    // chain = restR[b]                                              (root)
    //       = R_acc[parent] @ inv(restR[parent]) @ restR[b]         (non-root)
    if (parent < 0) {
      chain.set(restR.subarray(b*9, b*9+9));
    } else {
      const restPar = restR.subarray(parent*9, parent*9+9);
      const restB   = restR.subarray(b*9, b*9+9);
      const RaccPar = R_acc.subarray(parent*9, parent*9+9);
      transpose3Into(restPar, tmp9a);          // tmp9a = inv(restR[parent])
      mulMat3Into(tmp9a, restB, tmp9b);        // tmp9b = inv(restR[par]) @ restR[b]
      mulMat3Into(RaccPar, tmp9b, chain);      // chain = R_acc[par] @ tmp9b
    }

    const target = targets.get(b);
    let delta: Float32Array | null = null;

    if (target !== undefined) {
      transpose3Into(chain, invChain);

      if (target.kind === "dir") {
        // localTarget = inv(chain) @ worldY
        mulMat3VecInto(invChain, target.worldY, tmp3);
        normalize3(tmp3);
        delta = clampedRotFromTo(Y_AXIS, tmp3, maxAngle);
      } else {
        // kind === "rot": delta = inv(chain) @ worldR
        delta = new Float32Array(9);
        mulMat3Into(invChain, target.worldR, delta);
      }
      deltas[b] = delta;

      // R_acc[b] = chain @ delta
      mulMat3Into(chain, delta, tmp9a);
      R_acc.set(tmp9a, b * 9);
    } else {
      // No delta — R_acc[b] = chain (which equals pose_R[b] for identity delta)
      R_acc.set(chain, b * 9);
    }
  }

  return deltas;
}

// ── Public API ─────────────────────────────────────────────────────────────

export interface PoseInput {
  /** 33 MediaPipe Pose worldLandmarks (metric, hip-centered). Required. */
  pose: WorldLandmark[];
  /** 21 MediaPipe Hand worldLandmarks for the left hand. Optional. */
  leftHand?: WorldLandmark[];
  /** 21 MediaPipe Hand worldLandmarks for the right hand. Optional. */
  rightHand?: WorldLandmark[];
  /**
   * Flip Anny X axis. Only set true if your upstream is already mirrored.
   * Standard MediaPipe output is subject-relative — leave false. Default: false.
   */
  mirrorX?: boolean;
  /** Minimum landmark visibility to drive a bone (default 0.5). */
  visibilityMin?: number;
  /**
   * Maximum delta rotation angle per bone in radians, applied to `dir`
   * targets only. Bounds the damage a single bad landmark can do.
   * Default 2.5 rad (~143°).
   */
  maxAngleRad?: number;
}

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
 * @param boneIndex  Map from bone label → index (build once with buildBoneIndex).
 */
export function landmarksToPoseDeltas(
  input: PoseInput,
  model: AnnyModel,
  boneIndex: Map<string, number>,
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

  // 1. Collect named targets from body + hands.
  const named = new Map<string, BoneTarget>();
  buildBodyTargets(ctx, named);
  buildRootTarget(ctx, named, model, boneIndex);
  if (leftHand)  buildHandTargets(leftHand,  "L", mirrorX, named);
  if (rightHand) buildHandTargets(rightHand, "R", mirrorX, named);

  // 2. Resolve bone names → indices.
  const byIdx = new Map<number, BoneTarget>();
  for (const [name, t] of named) {
    const idx = boneIndex.get(name);
    if (idx !== undefined) byIdx.set(idx, t);
  }

  // 3. Run topological delta computation.
  return computeDeltas(model, byIdx, maxAngleRad);
}

// ── Hand-side assignment helper ────────────────────────────────────────────

/**
 * Pick which detected hand is left vs right, using proximity of each hand's
 * 2D wrist landmark to the body's LEFT_WRIST / RIGHT_WRIST in image space.
 *
 * MediaPipe Hand Landmarker also returns a `handedness` label per hand, but
 * the label is computed in raw camera space and can be flipped vs. user
 * expectation when the display is mirrored. Proximity-to-body-wrist is more
 * robust and doesn't depend on which side of the camera the user is on.
 *
 * @param handsImageLandmarks  HandLandmarker `result.landmarks` (image-space).
 * @param handsWorldLandmarks  HandLandmarker `result.worldLandmarks`.
 * @param poseImageLandmarks   PoseLandmarker `result.landmarks[0]` (image-space).
 * @returns                    Assigned world-space landmark arrays for the API.
 */
export function assignHands(
  handsImageLandmarks: Landmark[][],
  handsWorldLandmarks: WorldLandmark[][],
  poseImageLandmarks: Landmark[],
): { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } {
  const out: { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } = {};
  if (handsImageLandmarks.length === 0) return out;

  const bodyL = poseImageLandmarks[MP.LEFT_WRIST];
  const bodyR = poseImageLandmarks[MP.RIGHT_WRIST];
  if (!bodyL || !bodyR) return out;

  const sq = (a: Landmark, b: Landmark) =>
    (a.x - b.x)**2 + (a.y - b.y)**2;

  if (handsImageLandmarks.length === 1) {
    const w = handsImageLandmarks[0][MP_HAND.WRIST];
    if (sq(w, bodyL) < sq(w, bodyR)) out.leftHand = handsWorldLandmarks[0];
    else                              out.rightHand = handsWorldLandmarks[0];
    return out;
  }

  // ≥2 detected: pick the assignment with minimum total mismatch.
  const [a, b] = handsImageLandmarks;
  const wa = a[MP_HAND.WRIST];
  const wb = b[MP_HAND.WRIST];
  const costAB = sq(wa, bodyL) + sq(wb, bodyR);
  const costBA = sq(wb, bodyL) + sq(wa, bodyR);
  if (costAB <= costBA) {
    out.leftHand  = handsWorldLandmarks[0];
    out.rightHand = handsWorldLandmarks[1];
  } else {
    out.leftHand  = handsWorldLandmarks[1];
    out.rightHand = handsWorldLandmarks[0];
  }
  return out;
}
