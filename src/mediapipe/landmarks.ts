/**
 * MediaPipe Pose + Hand → Anny bone delta rotations.
 *
 * Two stages:
 *
 *   1. **Plan** — reduce the landmarks to world-space intents: body-part
 *      orientations (hips, torso, head, palms) as rotations away from the
 *      rest pose, limb segment directions, and hand-local finger angles.
 *   2. **Solve** — walk the bones parents-first, maintaining each bone's
 *      accumulated world rotation, and turn every intent into a local delta
 *      under the rig's own joint model: hinge axes for elbows/knees/fingers,
 *      forearm twist split out of the wrist, anatomical clamps.
 *
 * Coordinate facts this relies on (all measured, see docs/hand-tracking.md):
 *
 *   • MediaPipe Pose AND Hand `worldLandmarks` are metric and expressed in
 *     axes parallel to the camera (+x right, +y down, +z into the scene). For
 *     a subject facing the camera, +x is the subject's anatomical left.
 *   • Anny world is +x anatomical left, +y back, +z up, so
 *         anny = [mp.x, mp.z, −mp.y]
 *     is a proper rotation and pose + hand landmarks share one frame.
 *   • MediaPipe's hand landmarks have the chirality of its handedness label,
 *     which is anatomical for ordinary (non-mirrored) images.
 *
 * Fingers are driven by *angle transfer*: the flexion/abduction MediaPipe
 * measures between a finger's segments, minus the rig's rest angle, becomes a
 * rotation about the phalanx's own hinge axis. Fingers can therefore only
 * bend the way the rig's joints bend, independent of the palm estimate.
 */

import type { AnnyModel, PoseDeltas } from "../anny/types.js";
import { MP, MP_HAND } from "./constants.js";
import {
  axisAngle, clamp, DEG, frameXFirst, frameXY, IDENTITY, mmul, mslerp, mT, mvec, rotAngle,
  rotFromTo, smoothstep, twistAngle, vangle, vcross, vdot, vmid, vnorm, vsub,
  type Mat3, type Vec3,
} from "./geom.js";
import { fingerAngles, palmFrame, type HandSide } from "./hand-geometry.js";
import { rigInfo, type LimbRig, type RigInfo } from "./rig.js";
import type { PoseInput, WorldLandmark } from "./types.js";

// ── Anatomical limits (radians) ────────────────────────────────────────────

const LIMITS = {
  /** Forearm pronation/supination available from the rig's rest. */
  forearmTwist: 120 * DEG,
  /** Total wrist swing (flexion/deviation) after twist moved to the forearm. */
  wristSwing: 85 * DEG,
  /** Finger flexion ranges per joint, palmar positive: [MCP, PIP, DIP]. */
  fingerFlex: [[-30 * DEG, 100 * DEG], [-10 * DEG, 115 * DEG], [-20 * DEG, 95 * DEG]],
  /** MCP abduction range around the rig's rest abduction. */
  fingerAbd: 25 * DEG,
  /** Max rotation of any thumb bone away from its rest direction. */
  thumbBend: 100 * DEG,
} as const;

/** Elbow/knee bend below which the bend plane can't fix the limb's twist. */
const BEND_PLANE_MIN = 10 * DEG;
const BEND_PLANE_FULL = 30 * DEG;

// ── Stage 1: plan ──────────────────────────────────────────────────────────

interface LimbPlan {
  upperDir?: Vec3;
  lowerDir?: Vec3;
  /** Distal segment for legs: ankle → foot index. */
  footDir?: Vec3;
}

interface HandPlan {
  /** World rotation of the palm away from the rig's rest palm. */
  palm: Mat3;
  /** Finger angles (hand-geometry convention), radians. */
  flex: number[][];
  abd: number[];
  /** Thumb segment directions (CMC→MCP, MCP→IP, IP→TIP) in palm-frame coordinates. */
  thumb: Vec3[];
}

interface Plan {
  hips?: Mat3;
  torso?: Mat3;
  head?: Mat3;
  arms: Partial<Record<HandSide, LimbPlan>>;
  legs: Partial<Record<HandSide, LimbPlan>>;
  hands: Partial<Record<HandSide, HandPlan>>;
}

interface PlanCtx {
  pts: (Vec3 | undefined)[];
  vis: (i: number) => boolean;
}

function poseCtx(pose: WorldLandmark[] | undefined, mirrorX: boolean, visMin: number): PlanCtx {
  const sx = mirrorX ? -1 : 1;
  const pts = (pose ?? []).map((lm): Vec3 => [sx * lm.x, lm.z, -lm.y]);
  return {
    pts,
    vis: (i: number) => pose !== undefined && pose[i] !== undefined && (pose[i].visibility ?? 1) >= visMin,
  };
}

/** Pose landmark `i` in Anny coords, or undefined if missing / not visible. */
function at(ctx: PlanCtx, i: number): Vec3 | undefined {
  return ctx.vis(i) ? ctx.pts[i] : undefined;
}

function allAt(ctx: PlanCtx, ids: number[]): Vec3[] | undefined {
  const out: Vec3[] = [];
  for (const i of ids) {
    const p = at(ctx, i);
    if (!p) return undefined;
    out.push(p);
  }
  return out;
}

function planBody(ctx: PlanCtx, rig: RigInfo, plan: Plan): void {
  const core = allAt(ctx, [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER, MP.LEFT_HIP, MP.RIGHT_HIP]);
  if (core) {
    const [ls, rs, lh, rh] = core;
    // The hip and shoulder LINES are exact (they place the legs and arms);
    // "up" (hips → shoulders) only fixes the lean around them.
    const up = vsub(vmid(ls, rs), vmid(lh, rh));
    plan.hips = mmul(frameXFirst(vsub(lh, rh), up), mT(rig.hipsRest));
    plan.torso = mmul(frameXFirst(vsub(ls, rs), up), mT(rig.torsoRest));
  }

  // Head: x = right ear → left ear, y = mouth → eyes. At rest a level,
  // forward-facing head has x = +x, y = +z — the identity anatomical frame.
  const face = allAt(ctx, [MP.LEFT_EAR, MP.RIGHT_EAR, MP.LEFT_EYE, MP.RIGHT_EYE, MP.MOUTH_LEFT, MP.MOUTH_RIGHT]);
  if (face) {
    const [le, re, lEye, rEye, lm, rm] = face;
    const F = frameXY(vsub(le, re), vsub(vmid(lEye, rEye), vmid(lm, rm)));
    const REST_HEAD: Mat3 = [1, 0, 0, 0, 0, -1, 0, 1, 0];   // cols: +x, +z, −y
    plan.head = mmul(F, mT(REST_HEAD));
  }

  const limb = (a: number, b: number, c: number, d?: number): LimbPlan | undefined => {
    const pa = at(ctx, a), pb = at(ctx, b);
    if (!pa || !pb) return undefined;
    const pc = at(ctx, c);
    const pd = d !== undefined ? at(ctx, d) : undefined;
    return {
      upperDir: vnorm(vsub(pb, pa)),
      lowerDir: pc ? vnorm(vsub(pc, pb)) : undefined,
      footDir: pc && pd ? vnorm(vsub(pd, pc)) : undefined,
    };
  };
  const armL = limb(MP.LEFT_SHOULDER, MP.LEFT_ELBOW, MP.LEFT_WRIST);
  const armR = limb(MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW, MP.RIGHT_WRIST);
  const legL = limb(MP.LEFT_HIP, MP.LEFT_KNEE, MP.LEFT_ANKLE, MP.LEFT_FOOT_INDEX);
  const legR = limb(MP.RIGHT_HIP, MP.RIGHT_KNEE, MP.RIGHT_ANKLE, MP.RIGHT_FOOT_INDEX);
  if (armL) plan.arms.L = armL;
  if (armR) plan.arms.R = armR;
  if (legL) plan.legs.L = legL;
  if (legR) plan.legs.R = legR;

  // No wrist target from POSE: its wrist/index/pinky triangle is too coarse
  // to orient the palm — measured against MediaPipe Hand on the same images
  // it is worse (77° mean palm-normal error) than letting the hand follow
  // the forearm (67°). Without hand landmarks the wrist simply rides along.
}

/** Hand landmarks → Anny coords, depth-flipped if their chirality contradicts `side`. */
function handPoints(hand: WorldLandmark[], side: HandSide, mirrorX: boolean): Vec3[] {
  const sx = mirrorX ? -1 : 1;
  const pts = hand.map((lm): Vec3 => [sx * lm.x, lm.z, -lm.y]);
  if (curl(pts, side) > -CHIRALITY_CURL) return pts;
  // The fingers curl strongly *away* from the palm: MediaPipe picked the
  // mirror-image hand (its classic depth ambiguity — same 2D silhouette,
  // opposite chirality). Reflecting depth about the wrist restores a hand
  // of the right chirality with the same image projection.
  const w = pts[MP_HAND.WRIST];
  const flipped = pts.map((p): Vec3 => [p[0], 2 * w[1] - p[1], p[2]]);
  return curl(flipped, side) > curl(pts, side) ? flipped : pts;
}

/** Mean PIP+DIP flexion over fingers 2–5 (radians, palmar positive). */
function curl(pts: Vec3[], side: HandSide): number {
  const a = fingerAngles(pts, side);
  return a.flex.reduce((s, f) => s + f[1] + f[2], 0) / a.flex.length;
}
const CHIRALITY_CURL = 25 * DEG;

function planHand(pts: Vec3[], side: HandSide, rig: RigInfo): HandPlan {
  const F = palmFrame(pts);
  const { flex, abd } = fingerAngles(pts, side);
  const FT = mT(F);
  const seg = (a: number, b: number): Vec3 => mvec(FT, vnorm(vsub(pts[b], pts[a])));
  return {
    palm: mmul(F, mT(rig.hands[side].palmRest)),
    flex, abd,
    thumb: [
      seg(MP_HAND.THUMB_CMC, MP_HAND.THUMB_MCP),
      seg(MP_HAND.THUMB_MCP, MP_HAND.THUMB_IP),
      seg(MP_HAND.THUMB_IP, MP_HAND.THUMB_TIP),
    ],
  };
}

// ── Stage 2: solve ─────────────────────────────────────────────────────────

/** Computes a bone's local delta from its chain rotation (world rotation at identity delta). */
type Rule = (chain: Mat3) => Mat3 | null;

class Solver {
  readonly world: Mat3[];
  private rules = new Map<number, Rule>();

  constructor(private rig: RigInfo) {
    this.world = new Array<Mat3>(rig.boneCount);
  }

  on(bone: number, rule: Rule): void {
    this.rules.set(bone, rule);
  }

  run(): (Mat3 | null)[] {
    const { rig } = this;
    const out: (Mat3 | null)[] = new Array(rig.boneCount).fill(null);
    for (let b = 0; b < rig.boneCount; b++) {
      const p = rig.parents[b];
      const chain = p < 0 ? rig.restR[b] : mmul(mmul(this.world[p], mT(rig.restR[p])), rig.restR[b]);
      const rule = this.rules.get(b);
      const delta = rule ? rule(chain) : null;
      out[b] = delta;
      this.world[b] = delta ? mmul(chain, delta) : chain;
    }
    return out;
  }
}

/** Delta that makes a bone's world rotation equal `worldR`. */
const orient = (chain: Mat3, worldR: Mat3): Mat3 => mmul(mT(chain), worldR);

/**
 * Delta aligning local `aim` with world `aimW`; if a hint pair is given, also
 * align local `hint` with world `hintW` (fixing the twist about the aim).
 */
function aimDelta(chain: Mat3, aim: Vec3, aimW: Vec3, hint?: Vec3, hintW?: Vec3): Mat3 {
  if (!hint || !hintW) return rotFromTo(aim, vnorm(mvec(mT(chain), aimW)));
  const L = frameXY(hint, aim);
  const W = frameXY(hintW, aimW);
  return mmul(mmul(mT(chain), W), mT(L));
}

/** Limit a delta's rotation angle by slerping toward identity. */
function limitAngle(d: Mat3, max: number): Mat3 {
  const a = rotAngle(d);
  return a > max ? mslerp(IDENTITY, d, max / a) : d;
}

const X_AXIS: Vec3 = [1, 0, 0];
const Y_AXIS: Vec3 = [0, 1, 0];
const Z_AXIS: Vec3 = [0, 0, 1];

/**
 * Upper-limb rule: aim at the middle joint, and twist so the middle joint's
 * hinge matches the observed bend plane. A nearly straight limb has no
 * usable bend plane, so the twist fades to minimal-rotation below 30°.
 *
 * `guardBackward`: a bend whose plane points opposite to the hinge of the
 * minimal-twist solution is a hyperextension reading (knees: landmark noise
 * on a straight leg), not a 180° rotated hip — ignore its plane.
 */
function upperLimbRule(
  limb: LimbRig, upperDir: Vec3, lowerDir: Vec3 | undefined, maxAngle: number, guardBackward: boolean,
): Rule {
  return chain => {
    const minimal = aimDelta(chain, limb.upperAim, upperDir);
    if (!lowerDir) return limitAngle(minimal, maxAngle);
    const bend = vangle(upperDir, lowerDir);
    let w = smoothstep(bend, BEND_PLANE_MIN, BEND_PLANE_FULL);
    const hingeW = vnorm(vcross(upperDir, lowerDir));
    if (guardBackward && vdot(mvec(mmul(chain, minimal), limb.hinge), hingeW) < 0) w = 0;
    if (w === 0) return limitAngle(minimal, maxAngle);
    const hinted = aimDelta(chain, limb.upperAim, upperDir, limb.hinge, hingeW);
    return limitAngle(w === 1 ? hinted : mslerp(minimal, hinted, w), maxAngle);
  };
}

function planArm(solver: Solver, rig: RigInfo, side: HandSide, plan: Plan, maxAngle: number): void {
  const arm = rig.arms[side];
  const hand = plan.hands[side];
  const palm = hand?.palm;
  const wristTarget = palm ? mmul(palm, rig.restR[arm.end]) : undefined;
  // Hand without a pose arm: point the forearm the way the rest forearm
  // relates to the palm, so the hand reads as attached to an arm rather
  // than wrenched round at the wrist.
  const lp: LimbPlan | undefined = plan.arms[side]
    ?? (hand ? { lowerDir: mvec(hand.palm, arm.lowerRestDir) } : undefined);
  let twist = 0;

  if (lp?.upperDir) solver.on(arm.upper, upperLimbRule(arm, lp.upperDir, lp.lowerDir, maxAngle, false));
  if (lp?.lowerDir) {
    const lowerDir = lp.lowerDir;
    solver.on(arm.lower, chain => {
      let d = limitAngle(aimDelta(chain, arm.lowerAim, lowerDir), maxAngle);
      if (wristTarget) {
        // Move the palm's roll about the forearm out of the wrist and into
        // the forearm (pronation/supination), split over both forearm bones.
        const wristChain = mmul(mmul(mmul(chain, d), mT(rig.restR[arm.lower])), rig.restR[arm.end]);
        const need = twistAngle(mmul(mT(wristChain), wristTarget), arm.endTwistAxis);
        twist = clamp(need, -LIMITS.forearmTwist, LIMITS.forearmTwist);
        d = mmul(d, axisAngle(arm.lowerAim, twist / 2));
      }
      return d;
    });
  }
  if (wristTarget) {
    solver.on(arm.lower2, () => axisAngle(arm.lower2Aim, twist / 2));
    solver.on(arm.end, chain => limitAngle(orient(chain, wristTarget), LIMITS.wristSwing));
  }
}

function planLeg(solver: Solver, rig: RigInfo, side: HandSide, plan: Plan, maxAngle: number): void {
  const leg = rig.legs[side];
  const lp = plan.legs[side];
  if (!lp?.upperDir) return;
  solver.on(leg.upper, upperLimbRule(leg, lp.upperDir, lp.lowerDir, maxAngle, true));
  const { lowerDir, footDir } = lp;
  if (lowerDir) solver.on(leg.lower, chain => limitAngle(aimDelta(chain, leg.lowerAim, lowerDir), maxAngle));
  if (footDir) solver.on(leg.end, chain => limitAngle(aimDelta(chain, leg.footAim, footDir), maxAngle));
}

/**
 * Clamp [MCP, PIP, DIP] flexion to anatomical ranges. Flexion beyond a
 * joint's maximum spills into the joint above it instead of being dropped —
 * MediaPipe often reads a fist as a straight MCP with an impossible ~150°
 * PIP; spilling keeps the fingertip curled into the palm where it was seen.
 */
function limitFlex(flex: readonly number[]): number[] {
  const out = [...flex];
  for (let j = 2; j >= 0; j--) {
    const [lo, hi] = LIMITS.fingerFlex[j];
    if (out[j] > hi && j > 0) out[j - 1] += out[j] - hi;
    out[j] = clamp(out[j], lo, hi);
  }
  return out;
}

function planFingers(solver: Solver, rig: RigInfo, side: HandSide, hp: HandPlan): void {
  const hr = rig.hands[side];
  hr.fingers.forEach((f, i) => {
    const [mcp, pip, dip] = limitFlex(hp.flex[i]).map((a, j) => a - f.restFlex[j]);
    const abd = clamp(hp.abd[i] - f.restAbd, -LIMITS.fingerAbd, LIMITS.fingerAbd);
    solver.on(f.bones[0], () => mmul(axisAngle(X_AXIS, hr.flexSign * mcp), axisAngle(Z_AXIS, hr.abdSign * abd)));
    solver.on(f.bones[1], () => axisAngle(X_AXIS, hr.flexSign * pip));
    solver.on(f.bones[2], () => axisAngle(X_AXIS, hr.flexSign * dip));
  });

  // Thumb: carry MediaPipe's palm-relative segment directions over to the
  // rig's *current* palm and aim each thumb bone along them. The thumb's
  // joints don't hinge about axes that line up between MediaPipe's and the
  // rig's hand, so free (minimal-rotation) aims with a cone limit track it
  // far better than hinge projection.
  const palmNow = (): Mat3 => mmul(mmul(solver.world[hr.wrist], mT(rig.restR[hr.wrist])), hr.palmRest);
  hr.thumb.forEach((b, k) => {
    solver.on(b, chain => limitAngle(aimDelta(chain, Y_AXIS, mvec(palmNow(), hp.thumb[k])), LIMITS.thumbBend));
  });
}

function planTrunk(solver: Solver, rig: RigInfo, plan: Plan): void {
  const { hips, torso, head } = plan;
  if (hips) solver.on(rig.root, chain => orient(chain, mmul(hips, rig.restR[rig.root])));
  if (hips && torso) {
    // Spread the hips → chest rotation (mostly torso twist) over the spine.
    rig.spine.forEach((b, k) => {
      const t = (k + 1) / rig.spine.length;
      solver.on(b, chain => orient(chain, mmul(mslerp(hips, torso, t), rig.restR[b])));
    });
  }
  if (torso && head) {
    rig.neck.forEach((b, k) => {
      const t = Math.min(1, (k + 1) / (rig.neck.length - 1));
      solver.on(b, chain => orient(chain, mmul(mslerp(torso, head, t), rig.restR[b])));
    });
  }
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Convert MediaPipe Pose + Hand worldLandmarks to Anny PoseDeltas, ready to
 * feed into `forwardKinematics()`.
 *
 * `pose` drives the trunk, head and limbs; each supplied hand drives the
 * palm orientation (with forearm twist) and all finger joints. Hands work
 * without a pose too (the hand is then oriented in world, the arm stays put).
 *
 * @param input      Landmark inputs + options.
 * @param model      Loaded AnnyModel.
 * @param _boneIndex Unused — kept for API compatibility (rig facts are cached per model).
 */
export function landmarksToPoseDeltas(
  input: PoseInput,
  model: AnnyModel,
  _boneIndex?: Map<string, number>,
): PoseDeltas {
  const {
    pose,
    leftHand,
    rightHand,
    mirrorX = false,
    visibilityMin = 0.5,
    maxAngleRad = Math.PI,
    previousDeltas,
  } = input;
  const rig = rigInfo(model);

  const plan: Plan = { arms: {}, legs: {}, hands: {} };
  planBody(poseCtx(pose, mirrorX, visibilityMin), rig, plan);
  if (leftHand) plan.hands.L = planHand(handPoints(leftHand, "L", mirrorX), "L", rig);
  if (rightHand) plan.hands.R = planHand(handPoints(rightHand, "R", mirrorX), "R", rig);

  const solver = new Solver(rig);
  planTrunk(solver, rig, plan);
  for (const s of ["L", "R"] as const) {
    planArm(solver, rig, s, plan, maxAngleRad);
    planLeg(solver, rig, s, plan, maxAngleRad);
    const hp = plan.hands[s];
    if (hp) planFingers(solver, rig, s, hp);
  }

  const deltas: PoseDeltas = solver.run().map(d => (d ? Float32Array.from(d) : null));

  // Temporal stickiness: bones this frame couldn't drive keep last frame's
  // delta instead of snapping to rest (MP visibility dips, lost hands).
  if (previousDeltas && previousDeltas.length === deltas.length) {
    for (let i = 0; i < deltas.length; i++) {
      if (deltas[i] === null && previousDeltas[i] !== null) deltas[i] = previousDeltas[i];
    }
  }
  return deltas;
}
