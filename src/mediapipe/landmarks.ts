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
 * measures between a finger's segments becomes rotations of the rig's own
 * phalanges (the MCP solved exactly, PIP/DIP about their hinge axes), so
 * fingers only bend the way the rig's joints bend, independent of the palm.
 *
 * With hand quality (palm size in pixels, label agreement — `handsForDriver`)
 * each hand reading is fused with a prior over natural hands and wrists
 * (hand-fusion.ts), trusting it by how well MediaPipe reads a hand that size.
 * Physiology the prior can't express is enforced on top: joint ranges, the
 * DIP–PIP tendon coupling, abduction closing as a finger flexes, the thumb
 * kept out of the fingers (collide.ts), a forearm whose depth the pose got
 * wrong re-tilted to fit a clearly seen hand. See docs/hand-tracking.md.
 */

import type { AnnyModel, PoseDeltas } from "../anny/types.js";
import { MP } from "./constants.js";
import { clearThumb } from "./collide.js";
import { gaussianUpdate } from "./fusion.js";
import {
  axisAngle, clamp, DEG, frameXFirst, frameXY, fromRotVec, IDENTITY, mmul, mslerp, mT, mvec, rotAngle,
  rotFromTo, rotVec, signedAngle, smoothstep, twistAngle, vadd, vangle, vcross, vdot, vlen, vmid, vnorm, vreject,
  vscale, vsub,
  type Mat3, type Vec3,
} from "./geom.js";
import {
  fingerAngles, handPointsAnny, palmFrame, thumbDirections, type HandSide,
} from "./hand-geometry.js";
import {
  coupleFingertips, palmNoise, priorBelief, robustUpdate, shapeNoise, shapeToState, stateToShape,
  type HandBelief, type HandShape,
} from "./hand-fusion.js";
import { rigInfo, type FingerRig, type HandRig, type LimbRig, type RigInfo } from "./rig.js";
import type { HandQuality, PoseInput, WorldLandmark } from "./types.js";

// ── Anatomical limits (radians) ────────────────────────────────────────────

const LIMITS = {
  /**
   * Forearm pronation / supination from neutral. The rig's rest forearm is
   * mid-prone (palm facing the elbow's hinge axis), i.e. anatomical neutral.
   */
  forearmTwist: 90 * DEG,
  /** Humeral rotation the shoulder adds when the forearm alone can't roll the palm far enough. */
  humeralTwist: 70 * DEG,
  /** Wrist extension … palmar flexion (passive range; the prior covers what's typical). */
  wristFlex: [-85 * DEG, 85 * DEG] as const,
  /** Wrist ulnar … radial deviation. */
  wristDeviation: [-40 * DEG, 25 * DEG] as const,
  /** Finger flexion ranges per joint, palmar positive: [MCP, PIP, DIP]. */
  fingerFlex: [[-30 * DEG, 100 * DEG], [-10 * DEG, 115 * DEG], [-20 * DEG, 95 * DEG]],
  /**
   * MCP abduction range around the rig's rest abduction for an open finger;
   * it closes to `fingerAbdFlexed` as the MCP flexes from 30° to 90° (the
   * collateral ligaments tighten — a fist can't spread its fingers).
   */
  fingerAbd: 25 * DEG,
  fingerAbdFlexed: 5 * DEG,
  /** Thumb [MCP, IP] flexion ranges, pad-ward positive. */
  thumbFlex: [[-20 * DEG, 65 * DEG], [-20 * DEG, 85 * DEG]],
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
  /**
   * World rotation of the palm away from the rig's rest palm — absent for a
   * tracked hand MediaPipe didn't find this frame (driven from the belief).
   */
  palm?: Mat3;
  /** Finger angles and thumb directions to drive. */
  shape: HandShape;
  /** This frame's palm-orientation noise variances [roll, flex, deviation] (rad²). */
  palmNoise?: number[];
  /**
   * Belief going into this frame (natural-hand prior, or the tracker's
   * prediction) and coming out of it. Absent: the reading is taken at face
   * value (no quality given, not tracking).
   */
  belief?: HandBelief;
  posterior?: HandBelief;
  /** Whether `belief` is a tracked prediction (outlier-robust updates) rather than the bare prior. */
  tracking?: boolean;
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

/**
 * A hand's plan: its reading fused with the belief about it. With no belief
 * the reading is taken at face value; with a belief but no reading (a
 * tracked hand MediaPipe missed this frame) the belief alone drives it.
 */
function planHand(
  pts: Vec3[] | undefined, side: HandSide, rig: RigInfo, quality: HandQuality | undefined,
  belief: HandBelief | undefined, tracking: boolean,
): HandPlan | undefined {
  const hr = rig.hands[side];
  if (!pts) return belief && { shape: stateToShape(belief.shape.mean, hr), belief, posterior: copyBelief(belief) };
  const measured: HandShape = { ...fingerAngles(pts, side), thumb: thumbDirections(pts) };
  const palm = mmul(palmFrame(pts), mT(hr.palmRest));
  if (!belief) return { palm, shape: measured };
  const q = quality ?? ASSUMED_QUALITY;
  // Against last frame's estimate a wild reading is a blunder; against the
  // prior alone it may just be an unusual hand — only tracking discounts it.
  const update = tracking ? robustUpdate : gaussianUpdate;
  const shape = coupleFingertips(update(belief.shape, shapeToState(measured, hr), shapeNoise(q.palmPx)), hr);
  return {
    palm, shape: stateToShape(shape.mean, hr), palmNoise: palmNoise(q),
    belief, posterior: { ...copyBelief(belief), shape }, tracking,
  };
}

/** Quality assumed for a tracked hand given without one: a clear, mid-sized hand. */
const ASSUMED_QUALITY: HandQuality = { palmPx: 150, labelAgrees: true };

function copyBelief(b: HandBelief): HandBelief {
  return { ...b, wrist: { mean: [...b.wrist.mean], cov: [...b.wrist.cov] } };
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

/**
 * MediaPipe's 3D pose is weakest in depth, and a forearm's depth is what
 * decides how a visible hand must sit on it: a "stop" hand on a forearm read
 * as lying across the chest needs a 150°+ roll; the same hand on a forearm
 * pointing at the camera needs only wrist extension. SD of the pose
 * forearm's out-of-image-plane angle (radians).
 */
const FOREARM_DEPTH_SD = 25 * DEG;
/** Frame-to-frame SD of the forearm depth while tracking (radians). */
const FOREARM_DEPTH_STEP_SD = 15 * DEG;
/** Range-of-motion overshoot tolerated before it counts against a forearm depth. */
const ROM_SLACK = 5 * DEG;
const DEPTH_SEARCH = { range: 80 * DEG, steps: 32, refine: 12 };

/** A palm reading good enough to orient an arm nobody saw: a clear, mid-sized hand with consistent chirality. */
function reliablePalm(hand: HandPlan | undefined): boolean {
  return !hand?.palmNoise || Math.max(...hand.palmNoise) < (20 * DEG) ** 2;
}

function planArm(solver: Solver, rig: RigInfo, side: HandSide, plan: Plan, maxAngle: number, holdArm: boolean): void {
  const arm = rig.arms[side];
  const hr = rig.hands[side];
  const hand = plan.hands[side];
  const palm = hand?.palm;
  const wristTarget = palm ? mmul(palm, rig.restR[arm.end]) : undefined;
  const lp = plan.arms[side];
  if (!lp) {
    // No observed arm: the whole arm simply carries the palm — rotated with
    // it from the shoulder, as if arm and hand were one piece — when the palm
    // reading is all there is to go on (no pose: a close-up) or can be
    // trusted. In video with a doubtful hand on an out-of-sight arm, the arm
    // and wrist hold their last pose instead (previousDeltas): spinning a
    // forearm around to meet a noisy palm reading alone is exactly how a
    // wrist ends up wrung 180°.
    if (palm && (!holdArm || reliablePalm(hand))) {
      solver.on(arm.upper, chain => orient(chain, mmul(palm, rig.restR[arm.upper])));
    }
    return;
  }
  let twist = 0;

  if (lp.upperDir) solver.on(arm.upper, upperLimbRule(arm, lp.upperDir, lp.lowerDir, maxAngle, false));
  if (!lp.lowerDir) return;
  const poseDir = lp.lowerDir;
  const aim = (chain: Mat3, dir: Vec3): Mat3 => limitAngle(aimDelta(chain, arm.lowerAim, dir), maxAngle);
  if (!hand || (!wristTarget && !hand.belief)) {
    solver.on(arm.lower, chain => aim(chain, poseDir));
    return;
  }
  // Roll beyond the forearm's range comes from the shoulder (humeral
  // rotation). With a straight elbow the pose can't see it, so it's free;
  // a bent elbow's plane pins it.
  const free = lp.upperDir ? 1 - smoothstep(vangle(lp.upperDir, poseDir), BEND_PLANE_MIN, BEND_PLANE_FULL) : 0;
  const rollRange = LIMITS.forearmTwist + free * LIMITS.humeralTwist;

  const frame = wristFrame(arm, hr);
  const wristChainOf = (chain: Mat3, d: Mat3): Mat3 =>
    mmul(mmul(mmul(chain, d), mT(rig.restR[arm.lower])), rig.restR[arm.end]);
  /**
   * The wrist [roll, flexion, deviation] the palm reading asks of a forearm
   * with this chain and aim, and — when the reading's quality is known — its
   * noise covariance. MediaPipe's palm noise is about the camera axes (roll
   * in the image plane is read far better than tilt out of it), so it is
   * carried over to the wrist's axes at this pose.
   */
  const reading = (target: Mat3, chain: Mat3, d: Mat3): { z: number[]; R?: number[] } => {
    const wristChain = wristChainOf(chain, d);
    const z = wristReading(mmul(mT(wristChain), target), frame);
    if (!hand.palmNoise) return { z };
    const rolled = mmul(wristChain, axisAngle(frame.twist, z[0]));
    const axes = [mvec(wristChain, frame.twist), mvec(rolled, frame.flex), mvec(rolled, frame.radial)];
    return { z, R: covarianceAbout(axes, hand.palmNoise) };
  };
  /**
   * This frame's wrist: the reading fused with the belief (recorded as the
   * posterior); the belief alone for a tracked hand MediaPipe missed; the
   * reading as is without either.
   */
  const wrist = (chain: Mat3, d: Mat3): number[] => {
    const { belief, posterior, tracking } = hand;
    if (!wristTarget) return belief ? belief.wrist.mean : [0, 0, 0];
    const { z, R } = reading(wristTarget, chain, d);
    if (!belief || !posterior || !R) return z;
    posterior.wrist = (tracking ? robustUpdate : gaussianUpdate)(belief.wrist, z, R);
    return posterior.wrist.mean;
  };
  // The forearm direction, its depth settled against the hand (once, by
  // whichever of the arm's rules runs first).
  let lowerDir: Vec3 | undefined;
  const forearm = (lowerChain: Mat3): Vec3 => {
    if (!lowerDir) {
      lowerDir = wristTarget
        ? forearmDepth(poseDir, dir => reading(wristTarget, lowerChain, aim(lowerChain, dir)),
          rollRange, hand.belief?.forearmDepth)
        : poseDir;
      if (hand.posterior) hand.posterior.forearmDepth = depthAngle(lowerDir);
    }
    return lowerDir;
  };

  if (wristTarget && free > 0) {
    solver.on(arm.upper2, chain => {
      const lowerChain = mmul(mmul(chain, mT(rig.restR[arm.upper2])), rig.restR[arm.lower]);
      const want = wrist(lowerChain, aim(lowerChain, forearm(lowerChain)))[0];
      const excess = want - clamp(want, -LIMITS.forearmTwist, LIMITS.forearmTwist);
      return axisAngle(arm.upper2Aim, free * clamp(excess, -LIMITS.humeralTwist, LIMITS.humeralTwist));
    });
  }
  // The hand orientation the fused wrist describes, for the wrist bone.
  let handTarget: Mat3 = IDENTITY;
  solver.on(arm.lower, chain => {
    const d = aim(chain, forearm(chain));
    const [roll, flex, dev] = wrist(chain, d);
    handTarget = mmul(mmul(wristChainOf(chain, d), axisAngle(frame.twist, roll)),
      fromRotVec(vadd(vscale(frame.flex, flex), vscale(frame.radial, dev))));
    // Move the palm's roll about the forearm out of the wrist and into the
    // forearm (pronation/supination), split over both forearm bones.
    twist = clamp(roll, -LIMITS.forearmTwist, LIMITS.forearmTwist);
    return mmul(d, axisAngle(arm.lowerAim, twist / 2));
  });
  solver.on(arm.lower2, () => axisAngle(arm.lower2Aim, twist / 2));
  // The wrist keeps only its swing — palmar flexion and radial deviation —
  // held inside its range of motion: roll the forearm couldn't take is
  // dropped (the wrist itself barely rotates about the forearm), so the hand
  // still points where it was seen pointing.
  solver.on(arm.end, chain => {
    const [flex, dev] = wristSwing(orient(chain, handTarget), frame);
    return fromRotVec(vadd(
      vscale(frame.flex, clamp(flex, LIMITS.wristFlex[0], LIMITS.wristFlex[1])),
      vscale(frame.radial, clamp(dev, LIMITS.wristDeviation[0], LIMITS.wristDeviation[1])),
    ));
  });
}

/** Out-of-image-plane angle of a world direction (Anny +y points away from the camera). */
const depthAngle = (d: Vec3): number => Math.atan2(d[1], Math.hypot(d[0], d[2]));

/** Wrist-local axes: roll (along the forearm), palmar flexion, radial deviation — orthonormal. */
interface WristFrame {
  twist: Vec3;
  flex: Vec3;
  radial: Vec3;
}

function wristFrame(arm: LimbRig, hr: HandRig): WristFrame {
  const twist = arm.endTwistAxis;
  const flex = vnorm(vreject(hr.wristFlexAxis, twist));
  return { twist, flex, radial: vnorm(vreject(vreject(hr.wristRadialAxis, twist), flex)) };
}

/** A wrist delta's swing as [flexion, deviation] (radians), its roll about the forearm dropped. */
function wristSwing(delta: Mat3, f: WristFrame): [number, number] {
  const r = rotVec(mmul(delta, axisAngle(f.twist, -twistAngle(delta, f.twist))));
  return [vdot(r, f.flex), vdot(r, f.radial)];
}

/**
 * A wrist delta as [roll, flexion, deviation] (radians): the roll taken
 * first — by the forearm — then the swing in the rolled frame, so that
 * roll(a)·swing(b, c) rebuilds it exactly.
 */
function wristReading(delta: Mat3, f: WristFrame): number[] {
  const roll = twistAngle(delta, f.twist);
  const r = mvec(axisAngle(f.twist, -roll), rotVec(mmul(delta, axisAngle(f.twist, -roll))));
  return [roll, vdot(r, f.flex), vdot(r, f.radial)];
}

/** Covariance about `axes` (unit, roughly orthonormal) of a noise diagonal in world axes. */
function covarianceAbout(axes: Vec3[], worldVar: readonly number[]): number[] {
  const n = axes.length;
  const out = new Array<number>(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    out[i * n + j] = axes[i][0] * axes[j][0] * worldVar[0] + axes[i][1] * axes[j][1] * worldVar[1] + axes[i][2] * axes[j][2] * worldVar[2];
  }
  return out;
}

const excess = (x: number, lo: number, hi: number): number => Math.max(0, lo - x, x - hi);

/**
 * Re-estimate the forearm's depth from the hand. The forearm's direction in
 * the image plane is kept (that's what the camera sees well); its tilt out
 * of the plane is chosen to balance the pose's own estimate against the
 * wrist range of motion the hand reading would otherwise violate — weighted
 * by how much that reading can be trusted, so a 10 px hand can't overrule
 * the pose but a clear close-up hand can.
 */
function forearmDepth(
  poseDir: Vec3, need: (dir: Vec3) => { z: number[]; R?: number[] }, rollRange: number, previous?: number,
): Vec3 {
  const inPlane = Math.hypot(poseDir[0], poseDir[2]);
  if (inPlane < 0.05) return poseDir;   // pointing at the camera: no image-plane direction to keep
  const ux = poseDir[0] / inPlane, uz = poseDir[2] / inPlane;
  const dirAt = (phi: number): Vec3 => [Math.cos(phi) * ux, Math.sin(phi), Math.cos(phi) * uz];
  const phi0 = Math.atan2(poseDir[1], inPlane);
  const cost = (phi: number): number => {
    const { z, R } = need(dirAt(phi));
    const v = [
      excess(z[0], -rollRange, rollRange),
      excess(z[1], LIMITS.wristFlex[0], LIMITS.wristFlex[1]),
      excess(z[2], LIMITS.wristDeviation[0], LIMITS.wristDeviation[1]),
    ];
    const steady = previous === undefined ? 0 : ((phi - previous) / FOREARM_DEPTH_STEP_SD) ** 2;
    return ((phi - phi0) / FOREARM_DEPTH_SD) ** 2 + steady
      + v.reduce((s, x, k) => s + x * x / ((R ? R[k * 4] : 0) + ROM_SLACK ** 2), 0);
  };
  let best = phi0, bestCost = cost(phi0);
  if (bestCost === 0) return poseDir;
  const { range, steps, refine } = DEPTH_SEARCH;
  for (let i = 0; i <= steps; i++) {
    const phi = -range + (2 * range * i) / steps;
    const c = cost(phi);
    if (c < bestCost) { best = phi; bestCost = c; }
  }
  // Golden-section refinement around the best grid point.
  let lo = best - range / steps, hi = best + range / steps;
  const g = (Math.sqrt(5) - 1) / 2;
  for (let i = 0; i < refine; i++) {
    const a = hi - g * (hi - lo), b = lo + g * (hi - lo);
    if (cost(a) < cost(b)) hi = b; else lo = a;
  }
  const mid = (lo + hi) / 2;
  return cost(mid) < bestCost ? dirAt(mid) : best === phi0 ? poseDir : dirAt(best);
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

/**
 * MCP delta that puts the rig's proximal phalanx exactly where MCP flexion
 * `flex` and abduction `abd` (hand-geometry convention, absolute) place it.
 *
 * The rig's MCP axes don't line up with the anatomical flexion plane — its
 * local X tilts each finger sideways as it curls (index: ~10° over a 60°
 * curl) — so transferring angles as plain rotations about X and Z would add
 * that tilt on top of what was measured. Instead: build the target phalanx
 * direction in the rest hand, then solve Rx(x)·Rz(z)·p = t in closed form.
 */
function mcpDelta(f: FingerRig, side: HandSide, rest: Mat3, flex: number, abd: number): Mat3 {
  const inPlane = mvec(axisAngle(f.restHinge, flex), f.restMetacarpal);
  const a = (side === "R" ? 1 : -1) * abd;
  const t = mvec(mT(rest), vadd(vscale(inPlane, Math.cos(a)), vscale(f.restHinge, Math.sin(a))));
  const p = f.proximalLocal;
  // Rz(z) must give p the x-component of t (Rx keeps x): p.x cos z − p.y sin z = t.x.
  const r = Math.hypot(p[0], p[1]);
  const phi = Math.atan2(-p[1], p[0]);
  const spread = Math.acos(clamp(t[0] / r, -1, 1));
  const wrap = (x: number) => Math.atan2(Math.sin(x), Math.cos(x));
  const z1 = wrap(phi + spread), z2 = wrap(phi - spread);
  const z = Math.abs(z1) <= Math.abs(z2) ? z1 : z2;
  const v = mvec(axisAngle(Z_AXIS, z), p);
  const x = Math.atan2(v[1] * t[2] - v[2] * t[1], v[1] * t[1] + v[2] * t[2]);
  return mmul(axisAngle(X_AXIS, x), axisAngle(Z_AXIS, z));
}

function planFingers(solver: Solver, rig: RigInfo, side: HandSide, hp: HandPlan): void {
  const hr = rig.hands[side];
  hr.fingers.forEach((f, i) => {
    const flex = limitFlex(hp.shape.flex[i]);
    const [, pip, dip] = flex.map((a, j) => a - f.restFlex[j]);
    const closing = clamp((flex[0] - 30 * DEG) / (60 * DEG), 0, 1);
    const spread = LIMITS.fingerAbd + closing * (LIMITS.fingerAbdFlexed - LIMITS.fingerAbd);
    const abd = clamp(hp.shape.abd[i], f.restAbd - spread, f.restAbd + spread);
    const mcp = mcpDelta(f, side, rig.restR[f.bones[0]], flex[0], abd);
    solver.on(f.bones[0], () => mcp);
    solver.on(f.bones[1], () => axisAngle(X_AXIS, hr.flexSign * pip));
    solver.on(f.bones[2], () => axisAngle(X_AXIS, hr.flexSign * dip));
  });

  // Thumb: carry MediaPipe's palm-relative segment directions over to the
  // rig's *current* palm and aim each thumb bone along them. The thumb's
  // joints don't hinge about axes that line up between MediaPipe's and the
  // rig's hand, so free (minimal-rotation) aims with a cone limit track it
  // far better than hinge projection.
  const palmNow = (): Mat3 => mmul(mmul(solver.world[hr.wrist], mT(rig.restR[hr.wrist])), hr.palmRest);
  const thumb = limitThumb(hp.shape.thumb, hr.thumbPad);
  hr.thumb.forEach((b, k) => {
    solver.on(b, chain => limitAngle(aimDelta(chain, Y_AXIS, mvec(palmNow(), thumb[k])), LIMITS.thumbBend));
  });
}

/**
 * Hold the thumb's MCP and IP inside their range: each joint's flexion is
 * the bend between consecutive segments toward the pad side (the index
 * knuckle); a bend past the range — MediaPipe likes to bend the thumb tip
 * backwards — is rotated back to the limit, carrying the segments beyond.
 */
function limitThumb(dirs: readonly Vec3[], pad: readonly [Vec3, Vec3]): Vec3[] {
  const out = [...dirs];
  for (let k = 1; k <= 2; k++) {
    const hinge = vcross(out[k - 1], pad[k - 1]);
    if (vlen(hinge) < 1e-6) continue;
    const axis = vnorm(hinge);
    const bend = signedAngle(out[k - 1], out[k], axis);
    const [lo, hi] = LIMITS.thumbFlex[k - 1];
    const fix = clamp(bend, lo, hi) - bend;
    if (fix === 0) continue;
    const R = axisAngle(axis, fix);
    for (let j = k; j < out.length; j++) out[j] = mvec(R, out[j]);
  }
  return out;
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

/** Per-hand beliefs `PoseTracker` carries from frame to frame. */
export type HandBeliefs = Partial<Record<HandSide, HandBelief>>;

/**
 * Convert MediaPipe Pose + Hand worldLandmarks to Anny PoseDeltas, ready to
 * feed into `forwardKinematics()`.
 *
 * `pose` drives the trunk, head and limbs; each supplied hand drives the
 * palm orientation (with forearm twist) and all finger joints. Hands work
 * without a pose too (the hand is then oriented in world, the arm stays put).
 * With `leftHandQuality` / `rightHandQuality` (from `handsForDriver`), each
 * hand reading is fused with a prior over natural hands, weighted by how
 * reliable a reading of that size is. For video, `PoseTracker` additionally
 * carries each hand's estimate from frame to frame.
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
  return solvePose(input, model).deltas;
}

/**
 * The driver proper. `tracked` switches on tracking: each hand starts from
 * its belief there (or the prior on first sight), is driven from it when
 * MediaPipe misses it, and its updated belief is returned.
 */
export function solvePose(
  input: PoseInput, model: AnnyModel, tracked?: HandBeliefs,
): { deltas: PoseDeltas; beliefs: HandBeliefs } {
  const {
    pose,
    leftHand,
    rightHand,
    leftHandQuality,
    rightHandQuality,
    mirrorX = false,
    visibilityMin = 0.5,
    maxAngleRad = Math.PI,
    previousDeltas,
  } = input;
  const rig = rigInfo(model);

  const plan: Plan = { arms: {}, legs: {}, hands: {} };
  planBody(poseCtx(pose, mirrorX, visibilityMin), rig, plan);
  for (const s of ["L", "R"] as const) {
    const lm = s === "L" ? leftHand : rightHand;
    const quality = s === "L" ? leftHandQuality : rightHandQuality;
    const belief = tracked
      ? tracked[s] ?? (lm ? priorBelief() : undefined)
      : quality ? priorBelief() : undefined;
    const hp = planHand(lm && handPointsAnny(lm, s, mirrorX), s, rig, quality, belief, tracked !== undefined);
    if (hp) plan.hands[s] = hp;
  }

  const solver = new Solver(rig);
  planTrunk(solver, rig, plan);
  for (const s of ["L", "R"] as const) {
    // An out-of-sight arm can hold its last pose only in video, with a pose.
    planArm(solver, rig, s, plan, maxAngleRad, pose !== undefined && previousDeltas !== undefined);
    planLeg(solver, rig, s, plan, maxAngleRad);
    const hp = plan.hands[s];
    if (hp) planFingers(solver, rig, s, hp);
  }

  const solved = solver.run();
  for (const s of ["L", "R"] as const) {
    if (!plan.hands[s]) continue;
    const clearance = rig.hands[s].thumbClearance;
    const base = clearThumb(clearance, solver.world, rig.parents, rig.restR, rig.restHead);
    if (base) solved[clearance.base] = base;
  }
  const deltas: PoseDeltas = solved.map(d => (d ? Float32Array.from(d) : null));

  // Temporal stickiness: bones this frame couldn't drive keep last frame's
  // delta instead of snapping to rest (MP visibility dips, lost hands).
  if (previousDeltas && previousDeltas.length === deltas.length) {
    for (let i = 0; i < deltas.length; i++) {
      if (deltas[i] === null && previousDeltas[i] !== null) deltas[i] = previousDeltas[i];
    }
  }
  const beliefs: HandBeliefs = {};
  for (const s of ["L", "R"] as const) {
    const post = plan.hands[s]?.posterior;
    if (post) beliefs[s] = post;
  }
  return { deltas, beliefs };
}
