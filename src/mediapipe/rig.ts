/**
 * Rest-pose facts about the Anny rig that the driver needs, extracted once
 * per model and cached.
 *
 * Nothing here is hand-tuned per rig: aim vectors come from child-joint
 * offsets, hinge axes from the rest bend of each limb, sign conventions
 * from small numerical probes of the rest geometry. Re-baking the model
 * (different phenotype) re-derives everything.
 */

import type { AnnyModel } from "../anny/types.js";
import { MP_HAND } from "./constants.js";
import {
  axisAngle, frameXFirst, mmul, mT, mvec, vcross, vdot, vnorm, vsub,
  type Mat3, type Vec3,
} from "./geom.js";
import { phalanxCapsule, type ThumbClearanceRig } from "./collide.js";
import { FINGER_CHAINS, fingerAngles, palmarNormal, palmFrame, type HandSide } from "./hand-geometry.js";

export type Side = HandSide;

export interface LimbRig {
  upper: number;
  /** Twist bone between `upper` and `lower` (humeral / femoral rotation). */
  upper2: number;
  lower: number;
  lower2: number;
  /** Distal joint bone: wrist or foot. */
  end: number;
  /** `upper`-local unit vector toward the middle joint (elbow / knee). */
  upperAim: Vec3;
  /** `upper2`-local unit vector toward the middle joint (its twist axis). */
  upper2Aim: Vec3;
  /**
   * `upper`-local unit vector of the middle joint's flexion axis, oriented
   * like cross(upperDir, lowerDir) of a flexed limb.
   */
  hinge: Vec3;
  /** `lower`-local unit vector toward the distal joint. */
  lowerAim: Vec3;
  /** `lower2`-local unit vector toward the distal joint (its twist axis). */
  lower2Aim: Vec3;
  /** `end`-local unit vector along the segment (`lower` → `end`), for twist splits. */
  endTwistAxis: Vec3;
  /** World direction `lower` → `end` at rest. */
  lowerRestDir: Vec3;
}

export interface FingerRig {
  /** [proximal, middle, distal] phalanx bones. */
  bones: [number, number, number];
  /** Rest [MCP, PIP, DIP] flexion, radians (hand-geometry convention). */
  restFlex: number[];
  restAbd: number;
  /**
   * The MCP's measurement frame at rest (world, unit): the metacarpal
   * direction (wrist → MCP) and the flexion hinge ⊥ it and the palm normal,
   * exactly as `fingerAngles` defines them.
   */
  restMetacarpal: Vec3;
  restHinge: Vec3;
  /** Proximal phalanx direction (MCP → PIP) in its own bone's local frame. */
  proximalLocal: Vec3;
}

export interface HandRig {
  wrist: number;
  /** Rest palm frame (hand-geometry `palmFrame` of the rig's rest joints). */
  palmRest: Mat3;
  fingers: FingerRig[];
  /** finger1-1/2/3 — thumb metacarpal, proximal, distal. */
  thumb: [number, number, number];
  /** Δabduction per +rotation about a proximal phalanx's local Z (±1). */
  abdSign: number;
  /** Δflexion per +rotation about a phalanx's local X (±1). */
  flexSign: number;
  /** Wrist-local unit axis whose +rotation flexes the hand toward the palm. */
  wristFlexAxis: Vec3;
  /** Wrist-local unit axis whose +rotation deviates the hand toward the thumb (radial). */
  wristRadialAxis: Vec3;
  /** Capsules for keeping the thumb out of the fingers. */
  thumbClearance: ThumbClearanceRig;
  /**
   * Palm-frame unit directions from the thumb's MCP and IP joints toward the
   * index knuckle at rest: the side the thumb's pad faces, i.e. its flexion
   * direction (as the hand-angle metrics define it).
   */
  thumbPad: [Vec3, Vec3];
}

export interface RigInfo {
  boneCount: number;
  parents: Int32Array;
  restR: Mat3[];
  restHead: Vec3[];
  index: Map<string, number>;
  root: number;
  /** spine05 → spine01 (pelvis to chest), present bones only. */
  spine: number[];
  /** neck01 → head. */
  neck: number[];
  /** Rest anatomical frames (cols: left, up, forward-out-of-chest). */
  hipsRest: Mat3;
  torsoRest: Mat3;
  arms: Record<Side, LimbRig>;
  legs: Record<Side, LimbRig & { footAim: Vec3; toe: number }>;
  hands: Record<Side, HandRig>;
}

const cache = new WeakMap<AnnyModel, RigInfo>();

export function rigInfo(model: AnnyModel): RigInfo {
  let info = cache.get(model);
  if (!info) {
    info = buildRigInfo(model);
    cache.set(model, info);
  }
  return info;
}

function buildRigInfo(model: AnnyModel): RigInfo {
  const N = model.boneCount;
  const rp = model.restBonePoses;
  const restR: Mat3[] = [];
  const restHead: Vec3[] = [];
  for (let b = 0; b < N; b++) {
    const o = b * 16;
    restR.push([rp[o], rp[o + 1], rp[o + 2], rp[o + 4], rp[o + 5], rp[o + 6], rp[o + 8], rp[o + 9], rp[o + 10]]);
    restHead.push([rp[o + 3], rp[o + 7], rp[o + 11]]);
  }
  const index = new Map<string, number>();
  model.boneLabels.forEach((n, i) => index.set(n, i));
  const bone = (name: string): number => {
    const i = index.get(name);
    if (i === undefined) throw new Error(`Anny rig is missing bone "${name}"`);
    return i;
  };
  const head = (name: string): Vec3 => restHead[bone(name)];
  /** Unit vector from bone `a`'s head toward `b`'s head, in `a`'s local frame. */
  const aimLocal = (a: string, b: string): Vec3 =>
    vnorm(mvec(mT(restR[bone(a)]), vsub(head(b), head(a))));
  const xAxis = (b: number): Vec3 => [restR[b][0], restR[b][3], restR[b][6]];

  const limb = (upper: string, upper2: string, lower: string, lower2: string, end: string): LimbRig => {
    const upperDir = vnorm(vsub(head(lower), head(upper)));
    const lowerDir = vnorm(vsub(head(end), head(lower)));
    // The flexion axis is the middle bone's local X (elbow/knee hinge in this
    // rig); orient it like cross(upper, lower) so a flexed limb's bend-plane
    // normal can be matched to it directly.
    const bend = vcross(upperDir, lowerDir);
    let hingeWorld = xAxis(bone(lower));
    if (vdot(hingeWorld, bend) < 0) hingeWorld = [-hingeWorld[0], -hingeWorld[1], -hingeWorld[2]];
    return {
      upper: bone(upper), upper2: bone(upper2), lower: bone(lower), lower2: bone(lower2), end: bone(end),
      upperAim: aimLocal(upper, lower),
      upper2Aim: aimLocal(upper2, lower),
      hinge: vnorm(mvec(mT(restR[bone(upper)]), hingeWorld)),
      lowerAim: aimLocal(lower, end),
      lower2Aim: aimLocal(lower2, end),
      endTwistAxis: vnorm(mvec(mT(restR[bone(end)]), lowerDir)),
      lowerRestDir: lowerDir,
    };
  };

  const anatomical = (left: Vec3, right: Vec3, up: Vec3, down: Vec3): Mat3 =>
    frameXFirst(vsub(left, right), vsub(up, down));
  const hipsRest = anatomical(head("upperleg01.L"), head("upperleg01.R"),
    midpoint(head("upperarm01.L"), head("upperarm01.R")), midpoint(head("upperleg01.L"), head("upperleg01.R")));
  const torsoRest = anatomical(head("upperarm01.L"), head("upperarm01.R"),
    midpoint(head("upperarm01.L"), head("upperarm01.R")), midpoint(head("upperleg01.L"), head("upperleg01.R")));

  const hand = (s: Side): HandRig => {
    const pts = restHandPoints(s, head, restR, bone);
    const angles = fingerAngles(pts, s);
    const palmar = palmarNormal(palmFrame(pts), s);
    const fingers: FingerRig[] = [2, 3, 4, 5].map((f, i) => {
      const [mcp, pip] = FINGER_CHAINS[i];
      const proximal = bone(`finger${f}-1.${s}`);
      const metacarpal = vnorm(vsub(pts[mcp], pts[MP_HAND.WRIST]));
      return {
        bones: [proximal, bone(`finger${f}-2.${s}`), bone(`finger${f}-3.${s}`)],
        restFlex: angles.flex[i],
        restAbd: angles.abd[i],
        restMetacarpal: metacarpal,
        restHinge: vnorm(vcross(metacarpal, palmar)),
        proximalLocal: vnorm(mvec(mT(restR[proximal]), vsub(pts[pip], pts[mcp]))),
      };
    });
    // Sign probes: rotate the middle finger's proximal phalanx about its own
    // local Z / X by a small angle and see which way the hand-geometry
    // abduction / flexion moves. Mirrored rigs flip Z but not X; probing
    // keeps us honest either way.
    const probe = (axisCol: 0 | 2): number => {
      const b = fingers[1].bones[0];
      const R = restR[b];
      const axis: Vec3 = [R[axisCol], R[3 + axisCol], R[6 + axisCol]];
      const rot = axisAngle(axis, 0.1);
      const moved = pts.slice();
      const base = pts[MP_HAND.MIDDLE_MCP];
      for (const k of [MP_HAND.MIDDLE_PIP, MP_HAND.MIDDLE_DIP, MP_HAND.MIDDLE_TIP]) {
        const d = mvec(rot, vsub(pts[k], base));
        moved[k] = [base[0] + d[0], base[1] + d[1], base[2] + d[2]];
      }
      const after = fingerAngles(moved, s);
      const delta = axisCol === 2 ? after.abd[1] - angles.abd[1] : after.flex[1][0] - angles.flex[1][0];
      return Math.sign(delta) || 1;
    };
    const palmRest = palmFrame(pts);
    return {
      wrist: bone(`wrist.${s}`),
      palmRest,
      fingers,
      thumb: [bone(`finger1-1.${s}`), bone(`finger1-2.${s}`), bone(`finger1-3.${s}`)],
      abdSign: probe(2),
      flexSign: probe(0),
      ...wristAxes(bone(`wrist.${s}`), pts, palmRest, s, restR),
      thumbPad: [MP_HAND.THUMB_MCP, MP_HAND.THUMB_IP].map(j =>
        vnorm(mvec(mT(palmRest), vsub(pts[MP_HAND.INDEX_MCP], pts[j])))) as [Vec3, Vec3],
      thumbClearance: {
        base: bone(`finger1-1.${s}`),
        thumb: [capsule(`finger1-2.${s}`, `finger1-3.${s}`), capsule(`finger1-3.${s}`, null)],
        fingers: [2, 3, 4].flatMap(f => [
          capsule(`finger${f}-1.${s}`, `finger${f}-2.${s}`),
          capsule(`finger${f}-2.${s}`, `finger${f}-3.${s}`),
          capsule(`finger${f}-3.${s}`, null),
        ]),
      },
    };
  };
  const capsule = (b: string, child: string | null) =>
    phalanxCapsule(model, bone(b), child === null ? null : bone(child), restR, restHead);

  const leg = (s: Side) => ({
    ...limb(`upperleg01.${s}`, `upperleg02.${s}`, `lowerleg01.${s}`, `lowerleg02.${s}`, `foot.${s}`),
    toe: bone(`toe2-1.${s}`),
    footAim: aimLocal(`foot.${s}`, `toe2-1.${s}`),
  });

  const present = (names: string[]): number[] => names.map(n => index.get(n)).filter((i): i is number => i !== undefined);

  return {
    boneCount: N,
    parents: model.boneParents,
    restR, restHead, index,
    root: bone("root"),
    spine: present(["spine05", "spine04", "spine03", "spine02", "spine01"]),
    neck: present(["neck01", "neck02", "neck03", "head"]),
    hipsRest, torsoRest,
    arms: {
      L: limb("upperarm01.L", "upperarm02.L", "lowerarm01.L", "lowerarm02.L", "wrist.L"),
      R: limb("upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R"),
    },
    legs: { L: leg("L"), R: leg("R") },
    hands: { L: hand("L"), R: hand("R") },
  };
}

/**
 * Probe the wrist bone: rotate the rest hand a little about each local axis
 * and see where the middle knuckle goes. The axis that moves it most toward
 * the palm is the flexion axis; the other one the deviation axis, signed
 * toward the thumb. (In this rig: X deviates radially on both hands, Z flexes
 * with opposite signs on the two hands.)
 */
function wristAxes(
  wrist: number, pts: Vec3[], palmRest: Mat3, side: Side, restR: Mat3[],
): { wristFlexAxis: Vec3; wristRadialAxis: Vec3 } {
  const R = restR[wrist];
  const head = pts[MP_HAND.WRIST];
  const palmar = palmarNormal(palmRest, side);
  const radial: Vec3 = [palmRest[0], palmRest[3], palmRest[6]];
  const move = (local: Vec3): Vec3 => {
    const rot = mmul(mmul(R, axisAngle(local, 0.1)), mT(R));
    return vsub(mvec(rot, vsub(pts[MP_HAND.MIDDLE_MCP], head)), vsub(pts[MP_HAND.MIDDLE_MCP], head));
  };
  const axes: Vec3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const moves = axes.map(move);
  const flexK = [0, 1, 2].reduce((best, k) => (Math.abs(vdot(moves[k], palmar)) > Math.abs(vdot(moves[best], palmar)) ? k : best), 0);
  const devK = [0, 1, 2].filter(k => k !== flexK)
    .reduce((best, k) => (Math.abs(vdot(moves[k], radial)) > Math.abs(vdot(moves[best], radial)) ? k : best));
  const signed = (k: number, dir: Vec3): Vec3 => {
    const sg = Math.sign(vdot(moves[k], dir)) || 1;
    return [axes[k][0] * sg, axes[k][1] * sg, axes[k][2] * sg];
  };
  return { wristFlexAxis: signed(flexK, palmar), wristRadialAxis: signed(devK, radial) };
}

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
}

/**
 * The rig's rest hand as 21 MP_HAND-ordered points. Finger tips are the
 * distal bone's head pushed along its local Y (tips only matter as a
 * direction here).
 */
function restHandPoints(
  s: Side,
  head: (n: string) => Vec3,
  restR: Mat3[],
  bone: (n: string) => number,
): Vec3[] {
  const tip = (n: string): Vec3 => {
    const h = head(n), R = restR[bone(n)];
    return [h[0] + 0.02 * R[1], h[1] + 0.02 * R[4], h[2] + 0.02 * R[7]];
  };
  const p: Vec3[] = new Array(21);
  p[MP_HAND.WRIST] = head(`wrist.${s}`);
  p[MP_HAND.THUMB_CMC] = head(`finger1-1.${s}`);
  p[MP_HAND.THUMB_MCP] = head(`finger1-2.${s}`);
  p[MP_HAND.THUMB_IP] = head(`finger1-3.${s}`);
  p[MP_HAND.THUMB_TIP] = tip(`finger1-3.${s}`);
  const bases = [MP_HAND.INDEX_MCP, MP_HAND.MIDDLE_MCP, MP_HAND.RING_MCP, MP_HAND.PINKY_MCP];
  [2, 3, 4, 5].forEach((f, i) => {
    p[bases[i]] = head(`finger${f}-1.${s}`);
    p[bases[i] + 1] = head(`finger${f}-2.${s}`);
    p[bases[i] + 2] = head(`finger${f}-3.${s}`);
    p[bases[i] + 3] = tip(`finger${f}-3.${s}`);
  });
  return p;
}

