/**
 * Thumb–finger clearance.
 *
 * A thumb that MediaPipe reads a few degrees off — routine on a small hand —
 * ends up *inside* the index finger as soon as the fingers curl, because in
 * a fist the two are in contact. Real thumbs rest on the fingers instead.
 * After the solve, phalanges are treated as capsules (radii measured from
 * the mesh at load time) and the whole thumb is swung out about its base
 * until it clears the fingers it can wrap over.
 */

import type { AnnyModel } from "../anny/types.js";
import {
  axisAngle, mmul, mT, mvec, vadd, vcross, vdot, vlen, vnorm, vscale, vsub,
  type Mat3, type Vec3,
} from "./geom.js";

/** A phalanx: segment from the bone head along `end` (bone-local), inflated by `radius` (m). */
export interface Capsule {
  bone: number;
  end: Vec3;
  radius: number;
}

export interface ThumbClearanceRig {
  /** Thumb metacarpal (finger1-1): the whole thumb swings about its head. */
  base: number;
  /** Thumb proximal + distal phalanges. */
  thumb: Capsule[];
  /** Index, middle and ring finger phalanges (a thumb wraps over them in a fist). */
  fingers: Capsule[];
}

/**
 * Radius of a phalanx: 0.8 × the median distance of its vertices from the
 * axis — the median overestimates the collision core at the flat pads.
 */
const RADIUS_SCALE = 0.8;

/**
 * Capsule for `bone` from the mesh: axis from its head to `child`'s head (or,
 * for a distal phalanx, to the end of the vertices it carries, pulled in by
 * the radius so the end cap stays inside the skin).
 */
export function phalanxCapsule(
  model: AnnyModel, bone: number, child: number | null, restR: Mat3[], restHead: Vec3[],
): Capsule {
  const K = model.maxBonesPerVert;
  const R = restR[bone], head = restHead[bone];
  const axisLocal: Vec3 = child !== null ? mvec(mT(R), vsub(restHead[child], head)) : [0, 1, 0];
  const dir = vnorm(axisLocal);
  const dists: number[] = [];
  let reach = 0;
  for (let v = 0; v < model.vertCount; v++) {
    let best = -1, wBest = 0;
    for (let k = 0; k < K; k++) {
      const w = model.boneWeights[v * K + k];
      if (w > wBest) { wBest = w; best = model.boneIndices[v * K + k]; }
    }
    if (best !== bone) continue;
    const p = mvec(mT(R), vsub([model.restVertices[v * 3], model.restVertices[v * 3 + 1], model.restVertices[v * 3 + 2]], head));
    const along = vdot(p, dir);
    reach = Math.max(reach, along);
    dists.push(vlen(vsub(p, vscale(dir, along))));
  }
  dists.sort((a, b) => a - b);
  const radius = dists.length ? RADIUS_SCALE * dists[dists.length >> 1] : 0.006;
  const end = child !== null ? axisLocal : vscale(dir, Math.max(radius, reach - radius));
  return { bone, end, radius };
}

/** Closest points between segments p0p1 and q0q1. */
function closestPoints(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3): [Vec3, Vec3] {
  const d1 = vsub(p1, p0), d2 = vsub(q1, q0), r = vsub(p0, q0);
  const a = vdot(d1, d1), e = vdot(d2, d2), f = vdot(d2, r), c = vdot(d1, r), b = vdot(d1, d2);
  const den = a * e - b * b;
  let s = den > 1e-12 ? Math.min(1, Math.max(0, (b * f - c * e) / den)) : 0;
  let t = e > 1e-12 ? (b * s + f) / e : 0;
  if (t < 0) { t = 0; s = a > 1e-12 ? Math.min(1, Math.max(0, -c / a)) : 0; }
  else if (t > 1) { t = 1; s = a > 1e-12 ? Math.min(1, Math.max(0, (b - c) / a)) : 0; }
  return [vadd(p0, vscale(d1, s)), vadd(q0, vscale(d2, t))];
}

/** Clearance kept between the capsules (m): touching is fine, overlapping isn't. */
const SLACK = 0;
/** Largest swing considered (rad): past this the reading is wrong, not just close. */
const MAX_SWING = 70 * Math.PI / 180;
const SWING_STEP = 3 * Math.PI / 180;
/** Swing axes tried, spread around the plane ⊥ the thumb's metacarpal. */
const AXES = 16;

/**
 * Swing the thumb (in `world`, in place) until it clears the fingers.
 * Returns the thumb base's new local delta, or null if it was already clear.
 *
 * A small global search rather than pushing away from the deepest contact:
 * a thumb tucked between fingers would be pushed from one into the next.
 * Candidates are rotations of the whole thumb about its base; the winner
 * minimises squared penetration first, then the size of the swing.
 *
 * @param world     World rotations from the solve (mutated for the thumb bones).
 * @param parents   Bone parents.
 * @param restR     Rest world rotations.
 * @param restHead  Rest head positions.
 */
export function clearThumb(
  rig: ThumbClearanceRig, world: Mat3[], parents: Int32Array, restR: Mat3[], restHead: Vec3[],
): Mat3 | null {
  // Positions relative to an arbitrary origin: only differences matter.
  const pos = new Map<number, Vec3>();
  const posOf = (b: number): Vec3 => {
    const hit = pos.get(b);
    if (hit) return hit;
    const p = parents[b];
    const out: Vec3 = p < 0 || !world[p] ? restHead[b]
      : vadd(posOf(p), mvec(mmul(world[p], mT(restR[p])), vsub(restHead[b], restHead[p])));
    pos.set(b, out);
    return out;
  };
  const seg = (c: Capsule): [Vec3, Vec3] => {
    const a = posOf(c.bone);
    return [a, vadd(a, mvec(world[c.bone], c.end))];
  };
  const base = posOf(rig.base);
  const thumb = rig.thumb.map(c => seg(c).map(p => vsub(p, base)) as [Vec3, Vec3]);
  const fingers = rig.fingers.map(seg);
  const penetration = (R: Mat3 | null): number => {
    let sum = 0;
    rig.thumb.forEach((t, i) => {
      const [t0, t1] = thumb[i].map(p => vadd(base, R ? mvec(R, p) : p));
      rig.fingers.forEach((f, j) => {
        const [pt, pf] = closestPoints(t0, t1, fingers[j][0], fingers[j][1]);
        const d = t.radius + f.radius + SLACK - vlen(vsub(pt, pf));
        if (d > 0) sum += d * d;
      });
    });
    return sum;
  };
  if (penetration(null) === 0) return null;

  // Axes ⊥ the thumb's metacarpal direction.
  const meta = vnorm(mvec(world[rig.base], [0, 1, 0]));
  const u = vnorm(vcross(meta, Math.abs(meta[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  const v = vcross(meta, u);
  let best: { axis: Vec3; angle: number; cost: number } | undefined;
  const PENALTY = 1e7;   // per m² of penetration — any clearance beats any swing
  for (let k = 0; k < AXES; k++) {
    const t = (2 * Math.PI * k) / AXES;
    const axis = vadd(vscale(u, Math.cos(t)), vscale(v, Math.sin(t)));
    for (let a = SWING_STEP; a <= MAX_SWING + 1e-9; a += SWING_STEP) {
      const pen = penetration(axisAngle(axis, a));
      const cost = PENALTY * pen + a * a;
      if (!best || cost < best.cost) best = { axis, angle: a, cost };
      if (pen === 0) break;   // larger swings on this axis only cost more
    }
  }
  if (!best) return null;
  // The grid overshoots by up to a step: bisect for the smallest clearing swing.
  if (penetration(axisAngle(best.axis, best.angle)) === 0) {
    let lo = best.angle - SWING_STEP, hi = best.angle;
    for (let i = 0; i < 8; i++) {
      const mid = (lo + hi) / 2;
      if (penetration(axisAngle(best.axis, mid)) === 0) hi = mid; else lo = mid;
    }
    best.angle = hi;
  }
  const R = axisAngle(best.axis, best.angle);
  for (const b of [rig.base, ...rig.thumb.map(c => c.bone)]) world[b] = mmul(R, world[b]);
  const p = parents[rig.base];
  const chain = mmul(mmul(world[p], mT(restR[p])), restR[rig.base]);
  return mmul(mT(chain), world[rig.base]);
}
