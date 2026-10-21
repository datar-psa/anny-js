/**
 * Float64 vector / rotation helpers for the MediaPipe driver.
 *
 * Rotations are row-major 3×3 (`Mat3`) or unit quaternions [x, y, z, w]
 * (`Quat`) — quaternions only where we need slerp or swing/twist splits.
 * Plain tuples keep the solver readable; the driver touches ~60 bones per
 * frame so allocation cost is negligible next to LBS.
 */

export type Vec3 = [number, number, number];
export type Mat3 = [number, number, number, number, number, number, number, number, number];
export type Quat = [number, number, number, number];

export const DEG = Math.PI / 180;

export const vsub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vadd = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vscale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const vdot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vcross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const vlen = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export const vmid = (a: Vec3, b: Vec3): Vec3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
export const vneg = (a: Vec3): Vec3 => [-a[0], -a[1], -a[2]];

export function vnorm(a: Vec3): Vec3 {
  const l = vlen(a);
  return l > 1e-12 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}

/** Component of `v` orthogonal to unit `n`. */
export const vreject = (v: Vec3, n: Vec3): Vec3 => vsub(v, vscale(n, vdot(v, n)));

/** Unsigned angle between two vectors (radians). */
export function vangle(a: Vec3, b: Vec3): number {
  return Math.atan2(vlen(vcross(a, b)), vdot(a, b));
}

/** Signed angle from `a` to `b` about unit `axis`, both projected ⊥ axis (radians). */
export function signedAngle(a: Vec3, b: Vec3, axis: Vec3): number {
  const pa = vreject(a, axis), pb = vreject(b, axis);
  return Math.atan2(vdot(vcross(pa, pb), axis), vdot(pa, pb));
}

export const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/** 0 below `lo`, 1 above `hi`, smooth in between. */
export function smoothstep(x: number, lo: number, hi: number): number {
  const t = clamp((x - lo) / (hi - lo), 0, 1);
  return t * t * (3 - 2 * t);
}

// ── Mat3 ───────────────────────────────────────────────────────────────────

export const IDENTITY: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mmul(a: Mat3, b: Mat3): Mat3 {
  return [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
  ];
}

export const mT = (m: Mat3): Mat3 => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];

export const mvec = (m: Mat3, v: Vec3): Vec3 => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
  m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
];

/** Matrix with columns a, b, c. */
export const fromCols = (a: Vec3, b: Vec3, c: Vec3): Mat3 => [
  a[0], b[0], c[0],
  a[1], b[1], c[1],
  a[2], b[2], c[2],
];

/**
 * Right-handed orthonormal frame with columns (X, Y, Z): Y along `y`, X
 * along `x` made orthogonal to Y, Z = X × Y.
 */
export function frameXY(x: Vec3, y: Vec3): Mat3 {
  const Y = vnorm(y);
  const X = vnorm(vreject(x, Y));
  return fromCols(X, Y, vcross(X, Y));
}

/**
 * Right-handed orthonormal frame with columns (X, Y, Z): X exactly along
 * `x`, Y along `y` made orthogonal to X, Z = X × Y.
 */
export function frameXFirst(x: Vec3, y: Vec3): Mat3 {
  const X = vnorm(x);
  const Y = vnorm(vreject(y, X));
  return fromCols(X, Y, vcross(X, Y));
}

/** Rotation about `axis` (need not be unit) by `angle` radians. */
export function axisAngle(axis: Vec3, angle: number): Mat3 {
  const [x, y, z] = vnorm(axis);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ];
}

/** Smallest rotation taking unit `from` onto unit `to`. */
export function rotFromTo(from: Vec3, to: Vec3): Mat3 {
  const axis = vcross(from, to);
  const s = vlen(axis), c = vdot(from, to);
  if (s < 1e-9) {
    if (c > 0) return [...IDENTITY];
    const perp = Math.abs(from[0]) < 0.9 ? vcross(from, [1, 0, 0]) : vcross(from, [0, 1, 0]);
    return axisAngle(perp, Math.PI);
  }
  return axisAngle(axis, Math.atan2(s, c));
}

/** Rotation angle of a rotation matrix (radians). */
export function rotAngle(m: Mat3): number {
  return Math.acos(clamp((m[0] + m[4] + m[8] - 1) / 2, -1, 1));
}

// ── Quaternions ────────────────────────────────────────────────────────────

export function quatFromMat3(m: Mat3): Quat {
  const tr = m[0] + m[4] + m[8];
  let q: Quat;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [(m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s, 0.25 * s];
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
    q = [0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s, (m[7] - m[5]) / s];
  } else if (m[4] > m[8]) {
    const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
    q = [(m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s, (m[2] - m[6]) / s];
  } else {
    const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
    q = [(m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s, (m[3] - m[1]) / s];
  }
  return qnorm(q);
}

export function mat3FromQuat(q: Quat): Mat3 {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

export function qnorm(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

export function qslerp(a: Quat, b: Quat, t: number): Quat {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb: Quat = d < 0 ? [-b[0], -b[1], -b[2], -b[3]] : b;
  d = Math.abs(d);
  if (d > 0.9995) {
    return qnorm([
      a[0] + t * (bb[0] - a[0]), a[1] + t * (bb[1] - a[1]),
      a[2] + t * (bb[2] - a[2]), a[3] + t * (bb[3] - a[3]),
    ]);
  }
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [wa * a[0] + wb * bb[0], wa * a[1] + wb * bb[1], wa * a[2] + wb * bb[2], wa * a[3] + wb * bb[3]];
}

/** Rotation-matrix slerp: identity-at-0 → `b` at 1 when `a` is identity. */
export function mslerp(a: Mat3, b: Mat3, t: number): Mat3 {
  return mat3FromQuat(qslerp(quatFromMat3(a), quatFromMat3(b), t));
}

/**
 * Twist angle of rotation `m` about unit `axis` (swing–twist decomposition,
 * m = swing · twist). Radians in (−π, π].
 */
export function twistAngle(m: Mat3, axis: Vec3): number {
  const q = quatFromMat3(m);
  const p = q[0] * axis[0] + q[1] * axis[1] + q[2] * axis[2];
  return 2 * Math.atan2(p, q[3]);
}

/** Rotation vector (axis × angle, radians) of a rotation matrix. */
export function rotVec(m: Mat3): Vec3 {
  const q = quatFromMat3(m);
  const s = Math.hypot(q[0], q[1], q[2]);
  if (s < 1e-12) return [0, 0, 0];
  const angle = 2 * Math.atan2(s, Math.abs(q[3])) * (q[3] < 0 ? -1 : 1);
  return [q[0] / s * angle, q[1] / s * angle, q[2] / s * angle];
}

/** Rotation matrix of a rotation vector (axis × angle, radians). */
export function fromRotVec(r: Vec3): Mat3 {
  const a = vlen(r);
  return a < 1e-12 ? IDENTITY : axisAngle(vscale(r, 1 / a), a);
}
