/** Tiny immutable vec3 / mat3 helpers for validation code (clarity over speed). */

export type V3 = [number, number, number];
/** Row-major 3×3. */
export type M3 = [number, number, number, number, number, number, number, number, number];

export const add = (a: V3, b: V3): V3 => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
export const sub = (a: V3, b: V3): V3 => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
export const scale = (a: V3, s: number): V3 => [a[0]*s, a[1]*s, a[2]*s];
export const dot = (a: V3, b: V3): number => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
export const cross = (a: V3, b: V3): V3 => [
  a[1]*b[2] - a[2]*b[1],
  a[2]*b[0] - a[0]*b[2],
  a[0]*b[1] - a[1]*b[0],
];
export const len = (a: V3): number => Math.hypot(a[0], a[1], a[2]);
export const norm = (a: V3): V3 => {
  const l = len(a);
  return l > 1e-12 ? [a[0]/l, a[1]/l, a[2]/l] : [0, 0, 0];
};
export const mid = (a: V3, b: V3): V3 => [(a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2];
export const neg = (a: V3): V3 => [-a[0], -a[1], -a[2]];

export const RAD2DEG = 180 / Math.PI;

/** Unsigned angle between two vectors, degrees. */
export function angleDeg(a: V3, b: V3): number {
  const d = dot(norm(a), norm(b));
  return Math.acos(Math.max(-1, Math.min(1, d))) * RAD2DEG;
}

/** Component of `v` orthogonal to unit `n`. */
export const reject = (v: V3, n: V3): V3 => sub(v, scale(n, dot(v, n)));

/** Signed angle (deg) from `a` to `b` about unit axis `axis`, both projected ⊥ axis. */
export function signedAngleDeg(a: V3, b: V3, axis: V3): number {
  const pa = reject(a, axis), pb = reject(b, axis);
  return Math.atan2(dot(cross(pa, pb), axis), dot(pa, pb)) * RAD2DEG;
}

export const mat3MulVec = (m: M3, v: V3): V3 => [
  m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
  m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
  m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
];

/** Matrix whose columns are a, b, c. */
export const fromCols = (a: V3, b: V3, c: V3): M3 => [
  a[0], b[0], c[0],
  a[1], b[1], c[1],
  a[2], b[2], c[2],
];

export const transpose = (m: M3): M3 => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];

export function mat3Mul(a: M3, b: M3): M3 {
  const o = new Array(9).fill(0) as M3;
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      o[r*3+c] = a[r*3]*b[c] + a[r*3+1]*b[3+c] + a[r*3+2]*b[6+c];
  return o;
}

/** Rotation angle (deg) of the relative rotation aᵀb. */
export function rotationAngleDeg(a: M3, b: M3): number {
  const r = mat3Mul(transpose(a), b);
  const c = (r[0] + r[4] + r[8] - 1) / 2;
  return Math.acos(Math.max(-1, Math.min(1, c))) * RAD2DEG;
}

/** Axis-angle → row-major rotation matrix (axis need not be unit). */
export function axisAngle(axis: V3, rad: number): M3 {
  const [x, y, z] = norm(axis);
  const c = Math.cos(rad), s = Math.sin(rad), t = 1 - c;
  return [
    t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
    t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
    t*x*z - s*y, t*y*z + s*x, t*z*z + c,
  ];
}
