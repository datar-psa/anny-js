/** Minimal mat4 helpers — all operate on flat Float32Array, row-major. */

export function mat4Identity(): Float32Array {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** Multiply two 4×4 row-major matrices: out = a × b */
export function mat4Mul(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  out ??= new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = s;
    }
  }
  return out;
}

/**
 * Invert a rigid-body 4×4 matrix (rotation + translation only, no scale).
 * inv(R|t) = R^T | -R^T t
 */
export function mat4InvertRigid(m: Float32Array, out?: Float32Array): Float32Array {
  out ??= new Float32Array(16);
  // Transpose 3×3 rotation block
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      out[r * 4 + c] = m[c * 4 + r];
  // -R^T * t
  const tx = m[3], ty = m[7], tz = m[11];
  out[3]  = -(out[0]*tx + out[1]*ty + out[2]*tz);
  out[7]  = -(out[4]*tx + out[5]*ty + out[6]*tz);
  out[11] = -(out[8]*tx + out[9]*ty + out[10]*tz);
  out[12] = out[13] = out[14] = 0;
  out[15] = 1;
  return out;
}

/** Build a 4×4 from a 3×3 rotation (row-major) + translation (3-vec). */
export function mat4FromRot3Translation(
  r: Float32Array,
  tx: number, ty: number, tz: number,
  out?: Float32Array
): Float32Array {
  out ??= new Float32Array(16);
  out[0] = r[0]; out[1] = r[1]; out[2] = r[2]; out[3] = tx;
  out[4] = r[3]; out[5] = r[4]; out[6] = r[5]; out[7] = ty;
  out[8] = r[6]; out[9] = r[7]; out[10]= r[8]; out[11]= tz;
  out[12]= 0;    out[13]= 0;    out[14]= 0;    out[15]= 1;
  return out;
}

/** Slice a 4×4 sub-matrix out of a flat buffer at byte offset `boneIdx * 16`. */
export function mat4Slice(buf: Float32Array, boneIdx: number): Float32Array {
  return buf.subarray(boneIdx * 16, boneIdx * 16 + 16);
}

/** Multiply two 3×3 row-major matrices: out = a × b */
export function mulMat3(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[r*3+k] * b[k*3+c];
      out[r*3+c] = s;
    }
  return out;
}

/** Cross product of two 3-vectors. */
export function cross(a: Float32Array, b: Float32Array): Float32Array {
  return new Float32Array([
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ]);
}

/** Normalize a 3-vector in-place; returns it. */
export function normalize3(v: Float32Array): Float32Array {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len > 1e-8) { v[0] /= len; v[1] /= len; v[2] /= len; }
  return v;
}

/** Dot product of two 3-vectors. */
export function dot3(a: Float32Array, b: Float32Array): number {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * Build the smallest-angle rotation matrix that takes `from` unit-vec to `to`
 * unit-vec.  Returns a row-major 3×3 Float32Array.
 *
 * Pure geometric mapping with no angle clamp — callers that need to bound
 * the magnitude (e.g. the MediaPipe driver, where bad landmarks can produce
 * mesh-collapsing rotations) should clamp the result themselves.
 */
export function rotFromTo(from: Float32Array, to: Float32Array): Float32Array {
  const axis = cross(from, to);
  const sinA = Math.hypot(axis[0], axis[1], axis[2]);
  const cosA = dot3(from, to);

  if (sinA < 1e-8) {
    if (cosA > 0) return new Float32Array([1,0,0, 0,1,0, 0,0,1]);
    // Antiparallel: 180° around any axis ⟂ `from`.
    const perp = Math.abs(from[0]) < 0.9
      ? new Float32Array([1,0,0])
      : new Float32Array([0,1,0]);
    return rodriguesToMat3(normalize3(cross(from, perp)), Math.PI);
  }

  normalize3(axis);
  return rodriguesToMat3(axis, Math.atan2(sinA, cosA));
}

/** Rodrigues rotation matrix from axis (unit) + angle (radians). Row-major 3×3. */
export function rodriguesToMat3(axis: Float32Array, angle: number): Float32Array {
  const [ax, ay, az] = axis;
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return new Float32Array([
    t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay,
    t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax,
    t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c,
  ]);
}
