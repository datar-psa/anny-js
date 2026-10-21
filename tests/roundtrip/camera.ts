/**
 * Pinhole cameras for re-rendering the driven model.
 *
 * MediaPipe world landmarks (pose AND hand) are expressed in axes parallel to
 * the source camera: +x right, +y down, +z into the scene. Anny world is
 * (+x anat-left, +y back, +z up), and `annyToMp` is exactly the rotation
 * between the two. So a camera with R = annyToMp looks at the driven model
 * from the same direction the source camera looked at the subject — the
 * re-render is directly comparable with the source image.
 */

import type { CameraSpec } from "./protocol.js";
import { dot, mat3MulVec, sub, type M3, type V3 } from "./vec.js";

/** Anny world → MP camera axes. */
export const R_FRONT: M3 = [1, 0, 0, 0, 0, -1, 0, 1, 0];

export function project(cam: CameraSpec, p: V3): [number, number, number] {
  const c = mat3MulVec(cam.R as M3, sub(p, cam.C));
  return [cam.fx * c[0] / c[2] + cam.cx, cam.fy * c[1] / c[2] + cam.cy, c[2]];
}

/**
 * Camera with rotation `R` that frames `points` with a `margin` fraction of
 * empty border, using a vertical field of view `fovYDeg`.
 */
export function fitCamera(
  points: V3[], W: number, H: number, R: M3 = R_FRONT, fovYDeg = 30, margin = 0.12,
): CameraSpec {
  const f = (H / 2) / Math.tan(fovYDeg * Math.PI / 360);
  // Work in camera-rotated coordinates (no translation yet).
  const q = points.map(p => mat3MulVec(R, p));
  const minX = Math.min(...q.map(p => p[0])), maxX = Math.max(...q.map(p => p[0]));
  const minY = Math.min(...q.map(p => p[1])), maxY = Math.max(...q.map(p => p[1]));
  const minZ = Math.min(...q.map(p => p[2]));
  const ctr: V3 = [(minX + maxX) / 2, (minY + maxY) / 2, 0];
  const halfW = (maxX - minX) / 2, halfH = (maxY - minY) / 2;
  const usable = 1 - 2 * margin;
  // Distance so the extents fit: half-extent / dist * f ≤ usable * half-image.
  const dist = Math.max(halfW * f / (usable * W / 2), halfH * f / (usable * H / 2), 0.05);
  // Camera centre in rotated coords sits `dist` in front of the nearest point plane.
  const camRot: V3 = [ctr[0], ctr[1], minZ - dist];
  // Back to world: C = Rᵀ · camRot.
  const C: V3 = [
    dot([R[0], R[3], R[6]], camRot),
    dot([R[1], R[4], R[7]], camRot),
    dot([R[2], R[5], R[8]], camRot),
  ];
  return { W, H, fx: f, fy: f, cx: W / 2, cy: H / 2, R: [...R], C };
}
