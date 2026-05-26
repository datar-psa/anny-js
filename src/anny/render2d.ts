/**
 * Orthographic 2D projection + canvas renderer for Anny mesh.
 *
 * Projects the 3D mesh onto a canvas using a simple orthographic projection,
 * suitable for a 2D fighting game (side view, Y-up → Y-down canvas).
 *
 * The mesh is drawn as filled triangles sorted front-to-back (painter's
 * algorithm on Z depth — fast enough at 14K verts, 27K triangles).
 */

import type { SkinnedMesh } from "./types.js";

export interface RenderOptions {
  /** Canvas X center for the character (pixels). */
  cx: number;
  /** Canvas Y for the character's feet. */
  footY: number;
  /** Scale: canvas pixels per metre of body height. */
  scale: number;
  /** Fill color for the body mesh. */
  color: string;
  /** Optional outline color. Omit or set "" to skip. */
  outlineColor?: string;
  /**
   * Rotation around the vertical (Z) axis in radians (default 0).
   * Positive = Anny's left side swings toward camera (figure appears to turn right).
   */
  viewRotY?: number;
  /**
   * Rotation around the lateral (X) axis in radians (default 0).
   * Negative = tilt camera upward (bird's-eye view).  e.g. -Math.PI/2 = top-down.
   */
  viewRotX?: number;
}

/**
 * Render a skinned Anny mesh onto a Canvas 2D context.
 *
 * Projection: orthographic, side view.
 *   canvas_x = cx + vert_x * scale
 *   canvas_y = footY - vert_z * scale   (Z is up in Anny space)
 *   depth    = vert_y                   (Y is forward/backward in Anny space)
 *
 * Triangles are depth-sorted (painter's algorithm).
 */
export function renderAnny(
  ctx: CanvasRenderingContext2D,
  mesh: SkinnedMesh,
  opts: RenderOptions
): void {
  const { vertices, faces } = mesh;
  const { cx, footY, scale, color, outlineColor, viewRotY = 0, viewRotX = 0 } = opts;
  const cosR = Math.cos(viewRotY), sinR = Math.sin(viewRotY);
  const cosX = Math.cos(viewRotX), sinX = Math.sin(viewRotX);
  const faceCount = faces.length / 3;

  // No depth sort — back-face culling handles occlusion for closed meshes.

  // Batched render: chunk triangles into paths of BATCH_SIZE each.
  // ~26 fill() calls/view instead of 13K — fast on both GPU and CPU.
  const BATCH_SIZE = 14000;  // one fill() per view — 4 GPU draw calls for 4 views
  ctx.save();
  ctx.fillStyle = color;

  let batchN = 0;
  ctx.beginPath();
  for (let f = 0; f < faceCount; f++) {
    const i0 = faces[f * 3    ] * 3;
    const i1 = faces[f * 3 + 1] * 3;
    const i2 = faces[f * 3 + 2] * 3;
    const vx0 = vertices[i0], vy0 = vertices[i0+1], vz0 = vertices[i0+2];
    const vx1 = vertices[i1], vy1 = vertices[i1+1], vz1 = vertices[i1+2];
    const vx2 = vertices[i2], vy2 = vertices[i2+1], vz2 = vertices[i2+2];

    const rx0 = vx0*cosR - vy0*sinR, ry0 = vx0*sinR + vy0*cosR;
    const rx1 = vx1*cosR - vy1*sinR, ry1 = vx1*sinR + vy1*cosR;
    const rx2 = vx2*cosR - vy2*sinR, ry2 = vx2*sinR + vy2*cosR;
    const rz0 = ry0*sinX + vz0*cosX;
    const rz1 = ry1*sinX + vz1*cosX;
    const rz2 = ry2*sinX + vz2*cosX;
    const px0 = cx + rx0 * scale, py0 = footY - rz0*scale;
    const px1 = cx + rx1 * scale, py1 = footY - rz1*scale;
    const px2 = cx + rx2 * scale, py2 = footY - rz2*scale;

    // Back-face cull (third-person projection: anatomical-left renders on
    // screen right when subject faces camera; CCW triangles are front-faces).
    const cross2d = (px1-px0)*(py2-py0) - (py1-py0)*(px2-px0);
    if (cross2d >= 0) continue;

    ctx.moveTo(px0, py0);
    ctx.lineTo(px1, py1);
    ctx.lineTo(px2, py2);
    ctx.closePath();

    if (++batchN === BATCH_SIZE) {
      ctx.fill();
      ctx.beginPath();
      batchN = 0;
    }
  }
  if (batchN > 0) ctx.fill();

  if (outlineColor) {
    ctx.strokeStyle = outlineColor;
    ctx.lineWidth = 0.4;
    ctx.stroke();  // outline on last partial path only — acceptable for wireframe hint
  }
  ctx.restore();
}
