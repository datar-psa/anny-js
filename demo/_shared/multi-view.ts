/**
 * The 4-view layout used by every demo: FRONT, BACK, 3/4, ABOVE.
 *
 * Each view is one quadrant of the canvas. The body is anchored to the floor
 * grid in its quadrant (footY = grid Z=0 plus the model's negative-Z hang),
 * and the ABOVE view zooms out to fit a bird's-eye angle.
 */

import type { ViewParams, WebGLBundle } from "./webgl.js";

export interface QuadrantLayout {
  W: number;
  H: number;
  /** Pixels per metre of body height; same for all body views in the top row. */
  scale: number;
  /** Same vertical scale as `scale` but pre-multiplied by 0.6 for the ABOVE view. */
  aboveScale: number;
  /** Foot-level Z offset (= `model.restMinZ * scale`). Below 0 if rest mesh dips. */
  footOffset: number;
  /** ABOVE-view foot offset (scaled accordingly). */
  aboveFootOffset: number;
  /** Computed grid + body Y coordinates per quadrant. */
  fyTopGrid: number;
  fyBotGrid: number;
  bodyTopY: number;
  bodyBotY: number;
  /** ABOVE view foot Y (centered vertically in bottom-right quadrant). */
  aboveBodyY: number;
}

/** Compute pixel coordinates for the 4-view layout given canvas + scale + foot dip. */
export function computeQuadrants(
  W: number,
  H: number,
  rawScale: number,
  minRestZ: number,
): QuadrantLayout {
  const HH = H / 2;
  // Pick a sane scale: bound by the quadrant height (body must fit) and the raw value.
  const scale = Math.min(rawScale * 0.42, HH * 0.82 / 1.7);
  const bodyH = 1.7 * scale;
  const footOffset = minRestZ * scale;

  const fyTopGrid  = HH * 0.1 + bodyH;
  const fyBotGrid  = HH + HH * 0.1 + bodyH;

  return {
    W, H,
    scale,
    aboveScale: scale * 0.6,
    footOffset,
    aboveFootOffset: minRestZ * scale * 0.6,
    fyTopGrid,
    fyBotGrid,
    bodyTopY:   fyTopGrid + footOffset,
    bodyBotY:   fyBotGrid + footOffset,
    aboveBodyY: HH + HH * 0.55 + minRestZ * scale * 0.6,
  };
}

export const VIEW_COLORS = {
  FRONT: [0.88, 0.90, 0.95] as [number, number, number],
  BACK:  [0.95, 0.90, 0.85] as [number, number, number],
  THREE_QUARTER: [0.88, 0.95, 0.90] as [number, number, number],
  ABOVE: [0.92, 0.88, 0.96] as [number, number, number],
};

export interface FourViewLayout {
  body: { front: ViewParams; back: ViewParams; threeQuarter: ViewParams; above: ViewParams };
  grid: { front: ViewParams; back: ViewParams; threeQuarter: ViewParams; above: ViewParams };
}

export interface FourViewOptions {
  /** 3/4 view orbit angle (radians). live uses 0.35π, anim uses 0.4π. */
  threeQuarterAngle?: number;
  /** ABOVE view scale factor relative to body scale. live uses 0.6, anim uses 0.65. */
  aboveScaleFactor?: number;
  /** Vertical pixel offset added to every body footY (negative = lifts body up). */
  bodyYOffset?: number;
}

/**
 * Build the standard FRONT / BACK / 3-4 / ABOVE viewparams for a quadrant layout.
 * Demos can mutate the returned object before calling `drawFourViews` (e.g. anim
 * subtracts a jump arc from each body.footY).
 */
export function buildFourViews(q: QuadrantLayout, opts: FourViewOptions = {}): FourViewLayout {
  const HW = q.W / 2;
  const HH = q.H / 2;
  const threeQuarterAngle = opts.threeQuarterAngle ?? Math.PI * 0.35;
  const aboveScaleFactor = opts.aboveScaleFactor ?? 0.6;
  const bodyYOffset = opts.bodyYOffset ?? 0;

  const aboveScale = q.scale * aboveScaleFactor;
  const aboveFootOff = q.footOffset * aboveScaleFactor;
  const aboveBodyY = HH + HH * 0.55 + aboveFootOff;

  // Third-person rendering: anatomical-left lands on screen-right when the
  // subject faces camera (FRONT), screen-left when they face away (BACK) —
  // matching how a third-party camera would film the same subject. This is
  // unconditional in the pipeline now; the legacy "self-mirror" mode was
  // removed because comparing a self-mirrored model with third-person video
  // footage causes a confusing left/right swap.
  const body = {
    front:        { cx: HW * 0.5, footY: q.bodyTopY + bodyYOffset, scale: q.scale },
    back:         { cx: HW * 1.5, footY: q.bodyTopY + bodyYOffset, scale: q.scale, camY: Math.PI },
    threeQuarter: { cx: HW * 0.5, footY: q.bodyBotY + bodyYOffset, scale: q.scale, camY: threeQuarterAngle },
    above:        { cx: HW * 1.5, footY: aboveBodyY + bodyYOffset, scale: aboveScale, camX: 0.85 },
  };
  const grid = {
    front:        { ...body.front,        footY: q.fyTopGrid },
    back:         { ...body.back,         footY: q.fyTopGrid },
    threeQuarter: { ...body.threeQuarter, footY: q.fyBotGrid },
    above:        { ...body.above,        footY: HH + HH * 0.55 },
  };
  return { body, grid };
}

/**
 * Render a 4-view layout. Assumes the caller already uploaded posed vertices,
 * cleared the framebuffer, and (if needed) set world twist.
 */
export function drawFourViews(gl: WebGLBundle, views: FourViewLayout): void {
  gl.drawGrid(views.grid.front);        gl.drawBody(views.body.front,        VIEW_COLORS.FRONT);
  gl.drawGrid(views.grid.back);         gl.drawBody(views.body.back,         VIEW_COLORS.BACK);
  gl.drawGrid(views.grid.threeQuarter); gl.drawBody(views.body.threeQuarter, VIEW_COLORS.THREE_QUARTER);
  gl.drawGrid(views.grid.above);        gl.drawBody(views.body.above,        VIEW_COLORS.ABOVE);
}

/** Draw 2D quadrant dividers + labels on the overlay canvas. */
export function drawDividers(ctx: CanvasRenderingContext2D, W: number, H: number): void {
  const HW = W / 2, HH = H / 2;
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(HW, 0); ctx.lineTo(HW, H);
  ctx.moveTo(0, HH); ctx.lineTo(W, HH);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "11px monospace";
  ctx.fillText("FRONT", 8, 16);
  ctx.fillText("BACK", HW + 8, 16);
  ctx.fillText("3/4",  8, HH + 16);
  ctx.fillText("ABOVE", HW + 8, HH + 16);
}

// ── Side-by-side video|model layout ───────────────────────────────────────
//
// Mirrors the Meshcapade homepage comparison: source video on the left half,
// Anny model on the right half, both at the same vertical scale, model camera
// fixed (no orbit) so that as the subject rotates in the video, the model
// rotates with it — the most direct "is orientation tracking working?" view.

export interface SplitViewLayout {
  body: ViewParams;
  grid: ViewParams;
}

/**
 * Build a single FRONT-facing view sized to fill the right half of the
 * canvas. The video element is expected to occupy the left half (via CSS in
 * the demo HTML). Camera does NOT orbit — that's the whole point: a 90°
 * world-space turn by the subject should produce a 90° turn in the model.
 */
export function buildSplitView(
  W: number, H: number, minRestZ: number,
): SplitViewLayout {
  const rightCenterX = W * 0.75;
  const scale = 0.7 * H / 1.7;
  const bodyH = 1.7 * scale;
  const footOffset = minRestZ * scale;
  const footY = (H + bodyH) / 2 + footOffset;

  const body: ViewParams = { cx: rightCenterX, footY, scale };
  const grid: ViewParams = { ...body, footY: footY - footOffset };
  return { body, grid };
}

/**
 * Render the side-by-side layout into the GL canvas (right half only).
 * Caller is responsible for clearing the framebuffer and uploading vertices.
 */
export function drawSplitView(gl: WebGLBundle, view: SplitViewLayout): void {
  gl.drawGrid(view.grid);
  gl.drawBody(view.body, VIEW_COLORS.FRONT);
}

/** Draw a vertical divider + side labels for the split layout. */
export function drawSplitDivider(
  ctx: CanvasRenderingContext2D, W: number, H: number,
): void {
  const HW = W / 2;
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(HW, 0); ctx.lineTo(HW, H);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "11px monospace";
  ctx.fillText("VIDEO", 8, 16);
  ctx.fillText("MODEL", HW + 8, 16);
}
