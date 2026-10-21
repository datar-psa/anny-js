/**
 * The round trip:  image ─MP→ landmarks ─driver→ Anny pose ─render→ image' ─MP→ landmarks'
 *
 * For every case we compare three landmark sources with the same metrics:
 *   src    MediaPipe on the source image
 *   model  the driven rig's own joints (what the driver actually produced)
 *   rt     MediaPipe on a render of the driven rig (the user-visible result)
 *
 *   src↔model  = driver fidelity (does the rig do what MP saw?)
 *   src↔rt     = end-to-end round trip (does the render LOOK like the source?)
 *   model↔rt   = MediaPipe's own error on our renders (the noise floor)
 *
 * Body renders use a camera with the source camera's orientation (MP world
 * landmarks are camera-aligned), hand close-ups use the same orientation, so
 * direction metrics are comparable in camera axes.
 */

import {
  allocBoneTransforms, buildBoneIndex, forwardKinematics, lbs,
  type AnnyModel, type PoseDeltas,
} from "../../src/anny/index.js";
import {
  handLandmarks3D, handsForDriver, matchHandChirality, type PoseInput, type WorldLandmark,
} from "../../src/mediapipe/index.js";
import { fitCamera, project } from "./camera.js";
import type { HarnessClient } from "./client.js";
import {
  bodyDirections, compare2D, compareBody, compareHands, geometricSide, handAngles, toPx,
  type HandAngles, type HandDiff, type P2, type Side,
} from "./metrics.js";
import type { Detection, HandDetection } from "./protocol.js";
import {
  buildRigExtras, handJointsAnny, PosedSkeleton, poseJointsAnny, toMpWorld,
  type RigExtras,
} from "./skeleton.js";
import type { V3 } from "./vec.js";

export type Driver = (input: PoseInput, model: AnnyModel, boneIndex: Map<string, number>) => PoseDeltas;

export interface HandResult {
  side: Side;
  /** Source detection metadata. */
  labelSide: Side;
  geomSide: Side | undefined;
  src: HandAngles;
  model: HandAngles;
  rt?: HandAngles;
  srcVsModel: HandDiff;
  srcVsRt?: HandDiff;
  modelVsRt?: HandDiff;
  /** 2D landmark residuals (fraction of palm length), see `compare2D`. */
  srcVsModel2D?: number;
  srcVsRt2D?: number;
  modelVsRt2D?: number;
  renderPng?: string;
  rtDetection?: Detection;
}

export interface CaseResult {
  name: string;
  detection: Detection;
  body?: {
    srcVsModel: Record<string, number>;
    srcVsRt?: Record<string, number>;
    modelVsRt?: Record<string, number>;
    rtDetection?: Detection;
    renderPng: string;
  };
  hands: HandResult[];
  /** Why parts of the case were skipped, if any. */
  notes: string[];
}

export interface RunContext {
  client: HarnessClient;
  model: AnnyModel;
  boneIndex: Map<string, number>;
  extras: RigExtras;
  /** Vertices skinned mostly to each hand (wrist subtree) — for framing close-ups. */
  handVerts: Record<Side, number[]>;
  /** Which MediaPipe hand output feeds the driver and the 3D hand metrics. */
  handSource: HandSource;
}

/** "image": image landmarks lifted with `handLandmarks3D`; "world": `worldLandmarks`. */
export type HandSource = "image" | "world";

export function makeContext(client: HarnessClient, model: AnnyModel, handSource: HandSource = "image"): RunContext {
  return {
    client, model, boneIndex: buildBoneIndex(model), extras: buildRigExtras(model),
    handVerts: { L: handVertices(model, "L"), R: handVertices(model, "R") },
    handSource,
  };
}

function handVertices(model: AnnyModel, side: Side): number[] {
  const wrist = model.boneLabels.indexOf(`wrist.${side}`);
  const inHand = Array.from(model.boneParents, (_, b) => {
    let p = b;
    while (p >= 0 && p !== wrist) p = model.boneParents[p];
    return p === wrist;
  });
  const K = model.maxBonesPerVert, out: number[] = [];
  for (let v = 0; v < model.vertCount; v++) {
    let w = 0;
    for (let k = 0; k < K; k++) if (inHand[model.boneIndices[v * K + k]]) w += model.boneWeights[v * K + k];
    if (w > 0.5) out.push(v);
  }
  return out;
}

/** Close-up camera (source-camera orientation) framing the skinned hand mesh. */
export function handCamera(ctx: RunContext, deltas: PoseDeltas, side: Side, size = 384) {
  const bx = allocBoneTransforms(ctx.model.boneCount);
  forwardKinematics(ctx.model, deltas, bx);
  const { vertices } = lbs(ctx.model, bx);
  const pts: V3[] = ctx.handVerts[side].map(v => [vertices[v * 3], vertices[v * 3 + 1], vertices[v * 3 + 2]]);
  return fitCamera(pts, size, size, undefined, 30, 0.15);
}

function posed(ctx: RunContext, deltas: PoseDeltas): PosedSkeleton {
  const bx = allocBoneTransforms(ctx.model.boneCount);
  forwardKinematics(ctx.model, deltas, bx);
  return new PosedSkeleton(ctx.model, ctx.extras, bx);
}

/** All bone heads — frames the whole body including fingers and head. */
function allJoints(ctx: RunContext, sk: PosedSkeleton): V3[] {
  return ctx.model.boneLabels.map(n => sk.head(n));
}

/**
 * 3D points of a detected hand (from `source`) with chirality matched to the
 * side it will drive. `W`/`H` are the pixel size of the detection's image.
 */
export function handPoints(
  h: HandDetection, W: number, H: number, source: HandSource, side: Side = h.side,
): WorldLandmark[] {
  const pts = source === "image" ? handLandmarks3D(h.image, W, H) : h.world;
  return matchHandChirality(pts, side, h.labelSide === "L" ? "Left" : "Right");
}

/** Pick the hand to drive for each side from a detection. */
export function handsBySide(det: Detection): Partial<Record<Side, HandDetection>> {
  const out: Partial<Record<Side, HandDetection>> = {};
  for (const h of det.hands) if (!out[h.side]) out[h.side] = h;
  return out;
}

/** Best single hand in a close-up render (highest handedness confidence). */
function bestHand(det: Detection): HandDetection | undefined {
  return [...det.hands].sort((a, b) => b.labelScore - a.labelScore)[0];
}

/**
 * Driver input from a detection — through the library's own `handsForDriver`
 * (the path the demos use) for image-sourced hands.
 */
export function driverInput(det: Detection, source: HandSource): PoseInput | undefined {
  const pose = det.pose?.world as WorldLandmark[] | undefined;
  if (!pose && det.hands.length === 0) return undefined;
  const hands = source === "image"
    ? handsForDriver(
      { landmarks: det.hands.map(h => h.image), handedness: det.hands.map(h => (h.labelSide === "L" ? "Left" : "Right")) },
      det.width, det.height, det.pose?.image,
    )
    : {
      leftHand: handsBySide(det).L && handPoints(handsBySide(det).L!, det.width, det.height, source),
      rightHand: handsBySide(det).R && handPoints(handsBySide(det).R!, det.width, det.height, source),
    };
  return { pose, ...hands, visibilityMin: 0.2 };
}

export async function runImageCase(
  ctx: RunContext, name: string, imagePath: string, driver: Driver,
  opts: { renderBody?: boolean; renderHands?: boolean } = {},
): Promise<CaseResult> {
  const detection = await ctx.client.detectFile(imagePath);
  return runDetectionCase(ctx, name, detection, driver, opts);
}

export async function runDetectionCase(
  ctx: RunContext, name: string, detection: Detection, driver: Driver,
  opts: { renderBody?: boolean; renderHands?: boolean } = {},
): Promise<CaseResult> {
  const result: CaseResult = { name, detection, hands: [], notes: [] };
  const input = driverInput(detection, ctx.handSource);
  if (!input) {
    result.notes.push("nothing detected");
    return result;
  }
  let deltas: PoseDeltas;
  try {
    deltas = driver(input, ctx.model, ctx.boneIndex);
  } catch (err) {
    result.notes.push(`driver threw: ${(err as Error).message}`);
    return result;
  }
  const sk = posed(ctx, deltas);

  // ── Body ────────────────────────────────────────────────────────────────
  if (detection.pose && opts.renderBody !== false) {
    const srcDirs = bodyDirections(detection.pose.world.map(lm => ((lm.visibility ?? 1) >= 0.5 ? lm : undefined)));
    const modelDirs = bodyDirections(toMpWorld(poseJointsAnny(sk).map(p => p ?? [NaN, NaN, NaN])));
    const cam = fitCamera(allJoints(ctx, sk), 720, 1280, undefined, 30, 0.06);
    const rt = await ctx.client.renderAndDetect(deltas, cam);
    const rtDirs = rt.detection.pose ? bodyDirections(rt.detection.pose.world) : undefined;
    result.body = {
      srcVsModel: compareBody(srcDirs, modelDirs),
      srcVsRt: rtDirs && compareBody(srcDirs, rtDirs),
      modelVsRt: rtDirs && compareBody(modelDirs, rtDirs),
      rtDetection: rt.detection,
      renderPng: rt.png,
    };
    if (!rtDirs) result.notes.push("no pose on body render");
  }

  // ── Hands ───────────────────────────────────────────────────────────────
  const srcHands = handsBySide(detection);
  for (const side of ["L", "R"] as const) {
    const h = srcHands[side];
    if (!h) continue;
    const src = handAngles(handPoints(h, detection.width, detection.height, ctx.handSource), side);
    const modelJoints = handJointsAnny(sk, side);
    const model = handAngles(toMpWorld(modelJoints), side);
    const hr: HandResult = {
      side, labelSide: h.labelSide, geomSide: geometricSide(h.world),
      src, model, srcVsModel: compareHands(src, model),
    };
    if (opts.renderHands !== false) {
      const cam = handCamera(ctx, deltas, side);
      const rt = await ctx.client.renderAndDetect(deltas, cam, { handOnly: side }, { pose: false });
      hr.renderPng = rt.png;
      hr.rtDetection = rt.detection;
      const src2D = toPx(h.image, detection.width, detection.height);
      const model2D: P2[] = modelJoints.map(p => project(cam, p).slice(0, 2) as P2);
      hr.srcVsModel2D = compare2D(model2D, src2D).err;
      const rh = bestHand(rt.detection);
      if (rh) {
        const rt2D = toPx(rh.image, cam.W, cam.H);
        hr.srcVsRt2D = compare2D(rt2D, src2D).err;
        hr.modelVsRt2D = compare2D(rt2D, model2D).err;
        hr.rt = handAngles(handPoints(rh, cam.W, cam.H, ctx.handSource, side), side);
        hr.srcVsRt = compareHands(src, hr.rt);
        hr.modelVsRt = compareHands(model, hr.rt);
        if (rh.labelSide !== side) result.notes.push(`hand ${side}: render labelled ${rh.labelSide}`);
      } else {
        result.notes.push(`hand ${side}: no hand on close-up render`);
      }
    }
    result.hands.push(hr);
  }
  return result;
}
