/**
 * Browser side of the round-trip validator. Exposes `window.harness`, driven
 * by Playwright from Node (`tests/roundtrip/client.ts`):
 *
 *   init(opts)                       load Anny + MediaPipe
 *   detectUrl(url, det)              MediaPipe on an image served by the harness server
 *   render(deltas, cam, opts)        FK → LBS → pinhole render, returns PNG data URL
 *   renderAndDetect(deltas, cam, o)  render, then MediaPipe on the render
 *
 * The driver itself runs in Node; this page is only "MediaPipe + a GPU".
 */

import {
  allocBoneTransforms, allocNormalBuffer, allocVertexBuffer,
  forwardKinematics, lbs, loadAnnyModel,
  type AnnyModel, type PoseDeltas,
} from "../../src/anny/index.js";
import type { CameraSpec, Detection, DetectOptions, HarnessInit, RenderOptions } from "../roundtrip/protocol.js";
import { Detector } from "./detect.js";
import { PinholeRenderer } from "./render.js";

/** Deltas cross page.evaluate as plain arrays (9 numbers or null per bone). */
type WireDeltas = (number[] | null)[];

interface Harness {
  init(opts: HarnessInit): Promise<{ loadMs: number }>;
  detectUrl(url: string, det?: DetectOptions): Promise<Detection>;
  render(deltas: WireDeltas, cam: CameraSpec, opts?: RenderOptions): string;
  renderAndDetect(deltas: WireDeltas, cam: CameraSpec, opts?: RenderOptions, det?: DetectOptions): { detection: Detection; png: string };
}

let model: AnnyModel | undefined;
let detector: Detector | undefined;
let renderer: PinholeRenderer | undefined;
let boneXforms: Float32Array;
let verts: Float32Array;
let nrms: Float32Array;

function ready(): { model: AnnyModel; detector: Detector; renderer: PinholeRenderer } {
  if (!model || !detector || !renderer) throw new Error("harness.init() not called");
  return { model, detector, renderer };
}

function draw(deltas: WireDeltas, cam: CameraSpec, opts: RenderOptions = {}): HTMLCanvasElement {
  const h = ready();
  const d: PoseDeltas = deltas.map(m => (m ? new Float32Array(m) : null));
  forwardKinematics(h.model, d, boneXforms);
  const mesh = lbs(h.model, boneXforms, verts, nrms);
  return h.renderer.render(mesh.vertices, mesh.normals, cam, opts);
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

const harness: Harness = {
  async init(opts) {
    const t0 = performance.now();
    model = await loadAnnyModel("/anny/anny_model.json", "/anny/anny_model.bin");
    boneXforms = allocBoneTransforms(model.boneCount);
    verts = allocVertexBuffer(model);
    nrms = allocNormalBuffer(model);
    renderer = new PinholeRenderer(model);
    detector = await Detector.create(opts.poseModel);
    return { loadMs: performance.now() - t0 };
  },
  async detectUrl(url, det) {
    return ready().detector.detect(await loadImage(url), det);
  },
  render(deltas, cam, opts) {
    return draw(deltas, cam, opts).toDataURL("image/png");
  },
  renderAndDetect(deltas, cam, opts, det) {
    const canvas = draw(deltas, cam, opts);
    return { detection: ready().detector.detect(canvas, det), png: canvas.toDataURL("image/png") };
  },
};

(window as unknown as { harness: Harness }).harness = harness;
