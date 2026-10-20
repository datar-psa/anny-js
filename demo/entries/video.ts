/** Pre-recorded video → MediaPipe (heavy) → Anny demo with a fixture picker. */

import {
  loadAnnyModel, buildBoneIndex, allocBoneTransforms,
  lbs, allocVertexBuffer, allocNormalBuffer, forwardKinematics,
} from "../../src/anny/index.js";
import type { PoseDeltas } from "../../src/anny/index.js";
import { handsForDriver, landmarksToPoseDeltas, MP_HAND } from "../../src/mediapipe/index.js";
import type { HandDetections, Landmark, WorldLandmark } from "../../src/mediapipe/index.js";

import { setupWebGL, minRestZ } from "../_shared/webgl.js";
import { createMediaPipe, mediaPipeAssetsFromQuery } from "../_shared/mediapipe-loader.js";
import { smoothLandmarks } from "../_shared/smoothing.js";
import { HandRoiDetector, isHandOutlier } from "../_shared/hand-roi.js";
import {
  computeQuadrants, buildFourViews, drawFourViews, drawDividers,
  buildSplitView, drawSplitView, drawSplitDivider,
} from "../_shared/multi-view.js";

const W = window.innerWidth;
const H = window.innerHeight;

const statusEl = document.getElementById("status")!;
const picker   = document.getElementById("picker")!;
const layoutEl = document.getElementById("layout")!;
const video    = document.getElementById("src") as HTMLVideoElement;

// ── Layout state ───────────────────────────────────────────────────────────
// "four":  standard 4-view (FRONT/BACK/3-4/ABOVE), source video as a small
//          thumbnail in the top-left corner.
// "split": Meshcapade-style side-by-side — source video fills the left half,
//          a single FRONT-locked Anny render fills the right half. Useful
//          for visually verifying that orientation tracks the subject in
//          real time (subject turns 90° → model turns 90°).
type Layout = "four" | "split";
let layout: Layout = "four";

function setLayout(next: Layout): void {
  layout = next;
  document.body.classList.toggle("split-layout", next === "split");
  for (const b of layoutEl.querySelectorAll("button[data-layout]")) {
    b.classList.toggle("active", (b as HTMLButtonElement).dataset.layout === next);
  }
}
for (const btn of layoutEl.querySelectorAll("button[data-layout]")) {
  btn.addEventListener("click", () => {
    setLayout((btn as HTMLButtonElement).dataset.layout as Layout);
  });
}
const glCanvas = document.getElementById("gl")  as HTMLCanvasElement;
const canvas   = document.getElementById("c")   as HTMLCanvasElement;
canvas.width = W; canvas.height = H;
glCanvas.width = W; glCanvas.height = H;
const ctx = canvas.getContext("2d")!;

// ── Model + MP ──────────────────────────────────────────────────────────
const t0 = performance.now();
const model = await loadAnnyModel("/anny_model.json", "/anny_model.bin");
statusEl.textContent = `Anny loaded ${(performance.now() - t0).toFixed(0)} ms — starting MediaPipe…`;

const boneIndex  = buildBoneIndex(model);
const boneXforms = allocBoneTransforms(model.boneCount);
const vertBuf    = allocVertexBuffer(model);
const nrmBuf     = allocNormalBuffer(model);
const restMinZ   = minRestZ(model);

// Heavy pose model: offline video is latency-tolerant and benefits from accuracy.
// Add an IMAGE-mode HandLandmarker for the ROI detector — full-frame hand
// detection fails on far-away subjects (hands occupy < 5% of frame), so we
// crop a tight region around each pose-detected wrist instead.
const { poseLandmarker, handLandmarkerImage } = await createMediaPipe({
  poseModel: "heavy",
  handMode:  "image",
  ...mediaPipeAssetsFromQuery(),
});
const handRoi = new HandRoiDetector(handLandmarkerImage!);
statusEl.textContent = "MediaPipe ready — loading video…";

const gl = setupWebGL(glCanvas, model);

// ── Smoothed state (declared up-front so loadFixture can reset on switch) ──
let lastImageLm: Landmark[] | null = null;
let smoothImageLm: Landmark[] | null = null;
let smoothWorldLm: WorldLandmark[] | null = null;
let smoothLeftHand: WorldLandmark[] | null = null;
let smoothRightHand: WorldLandmark[] | null = null;
let lastHandDet: HandDetections | null = null;
let lastHands: { leftHand?: WorldLandmark[]; rightHand?: WorldLandmark[] } = {};
let handFrame = 0;
let lastTs = -1;
let lastDeltas: PoseDeltas | null = null;

async function loadFixture(src: string, label: string) {
  lastImageLm = smoothImageLm = smoothWorldLm = null;
  smoothLeftHand = smoothRightHand = null;
  lastHandDet = null;
  lastHands = {};
  lastDeltas = null;

  video.src = src;
  await new Promise<void>(r => video.addEventListener("loadeddata", () => r(), { once: true }));
  try { await video.play(); } catch { /* autoplay may be blocked; click to start */ }
  statusEl.textContent = `✓ Playing — ${label} (${video.videoWidth}×${video.videoHeight})`;
}

for (const btn of picker.querySelectorAll("button")) {
  btn.addEventListener("click", async () => {
    for (const b of picker.querySelectorAll("button")) b.classList.remove("active");
    btn.classList.add("active");
    const b = btn as HTMLButtonElement;
    await loadFixture(b.dataset.src!, b.dataset.label!);
  });
}
// `?clip=<url>` plays any clip the browser can decode (e.g. a WebM on
// Chromium builds without H.264) instead of the bundled boxer fixture.
const clipParam = new URLSearchParams(location.search).get("clip");
if (clipParam) for (const b of picker.querySelectorAll("button")) b.classList.remove("active");
await loadFixture(clipParam ?? "/fixtures/boxer.mp4", clipParam ? "custom clip" : "boxer");

// ── Loop ────────────────────────────────────────────────────────────────
const LM_S = 0.5;
// HAND_S is lower than the body LM_S because hold-last (line ~130) means a
// single noisy detection persists across dropout frames; aggressive smoothing
// (low alpha) damps the visible effect of any one outlier detection. The
// trade-off is finger motion lags real motion by ~3 frames, which is fine
// for video where finger micro-movements aren't the focus.
const HAND_S = 0.35;
const SMOOTH = 0.25;

// ── Hand tracking toggle ────────────────────────────────────────────────────
// Per-wrist ROI crops → MediaPipe Hand → `handsForDriver` (anatomical side,
// image landmarks lifted to 3D, chirality fixed) → palm + finger bones. On
// a far-away subject the finger detail is a best guess, but the anatomical
// joint model keeps it a plausible hand. Toggle with the ✋ button.
let trackHands = true;
const handsBtn = document.getElementById("hands") as HTMLButtonElement | null;
handsBtn?.addEventListener("click", () => {
  trackHands = !trackHands;
  handsBtn.classList.toggle("active", trackHands);
  if (!trackHands) { smoothLeftHand = smoothRightHand = null; lastHandDet = null; lastHands = {}; }
});

let sScale = 280;
let sCx = W / 2;
let sFootY = H * 0.88;

let frameCount = 0, fpsTs = performance.now(), fps = 0;
let lastFrame = 0;

function tick(): void {
  requestAnimationFrame(tick);
  const now = performance.now();
  if (now - lastFrame < 16) return;
  lastFrame = now;

  if (video.readyState >= 2 && now !== lastTs) {
    lastTs = now;
    const poseRes = poseLandmarker.detectForVideo(video, now);

    if (poseRes.landmarks.length > 0) {
      lastImageLm   = poseRes.landmarks[0];
      smoothImageLm = smoothLandmarks(smoothImageLm, lastImageLm, LM_S);
      smoothWorldLm = smoothLandmarks(smoothWorldLm, poseRes.worldLandmarks[0], LM_S);

      // Hands: run the ROI detector every other frame (saves ~10 ms/frame;
      // smoothing covers the gap). Full-frame detection fails on far-away
      // subjects (hands < 5 % of the frame), hence the per-wrist crops.
      if (trackHands && (handFrame++ & 1) === 0) {
        lastHandDet = handRoi.detect(video, smoothImageLm);
        lastHands = handsForDriver(lastHandDet, video.videoWidth, video.videoHeight, smoothImageLm);
      }
      // Hold-last + outlier reject: skip the smoothing update when the ROI
      // missed this frame (fingers would snap to rest) or when the fresh
      // detection jumps > 60° on any wrist→knuckle direction (a noise spike).
      const lH = lastHands.leftHand, rH = lastHands.rightHand;
      if (lH && !isHandOutlier(smoothLeftHand, lH)) smoothLeftHand = smoothLandmarks(smoothLeftHand, lH, HAND_S);
      if (rH && !isHandOutlier(smoothRightHand, rH)) smoothRightHand = smoothLandmarks(smoothRightHand, rH, HAND_S);
    }
  }

  frameCount++;
  if (now - fpsTs >= 1000) {
    fps = frameCount; frameCount = 0; fpsTs = now;
  }

  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, W, H);

  if (!smoothWorldLm || !smoothImageLm || !lastImageLm) {
    ctx.fillStyle = "#555";
    ctx.font = "18px monospace";
    ctx.fillText("Waiting for first pose…", W/2 - 120, H/2);
    return;
  }

  const t1 = performance.now();
  const deltas = landmarksToPoseDeltas({
    pose:      smoothWorldLm,
    leftHand:  smoothLeftHand  ?? undefined,
    rightHand: smoothRightHand ?? undefined,
    // Lower threshold so half-occluded limbs still drive — noisy beats stuck-at-rest.
    visibilityMin: 0.2,
    // Hold last good pose when MP visibility dips below threshold. Without
    // this the arm snaps to spine on every confidence dip (boxer's guard
    // hand sits at vis≈0.19 for ~15% of frames).
    previousDeltas: lastDeltas ?? undefined,
  }, model, boneIndex);
  lastDeltas = deltas;
  forwardKinematics(model, deltas, boneXforms);
  const mesh = lbs(model, boneXforms, vertBuf, nrmBuf);
  const poseMs = (performance.now() - t1).toFixed(1);

  // Screen-space placement from image-space pose
  const lS = smoothImageLm[11], rS = smoothImageLm[12];
  const torsoW = Math.abs(lS.x - rS.x);
  const rawScale = torsoW > 0.02 ? (torsoW * W) / 0.42 : sScale;
  const shoulderY = ((lS.y + rS.y) / 2) * H;
  const rawFootY  = shoulderY + 1.35 * rawScale;
  const clampedScale = rawFootY > H - 20 ? (H - 20 - shoulderY) / 1.35 : rawScale;
  sCx    += (((lS.x + rS.x) / 2) * W  - sCx)    * SMOOTH;
  sScale += (clampedScale              - sScale) * SMOOTH;
  sFootY += (shoulderY + 1.35 * sScale - sFootY) * SMOOTH;

  // Render
  gl.uploadVertices(mesh.vertices);
  gl.uploadNormals(mesh.normals);
  gl.clear();
  gl.setWorldTwist(0);

  ctx.clearRect(0, 0, W, H);

  if (layout === "split") {
    // Side-by-side: video fills the left half (via CSS), model fills the right.
    const sv = buildSplitView(W, H, restMinZ);
    drawSplitView(gl, sv);
    drawSplitDivider(ctx, W, H);
    drawSkeletonOverlay(ctx, lastImageLm, lastHandDet, video, /* splitLayout */ true);
  } else {
    const q = computeQuadrants(W, H, sScale, restMinZ);
    const views = buildFourViews(q);
    drawFourViews(gl, views);
    drawDividers(ctx, W, H);
    drawSkeletonOverlay(ctx, lastImageLm, lastHandDet, video, /* splitLayout */ false);
  }

  ctx.fillStyle = "#4f4";
  ctx.font = "12px monospace";
  const hands = (smoothLeftHand ? "L" : "-") + (smoothRightHand ? "R" : "-");
  ctx.fillText(`FK+LBS ${poseMs} ms   ${fps} fps   hands:${hands}`, 10, H - 10);
}

/**
 * Draw a skeleton overlay aligned to the source <video> thumbnail (320×240
 * with object-fit: contain). Landmarks are normalised to native aspect, so
 * we map them to the contain-fitted box.
 */
// All 33 MediaPipe Pose landmark connections, grouped by region.
// Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const POSE_EDGES: ReadonlyArray<[number, number]> = [
  // Face — eyes + ears + mouth (gives the head a recognisable shape on the
  // overlay so you can see which way the subject is looking).
  [0,1], [1,2], [2,3], [3,7],          // L eye chain → L ear
  [0,4], [4,5], [5,6], [6,8],          // R eye chain → R ear
  [9,10],                              // mouth
  // Torso
  [11,12], [11,23], [12,24], [23,24],
  // Arms
  [11,13], [13,15],                    // L upper + L lower
  [12,14], [14,16],                    // R upper + R lower
  // Hand-knuckle triangles (pose's coarse hand)
  [15,17], [15,19], [15,21], [17,19],  // L hand: wrist↔pinky↔index, wrist↔thumb
  [16,18], [16,20], [16,22], [18,20],  // R hand
  // Legs
  [23,25], [25,27], [27,29], [29,31], [27,31],   // L leg + L foot
  [24,26], [26,28], [28,30], [30,32], [28,32],   // R leg + R foot
];

// MediaPipe Hand Landmarker connections (21 landmarks per hand).
// Each finger chain WRIST(0) → MCP → PIP → DIP → TIP, plus the "palm web"
// connecting the MCPs across the palm.
const HAND_EDGES: ReadonlyArray<[number, number]> = [
  // Thumb
  [0,1], [1,2], [2,3], [3,4],
  // Index
  [0,5], [5,6], [6,7], [7,8],
  // Middle
  [9,10], [10,11], [11,12],
  // Ring
  [13,14], [14,15], [15,16],
  // Pinky
  [0,17], [17,18], [18,19], [19,20],
  // Palm web (MCP across the palm)
  [5,9], [9,13], [13,17],
];

function drawSkeletonOverlay(
  ctx: CanvasRenderingContext2D,
  lm: Landmark[],
  hands: HandDetections | null,
  v: HTMLVideoElement,
  splitLayout: boolean,
): void {
  // The video element is sized differently per layout:
  //   four:  320×240 thumbnail in the top-left corner
  //   split: 50% viewport-width × 100% viewport-height (left half)
  // Both use object-fit: contain so we mirror that letterbox/pillarbox math.
  const SRC_W = splitLayout ? W / 2 : 320;
  const SRC_H = splitLayout ? H     : 240;
  const vW = v.videoWidth || SRC_W;
  const vH = v.videoHeight || SRC_H;
  const fit = Math.min(SRC_W / vW, SRC_H / vH);
  const drawW = vW * fit, drawH = vH * fit;
  const offX = (SRC_W - drawW) / 2, offY = (SRC_H - drawH) / 2;

  function drawSkeleton(
    arr: Landmark[],
    edges: ReadonlyArray<[number, number]>,
    stroke: string,
    lineWidth: number,
    dotRadius: number,
  ): void {
    const px = (i: number) => offX + arr[i].x * drawW;
    const py = (i: number) => offY + arr[i].y * drawH;
    ctx.strokeStyle = stroke;
    ctx.lineWidth   = lineWidth;
    for (const [a, b] of edges) {
      if (!arr[a] || !arr[b]) continue;
      ctx.beginPath();
      ctx.moveTo(px(a), py(a));
      ctx.lineTo(px(b), py(b));
      ctx.stroke();
    }
    // Joint dots so individual landmarks are visible (especially fingers
    // and face landmarks that aren't connected by edges).
    ctx.fillStyle = stroke;
    for (let i = 0; i < arr.length; i++) {
      const p = arr[i]; if (!p) continue;
      ctx.beginPath();
      ctx.arc(offX + p.x * drawW, offY + p.y * drawH, dotRadius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.save();
  ctx.globalAlpha = 0.7;
  drawSkeleton(lm, POSE_EDGES, "#0f0", splitLayout ? 2 : 1.5, splitLayout ? 2 : 1.2);
  for (const hand of hands?.landmarks ?? []) {
    // Colour by which pose wrist the hand sits nearer: blue = left, orange = right.
    const w = hand[MP_HAND.WRIST], pl = lm[15], pr = lm[16];
    const left = Math.hypot(w.x - pl.x, w.y - pl.y) <= Math.hypot(w.x - pr.x, w.y - pr.y);
    drawSkeleton(hand, HAND_EDGES, left ? "#4af" : "#f84", splitLayout ? 1.5 : 1, splitLayout ? 1.6 : 1);
  }
  ctx.restore();
}

tick();
