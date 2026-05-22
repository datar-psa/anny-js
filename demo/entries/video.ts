/** Pre-recorded video → MediaPipe (heavy) → Anny demo with a fixture picker. */

import {
  loadAnnyModel, buildBoneIndex, allocBoneTransforms,
  lbs, allocVertexBuffer, forwardKinematics,
} from "../../src/anny/index.js";
import { landmarksToPoseDeltas, assignHands } from "../../src/mediapipe/index.js";
import type { Landmark, WorldLandmark } from "../../src/mediapipe/index.js";

import { setupWebGL, minRestZ } from "../_shared/webgl.js";
import { createMediaPipe } from "../_shared/mediapipe-loader.js";
import { smoothLandmarks } from "../_shared/smoothing.js";
import { computeQuadrants, buildFourViews, drawFourViews, drawDividers } from "../_shared/multi-view.js";

const W = window.innerWidth;
const H = window.innerHeight;

const statusEl = document.getElementById("status")!;
const picker   = document.getElementById("picker")!;
const video    = document.getElementById("src") as HTMLVideoElement;
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
const restMinZ   = minRestZ(model);

// Heavy pose model: offline video is latency-tolerant and benefits from accuracy.
const { poseLandmarker, handLandmarker } = await createMediaPipe({ poseModel: "heavy" });
statusEl.textContent = "MediaPipe ready — loading video…";

const gl = setupWebGL(glCanvas, model);

// ── Smoothed state (declared up-front so loadFixture can reset on switch) ──
let lastImageLm: Landmark[] | null = null;
let smoothImageLm: Landmark[] | null = null;
let smoothWorldLm: WorldLandmark[] | null = null;
let smoothLeftHand: WorldLandmark[] | null = null;
let smoothRightHand: WorldLandmark[] | null = null;
let lastHandRes: { landmarks: Landmark[][]; worldLandmarks: WorldLandmark[][] } | null = null;
let handFrame = 0;
let lastTs = -1;

async function loadFixture(src: string, label: string) {
  lastImageLm = smoothImageLm = smoothWorldLm = null;
  smoothLeftHand = smoothRightHand = null;
  lastHandRes = null;

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
await loadFixture("/fixtures/boxer.mp4", "boxer");

// ── Loop ────────────────────────────────────────────────────────────────
const LM_S = 0.5;
const HAND_S = 0.6;
const SMOOTH = 0.25;
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
    if ((handFrame++ & 1) === 0) {
      lastHandRes = handLandmarker.detectForVideo(video, now);
    }

    if (poseRes.landmarks.length > 0) {
      lastImageLm   = poseRes.landmarks[0];
      smoothImageLm = smoothLandmarks(smoothImageLm, lastImageLm, LM_S);
      smoothWorldLm = smoothLandmarks(smoothWorldLm, poseRes.worldLandmarks[0], LM_S);

      const hr = lastHandRes ?? { landmarks: [], worldLandmarks: [] };
      const { leftHand: lH, rightHand: rH } = assignHands(
        hr.landmarks, hr.worldLandmarks, smoothImageLm,
      );
      smoothLeftHand  = lH ? smoothLandmarks(smoothLeftHand,  lH, HAND_S) : null;
      smoothRightHand = rH ? smoothLandmarks(smoothRightHand, rH, HAND_S) : null;

      // Capture hook: the parity-fixture script (tools/capture_landmarks.mjs)
      // attaches `window.__captureFrames = []` and reads it back after playback.
      const w = window as unknown as { __captureFrames?: object[] };
      if (Array.isArray(w.__captureFrames)) {
        w.__captureFrames.push({
          worldLandmarks: poseRes.worldLandmarks[0].map(lm => ({ ...lm })),
        });
      }
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
  }, model, boneIndex);
  forwardKinematics(model, deltas, boneXforms);
  const mesh = lbs(model, boneXforms, vertBuf);
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
  const q = computeQuadrants(W, H, sScale, restMinZ);
  const views = buildFourViews(q);
  gl.uploadVertices(mesh.vertices);
  gl.clear();
  gl.setWorldTwist(0);
  drawFourViews(gl, views);

  // 2D overlay: dividers + skeleton overlay aligned to the source <video>
  ctx.clearRect(0, 0, W, H);
  drawDividers(ctx, W, H);
  drawSkeletonOverlay(ctx, lastImageLm, video);

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
function drawSkeletonOverlay(ctx: CanvasRenderingContext2D, lm: Landmark[], v: HTMLVideoElement): void {
  const SRC_W = 320, SRC_H = 240;
  const vW = v.videoWidth || SRC_W;
  const vH = v.videoHeight || SRC_H;
  const fit = Math.min(SRC_W / vW, SRC_H / vH);
  const drawW = vW * fit, drawH = vH * fit;
  const offX = (SRC_W - drawW) / 2, offY = (SRC_H - drawH) / 2;
  ctx.save();
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#0f0";
  const px = (i: number) => offX + lm[i].x * drawW;
  const py = (i: number) => offY + lm[i].y * drawH;
  const bone = (a: number, b: number) => {
    ctx.beginPath();
    ctx.moveTo(px(a), py(a));
    ctx.lineTo(px(b), py(b));
    ctx.stroke();
  };
  bone(11, 12); bone(11, 23); bone(12, 24); bone(23, 24);
  bone(11, 13); bone(13, 15); bone(12, 14); bone(14, 16);
  bone(23, 25); bone(25, 27); bone(24, 26); bone(26, 28);
  ctx.restore();
}

tick();
