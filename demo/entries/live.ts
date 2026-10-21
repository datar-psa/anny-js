/** Live webcam → MediaPipe → Anny demo. */

import {
  loadAnnyModel, allocBoneTransforms,
  lbs, allocVertexBuffer, allocNormalBuffer, forwardKinematics,
} from "../../src/anny/index.js";
import type { PoseDeltas } from "../../src/anny/index.js";
import { handsForDriver, PoseTracker } from "../../src/mediapipe/index.js";
import type { DriverHands, Landmark, WorldLandmark } from "../../src/mediapipe/index.js";

import { setupWebGL, minRestZ } from "../_shared/webgl.js";
import { createMediaPipe, mediaPipeAssetsFromQuery } from "../_shared/mediapipe-loader.js";
import { smoothLandmarks } from "../_shared/smoothing.js";
import { createCanvasRecorder } from "../_shared/recorder.js";
import { computeQuadrants, buildFourViews, drawFourViews, drawDividers } from "../_shared/multi-view.js";

const W = window.innerWidth;
const H = window.innerHeight;

const statusEl = document.getElementById("status")!;
const recBtn   = document.getElementById("rec")! as HTMLButtonElement;
const video    = document.getElementById("cam")! as HTMLVideoElement;
const glCanvas = document.getElementById("gl")  as HTMLCanvasElement;
const canvas   = document.getElementById("c")   as HTMLCanvasElement;
canvas.width = W; canvas.height = H;
glCanvas.width = W; glCanvas.height = H;
const ctx = canvas.getContext("2d")!;

// ── 1. Anny model ───────────────────────────────────────────────────────
const t0 = performance.now();
const model = await loadAnnyModel("/anny_model.json", "/anny_model.bin");
statusEl.textContent = `Anny loaded ${(performance.now() - t0).toFixed(0)} ms — starting MediaPipe…`;

const boneXforms = allocBoneTransforms(model.boneCount);
const vertBuf    = allocVertexBuffer(model);
const nrmBuf     = allocNormalBuffer(model);
const restMinZ   = minRestZ(model);

// ── 2. MediaPipe ────────────────────────────────────────────────────────
const { poseLandmarker, handLandmarker } = await createMediaPipe(mediaPipeAssetsFromQuery());
statusEl.textContent = "MediaPipe ready — requesting camera…";

// ── 3. Webcam ───────────────────────────────────────────────────────────
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
video.srcObject = stream;
await new Promise<void>(r => video.addEventListener("loadeddata", () => r(), { once: true }));

// ── 4. WebGL ────────────────────────────────────────────────────────────
const gl = setupWebGL(glCanvas, model);
statusEl.textContent = `✓ Live — GPU: ${gl.gpuName.substring(0, 40)}`;

// ── 5. Recorder ─────────────────────────────────────────────────────────
const recorder = createCanvasRecorder([glCanvas, canvas]);
recBtn.addEventListener("click", () => {
  if (recorder.isRecording()) {
    recorder.stop();
    recBtn.textContent = "⏺ REC";
    recBtn.classList.remove("recording");
  } else {
    recorder.start();
    recBtn.textContent = "⏹ STOP";
    recBtn.classList.add("recording");
  }
});

// ── 6. Render loop ──────────────────────────────────────────────────────
let lastImageLm: Landmark[] | null = null;
let smoothImageLm: Landmark[] | null = null;
let smoothWorldLm: WorldLandmark[] | null = null;
// Hands go raw into the tracker, which filters them against its natural-hand
// prior and carries a hand through frames where MediaPipe misses it.
const tracker = new PoseTracker(model);
let handFrame = 0;
let lastTs = -1;
let lastDeltas: PoseDeltas | null = null;
let handsSeen = "--";
let poseMs = "0.0";
let frameCount = 0, fpsTs = performance.now(), fps = 0;

const LM_S = 0.5;     // pose body smoothing
const SMOOTH = 0.25;  // screen-space placement smoothing

let sCx = W / 2, sScale = 280, sFootY = H * 0.88;
let lastFrame = 0;

function tick(): void {
  requestAnimationFrame(tick);
  const now = performance.now();
  if (now - lastFrame < 16) return;
  lastFrame = now;

  if (video.readyState >= 2 && now !== lastTs) {
    lastTs = now;
    const poseRes = poseLandmarker.detectForVideo(video, now);
    // Hands every other frame; in between the tracker carries them.
    const handRes = (handFrame++ & 1) === 0 ? handLandmarker.detectForVideo(video, now) : null;

    if (poseRes.landmarks.length > 0) {
      lastImageLm   = poseRes.landmarks[0];
      smoothImageLm = smoothLandmarks(smoothImageLm, lastImageLm, LM_S);
      smoothWorldLm = smoothLandmarks(smoothWorldLm, poseRes.worldLandmarks[0], LM_S);

      // Anatomical side from the pose wrists, 3D from the image landmarks,
      // chirality matched to the side, plus each hand's quality (size,
      // label agreement) — see `handsForDriver`.
      const hands: DriverHands = handRes
        ? handsForDriver(
          { landmarks: handRes.landmarks, handedness: handRes.handedness.map(h => h[0]?.categoryName ?? "") },
          video.videoWidth, video.videoHeight, smoothImageLm,
        )
        : {};
      if (handRes) handsSeen = (hands.leftHand ? "L" : "-") + (hands.rightHand ? "R" : "-");
      const t1 = performance.now();
      lastDeltas = tracker.update({ pose: smoothWorldLm, ...hands }, now);
      poseMs = (performance.now() - t1).toFixed(1);
    }
  }

  frameCount++;
  if (now - fpsTs >= 1000) {
    fps = frameCount; frameCount = 0; fpsTs = now;
  }

  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, W, H);

  if (!smoothWorldLm || !smoothImageLm || !lastImageLm || !lastDeltas) {
    ctx.fillStyle = "#555";
    ctx.font = "18px monospace";
    ctx.fillText("Stand in front of the camera…", W / 2 - 160, H / 2);
    return;
  }

  // ── Pose → mesh ──
  forwardKinematics(model, lastDeltas, boneXforms);
  const mesh = lbs(model, boneXforms, vertBuf, nrmBuf);

  // ── Screen position from image-space pose ──
  const lS = smoothImageLm[11], rS = smoothImageLm[12];
  const torsoW = Math.abs(lS.x - rS.x);
  const rawScale = torsoW > 0.02 ? (torsoW * W) / 0.42 : sScale;
  const shoulderY = ((lS.y + rS.y) / 2) * H;
  const rawFootY  = shoulderY + 1.35 * rawScale;
  const clampedScale = rawFootY > H - 20 ? (H - 20 - shoulderY) / 1.35 : rawScale;

  sCx    += (((lS.x + rS.x) / 2) * W   - sCx)    * SMOOTH;
  sScale += (clampedScale               - sScale) * SMOOTH;
  sFootY += (shoulderY + 1.35 * sScale  - sFootY) * SMOOTH;

  // ── Render 4 views ──
  const q = computeQuadrants(W, H, sScale, restMinZ);
  const views = buildFourViews(q);
  gl.uploadVertices(mesh.vertices);
  gl.uploadNormals(mesh.normals);
  gl.clear();
  gl.setWorldTwist(0);
  drawFourViews(gl, views);

  // ── 2D overlay: dividers, mini skeleton, stats ──
  ctx.clearRect(0, 0, W, H);
  drawDividers(ctx, W, H);

  drawMiniSkeleton(ctx, lastImageLm);

  ctx.fillStyle = "#4f4";
  ctx.font = "12px monospace";
  ctx.fillText(`driver ${poseMs} ms   ${fps} fps   hands:${handsSeen}`, 10, H - 10);
}

function drawMiniSkeleton(ctx: CanvasRenderingContext2D, lm: Landmark[]): void {
  const LW = 160, LH = 120;
  ctx.save();
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#0f0";
  const px = (i: number) => 8 + lm[i].x * LW;
  const py = (i: number) => 8 + lm[i].y * LH;
  const bone = (a: number, b: number) => {
    ctx.beginPath();
    ctx.moveTo(px(a), py(a));
    ctx.lineTo(px(b), py(b));
    ctx.stroke();
  };
  // torso
  bone(11, 12); bone(11, 23); bone(12, 24); bone(23, 24);
  // arms
  bone(11, 13); bone(13, 15); bone(12, 14); bone(14, 16);
  // legs
  bone(23, 25); bone(25, 27); bone(24, 26); bone(26, 28);
  ctx.restore();
}

tick();
