/** Pre-recorded video → MediaPipe (heavy) → Anny demo with a fixture picker. */

import {
  loadAnnyModel, buildBoneIndex, allocBoneTransforms,
  lbs, allocVertexBuffer, allocNormalBuffer, forwardKinematics,
} from "../../src/anny/index.js";
import type { PoseDeltas } from "../../src/anny/index.js";
import { landmarksToPoseDeltas } from "../../src/mediapipe/index.js";
import type { Landmark, WorldLandmark } from "../../src/mediapipe/index.js";

import { setupWebGL, minRestZ } from "../_shared/webgl.js";
import { createMediaPipe } from "../_shared/mediapipe-loader.js";
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
  for (const b of layoutEl.querySelectorAll("button")) {
    b.classList.toggle("active", (b as HTMLButtonElement).dataset.layout === next);
  }
}
for (const btn of layoutEl.querySelectorAll("button")) {
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
let lastHandRes: import("../_shared/hand-roi.js").RoiHandResult | null = null;
let handFrame = 0;
let lastTs = -1;
let lastDeltas: PoseDeltas | null = null;

async function loadFixture(src: string, label: string) {
  lastImageLm = smoothImageLm = smoothWorldLm = null;
  smoothLeftHand = smoothRightHand = null;
  lastHandRes = null;
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
await loadFixture("/fixtures/boxer.mp4", "boxer");

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
// Finger articulation from MediaPipe hand landmarks is unreliable on
// full-body footage (the hand is a few % of frame, so per-finger landmarks —
// especially the distal joints — are dominated by noise). The whole hand
// pipeline (ROI detection, frame alignment, palmar-clamped coordinated curl)
// is kept intact below and in `src/mediapipe/landmarks.ts`, but disabled here:
// with no hand landmarks fed to the driver, the hand bones inherit their
// natural rest pose (a relaxed, open hand) and the wrist follows the forearm.
// Flip to `true` to re-enable live finger tracking.
const TRACK_HANDS = false;

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

      // Hands: run the ROI detector every other frame (saves ~10ms/frame,
      // smoothing covers the gap). On far-away subjects this is the only way
      // to get hand landmarks — the full-frame HandLandmarker returns 0 hands
      // because the hands are < 5% of the frame. Skipped when TRACK_HANDS is
      // off, EXCEPT during fixture capture (where we still need raw hand
      // landmarks recorded to JSON for the hand tests).
      const capturing = Array.isArray((window as unknown as { __captureFrames?: unknown }).__captureFrames);
      if ((TRACK_HANDS || capturing) && (handFrame++ & 1) === 0) {
        lastHandRes = handRoi.detect(video, smoothImageLm);
        // Expose for devtools/Playwright inspection.
        (window as unknown as { __handRoi?: unknown }).__handRoi = handRoi.lastDebug;
      }

      // Expose model + bone buffers so the in-page hand verifier can compute
      // each finger bone's Y axis in world and compare it to the MP-derived
      // target direction. Set once; the buffers are mutated in place by FK.
      const wd = window as unknown as {
        __model?: unknown; __boneIndex?: unknown; __boneXforms?: unknown;
        __smoothLeftHand?: unknown; __smoothRightHand?: unknown;
      };
      if (!wd.__model) {
        wd.__model = model;
        wd.__boneIndex = boneIndex;
        wd.__boneXforms = boneXforms;
      }
      // The smoothed hands ARE what the driver consumes (lastHandRes is the
      // raw ROI result — smoothing + outlier reject sit between). Expose the
      // smoothed arrays each frame so the verifier compares like-for-like.
      wd.__smoothLeftHand  = smoothLeftHand;
      wd.__smoothRightHand = smoothRightHand;

      // The ROI detector returns side-keyed results directly — each crop is
      // centred on a specific pose wrist, so the detected hand IS the
      // anatomically-correct one. No re-disambiguation needed.
      const lH = lastHandRes?.leftHand?.world;
      const rH = lastHandRes?.rightHand?.world;

      // Debug hook: image-space hand wrists + both pose wrists, plus a few
      // world landmarks so we can verify hand chirality numerically.
      // For a LEFT hand: thumb sits on the medial (subject's right) side, so
      // in MP world (+x = anatomical-left) we expect thumb_CMC.x < wrist.x.
      // For a RIGHT hand: thumb on subject's left = +x → thumb_CMC.x > wrist.x.
      // If empirics disagree, MP's hand model is outputting chirality-flipped
      // landmarks (a known consequence of its "input is mirrored" training).
      (window as unknown as { __handAssign?: unknown }).__handAssign = {
        poseLeft:  smoothImageLm[15],
        poseRight: smoothImageLm[16],
        leftHandWrist:  lastHandRes?.leftHand?.image[0]  ?? null,
        rightHandWrist: lastHandRes?.rightHand?.image[0] ?? null,
        leftHandWorld:  lastHandRes?.leftHand?.world
          ? {
              wrist:    lastHandRes.leftHand.world[0],
              thumbCmc: lastHandRes.leftHand.world[1],
              indexMcp: lastHandRes.leftHand.world[5],
              pinkyMcp: lastHandRes.leftHand.world[17],
            }
          : null,
        rightHandWorld: lastHandRes?.rightHand?.world
          ? {
              wrist:    lastHandRes.rightHand.world[0],
              thumbCmc: lastHandRes.rightHand.world[1],
              indexMcp: lastHandRes.rightHand.world[5],
              pinkyMcp: lastHandRes.rightHand.world[17],
            }
          : null,
      };
      // Update only when fresh landmarks arrive AND they don't look like a
      // noise spike. Two safeguards in series:
      //   1) hold-last: skip the smoothLandmarks call when the ROI failed
      //      this frame, otherwise fingers would snap to rest on every miss.
      //   2) outlier reject: if the fresh detection differs from the held
      //      pose by > 60° on any wrist→knuckle direction (likely a noise
      //      spike rather than real motion), skip THAT update too.
      // Without (2), hold-last would "lock in" the first noisy detection and
      // keep it visible until the next clean detection lands — the cause of
      // the visible finger-jumping the user reported.
      if (lH && !isHandOutlier(smoothLeftHand,  lH)) {
        smoothLeftHand  = smoothLandmarks(smoothLeftHand,  lH, HAND_S);
      }
      if (rH && !isHandOutlier(smoothRightHand, rH)) {
        smoothRightHand = smoothLandmarks(smoothRightHand, rH, HAND_S);
      }

      // Capture hook: the parity-fixture script (tools/capture_landmarks.ts)
      // attaches `window.__captureFrames = []` and reads it back after playback.
      // We push everything the verifier needs: pose world + pose image + per-hand
      // (world + image) keyed by the anatomical side the ROI detector assigned.
      const w = window as unknown as { __captureFrames?: object[] };
      if (Array.isArray(w.__captureFrames)) {
        const handPayload = (h?: { image: Landmark[]; world: WorldLandmark[] }) =>
          h ? { world: h.world.map(lm => ({ ...lm })), image: h.image.map(lm => ({ ...lm })) } : undefined;
        w.__captureFrames.push({
          worldLandmarks: poseRes.worldLandmarks[0].map(lm => ({ ...lm })),
          imageLandmarks: poseRes.landmarks[0].map(lm => ({ ...lm })),
          leftHand:  handPayload(lastHandRes?.leftHand),
          rightHand: handPayload(lastHandRes?.rightHand),
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
    // Hands disabled (TRACK_HANDS) → undefined, so finger bones stay at their
    // open rest pose and the wrist inherits the forearm.
    leftHand:  TRACK_HANDS ? (smoothLeftHand  ?? undefined) : undefined,
    rightHand: TRACK_HANDS ? (smoothRightHand ?? undefined) : undefined,
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

  // Debug: also expose raw smoothed MP landmarks (image + world) so we can
  // verify coordinate conventions (e.g. is MP world +x anatomical left?).
  (window as unknown as { __lmDebug?: unknown }).__lmDebug = {
    image: smoothImageLm,
    world: smoothWorldLm,
  };

  // Debug: expose bone world positions so we can diagnose pose vs render
  // bugs from devtools (e.g. compare hand vs head depth from a back view).
  // boneXforms[b] = pose[b] @ inv(rest[b]); apply to rest-world bone origin
  // to recover pose-world bone origin.
  {
    const dbg = window as unknown as { __bones?: Record<string, [number, number, number]> };
    const t = (name: string): [number, number, number] | undefined => {
      const i = boneIndex.get(name); if (i === undefined) return;
      const m = i * 16, r = i * 16;
      const rx = model.restBonePoses[r + 3];
      const ry = model.restBonePoses[r + 7];
      const rz = model.restBonePoses[r + 11];
      return [
        boneXforms[m]*rx + boneXforms[m+1]*ry + boneXforms[m+2]*rz + boneXforms[m+3],
        boneXforms[m+4]*rx + boneXforms[m+5]*ry + boneXforms[m+6]*rz + boneXforms[m+7],
        boneXforms[m+8]*rx + boneXforms[m+9]*ry + boneXforms[m+10]*rz + boneXforms[m+11],
      ];
    };
    const bones: Record<string, [number, number, number]> = {};
    for (const n of ["head", "neck01", "wrist.L", "wrist.R", "lowerarm01.L", "lowerarm01.R", "upperarm01.L", "upperarm01.R"]) {
      const p = t(n); if (p) bones[n] = p;
    }
    dbg.__bones = bones;
  }

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
    drawSkeletonOverlay(
      ctx, lastImageLm,
      lastHandRes?.leftHand?.image ?? null,
      lastHandRes?.rightHand?.image ?? null,
      video, /* splitLayout */ true,
    );
  } else {
    const q = computeQuadrants(W, H, sScale, restMinZ);
    const views = buildFourViews(q);
    drawFourViews(gl, views);
    drawDividers(ctx, W, H);
    drawSkeletonOverlay(
      ctx, lastImageLm,
      lastHandRes?.leftHand?.image ?? null,
      lastHandRes?.rightHand?.image ?? null,
      video, /* splitLayout */ false,
    );
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
  leftHandImg: Landmark[] | null,
  rightHandImg: Landmark[] | null,
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
  if (leftHandImg)  drawSkeleton(leftHandImg,  HAND_EDGES, "#4af", splitLayout ? 1.5 : 1, splitLayout ? 1.6 : 1);
  if (rightHandImg) drawSkeleton(rightHandImg, HAND_EDGES, "#f84", splitLayout ? 1.5 : 1, splitLayout ? 1.6 : 1);
  ctx.restore();
}

tick();
