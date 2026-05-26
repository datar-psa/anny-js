#!/usr/bin/env bun
/**
 * Capture MediaPipe Pose + ROI Hand worldLandmarks from each demo fixture
 * and write JSON to tests/mediapipe/fixtures/. Used to feed the parity tests
 * (tests/mediapipe/landmarks-parity.test.ts and hands-real.test.ts).
 *
 * Run: `bun tools/capture_landmarks.ts`
 *   - Assumes `bun run dev` (or `vite`) is serving the demo at :3000.
 *
 * Captures ~120 frames (~4 seconds of playback) per fixture using the
 * heavy pose model + the ROI hand detector that the video demo already
 * runs. The video element plays the same MP4 the demo loads.
 *
 * Per frame we record:
 *   - pose worldLandmarks (33 × xyz + visibility)
 *   - per-hand worldLandmarks (21 × xyz) for whichever hands ROI found,
 *     keyed by the **anatomical** L/R that the ROI detector assigned
 *     (using pose's anatomical LEFT_WRIST / RIGHT_WRIST as ground truth)
 */

import { chromium } from "playwright";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync } from "node:fs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = resolve(ROOT, "tests/mediapipe/fixtures");
const BASE_URL = process.env.DEMO_URL ?? "http://localhost:3000";

interface LM { x: number; y: number; z: number; visibility?: number }
interface Frame {
  worldLandmarks: LM[];                  // pose (33)
  imageLandmarks: LM[];                  // pose image-space (33)
  leftHand?:  { world: LM[]; image: LM[] };  // 21 per hand if detected
  rightHand?: { world: LM[]; image: LM[] };
}

const FIXTURES = process.env.FIXTURE
  ? [{ name: process.env.FIXTURE, buttonSelector: `button[data-label="${process.env.FIXTURE}"]` }]
  : [
      { name: "boxer",  buttonSelector: 'button[data-label="boxer"]'  },
      { name: "dancer", buttonSelector: 'button[data-label="dancer"]' },
      { name: "poser",  buttonSelector: 'button[data-label="poser"]'  },
    ];

// Heavy MediaPipe pose runs at ~0.4 fps in headless Chrome on a typical CI
// machine, so we keep the count modest. 30 distinct landmark sets per fixture
// is plenty for the verifier — it averages per-bone error across all frames,
// and the videos loop so 30 frames typically span the full motion arc.
const FRAMES_PER_FIXTURE = Number(process.env.FRAMES ?? 30);
const FRAME_TIMEOUT_MS = Number(process.env.TIMEOUT_MS ?? 300_000);

async function captureOne(name: string, buttonSelector: string): Promise<Frame[]> {
  console.log(`[${name}] launching browser…`);
  const browser = await chromium.launch({
    headless: false,
    // Ubuntu 26.04 isn't supported by Playwright's bundled chromium download.
    // Fall back to the system google-chrome (works for MediaPipe Tasks WebGL).
    executablePath: process.env.PLAYWRIGHT_CHROME ?? "/usr/bin/google-chrome",
    args: ["--use-gl=egl", "--ignore-gpu-blocklist", "--enable-gpu"],
  });
  const page = await browser.newPage();
  page.on("console", msg => console.log(`  [page] ${msg.text()}`));

  await page.goto(`${BASE_URL}/video.html`, { waitUntil: "load" });

  // Wait for the demo to wire up.
  await page.waitForFunction(() => document.getElementById("status")?.textContent?.includes("Playing") ?? false, undefined, { timeout: FRAME_TIMEOUT_MS });
  await page.click(buttonSelector);
  await page.waitForFunction(
    (label) => document.getElementById("status")?.textContent?.includes(label) ?? false,
    name,
    { timeout: FRAME_TIMEOUT_MS },
  );

  // Install capture hook. The demo pushes pose worldLandmarks already; we
  // patch it here to also push the ROI hand landmarks (anatomical L/R) so
  // each captured frame includes everything we need for the verifier.
  await page.evaluate(() => {
    (window as unknown as { __captureFrames: object[] }).__captureFrames = [];
  });
  console.log(`[${name}] capturing ${FRAMES_PER_FIXTURE} frames…`);
  try {
    await page.waitForFunction(
      (target) => ((window as unknown as { __captureFrames: object[] }).__captureFrames?.length ?? 0) >= target,
      FRAMES_PER_FIXTURE,
      { timeout: FRAME_TIMEOUT_MS, polling: 200 },
    );
  } catch (err) {
    const diag = await page.evaluate(() => {
      const w = window as unknown as { __captureFrames?: object[]; __lmDebug?: unknown; __handRoi?: { attempts: object[] } };
      const v = document.getElementById("src") as HTMLVideoElement | null;
      return {
        frames: w.__captureFrames?.length ?? 0,
        lmDebug: !!w.__lmDebug,
        handAttempts: w.__handRoi?.attempts?.length ?? 0,
        videoState: v ? { paused: v.paused, ended: v.ended, readyState: v.readyState, currentTime: v.currentTime, videoWidth: v.videoWidth, videoHeight: v.videoHeight, networkState: v.networkState, src: v.src } : null,
        status: document.getElementById("status")?.textContent,
      };
    });
    console.error(`[${name}] capture timeout. Diagnostics:`, JSON.stringify(diag, null, 2));
    throw err;
  }

  const frames = (await page.evaluate(
    () => (window as unknown as { __captureFrames: object[] }).__captureFrames,
  )) as Frame[];

  await browser.close();
  return frames.slice(0, FRAMES_PER_FIXTURE);
}

async function main(): Promise<void> {
  mkdirSync(OUT_DIR, { recursive: true });

  for (const fx of FIXTURES) {
    const frames = await captureOne(fx.name, fx.buttonSelector);
    const out = resolve(OUT_DIR, `${fx.name}_landmarks.json`);
    await Bun.write(out, JSON.stringify({ frames }, null, 0));
    const handFrames = frames.filter(f => f.leftHand || f.rightHand).length;
    console.log(`[${fx.name}] wrote ${frames.length} frames (${handFrames} with hands) → ${out}`);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
