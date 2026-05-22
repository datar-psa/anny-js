#!/usr/bin/env bun
/**
 * Capture MediaPipe worldLandmarks from each demo fixture and write JSON
 * to tests/mediapipe/fixtures/. Used to feed the parity test
 * (tests/mediapipe/landmarks-parity.test.ts).
 *
 * Run: `bun tools/capture_landmarks.ts`
 *   - Assumes `bun run dev` (or `vite`) is serving the demo at :3000.
 *
 * Captures ~120 frames (~4 seconds of playback) per fixture using the
 * heavy pose model. The video element plays the same MP4 the demo loads.
 */

import { chromium } from "playwright";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync } from "node:fs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = resolve(ROOT, "tests/mediapipe/fixtures");
const BASE_URL = process.env.DEMO_URL ?? "http://localhost:3000";

interface Frame { worldLandmarks: Array<{ x: number; y: number; z: number; visibility?: number }> }

const FIXTURES = [
  { name: "boxer",  buttonSelector: 'button[data-label="boxer"]'  },
  { name: "dancer", buttonSelector: 'button[data-label="dancer"]' },
];

const FRAMES_PER_FIXTURE = 120;
const FRAME_TIMEOUT_MS = 30_000;

async function captureOne(name: string, buttonSelector: string): Promise<Frame[]> {
  console.log(`[${name}] launching browser…`);
  const browser = await chromium.launch({
    headless: false,  // GPU pipeline needs a window in headed mode for many MP builds
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

  // Install capture hook and let the demo collect frames.
  await page.evaluate(() => {
    (window as unknown as { __captureFrames: object[] }).__captureFrames = [];
  });
  console.log(`[${name}] capturing ${FRAMES_PER_FIXTURE} frames…`);
  await page.waitForFunction(
    (target) => ((window as unknown as { __captureFrames: object[] }).__captureFrames?.length ?? 0) >= target,
    FRAMES_PER_FIXTURE,
    { timeout: FRAME_TIMEOUT_MS, polling: 200 },
  );

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
    console.log(`[${fx.name}] wrote ${frames.length} frames → ${out}`);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
