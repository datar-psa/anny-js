#!/usr/bin/env bun
/**
 * Download the MediaPipe model files and the MediaPipe test images used by the
 * headless browser tests into `.cache/` (gitignored). Idempotent — files that
 * already exist with the expected size are skipped.
 *
 * Run: `bun tools/fetch_assets.ts`  (or `task assets`)
 *
 * Why a cache instead of the CDN at runtime: the browser tests must run
 * headless, offline-after-first-fetch and fast. The demos still load from the
 * public CDN in production (see demo/_shared/mediapipe-loader.ts).
 */

import { mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const CACHE_DIR = resolve(ROOT, ".cache");

const MODELS_BASE = "https://storage.googleapis.com/mediapipe-models";
const TESTDATA_BASE = "https://storage.googleapis.com/mediapipe-assets";

interface Asset { url: string; out: string }

export const ASSETS: readonly Asset[] = [
  { url: `${MODELS_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,   out: "mediapipe/pose_landmarker_lite.task" },
  { url: `${MODELS_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,   out: "mediapipe/pose_landmarker_full.task" },
  { url: `${MODELS_BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`, out: "mediapipe/pose_landmarker_heavy.task" },
  { url: `${MODELS_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,             out: "mediapipe/hand_landmarker.task" },
  // MediaPipe's own Apache-2.0 test images: real hands with known gestures.
  ...[
    "fist.jpg", "pointing_up.jpg", "thumb_up.jpg", "victory.jpg",
    "left_hands.jpg", "right_hands.jpg", "male_full_height_hands.jpg",
    "woman_hands.jpg", "hand-woman-man.jpg", "pose.jpg",
  ].map(f => ({ url: `${TESTDATA_BASE}/${f}`, out: `testdata/${f}` })),
];

function existingSize(path: string): number {
  try { return statSync(path).size; } catch { return 0; }
}

async function fetchOne(a: Asset): Promise<"cached" | "downloaded"> {
  const out = resolve(CACHE_DIR, a.out);
  if (existingSize(out) > 0) return "cached";
  mkdirSync(dirname(out), { recursive: true });
  const res = await fetch(a.url);
  if (!res.ok) throw new Error(`GET ${a.url} → HTTP ${res.status}`);
  await Bun.write(out, await res.arrayBuffer());
  return "downloaded";
}

export async function fetchAssets(log: (msg: string) => void = console.log): Promise<void> {
  for (const a of ASSETS) {
    const state = await fetchOne(a);
    if (state === "downloaded") log(`  ↓ ${a.out}`);
  }
}

if (import.meta.main) {
  await fetchAssets();
  console.log(`assets ready in ${CACHE_DIR}`);
}
