/** Real-image cases for the round trip: MediaPipe's test photos + frames of the demo clips. */

import { existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { CACHE_DIR } from "./client.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

/** MediaPipe's own Apache-2.0 test photos (fetched by tools/fetch_assets.ts). */
export const PHOTOS = [
  "fist.jpg", "pointing_up.jpg", "thumb_up.jpg", "victory.jpg",
  "left_hands.jpg", "right_hands.jpg", "male_full_height_hands.jpg",
  "woman_hands.jpg", "hand-woman-man.jpg", "pose.jpg",
] as const;

export const photoPath = (f: string): string => resolve(CACHE_DIR, "testdata", f);

/** Demo clips under demo/fixtures/. */
export const VIDEOS = ["dancer", "boxer"] as const;

/**
 * `n` evenly spaced frames of a demo clip as JPEGs in `.cache/frames/<clip>-<n>/`
 * (extracted once with ffmpeg; the browser's Chromium build can't decode H.264).
 */
export async function videoFrames(clip: string, n: number): Promise<string[]> {
  const src = resolve(ROOT, "demo/fixtures", `${clip}.mp4`);
  if (!existsSync(src)) return [];
  const dir = resolve(CACHE_DIR, "frames", `${clip}-${n}`);
  if (!existsSync(dir) || readdirSync(dir).length !== n) {
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    const total = 96;   // both clips are ~4 s @ 24–25 fps
    const step = Math.max(1, Math.floor(total / n));
    const proc = Bun.spawnSync([
      "ffmpeg", "-v", "error", "-y", "-i", src,
      "-vf", `select='not(mod(n\\,${step}))'`, "-vsync", "vfr", "-frames:v", String(n),
      "-q:v", "2", resolve(dir, "%03d.jpg"),
    ]);
    if (proc.exitCode !== 0) throw new Error(`ffmpeg failed: ${proc.stderr.toString()}`);
  }
  return readdirSync(dir).sort().map(f => resolve(dir, f));
}
