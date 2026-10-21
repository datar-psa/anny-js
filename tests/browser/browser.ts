/**
 * Headless Chromium launcher for the browser tests.
 *
 * Playwright pins a Chromium revision per release; CI images and cloud
 * sandboxes often ship a different one (e.g. a preinstalled
 * `$PLAYWRIGHT_BROWSERS_PATH/chromium-1194`). Resolution order:
 *   1. $ANNY_CHROMIUM (explicit path)
 *   2. Playwright's own pinned build, if installed
 *   3. the newest `chromium-*` build under $PLAYWRIGHT_BROWSERS_PATH
 *
 * WebGL runs on SwiftShader (software) so the tests need no GPU.
 */

import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { chromium, type Browser } from "playwright";

function newestInstalledChromium(): string | undefined {
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (!base || !existsSync(base)) return undefined;
  const builds = readdirSync(base)
    .filter(d => /^chromium-\d+$/.test(d))
    .sort((a, b) => Number(b.split("-")[1]) - Number(a.split("-")[1]));
  for (const b of builds) {
    const exe = join(base, b, "chrome-linux", "chrome");
    if (existsSync(exe)) return exe;
  }
  return undefined;
}

export function resolveChromium(): string | undefined {
  if (process.env.ANNY_CHROMIUM) return process.env.ANNY_CHROMIUM;
  const pinned = chromium.executablePath();
  if (existsSync(pinned)) return undefined;   // let Playwright use its own
  return newestInstalledChromium();
}

export function chromiumAvailable(): boolean {
  return resolveChromium() !== undefined || existsSync(chromium.executablePath());
}

export async function launchHeadless(): Promise<Browser> {
  return chromium.launch({
    headless: true,
    executablePath: resolveChromium(),
    args: [
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
    ],
  });
}
