/**
 * Node-side handle on the browser harness: starts the Vite server, launches
 * headless Chromium, and wraps `window.harness` calls with typed methods.
 */

import { relative, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Browser, Page } from "playwright";

import type { PoseDeltas } from "../../src/anny/index.js";
import { launchHeadless } from "../browser/browser.js";
import { startHarnessServer, type HarnessServer } from "../browser/server.js";
import type { CameraSpec, Detection, DetectOptions, HarnessInit, RenderOptions } from "./protocol.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
export const CACHE_DIR = resolve(ROOT, ".cache");

type WireDeltas = (number[] | null)[];
const toWire = (d: PoseDeltas): WireDeltas => d.map(m => (m ? Array.from(m) : null));

interface HarnessWindow {
  harness: {
    init(o: HarnessInit): Promise<{ loadMs: number }>;
    detectUrl(url: string, det?: DetectOptions): Promise<Detection>;
    render(d: WireDeltas, c: CameraSpec, o?: RenderOptions): string;
    renderAndDetect(d: WireDeltas, c: CameraSpec, o?: RenderOptions, det?: DetectOptions): { detection: Detection; png: string };
  };
}

export class HarnessClient {
  private constructor(
    private server: HarnessServer,
    private browser: Browser,
    private page: Page,
  ) {}

  static async start(init: HarnessInit = { poseModel: "heavy" }): Promise<HarnessClient> {
    const server = await startHarnessServer();
    const browser = await launchHeadless();
    try {
      const page = await browser.newPage();
      const errors: string[] = [];
      page.on("pageerror", e => errors.push(e.message));
      await page.goto(`${server.url}/harness.html`);
      await page.waitForFunction(() => "harness" in window, undefined, { timeout: 60_000 });
      await page.evaluate(o => (window as unknown as HarnessWindow).harness.init(o), init);
      if (errors.length) throw new Error(`harness page errors: ${errors.join("; ")}`);
      return new HarnessClient(server, browser, page);
    } catch (err) {
      await browser.close();
      await server.close();
      throw err;
    }
  }

  /** Run MediaPipe on an image file under `.cache/`. */
  async detectFile(path: string, det: DetectOptions = {}): Promise<Detection> {
    const rel = relative(CACHE_DIR, resolve(path));
    if (rel.startsWith("..")) throw new Error(`${path} must live under ${CACHE_DIR}`);
    const url = "/cache/" + rel.split("\\").join("/");
    return this.page.evaluate(([u, d]) => (window as unknown as HarnessWindow).harness.detectUrl(u, d), [url, det] as const);
  }

  async render(deltas: PoseDeltas, cam: CameraSpec, opts: RenderOptions = {}): Promise<string> {
    return this.page.evaluate(
      ([d, c, o]) => (window as unknown as HarnessWindow).harness.render(d, c, o),
      [toWire(deltas), cam, opts] as const,
    );
  }

  async renderAndDetect(
    deltas: PoseDeltas, cam: CameraSpec, opts: RenderOptions = {}, det: DetectOptions = {},
  ): Promise<{ detection: Detection; png: string }> {
    return this.page.evaluate(
      ([d, c, o, x]) => (window as unknown as HarnessWindow).harness.renderAndDetect(d, c, o, x),
      [toWire(deltas), cam, opts, det] as const,
    );
  }

  async close(): Promise<void> {
    await this.browser.close();
    await this.server.close();
  }
}

/** Decode a `data:image/png;base64,` URL to bytes. */
export function pngBytes(dataUrl: string): Uint8Array {
  return Buffer.from(dataUrl.slice(dataUrl.indexOf(",") + 1), "base64");
}
