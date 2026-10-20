/**
 * In-process Vite server for the headless browser tests.
 *
 * Serves the harness page (`tests/browser/harness.html`) plus three static
 * mounts the page needs, so nothing is fetched from a CDN at test time:
 *   /anny/     → assets/                                   (Anny model)
 *   /mp/wasm/  → node_modules/@mediapipe/tasks-vision/wasm (MediaPipe runtime)
 *   /cache/    → .cache/                                   (models, test images, frames)
 */

import { createReadStream, statSync } from "node:fs";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { IncomingMessage, ServerResponse } from "node:http";
import { createServer, type Plugin, type ViteDevServer } from "vite";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

const MOUNTS: ReadonlyArray<{ prefix: string; dir: string }> = [
  { prefix: "/anny/",    dir: resolve(ROOT, "assets") },
  { prefix: "/mp/wasm/", dir: resolve(ROOT, "node_modules/@mediapipe/tasks-vision/wasm") },
  { prefix: "/cache/",   dir: resolve(ROOT, ".cache") },
];

const MIME: Record<string, string> = {
  ".wasm": "application/wasm",
  ".js": "text/javascript",
  ".json": "application/json",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
};

function serveStatic(req: IncomingMessage, res: ServerResponse, next: () => void): void {
  const url = decodeURIComponent((req.url ?? "").split("?")[0]);
  const mount = MOUNTS.find(m => url.startsWith(m.prefix));
  if (!mount) return next();
  const file = resolve(mount.dir, url.slice(mount.prefix.length));
  if (!file.startsWith(mount.dir + sep)) {
    res.statusCode = 403;
    res.end();
    return;
  }
  let size: number;
  try {
    const st = statSync(file);
    if (!st.isFile()) throw new Error("not a file");
    size = st.size;
  } catch {
    res.statusCode = 404;
    res.end(`not found: ${url}`);
    return;
  }
  res.setHeader("Content-Type", MIME[extname(file)] ?? "application/octet-stream");
  res.setHeader("Content-Length", String(size));
  createReadStream(file).pipe(res);
}

const staticMounts: Plugin = {
  name: "anny-static-mounts",
  configureServer(server) {
    server.middlewares.use(serveStatic);
  },
};

export interface HarnessServer {
  url: string;
  close(): Promise<void>;
}

export async function startHarnessServer(): Promise<HarnessServer> {
  const server: ViteDevServer = await createServer({
    configFile: false,
    root: resolve(ROOT, "tests/browser"),
    logLevel: "error",
    plugins: [staticMounts],
    server: { port: 0, host: "127.0.0.1", fs: { allow: [ROOT] } },
    optimizeDeps: { include: ["@mediapipe/tasks-vision"] },
  });
  await server.listen();
  const addr = server.httpServer?.address();
  if (!addr || typeof addr === "string") throw new Error("vite: no listening address");
  return {
    url: `http://127.0.0.1:${addr.port}`,
    close: () => server.close(),
  };
}
