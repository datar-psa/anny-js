import { defineConfig, type Plugin } from "vite";
import { cpSync, createReadStream, existsSync, statSync } from "node:fs";
import { resolve, sep } from "node:path";

const root = resolve(__dirname, "demo");
const projectRoot = __dirname;

/**
 * Ship the demo clips. The video demo references them by runtime URL
 * (`/fixtures/*.mp4`), which Vite can't see, and they don't belong in
 * `assets/` (the npm package's `files`).
 */
function copyDemoFixtures(): Plugin {
  return {
    name: "copy-demo-fixtures",
    apply: "build",
    writeBundle(options) {
      const outDir = options.dir ?? resolve(projectRoot, "dist-demo");
      cpSync(resolve(root, "fixtures"), resolve(outDir, "fixtures"), { recursive: true });
    },
  };
}

/**
 * Dev-only: serve MediaPipe locally for `?mp=local` (offline / firewalled
 * machines). Wasm comes from node_modules, models from `task assets`; the
 * rest of `.cache/` (e.g. transcoded clips for `?clip=`) under /cache/.
 */
function localMediaPipe(): Plugin {
  const mounts = [
    { prefix: "/mp/wasm/", dir: resolve(projectRoot, "node_modules/@mediapipe/tasks-vision/wasm") },
    { prefix: "/mp/models/", dir: resolve(projectRoot, ".cache/mediapipe") },
    { prefix: "/cache/", dir: resolve(projectRoot, ".cache") },
  ];
  return {
    name: "local-mediapipe",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        const m = mounts.find(x => url.startsWith(x.prefix));
        if (!m) return next();
        const file = resolve(m.dir, url.slice(m.prefix.length));
        if (!file.startsWith(m.dir + sep) || !existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          res.end();
          return;
        }
        if (file.endsWith(".wasm")) res.setHeader("Content-Type", "application/wasm");
        if (file.endsWith(".webm")) res.setHeader("Content-Type", "video/webm");
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  root,
  // Serve assets/ at site root so demos can fetch /anny_model.json directly.
  publicDir: resolve(projectRoot, "assets"),
  plugins: [localMediaPipe(), copyDemoFixtures()],
  server: {
    port: 3000,
    fs: {
      // Allow Vite to read source files outside `root` (we import from ../src/).
      allow: [projectRoot],
    },
  },
  build: {
    outDir: resolve(projectRoot, "dist-demo"),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(root, "index.html"),
        anim:  resolve(root, "anim.html"),
        live:  resolve(root, "live.html"),
        video: resolve(root, "video.html"),
      },
    },
  },
});
