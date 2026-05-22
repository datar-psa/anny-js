import { defineConfig } from "vite";
import { resolve } from "node:path";

const root = resolve(__dirname, "demo");
const projectRoot = __dirname;

export default defineConfig({
  root,
  // Serve assets/ at site root so demos can fetch /anny_model.json directly.
  publicDir: resolve(projectRoot, "assets"),
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
