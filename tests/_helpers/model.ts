import { parseAnnyModel, type AnnyManifest, type AnnyModel } from "../../src/anny/index.js";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ASSETS = resolve(here, "../../assets");

let cached: Promise<AnnyModel> | undefined;

/** Load the bundled Anny model from `assets/`. Cached across tests in the run. */
export function loadFixtureModel(): Promise<AnnyModel> {
  if (cached) return cached;
  cached = (async () => {
    const manifest: AnnyManifest = JSON.parse(
      await Bun.file(resolve(ASSETS, "anny_model.json")).text(),
    );
    const buf = await Bun.file(resolve(ASSETS, "anny_model.bin")).arrayBuffer();
    return parseAnnyModel(manifest, buf);
  })();
  return cached;
}
