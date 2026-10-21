#!/usr/bin/env bun
/** Screenshot a local HTML file headlessly: bun tools/screenshot.ts in.html out.png [width] */
import { launchHeadless } from "../tests/browser/browser.js";
import { resolve } from "node:path";

const [input, output, width = "1400"] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: bun tools/screenshot.ts <in.html> <out.png> [width]");
  process.exit(2);
}
const browser = await launchHeadless();
const page = await browser.newPage({ viewport: { width: Number(width), height: 900 } });
await page.goto(`file://${resolve(input)}`);
await page.waitForLoadState("networkidle");
await page.screenshot({ path: output, fullPage: true });
await browser.close();
