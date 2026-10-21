/** Text table + HTML report for round-trip results. */

import { copyFileSync, mkdirSync } from "node:fs";
import { basename, resolve } from "node:path";

import { pngBytes } from "./client.js";
import { mean, percentile } from "./metrics.js";
import type { CaseResult } from "./run.js";
import type { Landmark } from "../../src/mediapipe/index.js";

const f1 = (x: number | undefined): string => (x === undefined || Number.isNaN(x) ? "  -  " : x.toFixed(1).padStart(5));

const pct = (x: number | undefined): string => f1(x === undefined ? undefined : x * 100);

const LIMBS = ["upperarm.L", "upperarm.R", "forearm.L", "forearm.R", "thigh.L", "thigh.R", "shin.L", "shin.R"];

function limbMean(r: Record<string, number> | undefined): number | undefined {
  if (!r) return undefined;
  const xs = LIMBS.map(k => r[k]).filter(x => x !== undefined && !Number.isNaN(x));
  return xs.length ? mean(xs) : undefined;
}

export function printTable(results: CaseResult[]): void {
  console.log("\ncase                          body°(src~mdl src~rt)  hand  fingers°(src~mdl src~rt mdl~rt)  thumb°(s~m s~r)  palm°(s~m s~r)  flips(s~m s~r)  2D%(s~m s~r m~r)");
  for (const r of results) {
    const body = r.body ? `${f1(limbMean(r.body.srcVsModel))} ${f1(limbMean(r.body.srcVsRt))}` : "  -     -  ";
    if (r.hands.length === 0) {
      console.log(`${r.name.padEnd(30)}${body}`);
      continue;
    }
    r.hands.forEach((h, i) => {
      const lead = i === 0 ? `${r.name.padEnd(30)}${body}` : " ".repeat(42);
      console.log(`${lead}    ${h.side}${h.labelSide !== h.side ? "!" : " "}  ${f1(h.srcVsModel.fingerFlex)} ${f1(h.srcVsRt?.fingerFlex)} ${f1(h.modelVsRt?.fingerFlex)}        ${f1(h.srcVsModel.thumbFlex)} ${f1(h.srcVsRt?.thumbFlex)}    ${f1(h.srcVsModel.palmNormal)} ${f1(h.srcVsRt?.palmNormal)}    ${String(h.srcVsModel.bendSignFlips).padStart(3)} ${String(h.srcVsRt?.bendSignFlips ?? "-").padStart(4)}    ${pct(h.srcVsModel2D)} ${pct(h.srcVsRt2D)} ${pct(h.modelVsRt2D)}`);
    });
  }
}

export interface Summary {
  cases: number;
  body: { srcVsModel: number; srcVsRt: number; p90SrcVsRt: number };
  hands: {
    n: number;
    rtDetected: number;
    fingerFlex: { srcVsModel: number; srcVsRt: number; modelVsRt: number };
    thumbFlex: { srcVsModel: number; srcVsRt: number };
    palmNormal: { srcVsModel: number; srcVsRt: number };
    handDir: { srcVsModel: number; srcVsRt: number };
    bendSignFlips: { srcVsModel: number; srcVsRt: number };
    /** 2D landmark residual, fraction of palm length. */
    shape2D: { srcVsModel: number; srcVsRt: number; modelVsRt: number };
  };
}

export function summarize(results: CaseResult[]): Summary {
  const bodies = results.flatMap(r => (r.body ? [r.body] : []));
  const hands = results.flatMap(r => r.hands);
  const rt = hands.filter(h => h.srcVsRt);
  const pick = (xs: (number | undefined)[]) => xs.filter((x): x is number => x !== undefined && !Number.isNaN(x));
  const rtLimbs = pick(bodies.map(b => limbMean(b.srcVsRt)));
  return {
    cases: results.length,
    body: {
      srcVsModel: mean(pick(bodies.map(b => limbMean(b.srcVsModel)))),
      srcVsRt: mean(rtLimbs),
      p90SrcVsRt: percentile(rtLimbs, 0.9),
    },
    hands: {
      n: hands.length,
      rtDetected: rt.length,
      fingerFlex: {
        srcVsModel: mean(hands.map(h => h.srcVsModel.fingerFlex)),
        srcVsRt: mean(rt.map(h => h.srcVsRt!.fingerFlex)),
        modelVsRt: mean(rt.map(h => h.modelVsRt!.fingerFlex)),
      },
      thumbFlex: {
        srcVsModel: mean(hands.map(h => h.srcVsModel.thumbFlex)),
        srcVsRt: mean(rt.map(h => h.srcVsRt!.thumbFlex)),
      },
      palmNormal: {
        srcVsModel: mean(hands.map(h => h.srcVsModel.palmNormal)),
        srcVsRt: mean(rt.map(h => h.srcVsRt!.palmNormal)),
      },
      handDir: {
        srcVsModel: mean(hands.map(h => h.srcVsModel.handDir)),
        srcVsRt: mean(rt.map(h => h.srcVsRt!.handDir)),
      },
      bendSignFlips: {
        srcVsModel: hands.reduce((s, h) => s + h.srcVsModel.bendSignFlips, 0),
        srcVsRt: rt.reduce((s, h) => s + h.srcVsRt!.bendSignFlips, 0),
      },
      shape2D: {
        srcVsModel: mean(pick(hands.map(h => h.srcVsModel2D))),
        srcVsRt: mean(pick(rt.map(h => h.srcVsRt2D))),
        modelVsRt: mean(pick(rt.map(h => h.modelVsRt2D))),
      },
    },
  };
}

// ── HTML ───────────────────────────────────────────────────────────────────

const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const POSE_EDGES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32]];

function overlay(lms: Landmark[] | undefined, edges: number[][], color: string): string {
  if (!lms) return "";
  return edges.map(([a, b]) =>
    `<line x1="${lms[a].x}" y1="${lms[a].y}" x2="${lms[b].x}" y2="${lms[b].y}" stroke="${color}" stroke-width="0.004"/>`,
  ).join("");
}

function figure(src: string, w: number, h: number, svg: string, caption: string): string {
  return `<figure><div class="fig" style="aspect-ratio:${w}/${h}"><img src="${src}"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${svg}</svg></div><figcaption>${caption}</figcaption></figure>`;
}

export async function writeReport(dir: string, results: CaseResult[], sources: Map<string, string>): Promise<void> {
  mkdirSync(resolve(dir, "img"), { recursive: true });
  const rows: string[] = [];
  for (const r of results) {
    const slug = r.name.replace(/[^a-z0-9]+/gi, "_");
    const srcPath = sources.get(r.name);
    const figs: string[] = [];
    if (srcPath) {
      const out = `img/${slug}_src_${basename(srcPath)}`;
      copyFileSync(srcPath, resolve(dir, out));
      const d = r.detection;
      const svg = overlay(d.pose?.image, POSE_EDGES, "#0f0") + d.hands.map(hd => overlay(hd.image, HAND_EDGES, hd.side === "L" ? "#4af" : "#f84")).join("");
      figs.push(figure(out, d.width, d.height, svg, "source + MediaPipe"));
    }
    if (r.body) {
      const out = `img/${slug}_body.png`;
      await Bun.write(resolve(dir, out), pngBytes(r.body.renderPng));
      const d = r.body.rtDetection;
      const svg = d ? overlay(d.pose?.image, POSE_EDGES, "#0f0") + d.hands.map(hd => overlay(hd.image, HAND_EDGES, hd.side === "L" ? "#4af" : "#f84")).join("") : "";
      figs.push(figure(out, 720, 1280, svg, "driven Anny + MediaPipe"));
    }
    for (const h of r.hands) {
      if (!h.renderPng) continue;
      const out = `img/${slug}_hand_${h.side}.png`;
      await Bun.write(resolve(dir, out), pngBytes(h.renderPng));
      const d = h.rtDetection;
      const svg = d ? d.hands.map(hd => overlay(hd.image, HAND_EDGES, hd.labelSide === h.side ? "#0f0" : "#f0f")).join("") : "";
      figs.push(figure(out, 1, 1, svg, `hand ${h.side} close-up (magenta = MP says other hand)`));
    }
    const handRows = r.hands.map(h => {
      const fmt = (rows: number[][]) => rows.map(row => row.map(x => x.toFixed(0)).join("/")).join(" · ");
      return `<tr><td>${h.side} (label ${h.labelSide}, geom ${h.geomSide ?? "?"})</td>
        <td>src ${fmt(h.src.flex)}<br>mdl ${fmt(h.model.flex)}<br>rt&nbsp; ${h.rt ? fmt(h.rt.flex) : "—"}</td>
        <td>${h.srcVsModel.fingerFlex.toFixed(1)} / ${h.srcVsRt?.fingerFlex.toFixed(1) ?? "—"}</td>
        <td>${h.srcVsModel.palmNormal.toFixed(1)} / ${h.srcVsRt?.palmNormal.toFixed(1) ?? "—"}</td>
        <td>${h.srcVsModel.bendSignFlips} / ${h.srcVsRt?.bendSignFlips ?? "—"}</td></tr>`;
    }).join("");
    const body = r.body ? Object.entries(r.body.srcVsRt ?? r.body.srcVsModel).map(([k, v]) => `${k} ${v.toFixed(0)}°`).join(", ") : "";
    rows.push(`<section><h2>${r.name}</h2><div class="figs">${figs.join("")}</div>
      <p class="notes">${r.notes.join("; ")}</p>
      ${body ? `<p>body src↔rt: ${body}</p>` : ""}
      ${handRows ? `<table><tr><th>hand</th><th>flex (thumb · index · middle · ring · pinky)</th><th>fingers° s↔m / s↔rt</th><th>palm° s↔m / s↔rt</th><th>sign flips</th></tr>${handRows}</table>` : ""}
    </section>`);
  }
  const html = `<!doctype html><meta charset="utf-8"><title>anny-js round trip</title>
<style>body{font:13px system-ui;background:#111;color:#ddd;margin:20px}section{border-top:1px solid #333;padding:10px 0}
.figs{display:flex;gap:10px;align-items:flex-start}figure{margin:0}.fig{position:relative;height:320px}.fig img{height:100%;display:block}
.fig svg{position:absolute;inset:0;width:100%;height:100%}figcaption{color:#888}table{border-collapse:collapse}td,th{border:1px solid #333;padding:3px 6px;vertical-align:top;font-family:monospace}.notes{color:#d88}</style>
<h1>anny-js round trip</h1><pre>${JSON.stringify(summarize(results), null, 1)}</pre>${rows.join("")}`;
  await Bun.write(resolve(dir, "index.html"), html);
}

/** Grid of synthetic renders with MediaPipe overlays and per-case errors. */
export async function writeSyntheticReport(
  dir: string,
  rows: { name: string; png: string; image?: { x: number; y: number }[]; caption: string }[],
): Promise<void> {
  mkdirSync(resolve(dir, "img"), { recursive: true });
  const figs: string[] = [];
  for (const r of rows) {
    const out = `img/${r.name}.png`;
    await Bun.write(resolve(dir, out), pngBytes(r.png));
    const svg = r.image ? overlay(r.image as Landmark[], HAND_EDGES, "#0f0") : "";
    figs.push(figure(out, 1, 1, svg, `${r.name}<br>${r.caption}`));
  }
  await Bun.write(resolve(dir, "index.html"), `<!doctype html><meta charset="utf-8"><title>anny-js synthetic hands</title>
<style>body{font:12px system-ui;background:#111;color:#ddd;margin:16px}.figs{display:flex;flex-wrap:wrap;gap:8px}
figure{margin:0;width:220px}.fig{position:relative;height:220px}.fig img{height:100%;display:block}
.fig svg{position:absolute;inset:0;width:100%;height:100%}figcaption{color:#999;font-family:monospace}</style>
<div class="figs">${figs.join("")}</div>`);
}
