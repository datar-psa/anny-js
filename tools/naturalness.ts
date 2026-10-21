#!/usr/bin/env bun
/**
 * Hand naturalness: validate the metrics, then score the driver.
 *
 *   bun tools/naturalness.ts validate [--critic]   labelled poses → per-metric AUC
 *   bun tools/naturalness.ts sequences             demo pipeline replay on cached clips
 *                                                  (tools/capture_sequences.ts)
 */

import { parseArgs } from "node:util";

import { buildBoneIndex } from "../src/anny/index.js";
import { landmarksToPoseDeltas } from "../src/mediapipe/index.js";
import { loadFixtureModel } from "../tests/_helpers/model.js";
import { mean } from "../tests/roundtrip/metrics.js";
import { buildRigExtras } from "../tests/roundtrip/skeleton.js";
import { buildPhysicalRig } from "../tests/naturalness/physical.js";
import { jointPoseDeltas, labelledPoses, type PoseClass } from "../tests/naturalness/poses.js";
import { auc, NaturalnessScorer } from "../tests/naturalness/score.js";
import type { Driver, ReplayOptions, Sequence } from "../tests/naturalness/replay.js";
import { printSequences, scoreSequence, type SequenceScore } from "../tests/naturalness/sequences.js";

const { values: args, positionals } = parseArgs({
  allowPositionals: true,
  options: { critic: { type: "boolean", default: false } },
});

const model = await loadFixtureModel();
const boneIndex = buildBoneIndex(model);
const extras = buildRigExtras(model);
const scorer = new NaturalnessScorer(model, extras, buildPhysicalRig(model));

type Row = { cls: PoseClass; name: string; m: Record<string, number> };

async function validate(): Promise<void> {
  const poses = labelledPoses();
  const rows: Row[] = [];
  let critic: ((d: ReturnType<typeof jointPoseDeltas>, s: "L" | "R") => Promise<Record<string, number>>) | undefined;
  let close: (() => Promise<void>) | undefined;
  if (args.critic) {
    const { HarnessClient } = await import("../tests/roundtrip/client.js");
    const { makeContext } = await import("../tests/roundtrip/run.js");
    const { criticScore } = await import("../tests/naturalness/critic.js");
    const client = await HarnessClient.start({ poseModel: "lite" });
    const ctx = makeContext(client, model);
    critic = async (d, s) => {
      const c = await criticScore(ctx, d, s);
      return { criticDrift: c.drift, criticMissed: c.missed, criticWrongSide: c.wrongSide };
    };
    close = () => client.close();
  }
  for (const p of poses) {
    const d = jointPoseDeltas(p.pose, p.side, boneIndex, model.boneCount);
    const r = scorer.score(d, p.side);
    const m: Record<string, number> = {
      anatomy: r.anatomy.total, rom: r.anatomy.rom, dipPip: r.anatomy.dipPip, enslaving: r.anatomy.enslaving,
      abdFlexed: r.anatomy.abductionWhileFlexed,
      penetration: r.physical.penetrationMm, collapsed: r.physical.collapsed * 100, flipped: r.physical.flipped * 100,
      distorted: r.physical.distorted * 100, armRom: r.physical.armRom, twistAbs: Math.abs(r.physical.forearmTwist),
    };
    if (critic) Object.assign(m, await critic(d, p.side));
    rows.push({ cls: p.cls, name: p.name, m });
  }
  await close?.();
  const metrics = Object.keys(rows[0].m);
  const classes: PoseClass[] = ["natural", "independent", "noisy", "hook", "hyperextended", "crossed", "twisted"];
  console.log("mean per class");
  console.log("metric".padEnd(16) + classes.map(c => c.slice(0, 10).padStart(12)).join(""));
  for (const k of metrics) {
    console.log(k.padEnd(16) + classes.map(c => mean(rows.filter(r => r.cls === c).map(r => r.m[k])).toFixed(2).padStart(12)).join(""));
  }
  console.log("\nAUC natural vs class (1.0 = perfectly separates, 0.5 = chance)");
  console.log("metric".padEnd(16) + classes.slice(1).map(c => c.slice(0, 10).padStart(12)).join(""));
  const nat = (k: string) => rows.filter(r => r.cls === "natural").map(r => r.m[k]);
  for (const k of metrics) {
    console.log(k.padEnd(16) + classes.slice(1).map(c => auc(nat(k), rows.filter(r => r.cls === c).map(r => r.m[k])).toFixed(2).padStart(12)).join(""));
  }
  await Bun.write(".cache/naturalness-validate.json", JSON.stringify(rows));
}

/** Score the demo pipeline (replay of cached detections) on every frame of both clips. */
async function scoreClips(driver: Driver, opts: ReplayOptions = {}): Promise<SequenceScore[]> {
  const out: SequenceScore[] = [];
  for (const clip of ["dancer", "boxer"]) {
    const seq: Sequence = await Bun.file(`.cache/sequences/${clip}.json`).json();
    out.push(...scoreSequence(seq, { model, extras, scorer }, driver, { handsEveryOther: true, ...opts }));
  }
  return out;
}

if (import.meta.main) {
  const cmd = positionals[0] ?? "validate";
  if (cmd === "validate") await validate();
  if (cmd === "sequences") {
    printSequences(await scoreClips(landmarksToPoseDeltas, { quality: false }), "stateless driver, no hand quality (demo's old landmark filtering)");
    printSequences(await scoreClips(landmarksToPoseDeltas), "stateless driver + hand quality (demo's old landmark filtering)");
    printSequences(await scoreClips(landmarksToPoseDeltas, { hands: "tracker" }), "PoseTracker (what the demos run)");
  }
}
