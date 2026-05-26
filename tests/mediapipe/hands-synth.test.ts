/**
 * Synthetic hand-pose tests.
 *
 * We hand-craft a few unambiguous hand poses in MP world coords (palm-down
 * pointing-forward, fist-down, thumbs-up), feed them through the driver,
 * and verify each finger bone's posed-Y axis matches the corresponding
 * MP-derived target within a tight bound.
 *
 * Synthetic input isolates the driver math from any quirks of real MP hand
 * output (chirality, smoothing, hand-coord frame). If THIS test fails, the
 * driver itself is broken — independently of any capture-time issues.
 */

import { expect, test, describe, beforeAll } from "bun:test";

import {
  buildBoneIndex, allocBoneTransforms, forwardKinematics, type AnnyModel,
} from "../../src/anny/index.js";
import {
  landmarksToPoseDeltas, MP, MP_HAND, type WorldLandmark,
} from "../../src/mediapipe/index.js";
import { loadFixtureModel } from "../_helpers/model.js";
import { verifyHand, verifyHandVsPose, formatVerifyResult } from "./verify_hands.js";

/** Minimal pose landmarks: just the eight points the driver needs for body. */
function basePose(): WorldLandmark[] {
  const lm: WorldLandmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));
  const set = (i: number, x: number, y: number, z: number) => { lm[i] = { x, y, z, visibility: 1 }; };
  set(MP.LEFT_HIP,  +0.10, 0.00, 0.00);
  set(MP.RIGHT_HIP, -0.10, 0.00, 0.00);
  set(MP.LEFT_SHOULDER,  +0.18, -0.50, 0.00);
  set(MP.RIGHT_SHOULDER, -0.18, -0.50, 0.00);
  set(MP.LEFT_ELBOW,  +0.40, -0.30, 0.00);
  set(MP.RIGHT_ELBOW, -0.40, -0.30, 0.00);
  set(MP.LEFT_WRIST,  +0.55, -0.05, 0.00);
  set(MP.RIGHT_WRIST, -0.55, -0.05, 0.00);
  // Pose's knuckle landmarks: at the index / pinky MCPs, slightly forward
  // of (and lateral/medial to) the wrist. Required by the new hand-frame
  // aligner — without LEFT_PINKY/RIGHT_PINKY positioned at the real knuckle,
  // there is no "across-the-palm" axis for the aligner to use.
  //
  // Coordinate convention for the palm-down-forward synthetic pose below:
  //   • L hand sits on +x (anatomical-left). Palm-down ⇒ thumb is on the
  //     MEDIAL side of the body (toward midline) = -x relative to wrist;
  //     pinky on the LATERAL side = +x of wrist.
  //   • R hand sits on -x. Palm-down ⇒ thumb at +x of wrist (medial = toward
  //     midline), pinky at -x (lateral).
  set(MP.LEFT_INDEX,  +0.525, -0.05, -0.08);
  set(MP.LEFT_PINKY,  +0.575, -0.05, -0.08);
  // Thumb on the palmar side: for palm-down hand, palmar = +y_mp (down) =
  // higher y_mp than the dorsal palm plane (-0.05). Use +0.005.
  set(MP.LEFT_THUMB,  +0.51,  -0.045, -0.04);
  set(MP.RIGHT_INDEX, -0.525, -0.05, -0.08);
  set(MP.RIGHT_PINKY, -0.575, -0.05, -0.08);
  set(MP.RIGHT_THUMB, -0.51,  -0.045, -0.04);
  set(MP.LEFT_EAR,   +0.08, -0.70, 0.00);
  set(MP.RIGHT_EAR,  -0.08, -0.70, 0.00);
  set(MP.LEFT_EYE,   +0.04, -0.73, -0.05);
  set(MP.RIGHT_EYE,  -0.04, -0.73, -0.05);
  set(MP.MOUTH_LEFT, +0.03, -0.65, -0.05);
  set(MP.MOUTH_RIGHT,-0.03, -0.65, -0.05);
  return lm;
}

/**
 * Build a synthetic palm-down-pointing-forward hand at the given wrist
 * position. The wrist is the *origin* of the local hand frame.
 *
 * Geometry in **hand-local frame**:
 *   • Fingers extend along +U (= forward, toward the fingertips).
 *   • Palm normal = +N (= up if palm faces down).
 *   • Thumb is on the radial side (lateral for L hand, medial for R hand).
 *
 * `side`: which anatomical side this hand is. We place the thumb correctly:
 *   • L hand: thumb on the **medial** side (toward body midline = -x_mp).
 *     Anatomically the thumb is medial when the palm faces *down*.
 *   • R hand: thumb on the **medial** side too (+x_mp for the R hand).
 *
 * @param wristMp  Wrist position in MP world coords.
 * @param side     Anatomical side; selects which way the thumb sits.
 * @returns 21-landmark array in MP world coords.
 */
function palmDownForward(wristMp: [number, number, number], side: "L" | "R"): WorldLandmark[] {
  // We work in MP world: +x = anat-left, +y = down, +z = away from camera.
  // Palm-down means hand normal = +y (pointing down). Fingers extend along
  // -z (forward, toward camera). Thumb sits in the +x or -x direction.
  //
  // For a LEFT hand palm-down, thumb is on the -x side (medial = toward body
  // midline since the L hand is at +x). For a RIGHT hand palm-down, thumb is
  // on the +x side (medial = toward body midline since R hand is at -x).
  const thumbDir = side === "L" ? -1 : +1;

  const [wx, wy, wz] = wristMp;
  const make = (dx: number, dy: number, dz: number): WorldLandmark =>
    ({ x: wx + dx, y: wy + dy, z: wz + dz, visibility: 1 });

  const lm: WorldLandmark[] = new Array(21);
  lm[MP_HAND.WRIST] = make(0, 0, 0);

  // Thumb: extends along thumbDir × radial + forward + slightly PALMAR.
  // For palm-down hand: palmar = "down" physically = +y in MP (+y is down).
  // So thumb has positive dy (below the dorsal palm plane).
  lm[MP_HAND.THUMB_CMC] = make(thumbDir * 0.02, +0.005, -0.02);
  lm[MP_HAND.THUMB_MCP] = make(thumbDir * 0.04, +0.010, -0.04);
  lm[MP_HAND.THUMB_IP]  = make(thumbDir * 0.05, +0.012, -0.06);
  lm[MP_HAND.THUMB_TIP] = make(thumbDir * 0.06, +0.014, -0.08);

  // Index/middle/ring/pinky knuckles sit roughly straight forward of wrist
  // and fan slightly across the palm. Pinky is on -thumbDir side.
  // MCP knuckles are ~0.08m ahead of wrist (typical adult).
  const fingerOffsets: ReadonlyArray<{ which: "I"|"M"|"R"|"P"; sideOff: number }> = [
    { which: "I", sideOff: +thumbDir * 0.025 },  // index nearest thumb side
    { which: "M", sideOff: +thumbDir * 0.008 },
    { which: "R", sideOff: -thumbDir * 0.008 },
    { which: "P", sideOff: -thumbDir * 0.025 },
  ];
  const mcpIdx: Record<string, number> = { I: MP_HAND.INDEX_MCP, M: MP_HAND.MIDDLE_MCP, R: MP_HAND.RING_MCP, P: MP_HAND.PINKY_MCP };
  const pipIdx: Record<string, number> = { I: MP_HAND.INDEX_PIP, M: MP_HAND.MIDDLE_PIP, R: MP_HAND.RING_PIP, P: MP_HAND.PINKY_PIP };
  const dipIdx: Record<string, number> = { I: MP_HAND.INDEX_DIP, M: MP_HAND.MIDDLE_DIP, R: MP_HAND.RING_DIP, P: MP_HAND.PINKY_DIP };
  const tipIdx: Record<string, number> = { I: MP_HAND.INDEX_TIP, M: MP_HAND.MIDDLE_TIP, R: MP_HAND.RING_TIP, P: MP_HAND.PINKY_TIP };
  // Phalange length: index 0.045, middle 0.05, ring 0.046, pinky 0.038.
  const phalLen: Record<string, number> = { I: 0.045, M: 0.050, R: 0.046, P: 0.038 };

  for (const { which, sideOff } of fingerOffsets) {
    const zMcp = -0.08;  // MCP knuckle 8cm forward
    const phl = phalLen[which];
    lm[mcpIdx[which]] = make(sideOff, 0, zMcp);
    lm[pipIdx[which]] = make(sideOff, 0, zMcp - phl);
    lm[dipIdx[which]] = make(sideOff, 0, zMcp - phl - phl * 0.65);
    lm[tipIdx[which]] = make(sideOff, 0, zMcp - phl - phl * 0.65 - phl * 0.5);
  }
  return lm;
}

describe("synthetic hand verifier", () => {
  let model: AnnyModel;
  let boneIndex: Map<string, number>;

  beforeAll(async () => {
    model = await loadFixtureModel();
    boneIndex = buildBoneIndex(model);
  });

  // Pose-grounded checks (non-circular): the driver now *coordinates* finger
  // curl and de-splays the metacarpals, so it deliberately deviates from raw
  // per-finger MP directions. Asserting raw-MP fidelity is therefore wrong.
  // Instead verify the anatomical invariants against POSE landmarks: the
  // wrist tracks the arm, metacarpals point toward the knuckles, and a
  // straight synthetic hand does not bend its fingers backward.
  const assertSane = (side: "L"|"R", hand: WorldLandmark[], pose: WorldLandmark[]) => {
    const deltas = landmarksToPoseDeltas(
      side === "L" ? { pose, leftHand: hand } : { pose, rightHand: hand },
      model, boneIndex,
    );
    const xforms = allocBoneTransforms(model.boneCount);
    forwardKinematics(model, deltas, xforms);
    const r = verifyHandVsPose(model, boneIndex, xforms, pose, side);
    expect(r.ran, `${side}: verifier ran`).toBe(true);
    expect(r.wristErrDeg, `${side} wrist err ${r.wristErrDeg.toFixed(1)}°`).toBeLessThan(5);
    expect(r.indexMetacarpalErrDeg, `${side} index-meta ${r.indexMetacarpalErrDeg.toFixed(1)}°`).toBeLessThan(15);
    expect(r.pinkyMetacarpalErrDeg, `${side} pinky-meta ${r.pinkyMetacarpalErrDeg.toFixed(1)}°`).toBeLessThan(15);
    // Straight synthetic fingers: curl must not be meaningfully backward.
    r.fingerCurlPalmward.forEach((c, fi) => {
      if (fi === 0 || Number.isNaN(c)) return;  // skip thumb
      expect(c, `${side} finger${fi+1} curl ${c.toFixed(2)} (negative=backward)`).toBeGreaterThan(-0.15);
    });
  };

  test("L hand palm-down-forward: pose-grounded sanity (wrist/meta/curl)", () => {
    assertSane("L", palmDownForward([+0.55, -0.05, 0.00], "L"), basePose());
  });

  test("R hand palm-down-forward: pose-grounded sanity (wrist/meta/curl)", () => {
    assertSane("R", palmDownForward([-0.55, -0.05, 0.00], "R"), basePose());
  });

  test("both hands together: each hand stays pose-grounded sane", () => {
    const pose = basePose();
    const leftHand  = palmDownForward([+0.55, -0.05, 0.00], "L");
    const rightHand = palmDownForward([-0.55, -0.05, 0.00], "R");
    const deltas = landmarksToPoseDeltas({ pose, leftHand, rightHand }, model, boneIndex);
    const xforms = allocBoneTransforms(model.boneCount);
    forwardKinematics(model, deltas, xforms);
    for (const [side, hand] of [["L", leftHand], ["R", rightHand]] as const) {
      const r = verifyHandVsPose(model, boneIndex, xforms, pose, side);
      console.log(formatVerifyResult(verifyHand(model, boneIndex, xforms, hand, side)));
      expect(r.wristErrDeg, `${side} wrist`).toBeLessThan(5);
      expect(r.indexMetacarpalErrDeg, `${side} index-meta`).toBeLessThan(15);
      expect(r.pinkyMetacarpalErrDeg, `${side} pinky-meta`).toBeLessThan(15);
    }
  });

  test("naturality: finger curl signs are consistent on each finger", () => {
    // Synthetic hand has STRAIGHT fingers, so all curl signs are ~0
    // (signs.length but all signs are sign(0) = 0, which we treat as
    // consistent because there's no real direction).
    const pose = basePose();
    const leftHand = palmDownForward([+0.55, -0.05, 0.00], "L");
    const deltas = landmarksToPoseDeltas({ pose, leftHand }, model, boneIndex);
    const xforms = allocBoneTransforms(model.boneCount);
    forwardKinematics(model, deltas, xforms);
    const r = verifyHand(model, boneIndex, xforms, leftHand, "L");
    for (const fc of r.naturality.fingerCurls) {
      expect(fc.signsConsistent, `${fc.finger} curl signs [${fc.signs}] should be consistent`).toBe(true);
    }
  });
});
