/**
 * A hand reading fused with what hands are like.
 *
 * The driver keeps a belief per hand — the hand's shape (finger angles and
 * thumb, hand-prior.ts state) and its wrist relative to the forearm (roll,
 * flexion, deviation) — and updates it with each MediaPipe reading, trusting
 * the reading by how large the hand is in the image. For a single image the
 * belief starts as the natural-hand prior; under `PoseTracker` it is last
 * frame's posterior, relaxed toward the prior by the time elapsed (an
 * Ornstein–Uhlenbeck process whose stationary distribution is that prior).
 */

import { diagonal, gaussianUpdate, linearUpdate, mahalanobis2, ouPredict, type Gaussian } from "./fusion.js";
import { clamp, DEG, vnorm, type Vec3 } from "./geom.js";
import { HAND_NOISE, HAND_PRIOR, PALM_NOISE, REAL_FOOTAGE } from "./hand-prior.js";
import type { HandRig } from "./rig.js";
import type { HandQuality } from "./types.js";

/** Hand shape in the driver's terms: hand-geometry finger angles (radians) and palm-frame thumb directions. */
export interface HandShape {
  flex: number[][];
  abd: number[];
  thumb: Vec3[];
}

export interface HandBelief {
  /** Hand shape, in hand-prior.ts state units (degrees). */
  shape: Gaussian;
  /** Wrist [roll, flexion, deviation] relative to the forearm, radians. */
  wrist: Gaussian;
  /** Forearm out-of-image-plane angle settled last frame (radians), when known. */
  forearmDepth?: number;
}

/**
 * How wrists usually sit relative to the forearm (SD, radians; zero mean =
 * neutral): functional range-of-motion studies put everyday forearm rotation
 * within ±50°, wrist flexion/extension within ±40° and deviation within ±15°.
 */
const WRIST_PRIOR: Gaussian = { mean: [0, 0, 0], cov: diagonal([40 * DEG, 30 * DEG, 12 * DEG].map(s => s * s)) };

/** The belief before any reading: natural hands, neutral wrist. */
export function priorBelief(): HandBelief {
  return {
    shape: { mean: [...HAND_PRIOR.mean], cov: [...HAND_PRIOR.cov] },
    wrist: { mean: [...WRIST_PRIOR.mean], cov: [...WRIST_PRIOR.cov] },
  };
}

/** Time constants (s) with which an unobserved hand relaxes toward the prior. */
export interface RelaxTimes {
  shape: number;
  wrist: number;
}

/** Belief `dt` seconds later, before seeing the new frame. */
export function predictBelief(b: HandBelief, dt: number, tau: RelaxTimes): HandBelief {
  return {
    shape: ouPredict(b.shape, HAND_PRIOR, dt, tau.shape),
    wrist: ouPredict(b.wrist, WRIST_PRIOR, dt, tau.wrist),
    forearmDepth: b.forearmDepth,
  };
}

/** Palm length below which a reading carries no information at all. */
const MIN_PALM_PX = 1;
/** Real-footage calibration range (outside the measured palm sizes it's extrapolated). */
const FOOTAGE_RANGE = [0.25, 8] as const;

/** Render-fitted noise → real footage, κ(palmPx) (hand-prior.ts REAL_FOOTAGE). */
function footage(palmPx: number): number {
  const { px0, k0, exponent } = REAL_FOOTAGE;
  return clamp(k0 * (px0 / palmPx) ** exponent, FOOTAGE_RANGE[0], FOOTAGE_RANGE[1]);
}

/** MediaPipe's noise covariance on the hand-shape state for a hand of this size. */
export function shapeNoise(palmPx: number): number[] {
  const px = Math.max(MIN_PALM_PX, palmPx);
  const k = footage(px);
  const sd = HAND_NOISE.a.map((a, i) => Math.sqrt(k * (a + HAND_NOISE.b[i] / px)));
  const n = sd.length;
  return HAND_NOISE.corr.map((c, idx) => c * sd[Math.floor(idx / n)] * sd[idx % n]);
}

/**
 * MediaPipe's palm-orientation noise variances (rad²) about the world axes,
 * which are the camera's: [image right, view direction, image up].
 */
export function palmNoise(q: HandQuality): number[] {
  const px = Math.max(MIN_PALM_PX, q.palmPx);
  const n = q.labelAgrees ? PALM_NOISE.agree : PALM_NOISE.disagree;
  const k = footage(px);
  return [0, 1, 2].map(i => k * (n.a[i] + n.b[i] / px) * DEG * DEG);
}

export function shapeToState(m: HandShape, hr: HandRig): number[] {
  return [
    ...m.flex.flatMap((f, i) => f.map((x, j) => (x - hr.fingers[i].restFlex[j]) / DEG)),
    ...m.abd.map((x, i) => (x - hr.fingers[i].restAbd) / DEG),
    ...m.thumb.flatMap(d => d.map(x => x / DEG)),
  ];
}

export function stateToShape(x: readonly number[], hr: HandRig): HandShape {
  return {
    flex: hr.fingers.map((f, i) => f.restFlex.map((r, j) => r + x[3 * i + j] * DEG)),
    abd: hr.fingers.map((f, i) => f.restAbd + x[12 + i] * DEG),
    thumb: [0, 1, 2].map(k => vnorm([x[16 + 3 * k], x[17 + 3 * k], x[18 + 3 * k]])),
  };
}

/** Degrees of freedom of the Student-t measurement model. */
const ROBUST_NU = 4;

/**
 * Kalman update with a Student-t measurement: a reading far outside what
 * belief + noise can explain (MediaPipe's occasional blunders — a mirrored
 * hand, a finger snapped to the wrong knuckle) has its noise inflated by
 * its Mahalanobis distance instead of being trusted or dropped outright.
 */
export function robustUpdate(belief: Gaussian, z: readonly number[], R: readonly number[]): Gaussian {
  const d2 = mahalanobis2(belief, z, R);
  const w = (ROBUST_NU + z.length) / (ROBUST_NU + d2);
  return gaussianUpdate(belief, z, w < 1 ? R.map(r => r / w) : R);
}

/** DIP flexion per degree of PIP flexion in natural, unforced motion (tendon coupling). */
const DIP_PIP_RATIO = 2 / 3;
/** SD (degrees) of DIP around that coupling. */
const DIP_PIP_SD = 10;

/**
 * Tendon coupling as a soft constraint: in each finger the DIP follows the
 * PIP (DIP ≈ ⅔·PIP, in absolute terms) — a fact of the flexor tendons and
 * retinacular ligaments, not of any one pose. MediaPipe breaks it
 * systematically when a fist hides the fingertips (a 150° PIP with a
 * hyperextended tip); being a bias rather than noise, the noise model can't
 * undo that, the constraint can — the other joints move with it through the
 * prior's covariance.
 */
export function coupleFingertips(shape: Gaussian, hr: HandRig): Gaussian {
  let out = shape;
  hr.fingers.forEach((f, i) => {
    // Absolute angles: x + rest (degrees); DIP − r·PIP = 0 ⇔ hᵀx = r·restPIP − restDIP.
    const h = new Array<number>(shape.mean.length).fill(0);
    h[3 * i + 2] = 1;
    h[3 * i + 1] = -DIP_PIP_RATIO;
    const z = (DIP_PIP_RATIO * f.restFlex[1] - f.restFlex[2]) / DEG;
    out = linearUpdate(out, h, z, DIP_PIP_SD ** 2);
  });
  return out;
}
