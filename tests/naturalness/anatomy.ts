/**
 * Rule-based hand plausibility from joint angles (tests/roundtrip/metrics
 * `handAngles` convention: degrees, flexion palmar-positive).
 *
 * Each rule returns degrees of *excess* beyond what a real hand does, so 0 is
 * "nothing implausible" and the terms add up. Ranges and couplings are the
 * standard ones from the hand-biomechanics literature:
 *   • range of motion (passive limits, so firm voluntary poses still pass):
 *     MCP −25…100, PIP −5…120, DIP −15…90 (thumb MCP −25…70, IP −25…90)
 *   • DIP–PIP tendon coupling: DIP ≈ ⅔·PIP during natural flexion
 *     (Rijpkema & Girard 1991; Kamper et al. 2003) — a fully bent PIP with a
 *     straight fingertip ("hook" missing) or a curled tip on a straight PIP
 *     is not something people do unconsciously
 *   • finger enslaving: the ring finger is the least independent (Häger-Ross
 *     & Schieber 2000) — pointing or a V sign are fine, but a ring finger far
 *     from BOTH neighbours (straight between two curled fingers, or curled
 *     between two straight ones) is not
 *   • abduction collapses as the MCP flexes (collateral ligaments tighten)
 */

export interface AnatomyScore {
  /** Degrees outside active range of motion, summed over joints. */
  rom: number;
  /** DIP deviating from the ⅔·PIP coupling by more than a tolerance. */
  dipPip: number;
  /** Ring finger's total curl far from both neighbours. */
  enslaving: number;
  /** Abduction kept while the MCP is strongly flexed. */
  abductionWhileFlexed: number;
  /** Sum of the above. */
  total: number;
}

const ROM: ReadonlyArray<ReadonlyArray<[number, number]>> = [
  [[-90, 90], [-25, 70], [-25, 90]],          // thumb: [CMC elevation (unused), MCP, IP]
  [[-25, 100], [-5, 120], [-15, 90]],         // index
  [[-25, 100], [-5, 120], [-15, 90]],         // middle
  [[-25, 100], [-5, 120], [-15, 90]],         // ring
  [[-25, 100], [-5, 120], [-15, 90]],         // little
];

/** Allowed total-curl difference (MCP+PIP+DIP) of the ring finger from its nearer neighbour. */
const RING_TOL = 70;
const DIP_PIP_RATIO = 2 / 3;
const DIP_PIP_TOL = 25;

const excess = (x: number, lo: number, hi: number): number => Math.max(0, lo - x, x - hi);

export function anatomyScore(flex: number[][], abd: number[], restAbd: number[]): AnatomyScore {
  let rom = 0;
  flex.forEach((row, f) => row.forEach((a, j) => {
    if (f === 0 && j === 0) return;
    rom += excess(a, ROM[f][j][0], ROM[f][j][1]);
  }));
  let dipPip = 0;
  for (let f = 1; f <= 4; f++) {
    const [, pip, dip] = flex[f];
    dipPip += Math.max(0, Math.abs(dip - DIP_PIP_RATIO * Math.max(0, pip)) - DIP_PIP_TOL);
  }
  const curl = [1, 2, 3, 4].map(f => flex[f][0] + flex[f][1] + flex[f][2]);
  const enslaving = Math.max(0, Math.min(Math.abs(curl[2] - curl[1]), Math.abs(curl[2] - curl[3])) - RING_TOL);
  let abductionWhileFlexed = 0;
  for (let k = 0; k < 4; k++) {
    const mcp = flex[k + 1][0];
    const spread = Math.abs(abd[k] - restAbd[k]);
    const allowed = 25 * Math.max(0, 1 - Math.max(0, mcp - 30) / 60) + 5;   // 30° open → 5° at 90°
    abductionWhileFlexed += Math.max(0, spread - allowed);
  }
  return { rom, dipPip, enslaving, abductionWhileFlexed, total: rom + dipPip + enslaving + abductionWhileFlexed };
}
