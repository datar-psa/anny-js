/**
 * Gaussian fusion of MediaPipe readings with a prior over natural poses.
 *
 * One model serves single images and video:
 *
 *   prior        x ~ N(μ, Σ)                 how hands (wrists, …) usually are
 *   measurement  z = x + ε,  ε ~ N(0, R(q))   R grows as the hand gets smaller
 *   single frame x̂ = μ + Σ(Σ + R)⁻¹(z − μ)    (`gaussianUpdate` from the prior)
 *   video        the prior becomes an Ornstein–Uhlenbeck process: between
 *                frames x relaxes toward μ with time constant τ and gains
 *                exactly enough noise to keep N(μ, Σ) stationary
 *                (`ouPredict`), then the same update — a Kalman filter.
 *
 * With a correlated Σ (finger synergies: fingers curl together, DIP follows
 * PIP) a noisy reading is pulled toward the nearest *natural* hand, not just
 * toward the mean — independent finger detail survives only as far as the
 * measurement can support it.
 *
 * Matrices are row-major `number[]`; dimensions are small (≤ 16).
 */

export interface Gaussian {
  mean: number[];
  /** Row-major n×n covariance. */
  cov: number[];
}

/** Cholesky factor L (row-major, lower) of SPD `a`; throws if not positive definite. */
function cholesky(a: readonly number[], n: number): number[] {
  const L = new Array<number>(n * n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) throw new Error("fusion: covariance is not positive definite");
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return L;
}

/** Solve (L Lᵀ) x = b in place. */
function cholSolve(L: readonly number[], n: number, b: number[]): number[] {
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * b[k];
    b[i] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * b[k];
    b[i] = s / L[i * n + i];
  }
  return b;
}

/**
 * Posterior of x given belief N(m, P) and a measurement z with noise
 * covariance R: the Kalman update.
 */
export function gaussianUpdate(belief: Gaussian, z: readonly number[], R: readonly number[]): Gaussian {
  const n = z.length;
  const { mean: m, cov: P } = belief;
  const S = P.map((p, i) => p + R[i]);
  const L = cholesky(S, n);
  const innov = cholSolve(L, n, z.map((v, i) => v - m[i]));
  // K = P S⁻¹; mean = m + P S⁻¹ (z − m); cov = P − P S⁻¹ P.
  const mean = m.map((mi, i) => {
    let s = mi;
    for (let k = 0; k < n; k++) s += P[i * n + k] * innov[k];
    return s;
  });
  const SinvP: number[] = new Array(n * n);
  for (let c = 0; c < n; c++) {
    const col = cholSolve(L, n, Array.from({ length: n }, (_, r) => P[r * n + c]));
    for (let r = 0; r < n; r++) SinvP[r * n + c] = col[r];
  }
  const cov = P.map((p, idx) => {
    const i = Math.floor(idx / n), j = idx % n;
    let s = p;
    for (let k = 0; k < n; k++) s -= P[i * n + k] * SinvP[k * n + j];
    return s;
  });
  return { mean, cov };
}

/**
 * Update with one scalar linear observation hᵀx = z (noise variance r) — a
 * soft constraint when z is what physiology says hᵀx should be.
 */
export function linearUpdate(belief: Gaussian, h: readonly number[], z: number, r: number): Gaussian {
  const n = h.length;
  const { mean: m, cov: P } = belief;
  const Ph = Array.from({ length: n }, (_, i) => h.reduce((s, hk, k) => s + P[i * n + k] * hk, 0));
  const S = h.reduce((s, hi, i) => s + hi * Ph[i], 0) + r;
  const innov = z - h.reduce((s, hi, i) => s + hi * m[i], 0);
  return {
    mean: m.map((mi, i) => mi + Ph[i] * innov / S),
    cov: P.map((p, idx) => p - Ph[Math.floor(idx / n)] * Ph[idx % n] / S),
  };
}

/** Mean of `gaussianUpdate` alone (skips the covariance). */
export function posteriorMean(belief: Gaussian, z: readonly number[], R: readonly number[]): number[] {
  const n = z.length;
  const { mean: m, cov: P } = belief;
  const innov = cholSolve(cholesky(P.map((p, i) => p + R[i]), n), n, z.map((v, i) => v - m[i]));
  return m.map((mi, i) => {
    let s = mi;
    for (let k = 0; k < n; k++) s += P[i * n + k] * innov[k];
    return s;
  });
}

/**
 * Ornstein–Uhlenbeck prediction over `dt` seconds: the belief relaxes toward
 * the prior with time constant `tau`, keeping the prior stationary.
 */
export function ouPredict(belief: Gaussian, prior: Gaussian, dt: number, tau: number): Gaussian {
  const a = Math.exp(-Math.max(0, dt) / tau);
  return {
    mean: belief.mean.map((m, i) => prior.mean[i] + a * (m - prior.mean[i])),
    cov: belief.cov.map((p, i) => a * a * p + (1 - a * a) * prior.cov[i]),
  };
}

/** Squared Mahalanobis distance of z from belief N(m, P) under extra noise R. */
export function mahalanobis2(belief: Gaussian, z: readonly number[], R: readonly number[]): number {
  const n = z.length;
  const L = cholesky(belief.cov.map((p, i) => p + R[i]), n);
  const d = z.map((v, i) => v - belief.mean[i]);
  const x = cholSolve(L, n, [...d]);
  return d.reduce((s, v, i) => s + v * x[i], 0);
}

/** Diagonal covariance from per-axis variances. */
export function diagonal(vars: readonly number[]): number[] {
  const n = vars.length;
  const out = new Array<number>(n * n).fill(0);
  vars.forEach((v, i) => { out[i * n + i] = v; });
  return out;
}
