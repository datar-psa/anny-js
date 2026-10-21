/**
 * Temporal naturalness of a joint-angle sequence (degrees per frame).
 * Real finger motion is smooth: at 24–30 fps joint angles change by a few
 * degrees per frame, and accelerations are small. Noise shows up as jitter
 * (second differences); bad filtering shows up as freezes followed by snaps.
 */

export interface TemporalScore {
  /** RMS second difference over all joints and frames (deg/frame²). */
  jitter: number;
  /** Frames with any joint moving more than 30° from the previous frame. */
  snaps: number;
  /** Longest run of frames with all joints exactly unchanged while driven. */
  longestFreeze: number;
}

/** `frames[t]` = flattened joint angles at frame t, or null when not driven. */
export function temporalScore(frames: (number[] | null)[]): TemporalScore {
  let sq = 0, n = 0, snaps = 0, freeze = 0, longest = 0;
  for (let t = 1; t < frames.length; t++) {
    const a = frames[t], b = frames[t - 1];
    if (!a || !b) { freeze = 0; continue; }
    let maxStep = 0, same = true;
    a.forEach((x, j) => {
      const d = Math.abs(x - b[j]);
      maxStep = Math.max(maxStep, d);
      if (d > 1e-6) same = false;
    });
    if (maxStep > 30) snaps++;
    freeze = same ? freeze + 1 : 0;
    longest = Math.max(longest, freeze);
    const c = frames[t - 2];
    if (t >= 2 && c) a.forEach((x, j) => { const d2 = x - 2 * b[j] + c[j]; sq += d2 * d2; n++; });
  }
  return { jitter: n ? Math.sqrt(sq / n) : 0, snaps, longestFreeze: longest };
}
