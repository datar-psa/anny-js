import type { Landmark, WorldLandmark } from "../../src/mediapipe/index.js";

/**
 * Exponential moving average smoother for a landmark array. Mutates `acc` and
 * returns it (or allocates a clone on the first call). Use one accumulator per
 * stream (pose body, left hand, right hand) and one alpha per stream.
 *
 *   alpha = 1.0  → no smoothing (track fresh)
 *   alpha = 0.5  → light damping (works for body)
 *   alpha = 0.6  → slightly more responsive (fingers move fast)
 */
export function smoothLandmarks<T extends Landmark | WorldLandmark>(
  acc: T[] | null,
  fresh: T[],
  alpha: number,
): T[] {
  if (!acc) return fresh.map(lm => ({ ...lm }));
  for (let i = 0; i < fresh.length; i++) {
    const a = acc[i] as Landmark;
    const f = fresh[i] as Landmark;
    a.x += (f.x - a.x) * alpha;
    a.y += (f.y - a.y) * alpha;
    a.z = (a.z ?? 0) + ((f.z ?? 0) - (a.z ?? 0)) * alpha;
    a.visibility = (a.visibility ?? 1) + ((f.visibility ?? 1) - (a.visibility ?? 1)) * alpha;
  }
  return acc;
}
