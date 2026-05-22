/** MediaPipe → Anny driver: turns MP Pose+Hand worldLandmarks into bone deltas. */

export { MP, MP_HAND } from "./constants.js";
export type { Landmark, WorldLandmark, PoseInput } from "./types.js";
export { landmarksToPoseDeltas } from "./landmarks.js";
export { assignHands } from "./hands.js";
