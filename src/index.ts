/** anny-js public surface.
 *
 * Two domains, importable separately or together:
 *   • `anny-js/anny`      — core: model loading, FK, LBS, 2D renderer (no MP dep)
 *   • `anny-js/mediapipe` — MediaPipe Pose + Hand → Anny bone deltas
 *
 * Importing from `anny-js` root re-exports everything for back-compat. */

export * from "./anny/index.js";
export * from "./mediapipe/index.js";
