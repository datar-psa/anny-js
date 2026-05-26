# anny-js

**A browser-native JavaScript runtime for [Anny](https://github.com/naver/anny)** — Naver's open-source parametric 3D human body model (Apache 2.0) — **plus a MediaPipe Pose + Hand driver** that maps live landmarks onto Anny's 163-bone rig.

Two pillars:

1. **Anny in the browser.** The canonical Anny package is Python + PyTorch. `anny-js` ports the runtime — forward kinematics and linear blend skinning over a 13 718-vertex / 163-bone mesh — to TypeScript, with no Python, no native modules, no server. The implementation is parity-tested against the reference Python `anny`: bone transforms agree within 1e-5, vertices within 0.1 mm.
2. **MediaPipe → Anny.** A driver that turns 33 pose landmarks + 21×2 hand landmarks per frame into per-bone rotation deltas — parent-corrected so child bones see the true parent rotation, angle-clamped so a single bad landmark doesn't crumple the rig, and bundled with proximity-based left/right hand assignment.

End-to-end (MediaPipe pose → posed mesh) is **< 6 ms** on integrated GPU. The library itself is a 1.4 MB binary asset + ~15 KB of TypeScript.

**[▶ Animation demo](https://clad.you/anny-js/demo/anim.html)**  ·  **[📷 Live pose demo](https://clad.you/anny-js/demo/live.html)**  ·  **[🎞 Pre-recorded video demo](https://clad.you/anny-js/demo/video.html)**

---

## What is Anny?

[Anny](https://github.com/naver/anny) is Naver's open-source parametric 3D human body model — Apache 2.0, commercial use allowed. Think SMPL/SMPL-X but with a clean permissive license. Anny has:

- **13 718 vertices**, 163 bones, triangulated mesh
- **11 semantic shape parameters** (gender, height, weight, muscle, proportions…)
- **256 local blendshapes** for detailed shape control
- Full LBS skinning with 8 influence bones per vertex
- A Blender-compatible rig (bone rolls, parent hierarchy, A-pose)

The Naver release ships as a Python/PyTorch package — great for training and offline rendering, but unusable in the browser. `anny-js` fills that gap:

- **Anny runtime in TypeScript** — `forwardKinematics` + `lbs` parity-tested against the reference `anny.utils.kinematics` and `anny.skinning.skinning`.
- **MediaPipe Pose + Hand driver** — 33 + 42 landmarks → 163 bone deltas, with the spine-root frame, hand assignment, and angle-clamping handled.
- **Renderers** — a Canvas 2D renderer for quick previews, plus the WebGL pipeline used by the demos (back-face culled, depth-tested, 4 camera views per draw call).
- **Ergonomic posing API** — `new Pose(model).set(boneName, rot)…` for custom animation; `landmarksToPoseDeltas` for the MP path.

The included `assets/anny_model.bin` is the average Anny body (neutral phenotype, A-pose) baked once into the wire format `anny-js` reads. Want a different body? Re-bake with different phenotype params — see [Regenerating the model binary](#regenerating-the-model-binary) below.

---

## Installation

```bash
npm install anny-js
```

**CDN** (no bundler):

```html
<script type="module">
  import { loadAnnyModel, lbs, forwardKinematics }
    from 'https://cdn.jsdelivr.net/npm/anny-js/dist/index.js';
</script>
```

---

## Package layout

Two subpath imports, each tree-shakeable:

| Import | Contains |
|---|---|
| `anny-js/anny` | Model loading, forward kinematics, linear blend skinning, 2D renderer, `Pose` |
| `anny-js/mediapipe` | MediaPipe Pose + Hand → Anny bone deltas |
| `anny-js` | Everything from both, for the common case |

If you only want headless animation, importing from `anny-js/anny` keeps the ~30 KB MediaPipe driver out of your bundle.

---

## Quick start — custom animation (`Pose` API)

```ts
import {
  loadAnnyModel,
  Pose, allocBoneTransforms, forwardKinematics,
  lbs, allocVertexBuffer,
  rodriguesToMat3,
} from 'anny-js/anny';

// 1. Load model once (~20 ms, 1.4 MB)
const model = await loadAnnyModel('/anny_model.json', '/anny_model.bin');

// Allocate reusable frame buffers
const boneXforms = allocBoneTransforms(model.boneCount);  // (163×16) Float32Array
const vertBuf    = allocVertexBuffer(model);              // (13718×3) Float32Array

// 2. Build a pose — both arms raised 90°
const up = rodriguesToMat3(new Float32Array([0, 0, 1]), Math.PI / 2);
const pose = new Pose(model)
  .set('upperarm01.L', up)
  .set('upperarm01.R', up);

// 3. FK → LBS → posed vertices every frame
forwardKinematics(model, pose.deltas, boneXforms);
const { vertices, faces } = lbs(model, boneXforms, vertBuf);
// vertices: Float32Array (V×3, Anny Z-up metres)
// faces:    Int32Array   (F×3)
```

Hot loops can skip the `Pose` wrapper entirely and use the underlying functional API (`identityDeltas` + `setDelta` + `setDeltas`) — see "Posing" below.

---

## Quick start — drive from MediaPipe Pose

```ts
import {
  loadAnnyModel,
  allocBoneTransforms, lbs, allocVertexBuffer, forwardKinematics,
} from 'anny-js/anny';
import { landmarksToPoseDeltas } from 'anny-js/mediapipe';

const model     = await loadAnnyModel('/anny_model.json', '/anny_model.bin');
const boneXf    = allocBoneTransforms(model.boneCount);
const vertBuf   = allocVertexBuffer(model);

// In your MediaPipe Pose detection loop:
function onLandmarks(landmarks) {         // 33 × {x,y,z,visibility}
  const deltas = landmarksToPoseDeltas({ pose: landmarks }, model);
  forwardKinematics(model, deltas, boneXf);
  const { vertices, faces } = lbs(model, boneXf, vertBuf);
  // → feed into WebGL / Three.js / your renderer
}
```

See [`demo/live.html`](demo/live.html) for the full 4-view WebGL demo with recording.

> **Hands:** pose tracking is solid; per-finger hand tracking from full-body
> footage is unreliable (the hand is a few % of frame) and is **disabled by
> default** in the video demo. The whole hand pipeline is kept intact and
> re-enables with one flag. The investigation — MediaPipe's hand-local
> coordinate frame, the rig's palmar/dorsal convention, the natural curl
> synergy, and how to verify hands numerically — is written up in
> [`docs/hand-tracking.md`](docs/hand-tracking.md).

---

## Demos

Three demos, each isolating one capability:

| | Demo | What it demonstrates |
|--|------|----------------------|
| 🎬 | [`demo/anim.html`](demo/anim.html) | **The Anny runtime, no MediaPipe.** A procedural backflip drives the rig directly with rotation matrices — proves FK + LBS work in the browser at 30 fps with 4 simultaneous camera views. No webcam, no inference. |
| 📷 | [`demo/live.html`](demo/live.html) | **Anny + MediaPipe, live.** Your webcam → MediaPipe Pose + Hand → Anny. All 163 bones (including 38 finger bones) drive in real time. 4-view WebGL render + WebM recorder. |
| 🎞 | [`demo/video.html`](demo/video.html) | **Anny + MediaPipe, offline.** Same pipeline as `live`, but on bundled clips with MediaPipe's *heavy* pose model — useful for inspecting driver quality without webcam noise. |

**Run locally:**

```bash
git clone https://github.com/datar-psa/anny-js
cd anny-js && bun install
bun run demo    # Vite dev server with HMR — http://localhost:3000
```

---

## API reference

### Loading

```ts
loadAnnyModel(manifestUrl: string, binUrl: string): Promise<AnnyModel>
```

Fetches the binary model. ~20 ms cold, instant from browser cache. Call once.

---

### Bone access

```ts
buildBoneIndex(model): Map<string, number>   // build once, reuse
```

Key bone names: `spine01`, `neck01`, `head`, `upperarm01.L/R`, `lowerarm01.L/R`,
`wrist.L/R`, `upperleg01.L/R`, `lowerleg01.L/R`.

---

### Posing

**Recommended: the `Pose` class.** Holds the underlying `PoseDeltas` array plus
a cached bone index, supports method chaining, and won't allocate beyond the
constructor:

```ts
const pose = new Pose(model)
  .set('upperarm01.L', rot)         // silent no-op on unknown bones
  .setStrict('spine03', rot)        // throws on unknown bones
  .setMany([{ bone: 'neck01', rot }, ...])
  .reset();                         // back to rest

forwardKinematics(model, pose.deltas, boneXforms);
```

**Low-level functional API**, when you already manage the `PoseDeltas` array
yourself or want the absolute fastest path:

```ts
// Array of per-bone local delta rotations (3×3 Float32Array or null = identity)
identityDeltas(boneCount: number): PoseDeltas

setDelta(deltas, model, 'upperarm01.L', rot3x3Float32)
setDeltas(deltas, model, [{ bone: 'spine01', rot }, ...])
```

**From MediaPipe Pose landmarks** (parent-corrected, world-coord deltas):

```ts
landmarksToPoseDeltas(
  input: PoseInput,                    // { pose, leftHand?, rightHand?, mirrorX?, visibilityMin?, maxAngleRad? }
  model: AnnyModel,
  boneIndex?: Map<string, number>,     // optional — built on demand if omitted
): PoseDeltas
```

---

### Kinematics + skinning

```ts
// Forward kinematics — updates boneXforms in-place
forwardKinematics(model, deltas, boneXforms: Float32Array): void

// Linear blend skinning — returns posed vertices
lbs(model, boneXforms, outVertices?: Float32Array): SkinnedMesh
// SkinnedMesh: { vertices: Float32Array (V×3), faces: Int32Array (F×3) }
```

Both are **zero-allocation** when called with pre-allocated output buffers (see `allocBoneTransforms` / `allocVertexBuffer`). Call every frame without GC pressure.

---

### Canvas 2D renderer

```ts
renderAnny(ctx: CanvasRenderingContext2D, mesh: SkinnedMesh, opts: RenderOptions): void

type RenderOptions = {
  cx: number;         // canvas X of body centre
  footY: number;      // canvas Y of feet
  scale: number;      // pixels per metre (body height ≈ 1.7 m)
  color: string;      // fill colour
  outlineColor?: string;
  flipX?: boolean;    // mirror for self-view (default false)
  viewRotY?: number;  // horizontal orbit angle (radians)
  viewRotX?: number;  // vertical tilt (negative = bird's-eye)
}
```

For production use the **WebGL path** in the demos (4 draw calls for 4 views).

---

## Performance

Measured on Intel Iris Xe (integrated, 12th-gen laptop):

| Operation | Time |
|-----------|------|
| Model load | ~20 ms (cold), instant (cached) |
| FK (163 bones) | ~0.4 ms |
| LBS (13 718 verts, 8 weights/vert) | ~1.2 ms |
| WebGL 4-view render | ~4 ms |
| **End-to-end (pose → render)** | **< 6 ms** |
| **Frame rate** | **30 fps (GPU-limited by MediaPipe)** |

---

## Parity with Python anny

The runtime is verified against the canonical Python `anny` package via golden fixtures committed under [`tests/parity/fixtures/`](tests/parity/fixtures/). A Python harness ([`tools/gen_parity_fixtures.py`](tools/gen_parity_fixtures.py)) loads the same `anny_model.{json,bin}` this library ships, runs `anny.utils.kinematics.forward_kinematic` + `anny.skinning.skinning.linear_blend_skinning` on a handful of canonical poses (rest, arms-up, knee-bend, torso twist, compound), and dumps the expected outputs. The JS suite asserts:

- Bone transforms agree to `‖js - py‖∞ < 1e-5`
- Posed vertices agree to `‖js - py‖₂ < 0.1 mm`

Re-generate fixtures (only needed if the Python `anny` package changes its algorithm):

```bash
source /path/to/vton-exp/hmr/body-tuning/venv/bin/activate
task fixtures   # → tests/parity/fixtures/*.json
```

The MediaPipe driver has its own parity test ([`tests/mediapipe/landmarks-parity.test.ts`](tests/mediapipe/landmarks-parity.test.ts)) that compares each posed bone's axis against the MediaPipe-derived direction. It runs on synthetic A-pose landmarks (mean error must stay < 0.5°) and on optional video-captured fixtures (regenerate via `task fixtures:landmarks`).

---

## Regenerating the model binary

The included binary is the average Anny body (neutral parameters, A-pose). To rebake with different phenotype params:

```bash
# Requires the anny Python package (from vton-exp's body-tuning venv)
source /path/to/venv/bin/activate
python tools/extract_anny.py --out assets/ \
  --params '{"gender":0.5,"height":0.55,"weight":0.4}'
```

Once you have a custom binary, re-run `task fixtures` to refresh the parity goldens against the new mesh.

---

## License

**anny-js**: Apache 2.0  
**Anny body model** (Naver): Apache 2.0  
`assets/anny_model.bin` is derived from Anny and inherits Apache 2.0.
