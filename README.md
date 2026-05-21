# anny-js

**Live 3D body animation entirely in the browser** — powered by [Anny](https://github.com/naver/anny) (Naver, Apache 2.0).

No server. No Python. No native modules. Just a TypeScript library and a 1.4 MB binary.

FK + LBS on a **14 718-vertex / 163-bone** parametric body in **< 2 ms**.  
WebGL 4-view rendering at **30 fps** on integrated GPU.

**[▶ Animation demo](https://clad.you/anny-js/demo/anim.html)**  ·  **[📷 Live pose demo](https://clad.you/anny-js/demo/live.html)**

---

## What is Anny?

[Anny](https://github.com/naver/anny) is Naver's open-source parametric 3D human body model — Apache 2.0, commercial use allowed. Think SMPL/SMPL-X but with a clean permissive license. Anny has:

- **14 718 vertices**, 163 bones, triangulated mesh
- **11 semantic shape parameters** (gender, height, weight, muscle, proportions…)
- **256 local blendshapes** for detailed shape control
- Full LBS skinning with 8 influence bones per vertex
- A Blender-compatible rig (bone rolls, parent hierarchy, A-pose)

**anny-js** exposes Anny's forward kinematics and linear blend skinning as a clean browser API, plus:
- A **MediaPipe Pose** driver (33 landmarks → 163 bone deltas, parent-corrected)
- A **WebGL renderer** (back-face culled, depth-tested, 4-camera multi-view)
- An **animation API** to drive any bones with rotation matrices

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

## Quick start — custom animation

```ts
import {
  loadAnnyModel, buildBoneIndex,
  allocBoneTransforms, identityDeltas, setDeltas,
  lbs, allocVertexBuffer, rodriguesToMat3,
} from 'anny-js';

// 1. Load model once (~20 ms, 1.4 MB)
const model     = await loadAnnyModel('/anny_model.json', '/anny_model.bin');
const boneIndex = buildBoneIndex(model);

// Allocate reusable frame buffers
const boneXforms = allocBoneTransforms(model.boneCount);  // (163×16) Float32Array
const vertBuf    = allocVertexBuffer(model);              // (14718×3) Float32Array

// 2. Build a pose — both arms raised 90°
const deltas = identityDeltas(model.boneCount);
const up = rodriguesToMat3(new Float32Array([0, 0, 1]), Math.PI / 2);
setDeltas(deltas, model, [
  { bone: 'upperarm01.L', rot: up },
  { bone: 'upperarm01.R', rot: up },
]);

// 3. FK → LBS → raw vertex positions every frame
forwardKinematics(model, deltas, boneXforms);
const { vertices, faces } = lbs(model, boneXforms, vertBuf);
// vertices: Float32Array (V×3, Anny Z-up metres)
// faces:    Int32Array   (F×3)
```

---

## Quick start — drive from MediaPipe Pose

```ts
import {
  loadAnnyModel, buildBoneIndex,
  allocBoneTransforms, lbs, allocVertexBuffer,
  landmarksToPoseDeltas, forwardKinematics,
} from 'anny-js';

const model     = await loadAnnyModel('/anny_model.json', '/anny_model.bin');
const boneIndex = buildBoneIndex(model);
const boneXf    = allocBoneTransforms(model.boneCount);
const vertBuf   = allocVertexBuffer(model);

// In your MediaPipe Pose detection loop:
function onLandmarks(landmarks) {         // 33 × {x,y,z,visibility}
  const deltas = landmarksToPoseDeltas(landmarks, model, boneIndex);
  forwardKinematics(model, deltas, boneXf);
  const { vertices, faces } = lbs(model, boneXf, vertBuf);
  // → feed into WebGL / Three.js / your renderer
}
```

See [`demo/live.html`](demo/live.html) for the full 4-view WebGL demo with recording.

---

## Demos

| | Demo | Description |
|--|------|-------------|
| 🎬 | [`demo/anim.html`](demo/anim.html) | **No camera.** Pose cycle (A-pose → T-pose → guard → wave…), 4 simultaneous camera views (front, back, 3/4, above), smooth blending between poses. Pure performance showcase. |
| 📷 | [`demo/live.html`](demo/live.html) | **Webcam required.** MediaPipe Pose drives all 163 bones live. 4-view WebGL, skeleton overlay, built-in WebM recorder (⏺ REC button). |

**Run locally:**

```bash
git clone https://github.com/datar-psa/anny-js
cd anny-js && bun install
bun run demo    # http://localhost:3000
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

```ts
// Array of per-bone local delta rotations (3×3 Float32Array or null = identity)
identityDeltas(boneCount: number): PoseDeltas

// Single bone
setDelta(deltas, model, 'upperarm01.L', rot3x3Float32)

// Multiple at once
setDeltas(deltas, model, [{ bone: 'spine01', rot }, ...])

// From MediaPipe Pose landmarks (parent-corrected, XZ-projected for depth stability)
landmarksToPoseDeltas(
  landmarks: Landmark[],   // 33 × {x,y,z?,visibility?}
  model: AnnyModel,
  boneIndex: Map<string, number>,
  mirrorX?: boolean        // default false (standard front-facing webcam)
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

### Rotation helpers

```ts
rodriguesToMat3(axis: Float32Array, angle: number): Float32Array  // axis-angle → 3×3
rotFromTo(from: Float32Array, to: Float32Array):    Float32Array  // shortest arc rotation
```

---

## Performance

Measured on Intel Iris Xe (integrated, 12th-gen laptop):

| Operation | Time |
|-----------|------|
| Model load | ~20 ms (cold), instant (cached) |
| FK (163 bones) | ~0.4 ms |
| LBS (14 718 verts, 8 weights/vert) | ~1.2 ms |
| WebGL 4-view render | ~4 ms |
| **End-to-end (pose → render)** | **< 6 ms** |
| **Frame rate** | **30 fps (GPU-limited by MediaPipe)** |

---

## Regenerating the model binary

The included binary is the average Anny body (neutral parameters, A-pose). To rebake:

```bash
# Requires anny Python package (from vton-exp venv)
source /path/to/venv/bin/activate
python tools/extract_anny.py --out assets/ \
  --params '{"gender":0.5,"height":0.55,"weight":0.4}'
```

---

## License

**anny-js**: Apache 2.0  
**Anny body model** (Naver): Apache 2.0  
`assets/anny_model.bin` is derived from Anny and inherits Apache 2.0.
