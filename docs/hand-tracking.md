# MediaPipe → Anny: hands, how they're driven, and how we know they're right

Hands are the hard part of driving Anny from MediaPipe. This document records
what we measured, the driver design that came out of it, and the automated
validation that keeps it honest. Everything below is reproducible with
`task validate` and gated by `task test:browser`.

**TL;DR** — The earlier hand pipeline was built on a wrong premise (that
MediaPipe's hand `worldLandmarks` live in a hand-local frame) and compensated
with pose-anchored re-alignment, a palmar clamp and a fixed curl synergy. All
three are gone. Hand landmarks are camera-aligned; the driver now orients the
palm directly from them (forearm takes the roll), transfers each finger's
joint angles onto the rig's own hinge axes, and feeds on MediaPipe's *image*
landmarks lifted to 3D, which are markedly more accurate than its
`worldLandmarks`. On the same 34 real images (10 labelled hand photos + 24
frames of the demo clips):

| | old driver | new driver |
|---|---:|---:|
| palm normal, MediaPipe vs driven rig | 80.7° | **1.5°** |
| palm normal, photo vs render (round trip) | 80.6° | **26.3°** |
| finger flexion, photo vs render | 29.4° | **15.3°** |
| fingers bent the opposite way, photo vs render | 21 joints | **5** |
| 2D landmark residual, photo vs render | 37 % of a palm | **12 %** |
| hands MediaPipe finds again on the render | 26 / 32 | **32 / 32** |

(The old driver could not drive hands without a pose, which inflates its
numbers on the four close-up photos; its full-body numbers were just as far
off.) The remaining round-trip error is dominated by MediaPipe re-reading our
renders — see "The noise floor" below.

---

## What we measured

### 1. Hand `worldLandmarks` ARE camera-aligned

The previous docs claimed MediaPipe emits hand world landmarks in a
hand-local canonical frame, so the driver re-aligned every hand through a
frame built from the *pose* model's coarse wrist/index/pinky/thumb points.

Measured on all ten MediaPipe test photos: the in-plane rotation between each
hand's world landmarks (x, y) and its image landmarks is −8°…+4°, and world z
correlates 0.9+ with image z. The landmarks are in camera axes — exactly what
`world_landmark_projection_calculator.cc` does (it undoes the crop's in-plane
rotation). Pose world landmarks are camera-aligned the same way (−1.7°…+1.3°
on eleven full-body images), so pose and hand landmarks share one frame. The
re-alignment was not just unnecessary but harmful: it flipped
its Z axis per frame depending on which side of the palm the pose's noisy
thumb point fell, producing an improper (mirroring) rotation — a mirrored
hand is precisely "fingers bending backward".

### 2. The handedness label is anatomical, and the 3D follows it

MediaPipe's docs say handedness "assumes a mirrored image — swap it if not".
For the Tasks API that's wrong for ordinary footage: on `left_hands.jpg`,
`right_hands.jpg` and `victory.jpg` (a right hand, palm to camera) the label is
the true anatomical side, and the 3D landmarks have that label's chirality
(the fingertips sit on the palm side the label implies). Renders of the rig's
known L/R hands agree.

When the label disagrees with the side you assign a hand to (pose-wrist
proximity), MediaPipe has fallen for the depth ambiguity — a left hand
palm-forward and a right hand palm-back have the same silhouette.
`matchHandChirality` reflects depth about the wrist, which keeps the image
projection and fixes the chirality. As a second line of defence the driver
itself depth-flips a hand whose fingers curl strongly *away* from the palm.

### 3. Image landmarks beat `worldLandmarks` for 3D

On 153 renders of known hand poses (synthetic ground truth, below):

| hand source | finger joints | thumb | palm normal | bend-direction flips |
|---|---:|---:|---:|---:|
| `worldLandmarks` | 32.2° | 28.9° | 26.2° | 94 |
| image landmarks, z × width | **21.3°** | **18.9°** | **16.5°** | **38** |
| image landmarks, z × 0.5 / × 1.5 | 24.3° / 22.5° | 20.4° / 18.8° | 22.8° / 17.9° | 39 / 37 |
| image x, y + world z | 26.6° | 23.9° | 19.9° | 80 |

The documented "z uses roughly the same scale as x" holds (fitted scale
0.88, IQR 0.64–1.17). `handLandmarks3D` does the lift. **ROI crops**: z is in
units of the *crop* width, so it must be rescaled to the frame width when
crop landmarks are mapped back (the old demo kept it as-is, making ROI hands
4–8× too deep).

### 4. Pose alone cannot orient a palm

Pose's wrist/index/pinky triangle, scored against MediaPipe Hand on the same
images: 77° mean palm-normal error — worse than simply letting the hand follow
the forearm (67°). The driver no longer tries; without hand landmarks the
wrist rides along with the forearm.

### 5. Rig facts (now probed, not hard-coded)

- Every phalanx (fingers and thumb, both hands) flexes toward the palm about
  its local **+X**; abduction is about local Z, **mirrored** between hands.
- Elbows and knees flex about the lower bone's local **+X** too.
- Limb bones' Y axes are **not** on the joint-to-joint line (upper arm 4.6°,
  thigh 10°): the old driver aligned Y and was off by that much. The new one
  aims each bone's offset to its child joint.

`src/mediapipe/rig.ts` derives all of this from the model at load time, and
`tests/mediapipe/conventions.test.ts` pins it.

---

## Driver design (`src/mediapipe/landmarks.ts`)

**Plan**: landmarks → world intents. Hip and shoulder *lines* (exact) with
hips→shoulders as up, a head frame from ears/eyes/mouth, limb segment
directions, and per hand: palm orientation (`palmFrame` of MediaPipe vs the
rig's rest palm) plus hand-local finger angles.

**Solve**: one pass over the bones, parents first, tracking world rotations.

- Trunk: root takes the hip frame; the hips→chest rotation is slerped over
  the five spine bones; torso→head over the neck.
- Upper arm / thigh: aim at the elbow / knee and twist so the joint's hinge
  matches the observed bend plane (faded out below 30° of bend, where the
  plane is noise; for knees a backward-pointing plane is treated as a
  hyperextension reading, not a 180° hip rotation).
- Forearm: aim at the wrist, then take the palm's roll about the forearm
  (swing/twist split of the wrist target) as pronation/supination, half on
  each forearm bone; the wrist keeps only the swing (≤ 85°).
- Fingers 2–5: MediaPipe's per-joint flexion and MCP abduction minus the
  rig's rest angles, applied about each phalanx's own X/Z. Fingers can only
  bend the way the rig's joints bend. Flexion beyond a joint's range spills
  into the joint above (MediaPipe often reads a fist as a straight MCP with an
  impossible ~150° PIP).
- Thumb: MediaPipe's palm-relative segment directions, carried over to the
  rig's current palm, as free aims with a 100° cone (its joints don't hinge
  about axes that line up between the two hands).
- Hands without a pose: the hand is oriented in world and the forearm is
  pointed to match.

`landmarksToPoseDeltas` runs in ~0.1–0.3 ms.

### Recommended call

```ts
const hands = handsForDriver(
  { landmarks: handRes.landmarks, handedness: handRes.handedness.map(h => h[0].categoryName) },
  video.videoWidth, video.videoHeight,
  poseRes.landmarks[0],              // optional: better L/R assignment
);
const deltas = landmarksToPoseDeltas({ pose: poseRes.worldLandmarks[0], ...hands }, model);
```

---

## Validation: three loops

Visual review of hands is unreliable and a verifier that derives its ground
truth from the same data as the driver proves nothing. So every check here
has an independent reference.

### 1. Inverse-kinematics recovery (pure TS, `bun test`)

`tests/mediapipe/recovery.test.ts`: pose the rig with known random deltas
(built on its hinge axes, not by the driver), read off the landmarks
MediaPipe would ideally report, drive, and compare. Limb directions, hip and
shoulder lines, head and palm orientation recover to < 0.5°, finger joints to
~1° mean, the thumb exactly. Also: hands without a pose, depth-mirrored hands
(must still curl palmward — mutation-tested), garbage input (no
hyperextension past the limits), and per-frame cost.

### 2. Synthetic ground truth through MediaPipe (`tools/synthetic.ts`)

Render known hand poses (named gestures × six views × both hands + seeded
random natural hands), run MediaPipe on the render, drive the rig from its
reading, and compare both against the known pose. This measures MediaPipe's
own error on our renders and checks that the driver never adds to it.

### 3. Round trip on real images (`tools/roundtrip.ts`)

The validation proposed in the brief: **image → MediaPipe → driven rig →
render → MediaPipe**, on MediaPipe's labelled hand photos and frames of the
demo clips. The render uses the source camera's orientation (MediaPipe's
world landmarks are camera-aligned), so the two readings are comparable.
Three sources are compared with one metric implementation
(`tests/roundtrip/metrics.ts`):

- `src↔model` — driver fidelity: does the rig do what MediaPipe saw?
- `src↔rt` — the round trip: does the render read back like the photo?
- `model↔rt` — MediaPipe's error on our render (the noise floor).

Metrics are view-invariant where possible: signed joint flexion (palm side
positive, so a finger bent the wrong way shows up as a sign flip), palm
normal and hand direction in camera axes, limb directions, and a 2D landmark
residual after removing translation and scale.

Both tools write an HTML report (source with MediaPipe overlay, driven
render, hand close-ups with MediaPipe overlay, per-joint tables) to `.cache/`.

### The noise floor

MediaPipe reads our flat-shaded renders worse than photos, so the round trip
can't score the driver more finely than MediaPipe can read the render. The
renderer helps it: skin tone, a lighter palm side and fingernails (MediaPipe
uses both to tell palm from back), forearm included in hand close-ups, no
pose model on close-ups (it hallucinates a body around a lone forearm). Even
so, on synthetic hands MediaPipe's own error is ~21° per finger joint and
~18° on the palm normal; the driver tracks it (and removes about half of its
bend-direction flips). In the round trip `model↔rt` is ~10° fingers / 7 % 2D,
of the same order as `src↔rt`.

### Known limits

- Full-body footage: hands are a few percent of the frame. ROI crops make
  MediaPipe find them; finger detail is then a plausible guess, not a
  measurement.
- Edge-on hands (palm up vs palm down seen from the side) are genuinely
  depth-ambiguous for MediaPipe; the round trip shows them as palm-normal
  outliers.
- The forearm twist is split evenly between the two forearm bones — not
  observable from landmarks.

---

## Coordinate conventions

```
MediaPipe world (pose AND hand): +x right (= subject's left when facing the camera),
                                 +y down, +z into the scene. Camera-aligned.
MediaPipe image landmarks:       x, y normalised to the image; z relative depth in
                                 units of the input WIDTH (crop width for ROI crops).
Anny world:                      +x anatomical left, +y back, +z up
  ⇒ anny = [ mp.x, mp.z, −mp.y ]   (a proper rotation)
Hands: fwd = wrist → middle MCP, across = pinky MCP → index MCP;
       palm normal = −(fwd × across) for a RIGHT hand, +(fwd × across) for a LEFT one.
```

## Running it

```sh
task assets         # MediaPipe models + test photos → .cache/ (once)
task test           # unit + IK recovery + parity (fast, no browser)
task test:browser   # headless Chromium + real MediaPipe regression gates (~1 min)
task validate       # synthetic + round-trip reports → .cache/{synthetic,roundtrip}/index.html
task demo           # then open /video.html?mp=local to load MediaPipe from local files
```

The browser tests resolve Chromium from `$ANNY_CHROMIUM`, Playwright's own
build, or the newest `chromium-*` under `$PLAYWRIGHT_BROWSERS_PATH`, and run
WebGL on SwiftShader — no GPU needed. Chromium builds without H.264 can play
the demo clips as WebM via `video.html?clip=<url>`.
