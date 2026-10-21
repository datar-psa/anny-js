# MediaPipe → Anny: hands, how they're driven, and how we know they're right

Hands are the hard part of driving Anny from MediaPipe. This document records
what we measured, the driver design that came out of it, and the automated
validation that keeps it honest. It has two parts: **making hands correct**
(the right bend directions, the right palm orientation) and **making them
natural** (poses a person could actually make, in video too) — the second
because a hand can follow MediaPipe faithfully and still look wrong, since
MediaPipe itself is wrong far more often than expected on small hands.
Everything below is reproducible (`task validate`, `task naturalness`,
`task hand-model`) and gated in CI by `task test:browser`.

**TL;DR (naturalness)** — Judged by eye, the driven hands "looked unnatural";
judged by six validated numerical metrics on every frame of the demo clips,
72–100 % of driven hands were physically or anatomically impossible. Causes:
on a full-body video a hand is 9–30 px long, where MediaPipe's finger angles
are *worse than assuming a relaxed hand* and its handedness label is a coin
flip; its systematic errors (a fist read as a hook, thumbs bent backwards);
and transfer errors of our own. The driver now fuses each reading with a
fitted prior over natural hands, trusting it by how well MediaPipe reads a
hand that size on real footage, enforces the physiology a prior can't
express, and tracks hands over time with a Kalman filter built on the same
prior. On the clips: **0 % implausible frames, 1–3 snaps instead of 14–23**;
on held-out ground truth, finger error −30 % and wrong-way bends 15 → 1.

**TL;DR (correctness)** — The earlier hand pipeline was built on a wrong premise (that
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

# Part I — Correct hands

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
- Forearm: aim at the wrist — its depth re-estimated from the hand when the
  pose's would bend the wrist out of range (Part II) — then take the palm's
  roll about the forearm as pronation/supination (≤ 90° from neutral, half on
  each forearm bone; with a straight elbow the excess goes to the humerus);
  the wrist keeps only the swing, within its range of motion.
- Fingers 2–5: the MCP is solved exactly — the proximal phalanx aimed where
  MediaPipe's flexion and abduction put it (the rig's MCP axes tilt a finger
  ~10° sideways as it curls, so plain rotations about X/Z don't reproduce the
  angles); PIP/DIP rotate about their hinge. Flexion beyond a joint's range
  spills into the joint above (MediaPipe often reads a fist as a straight MCP
  with an impossible ~150° PIP); abduction closes as the MCP flexes.
- Thumb: MediaPipe's palm-relative segment directions, carried over to the
  rig's current palm, as free aims with a 100° cone (its joints don't hinge
  about axes that line up between the two hands), held inside the thumb's
  joint ranges, then swung clear of any finger it would pass through.
- Hands without a pose: the whole arm carries the palm from the shoulder.

### Recommended call

```ts
const hands = handsForDriver(
  { landmarks: handRes.landmarks, handedness: handRes.handedness.map(h => h[0].categoryName) },
  video.videoWidth, video.videoHeight,
  poseRes.landmarks[0],              // optional: better L/R assignment
);
// video: once per new frame
const deltas = tracker.update({ pose: poseRes.worldLandmarks[0], ...hands }, video.currentTime * 1000);
// a single image
const deltas = landmarksToPoseDeltas({ pose: poseRes.worldLandmarks[0], ...hands }, model);
```

`handsForDriver` attaches each hand's quality (palm length in pixels, label
agreement); without it, readings are taken at face value (within the ranges
of motion). Hands go to the tracker raw — no landmark smoothing.

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

# Part II — Natural hands

A hand can follow MediaPipe faithfully and still look wrong, because on the
hands that matter most — small ones in full-body video — MediaPipe is wrong
far more than it looks. Eyeballing renders can't settle that, so naturalness
is measured, the measures are themselves validated, and every change below
was kept or dropped on the numbers.

## Measuring naturalness (`tests/naturalness/`, `task naturalness`)

Six independent families of measures, each with a reference that doesn't
come from the driver:

| measure | what it catches | reference |
|---|---|---|
| anatomy (`anatomy.ts`) | joints past their range; DIP not following PIP (tendon coupling, DIP ≈ ⅔·PIP); a ring finger far from both neighbours (enslaving); fingers spread while flexed | hand-biomechanics literature |
| penetration (`physical.ts`) | fingers / thumb passing through each other — capsules measured from the mesh, beyond what the rest pose already has | the skinned mesh |
| arm range (`physical.ts`) | forearm roll beyond pronation/supination, wrist beyond flexion/extension/deviation — read off the bone rotations as twist/swing, independent of how a driver splits them | passive ranges of motion |
| skin damage (`physical.ts`) | collapsed / flipped / stretched triangles of forearm and hand | the rest mesh |
| temporal (`temporal.ts`) | jitter (RMS second difference), snaps (> 30°/frame), freezes | — |
| critic (`critic.ts`) | MediaPipe re-reading the driven hand from six views | MediaPipe |

They are validated on labelled poses built directly on the rig — natural hands
(synergistic curls, named gestures) versus five families of implausible ones —
as the ROC AUC of "metric says implausible" (`bun tools/naturalness.ts validate`):

| metric | independent fingers | noise | hooked tips | hyperextended | crossed | twisted wrist |
|---|---:|---:|---:|---:|---:|---:|
| anatomy | 0.95 | 0.94 | 1.00 | 0.95 | 0.96 | — |
| penetration | 0.61 | 0.76 | — | — | **1.00** | — |
| arm range | — | — | — | — | — | **1.00** |
| critic (missed) | 0.75 | — | — | — | — | 1.00 |

Skin damage didn't separate anything (≈ 0.5) and is reported, not gated. The
critic is weak and slow and is kept as a cross-check only.

Applied to every frame of the demo clips (`tools/capture_sequences.ts` caches
MediaPipe's detections; `tests/naturalness/replay.ts` replays the demo's
pipeline in milliseconds), the driver of Part I scored:

| clip / hand | implausible frames | penetration | arm out of range | forearm roll p90 | jitter | snaps |
|---|---:|---:|---:|---:|---:|---:|
| dancer L | 79 % | 4.9 mm | 26° | 156° | 10.4 | 17 |
| dancer R | 72 % | 2.2 mm | 47° | 173° | 9.7 | 15 |
| boxer L | 100 % | 13.8 mm | 67° | 173° | 6.7 | 23 |
| boxer R | 100 % | 8.2 mm | 4° | 56° | 21.1 | 14 |

(Building the measures also caught two bugs in themselves: the wrist's local X
is *deviation*, not flexion — probed now, see `rig.ts` — and a projection-based
twist that leaked wrist swing into forearm roll.)

## Why: how wrong MediaPipe is, by hand size

`tools/hand_dataset.ts` renders 207 physically valid ground-truth hands (16
gesture families + random natural hands) at seven sizes, upscaled to 256 px
like the demo's ROI crops, and reads them back with MediaPipe — 1449 readings.
RMS error of MediaPipe against the truth, next to simply assuming the average
hand (`bun tools/hand_trust.ts`):

| palm px | detected | finger joint: MediaPipe | average hand | abduction: MediaPipe | average hand |
|---:|---:|---:|---:|---:|---:|
| 8 | 66 % | 54.5° | 36.1° | 20.4° | 4.2° |
| 16 | 96 % | 44.7° | 35.2° | 20.5° | 4.4° |
| 33 | 98 % | 38.5° | 35.7° | 16.5° | 4.4° |
| 132 | 89 % | 28.6° | 35.2° | 12.8° | 4.3° |

Below ~30 px — every hand in the demo clips (palm 9–31 px) — MediaPipe's
finger angles are worse than no reading at all. Its palm orientation is
anisotropic: roll in the image plane is read to ~10° at any size, tilt
toward/away from the camera only to 20–30°. When its handedness label
disagrees with the side the pose assigns the hand to — 45 % of the dancer's
hands — the palm reading is essentially random (60–100° error).

**Real footage differs from renders**, in both directions. Without ground
truth, test-time augmentation measures it: re-detect each image shifted /
scaled / rotated slightly; joint angles are invariant to all three, so the
spread of the readings is MediaPipe's instability, and on renders the true
error is a stable 2.2–2.8 × that spread (`tools/hand_tta.ts`):

| | palm px | spread vs renders of that size |
|---|---:|---:|
| MediaPipe's labelled photos | 93 | **0.36 ×** (crisp photos: much better) |
| demo clip frames (ROI) | 15 | **3.97 ×** (blurry, compressed video: much worse) |

## Fusion: what the reading says, weighed against what hands are like

`src/mediapipe/hand-fusion.ts`, constants fitted by `tools/hand_trust.ts`
into `src/mediapipe/hand-prior.ts` (`task hand-model` regenerates them).

**Hand shape.** A 25-dimensional Gaussian: 12 finger flexions, 4 abductions,
3 thumb directions. The prior is the frequency-weighted mean and covariance
of the ground-truth hands (relaxed and open hands weigh most; signs least);
its off-diagonal terms are the finger synergies — fingers curl together, DIP
follows PIP, the thumb folds with the fingers. MediaPipe's noise is a
per-dimension variance curve a + b / px times one correlation matrix, scaled
to real footage by κ(px) = 0.36 · (93 / px)^1.32. The posterior mean is the
driven hand: a noisy reading is pulled toward the nearest *natural* hand, not
just the average one. Cross-validated by leaving out whole gesture families:

| palm px | finger joint: MediaPipe → fused | finger curl | thumb |
|---:|---:|---:|---:|
| 8 | 54.5° → 34.8° | 136° → 96° | 40.5° → 36.0° |
| 16 | 44.7° → 30.1° | 105° → 83° | 36.2° → 29.9° |
| 33 | 38.5° → 26.8° | 89° → 73° | 30.3° → 23.9° |
| 132 | 28.6° → 23.0° | 71° → 61° | 25.8° → 18.4° |

Ablations: an independent (diagonal) prior is 3–4° worse per joint and 6–10°
on curl; the thumb gains nothing on its own and 8° jointly with the fingers.

**Wrist.** Roll, flexion and deviation relative to the forearm are fused
jointly with a neutral-wrist prior (SD 40° / 30° / 12°), MediaPipe's noise
carried from camera axes into wrist axes at the current pose — so the hand
keeps pointing where it is *seen* pointing while the depth tilt the camera
can't see is left to the prior.

**Physiology the prior can't express**, enforced on top:

- passive ranges: forearm roll ±90° (the rig's rest is mid-prone), wrist
  −85…85° / −40…25°; roll beyond the forearm goes to the humerus when the
  elbow is straight (the pose can't see humeral rotation then);
- DIP–PIP tendon coupling as a soft constraint (SD 10°) — MediaPipe's fist
  error is a *bias* (fingertips hidden in the fist, read as a hook) that a
  zero-mean noise model can't undo;
- abduction closing from ±25° to ±5° as the MCP flexes 30° → 90°;
- thumb MCP/IP ranges, and a search for the smallest swing of the thumb
  that clears every finger it would pass through.

**Forearm depth from the hand.** MediaPipe's 3D pose is weakest in depth, and
a forearm's depth decides how a visible hand must sit on it: a "stop" hand on
a forearm read as lying across the chest needs a 150° roll, on one pointing
at the camera just wrist extension. The forearm keeps its image-plane
direction; its out-of-plane tilt balances the pose's estimate (SD 25°)
against the range of motion the hand would violate, weighted by how much the
hand reading can be trusted.

**Out-of-sight arms.** With no pose (a close-up) or a single image, the
whole arm carries the palm; in video a doubtful hand on an unseen arm holds
the arm's last pose instead of spinning the forearm to meet a noisy reading.

## Tracking (`src/mediapipe/tracker.ts`)

`PoseTracker` carries each hand's belief between frames as an
Ornstein–Uhlenbeck process whose stationary distribution *is* the prior —
the belief relaxes toward natural hands with correlation time τ — and updates
it with each reading: a Kalman filter in which small hands are smoothed hard
and large ones barely, a lost hand relaxes into a natural pose instead of
freezing, and a surprising reading is down-weighted by its Mahalanobis
distance (Student-t update). It replaces the demo's landmark EMA, outlier
gate and hold. τ is a trade-off measured in `tests/mediapipe/tracker.test.ts`:
at 1 s a single garbled frame moves a hand ~40 % of the way (a one-frame
blunder and the first frame of a real change look alike), a real change is
followed within ~3 frames at 30 fps.

Cost: ~1 ms per frame for the body and two fused hands (0.4 ms taking the
readings at face value) — MediaPipe itself takes 20–50 ms.

## Results

**Demo clips** — same cached MediaPipe detections, Part I driver with the
demo's old filtering vs `PoseTracker` (`task naturalness`; gated in CI by
`tests/browser/naturalness.test.ts`):

| clip / hand | implausible frames | penetration | arm out of range | forearm roll p90 | jitter | snaps |
|---|---:|---:|---:|---:|---:|---:|
| dancer L | 79 % → **0 %** | 4.9 → 0.2 mm | 26° → 0° | 156° → 24° | 10.4 → 1.8 | 17 → 3 |
| dancer R | 72 % → **0 %** | 2.2 → 0.1 mm | 47° → 0° | 173° → 32° | 9.7 → 2.5 | 15 → 1 |
| boxer L | 100 % → **0 %** | 13.8 → 0.1 mm | 67° → 0° | 173° → 51° | 6.7 → 3.9 | 23 → 2 |
| boxer R | 100 % → **0 %** | 8.2 → 0.1 mm | 4° → 0° | 56° → 56° | 21.1 → 4.8 | 14 → 2 |

**Held-out ground truth** — the synthetic set of Part I (a different
generator and seed from the one the prior was fitted on), rendered small and
upscaled like ROI crops:

| palm px | finger flexion error: MediaPipe / Part I driver / fused | wrong-way bends |
|---:|---:|---:|
| 8 | 41.6° / 33.4° / **29.3°** | 31 / 8 / **0** |
| 18 | 33.1° / 27.3° / **23.6°** | 25 / 15 / **1** |
| 108 | 22.5° / 21.6° / **14.3°** | 15 / 7 / **1** |

**Real photos** (MediaPipe's labelled hands; heavy pose model):

| | palm MP↔rig | fingers MP↔rig | round trip: 2D | fingers | body |
|---|---:|---:|---:|---:|---:|
| Part I driver | 0.1° | 4.3° | 0.105 | 13.4° | 6.9° |
| face value (no quality) | 0.3° | 4.2° | **0.091** | **12.1°** | **6.6°** |
| fused | 2.4° | 21.8° | 0.136 | 17.3° | 6.7° |

The geometry alone (exact MCP transfer, ranges, forearm depth, thumb
clearance) reads back *better* than Part I. Fused, the rig departs from
MediaPipe on purpose — every photo hand is now anatomically plausible, but
the round trip, which can only measure agreement with MediaPipe, counts the
corrections (fist.jpg: a 147° PIP with the fingertip at −31° becomes a fist)
as error.

### Trade-offs, stated plainly

- On tiny video hands the fused hand agrees with MediaPipe's reading of that
  hand much less (2D landmark residual 0.12 → 0.31 of a palm on the clip
  frames, ≈ 5 px on a 15 px hand). By the measured noise that reading is
  ~70° RMS wrong per joint, so this is the point — but it is a choice: a
  hand MediaPipe happens to read right at 15 px is pulled toward natural too.
- The prior is a modelling choice (which gestures, how often), fitted on
  renders of one rig; the real-footage calibration rests on two operating
  points (photos, demo clips).
- Open hands read from large photos land ~13° more curled than MediaPipe
  says: under the calibrated noise (~17° per joint even at 150 px) that is
  what the evidence supports.
- τ = 1 s: one-frame blunders are halved, not removed.

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
task test           # unit + IK recovery + fusion + tracker + parity (fast, no browser)
task test:browser   # headless Chromium + real MediaPipe regression gates (~2 min)
task validate       # synthetic + round-trip reports → .cache/{synthetic,roundtrip}/index.html
task sequences      # cache MediaPipe on every frame of the demo clips (once)
task naturalness    # validate the naturalness metrics, score the clips
task hand-model     # refit prior + noise model → src/mediapipe/hand-prior.ts
task demo           # then open /video.html?mp=local to load MediaPipe from local files
```

The browser tests resolve Chromium from `$ANNY_CHROMIUM`, Playwright's own
build, or the newest `chromium-*` under `$PLAYWRIGHT_BROWSERS_PATH`, and run
WebGL on SwiftShader — no GPU needed. Chromium builds without H.264 can play
the demo clips as WebM via `video.html?clip=<url>`.
