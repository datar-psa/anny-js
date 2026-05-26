# Hand tracking: findings, design, and why it's off by default

This document records what we learned mapping **MediaPipe hand landmarks → Anny
finger bones**, the traps we hit, and the design that resulted. It exists
because the work was unusually subtle: several "fixes" produced numerically
perfect-looking results that were visually wrong, and the only way through was
to build *independent* numerical checks and read what they actually said.

**TL;DR** — Per-finger articulation from **full-body** footage is unreliable:
the hand is a few % of the frame, so MediaPipe's per-finger landmarks (the
distal joints especially) are dominated by noise. Live finger tracking in the
video demo is therefore **disabled by default** (`TRACK_HANDS = false` in
[`demo/entries/video.ts`](../demo/entries/video.ts)) — the hand rests in a
natural open pose and the wrist follows the forearm. All the machinery below is
kept intact and re-enables with one flag, because it works well on close-up
hand footage and is the right foundation if/when we drive hands from a
dedicated hand camera or a closer crop.

---

## The symptom

Body tracking looked great; hands looked "alien". Across iterations the hands
were variously: mirror-flipped, bent backward at the knuckles, splayed like a
claw, or robotically uniform. Visual debugging was nearly useless — a half-
curled splayed hand and a natural fist look similar in a still frame, and the
failure was intermittent (some frames fine, some broken).

## Trap #1 — the circular verifier (0° error, still wrong)

The first verifier compared each posed bone's local-Y axis to a target
direction **computed the same way the driver computed it**. So any *shared*
systematic error — a wrong axis, a mirrored frame, a depth flip — was present on
both sides and cancelled to ~0°. It proved "the bone went where the driver
intended", never "the driver is anatomically correct".

**Lesson:** a verifier that derives its ground truth from the same data the
code-under-test consumes is worthless. Ground truth must be *independent*.

The fix was [`verifyHandVsPose`](../tests/mediapipe/verify_hands.ts): check the
posed finger/wrist bones against the **pose** landmarks (WRIST / INDEX / PINKY /
THUMB) — a different model, a different coordinate pipeline, already in world
coordinates. The metacarpal bones must point toward pose's own knuckle
landmarks; the wrist toward the knuckle centroid. Those checks share no math
with the hand-landmark driver.

## Trap #2 — MediaPipe hand `worldLandmarks` are NOT world-aligned

The MediaPipe docs call hand `worldLandmarks` "real-world 3D coordinates in
meters". We initially read that as camera/world-aligned, like Pose's world
landmarks. **It isn't.** Empirically
([`tools/inspect_hand_frame.ts`](../tools/inspect_hand_frame.ts)): as the
subject's forearm rotates 90°+ across frames, the raw `WRIST → MIDDLE_MCP`
direction in MP "world" stays nearly constant. That can only happen if MP emits
landmarks in a **hand-local canonical frame**.

The MediaPipe source confirms it: `world_landmark_projection_calculator.cc`
applies only a **2D rotation** (the hand-rect's in-image rotation) to the
model's output and **leaves Z untouched**. So:

- **X, Y** are roughly image/camera-aligned, and
- **Z (depth)** is in the model's canonical direction — it does **not** track
  whether the palm faces toward or away from the camera.

Only the metric *scale* is "real-world"; the *orientation* is hand-canonical
with an ambiguous depth axis.

**Consequence:** driving Anny bones with the raw landmarks makes the model's
hand point in a near-constant direction regardless of the actual arm pose (the
"wrist totally different way" symptom). And the depth ambiguity means finger
curl flips palmar↔dorsal arbitrarily ("fingers bend backward").

### The fix: align hand-local → world via pose anchors

[`buildHandTargets`](../src/mediapipe/landmarks.ts) builds a hand frame from
**pose** landmarks (WRIST, INDEX-knuckle, PINKY-knuckle — all in world) and the
matching frame from the hand landmarks, then rotates every hand direction
through `R = W·Lᵀ`. The wrist bone is driven straight from pose
(`WRIST → mid(INDEX, PINKY)`) and never touches the hand-local frame at all.

## Trap #3 — "backward" is a rig-specific, chirality-dependent direction

We needed to know, in Anny's own rig, which way is palmar (natural curl) vs
dorsal (backward). [`tools/probe_curl_direction.ts`](../tools/probe_curl_direction.ts)
answers it empirically by flexing a finger in the rig and watching the tip:

- The metacarpal-fan normal `cross(indexMeta_Y, pinkyMeta_Y)` points **dorsal**
  (back of hand) on the **left** hand.
- A natural palmar curl bends each phalange toward **−fanNormal**.
- **Index and pinky are mirror-arranged on the two hands**, so that same cross
  product points **palmar on the right** hand — it must be **negated for R**.

This chirality flip was the subtle one: a single sign that's correct for the
left hand and inverted for the right. With it wrong, the right hand bent
backward while the (also-flipped) verifier reported it as fine. Both the driver
clamp and the verifier must use the per-hand-corrected dorsal.

### The fix: a palmar clamp anchored to the rig

Rather than try to detect & undo MP's depth flip, we enforce anatomy directly:
each phalange's dorsal projection may never exceed its parent's. If it does,
reflect it to the palmar side (preserving bend magnitude). A finger then
*cannot* bend backward, whatever MP's depth guess was.

## Trap #4 — per-joint MP angles are noise; coordinate the curl

Even with direction correct, the hand looked like a splayed, half-curled claw.
The naturality metrics ([`handNaturality`](../tests/mediapipe/verify_hands.ts))
showed why:

- **cross-finger curl std = 0°** when we forced identical curl — robotic; real
  hands cascade slightly (pinky curls more than index).
- **DIP/PIP ratio ranged 0.0–1.2** off raw MP — the distal-joint estimate on a
  distant hand is pure noise. Natural tendon coupling is ~0.6–0.8.

### The fix: natural curl synergy

Take only the **reliable total curl** from MP (summed palmar bend across the
finger), then redistribute it via fixed anatomical ratios — MCP : PIP : DIP ≈
**0.35 : 0.40 : 0.25** (DIP/PIP ≈ 0.62) — plus a slight per-finger cascade
(index 0.90 → pinky 1.10). De-splay the metacarpals toward their mean knuckle
direction (`SPLAY = 0.45`) so the fingers fan only a little. Result: DIP/PIP
≈ 0.62 on every finger, cross-finger std ≈ 6–8°, adjacent fingertips within a
few degrees — anatomically natural by the numbers.

## Why it's still off by default

After all of the above, the geometry is correct and the metrics are clean, but
on full-body footage the hand is simply too small for MediaPipe to recover
*which* gesture the fingers are making — the recovered curl amount is a noisy
guess. A correct-but-guessed hand still reads as subtly off. So the pragmatic
call is to show a natural open hand and keep the pipeline ready for better
input (close-up hand crop / dedicated hand camera), where the same code
produces good results.

---

## Numerical verification toolkit

Visual review of hands is unreliable; use the numbers.

| Tool | What it tells you |
|---|---|
| [`tools/audit_hand_rig.ts`](../tools/audit_hand_rig.ts) | Rest orientation (X/Y/Z axes) of every hand bone — the ground truth for the rig. |
| [`tools/probe_curl_direction.ts`](../tools/probe_curl_direction.ts) | Which rig rotation is palmar vs dorsal (defines "backward"). |
| [`tools/inspect_hand_frame.ts`](../tools/inspect_hand_frame.ts) | Whether MP hand landmarks are world- or hand-local (the constant-vs-tracking test). |
| [`tools/analyze_hand_fixture.ts`](../tools/analyze_hand_fixture.ts) | Per-bone error, pose-grounded error, palmar curl, and naturality (joint angles, DIP/PIP, cascade) across a captured fixture. |
| [`tests/mediapipe/verify_hands.ts`](../tests/mediapipe/verify_hands.ts) | `verifyHand` (per-bone, circular — use with care), `verifyHandVsPose` (independent, pose-grounded), `handNaturality` (vs human norms). |
| [`tests/mediapipe/hands-synth.test.ts`](../tests/mediapipe/hands-synth.test.ts) | Synthetic poses; pose-grounded sanity (wrist/metacarpal/curl). |
| [`tests/mediapipe/hands-real.test.ts`](../tests/mediapipe/hands-real.test.ts) | Real captured fixtures; asserts fingers never bend backward + metacarpals track pose. |

Regenerate fixtures (hand landmarks are captured even when `TRACK_HANDS` is off):

```sh
task demo                       # serve the demos
bun tools/capture_landmarks.ts  # writes tests/mediapipe/fixtures/*_landmarks.json
```

## Coordinate conventions (quick reference)

```
MediaPipe Pose world:  +x = anatomical-left, +y = down,        +z = away from camera
Anny world:            +x = anatomical-left, +y = depth (back), +z = up
  ⇒  anny = [ +mp.x, +mp.z, −mp.y ]

MediaPipe Hand world:  X,Y ≈ image-aligned; Z = model-canonical (depth-ambiguous);
                       origin at hand centroid; orientation hand-local.
```

## Re-enabling live finger tracking

Set `TRACK_HANDS = true` in [`demo/entries/video.ts`](../demo/entries/video.ts).
Best on close-up hand footage. If finger quality regresses, run
`bun tools/analyze_hand_fixture.ts <fixture>` and read the pose-grounded +
naturality sections — do **not** trust the circular per-bone error.
