/** Pre-baked animation demo: 4-second salto loop, no camera/MP needed. */

import {
  loadAnnyModel, buildBoneIndex,
  allocBoneTransforms, identityDeltas, setDeltas,
  lbs, allocVertexBuffer, allocNormalBuffer,
  forwardKinematics,
  rodriguesToMat3, rotFromTo,
} from "../../src/anny/index.js";
import { setupWebGL, minRestZ } from "../_shared/webgl.js";
import { computeQuadrants, buildFourViews, drawFourViews } from "../_shared/multi-view.js";

const W = window.innerWidth;
const H = window.innerHeight;
const statsEl = document.getElementById("stats")!;
const poseEl  = document.getElementById("pose")!;
const glCanvas = document.getElementById("glc") as HTMLCanvasElement;
glCanvas.width = W; glCanvas.height = H;

const model = await loadAnnyModel("/anny_model.json", "/anny_model.bin");
statsEl.textContent = "model ready";

const boneIndex  = buildBoneIndex(model);
const boneXforms = allocBoneTransforms(model.boneCount);
const vertBuf    = allocVertexBuffer(model);
const nrmBuf     = allocNormalBuffer(model);
const restMinZ   = minRestZ(model);
const gl = setupWebGL(glCanvas, model);

// ── Pose helpers ────────────────────────────────────────────────────────
const deg = (a: number) => a * Math.PI / 180;
const rotX = (a: number) => rodriguesToMat3(new Float32Array([1, 0, 0]), a);

const Y_UNIT = new Float32Array([0, 1, 0]);

/** Aim a bone's local +Y axis at a world-space direction. */
function aimAt(boneName: string, worldDir: [number, number, number]): Float32Array {
  const idx = boneIndex.get(boneName) ?? -1;
  if (idx < 0) return new Float32Array([1,0,0, 0,1,0, 0,0,1]);
  const b = idx * 16;
  const R = model.restBonePoses;
  const lx = R[b  ]*worldDir[0] + R[b+4]*worldDir[1] + R[b+ 8]*worldDir[2];
  const ly = R[b+1]*worldDir[0] + R[b+5]*worldDir[1] + R[b+ 9]*worldDir[2];
  const lz = R[b+2]*worldDir[0] + R[b+6]*worldDir[1] + R[b+10]*worldDir[2];
  const n = Math.hypot(lx, ly, lz) || 1;
  return rotFromTo(Y_UNIT, new Float32Array([lx/n, ly/n, lz/n]));
}

function blendDirs(
  rest: [number, number, number],
  weighted: Array<[number, [number, number, number]]>,
): [number, number, number] {
  let x = 0, y = 0, z = 0, w = 0;
  for (const [wi, d] of weighted) {
    if (wi <= 0) continue;
    x += wi*d[0]; y += wi*d[1]; z += wi*d[2]; w += wi;
  }
  if (w < 1) {
    const rw = 1 - w;
    x += rw*rest[0]; y += rw*rest[1]; z += rw*rest[2];
  }
  return [x, y, z];
}

const smooth  = (t: number) => t*t*(3-2*t);
const clamp01 = (t: number) => Math.max(0, Math.min(1, t));
const phase   = (t: number, a: number, b: number) => smooth(clamp01((t-a)/(b-a)));

const ARM_REST_L: [number, number, number]  = [ 0.73, 0.00, -0.69];
const ARM_REST_R: [number, number, number]  = [-0.73, 0.00, -0.69];
const ARM_BACK_L: [number, number, number]  = [ 0.45, 0.35, -0.82];
const ARM_BACK_R: [number, number, number]  = [-0.45, 0.35, -0.82];
const ARM_UP_L:   [number, number, number]  = [ 0.30, 0.00,  0.95];
const ARM_UP_R:   [number, number, number]  = [-0.30, 0.00,  0.95];
const ARM_TUCK_L: [number, number, number]  = [ 0.18,-0.70, -0.70];
const ARM_TUCK_R: [number, number, number]  = [-0.18,-0.70, -0.70];
const ARM_OPEN_L: [number, number, number]  = [ 0.92,-0.20, -0.34];
const ARM_OPEN_R: [number, number, number]  = [-0.92,-0.20, -0.34];

function makeSaltoPose(t: number) {
  const d = identityDeltas(model.boneCount);

  const crouch    = phase(t, 0.10, 0.22) - phase(t, 0.22, 0.30)
                  + phase(t, 0.74, 0.84) - phase(t, 0.92, 1.00);
  const launch    = phase(t, 0.22, 0.30) * (1 - phase(t, 0.30, 0.38));
  const tuck      = phase(t, 0.32, 0.44) * (1 - phase(t, 0.55, 0.60));
  const armsBack  = phase(t, 0.10, 0.20) * (1 - phase(t, 0.20, 0.28));
  const armsUp    = phase(t, 0.20, 0.30) * (1 - phase(t, 0.32, 0.42));
  const armsOpen  = phase(t, 0.55, 0.62) * (1 - phase(t, 0.82, 0.92));

  const tuckHip   = -tuck * 130;
  const tuckKnee  =  tuck * 145;
  const tuckElbow =  tuck * 105;

  const armDirL = blendDirs(ARM_REST_L, [
    [armsBack, ARM_BACK_L], [armsUp, ARM_UP_L],
    [tuck,     ARM_TUCK_L], [armsOpen, ARM_OPEN_L],
  ]);
  const armDirR = blendDirs(ARM_REST_R, [
    [armsBack, ARM_BACK_R], [armsUp, ARM_UP_R],
    [tuck,     ARM_TUCK_R], [armsOpen, ARM_OPEN_R],
  ]);

  setDeltas(d, model, [
    { bone: 'upperleg01.L', rot: rotX(deg(-5 - crouch*32 + launch*8 + tuckHip)) },
    { bone: 'upperleg01.R', rot: rotX(deg(-5 - crouch*32 + launch*8 + tuckHip)) },
    { bone: 'lowerleg01.L', rot: rotX(deg(crouch*55 + tuckKnee)) },
    { bone: 'lowerleg01.R', rot: rotX(deg(crouch*55 + tuckKnee)) },

    { bone: 'upperarm01.L', rot: aimAt('upperarm01.L', armDirL) },
    { bone: 'upperarm01.R', rot: aimAt('upperarm01.R', armDirR) },
    { bone: 'lowerarm01.L', rot: rotX(deg(armsUp*10 + tuckElbow)) },
    { bone: 'lowerarm01.R', rot: rotX(deg(armsUp*10 + tuckElbow)) },

    { bone: 'spine01', rot: rotX(deg(crouch*5 - launch*6 + tuck*15)) },
    { bone: 'spine02', rot: rotX(deg(crouch*4 - launch*4 + tuck*15)) },
    { bone: 'spine03', rot: rotX(deg(tuck*12)) },
    { bone: 'spine04', rot: rotX(deg(tuck*10)) },
    { bone: 'spine05', rot: rotX(deg(tuck*8)) },

    { bone: 'neck01', rot: rotX(deg(tuck*30)) },
  ]);
  return d;
}

const flipAngle = (t: number) => -phase(t, 0.30, 0.75) * Math.PI * 2;
const jumpArc   = (t: number) => phase(t, 0.24, 0.35) * (1 - phase(t, 0.72, 0.84));

const CYCLE_MS = 4200;

const qs = new URLSearchParams(location.search);
const FROZEN_T = qs.has("t") ? parseFloat(qs.get("t")!) : null;
const FLIP_ON  = qs.get("flip") !== "0";

let fc = 0, fpsTs = performance.now(), fps = 0;

function tick() {
  requestAnimationFrame(tick);
  const now = performance.now();
  if (++fc % 60 === 0) {
    fps = Math.round(fc / ((now - fpsTs) / 1000));
    fc = 0; fpsTs = now;
  }

  const tOverride = (window as unknown as { __t?: number }).__t;
  const t = typeof tOverride === "number" ? tOverride : (FROZEN_T ?? (now % CYCLE_MS) / CYCLE_MS);

  const deltas = makeSaltoPose(t);
  const rx = flipAngle(t);

  const t1 = performance.now();
  forwardKinematics(model, deltas, boneXforms);
  const mesh = lbs(model, boneXforms, vertBuf, nrmBuf);
  const ms = (performance.now() - t1).toFixed(1);

  // Body height in pixels → vertical lift during aerial phase
  const q = computeQuadrants(W, H, /* rawScale = */ W * 1.4, restMinZ);
  const bodyH = 1.7 * q.scale;
  const jumpY = -jumpArc(t) * bodyH * 0.18;

  // anim layout uses a slightly wider 3/4 angle + bigger ABOVE view
  const views = buildFourViews(q, {
    threeQuarterAngle: Math.PI * 0.4,
    aboveScaleFactor: 0.65,
    bodyYOffset: jumpY,
  });

  gl.uploadVertices(mesh.vertices);
  gl.uploadNormals(mesh.normals);
  gl.clear();
  gl.setWorldTwist(FLIP_ON ? rx : 0);
  drawFourViews(gl, views);

  const airborne = t > 0.30 && t < 0.74;
  const phaseName = t<0.10?"stand": t<0.22?"crouch": t<0.30?"launch":
                    t<0.62?"tuck":  t<0.72?"open":   t<0.86?"land":  "recover";
  statsEl.textContent = `FK+LBS ${ms} ms · ${fps} fps · ${model.vertCount.toLocaleString()} verts · ${model.boneCount} bones`;
  poseEl.textContent = airborne ? `✈ ${phaseName} (${Math.round(-rx/Math.PI/2*360)}°)` : phaseName;
}

tick();
