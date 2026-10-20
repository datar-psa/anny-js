/**
 * Pinhole WebGL renderer for the round-trip harness.
 *
 * Unlike the demos' orthographic quad-view renderer, this one reproduces a
 * real camera (intrinsics + extrinsics from `CameraSpec`) and shades the body
 * with a plain skin tone, so MediaPipe sees something close to a photo of a
 * person. `preserveDrawingBuffer` keeps the frame readable by MediaPipe and
 * `toDataURL` after the draw call returns.
 */

import type { AnnyModel } from "../../src/anny/index.js";
import type { CameraSpec, HandSide, RenderOptions } from "../roundtrip/protocol.js";

const VS = `
  attribute vec3 aPos;
  attribute vec3 aNrm;
  attribute vec3 aCol;
  attribute float aMask;
  uniform mat4 uMVP;
  uniform mat3 uR;
  varying vec3 vNrm;
  varying vec3 vCol;
  varying float vMask;
  void main() {
    vNrm = uR * aNrm;
    vCol = aCol;
    vMask = aMask;
    gl_Position = uMVP * vec4(aPos, 1.0);
  }
`;

// Camera-space lighting: +x right, +y down, +z into the scene. A key light
// from the upper-left front, a dimmer fill from the right and a little
// ambient — enough relief for MediaPipe to read finger creases.
const FS = `
  precision mediump float;
  uniform float uMaskOn;
  varying vec3 vNrm;
  varying vec3 vCol;
  varying float vMask;
  void main() {
    if (uMaskOn > 0.5 && vMask < 0.5) discard;
    vec3 n = normalize(vNrm);
    if (n.z > 0.0) n = -n;   // two-sided: always light the camera-facing side
    vec3 keyDir = normalize(vec3(-0.45, -0.6, -0.65));
    float key  = max(dot(n, keyDir), 0.0);
    float fill = max(dot(n, normalize(vec3( 0.7,  0.1, -0.7))), 0.0);
    // Weak specular (camera looks down +z) so nails and knuckles catch light.
    vec3 h = normalize(keyDir + vec3(0.0, 0.0, -1.0));
    float spec = pow(max(dot(n, h), 0.0), 24.0) * 0.12;
    vec3 c = vCol * (0.28 + 0.62 * key + 0.22 * fill) + vec3(spec);
    gl_FragColor = vec4(c, 1.0);
  }
`;

function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type);
  if (!s) throw new Error("createShader failed");
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(`shader compile: ${gl.getShaderInfoLog(s)}`);
  }
  return s;
}

/** Column-major MVP for `cam` (WebGL clip space, depth in [near, far]). */
function mvpColumnMajor(cam: CameraSpec, near = 0.02, far = 20): Float32Array {
  const { W, H, fx, fy, cx, cy, R, C } = cam;
  // Projection P (row-major) acting on camera coords (x right, y down, z fwd).
  const P = [
    2 * fx / W, 0, 2 * cx / W - 1, 0,
    0, -2 * fy / H, 1 - 2 * cy / H, 0,
    0, 0, (far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, 1, 0,
  ];
  // View V = [R | -R·C].
  const t = [
    -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
    -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
    -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
  ];
  const V = [
    R[0], R[1], R[2], t[0],
    R[3], R[4], R[5], t[1],
    R[6], R[7], R[8], t[2],
    0, 0, 0, 1,
  ];
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += P[r * 4 + k] * V[k * 4 + c];
      out[c * 4 + r] = s;   // column-major
    }
  }
  return out;
}

const SKIN: [number, number, number] = [0.86, 0.66, 0.54];
const PALM: [number, number, number] = [0.93, 0.70, 0.60];
const NAIL: [number, number, number] = [0.95, 0.83, 0.80];

/**
 * Per-vertex albedo with the two cues MediaPipe uses to tell a palm from the
 * back of a hand: a lighter palm/finger-pad side and fingernails. Both are
 * read off rest geometry — no UVs needed. Fingers flex about their local +X,
 * so a phalanx's local +Z points palmar and −Z dorsal (nail side).
 */
function skinColors(model: AnnyModel): Float32Array {
  const K = model.maxBonesPerVert;
  const rp = model.restBonePoses;
  const col = new Float32Array(model.vertCount * 3);
  const label = model.boneLabels;
  const isPhalanx = (b: number) => /^finger\d-\d\.[LR]$/.test(label[b]);
  const isDistal = (b: number) => /^finger\d-3\.[LR]$/.test(label[b]);
  const isPalm = (b: number) => /^(metacarpal\d|wrist)\.[LR]$/.test(label[b]);
  // Palm normal per side: average +Z of the metacarpals.
  const palmNormal = (side: string): number[] => {
    const n = [0, 0, 0];
    for (let m = 1; m <= 4; m++) {
      const o = label.indexOf(`metacarpal${m}.${side}`) * 16;
      n[0] += rp[o + 2]; n[1] += rp[o + 6]; n[2] += rp[o + 10];
    }
    const l = Math.hypot(n[0], n[1], n[2]);
    return n.map(x => x / l);
  };
  const palmN: Record<string, number[]> = { L: palmNormal("L"), R: palmNormal("R") };
  for (let v = 0; v < model.vertCount; v++) {
    let best = -1, bw = 0;
    for (let k = 0; k < K; k++) {
      const w = model.boneWeights[v * K + k];
      if (w > bw) { bw = w; best = model.boneIndices[v * K + k]; }
    }
    let c = SKIN;
    if (best >= 0 && (isPhalanx(best) || isPalm(best))) {
      const o = best * 16;
      const nx = model.restNormals[v * 3], ny = model.restNormals[v * 3 + 1], nz = model.restNormals[v * 3 + 2];
      const side = label[best].slice(-1);
      const pd = isPhalanx(best) ? [rp[o + 2], rp[o + 6], rp[o + 10]] : palmN[side];
      const palmar = nx * pd[0] + ny * pd[1] + nz * pd[2];
      const t = Math.min(1, Math.max(0, (palmar - 0.1) / 0.5));
      c = [0, 1, 2].map(i => SKIN[i] + t * (PALM[i] - SKIN[i])) as [number, number, number];
      if (isDistal(best) && bw > 0.5) {
        const px = model.restVertices[v * 3] - rp[o + 3];
        const py = model.restVertices[v * 3 + 1] - rp[o + 7];
        const pz = model.restVertices[v * 3 + 2] - rp[o + 11];
        const along = px * rp[o + 1] + py * rp[o + 5] + pz * rp[o + 9];
        if (palmar < -0.6 && along > 0.006) c = NAIL;
      }
    }
    col.set(c, v * 3);
  }
  return col;
}

/**
 * Per-vertex weight on one forearm + hand (lowerarm01 and every descendant).
 * The forearm stays in: MediaPipe's hand model expects a wrist leading into
 * an arm, and reads a hand cut off at the wrist noticeably worse.
 */
function handMask(model: AnnyModel, side: HandSide): Float32Array {
  const forearm = model.boneLabels.indexOf(`lowerarm01.${side}`);
  const inHand = new Uint8Array(model.boneCount);
  for (let b = 0; b < model.boneCount; b++) {
    let p = b;
    while (p >= 0 && p !== forearm) p = model.boneParents[p];
    inHand[b] = p === forearm ? 1 : 0;
  }
  const K = model.maxBonesPerVert;
  const mask = new Float32Array(model.vertCount);
  for (let v = 0; v < model.vertCount; v++) {
    let w = 0;
    for (let k = 0; k < K; k++) {
      if (inHand[model.boneIndices[v * K + k]]) w += model.boneWeights[v * K + k];
    }
    mask[v] = w;
  }
  return mask;
}

export class PinholeRenderer {
  readonly canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext;
  private prog: WebGLProgram;
  private posBuf: WebGLBuffer;
  private nrmBuf: WebGLBuffer;
  private maskBufs: Record<"none" | HandSide, WebGLBuffer>;
  private colBuf: WebGLBuffer;
  private idxBuf: WebGLBuffer;
  private indexCount: number;
  private indexType: number;

  constructor(model: AnnyModel) {
    this.canvas = document.createElement("canvas");
    const gl = this.canvas.getContext("webgl", {
      antialias: true, preserveDrawingBuffer: true, alpha: false,
    });
    if (!gl) throw new Error("WebGL unavailable");
    this.gl = gl;

    const prog = gl.createProgram();
    if (!prog) throw new Error("createProgram failed");
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`program link: ${gl.getProgramInfoLog(prog)}`);
    }
    this.prog = prog;

    const buf = (): WebGLBuffer => {
      const b = gl.createBuffer();
      if (!b) throw new Error("createBuffer failed");
      return b;
    };
    this.posBuf = buf();
    this.nrmBuf = buf();
    this.idxBuf = buf();
    const uploadMask = (m: Float32Array): WebGLBuffer => {
      const b = buf();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, m, gl.STATIC_DRAW);
      return b;
    };
    this.colBuf = uploadMask(skinColors(model));
    this.maskBufs = {
      none: uploadMask(new Float32Array(model.vertCount).fill(1)),
      L: uploadMask(handMask(model, "L")),
      R: uploadMask(handMask(model, "R")),
    };

    // 13 718 vertices fit in 16-bit indices; keep the 32-bit path for re-baked meshes.
    const uint32 = gl.getExtension("OES_element_index_uint");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    if (model.vertCount < 65536) {
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint16Array.from(model.faces), gl.STATIC_DRAW);
      this.indexType = gl.UNSIGNED_SHORT;
    } else {
      if (!uint32) throw new Error("mesh needs 32-bit indices");
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, Uint32Array.from(model.faces), gl.STATIC_DRAW);
      this.indexType = gl.UNSIGNED_INT;
    }
    this.indexCount = model.faces.length;
  }

  render(vertices: Float32Array, normals: Float32Array, cam: CameraSpec, opts: RenderOptions = {}): HTMLCanvasElement {
    const gl = this.gl;
    this.canvas.width = cam.W;
    this.canvas.height = cam.H;
    gl.viewport(0, 0, cam.W, cam.H);
    const [br, bg, bb] = opts.background ?? [0.86, 0.86, 0.84];
    gl.clearColor(br, bg, bb, 1);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, "uMVP"), false, mvpColumnMajor(cam));
    const R = cam.R;
    // mat3 uniform is column-major: transpose of row-major R.
    gl.uniformMatrix3fv(gl.getUniformLocation(this.prog, "uR"), false,
      new Float32Array([R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]));
    gl.uniform1f(gl.getUniformLocation(this.prog, "uMaskOn"), opts.handOnly ? 1 : 0);

    const bindAttr = (name: string, b: WebGLBuffer, size: number, data?: Float32Array): void => {
      const loc = gl.getAttribLocation(this.prog, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      if (data) gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(loc);
    };
    bindAttr("aPos", this.posBuf, 3, vertices);
    bindAttr("aNrm", this.nrmBuf, 3, normals);
    bindAttr("aMask", this.maskBufs[opts.handOnly ?? "none"], 1);
    bindAttr("aCol", this.colBuf, 3);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.drawElements(gl.TRIANGLES, this.indexCount, this.indexType, 0);
    gl.finish();
    return this.canvas;
  }
}
