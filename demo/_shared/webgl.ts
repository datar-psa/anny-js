/**
 * WebGL pipeline shared by every demo: a face-normal-lit body shader and a
 * static floor-grid shader. Both honor a single screen-space projection
 * (`{cx, footY, scale, xSign}`) plus camera orbit (camY) and tilt (camX).
 *
 * The "world twist" uniform on the body lets the demo spin the model around
 * its own Z axis without spinning the camera or the grid — handy if you want
 * to bake hip yaw into the rig rather than the view.
 */

import type { AnnyModel } from "../../src/anny/index.js";

export interface ViewParams {
  /** Canvas X of the body center. */
  cx: number;
  /** Canvas Y of the floor (Z = 0). */
  footY: number;
  /** Pixels per metre. */
  scale: number;
  /** +1 (default) or -1 to mirror horizontally. */
  xSign?: number;
  /** Camera orbit around vertical axis, radians. */
  camY?: number;
  /** Camera tilt around horizontal axis, radians (negative = bird's-eye). */
  camX?: number;
}

export interface WebGLBundle {
  gl: WebGLRenderingContext;
  /** Upload posed vertices for this frame (call once before any `drawBody`). */
  uploadVertices(vertices: Float32Array): void;
  /** Draw the body in one view, RGB in 0..1. */
  drawBody(view: ViewParams, color: [number, number, number]): void;
  /** Draw the floor grid in one view (no world-twist applied). */
  drawGrid(view: ViewParams): void;
  /** Set hip-yaw world rotation for upcoming body draws (radians). */
  setWorldTwist(angle: number): void;
  /** Clear the framebuffer to bg, prepare for new frame. */
  clear(r?: number, g?: number, b?: number): void;
  /** Resize the GL canvas and viewport. */
  resize(w: number, h: number): void;
  /** Detected GPU name (best effort), or "unknown". */
  gpuName: string;
}

const BODY_VS = `
  attribute vec3 aPos;
  uniform float uSc, uCx, uFy, uXs, uCY, uSY, uCX, uSX, uWX, uWS;
  uniform vec2 uVP;
  varying vec3 vViewPos;
  varying float vH, vD;
  void main() {
    float py = aPos.y * uWX - aPos.z * uWS;
    float pz = aPos.y * uWS + aPos.z * uWX;
    float rx = aPos.x * uCY - py * uSY;
    float ry = aPos.x * uSY + py * uCY;
    float rz = ry * uSX + pz * uCX;
    float rd = ry * uCX - pz * uSX;
    float sx = uCx + uXs * rx * uSc;
    float sy = uFy - rz * uSc;
    vViewPos = vec3(sx, sy, rd * uSc);
    vH = pz; vD = rd;
    gl_Position = vec4(sx / uVP.x * 2.0 - 1.0, 1.0 - sy / uVP.y * 2.0, rd / 5.0, 1.0);
  }
`;

const BODY_FS_DERIV = `
  #extension GL_OES_standard_derivatives : enable
  precision mediump float;
  uniform vec4 uCol;
  varying vec3 vViewPos;
  varying float vH, vD;
  void main() {
    vec3 n = normalize(cross(dFdx(vViewPos), dFdy(vViewPos)));
    float key  = max(dot(n, normalize(vec3( 0.3, -0.8,  0.6))), 0.0);
    float fill = max(dot(n, normalize(vec3( 0.8,  0.2,  0.3))), 0.0);
    float rim  = max(dot(n, normalize(vec3(-0.2,  0.6, -0.8))), 0.0);
    gl_FragColor = vec4(uCol.rgb * (0.28 + 0.52 * key + 0.14 * fill + 0.06 * rim * rim), 1.0);
  }
`;

const BODY_FS_FALLBACK = `
  precision mediump float;
  uniform vec4 uCol;
  varying float vH, vD;
  void main() {
    float top   = 0.35 + 0.65 * clamp(vH / 1.8, 0.0, 1.0);
    float front = 0.50 + 0.50 * clamp(-vD * 1.4, 0.0, 1.0);
    gl_FragColor = vec4(uCol.rgb * (top * 0.5 + front * 0.5), 1.0);
  }
`;

const GRID_VS = `
  attribute vec3 aPos;
  uniform float uSc, uCx, uFy, uXs, uCY, uSY, uCX, uSX;
  uniform vec2 uVP;
  void main() {
    float rx = aPos.x * uCY - aPos.y * uSY;
    float ry = aPos.x * uSY + aPos.y * uCY;
    float rz = ry * uSX + aPos.z * uCX;
    float rd = ry * uCX - aPos.z * uSX;
    gl_Position = vec4(
      (uCx + uXs * rx * uSc) / uVP.x * 2.0 - 1.0,
      1.0 - (uFy - rz * uSc) / uVP.y * 2.0,
      rd / 5.0 + 0.001,
      1.0);
  }
`;

const GRID_FS = `
  precision mediump float;
  void main() { gl_FragColor = vec4(0.28, 0.28, 0.34, 0.5); }
`;

function compileShader(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error("Shader compile error:", gl.getShaderInfoLog(s));
  }
  return s;
}

function linkProgram(gl: WebGLRenderingContext, vs: string, fs: string): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(prog));
  }
  return prog;
}

const BODY_UNIFORMS = ["uSc","uCx","uFy","uXs","uCY","uSY","uCX","uSX","uWX","uWS","uVP","uCol"] as const;
const GRID_UNIFORMS = ["uSc","uCx","uFy","uXs","uCY","uSY","uCX","uSX","uVP"] as const;

type UniformMap = Record<string, WebGLUniformLocation | null>;

function uniforms(gl: WebGLRenderingContext, prog: WebGLProgram, names: readonly string[]): UniformMap {
  const out: UniformMap = {};
  for (const n of names) out[n] = gl.getUniformLocation(prog, n);
  return out;
}

/** Build the grid geometry (Z = 0 plane, 20×20 lines from -1.4 to 1.4 m). */
function buildGridVerts(): Float32Array {
  const verts: number[] = [];
  const range = 1.4, count = 20, step = range * 2 / count;
  for (let i = 0; i <= count; i++) {
    const v = -range + i * step;
    verts.push(v, -range, 0, v,  range, 0);   // X lines
    verts.push(-range, v, 0,  range, v, 0);   // Y lines
  }
  return new Float32Array(verts);
}

export function setupWebGL(canvas: HTMLCanvasElement, model: AnnyModel): WebGLBundle {
  const ctx = canvas.getContext("webgl", { alpha: true, antialias: true, depth: true });
  if (!ctx) throw new Error("WebGL not supported");
  const gl: WebGLRenderingContext = ctx;

  // Best-effort GPU name (only available via debug extension)
  const dbgExt = gl.getExtension("WEBGL_debug_renderer_info");
  const gpuName = dbgExt
    ? String(gl.getParameter((dbgExt as { UNMASKED_RENDERER_WEBGL: number }).UNMASKED_RENDERER_WEBGL))
    : "unknown";

  const hasDerivatives = !!gl.getExtension("OES_standard_derivatives");
  const bodyProg = linkProgram(gl, BODY_VS, hasDerivatives ? BODY_FS_DERIV : BODY_FS_FALLBACK);
  const gridProg = linkProgram(gl, GRID_VS, GRID_FS);

  const aBodyPos = gl.getAttribLocation(bodyProg, "aPos");
  const aGridPos = gl.getAttribLocation(gridProg, "aPos");
  const bodyU = uniforms(gl, bodyProg, BODY_UNIFORMS);
  const gridU = uniforms(gl, gridProg, GRID_UNIFORMS);

  // Buffers
  const idxBuf = gl.createBuffer()!;
  const faces16 = new Uint16Array(model.faces.length);
  for (let i = 0; i < model.faces.length; i++) faces16[i] = model.faces[i];
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, faces16, gl.STATIC_DRAW);

  const vtxBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, vtxBuf);
  gl.bufferData(gl.ARRAY_BUFFER, model.vertCount * 12, gl.DYNAMIC_DRAW);

  const gridVerts = buildGridVerts();
  const gridBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
  gl.bufferData(gl.ARRAY_BUFFER, gridVerts, gl.STATIC_DRAW);
  const gridVertCount = gridVerts.length / 3;

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  gl.frontFace(gl.CCW);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  let W = canvas.width, H = canvas.height;
  let twistX = 1, twistS = 0;

  function applyProj(u: UniformMap, v: ViewParams): void {
    const xs = v.xSign ?? -1;
    const camY = v.camY ?? 0;
    const camX = v.camX ?? 0;
    gl.uniform1f(u.uSc!, v.scale);
    gl.uniform1f(u.uCx!, v.cx);
    gl.uniform1f(u.uFy!, v.footY);
    gl.uniform1f(u.uXs!, xs);
    gl.uniform1f(u.uCY!, Math.cos(camY));
    gl.uniform1f(u.uSY!, Math.sin(camY));
    gl.uniform1f(u.uCX!, Math.cos(camX));
    gl.uniform1f(u.uSX!, Math.sin(camX));
    gl.uniform2f(u.uVP!, W, H);
  }

  return {
    gl,
    gpuName,
    uploadVertices(vertices) {
      gl.bindBuffer(gl.ARRAY_BUFFER, vtxBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertices);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    },
    drawBody(view, [r, g, b]) {
      gl.useProgram(bodyProg);
      gl.uniform1f(bodyU.uWX!, twistX);
      gl.uniform1f(bodyU.uWS!, twistS);
      applyProj(bodyU, view);
      gl.uniform4f(bodyU.uCol!, r, g, b, 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, vtxBuf);
      gl.vertexAttribPointer(aBodyPos, 3, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(aBodyPos);
      gl.drawElements(gl.TRIANGLES, model.faceCount * 3, gl.UNSIGNED_SHORT, 0);
    },
    drawGrid(view) {
      gl.useProgram(gridProg);
      applyProj(gridU, view);
      gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
      gl.vertexAttribPointer(aGridPos, 3, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(aGridPos);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
      gl.drawArrays(gl.LINES, 0, gridVertCount);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    },
    setWorldTwist(angle) {
      twistX = Math.cos(angle);
      twistS = Math.sin(angle);
    },
    clear(r = 0.047, g = 0.047, b = 0.071) {
      gl.clearColor(r, g, b, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    },
    resize(w, h) {
      W = w; H = h;
      canvas.width = w; canvas.height = h;
      gl.viewport(0, 0, w, h);
    },
  };
}

/** The min-Z of the rest mesh — usually slightly below 0 (feet anchor). */
export function minRestZ(model: AnnyModel): number {
  let z = 0;
  const v = model.restVertices;
  for (let i = 2; i < v.length; i += 3) {
    if (v[i] < z) z = v[i];
  }
  return z;
}
