/**
 * WebGL pipeline shared by every demo: a face-normal-lit body shader and a
 * static floor-grid shader. Both honor a single screen-space projection
 * (`{cx, footY, scale}`) plus camera orbit (camY) and tilt (camX).
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
  /** Camera orbit around vertical axis, radians. */
  camY?: number;
  /** Camera tilt around horizontal axis, radians (negative = bird's-eye). */
  camX?: number;
}

export interface WebGLBundle {
  gl: WebGLRenderingContext;
  /** Upload posed vertices for this frame (call once before any `drawBody`). */
  uploadVertices(vertices: Float32Array): void;
  /** Upload posed per-vertex normals for this frame (call once per frame). */
  uploadNormals(normals: Float32Array): void;
  /** Draw the body in one view, RGB in 0..1. */
  drawBody(view: ViewParams, color: [number, number, number]): void;
  /** Draw the floor grid in one view (no world-twist applied). */
  drawGrid(view: ViewParams): void;
  /** Set hip-yaw world rotation for upcoming body draws (radians). */
  setWorldTwist(angle: number): void;
  /**
   * Toggle the anatomical front/back tint. When enabled, the rendered body
   * is warm where the underlying rest vertex sits on the anatomical front
   * (rest_y < 0 in Anny world) and cool on the back. Use it to verify
   * orientation visually: FRONT view should be predominantly warm, BACK view
   * predominantly cool. Default: enabled.
   */
  setOrientationTint(enabled: boolean): void;
  /** Clear the framebuffer to bg, prepare for new frame. */
  clear(r?: number, g?: number, b?: number): void;
  /** Resize the GL canvas and viewport. */
  resize(w: number, h: number): void;
  /** Detected GPU name (best effort), or "unknown". */
  gpuName: string;
}

// Smooth-shading body shader. Takes per-vertex positions + per-vertex normals
// (LBS-skinned) + the rest-pose Y coordinate. The world-twist + camera-orbit
// rotations are applied to *both* the position and the normal so lighting
// stays consistent across views. The third attribute `aRY` is the rest-pose
// anatomical Y (Anny world: -y = anatomical front), used by the fragment
// shader for the front/back orientation tint.
const BODY_VS = `
  attribute vec3 aPos;
  attribute vec3 aNrm;
  attribute float aRY;
  uniform float uSc, uCx, uFy, uCY, uSY, uCX, uSX, uWX, uWS;
  uniform vec2 uVP;
  varying vec3 vNrm;
  varying float vRY;

  // Apply: world-twist (around z), camera orbit-Y, camera tilt-X. Identical
  // chain for points and directions — directions skip the screen-space scale
  // and the translation, but the rotation composition is the same.
  vec3 applyRot(vec3 p) {
    float py = p.y * uWX - p.z * uWS;
    float pz = p.y * uWS + p.z * uWX;
    float rx = p.x * uCY - py * uSY;
    float ry = p.x * uSY + py * uCY;
    float rz = ry * uSX + pz * uCX;
    float rd = ry * uCX - pz * uSX;
    return vec3(rx, rd, rz);
  }

  void main() {
    vec3 r = applyRot(aPos);
    vec3 n = applyRot(aNrm);
    vNrm = normalize(n);
    vRY  = aRY;
    // Third-person projection: anatomical-left (world +x) lands on screen
    // right when the subject faces camera. Self-mirror is intentionally NOT
    // supported — see the changelog where it was removed because comparing
    // a self-mirrored model side-by-side with a third-person video caused
    // a confusing left/right swap.
    float sx = uCx + r.x * uSc;
    float sy = uFy - r.z * uSc;
    gl_Position = vec4(sx / uVP.x * 2.0 - 1.0, 1.0 - sy / uVP.y * 2.0, r.y / 5.0, 1.0);
  }
`;

// Soft three-light setup (key, fill, rim) on per-vertex normals — gives the
// clay-matte look. The orientation tint mixes a warm hue toward the
// anatomical front (rest_y < 0) and a cool hue toward the back. uTint = 0
// disables the tint entirely so we can A/B against the plain look.
const BODY_FS = `
  precision mediump float;
  uniform vec4 uCol;
  uniform float uTint;
  varying vec3 vNrm;
  varying float vRY;
  void main() {
    vec3 n = normalize(vNrm);
    float key  = max(dot(n, normalize(vec3( 0.3, -0.7,  0.7))), 0.0);
    float fill = max(dot(n, normalize(vec3(-0.6, -0.3,  0.4))), 0.0);
    float rim  = max(dot(n, normalize(vec3( 0.0,  0.8, -0.5))), 0.0);
    float L = 0.42 + 0.55 * key + 0.20 * fill + 0.10 * rim * rim;

    // Anatomical tint: rest_y is hip-centered, body depth roughly ±0.15 m,
    // so 6.5x makes the tint saturate near the edges of the body. Sign flip
    // because Anny -y = anatomical front.
    float frontness = clamp(-vRY * 6.5, -1.0, 1.0);
    vec3 warm = vec3(1.10, 0.78, 0.68);  // chest/face side
    vec3 cool = vec3(0.74, 0.84, 1.05);  // back/spine side
    vec3 tint = mix(cool, warm, frontness * 0.5 + 0.5);
    vec3 base = mix(uCol.rgb, uCol.rgb * tint, uTint);

    gl_FragColor = vec4(base * L, 1.0);
  }
`;

const GRID_VS = `
  attribute vec3 aPos;
  uniform float uSc, uCx, uFy, uCY, uSY, uCX, uSX;
  uniform vec2 uVP;
  void main() {
    float rx = aPos.x * uCY - aPos.y * uSY;
    float ry = aPos.x * uSY + aPos.y * uCY;
    float rz = ry * uSX + aPos.z * uCX;
    float rd = ry * uCX - aPos.z * uSX;
    gl_Position = vec4(
      (uCx + rx * uSc) / uVP.x * 2.0 - 1.0,
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

const BODY_UNIFORMS = ["uSc","uCx","uFy","uCY","uSY","uCX","uSX","uWX","uWS","uVP","uCol","uTint"] as const;
const GRID_UNIFORMS = ["uSc","uCx","uFy","uCY","uSY","uCX","uSX","uVP"] as const;

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

  const bodyProg = linkProgram(gl, BODY_VS, BODY_FS);
  const gridProg = linkProgram(gl, GRID_VS, GRID_FS);

  const aBodyPos = gl.getAttribLocation(bodyProg, "aPos");
  const aBodyNrm = gl.getAttribLocation(bodyProg, "aNrm");
  const aBodyRY  = gl.getAttribLocation(bodyProg, "aRY");
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

  const nrmBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
  gl.bufferData(gl.ARRAY_BUFFER, model.vertCount * 12, gl.DYNAMIC_DRAW);

  // Static rest-Y attribute — one float per vertex, uploaded once.
  const restY = new Float32Array(model.vertCount);
  for (let v = 0; v < model.vertCount; v++) restY[v] = model.restVertices[v * 3 + 1];
  const ryBuf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, ryBuf);
  gl.bufferData(gl.ARRAY_BUFFER, restY, gl.STATIC_DRAW);

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
  let tintEnabled = 1.0;

  function applyProj(u: UniformMap, v: ViewParams): void {
    const camY = v.camY ?? 0;
    const camX = v.camX ?? 0;
    gl.uniform1f(u.uSc!, v.scale);
    gl.uniform1f(u.uCx!, v.cx);
    gl.uniform1f(u.uFy!, v.footY);
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
    uploadNormals(normals) {
      gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, normals);
    },
    drawBody(view, [r, g, b]) {
      gl.useProgram(bodyProg);
      gl.uniform1f(bodyU.uWX!, twistX);
      gl.uniform1f(bodyU.uWS!, twistS);
      applyProj(bodyU, view);
      gl.uniform4f(bodyU.uCol!, r, g, b, 1);
      gl.uniform1f(bodyU.uTint!, tintEnabled);

      gl.bindBuffer(gl.ARRAY_BUFFER, vtxBuf);
      gl.vertexAttribPointer(aBodyPos, 3, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(aBodyPos);

      gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
      gl.vertexAttribPointer(aBodyNrm, 3, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(aBodyNrm);

      gl.bindBuffer(gl.ARRAY_BUFFER, ryBuf);
      gl.vertexAttribPointer(aBodyRY, 1, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(aBodyRY);

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
    setOrientationTint(enabled) {
      tintEnabled = enabled ? 1.0 : 0.0;
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
