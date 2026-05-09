// OpenGL → WebGL2 host implementation for wasm modules compiled with wasi-sdk.
// Covers modern GL (ES 3.0), immediate mode emulation, and fixed-function matrices.
//
// Usage:
//   const gl = new GL({ memory })
//   // after _start() runs (window + context exist):
//   gl.setGL(glfw.getContextGL())
//   // include in env imports:
//   env: { ...glfw, ...gl, memory }

const dec = new TextDecoder()
const enc = new TextEncoder()

// ── Matrix math (column-major, matches OpenGL) ───────────────────────────────

function m4id() {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
}

function m4mul(a, b) {
  const o = new Float32Array(16)
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0
      for (let k = 0; k < 4; k++) s += a[r + k*4] * b[k + c*4]
      o[r + c*4] = s
    }
  return o
}

function m4ortho(l, r, b, t, n, f) {
  return new Float32Array([
    2/(r-l),       0,         0, 0,
    0,       2/(t-b),         0, 0,
    0,             0,  -2/(f-n), 0,
    -(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n), 1
  ])
}

function m4frustum(l, r, b, t, n, f) {
  return new Float32Array([
    2*n/(r-l),         0,            0,  0,
    0,         2*n/(t-b),            0,  0,
    (r+l)/(r-l), (t+b)/(t-b), -(f+n)/(f-n), -1,
    0,                 0,  -2*f*n/(f-n),  0
  ])
}

function m4translate(x, y, z) {
  const m = m4id()
  m[12] = x; m[13] = y; m[14] = z
  return m
}

function m4scale(x, y, z) {
  return new Float32Array([x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1])
}

function m4rotate(angleDeg, x, y, z) {
  const a = angleDeg * Math.PI / 180
  const c = Math.cos(a), s = Math.sin(a)
  const len = Math.sqrt(x*x + y*y + z*z)
  if (len === 0) return m4id()
  x /= len; y /= len; z /= len
  const t = 1 - c
  return new Float32Array([
    t*x*x+c,   t*x*y+s*z, t*x*z-s*y, 0,
    t*x*y-s*z, t*y*y+c,   t*y*z+s*x, 0,
    t*x*z+s*y, t*y*z-s*x, t*z*z+c,   0,
    0,         0,         0,          1
  ])
}

// Immediate mode: stride in floats per vertex: pos(3)+color(4)+uv(2)+normal(3)
const IM_STRIDE = 12

const IM_VERT_SRC = `#version 300 es
in vec3 a_pos; in vec4 a_col; in vec2 a_uv; in vec3 a_nrm;
uniform mat4 u_mv; uniform mat4 u_proj;
out vec4 v_col; out vec2 v_uv;
void main() {
  gl_Position = u_proj * u_mv * vec4(a_pos, 1.0);
  v_col = a_col; v_uv = a_uv;
}`

const IM_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_col; in vec2 v_uv;
uniform sampler2D u_tex; uniform int u_use_tex;
out vec4 fragColor;
void main() {
  fragColor = u_use_tex != 0 ? v_col * texture(u_tex, v_uv) : v_col;
}`

// GL_QUADS → triangles: every 4 verts → 2 tris
function quadsToTris(verts, stride) {
  const out = []
  for (let i = 0; i < verts.length; i += stride * 4) {
    const v = (j) => verts.slice(i + j*stride, i + (j+1)*stride)
    out.push(...v(0), ...v(1), ...v(2))
    out.push(...v(0), ...v(2), ...v(3))
  }
  return out
}

export class GL {
  constructor({ memory } = {}) {
    this._memory = memory || null
    this._gl = null
    this._instance = null

    // Object registries: integer ID → WebGL object
    this._buffers   = new Map()
    this._textures  = new Map()
    this._vaos      = new Map()
    this._shaders   = new Map()
    this._programs  = new Map()
    this._fbos      = new Map()
    this._rbos      = new Map()
    this._ulocs     = new Map()  // uniform location ID → WebGLUniformLocation
    this._nextId    = 1

    // String allocation cache (same pattern as Glfw)
    this._strCache  = new Map()

    // Fixed-function matrix stacks
    this._matMode   = 0x1700  // GL_MODELVIEW
    this._mvStack   = [m4id()]
    this._projStack = [m4id()]
    this._texStack  = [m4id()]

    // Immediate mode state
    this._imMode    = -1
    this._imVerts   = []
    this._imColor   = [1, 1, 1, 1]
    this._imNormal  = [0, 0, 1]
    this._imUV      = [0, 0]
    this._imVao     = null
    this._imVbo     = null
    this._imProg    = null
    this._imLocs    = null  // uniform locations for im program
    this._imTex     = false

    // Bind all methods for use as wasm imports
    const proto = Object.getPrototypeOf(this)
    for (const key of Object.getOwnPropertyNames(proto)) {
      if (key === 'constructor') continue
      const desc = Object.getOwnPropertyDescriptor(proto, key)
      if (typeof desc.value === 'function') this[key] = desc.value.bind(this)
    }
  }

  // Call after instantiation with the wasm exports object (mirrors WasiPreview1.start).
  // Call setGL(glfw.getContextGL()) after exports._start() to wire the WebGL2 context.
  start(exports) {
    this._instance = { exports }
    if (!this._memory && exports.memory) this._memory = exports.memory
  }

  // Legacy: call with the full WebAssembly.Instance object.
  setInstance(instance) {
    this._instance = instance
    if (!this._memory && instance.exports.memory) {
      this._memory = instance.exports.memory
    }
  }

  setGL(gl) {
    this._gl = gl
    this._initImmediate()
  }

  // ── Memory helpers ──────────────────────────────────────────────────────────

  get _view() { return new DataView(this._memory.buffer) }

  _readStr(ptr) {
    if (!ptr) return ''
    const buf = new Uint8Array(this._memory.buffer)
    let end = ptr; while (buf[end]) end++
    return dec.decode(buf.subarray(ptr, end))
  }

  _wi32(ptr, v) { if (ptr) this._view.setInt32(ptr, v | 0, true) }
  _wf32(ptr, v) { if (ptr) this._view.setFloat32(ptr, v, true) }

  _allocStr(str) {
    if (!str) return 0
    if (this._strCache.has(str)) return this._strCache.get(str)
    const fn = this._instance?.exports?.malloc
    if (!fn) return 0
    const bytes = enc.encode(str + '\0')
    const ptr = fn(bytes.length)
    new Uint8Array(this._memory.buffer, ptr, bytes.length).set(bytes)
    this._strCache.set(str, ptr)
    return ptr
  }

  _f32s(ptr, n) {
    // Return Float32Array view into wasm memory (ptr must be 4-byte aligned)
    if (!ptr) return null
    return new Float32Array(this._memory.buffer, ptr, n)
  }

  _i32s(ptr, n) {
    if (!ptr) return null
    return new Int32Array(this._memory.buffer, ptr, n)
  }

  _bytes(ptr, n) {
    if (!ptr) return null
    return new Uint8Array(this._memory.buffer, ptr, n)
  }

  // ── Immediate mode setup ────────────────────────────────────────────────────

  _initImmediate() {
    const g = this._gl
    const vs = g.createShader(g.VERTEX_SHADER)
    g.shaderSource(vs, IM_VERT_SRC)
    g.compileShader(vs)
    const fs = g.createShader(g.FRAGMENT_SHADER)
    g.shaderSource(fs, IM_FRAG_SRC)
    g.compileShader(fs)
    const prog = g.createProgram()
    g.attachShader(prog, vs); g.attachShader(prog, fs)
    g.linkProgram(prog)
    g.deleteShader(vs); g.deleteShader(fs)

    this._imProg = prog
    this._imLocs = {
      mv:      g.getUniformLocation(prog, 'u_mv'),
      proj:    g.getUniformLocation(prog, 'u_proj'),
      tex:     g.getUniformLocation(prog, 'u_tex'),
      useTex:  g.getUniformLocation(prog, 'u_use_tex'),
    }

    this._imVao = g.createVertexArray()
    g.bindVertexArray(this._imVao)
    this._imVbo = g.createBuffer()
    g.bindBuffer(g.ARRAY_BUFFER, this._imVbo)

    const stride = IM_STRIDE * 4
    const al = (name, size, off) => {
      const loc = g.getAttribLocation(prog, name)
      g.vertexAttribPointer(loc, size, g.FLOAT, false, stride, off * 4)
      g.enableVertexAttribArray(loc)
    }
    al('a_pos', 3, 0)
    al('a_col', 4, 3)
    al('a_uv',  2, 7)
    al('a_nrm', 3, 9)

    g.bindVertexArray(null)
    g.bindBuffer(g.ARRAY_BUFFER, null)
  }

  _flushImmediate() {
    if (!this._imVerts.length) return
    const g = this._gl

    let mode = this._imMode
    let verts = this._imVerts

    // Emulate GL_QUADS → GL_TRIANGLES
    if (mode === 7 /*GL_QUADS*/) {
      verts = quadsToTris(verts, IM_STRIDE)
      mode = 4 /*GL_TRIANGLES*/
    }
    // GL_POLYGON treated as triangle fan
    if (mode === 9 /*GL_POLYGON*/) mode = 6 /*GL_TRIANGLE_FAN*/

    const data = new Float32Array(verts)

    // Save state
    const prevProg = g.getParameter(g.CURRENT_PROGRAM)
    const prevVao  = g.getParameter(g.VERTEX_ARRAY_BINDING)
    const prevVbo  = g.getParameter(g.ARRAY_BUFFER_BINDING)

    g.useProgram(this._imProg)
    g.uniformMatrix4fv(this._imLocs.mv,   false, this._mvStack[this._mvStack.length-1])
    g.uniformMatrix4fv(this._imLocs.proj, false, this._projStack[this._projStack.length-1])
    g.uniform1i(this._imLocs.useTex, this._imTex ? 1 : 0)

    g.bindVertexArray(this._imVao)
    g.bindBuffer(g.ARRAY_BUFFER, this._imVbo)
    g.bufferData(g.ARRAY_BUFFER, data, g.STREAM_DRAW)
    g.drawArrays(mode, 0, verts.length / IM_STRIDE)

    // Restore state
    g.bindVertexArray(prevVao)
    g.bindBuffer(g.ARRAY_BUFFER, prevVbo)
    g.useProgram(prevProg)

    this._imVerts = []
  }

  _curMatrix() {
    if (this._matMode === 0x1701) return this._projStack
    if (this._matMode === 0x1702) return this._texStack
    return this._mvStack
  }

  // ── Clear ───────────────────────────────────────────────────────────────────

  glClearColor(r, g, b, a) { this._gl.clearColor(r, g, b, a) }
  glClearDepth(d)           { this._gl.clearDepth(d) }
  glClearDepthf(d)          { this._gl.clearDepth(d) }
  glClearStencil(s)         { this._gl.clearStencil(s) }
  glClear(mask)             { this._gl.clear(mask) }

  // ── State ───────────────────────────────────────────────────────────────────

  glEnable(cap)                        { this._gl.enable(cap) }
  glDisable(cap)                       { this._gl.disable(cap) }
  glIsEnabled(cap)                     { return this._gl.isEnabled(cap) ? 1 : 0 }
  glBlendFunc(sfactor, dfactor)        { this._gl.blendFunc(sfactor, dfactor) }
  glBlendFuncSeparate(srcRGB, dstRGB, srcA, dstA) { this._gl.blendFuncSeparate(srcRGB, dstRGB, srcA, dstA) }
  glBlendEquation(mode)                { this._gl.blendEquation(mode) }
  glBlendEquationSeparate(modeRGB, modeA) { this._gl.blendEquationSeparate(modeRGB, modeA) }
  glBlendColor(r, g, b, a)            { this._gl.blendColor(r, g, b, a) }
  glDepthFunc(func)                    { this._gl.depthFunc(func) }
  glDepthMask(flag)                    { this._gl.depthMask(!!flag) }
  glDepthRange(near, far)              { this._gl.depthRange(near, far) }
  glDepthRangef(near, far)             { this._gl.depthRange(near, far) }
  glColorMask(r, g, b, a)             { this._gl.colorMask(!!r, !!g, !!b, !!a) }
  glCullFace(mode)                     { this._gl.cullFace(mode) }
  glFrontFace(mode)                    { this._gl.frontFace(mode) }
  glLineWidth(w)                       { this._gl.lineWidth(w) }
  glPolygonOffset(factor, units)       { this._gl.polygonOffset(factor, units) }
  glScissor(x, y, w, h)               { this._gl.scissor(x, y, w, h) }
  glViewport(x, y, w, h)              { this._gl.viewport(x, y, w, h) }
  glSampleCoverage(value, invert)      { this._gl.sampleCoverage(value, !!invert) }
  glStencilFunc(func, ref, mask)       { this._gl.stencilFunc(func, ref, mask) }
  glStencilFuncSeparate(face, f, r, m) { this._gl.stencilFuncSeparate(face, f, r, m) }
  glStencilOp(fail, zfail, zpass)      { this._gl.stencilOp(fail, zfail, zpass) }
  glStencilOpSeparate(face, f, zf, zp) { this._gl.stencilOpSeparate(face, f, zf, zp) }
  glStencilMask(mask)                  { this._gl.stencilMask(mask) }
  glStencilMaskSeparate(face, mask)    { this._gl.stencilMaskSeparate(face, mask) }
  glPixelStorei(pname, param)          { this._gl.pixelStorei(pname, param) }
  glFinish()                           { this._gl.finish() }
  glFlush()                            { this._gl.flush() }

  // ── Buffers ─────────────────────────────────────────────────────────────────

  glGenBuffers(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._nextId++
      this._buffers.set(id, this._gl.createBuffer())
      this._wi32(ptr + i * 4, id)
    }
  }

  glDeleteBuffers(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._view.getInt32(ptr + i * 4, true)
      const buf = this._buffers.get(id)
      if (buf) { this._gl.deleteBuffer(buf); this._buffers.delete(id) }
    }
  }

  glBindBuffer(target, id) {
    this._gl.bindBuffer(target, this._buffers.get(id) ?? null)
  }

  glBufferData(target, size, dataPtr, usage) {
    const data = dataPtr ? new Uint8Array(this._memory.buffer, dataPtr, size) : null
    this._gl.bufferData(target, data ?? size, usage)
  }

  glBufferSubData(target, offset, size, dataPtr) {
    const data = new Uint8Array(this._memory.buffer, dataPtr, size)
    this._gl.bufferSubData(target, offset, data)
  }

  glIsBuffer(id) { return this._buffers.has(id) ? 1 : 0 }

  // ── Vertex Arrays ────────────────────────────────────────────────────────────

  glGenVertexArrays(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._nextId++
      this._vaos.set(id, this._gl.createVertexArray())
      this._wi32(ptr + i * 4, id)
    }
  }

  glDeleteVertexArrays(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._view.getInt32(ptr + i * 4, true)
      const vao = this._vaos.get(id)
      if (vao) { this._gl.deleteVertexArray(vao); this._vaos.delete(id) }
    }
  }

  glBindVertexArray(id) {
    this._gl.bindVertexArray(this._vaos.get(id) ?? null)
  }

  glIsVertexArray(id) { return this._vaos.has(id) ? 1 : 0 }

  glEnableVertexAttribArray(index)  { this._gl.enableVertexAttribArray(index) }
  glDisableVertexAttribArray(index) { this._gl.disableVertexAttribArray(index) }

  glVertexAttribPointer(index, size, type, normalized, stride, offset) {
    this._gl.vertexAttribPointer(index, size, type, !!normalized, stride, offset)
  }

  glVertexAttribIPointer(index, size, type, stride, offset) {
    this._gl.vertexAttribIPointer(index, size, type, stride, offset)
  }

  glVertexAttrib1f(i, x)          { this._gl.vertexAttrib1f(i, x) }
  glVertexAttrib2f(i, x, y)       { this._gl.vertexAttrib2f(i, x, y) }
  glVertexAttrib3f(i, x, y, z)    { this._gl.vertexAttrib3f(i, x, y, z) }
  glVertexAttrib4f(i, x, y, z, w) { this._gl.vertexAttrib4f(i, x, y, z, w) }

  glVertexAttrib1fv(i, ptr) { this._gl.vertexAttrib1fv(i, this._f32s(ptr, 1)) }
  glVertexAttrib2fv(i, ptr) { this._gl.vertexAttrib2fv(i, this._f32s(ptr, 2)) }
  glVertexAttrib3fv(i, ptr) { this._gl.vertexAttrib3fv(i, this._f32s(ptr, 3)) }
  glVertexAttrib4fv(i, ptr) { this._gl.vertexAttrib4fv(i, this._f32s(ptr, 4)) }

  glVertexAttribDivisor(index, divisor) { this._gl.vertexAttribDivisor(index, divisor) }

  // ── Shaders ──────────────────────────────────────────────────────────────────

  glCreateShader(type) {
    const id = this._nextId++
    this._shaders.set(id, this._gl.createShader(type))
    return id
  }

  glDeleteShader(id) {
    const s = this._shaders.get(id)
    if (s) { this._gl.deleteShader(s); this._shaders.delete(id) }
  }

  glShaderSource(id, count, stringsPtrPtr, lengthsPtrPtr) {
    const v = this._view
    let src = ''
    for (let i = 0; i < count; i++) {
      const strPtr = v.getInt32(stringsPtrPtr + i * 4, true)
      const len = lengthsPtrPtr ? v.getInt32(lengthsPtrPtr + i * 4, true) : -1
      if (len >= 0) {
        src += dec.decode(new Uint8Array(this._memory.buffer, strPtr, len))
      } else {
        src += this._readStr(strPtr)
      }
    }
    this._gl.shaderSource(this._shaders.get(id), src)
  }

  glCompileShader(id) {
    this._gl.compileShader(this._shaders.get(id))
  }

  glGetShaderiv(id, pname, paramsPtr) {
    const g = this._gl
    const s = this._shaders.get(id)
    let val
    if (pname === 0x8B81 /*COMPILE_STATUS*/) val = g.getShaderParameter(s, pname) ? 1 : 0
    else if (pname === 0x8B84 /*INFO_LOG_LENGTH*/) val = (g.getShaderInfoLog(s) || '').length + 1
    else if (pname === 0x8B4F /*SHADER_TYPE*/) val = g.getShaderParameter(s, pname)
    else if (pname === 0x8B80 /*DELETE_STATUS*/) val = g.getShaderParameter(s, pname) ? 1 : 0
    else val = 0
    this._wi32(paramsPtr, val)
  }

  glGetShaderInfoLog(id, maxLen, lengthPtr, logPtr) {
    const log = this._gl.getShaderInfoLog(this._shaders.get(id)) || ''
    const bytes = enc.encode(log.substring(0, maxLen - 1) + '\0')
    new Uint8Array(this._memory.buffer, logPtr, bytes.length).set(bytes)
    this._wi32(lengthPtr, bytes.length - 1)
  }

  glIsShader(id) { return this._shaders.has(id) ? 1 : 0 }

  // ── Programs ─────────────────────────────────────────────────────────────────

  glCreateProgram() {
    const id = this._nextId++
    this._programs.set(id, this._gl.createProgram())
    return id
  }

  glDeleteProgram(id) {
    const p = this._programs.get(id)
    if (p) { this._gl.deleteProgram(p); this._programs.delete(id) }
  }

  glAttachShader(progId, shaderId) {
    this._gl.attachShader(this._programs.get(progId), this._shaders.get(shaderId))
  }

  glDetachShader(progId, shaderId) {
    this._gl.detachShader(this._programs.get(progId), this._shaders.get(shaderId))
  }

  glBindAttribLocation(progId, index, namePtr) {
    this._gl.bindAttribLocation(this._programs.get(progId), index, this._readStr(namePtr))
  }

  glLinkProgram(id) {
    this._gl.linkProgram(this._programs.get(id))
  }

  glValidateProgram(id) {
    this._gl.validateProgram(this._programs.get(id))
  }

  glUseProgram(id) {
    this._gl.useProgram(id ? this._programs.get(id) : null)
  }

  glGetProgramiv(id, pname, paramsPtr) {
    const g = this._gl
    const p = this._programs.get(id)
    let val
    switch (pname) {
      case 0x8B82: val = g.getProgramParameter(p, pname) ? 1 : 0; break // LINK_STATUS
      case 0x8B80: val = g.getProgramParameter(p, pname) ? 1 : 0; break // DELETE_STATUS
      case 0x8B83: val = g.getProgramParameter(p, pname) ? 1 : 0; break // VALIDATE_STATUS
      case 0x8B85: val = (g.getProgramInfoLog(p) || '').length + 1; break // INFO_LOG_LENGTH
      default: val = g.getProgramParameter(p, pname) ?? 0
    }
    this._wi32(paramsPtr, typeof val === 'boolean' ? (val ? 1 : 0) : val)
  }

  glGetProgramInfoLog(id, maxLen, lengthPtr, logPtr) {
    const log = this._gl.getProgramInfoLog(this._programs.get(id)) || ''
    const bytes = enc.encode(log.substring(0, maxLen - 1) + '\0')
    new Uint8Array(this._memory.buffer, logPtr, bytes.length).set(bytes)
    this._wi32(lengthPtr, bytes.length - 1)
  }

  glGetAttribLocation(progId, namePtr) {
    return this._gl.getAttribLocation(this._programs.get(progId), this._readStr(namePtr))
  }

  glIsProgram(id) { return this._programs.has(id) ? 1 : 0 }

  // ── Uniforms ─────────────────────────────────────────────────────────────────

  glGetUniformLocation(progId, namePtr) {
    const loc = this._gl.getUniformLocation(this._programs.get(progId), this._readStr(namePtr))
    if (loc == null) return -1
    const id = this._nextId++
    this._ulocs.set(id, loc)
    return id
  }

  _ul(id) { return id >= 0 ? this._ulocs.get(id) : null }

  glUniform1f(l, x)          { this._gl.uniform1f(this._ul(l), x) }
  glUniform2f(l, x, y)       { this._gl.uniform2f(this._ul(l), x, y) }
  glUniform3f(l, x, y, z)    { this._gl.uniform3f(this._ul(l), x, y, z) }
  glUniform4f(l, x, y, z, w) { this._gl.uniform4f(this._ul(l), x, y, z, w) }
  glUniform1i(l, x)          { this._gl.uniform1i(this._ul(l), x) }
  glUniform2i(l, x, y)       { this._gl.uniform2i(this._ul(l), x, y) }
  glUniform3i(l, x, y, z)    { this._gl.uniform3i(this._ul(l), x, y, z) }
  glUniform4i(l, x, y, z, w) { this._gl.uniform4i(this._ul(l), x, y, z, w) }
  glUniform1ui(l, x)          { this._gl.uniform1ui(this._ul(l), x) }
  glUniform2ui(l, x, y)       { this._gl.uniform2ui(this._ul(l), x, y) }
  glUniform3ui(l, x, y, z)    { this._gl.uniform3ui(this._ul(l), x, y, z) }
  glUniform4ui(l, x, y, z, w) { this._gl.uniform4ui(this._ul(l), x, y, z, w) }

  glUniform1fv(l, n, ptr)   { this._gl.uniform1fv(this._ul(l), this._f32s(ptr, n*1)) }
  glUniform2fv(l, n, ptr)   { this._gl.uniform2fv(this._ul(l), this._f32s(ptr, n*2)) }
  glUniform3fv(l, n, ptr)   { this._gl.uniform3fv(this._ul(l), this._f32s(ptr, n*3)) }
  glUniform4fv(l, n, ptr)   { this._gl.uniform4fv(this._ul(l), this._f32s(ptr, n*4)) }
  glUniform1iv(l, n, ptr)   { this._gl.uniform1iv(this._ul(l), this._i32s(ptr, n*1)) }
  glUniform2iv(l, n, ptr)   { this._gl.uniform2iv(this._ul(l), this._i32s(ptr, n*2)) }
  glUniform3iv(l, n, ptr)   { this._gl.uniform3iv(this._ul(l), this._i32s(ptr, n*3)) }
  glUniform4iv(l, n, ptr)   { this._gl.uniform4iv(this._ul(l), this._i32s(ptr, n*4)) }
  glUniform1uiv(l, n, ptr)  { this._gl.uniform1uiv(this._ul(l), new Uint32Array(this._memory.buffer, ptr, n)) }
  glUniform2uiv(l, n, ptr)  { this._gl.uniform2uiv(this._ul(l), new Uint32Array(this._memory.buffer, ptr, n*2)) }
  glUniform3uiv(l, n, ptr)  { this._gl.uniform3uiv(this._ul(l), new Uint32Array(this._memory.buffer, ptr, n*3)) }
  glUniform4uiv(l, n, ptr)  { this._gl.uniform4uiv(this._ul(l), new Uint32Array(this._memory.buffer, ptr, n*4)) }

  glUniformMatrix2fv(l, n, t, ptr) { this._gl.uniformMatrix2fv(this._ul(l), !!t, this._f32s(ptr, n*4)) }
  glUniformMatrix3fv(l, n, t, ptr) { this._gl.uniformMatrix3fv(this._ul(l), !!t, this._f32s(ptr, n*9)) }
  glUniformMatrix4fv(l, n, t, ptr) { this._gl.uniformMatrix4fv(this._ul(l), !!t, this._f32s(ptr, n*16)) }
  glUniformMatrix2x3fv(l,n,t,ptr)  { this._gl.uniformMatrix2x3fv(this._ul(l), !!t, this._f32s(ptr, n*6)) }
  glUniformMatrix3x2fv(l,n,t,ptr)  { this._gl.uniformMatrix3x2fv(this._ul(l), !!t, this._f32s(ptr, n*6)) }
  glUniformMatrix2x4fv(l,n,t,ptr)  { this._gl.uniformMatrix2x4fv(this._ul(l), !!t, this._f32s(ptr, n*8)) }
  glUniformMatrix4x2fv(l,n,t,ptr)  { this._gl.uniformMatrix4x2fv(this._ul(l), !!t, this._f32s(ptr, n*8)) }
  glUniformMatrix3x4fv(l,n,t,ptr)  { this._gl.uniformMatrix3x4fv(this._ul(l), !!t, this._f32s(ptr, n*12)) }
  glUniformMatrix4x3fv(l,n,t,ptr)  { this._gl.uniformMatrix4x3fv(this._ul(l), !!t, this._f32s(ptr, n*12)) }

  // ── Textures ─────────────────────────────────────────────────────────────────

  glGenTextures(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._nextId++
      this._textures.set(id, this._gl.createTexture())
      this._wi32(ptr + i * 4, id)
    }
  }

  glDeleteTextures(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._view.getInt32(ptr + i * 4, true)
      const t = this._textures.get(id)
      if (t) { this._gl.deleteTexture(t); this._textures.delete(id) }
    }
  }

  glBindTexture(target, id) {
    this._gl.bindTexture(target, id ? this._textures.get(id) : null)
  }

  glActiveTexture(unit) { this._gl.activeTexture(unit) }

  glTexImage2D(target, level, internalFormat, width, height, border, format, type, dataPtr) {
    const g = this._gl
    if (!dataPtr) {
      g.texImage2D(target, level, internalFormat, width, height, border, format, type, null)
      return
    }
    const bytesPerPixel = this._glBytesPerPixel(format, type)
    const size = width * height * bytesPerPixel
    const src = this._typedView(type, dataPtr, size / this._glTypeSize(type))
    g.texImage2D(target, level, internalFormat, width, height, border, format, type, src)
  }

  glTexImage3D(target, level, internalFmt, w, h, depth, border, fmt, type, dataPtr) {
    const g = this._gl
    if (!dataPtr) {
      g.texImage3D(target, level, internalFmt, w, h, depth, border, fmt, type, null)
      return
    }
    const bytesPerPixel = this._glBytesPerPixel(fmt, type)
    const size = w * h * depth * bytesPerPixel
    const src = this._typedView(type, dataPtr, size / this._glTypeSize(type))
    g.texImage3D(target, level, internalFmt, w, h, depth, border, fmt, type, src)
  }

  glTexSubImage2D(target, level, xoff, yoff, w, h, format, type, dataPtr) {
    const size = w * h * this._glBytesPerPixel(format, type)
    const src = this._typedView(type, dataPtr, size / this._glTypeSize(type))
    this._gl.texSubImage2D(target, level, xoff, yoff, w, h, format, type, src)
  }

  glTexParameteri(target, pname, param) { this._gl.texParameteri(target, pname, param) }
  glTexParameterf(target, pname, param) { this._gl.texParameterf(target, pname, param) }
  glGenerateMipmap(target)              { this._gl.generateMipmap(target) }
  glIsTexture(id)                       { return this._textures.has(id) ? 1 : 0 }

  _glTypeSize(type) {
    const map = { 0x1400:1, 0x1401:1, 0x1402:2, 0x1403:2, 0x1404:4, 0x1405:4, 0x1406:4 }
    return map[type] ?? 1
  }

  _glBytesPerPixel(format, type) {
    const channels = { 0x1903:1, 0x1902:1, 0x1904:3, 0x1908:4, 0x8227:2, 0x8228:4, 0x8D94:1, 0x8D95:2, 0x8D96:3, 0x8D97:4 }
    return (channels[format] ?? 4) * this._glTypeSize(type)
  }

  _typedView(type, ptr, count) {
    switch (type) {
      case 0x1406: return new Float32Array(this._memory.buffer, ptr, count)   // FLOAT
      case 0x1405: return new Uint32Array(this._memory.buffer, ptr, count)    // UNSIGNED_INT
      case 0x1404: return new Int32Array(this._memory.buffer, ptr, count)     // INT
      case 0x1403: return new Uint16Array(this._memory.buffer, ptr, count)    // UNSIGNED_SHORT
      case 0x1402: return new Int16Array(this._memory.buffer, ptr, count)     // SHORT
      case 0x1401: return new Uint8Array(this._memory.buffer, ptr, count)     // UNSIGNED_BYTE
      case 0x1400: return new Int8Array(this._memory.buffer, ptr, count)      // BYTE
      default:     return new Uint8Array(this._memory.buffer, ptr, count)
    }
  }

  // ── Framebuffers ─────────────────────────────────────────────────────────────

  glGenFramebuffers(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._nextId++
      this._fbos.set(id, this._gl.createFramebuffer())
      this._wi32(ptr + i * 4, id)
    }
  }

  glDeleteFramebuffers(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._view.getInt32(ptr + i * 4, true)
      const f = this._fbos.get(id)
      if (f) { this._gl.deleteFramebuffer(f); this._fbos.delete(id) }
    }
  }

  glBindFramebuffer(target, id) {
    this._gl.bindFramebuffer(target, id ? this._fbos.get(id) : null)
  }

  glFramebufferTexture2D(target, attachment, texTarget, texId, level) {
    this._gl.framebufferTexture2D(target, attachment, texTarget, this._textures.get(texId) ?? null, level)
  }

  glFramebufferRenderbuffer(target, attachment, rbTarget, rbId) {
    this._gl.framebufferRenderbuffer(target, attachment, rbTarget, this._rbos.get(rbId) ?? null)
  }

  glCheckFramebufferStatus(target) { return this._gl.checkFramebufferStatus(target) }
  glIsFramebuffer(id) { return this._fbos.has(id) ? 1 : 0 }

  glGenRenderbuffers(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._nextId++
      this._rbos.set(id, this._gl.createRenderbuffer())
      this._wi32(ptr + i * 4, id)
    }
  }

  glDeleteRenderbuffers(n, ptr) {
    for (let i = 0; i < n; i++) {
      const id = this._view.getInt32(ptr + i * 4, true)
      const r = this._rbos.get(id)
      if (r) { this._gl.deleteRenderbuffer(r); this._rbos.delete(id) }
    }
  }

  glBindRenderbuffer(target, id) {
    this._gl.bindRenderbuffer(target, id ? this._rbos.get(id) : null)
  }

  glRenderbufferStorage(target, internalFormat, w, h) {
    this._gl.renderbufferStorage(target, internalFormat, w, h)
  }

  glRenderbufferStorageMultisample(target, samples, internalFormat, w, h) {
    this._gl.renderbufferStorageMultisample(target, samples, internalFormat, w, h)
  }

  glIsRenderbuffer(id) { return this._rbos.has(id) ? 1 : 0 }

  glBlitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter) {
    this._gl.blitFramebuffer(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter)
  }

  // ── Drawing ──────────────────────────────────────────────────────────────────

  glDrawArrays(mode, first, count) { this._gl.drawArrays(mode, first, count) }

  glDrawElements(mode, count, type, offset) {
    this._gl.drawElements(mode, count, type, offset)
  }

  glDrawArraysInstanced(mode, first, count, instanceCount) {
    this._gl.drawArraysInstanced(mode, first, count, instanceCount)
  }

  glDrawElementsInstanced(mode, count, type, offset, instanceCount) {
    this._gl.drawElementsInstanced(mode, count, type, offset, instanceCount)
  }

  glDrawRangeElements(mode, start, end, count, type, offset) {
    this._gl.drawRangeElements(mode, start, end, count, type, offset)
  }

  glMultiDrawArrays(mode, firstsPtr, countsPtr, drawCount) {
    const v = this._view
    for (let i = 0; i < drawCount; i++) {
      const first = v.getInt32(firstsPtr + i * 4, true)
      const count = v.getInt32(countsPtr + i * 4, true)
      this._gl.drawArrays(mode, first, count)
    }
  }

  // ── Read ─────────────────────────────────────────────────────────────────────

  glReadPixels(x, y, w, h, format, type, dataPtr) {
    const size = w * h * this._glBytesPerPixel(format, type)
    const dst = this._typedView(type, dataPtr, size / this._glTypeSize(type))
    this._gl.readPixels(x, y, w, h, format, type, dst)
  }

  // ── Queries / getters ────────────────────────────────────────────────────────

  glGetError() { return this._gl.getError() }

  glGetIntegerv(pname, ptr) {
    const val = this._gl.getParameter(pname)
    if (val == null) return
    const v = this._view
    if (typeof val === 'number' || typeof val === 'boolean') {
      v.setInt32(ptr, (typeof val === 'boolean' ? (val ? 1 : 0) : val) | 0, true)
    } else if (val instanceof Int32Array || val instanceof Uint32Array) {
      for (let i = 0; i < val.length; i++) v.setInt32(ptr + i * 4, val[i], true)
    } else if (val instanceof Float32Array) {
      for (let i = 0; i < val.length; i++) v.setInt32(ptr + i * 4, val[i] | 0, true)
    }
  }

  glGetFloatv(pname, ptr) {
    const val = this._gl.getParameter(pname)
    if (val == null) return
    const v = this._view
    if (typeof val === 'number') {
      v.setFloat32(ptr, val, true)
    } else if (val instanceof Float32Array || val instanceof Int32Array) {
      for (let i = 0; i < val.length; i++) v.setFloat32(ptr + i * 4, val[i], true)
    }
  }

  glGetBooleanv(pname, ptr) {
    const val = this._gl.getParameter(pname)
    new Uint8Array(this._memory.buffer)[ptr] = val ? 1 : 0
  }

  glGetString(name) {
    const g = this._gl
    const str = {
      0x1F00: 'WebGL',                              // GL_VENDOR
      0x1F01: g.getParameter(g.RENDERER),           // GL_RENDERER
      0x1F02: '3.0.0 WebGL2',                       // GL_VERSION
      0x1F03: '',                                    // GL_EXTENSIONS
      0x8B8C: '3.00 ES',                            // GL_SHADING_LANGUAGE_VERSION
    }[name]
    return str != null ? this._allocStr(str) : 0
  }

  glGetStringi(name, index) {
    if (name === 0x1F03 /*GL_EXTENSIONS*/) {
      const ext = this._gl.getSupportedExtensions() ?? []
      return index < ext.length ? this._allocStr(ext[index]) : 0
    }
    return 0
  }

  glGetActiveUniform(progId, index, bufSize, lengthPtr, sizePtr, typePtr, namePtr) {
    const info = this._gl.getActiveUniform(this._programs.get(progId), index)
    if (!info) return
    const bytes = enc.encode(info.name.substring(0, bufSize - 1) + '\0')
    new Uint8Array(this._memory.buffer, namePtr, bytes.length).set(bytes)
    this._wi32(lengthPtr, bytes.length - 1)
    this._wi32(sizePtr, info.size)
    this._wi32(typePtr, info.type)
  }

  glGetActiveAttrib(progId, index, bufSize, lengthPtr, sizePtr, typePtr, namePtr) {
    const info = this._gl.getActiveAttrib(this._programs.get(progId), index)
    if (!info) return
    const bytes = enc.encode(info.name.substring(0, bufSize - 1) + '\0')
    new Uint8Array(this._memory.buffer, namePtr, bytes.length).set(bytes)
    this._wi32(lengthPtr, bytes.length - 1)
    this._wi32(sizePtr, info.size)
    this._wi32(typePtr, info.type)
  }

  glGetUniformfv(progId, locId, paramsPtr) {
    const vals = this._gl.getUniform(this._programs.get(progId), this._ul(locId))
    const v = this._view
    if (typeof vals === 'number') { v.setFloat32(paramsPtr, vals, true); return }
    if (vals?.length) for (let i = 0; i < vals.length; i++) v.setFloat32(paramsPtr + i*4, vals[i], true)
  }

  glGetUniformiv(progId, locId, paramsPtr) {
    const vals = this._gl.getUniform(this._programs.get(progId), this._ul(locId))
    const v = this._view
    if (typeof vals === 'number') { v.setInt32(paramsPtr, vals, true); return }
    if (vals?.length) for (let i = 0; i < vals.length; i++) v.setInt32(paramsPtr + i*4, vals[i], true)
  }

  // ── Fixed-function matrix stack ──────────────────────────────────────────────

  glMatrixMode(mode) { this._matMode = mode }

  glLoadIdentity() {
    const s = this._curMatrix()
    s[s.length - 1] = m4id()
  }

  glLoadMatrixf(ptr) {
    const s = this._curMatrix()
    s[s.length - 1] = new Float32Array(this._memory.buffer.slice(ptr, ptr + 64))
  }

  glLoadMatrixd(ptr) {
    const s = this._curMatrix()
    const v = this._view
    const m = new Float32Array(16)
    for (let i = 0; i < 16; i++) m[i] = v.getFloat64(ptr + i * 8, true)
    s[s.length - 1] = m
  }

  glMultMatrixf(ptr) {
    const s = this._curMatrix()
    const b = new Float32Array(this._memory.buffer.slice(ptr, ptr + 64))
    s[s.length - 1] = m4mul(s[s.length - 1], b)
  }

  glMultMatrixd(ptr) {
    const s = this._curMatrix()
    const v = this._view
    const b = new Float32Array(16)
    for (let i = 0; i < 16; i++) b[i] = v.getFloat64(ptr + i * 8, true)
    s[s.length - 1] = m4mul(s[s.length - 1], b)
  }

  glPushMatrix() {
    const s = this._curMatrix()
    s.push(new Float32Array(s[s.length - 1]))
  }

  glPopMatrix() {
    const s = this._curMatrix()
    if (s.length > 1) s.pop()
  }

  glOrtho(l, r, b, t, n, f) {
    const s = this._curMatrix()
    s[s.length - 1] = m4mul(s[s.length - 1], m4ortho(l, r, b, t, n, f))
  }

  glOrthof(l, r, b, t, n, f) { this.glOrtho(l, r, b, t, n, f) }

  glFrustum(l, r, b, t, n, f) {
    const s = this._curMatrix()
    s[s.length - 1] = m4mul(s[s.length - 1], m4frustum(l, r, b, t, n, f))
  }

  glTranslatef(x, y, z) {
    const s = this._curMatrix()
    s[s.length - 1] = m4mul(s[s.length - 1], m4translate(x, y, z))
  }
  glTranslated(x, y, z) { this.glTranslatef(x, y, z) }

  glScalef(x, y, z) {
    const s = this._curMatrix()
    s[s.length - 1] = m4mul(s[s.length - 1], m4scale(x, y, z))
  }
  glScaled(x, y, z) { this.glScalef(x, y, z) }

  glRotatef(angle, x, y, z) {
    const s = this._curMatrix()
    s[s.length - 1] = m4mul(s[s.length - 1], m4rotate(angle, x, y, z))
  }
  glRotated(angle, x, y, z) { this.glRotatef(angle, x, y, z) }

  // ── Immediate mode ────────────────────────────────────────────────────────────

  glBegin(mode) {
    this._imMode = mode
    this._imVerts = []
  }

  glEnd() {
    this._flushImmediate()
    this._imMode = -1
  }

  _pushVert(x, y, z) {
    this._imVerts.push(
      x, y, z,
      ...this._imColor,
      ...this._imUV,
      ...this._imNormal
    )
  }

  glVertex2f(x, y)       { this._pushVert(x, y, 0) }
  glVertex3f(x, y, z)    { this._pushVert(x, y, z) }
  glVertex4f(x, y, z, w) { this._pushVert(x/w, y/w, z/w) }
  glVertex2i(x, y)       { this._pushVert(x, y, 0) }
  glVertex3i(x, y, z)    { this._pushVert(x, y, z) }
  glVertex2d(x, y)       { this._pushVert(x, y, 0) }
  glVertex3d(x, y, z)    { this._pushVert(x, y, z) }

  glVertex2fv(ptr) { const f = this._f32s(ptr, 2); this._pushVert(f[0], f[1], 0) }
  glVertex3fv(ptr) { const f = this._f32s(ptr, 3); this._pushVert(f[0], f[1], f[2]) }

  glColor3f(r, g, b)       { this._imColor = [r, g, b, 1] }
  glColor4f(r, g, b, a)    { this._imColor = [r, g, b, a] }
  glColor3d(r, g, b)       { this._imColor = [r, g, b, 1] }
  glColor4d(r, g, b, a)    { this._imColor = [r, g, b, a] }
  glColor3ub(r, g, b)      { this._imColor = [r/255, g/255, b/255, 1] }
  glColor4ub(r, g, b, a)   { this._imColor = [r/255, g/255, b/255, a/255] }
  glColor3i(r, g, b)       { this._imColor = [r/0x7fffffff, g/0x7fffffff, b/0x7fffffff, 1] }
  glColor4i(r, g, b, a)    { this._imColor = [r/0x7fffffff, g/0x7fffffff, b/0x7fffffff, a/0x7fffffff] }
  glColor3fv(ptr)          { const f = this._f32s(ptr, 3); this._imColor = [f[0], f[1], f[2], 1] }
  glColor4fv(ptr)          { const f = this._f32s(ptr, 4); this._imColor = [f[0], f[1], f[2], f[3]] }

  glNormal3f(x, y, z)      { this._imNormal = [x, y, z] }
  glNormal3d(x, y, z)      { this._imNormal = [x, y, z] }
  glNormal3fv(ptr)         { const f = this._f32s(ptr, 3); this._imNormal = [f[0], f[1], f[2]] }

  glTexCoord1f(s)          { this._imUV = [s, 0] }
  glTexCoord2f(s, t)       { this._imUV = [s, t] }
  glTexCoord3f(s, t, r)    { this._imUV = [s, t] }
  glTexCoord2fv(ptr)       { const f = this._f32s(ptr, 2); this._imUV = [f[0], f[1]] }

  glRectf(x1, y1, x2, y2) {
    this.glBegin(7 /*GL_QUADS*/)
    this._pushVert(x1, y1, 0); this._pushVert(x2, y1, 0)
    this._pushVert(x2, y2, 0); this._pushVert(x1, y2, 0)
    this.glEnd()
  }

  glCopyTexImage2D(target, level, internalFormat, x, y, w, h, border) {
    this._gl.copyTexImage2D(target, level, internalFormat, x, y, w, h, border)
  }
  glCopyTexSubImage2D(target, level, xoff, yoff, x, y, w, h) {
    this._gl.copyTexSubImage2D(target, level, xoff, yoff, x, y, w, h)
  }

/*

  // ── Legacy stubs (no-ops or trivial mappings) ─────────────────────────────────

  glEnableClientState()  {}
  glDisableClientState() {}
  glClientActiveTexture() {}
  glShadeModel()         {}
  glTexEnvi()            {}
  glTexEnvf()            {}
  glTexEnvfv()           {}
  glFogi()               {}
  glFogf()               {}
  glFogfv()              {}
  glFogiv()              {}
  glAlphaFunc()          {}  // emulate via discard in shader if needed
  glLightf()             {}
  glLightfv()            {}
  glLighti()             {}
  glLightiv()            {}
  glMaterialf()          {}
  glMaterialfv()         {}
  glMateriali()          {}
  glMaterialiv()         {}
  glPointSize()          {}
  glColorMaterial()      {}
  glLogicOp()            {}
  glPassThrough()        {}
  glSelectBuffer()       {}
  glFeedbackBuffer()     {}
  glRenderMode()         { return 0 }
  glInitNames()          {}
  glPushName()           {}
  glPopName()            {}
  glLoadName()           {}
  glPushAttrib()         {}
  glPopAttrib()          {}
  glPushClientAttrib()   {}
  glPopClientAttrib()    {}

  // Legacy array drawing (client-side arrays — use VBOs instead)
  glVertexPointer()  {}
  glColorPointer()   {}
  glNormalPointer()  {}
  glTexCoordPointer() {}
  glIndexPointer()   {}

  // ── Misc ─────────────────────────────────────────────────────────────────────

  glLineStipple()        {}
  glPolygonStipple()     {}
  glPolygonMode()        {}  // no equivalent in WebGL2


*/
}
