// Web host implementation of the GLFW 3.5 API for WebAssembly modules compiled
// with wasi-sdk.  wasi-sdk puts undefined symbols in the "env" import namespace
// by default, so pass the Glfw instance as the env import:
//
//   const glfw = new Glfw({ canvas })
//   const wasi = new WasiPreview1()
//   const { instance } = await WebAssembly.instantiateStreaming(fetch('app.wasm'), {
//     env: glfw,
//     wasi_snapshot_preview1: wasi,
//   })
//   glfw.setInstance(instance)
//
// If _start runs during instantiation (standard C main) and any GLFW function
// called at startup needs memory access, pre-create a WebAssembly.Memory and
// pass it to both Glfw({ memory }) and env: { ...glfw, memory }:
//
//   const memory = new WebAssembly.Memory({ initial: 256 })
//   const glfw = new Glfw({ canvas, memory })
//   ...{ env: { ...glfw, memory }, ... }
//
// Blocking while(!glfwWindowShouldClose) loops need asyncify or JSPI.
// Alternative: export a frame() function and drive it from requestAnimationFrame.

const enc = new TextEncoder()
const dec = new TextDecoder()

// DOM event.code → GLFW key constant (US layout)
const KEY_MAP = {
  Space: 32, Quote: 39, Comma: 44, Minus: 45, Period: 46, Slash: 47,
  Digit0: 48, Digit1: 49, Digit2: 50, Digit3: 51, Digit4: 52,
  Digit5: 53, Digit6: 54, Digit7: 55, Digit8: 56, Digit9: 57,
  Semicolon: 59, Equal: 61,
  KeyA: 65, KeyB: 66, KeyC: 67, KeyD: 68, KeyE: 69, KeyF: 70,
  KeyG: 71, KeyH: 72, KeyI: 73, KeyJ: 74, KeyK: 75, KeyL: 76,
  KeyM: 77, KeyN: 78, KeyO: 79, KeyP: 80, KeyQ: 81, KeyR: 82,
  KeyS: 83, KeyT: 84, KeyU: 85, KeyV: 86, KeyW: 87, KeyX: 88,
  KeyY: 89, KeyZ: 90,
  BracketLeft: 91, Backslash: 92, BracketRight: 93, Backquote: 96,
  Escape: 256, Enter: 257, Tab: 258, Backspace: 259,
  Insert: 260, Delete: 261,
  ArrowRight: 262, ArrowLeft: 263, ArrowDown: 264, ArrowUp: 265,
  PageUp: 266, PageDown: 267, Home: 268, End: 269,
  CapsLock: 280, ScrollLock: 281, NumLock: 282, PrintScreen: 283, Pause: 284,
  F1: 290, F2: 291, F3: 292, F4: 293, F5: 294, F6: 295, F7: 296,
  F8: 297, F9: 298, F10: 299, F11: 300, F12: 301, F13: 302, F14: 303,
  F15: 304, F16: 305, F17: 306, F18: 307, F19: 308, F20: 309, F21: 310,
  F22: 311, F23: 312, F24: 313, F25: 314,
  Numpad0: 320, Numpad1: 321, Numpad2: 322, Numpad3: 323, Numpad4: 324,
  Numpad5: 325, Numpad6: 326, Numpad7: 327, Numpad8: 328, Numpad9: 329,
  NumpadDecimal: 330, NumpadDivide: 331, NumpadMultiply: 332,
  NumpadSubtract: 333, NumpadAdd: 334, NumpadEnter: 335, NumpadEqual: 336,
  ShiftLeft: 340, ControlLeft: 341, AltLeft: 342, MetaLeft: 343,
  ShiftRight: 344, ControlRight: 345, AltRight: 346, MetaRight: 347,
  ContextMenu: 348,
}

// GLFW key → printable name used by glfwGetKeyName
const KEY_NAMES = {
  32: 'SPACE', 39: "'", 44: ',', 45: '-', 46: '.', 47: '/',
  48: '0', 49: '1', 50: '2', 51: '3', 52: '4',
  53: '5', 54: '6', 55: '7', 56: '8', 57: '9',
  59: ';', 61: '=',
  65: 'A', 66: 'B', 67: 'C', 68: 'D', 69: 'E', 70: 'F',
  71: 'G', 72: 'H', 73: 'I', 74: 'J', 75: 'K', 76: 'L',
  77: 'M', 78: 'N', 79: 'O', 80: 'P', 81: 'Q', 82: 'R',
  83: 'S', 84: 'T', 85: 'U', 86: 'V', 87: 'W', 88: 'X',
  89: 'Y', 90: 'Z',
  91: '[', 92: '\\', 93: ']', 96: '`',
}

// DOM MouseEvent.button → GLFW mouse button (left=0, right=1, middle=2)
const BTN_MAP = [0, 2, 1]

function getMods(e) {
  return (e.shiftKey ? 0x0001 : 0) |
         (e.ctrlKey  ? 0x0002 : 0) |
         (e.altKey   ? 0x0004 : 0) |
         (e.metaKey  ? 0x0008 : 0) |
         (e.getModifierState?.('CapsLock') ? 0x0010 : 0) |
         (e.getModifierState?.('NumLock')  ? 0x0020 : 0)
}

// Standard cursor shape → CSS cursor value
const CURSOR_CSS = {
  0x00036001: 'default',
  0x00036002: 'text',
  0x00036003: 'crosshair',
  0x00036004: 'pointer',
  0x00036005: 'ew-resize',
  0x00036006: 'ns-resize',
  0x00036007: 'nwse-resize',
  0x00036008: 'nesw-resize',
  0x00036009: 'move',
  0x0003600A: 'not-allowed',
}

export class Glfw {
  constructor({ memory, canvas } = {}) {
    this._memory = memory || null
    this._canvas = canvas || null
    this._instance = null

    this._windows = new Map()   // handle (i32) → WindowState
    this._monitors = new Map()  // handle (i32) → MonitorState
    this._cursors = new Map()   // handle (i32) → { css }
    this._nextHandle = 1

    this._currentContext = 0
    this._hints = {}

    this._errorCb = 0
    this._monitorCb = 0
    this._joystickCb = 0

    this._timeBase = performance.now() / 1000
    this._timeOffset = 0

    // malloc'd string ptrs keyed by string value, never freed (static lifetime)
    this._strCache = new Map()

    // malloc'd struct ptrs for vidmode etc.
    this._vidmodePtr = 0
    this._monitorsPtr = 0

    // Bind all methods so wasm can call them as plain function references
    const proto = Object.getPrototypeOf(this)
    for (const key of Object.getOwnPropertyNames(proto)) {
      if (key === 'constructor') continue
      const desc = Object.getOwnPropertyDescriptor(proto, key)
      if (typeof desc.value === 'function') this[key] = desc.value.bind(this)
    }
  }

  // Returns the WebGL2 context for a window handle
  getGL(winId) {
    return this._windows.get(winId)?.gl ?? null
  }

  // Returns the WebGL2 context for whichever window is current context
  getContextGL() {
    return this._windows.get(this._currentContext)?.gl ?? null
  }

  // Call after instantiation with the wasm exports object (mirrors WasiPreview1.start).
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

  // ── Memory helpers ──────────────────────────────────────────────────────────

  get _view() { return new DataView(this._memory.buffer) }

  _readStr(ptr) {
    if (!ptr) return ''
    const buf = new Uint8Array(this._memory.buffer)
    let end = ptr
    while (buf[end]) end++
    return dec.decode(buf.subarray(ptr, end))
  }

  _wi32(ptr, v) { if (ptr) this._view.setInt32(ptr, v | 0, true) }
  _wf32(ptr, v) { if (ptr) this._view.setFloat32(ptr, v, true) }
  _wf64(ptr, v) { if (ptr) this._view.setFloat64(ptr, v, true) }

  _malloc(size) {
    const fn = this._instance?.exports?.malloc
    if (!fn) {
      console.warn('[glfw] malloc not exported — add -Wl,--export=malloc to link flags')
      return 0
    }
    return fn(size)
  }

  _allocStr(str) {
    if (!str) return 0
    if (this._strCache.has(str)) return this._strCache.get(str)
    const bytes = enc.encode(str + '\0')
    const ptr = this._malloc(bytes.length)
    if (!ptr) return 0
    new Uint8Array(this._memory.buffer, ptr, bytes.length).set(bytes)
    this._strCache.set(str, ptr)
    return ptr
  }

  // Call a wasm function pointer via the indirect function table
  _callFn(fnPtr, ...args) {
    if (!fnPtr) return
    this._instance?.exports?.__indirect_function_table?.get(fnPtr)?.(...args)
  }

  // ── Init ────────────────────────────────────────────────────────────────────

  glfwInit() {
    const id = this._nextHandle++
    this._monitors.set(id, {
      id,
      name: 'Web Display',
      x: 0, y: 0,
      width: screen.width,
      height: screen.height,
      workX: 0, workY: 0,
      workWidth: screen.availWidth,
      workHeight: screen.availHeight,
      widthMM: Math.round(screen.width / 96 * 25.4),
      heightMM: Math.round(screen.height / 96 * 25.4),
      userPointer: 0,
    })
    this._primaryMonitor = id
    return 1  // GLFW_TRUE
  }

  glfwTerminate() {
    for (const [id] of this._windows) this.glfwDestroyWindow(id)
    this._windows.clear()
    this._monitors.clear()
    this._cursors.clear()
    this._currentContext = 0
  }

  glfwInitHint() {}
  glfwInitAllocator() {}
  glfwInitVulkanLoader() {}

  glfwGetVersion(majPtr, minPtr, revPtr) {
    this._wi32(majPtr, 3)
    this._wi32(minPtr, 5)
    this._wi32(revPtr, 0)
  }

  glfwGetVersionString() { return this._allocStr('3.5.0 WASM') }

  glfwGetError(descPtrPtr) {
    this._wi32(descPtrPtr, 0)
    return 0  // GLFW_NO_ERROR
  }

  glfwSetErrorCallback(cb) {
    const prev = this._errorCb
    this._errorCb = cb
    return prev
  }

  glfwGetPlatform() { return 0x00060005 }  // GLFW_PLATFORM_NULL
  glfwPlatformSupported(p) { return p === 0x00060005 ? 1 : 0 }

  // ── Monitors ────────────────────────────────────────────────────────────────

  glfwGetMonitors(countPtr) {
    const count = this._monitors.size
    this._wi32(countPtr, count)
    if (!count) return 0
    // Allocate / refresh array of i32 handles
    if (!this._monitorsPtr) this._monitorsPtr = this._malloc(count * 4)
    if (!this._monitorsPtr) return 0
    let i = 0
    for (const id of this._monitors.keys()) {
      this._view.setInt32(this._monitorsPtr + i * 4, id, true)
      i++
    }
    return this._monitorsPtr
  }

  glfwGetPrimaryMonitor() { return this._primaryMonitor || 0 }

  glfwGetMonitorPos(monId, xPtr, yPtr) {
    const m = this._monitors.get(monId)
    if (!m) return
    this._wi32(xPtr, m.x)
    this._wi32(yPtr, m.y)
  }

  glfwGetMonitorWorkarea(monId, xPtr, yPtr, wPtr, hPtr) {
    const m = this._monitors.get(monId)
    if (!m) return
    this._wi32(xPtr, m.workX); this._wi32(yPtr, m.workY)
    this._wi32(wPtr, m.workWidth); this._wi32(hPtr, m.workHeight)
  }

  glfwGetMonitorPhysicalSize(monId, wmmPtr, hmmPtr) {
    const m = this._monitors.get(monId)
    if (!m) return
    this._wi32(wmmPtr, m.widthMM)
    this._wi32(hmmPtr, m.heightMM)
  }

  glfwGetMonitorContentScale(monId, xsPtr, ysPtr) {
    const dpr = window.devicePixelRatio || 1
    this._wf32(xsPtr, dpr)
    this._wf32(ysPtr, dpr)
  }

  glfwGetMonitorName(monId) {
    return this._allocStr(this._monitors.get(monId)?.name ?? '')
  }

  glfwSetMonitorUserPointer(monId, ptr) {
    const m = this._monitors.get(monId)
    if (m) m.userPointer = ptr
  }

  glfwGetMonitorUserPointer(monId) {
    return this._monitors.get(monId)?.userPointer ?? 0
  }

  glfwSetMonitorCallback(cb) {
    const prev = this._monitorCb
    this._monitorCb = cb
    return prev
  }

  glfwGetVideoModes(monId, countPtr) {
    this._wi32(countPtr, 1)
    return this.glfwGetVideoMode(monId)
  }

  glfwGetVideoMode(monId) {
    // GLFWvidmode: int width, height, redBits, greenBits, blueBits, refreshRate (24 bytes)
    if (!this._vidmodePtr) {
      this._vidmodePtr = this._malloc(24)
      if (this._vidmodePtr) {
        const v = this._view
        v.setInt32(this._vidmodePtr +  0, screen.width, true)
        v.setInt32(this._vidmodePtr +  4, screen.height, true)
        v.setInt32(this._vidmodePtr +  8, 8, true)
        v.setInt32(this._vidmodePtr + 12, 8, true)
        v.setInt32(this._vidmodePtr + 16, 8, true)
        v.setInt32(this._vidmodePtr + 20, 60, true)
      }
    }
    return this._vidmodePtr
  }

  glfwSetGamma() {}
  glfwGetGammaRamp() { return 0 }
  glfwSetGammaRamp() {}

  // ── Window hints ─────────────────────────────────────────────────────────────

  glfwDefaultWindowHints() { this._hints = {} }
  glfwWindowHint(hint, value) { this._hints[hint] = value }
  glfwWindowHintString(hint, valuePtr) { this._hints[hint] = this._readStr(valuePtr) }

  // ── Windows ──────────────────────────────────────────────────────────────────

  glfwCreateWindow(width, height, titlePtr, monitorPtr, sharePtr) {
    const title = this._readStr(titlePtr)

    let canvas = this._canvas
    if (!canvas) {
      canvas = document.createElement('canvas')
      canvas.style.display = 'block'
      document.body.appendChild(canvas)
    }
    canvas.width = width
    canvas.height = height
    canvas.title = title

    const noApi = this._hints[0x00022001] === 0  // GLFW_CLIENT_API = GLFW_NO_API
    const gl = noApi ? null : canvas.getContext('webgl2', {
      antialias: (this._hints[0x0002100D] || 0) > 0,  // GLFW_SAMPLES
      depth:     (this._hints[0x00021005] ?? 24) > 0,  // GLFW_DEPTH_BITS
      stencil:   (this._hints[0x00021006] || 0) > 0,  // GLFW_STENCIL_BITS
      alpha: true,
      premultipliedAlpha: false,
      powerPreference: 'high-performance',
    })

    if (gl === null && !noApi) {
      console.error('[glfw] WebGL2 context creation failed')
      return 0
    }

    const id = this._nextHandle++
    const win = {
      id, canvas, gl, width, height, x: 0, y: 0, title,
      shouldClose: false, userPointer: 0,
      visible: this._hints[0x00020004] !== 0,  // GLFW_VISIBLE
      focused: true, iconified: false, maximized: false,
      cb: {},
      keys: new Map(),    // GLFW key → 0/1/2 (release/press/repeat)
      buttons: new Map(), // GLFW button → 0/1
      cx: 0, cy: 0,
      inputMode: {
        cursor: 0x00034001,  // GLFW_CURSOR_NORMAL
        stickyKeys: 0, stickyMouseButtons: 0,
        lockKeyMods: 0, rawMouseMotion: 0,
      },
      cleanups: [],
    }
    this._windows.set(id, win)
    this._attachEvents(win)
    return id
  }

  glfwDestroyWindow(winId) {
    const win = this._windows.get(winId)
    if (!win) return
    for (const fn of win.cleanups) fn()
    this._windows.delete(winId)
    if (this._currentContext === winId) this._currentContext = 0
  }

  glfwWindowShouldClose(winId) {
    return this._windows.get(winId)?.shouldClose ? 1 : 0
  }

  glfwSetWindowShouldClose(winId, value) {
    const win = this._windows.get(winId)
    if (win) win.shouldClose = value !== 0
  }

  glfwGetWindowTitle(winId) {
    return this._allocStr(this._windows.get(winId)?.title ?? '')
  }

  glfwSetWindowTitle(winId, titlePtr) {
    const win = this._windows.get(winId)
    if (!win) return
    win.title = this._readStr(titlePtr)
    win.canvas.title = win.title
  }

  glfwSetWindowIcon() {}

  glfwGetWindowPos(winId, xPtr, yPtr) {
    const win = this._windows.get(winId)
    if (!win) return
    const r = win.canvas.getBoundingClientRect()
    this._wi32(xPtr, Math.round(r.left))
    this._wi32(yPtr, Math.round(r.top))
  }

  glfwSetWindowPos(winId, x, y) {
    const win = this._windows.get(winId)
    if (!win) return
    win.canvas.style.position = 'absolute'
    win.canvas.style.left = x + 'px'
    win.canvas.style.top = y + 'px'
    win.x = x; win.y = y
  }

  glfwGetWindowSize(winId, wPtr, hPtr) {
    const win = this._windows.get(winId)
    if (!win) return
    this._wi32(wPtr, win.width)
    this._wi32(hPtr, win.height)
  }

  glfwSetWindowSizeLimits() {}
  glfwSetWindowAspectRatio() {}

  glfwSetWindowSize(winId, w, h) {
    const win = this._windows.get(winId)
    if (!win) return
    win.canvas.width = w; win.canvas.height = h
    win.width = w; win.height = h
    this._callFn(win.cb.windowSize, winId, w, h)
    const dpr = window.devicePixelRatio || 1
    this._callFn(win.cb.framebufferSize, winId,
      Math.round(w * dpr), Math.round(h * dpr))
  }

  glfwGetFramebufferSize(winId, wPtr, hPtr) {
    const win = this._windows.get(winId)
    if (!win) return
    const dpr = window.devicePixelRatio || 1
    this._wi32(wPtr, Math.round(win.width * dpr))
    this._wi32(hPtr, Math.round(win.height * dpr))
  }

  glfwGetWindowFrameSize(winId, lPtr, tPtr, rPtr, bPtr) {
    this._wi32(lPtr, 0); this._wi32(tPtr, 0)
    this._wi32(rPtr, 0); this._wi32(bPtr, 0)
  }

  glfwGetWindowContentScale(winId, xsPtr, ysPtr) {
    const dpr = window.devicePixelRatio || 1
    this._wf32(xsPtr, dpr)
    this._wf32(ysPtr, dpr)
  }

  glfwGetWindowOpacity() { return 1.0 }

  glfwSetWindowOpacity(winId, opacity) {
    const win = this._windows.get(winId)
    if (win) win.canvas.style.opacity = opacity
  }

  glfwIconifyWindow() {}
  glfwRestoreWindow() {}
  glfwMaximizeWindow() {}

  glfwShowWindow(winId) {
    const win = this._windows.get(winId)
    if (win) { win.canvas.style.display = 'block'; win.visible = true }
  }

  glfwHideWindow(winId) {
    const win = this._windows.get(winId)
    if (win) { win.canvas.style.display = 'none'; win.visible = false }
  }

  glfwFocusWindow(winId) {
    this._windows.get(winId)?.canvas.focus()
  }

  glfwRequestWindowAttention() {}

  glfwGetWindowMonitor() { return 0 }

  glfwSetWindowMonitor(winId, monId, x, y, w, h, refreshRate) {
    const win = this._windows.get(winId)
    if (!win) return
    if (monId) {
      win.canvas.style.width = '100vw'
      win.canvas.style.height = '100vh'
      win.canvas.style.position = 'fixed'
      win.canvas.style.left = '0'
      win.canvas.style.top = '0'
    } else {
      win.canvas.style.width = ''
      win.canvas.style.height = ''
      win.canvas.style.position = ''
      this.glfwSetWindowPos(winId, x, y)
      this.glfwSetWindowSize(winId, w, h)
    }
  }

  glfwGetWindowAttrib(winId, attrib) {
    const win = this._windows.get(winId)
    if (!win) return 0
    switch (attrib) {
      case 0x00020001: return win.focused ? 1 : 0
      case 0x00020002: return win.iconified ? 1 : 0
      case 0x00020003: return 1  // resizable
      case 0x00020004: return win.visible ? 1 : 0
      case 0x00020005: return 1  // decorated
      case 0x00020008: return win.maximized ? 1 : 0
      case 0x0002000B: return win.focused ? 1 : 0  // hovered
      case 0x00022001: return win.gl ? 0x00030001 : 0  // CLIENT_API
      case 0x00022002: return 4  // CONTEXT_VERSION_MAJOR (WebGL2 ≈ GL 4.x)
      case 0x00022003: return 6  // CONTEXT_VERSION_MINOR
      default: return 0
    }
  }

  glfwSetWindowAttrib() {}

  glfwSetWindowUserPointer(winId, ptr) {
    const win = this._windows.get(winId)
    if (win) win.userPointer = ptr
  }

  glfwGetWindowUserPointer(winId) {
    return this._windows.get(winId)?.userPointer ?? 0
  }

  // Window callbacks
  glfwSetWindowPosCallback(w, cb)          { return this._setCb(w, 'windowPos', cb) }
  glfwSetWindowSizeCallback(w, cb)         { return this._setCb(w, 'windowSize', cb) }
  glfwSetWindowCloseCallback(w, cb)        { return this._setCb(w, 'windowClose', cb) }
  glfwSetWindowRefreshCallback(w, cb)      { return this._setCb(w, 'windowRefresh', cb) }
  glfwSetWindowFocusCallback(w, cb)        { return this._setCb(w, 'windowFocus', cb) }
  glfwSetWindowIconifyCallback(w, cb)      { return this._setCb(w, 'windowIconify', cb) }
  glfwSetWindowMaximizeCallback(w, cb)     { return this._setCb(w, 'windowMaximize', cb) }
  glfwSetFramebufferSizeCallback(w, cb)    { return this._setCb(w, 'framebufferSize', cb) }
  glfwSetWindowContentScaleCallback(w, cb) { return this._setCb(w, 'contentScale', cb) }

  _setCb(winId, name, cb) {
    const win = this._windows.get(winId)
    if (!win) return 0
    const prev = win.cb[name] ?? 0
    win.cb[name] = cb
    return prev
  }

  // ── Events ───────────────────────────────────────────────────────────────────

  glfwPollEvents() {
    // DOM events arrive asynchronously via listeners; key/button state is kept
    // current.  In an asyncify/JSPI setup, yielding here lets the event loop
    // flush.  In a synchronous loop this is effectively a no-op.
  }

  glfwWaitEvents() {}
  glfwWaitEventsTimeout() {}
  glfwPostEmptyEvent() {}

  // ── Input ────────────────────────────────────────────────────────────────────

  glfwGetInputMode(winId, mode) {
    const win = this._windows.get(winId)
    if (!win) return 0
    switch (mode) {
      case 0x00033001: return win.inputMode.cursor
      case 0x00033002: return win.inputMode.stickyKeys
      case 0x00033003: return win.inputMode.stickyMouseButtons
      case 0x00033004: return win.inputMode.lockKeyMods
      case 0x00033005: return win.inputMode.rawMouseMotion
      default: return 0
    }
  }

  glfwSetInputMode(winId, mode, value) {
    const win = this._windows.get(winId)
    if (!win) return
    switch (mode) {
      case 0x00033001:
        win.inputMode.cursor = value
        if (value === 0x00034002) {  // HIDDEN
          win.canvas.style.cursor = 'none'
        } else if (value === 0x00034003) {  // DISABLED
          win.canvas.requestPointerLock?.()
          win.canvas.style.cursor = 'none'
        } else {
          win.canvas.style.cursor = ''
          if (document.pointerLockElement === win.canvas) document.exitPointerLock?.()
        }
        break
      case 0x00033002: win.inputMode.stickyKeys = value; break
      case 0x00033003: win.inputMode.stickyMouseButtons = value; break
      case 0x00033004: win.inputMode.lockKeyMods = value; break
      case 0x00033005: win.inputMode.rawMouseMotion = value; break
    }
  }

  glfwRawMouseMotionSupported() { return 0 }

  glfwGetKeyName(key, scancode) { return this._allocStr(KEY_NAMES[key] ?? '') }
  glfwGetKeyScancode(key) { return key }

  glfwGetKey(winId, key) {
    const win = this._windows.get(winId)
    if (!win) return 0
    const state = win.keys.get(key) ?? 0
    if (win.inputMode.stickyKeys && state) { win.keys.set(key, 0); return 1 }
    return state
  }

  glfwGetMouseButton(winId, button) {
    const win = this._windows.get(winId)
    if (!win) return 0
    const state = win.buttons.get(button) ?? 0
    if (win.inputMode.stickyMouseButtons && state) { win.buttons.set(button, 0); return 1 }
    return state
  }

  glfwGetCursorPos(winId, xPtr, yPtr) {
    const win = this._windows.get(winId)
    if (!win) return
    this._wf64(xPtr, win.cx)
    this._wf64(yPtr, win.cy)
  }

  glfwSetCursorPos(winId, x, y) {
    const win = this._windows.get(winId)
    if (win) { win.cx = x; win.cy = y }
  }

  // ── Cursors ──────────────────────────────────────────────────────────────────

  glfwCreateCursor(imagePtr, xhot, yhot) {
    const id = this._nextHandle++
    this._cursors.set(id, { css: 'default' })
    return id
  }

  glfwCreateStandardCursor(shape) {
    const id = this._nextHandle++
    this._cursors.set(id, { css: CURSOR_CSS[shape] ?? 'default' })
    return id
  }

  glfwDestroyCursor(curId) { this._cursors.delete(curId) }

  glfwSetCursor(winId, curId) {
    const win = this._windows.get(winId)
    if (!win) return
    win.canvas.style.cursor = this._cursors.get(curId)?.css ?? ''
  }

  // Input callbacks
  glfwSetKeyCallback(w, cb)         { return this._setCb(w, 'key', cb) }
  glfwSetCharCallback(w, cb)        { return this._setCb(w, 'char', cb) }
  glfwSetCharModsCallback(w, cb)    { return this._setCb(w, 'charMods', cb) }
  glfwSetMouseButtonCallback(w, cb) { return this._setCb(w, 'mouseButton', cb) }
  glfwSetCursorPosCallback(w, cb)   { return this._setCb(w, 'cursorPos', cb) }
  glfwSetCursorEnterCallback(w, cb) { return this._setCb(w, 'cursorEnter', cb) }
  glfwSetScrollCallback(w, cb)      { return this._setCb(w, 'scroll', cb) }
  glfwSetDropCallback(w, cb)        { return this._setCb(w, 'drop', cb) }

  // ── Joystick / Gamepad ───────────────────────────────────────────────────────

  glfwJoystickPresent(jid) {
    return (navigator.getGamepads?.()[jid]) ? 1 : 0
  }

  glfwGetJoystickAxes(jid, countPtr) {
    const pad = navigator.getGamepads?.()[jid]
    this._wi32(countPtr, pad ? pad.axes.length : 0)
    if (!pad || !pad.axes.length) return 0
    const ptr = this._malloc(pad.axes.length * 4)
    if (!ptr) return 0
    for (let i = 0; i < pad.axes.length; i++) {
      this._view.setFloat32(ptr + i * 4, pad.axes[i], true)
    }
    return ptr
  }

  glfwGetJoystickButtons(jid, countPtr) {
    const pad = navigator.getGamepads?.()[jid]
    this._wi32(countPtr, pad ? pad.buttons.length : 0)
    if (!pad || !pad.buttons.length) return 0
    const ptr = this._malloc(pad.buttons.length)
    if (!ptr) return 0
    const buf = new Uint8Array(this._memory.buffer, ptr, pad.buttons.length)
    for (let i = 0; i < pad.buttons.length; i++) buf[i] = pad.buttons[i].pressed ? 1 : 0
    return ptr
  }

  glfwGetJoystickHats(jid, countPtr) { this._wi32(countPtr, 0); return 0 }

  glfwGetJoystickName(jid) {
    return this._allocStr(navigator.getGamepads?.()[jid]?.id ?? '')
  }

  glfwGetJoystickGUID() { return 0 }
  glfwSetJoystickUserPointer() {}
  glfwGetJoystickUserPointer() { return 0 }
  glfwJoystickIsGamepad(jid) { return this.glfwJoystickPresent(jid) }

  glfwSetJoystickCallback(cb) {
    const prev = this._joystickCb
    this._joystickCb = cb
    return prev
  }

  glfwUpdateGamepadMappings() { return 1 }

  glfwGetGamepadName(jid) { return this.glfwGetJoystickName(jid) }

  glfwGetGamepadState(jid, statePtr) {
    // GLFWgamepadstate: unsigned char buttons[15] + pad + float axes[6] = 40 bytes
    const pad = navigator.getGamepads?.()[jid]
    if (!pad || !statePtr) return 0
    const mem = new Uint8Array(this._memory.buffer)
    for (let i = 0; i < 15; i++) {
      mem[statePtr + i] = pad.buttons[i]?.pressed ? 1 : 0
    }
    for (let i = 0; i < 6; i++) {
      this._view.setFloat32(statePtr + 16 + i * 4, pad.axes[i] ?? 0, true)
    }
    return 1
  }

  // ── Clipboard ────────────────────────────────────────────────────────────────

  glfwSetClipboardString(winId, strPtr) {
    navigator.clipboard?.writeText(this._readStr(strPtr))
  }

  glfwGetClipboardString() {
    // Clipboard API is async; synchronous access not possible in web.
    return 0
  }

  // ── Time ─────────────────────────────────────────────────────────────────────

  glfwGetTime() {
    return this._timeOffset + (performance.now() / 1000 - this._timeBase)
  }

  glfwSetTime(t) {
    this._timeOffset = t
    this._timeBase = performance.now() / 1000
  }

  // Returns i64 (BigInt in JS ↔ wasm i64 boundary)
  glfwGetTimerValue() {
    return BigInt(Math.floor(performance.now() * 1000))
  }

  glfwGetTimerFrequency() {
    return BigInt(1_000_000)
  }

  // ── Context ──────────────────────────────────────────────────────────────────

  glfwMakeContextCurrent(winId) { this._currentContext = winId }
  glfwGetCurrentContext() { return this._currentContext }

  // WebGL auto-presents; swap and interval are no-ops
  glfwSwapBuffers() {}
  glfwSwapInterval() {}

  glfwExtensionSupported() { return 0 }

  // WebGL functions cannot be returned as raw function pointers to wasm.
  // Use a dedicated WebGL import namespace (e.g. @easywasm/gl) for GL calls.
  glfwGetProcAddress() { return 0 }

  // ── Vulkan (not supported in web) ────────────────────────────────────────────

  glfwVulkanSupported() { return 0 }
  glfwGetRequiredInstanceExtensions(countPtr) { this._wi32(countPtr, 0); return 0 }
  glfwGetInstanceProcAddress() { return 0 }
  glfwGetPhysicalDevicePresentationSupport() { return 0 }
  glfwCreateWindowSurface() { return -1 }  // VK_ERROR_EXTENSION_NOT_PRESENT

  // ── Private: DOM event wiring ────────────────────────────────────────────────

  _on(win, target, type, fn, opts) {
    target.addEventListener(type, fn, opts)
    win.cleanups.push(() => target.removeEventListener(type, fn))
  }

  _attachEvents(win) {
    const { canvas } = win
    canvas.setAttribute('tabindex', '0')

    this._on(win, canvas, 'keydown', e => {
      const key = KEY_MAP[e.code] ?? -1
      const mods = getMods(e)
      const action = e.repeat ? 2 : 1  // REPEAT : PRESS
      win.keys.set(key, action)
      this._callFn(win.cb.key, win.id, key, key, action, mods)
      if (e.key.length === 1) {
        const cp = e.key.codePointAt(0)
        this._callFn(win.cb.char, win.id, cp)
        this._callFn(win.cb.charMods, win.id, cp, mods)
      }
      // Only suppress default for keys the app likely handles
      if (!['F5', 'F12', 'Tab'].includes(e.key)) e.preventDefault()
    })

    this._on(win, canvas, 'keyup', e => {
      const key = KEY_MAP[e.code] ?? -1
      win.keys.set(key, 0)
      this._callFn(win.cb.key, win.id, key, key, 0, getMods(e))
    })

    this._on(win, canvas, 'mousedown', e => {
      canvas.focus()
      const btn = BTN_MAP[e.button] ?? e.button
      win.buttons.set(btn, 1)
      this._callFn(win.cb.mouseButton, win.id, btn, 1, getMods(e))
      e.preventDefault()
    })

    this._on(win, canvas, 'mouseup', e => {
      const btn = BTN_MAP[e.button] ?? e.button
      win.buttons.set(btn, 0)
      this._callFn(win.cb.mouseButton, win.id, btn, 0, getMods(e))
    })

    this._on(win, canvas, 'mousemove', e => {
      const r = canvas.getBoundingClientRect()
      win.cx = e.clientX - r.left
      win.cy = e.clientY - r.top
      this._callFn(win.cb.cursorPos, win.id, win.cx, win.cy)
    })

    this._on(win, canvas, 'mouseenter', () => {
      this._callFn(win.cb.cursorEnter, win.id, 1)
    })

    this._on(win, canvas, 'mouseleave', () => {
      this._callFn(win.cb.cursorEnter, win.id, 0)
    })

    this._on(win, canvas, 'wheel', e => {
      // Normalize to "lines" — deltaMode 0=pixel, 1=line, 2=page
      const scale = e.deltaMode === 0 ? 1 / 100 : e.deltaMode === 2 ? 10 : 1
      this._callFn(win.cb.scroll, win.id, -e.deltaX * scale, -e.deltaY * scale)
      e.preventDefault()
    }, { passive: false })

    this._on(win, canvas, 'contextmenu', e => e.preventDefault())

    this._on(win, canvas, 'dragover', e => e.preventDefault())

    this._on(win, canvas, 'drop', e => {
      e.preventDefault()
      // TODO: allocate path strings and call drop callback
    })

    this._on(win, canvas, 'focus', () => {
      win.focused = true
      this._callFn(win.cb.windowFocus, win.id, 1)
    })

    this._on(win, canvas, 'blur', () => {
      win.focused = false
      // Clear all held keys on focus loss to avoid stuck-key issues
      win.keys.clear()
      win.buttons.clear()
      this._callFn(win.cb.windowFocus, win.id, 0)
    })

    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const w = Math.round(entry.contentRect.width)
        const h = Math.round(entry.contentRect.height)
        if (w === win.width && h === win.height) continue
        win.width = w; win.height = h
        win.canvas.width = w; win.canvas.height = h
        this._callFn(win.cb.windowSize, win.id, w, h)
        const dpr = window.devicePixelRatio || 1
        this._callFn(win.cb.framebufferSize, win.id, Math.round(w * dpr), Math.round(h * dpr))
      }
    })
    ro.observe(canvas)
    win.cleanups.push(() => ro.disconnect())

    const onUnload = () => {
      win.shouldClose = true
      this._callFn(win.cb.windowClose, win.id)
    }
    window.addEventListener('beforeunload', onUnload)
    win.cleanups.push(() => window.removeEventListener('beforeunload', onUnload))
  }
}
