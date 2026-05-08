// Compile example.wasm first:
//   /opt/wasi-sdk/bin/clang -DGLFW_INCLUDE_NONE \
//     -Wl,--export=malloc -Wl,--allow-undefined -Wl,--import-memory \
//     -O3 ../example.c -o example.wasm


import { Glfw } from './glfw.js'

const status = document.getElementById('status')
const canvas = document.getElementById('canvas')

// Pre-create memory so it's available before _start runs.
// --import-memory compile flag makes wasm import this instead of creating its own.
const memory = new WebAssembly.Memory({ initial: 256, maximum: 16384 })

const glfw = new Glfw({ canvas, memory })

// ── Minimal wasi_snapshot_preview1 ──────────────────────────────────────────
// Only implements what a typical wasi-sdk GLFW program actually calls.
const dec = new TextDecoder()

const wasi_snapshot_preview1 = {
  args_sizes_get(argc_ptr, argv_buf_size_ptr) {
    const v = new DataView(memory.buffer)
    v.setInt32(argc_ptr, 1, true)
    v.setInt32(argv_buf_size_ptr, 8, true)
    return 0
  },
  args_get(argv_ptr, argv_buf_ptr) {
    const v = new DataView(memory.buffer)
    // argv[0] = "app\0"
    new Uint8Array(memory.buffer).set([97, 112, 112, 0], argv_buf_ptr)
    v.setInt32(argv_ptr, argv_buf_ptr, true)
    return 0
  },
  environ_sizes_get(count_ptr, buf_size_ptr) {
    const v = new DataView(memory.buffer)
    v.setInt32(count_ptr, 0, true)
    v.setInt32(buf_size_ptr, 0, true)
    return 0
  },
  environ_get() { return 0 },
  fd_write(fd, iovs_ptr, iovs_len, nwritten_ptr) {
    const v = new DataView(memory.buffer)
    const buf = new Uint8Array(memory.buffer)
    let written = 0
    for (let i = 0; i < iovs_len; i++) {
      const ptr = v.getInt32(iovs_ptr + i * 8, true)
      const len = v.getInt32(iovs_ptr + i * 8 + 4, true)
      const text = dec.decode(buf.subarray(ptr, ptr + len))
      if (fd === 1) console.log(text)
      else console.error(text)
      written += len
    }
    v.setInt32(nwritten_ptr, written, true)
    return 0
  },
  fd_read() { return 8 },   // EBADF
  fd_close() { return 0 },
  fd_seek() { return 70 },  // ESPIPE
  fd_fdstat_get(fd, stat_ptr) {
    // Minimal stat: filetype=2 (char device), flags=0, rights=all
    const v = new DataView(memory.buffer)
    v.setUint8(stat_ptr, 2)       // fs_filetype
    v.setUint8(stat_ptr + 1, 0)   // fs_flags
    v.setBigUint64(stat_ptr + 8,  0xFFFFFFFFFFFFFFFFn, true) // fs_rights_base
    v.setBigUint64(stat_ptr + 16, 0xFFFFFFFFFFFFFFFFn, true) // fs_rights_inheriting
    return 0
  },
  proc_exit(code) {
    // main() returned — normal for frame-based wasm apps
    if (code !== 0) console.error('[wasi] exit', code)
  },
  clock_time_get(id, precision, time_ptr) {
    const v = new DataView(memory.buffer)
    v.setBigUint64(time_ptr, BigInt(Math.floor(performance.now() * 1_000_000)), true)
    return 0
  },
  clock_res_get(id, res_ptr) {
    const v = new DataView(memory.buffer)
    v.setBigUint64(res_ptr, 1000n, true)
    return 0
  },
  sched_yield() { return 0 },
  random_get(buf_ptr, buf_len) {
    crypto.getRandomValues(new Uint8Array(memory.buffer, buf_ptr, buf_len))
    return 0
  },
  poll_oneoff() { return 52 },  // ENOSYS
}

// ── GL stubs — only what example.c uses ─────────────────────────────────────
// More GL functions live in a future @easywasm/gl package.
let gl = null

const glStubs = {
  glClearColor: (r, g, b, a) => gl?.clearColor(r, g, b, a),
  glClear:      (mask)       => gl?.clear(mask),
}

// ── Instantiate ──────────────────────────────────────────────────────────────
try {
  status.textContent = 'fetching wasm…'

  const { instance } = await WebAssembly.instantiateStreaming(fetch('example.wasm'), {
    // All GLFW functions + GL stubs + imported memory go in env.
    // wasi-sdk puts undefined symbols in "env" by default.
    env: { ...glfw, ...glStubs, memory },
    wasi_snapshot_preview1,
  })

  // Give glfw access to the wasm instance (function table for callbacks)
  glfw.setInstance(instance)

  // Run main() — creates the GLFW window + GL context, then returns
  instance.exports._start()

  // Grab the WebGL2 context that glfwCreateWindow created
  gl = glfw.getContextGL()
  if (!gl) throw new Error('WebGL2 not available')

  status.textContent = 'running — press Escape to close'

  // ── Frame loop ─────────────────────────────────────────────────────────────
  function loop() {
    if (instance.exports.frame()) {
      requestAnimationFrame(loop)
    } else {
      status.textContent = 'window closed'
    }
  }
  requestAnimationFrame(loop)

} catch (e) {
  status.textContent = 'error: ' + e.message
  console.error(e)
}
