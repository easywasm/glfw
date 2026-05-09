# @easywasm/gl

The idea with this is easy GLFW/GL shim-layer for wasm.

It should make existing projects that use those libs easier to compile for wasm, without emscripten, but it's also an API you can link against in any wasm.

Think of it as a platform-layer. You have WASI, which covers a lot of "OS things" like filesystem, time, etc, and you have this covers all the basic stuff you need to make a game (window, graphics, input, sound.)

## basic usage

You can assume GLFW (and most of OpenGL) is exposed, and then you can use it normally. You can see an example in [index.html](./web/index.html). The code is in [example.c](./src/example.c), and can be compiled as native or wasm.

- `npm start` - compile example for web - requires wasi-sdk
- `npm run example:mac` - compiles same example for native mac. This is what I have for testing. Might add more build commands, later
