// native mac: clang $(pkg-config --cflags --libs glfw3) -isysroot $(xcrun --show-sdk-path) -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo example.c -o example
// wasi-sdk:  /opt/wasi-sdk/bin/clang --target=wasm32-wasip1 -Wl,--export=malloc -Wl,--allow-undefined -Wl,--import-memory -O3 example.c -o demo/example.wasm

#ifdef __wasm__
#  define GLFW_INCLUDE_NONE
#endif
#include "glfw3.h"
#include <math.h>

#ifdef __wasm__
// GL functions imported from JS host (env namespace)
void glClearColor(float r, float g, float b, float a);
void glClear(unsigned int mask);
#  define GL_COLOR_BUFFER_BIT 0x4000
#elif defined(__APPLE__)
#  include <OpenGL/gl.h>
#else
#  include <GL/gl.h>
#endif

static GLFWwindow* window = NULL;

static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}

// On wasm: exported and called each tick by JS requestAnimationFrame.
// On native: called in the main loop below.
// Returns 1 to keep going, 0 to stop.
#ifdef __wasm__
__attribute__((export_name("frame")))
#endif
int frame(void) {
    if (!window || glfwWindowShouldClose(window)) return 0;

    float t = (float)glfwGetTime();
    glClearColor(
        0.5f + 0.5f * sinf(t),
        0.5f + 0.5f * sinf(t + 2.094f),
        0.5f + 0.5f * sinf(t + 4.189f),
        1.0f
    );
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);
    glfwPollEvents();
    return 1;
}

int main(void) {
    if (!glfwInit()) return -1;

    window = glfwCreateWindow(640, 480, "GLFW Demo", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

#ifndef __wasm__
    while (frame()) {}
    glfwTerminate();
#endif
    return 0;
}