// this sets up GLFW/GL for native or wasm

#include "glfw3.h"

#ifdef __wasm__
    #include "gl.h"
#elif defined(__APPLE__)
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
#endif

int update(void);
int setup();

int main(void) {
    int ok = setup();
    if (ok != 0) {
        return ok;
    } 

#ifndef __wasm__
    while (update()) {}
    glfwTerminate();
#endif

    return 0;
}