// this is an exmaple of wasm/native code

#include <math.h>
#include "platform.h"

static GLFWwindow* window = NULL;

static void key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}

// return "keep going"
int update() {
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

// return exit-status
int setup() {
    if (!glfwInit()) return -1;

    window = glfwCreateWindow(640, 480, "GLFW Demo", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    return 0;
}
