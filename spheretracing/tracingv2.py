import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np


camera_pos = np.array([0.0, 0.0, -5.0], dtype=np.float32)  # Camera position
camera_matrix=np.eye(3)#cam to world
camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)     # Up vector
camera_speed = 0.05  # Movement speed
yaw, pitch = 90., 0.0  # Camera angles
sensitivity = 0.1  # Mouse sensitivity


def getCam(forward):
    camF = normalize(forward)
    #camF = normalize(lookat - ro)  # forward
    camR = normalize(np.cross(camera_up, camF))  # right
    camU = np.cross(camF, camR)  # up
    
    # Construct the camera matrix (column-major)
    camera_matrix = np.array([camR, camU, camF]).T  # Transpose to match OpenGL column-major format
    return camera_matrix

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v



# Initialize GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

windowsize=(800, 600)
# Create a window
window = glfw.create_window(*windowsize, "Sphere Tracer", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

glfw.make_context_current(window)

# Define vertices for a fullscreen quad
vertices = np.array([
    -1.0, -1.0, 0.0,
     1.0, -1.0, 0.0,
     1.0,  1.0, 0.0,
    -1.0,  1.0, 0.0
], dtype=np.float32)#*np.array([])

indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

def readfile(path):
    with open(path) as file:
        return file.read()


# Compile shaders
vertex_src = readfile("./spheretracing/vertexshader.glsl")

fragment_src = readfile("./spheretracing/fragmentshader.glsl")

scene=readfile("./scene.glsl")
fragment_src=fragment_src.split("//cutoff")[0]+scene
print(fragment_src)

shader = compileProgram(
    compileShader(vertex_src, GL_VERTEX_SHADER),
    compileShader(fragment_src, GL_FRAGMENT_SHADER)
)

# Create VAO and VBO
vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)
ebo = glGenBuffers(1)

glBindVertexArray(vao)

glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)



def handle_keys(window):
    global camera_pos, camera_up, camera_speed
    camera_front=camera_matrix@np.array([0,0,1])
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:  # Move forward
        camera_pos += camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:  # Move backward
        camera_pos -= camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:  # Move left
        camera_pos += camera_matrix@np.array([-1,0,0]) * camera_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:  # Move right
        camera_pos += camera_matrix@np.array([1,0,0]) * camera_speed
last_x, last_y = 400, 300  # Center of the screen

mouse_pressed = False
def mouse_callback(window, xpos, ypos):
    global yaw, pitch, first_mouse, last_x, last_y


    x_offset = (xpos - last_x) * sensitivity
    y_offset = (last_y - ypos) * sensitivity  # Reversed: y-coordinates go from bottom to top
    last_x, last_y = xpos, ypos


    if not mouse_pressed:
        return

    yaw += x_offset
    pitch -= y_offset


    # Constrain pitch
    pitch = max(-89.0, min(89.0, pitch))

    # Update direction vector
    front = np.array([
        np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
        np.sin(np.radians(pitch)),
        np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    ], dtype=np.float32)
    global camera_matrix
    camera_matrix = getCam( front)
glfw.set_cursor_pos_callback(window, mouse_callback)


def mouse_button_callback(window, button, action, mods):
    global mouse_pressed
    if button == glfw.MOUSE_BUTTON_LEFT:  # Left mouse button
        if action == glfw.PRESS:
            mouse_pressed = True
        elif action == glfw.RELEASE:
            mouse_pressed = False
glfw.set_mouse_button_callback(window, mouse_button_callback)


glUseProgram(shader)
glUniform2f(glGetUniformLocation(shader, "windowsize"), *windowsize)
glUniform3f(glGetUniformLocation(shader, "lookat"), 0.0, 0.0, 0.0)
glUniform3f(glGetUniformLocation(shader, "lightPos"), 5.0, 5.0, -5.0)

# Render loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    handle_keys(window)  # Process keyboard input

    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(shader)

   
    #camera_matrix = getCam(camera_pos, camera_pos + camera_front)
    glUniform3f(glGetUniformLocation(shader, "cameraPos"), *camera_pos)
    #glUniform3f(glGetUniformLocation(shader, "lookat"), *lookat)
    glUniform1f(glGetUniformLocation(shader, "time"), glfw.get_time())
    glUniformMatrix3fv(glGetUniformLocation(shader, "cameraMatrix"), 1, GL_TRUE, camera_matrix)

    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

glfw.terminate()
