import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import time
import sys
import glslprog
import opgraphtofun
sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
server=Server()

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

def updatecammat():
    global yaw,pitch
    pitch = max(-89.0, min(89.0, pitch))

    # Update direction vector
    front = np.array([
        np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
        np.sin(np.radians(pitch)),
        np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    ], dtype=np.float32)
    global camera_matrix
    camera_matrix = getCam( front)

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
vertex_src = readfile("./spheretracing2/vertexshader.glsl")

def funtovisualize(x,y,z):


    t=dcga.Translator(3,4,5)
    #print((t*t.reverse()).blades)
    def sanwich(V,m):
        return V*m*dcga.inverse(V)

    #xyz=(opg.VarNode("x"),opg.VarNode("y"),opg.VarNode("z"))
    point=dcga.point(x,y,z)
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,1)

    #obj=sanwich(t,obj)
    iprod=point.inner(obj)
    return list(iprod.blades.values())
#import opgraphtofuncasadi
scene,_=opgraphtofun.makefuntailor(funtovisualize)
#scene=opgraphtofuncasadi.generate_glsl_code(funtovisualize)
prog=glslprog.glslprogramm(version="440")
prog.parts.append(glslprog.glslprogrammpart(bodypath="./glslcompile/fragmentshader.glsl"))
prog.parts.append(scene)


fragment_src = str(prog)
#print(fragment_src)
with open("./glslcompile/lastfragment.glsl","w")as f:
    f.write(fragment_src)

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
    global camera_pos, camera_speed,yaw
    camera_front=camera_matrix@np.array([0,0,1])

    camera_speedlocal=camera_speed
    rotspeed=0.5
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
        camera_speedlocal*=4
        rotspeed*=4
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
        camera_speedlocal*=0.01
        rotspeed*=0.01

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:  # Move forward
        camera_pos += camera_speedlocal * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:  # Move backward
        camera_pos -= camera_speedlocal * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:  # Move left
        camera_pos += camera_matrix@np.array([-1,0,0]) * camera_speedlocal
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:  # Move right
        camera_pos += camera_matrix@np.array([1,0,0]) * camera_speedlocal
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:  # Move right
        yaw += rotspeed
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:  # Move right
        yaw -= rotspeed
    updatecammat()
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


    updatecammat()
    
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

raydyinit=1000
rayfactor=1
rayalpha=0.5
rayepsilon=0.00001
raypow=1

varupdates=server.variableupdates

while not glfw.window_should_close(window):
    cammatold,camposold=camera_matrix.copy(),camera_pos.copy()
    glfw.poll_events()
    handle_keys(window)  # Process keyboard input
    if np.array_equal(cammatold,camera_matrix) and np.array_equal(camposold,camera_pos) and varupdates==server.variableupdates:
        time.sleep(0.1)
        continue
    varupdates=server.variableupdates

    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader)

    glUniform1f(glGetUniformLocation(shader, "raydyinit"),raydyinit )
    glUniform1f(glGetUniformLocation(shader, "rayfactor"),rayfactor )
    glUniform1f(glGetUniformLocation(shader, "rayalpha"),rayalpha )
    glUniform1f(glGetUniformLocation(shader, "rayepsilon"),rayepsilon )
    glUniform1f(glGetUniformLocation(shader, "raypow"),raypow )


   
    #camera_matrix = getCam(camera_pos, camera_pos + camera_front)
    glUniform3f(glGetUniformLocation(shader, "cameraPos"), *camera_pos)
    #glUniform3f(glGetUniformLocation(shader, "lookat"), *lookat)
    glUniform1f(glGetUniformLocation(shader, "time"), glfw.get_time())
    glUniformMatrix3fv(glGetUniformLocation(shader, "cameraMatrix"), 1, GL_TRUE, camera_matrix)


    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glfw.swap_buffers(window)

glfw.terminate()
