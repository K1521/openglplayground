import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import glfw
import pyvista as pv
import time

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW cannot be initialized")
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

# Create a windowed mode window and OpenGL context
window = glfw.create_window(800, 600, "OpenGL Compute Shader", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created")

# Make the OpenGL context current
glfw.make_context_current(window)
print("OpenGL version:", gl.glGetString(gl.GL_VERSION))

# Set up compute shader source code
grid_size = (5, 1, 1)
num_elements = grid_size[0] * grid_size[1] * grid_size[2]

pointsshader = """
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, offset = 0) uniform atomic_uint appendCounter;
layout(std430, binding = 1) buffer AppendBuffer {
    vec3 vert[];
};

void main() {
    ivec3 p = ivec3(gl_GlobalInvocationID.xyz);  // Corrected
    int idx = int(atomicCounterIncrement(appendCounter));
    vert[idx] = vec3(p);
}
"""

try:
    compute_shader = shaders.compileShader(pointsshader, gl.GL_COMPUTE_SHADER)
    shader_program = shaders.compileProgram(compute_shader)
except Exception as e:
    print(str(e).replace("\\\\n", "\n").replace("\\n", "\n"))
    glfw.terminate()
    exit()

gl.glUseProgram(shader_program)

# Atomic counter buffer setup
appendCounter = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ATOMIC_COUNTER_BUFFER, appendCounter)
gl.glBufferData(gl.GL_ATOMIC_COUNTER_BUFFER, 4, np.uint32(0), gl.GL_DYNAMIC_DRAW)
gl.glBindBufferBase(gl.GL_ATOMIC_COUNTER_BUFFER, 0, appendCounter)

# Correct size for storing vec3 (3 floats per vertex, no padding)
vertsbuffer = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, vertsbuffer)
gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, num_elements * 3 * 4, None, gl.GL_DYNAMIC_COPY)  # 3 floats * 4 bytes
gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, vertsbuffer)

# Dispatch compute shader
gl.glDispatchCompute(*grid_size)
gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT | gl.GL_ATOMIC_COUNTER_BARRIER_BIT)

# Pause to allow compute shader to execute
time.sleep(0.1)

# Retrieve the current value of the atomic counter
gl.glBindBuffer(gl.GL_ATOMIC_COUNTER_BUFFER, appendCounter)
append_counter_value = gl.glGetBufferSubData(gl.GL_ATOMIC_COUNTER_BUFFER, 0, 4)
append_counter_value = np.frombuffer(append_counter_value, dtype=np.uint32)[0]

num_vert = append_counter_value
print("Number of vertices generated:", num_vert)

# Retrieve the vertex data from the buffer
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, vertsbuffer)
vert_data = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, num_vert * 3 * 4)  # Each vertex is 3 floats
verts = np.frombuffer(vert_data, dtype=np.float32).reshape(-1, 3)

print("Vertices array:")
print(verts)

# Clean up and terminate
gl.glDeleteBuffers(1, [appendCounter])
gl.glDeleteBuffers(1, [vertsbuffer])
glfw.terminate()

# Create and visualize the mesh in PyVista
mesh = pv.PolyData(verts)
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="black", show_edges=True)
plotter.show()
