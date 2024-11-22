import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import glfw
import pyvista as pv
import tableconvert
import time

class deltastopwatch:
    def __init__(self):
        self.start=time.time()
    def __call__(self):
        now=time.time()
        delta=now-self.start
        self.start=now
        return delta

watch=deltastopwatch()

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

# Ensure OpenGL is initialized
print("OpenGL version:", gl.glGetString(gl.GL_VERSION))
print("contextcreation",watch())

# Compute shader source code
compute_shader_code = """
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer Result {
    float result[];
};

uniform ivec3 gridSize;
uniform vec3 minVal;
uniform vec3 maxVal;

void main() {
    ivec3 idx = ivec3(gl_GlobalInvocationID.xyz);
    int linear_idx = idx.x + gridSize.x * (idx.y + gridSize.y * idx.z);

    vec3 p = mix(minVal, maxVal, vec3(idx) / vec3(gridSize - 1));
    
    result[linear_idx] = p.x * p.x + p.y * p.y + p.z * p.z - 0.5;
}
"""

# Compile the compute shader
compute_shader = shaders.compileShader(compute_shader_code, gl.GL_COMPUTE_SHADER)
shader_program = shaders.compileProgram(compute_shader)

# Grid parameters
grid_size = (30, 30, 30)
min_val = (-1.0, -1, -1)
max_val = (1.0, 1, 1)
num_elements = grid_size[0] * grid_size[1] * grid_size[2]

# Create an SSBO to store the results
ssbo = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, num_elements * 4, None, gl.GL_DYNAMIC_COPY)  # 4 bytes per float
gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, ssbo)

# Set shader program uniforms
gl.glUseProgram(shader_program)
gl.glUniform3i(gl.glGetUniformLocation(shader_program, "gridSize"), *grid_size)
gl.glUniform3f(gl.glGetUniformLocation(shader_program, "minVal"), *min_val)
gl.glUniform3f(gl.glGetUniformLocation(shader_program, "maxVal"), *max_val)

# Dispatch the compute shader
gl.glDispatchCompute(*grid_size)
gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

# Check for OpenGL errors
if (error := gl.glGetError()) != gl.GL_NO_ERROR:
    print(f"OpenGL Error: {error}")
print("implicitshader",watch())
# Marching cubes shader
marchingcubesshader = """
#version 430
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer Result {
    float field[];
};

layout(binding = 1, offset = 0) uniform atomic_uint appendCounter;  // Atomic counter for indexing
layout(std430, binding = 2) buffer AppendBuffer {  // Buffer for appended values
    vec3 triangles[];
};

uniform ivec3 gridSize;
uniform vec3 minVal;
uniform vec3 maxVal;

const ivec2[12] EdgeVertexIndices = ivec2[12](
    ivec2(0, 1),
    ivec2(1, 3),
    ivec2(3, 2),
    ivec2(2, 0),
    ivec2(4, 5),
    ivec2(5, 7),
    ivec2(7, 6),
    ivec2(6, 4),
    ivec2(0, 4),
    ivec2(1, 5),
    ivec2(3, 7),
    ivec2(2, 6)
);

const ivec3[8] vertexoffset = ivec3[8](
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 1, 0),
    ivec3(1, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 0, 1),
    ivec3(0, 1, 1),
    ivec3(1, 1, 1)
);

// Lookup Tables for Marching Cubes
{marchingcubestable}

int idot(ivec3 a, ivec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

void main() {
    ivec3 pos0 = ivec3(gl_GlobalInvocationID.xyz);
    ivec3 delta = ivec3(1, gridSize.x, gridSize.y * gridSize.x);
    float vertvalues[8];
    vec3 vertpos[8];

    float isoValue = 0;

    int tableindex = 0;
    for (int i = 0; i < 8; i++) {
        ivec3 pos=vertexoffset[i]+pos0;
        vertvalues[i] = field[idot(pos, delta)];
        vertpos[i] = mix(minVal, maxVal, vec3(pos) / vec3(gridSize - 1));
        if (vertvalues[i] >= isoValue) {
            tableindex |= (1 << i);  // Set the bit for the i-th corner
        }
    }




    // Iterate through triangles and append them
    for (int i = 0; marchingcubestable[tableindex][i] != -1; i += 3) {
        //int idx = int(atomicCounterAdd(appendCounter, uint(3)));
        int idx=int(atomicCounterIncrement(appendCounter))*3;
        for (int j = 0; j < 3; j++) {
            int edgeindex = marchingcubestable[tableindex][i + j];
            ivec2 edge = EdgeVertexIndices[edgeindex];
            float t = (isoValue - vertvalues[edge.x]) / (vertvalues[edge.y] - vertvalues[edge.x]);
            triangles[idx + j] = mix(vertpos[edge.x], vertpos[edge.y], t);
            //int idx=int(atomicCounterIncrement(appendCounter));
            //triangles[idx] = mix(vertpos[edge.x], vertpos[edge.y], t);
        }

    }
}
""".replace("{marchingcubestable}",tableconvert.tablestring)

# Compile the marching cubes compute shader
try:
    compute_shader2 = shaders.compileShader(marchingcubesshader, gl.GL_COMPUTE_SHADER)
    shader_program2 = shaders.compileProgram(compute_shader2)
except Exception as e:
    print(str(e).replace("\\\\n","\n").replace("\\n","\n"))
    #for a in e.args:
    #    print(a)
    exit()
gl.glUseProgram(shader_program2)

gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, ssbo)

# Create an atomic counter buffer for counting the number of triangles
appendCounter = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_ATOMIC_COUNTER_BUFFER, appendCounter)
gl.glBufferData(gl.GL_ATOMIC_COUNTER_BUFFER, 4, np.uint32(0), gl.GL_DYNAMIC_DRAW)  # Initial value of 0
gl.glBindBufferBase(gl.GL_ATOMIC_COUNTER_BUFFER, 1, appendCounter)

# Create an SSBO for the triangles
triangles = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, triangles)
gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, num_elements * 4 * 4 * 13, None, gl.GL_DYNAMIC_COPY)  # 4 bytes per float
gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, triangles)

# Set shader program uniforms

gl.glUniform3i(gl.glGetUniformLocation(shader_program2, "gridSize"), *grid_size)
gl.glUniform3f(gl.glGetUniformLocation(shader_program2, "minVal"), *min_val)
gl.glUniform3f(gl.glGetUniformLocation(shader_program2, "maxVal"), *max_val)

# Dispatch the marching cubes compute shader
gl.glDispatchCompute(*[x - 1 for x in grid_size])
gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

# Fetch the current append counter
gl.glBindBuffer(gl.GL_ATOMIC_COUNTER_BUFFER, appendCounter)
append_counter_value = gl.glGetBufferSubData(gl.GL_ATOMIC_COUNTER_BUFFER, 0, 4)  # 1 int = 4 bytes
append_counter_value = np.frombuffer(append_counter_value, dtype=np.uint32)[0]

# Calculate the number of triangles
num_triangles = append_counter_value 
print(num_triangles)



# Fetch the triangle data
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, triangles)
triangle_data = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, num_triangles * 3*4 * 4)  # 3 vertices per triangle, 4 bytes per float
triangle_data = np.frombuffer(triangle_data, dtype=np.float32).reshape(-1, 4)
triangle_data=triangle_data[:,:3]
print(len(triangle_data))
print("marchingcubes",watch())
# Create a PyVista mesh from the triangle data
#triangles = triangle_data.reshape(-1, 3,3)

# Create a PyVista PolyData object
faces = np.hstack([np.full((num_triangles, 1), 3), np.arange(num_triangles*3).reshape(-1,3)])
mesh = pv.PolyData(triangle_data,faces)

#print(np.arange(num_triangles*3).reshape(-1,3))

# Add faces (PyVista requires faces to be in a specific format)

#print(faces)
mesh.faces = faces.ravel()

# Visualize the
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="red",lighting=True,show_vertices=False)
print("pyvista",watch())
plotter.show()