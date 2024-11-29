import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import glfw
import pyvista as pv
import tableconvert
import time
import ctypes

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

float scene(vec3 p);

void main() {
    ivec3 idx = ivec3(gl_GlobalInvocationID.xyz);
    int linear_idx = idx.x + gridSize.x * (idx.y + gridSize.y * idx.z);

    vec3 p = mix(minVal, maxVal, vec3(idx) / vec3(gridSize - 1));
    


// Implement the torus equation
result[linear_idx] = scene(p);
}

//cutoff
    float scene(vec3 p){
    float x2 = p.x * p.x;
    float y2 = p.y * p.y;
    float z2 = p.z * p.z;
    float R2 = 0.5;//R * R;
    float r2 = 0.1;//r * r;
    return pow(x2 + y2 + z2 + R2 - r2, 2.0) - 4.0 * R2 * (x2 + y2);
}
"""

def readfile(path):
    with open(path) as file:
        return file.read()

scene=readfile("./scene.glsl")
compute_shader_code=compute_shader_code.split("//cutoff")[0]+scene

# Compile the compute shader
compute_shader = shaders.compileShader(compute_shader_code, gl.GL_COMPUTE_SHADER)
shader_program = shaders.compileProgram(compute_shader)

# Grid parameters
grid_size = (100, 100, 100)
#grid_size = [8*10+1]*3
min_val = (-5, -5, -5)
max_val = (5, 5, 5)
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

threadnum=[8,8,8]
marchingcubesshader = """
#version 430
#define NUM_THREADS_X {threadnum1}
#define NUM_THREADS_Y {threadnum2}
#define NUM_THREADS_Z {threadnum3}
#define NUM_THREADS_TOTAL (NUM_THREADS_X * NUM_THREADS_Y * NUM_THREADS_Z)
//const ivec3 NUM_THREADS = ivec3(NUM_THREADS_X,NUM_THREADS_Y,NUM_THREADS_Z);  // Used for internal loop size
layout(local_size_x = NUM_THREADS_X, local_size_y = NUM_THREADS_Y, local_size_z = NUM_THREADS_Z) in;
//shared vec3[NUM_THREADS_TOTAL*12] sharedVertices; //vertices written by each thread
//shared int[NUM_THREADS_TOTAL] sharedNumVertices;

layout(std430, binding = 0) buffer Result {
    float field[];
};


layout(std430, binding = 1) buffer CounterBuffer {
    uint appendCounter;
};
//layout(binding = 1, offset = 0) uniform atomic_uint appendCounter;  // Atomic counter for indexing
layout(std430, binding = 2) buffer AppendBuffer {  // Buffer for appended values
    vec3 triangles[];
};

uniform ivec3 gridSize;
uniform vec3 minVal;
uniform vec3 maxVal;

uniform ivec2[12] EdgeVertexIndices = ivec2[12](
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

uniform ivec3[8] vertexoffset = ivec3[8](
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
    if(pos0.x>=gridSize.x-1||pos0.y>=gridSize.y-1||pos0.z>=gridSize.z-1)return;
    

    ivec3 delta = ivec3(1, gridSize.x, gridSize.y * gridSize.x);
    float vertvalues[8];
    //vec3 vertpos[8];

    float isoValue = 0;

    int tableindex = 0;
    for (int i = 0; i < 8; i++) {
        ivec3 pos=vertexoffset[i]+pos0;
        vertvalues[i] = field[idot(pos, delta)];

        if (vertvalues[i] >= isoValue) {
            tableindex |= (1 << i);  // Set the bit for the i-th corner
        }
    }
    if(tableindex==0 || tableindex==255)return;




    // Calculate the number of triangles this thread will generate
    const int[] tablerow=marchingcubestable[tableindex];

    int numVertices = 0;
    for (int i = 0; tablerow[i] != -1; i += 3) {
        numVertices+=3;
    }

    

    // Reserve space for triangles
    //int baseIdx = int(atomicAdd(appendCounter, uint(numVertices)));
    int baseIdx = int(atomicAdd(appendCounter, uint(numVertices)));
    //int baseIdx = idot(pos0, delta)*12;


    // Append triangles
    for (int i = 0; i<numVertices; i++) {
    //for (int i = 0; tablerow[i] != -1; i++) {
    //int i=0;
        int edgeindex = tablerow[i];
        ivec2 edge = EdgeVertexIndices[edgeindex];
        float t = (isoValue - vertvalues[edge.x]) / (vertvalues[edge.y] - vertvalues[edge.x]);
        
        vec3 vertex=mix(vertexoffset[edge.x], vertexoffset[edge.y], t);
        vertex=mix(minVal, maxVal,(vertex+pos0)/ vec3(gridSize - 1));
        triangles[baseIdx+i] = vertex;
    }
}
""".replace("{marchingcubestable}",tableconvert.tablestring).replace("{threadnum1}",str(threadnum[0])).replace("{threadnum2}",str(threadnum[1])).replace("{threadnum3}",str(threadnum[2]))

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
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, appendCounter)
gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, 4, np.uint32(0), gl.GL_DYNAMIC_DRAW)  # Initial value of 0
gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, appendCounter)

# Create an SSBO for the triangles
triangles = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, triangles)
gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, num_elements * 4 * 4 * 12, None, gl.GL_DYNAMIC_COPY)  # 4 bytes per float,4 floats per vertex,12 vertices per cube
gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, triangles)

# Set shader program uniforms

gl.glUniform3i(gl.glGetUniformLocation(shader_program2, "gridSize"), *grid_size)
gl.glUniform3f(gl.glGetUniformLocation(shader_program2, "minVal"), *min_val)
gl.glUniform3f(gl.glGetUniformLocation(shader_program2, "maxVal"), *max_val)

# Dispatch the marching cubes compute shader
query, = gl.glGenQueries(1)
gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)
gl.glDispatchCompute(*[-(-(x - 1)//t) for x,t in zip(grid_size,threadnum)])
gl.glEndQuery(gl.GL_TIME_ELAPSED)
result = ctypes.c_uint64()
gl.glGetQueryObjectui64v(query, gl.GL_QUERY_RESULT, ctypes.byref(result))
elapsed_time_ns = result.value  # Result is in nanoseconds
print(f"Elapsed time: {elapsed_time_ns / 1e6:.3f} ms")
gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

# Fetch the current append counter
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, appendCounter)
append_counter_value = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, 4)  # 1 int = 4 bytes
append_counter_value = np.frombuffer(append_counter_value, dtype=np.uint32)[0]

# Calculate the number of triangles
num_vertices = append_counter_value 
print("numvert",num_vertices)



# Fetch the triangle data
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, triangles)
triangle_data = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, num_vertices *4 * 4)  # 3 vertices per triangle, 4 bytes per float
triangle_data = np.frombuffer(triangle_data, dtype=np.float32).reshape(-1, 4)
triangle_data=triangle_data[:,:3]
#print(len(triangle_data))
print("marchingcubes",watch())
# Create a PyVista mesh from the triangle data
#triangles = triangle_data.reshape(-1, 3,3)
if num_vertices==0:
    print("empty mesh")
    exit()
# Create a PyVista PolyData object
faces = np.hstack([np.full((num_vertices//3, 1), 3), np.arange(num_vertices).reshape(-1,3)])
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