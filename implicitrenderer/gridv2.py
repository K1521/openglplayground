import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import glfw

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
grid_size = 10,20,30
min_val = -1.0,-1,-1
max_val = 1.0,1,1
num_elements = grid_size[0]*grid_size[1]*grid_size[2]

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

if (error := gl.glGetError()) != gl.GL_NO_ERROR:
    print(f"OpenGL Error: {error}")

# Retrieve the data from the SSBO
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)

import tableconvert
marchingcubesshader="""
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

const ivec2 EdgeVertexIndices[12] = ivec2[12](
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
// Lookup Tables for Marching Cubes
//
// These tables differ from the original paper (Marching Cubes: A High Resolution 3D Surface Construction Algorithm)
//
// The co-ordinate system has the more convenient properties:
//
//    i = cube index [0, 7]
//    x = (i & 1) >> 0
//    y = (i & 2) >> 1
//    z = (i & 4) >> 2
//
// Axes are:
//
//      y
//      |     z
//      |   /
//      | /
//      +----- x
//
// Vertex and edge layout:
//
//            6             7
//            +-------------+               +-----6-------+   
//          / |           / |             / |            /|   
//        /   |         /   |          11   7         10   5
//    2 +-----+-------+  3  |         +-----+2------+     |   
//      |   4 +-------+-----+ 5       |     +-----4-+-----+   
//      |   /         |   /           3   8         1   9
//      | /           | /             | /           | /       
//    0 +-------------+ 1             +------0------+         
//
ivec3 vertexoffset[8] = ivec3[8]{
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 1, 0),
    ivec3(1, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 0, 1),
    ivec3(0, 1, 1),
    ivec3(1, 1, 1)
};

{marchingcubestable}

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
    ivec delta=ivec3(1, gridSize.x, gridSize.y*gridSize.x);
    float vertvalues[8];
    vec3 vertpos[8];

    float isoValue=0;

    int tableindex=0;
    for(int i=0;i<8;i++){
        vertvalues[i]=field[dot(vertexoffset[i]+pos,delta)];
        vertpos[i] = mix(minVal, maxVal, vec3(pos) / vec3(gridSize - 1));
        if (vertvalues[i] >= isoValue) {
            tableindex |= (1 << i);  // Set the bit for the i-th corner
        }
    }

    for(int i=0;marchingcubestable[tableindex][i]!=-1;i+=3){
        int idx=atomicAdd(appendCounter, 3);
        for(int j=0;j<3;j++){
            int edgeindex=marchingcubestable[tableindex][i+j];
            ivec2 edge=EdgeVertexIndices[edgeindex];
            float t = (isoValue - vertvalues[edge.x]) / (vertvalues[edge.y] - vertvalues[edge.x]);
            triangles[idx+j]=mix(vertpos[edge.x], vertpos[edge.y], t);
        }
    }
}
""".replace("{marchingcubestable}",tableconvert.tablestring)

compute_shader = shaders.compileShader(marchingcubesshader, gl.GL_COMPUTE_SHADER)
shader_program2 = shaders.compileProgram(compute_shader)

triangles = gl.glGenBuffers(1)
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, triangles)
gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, num_elements * 4*3*13, None, gl.GL_DYNAMIC_COPY)  # 4 bytes per float
gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, triangles)

# Set shader program uniforms
gl.glUseProgram(shader_program2)
gl.glUniform3i(gl.glGetUniformLocation(shader_program2, "gridSize"), *grid_size)
gl.glUniform3f(gl.glGetUniformLocation(shader_program2, "minVal"), *min_val)
gl.glUniform3f(gl.glGetUniformLocation(shader_program2, "maxVal"), *max_val)
gl.glUniform3f(gl.glGetUniformLocation(shader_program2, "maxVal"), *max_val)
# Dispatch the compute shader
gl.glDispatchCompute(*[x-1 for x in grid_size])
gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

if (error := gl.glGetError()) != gl.GL_NO_ERROR:
    print(f"OpenGL Error: {error}")

# result = np.empty(num_elements, dtype=np.float32)
# gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, result.nbytes,result.data)

# result = result.reshape(grid_size[::-1])  # Reshape to 3D array

# # Cleanup
# gl.glDeleteBuffers(1, [ssbo])

glfw.terminate()
import pyvista as pv
plotter = pv.Plotter()
plotter.set_background("black") 

x,y,z = [np.linspace(mi,ma,s)for mi,ma,s in zip(min_val,max_val,grid_size[::-1])]

grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))
grid["values"] = result.ravel(order="F")

contours = grid.contour([0])
print(contours)
if contours.n_points != 0:
    plotter.add_mesh(contours)#, color="white", opacity=0.02)
plotter.show(auto_close=False)