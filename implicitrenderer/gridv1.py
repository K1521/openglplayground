import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import glfw
import time
class deltastopwatch:
    def __init__(self):
        self.start=time.time()
    def __call__(self):
        now=time.time()
        delta=now-self.start
        self.start=now
        return delta
    
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
int idot(ivec3 a, ivec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
void main() {
    ivec3 idx = ivec3(gl_GlobalInvocationID.xyz);
    ivec3 delta = ivec3(1, gridSize.x, gridSize.y * gridSize.x);

    //int linear_idx = idx.x + gridSize.x * (idx.y + gridSize.y * idx.z);
    int linear_idx = idot(idx,delta);


    vec3 p = mix(minVal, maxVal, vec3(idx) / vec3(gridSize - 1));
    
    //result[linear_idx] = p.x * p.x + p.y * p.y + p.z * p.z - 0.5;

    float x2 = p.x * p.x;
float y2 = p.y * p.y;
float z2 = p.z * p.z;
float R2 = 0.35;//R * R;
float r2 = 0.1;//r * r;

// Implement the torus equation
result[linear_idx] = pow(x2 + y2 + z2 + R2 - r2, 2.0) - 4.0 * R2 * (x2 + y2);

}
"""

# Compile the compute shader
compute_shader = shaders.compileShader(compute_shader_code, gl.GL_COMPUTE_SHADER)
shader_program = shaders.compileProgram(compute_shader)

# Grid parameters
grid_size = 200,200,200
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
error = gl.glGetError()
if error != gl.GL_NO_ERROR:
    print(f"OpenGL Error: {error}")

# Retrieve the data from the SSBO
gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, ssbo)
result = np.empty(num_elements, dtype=np.float32)
gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, result.nbytes,result.data)

result = result.reshape(grid_size)  # Reshape to 3D array
print("hi")
print(result.min(),result.max())
# Cleanup
gl.glDeleteBuffers(1, [ssbo])
print("hi")
glfw.terminate()
import pyvista as pv
plotter = pv.Plotter()
plotter.set_background("black") 

x,y,z = [np.linspace(mi,ma,s)for mi,ma,s in zip(min_val,max_val,grid_size)]

grid = pv.StructuredGrid(*np.meshgrid(x,y,z))
grid["values"] = result.ravel(order="F")
s=deltastopwatch()
contours = grid.contour([0])
print(s())
print(contours)
if contours.n_points != 0:
    plotter.add_mesh(contours)#, color="white", opacity=0.02)
plotter.show(auto_close=False)