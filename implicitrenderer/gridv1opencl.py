import pyopencl as cl
import numpy as np

# Platform and device selection
platform = cl.get_platforms()[0]  # First platform
device = platform.get_devices()[0]  # First device

# Create OpenCL context and queue
context = cl.Context([device])
queue = cl.CommandQueue(context, device)

# Input data (Example: a simple array)
data = np.random.rand(100).astype(np.float32)

# Create buffer for data
data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)

# Create buffer for results
result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, data.nbytes)

# OpenCL kernel code
program_src = """
__kernel void square(__global const float* input, __global float* output) {
    int i = get_global_id(0);
    output[i] = input[i] * input[i];
}
"""

# Compile and build the program
program = cl.Program(context, program_src).build()

# Execute the kernel
program.square(queue, data.shape, None, data_buffer, result_buffer)

# Retrieve the results
result = np.empty_like(data)
cl.enqueue_copy(queue, result, result_buffer).wait()

print("Input data:", data)
print("Squared results:", result)
