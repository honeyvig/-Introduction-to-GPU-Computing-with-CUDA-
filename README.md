# -Introduction-to-GPU-Computing-with-CUDA
Introduction to GPU Computing with CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to use GPUs (Graphics Processing Units) for general-purpose computing, which can greatly accelerate computationally intensive tasks.

In this introduction, we will explore the basic concepts of GPU computing using CUDA. Specifically, we will show how to:

    Setup a CUDA environment.
    Write and run a simple CUDA kernel for array addition.
    Understand how the GPU architecture works for parallel computing.

We will use the pycuda library to access CUDA from Python. This allows us to interact with the GPU directly from Python code and utilize CUDA for parallel computation.
Prerequisites

    CUDA Toolkit: Install the NVIDIA CUDA toolkit. You can download it from NVIDIA's official site.
    NVIDIA GPU: Ensure you have an NVIDIA GPU that supports CUDA (for example, Tesla, Quadro, or GeForce series).
    pycuda: Install the pycuda library. You can install it via pip:

    pip install pycuda

Step-by-Step Guide

Let's walk through a basic example of GPU computing with CUDA. The goal is to create a program that adds two arrays together using the GPU in parallel.
Step 1: Setting Up the CUDA Kernel

We'll write a simple CUDA kernel to add two arrays element-wise on the GPU.
CUDA Kernel Code:

The kernel will take two input arrays, add their corresponding elements, and store the result in a third array.

import numpy as np
import pycuda.autoinit  # Automatically initialize CUDA driver
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Define a CUDA kernel for array addition
kernel_code = """
__global__ void array_addition(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Calculate global index
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Perform element-wise addition
    }
}
"""

# Compile the CUDA kernel code
mod = SourceModule(kernel_code)

Step 2: Initialize Input Data and GPU Memory

Next, we will:

    Create input arrays a and b.
    Allocate memory for arrays on the GPU.
    Copy the data from the host (CPU) to the GPU.

# Initialize input data
N = 1024  # Size of the arrays
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros_like(a)  # Output array initialized to zero

# Allocate memory on the GPU for input and output arrays
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy input arrays from host (CPU) to device (GPU)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

Step 3: Configure CUDA Blocks and Threads

CUDA operates on a grid of blocks, where each block contains multiple threads. The size of each block and grid needs to be configured based on the number of elements you want to process.

Here, we use 256 threads per block, and calculate how many blocks are needed to cover the entire array.

# Define block size and grid size
block_size = 256  # Number of threads per block
grid_size = (N + block_size - 1) // block_size  # Number of blocks (ceil(N / block_size))

Step 4: Launch the CUDA Kernel

Now we launch the kernel on the GPU. This will start the computation on the GPU, using the configuration defined in the previous step.

# Get the kernel function from the compiled module
array_add_kernel = mod.get_function("array_addition")

# Launch the kernel on the GPU
array_add_kernel(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

Step 5: Copy the Results from GPU to CPU

Once the kernel finishes running, we need to copy the result back from the GPU memory to the CPU memory.

# Copy the result from device (GPU) to host (CPU)
cuda.memcpy_dtoh(c, c_gpu)

# Print results
print("Array A:", a[:10])  # Print first 10 elements for brevity
print("Array B:", b[:10])  # Print first 10 elements for brevity
print("Array C (A + B):", c[:10])  # Print first 10 elements of the result

Full Code Example:

Here’s the full code for GPU-based array addition using CUDA:

import numpy as np
import pycuda.autoinit  # Automatically initialize CUDA driver
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Define the CUDA kernel for array addition
kernel_code = """
__global__ void array_addition(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Calculate global index
    if (idx < N) {
        c[idx] = a[idx] + b[idx];  // Perform element-wise addition
    }
}
"""

# Compile the CUDA kernel code
mod = SourceModule(kernel_code)

# Initialize input data
N = 1024  # Size of the arrays
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros_like(a)  # Output array initialized to zero

# Allocate memory on the GPU for input and output arrays
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy input arrays from host (CPU) to device (GPU)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Define block size and grid size
block_size = 256  # Number of threads per block
grid_size = (N + block_size - 1) // block_size  # Number of blocks (ceil(N / block_size))

# Get the kernel function from the compiled module
array_add_kernel = mod.get_function("array_addition")

# Launch the kernel on the GPU
array_add_kernel(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

# Copy the result from device (GPU) to host (CPU)
cuda.memcpy_dtoh(c, c_gpu)

# Print results
print("Array A:", a[:10])  # Print first 10 elements for brevity
print("Array B:", b[:10])  # Print first 10 elements for brevity
print("Array C (A + B):", c[:10])  # Print first 10 elements of the result

Output:

The output will show the first 10 elements of arrays a, b, and c. For example:

Array A: [0.4807465  0.40623428 0.76323849 0.14005231 0.96741059 0.89802994 0.40505687 0.32418419 0.59448478 0.35014969]
Array B: [0.83887111 0.36233114 0.44522742 0.12445361 0.8324241  0.34603153 0.59117599 0.94652722 0.6488045  0.37405233]
Array C (A + B): [1.31961762 0.76856542 1.20846591 0.26450592 1.79983469 1.24406147 0.99623286 1.27071141 1.24328928 0.72420202]

Explanation:

    CUDA Kernel: The CUDA kernel (array_addition) is executed in parallel by many threads on the GPU. Each thread computes one element of the result by adding the corresponding elements from arrays a and b.

    Memory Allocation: We allocate memory on the GPU for the input arrays a, b, and the output array c using cuda.mem_alloc(). Data is then copied from the host (CPU) to the device (GPU) using cuda.memcpy_htod().

    Parallel Execution: The kernel is launched with a block size of 256 threads and a grid size calculated to ensure that every element in the arrays is processed. The global index of each thread is calculated using threadIdx and blockIdx.

    Copying Results: After the kernel finishes executing, the result is copied back from the GPU memory to the CPU memory using cuda.memcpy_dtoh().

Conclusion

This simple example demonstrates how to use CUDA to perform parallel computation on a GPU for array operations. By leveraging the power of the GPU, operations on large datasets can be sped up significantly.

From here, you can experiment with more complex operations like matrix multiplication, element-wise functions, or even multi-dimensional arrays. CUDA programming is a powerful tool for accelerating computationally intensive tasks.
