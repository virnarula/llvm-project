#include <stdio.h>
#include <tb_size_marker.cpp>

// CUDA kernel for vector scalar multiplication
__global__ void vectorScalarMultiply(const float *input, float *output, float scalar, int numElements) {
    __tb_size_marker_1D(16);
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        output[i] = input[i] * scalar;
    }
}

// CUDA kernel for vector scalar multiplication
__global__ void vectorScalarMultiply_coalesced(const float *input, float *output, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        __shared__ float input_shared[16];
        input_shared[threadIdx.x] = input[i];
        __syncthreads();
        output[i] = input_shared[threadIdx.x] * scalar;
    }
}
