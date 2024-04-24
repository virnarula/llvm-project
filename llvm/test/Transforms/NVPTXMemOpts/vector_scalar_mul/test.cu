#include <stdio.h>

// CUDA kernel for vector scalar multiplication
__global__ void vectorScalarMultiply(const float *input, float *output, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i+16 < numElements)
        auto tmp = input[i];
    if (i < numElements) {
        output[i] = input[i] * scalar;
    }
}

// CUDA kernel for vector scalar multiplication
__global__ void vectorScalarMultiply_coalesced(const float *input, float *output, float scalar, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float input_shared[16];
    input_shared[threadIdx.x] = input[i];
    __syncthreads();
    if (i < numElements) {
        output[i] = input_shared[threadIdx.x] * scalar;
    }
}
