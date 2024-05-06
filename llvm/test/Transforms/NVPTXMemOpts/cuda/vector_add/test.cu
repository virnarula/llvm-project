#include <stdio.h>
#include <tb_size_marker.cpp>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    __tb_size_marker_1D(256);
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Cuda kernel for vector addition with memory coalescing
__global__ void vectorAdd_coalesced(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float A_shared[16];
    __shared__ float B_shared[16];
    A_shared[threadIdx.x] = A[i];
    B_shared[threadIdx.x] = B[i];
    __syncthreads();
    if (i < numElements) {
        C[i] = A_shared[threadIdx.x] + B_shared[threadIdx.x];
    }
}
