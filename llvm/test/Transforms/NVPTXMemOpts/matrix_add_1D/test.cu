#include <cuda_runtime.h>

// CUDA kernel for element-wise addition of two matrices
__global__ void matrixAdd(const float *A, const float *B, float *C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols) {
        int idx = row * numCols + col;
        C[idx] = A[idx] + B[idx];
    }
}   


// CUDA kernel for element-wise addition of two matrices with memory coalescing
__global__ void matrixAdd_coalesced(const float *A, const float *B, float *C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int A_shared[16][16];
    __shared__ int B_shared[16][16];
    A_shared[threadIdx.y][threadIdx.x] = A[row * numCols + col];
    B_shared[threadIdx.y][threadIdx.x] = B[row * numCols + col];
    __syncthreads();
    if (row < numRows && col < numCols) {
        int idx = row * numCols + col;
        C[idx] = A_shared[threadIdx.y][threadIdx.x] + B_shared[threadIdx.y][threadIdx.x];
    }
}

 