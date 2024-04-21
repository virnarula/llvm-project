#include <stdio.h>

// CUDA kernel for mat mul
__global__ void naiveMM(const float **A, const float **B, float **C, int w) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;
    for (int i = 0; i < w; i++) {
        sum += A[idy][i] * B[i][idx];
    }
    C[idy][idx] = sum;

}

// CUDA kernel for mat mul with shared memory
__global__ void sharedMM(const float **A, const float **B, float **C, int w) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;
    for (int i = 0; i < w; i++) {
        As[threadIdx.y][i] = A[idy][i];
        Bs[i][threadIdx.x] = B[i][idx];
        __syncthreads();
        for (int j = 0; j < w; j++) {
            sum += As[threadIdx.y][j] * Bs[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[idy][idx] = sum;
}