#include <stdio.h>

// the reads in this kernel are of constant indexes and therefore cannot be coallesced

// CUDA kernel for constant access
__global__ void naiveMM(const float **A, const float **B, float **C, int w) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            sum += A[10][15] * B[23][54];
        }
    }
    C[idy][idx] = sum;

}