#include <cuda_runtime.h>
#include <tb_size_marker.cpp>

__global__ void naive(float *input, float *output, int row, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    __tb_size_marker_1D(256);
    if (idx < cols && idy < row) {
        float sum = 0;
        for (int i = 0; i < cols; i++) {
            sum += input[idy * cols + i];
        }
        output[idx] = sum;
    }
}

__global__ void coalesced(float *input, float *output, int row, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0;
    for (int i = 0; i < cols; i += 256) {
        __shared__ float shared[256];
        shared[threadIdx.x] = input[idy * cols + i + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < 256; k++) {
            sum += shared[k];
        }
        __syncthreads();
    }
    if (idx < cols && idy < row) {
        output[idx] = sum;
    }
}

