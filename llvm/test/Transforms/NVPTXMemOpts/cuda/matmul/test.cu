#include "tb_size_marker.cpp"

// CUDA Kernel to implement naive matrix multiplication
__global__ void matmul(const float *A, const float *B, float *C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __tb_size_marker_1D(16);

    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
/*
// CUDA Kernel to implement matrix multiplication with shared memory (coalesced)
__global__ void matmul_coalesced(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for (int i = 0; i < N; i += 16) {
        __shared__ float A_shared[16];
        A_shared[threadIdx.x] = A[row * N + i + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < 16; j++) {
            sum += A_shared[j] * B[i * N + col];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
*/

// CUDA Kernel to implement matrix multiplication with shared memory (coalesced) and prefetching
__global__ void matmul_coalesced_prefetch(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    float tmp = A[row * N + threadIdx.x];
    for (int i = 0; i < N; i += 16) {
        __shared__ float A_shared[16];
        A_shared[threadIdx.x] = tmp;
        __syncthreads();

        if (i + 16 < N)
            tmp = A[row * N + i + 16 + threadIdx.x];

        for (int j = 0; j < 16; j++) {
            sum += A_shared[j] * B[i * N + col];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
