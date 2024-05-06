#include <cuda_runtime.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdint> // Needed for int64_t

// CUDA kernel for element-wise addition of two matrices
__global__ void matrixAdd(const int64_t *A, const int64_t *B, int64_t *C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols) {
        int idx = row * numCols + col;
        C[idx] = A[idx] + B[idx];
    }
}   


// CUDA kernel for element-wise addition of two matrices with memory coalescing
__global__ void matrixAdd_coalesced(const int64_t *A, const int64_t *B, int64_t *C, int numRows, int numCols) {
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

 

int main() {
    size_t N = 4096;
    size_t bytes = N * N * sizeof(int64_t);
    int64_t *h_A = new int64_t[N * N];
    int64_t *h_B = new int64_t[N * N];
    int64_t *h_C = new int64_t[N * N]; 
    int64_t *h_C_coalesced = new int64_t[N * N];  

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1; 
        h_B[i] = 2;
    }

    int64_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // Naive matrix multiplication
    cudaEventRecord(start);
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Coalesced matrix multiplication
    cudaEventRecord(start);
    matrixAdd_coalesced<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C_coalesced, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != 3 || h_C_coalesced[i] != 3) {
            std::cerr << "Error: Matrix result is incorrect at index " << i << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "All results are correct." << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_coalesced;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}