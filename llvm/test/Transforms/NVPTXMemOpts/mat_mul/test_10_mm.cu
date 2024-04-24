#include <iostream>
#include <cuda_runtime.h>

#define N 4096  // Define matrix size (N x N)

// Naive Matrix Multiplication
__global__ void matrixMultiplyNaive(const float *A, const float *B, float *C, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < width) {
        float sum = 0;
        for (int i = 0; i < width; i++) {
            sum += A[idy * width + i] * B[i * width + idx];
        }
        C[idy * width + idx] = sum;
    }
}

// Coalesced Matrix Multiplication
__global__ void matrixMultiplyCoalesced(const float *A, const float *B, float *C, int width) {
    __shared__ float sharedA[16];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    for (int i = 0; i < width; i += 16) {
        sharedA[threadIdx.x] = A[idy * width + i + threadIdx.x];
        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += sharedA[k] * B[(i + k) * width + idx];
        }
        __syncthreads();
    }
    if (idx < width && idy < width) {
        C[idy * width + idx] = sum;
    }
}

// Coalesced and Prefetched Matrix Multiplication
__global__ void matrixMultiplyCoalescedPrefetched(const float *A, const float *B, float *C, int width) {
    __shared__ float sharedA[16];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    float tmp = A[idy * width + threadIdx.x];
    for (int i = 0; i < width; i += 16) {
        sharedA[threadIdx.x] = tmp;
        __syncthreads();
        if (i + 16 < width) {
            tmp = A[idy * width + i + 16 + threadIdx.x];
        }
        for (int k = 0; k < 16; k++) {
            sum += sharedA[k] * B[(i + k) * width + idx];
        }
        __syncthreads();
    }
    if (idx < width && idy < width) {
        C[idy * width + idx] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];  
    float *h_C_coalesced = new float[N * N];  
    float *h_C_prefetched = new float[N * N]; 

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f; 
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
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
    matrixMultiplyNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Coalesced matrix multiplication
    cudaEventRecord(start);
    matrixMultiplyCoalesced<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C_coalesced, d_C, bytes, cudaMemcpyDeviceToHost);

    // Coalesced and Prefetched matrix multiplication
    cudaEventRecord(start);
    matrixMultiplyCoalescedPrefetched<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced and Prefetched execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C_prefetched, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != 2 * N || h_C_coalesced[i] != 2 * N || h_C_prefetched[i] != 2 * N) {
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
    delete[] h_C_prefetched;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}