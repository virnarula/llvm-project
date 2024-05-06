#include <iostream>
#include <cuda_runtime.h>

#define N 4096  // Define matrix size (N x N)

// Naive Matrix-Vector Multiplication
__global__ void matrixVectorMultiplyNaiveInt64(const long long *A, const long long *B, long long *C, int width) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < width) {
        long long sum = 0;
        for (int i = 0; i < width; i++) {
            sum += A[id * width + i] * B[i];
        }
        C[id] = sum;
    }
}

// Coalesced Matrix-Vector Multiplication
__global__ void matrixVectorMultiplyCoalescedInt64(const long long *A, const long long *B, long long *C, int width) {
    __shared__ long long sharedB[16];
    __shared__ long long sharedA[16][16 + 1];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int idx = bidx * blockDim.x + tidx;
    long long sum = 0;

    for (int i = 0; i < width; i += 16) {
        if (i + tidx < width) sharedB[tidx] = B[i + tidx];
        
        for (int l = 0; l < 16; l++) {
            if ((idx - tidx + l < width) && (i + tidx < width)) {
                sharedA[l][tidx] = A[(idx - tidx + l) * width + (i + tidx)];
            }
        }
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            if (i + k < width) {
                sum += sharedA[tidx][k] * sharedB[k];
            }
        }
        __syncthreads(); 
    }
    if (idx < width) C[idx] = sum;
}

// Coalesced and Prefetched Matrix-Vector Multiplication
__global__ void matrixVectorMultiplyCoalescedPrefetchedInt64(const long long *A, const long long *B, long long *C, int width) {
    __shared__ long long sharedB[16];
    __shared__ long long sharedA[16][16 + 1];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int idx = bidx * blockDim.x + tidx;
    long long sum = 0;
    long long prefetchedB = 0;

    if (tidx < width) prefetchedB = B[tidx];

    for (int i = 0; i < width; i += 16) {
        sharedB[tidx] = prefetchedB;
        if (i + 16 + tidx < width) {
            prefetchedB = B[i + 16 + tidx];
        }
        for (int l = 0; l < 16; l++) {
            if ((idx - tidx + l < width) && (i + tidx < width)) {
                sharedA[l][tidx] = A[(idx - tidx + l) * width + (i + tidx)];
            }
        }
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            if (i + k < width) {
                sum += sharedA[tidx][k] * sharedB[k];
            }
        }
        __syncthreads();
    }
    if (idx < width) C[idx] = sum;
}

int main() {
    size_t bytes = N * N * sizeof(long long);
    size_t vectorBytes = N * sizeof(long long);
    long long *h_A = new long long[N * N];
    long long *h_B = new long long[N];
    long long *h_C = new long long[N];
    long long *h_C_coalesced = new long long[N];
    long long *h_C_prefetched = new long long[N];

    // Initialize matrices and vector
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
    }
    for (int i = 0; i < N; i++) {
        h_B[i] = 2;
    }

    long long *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, vectorBytes);
    cudaMalloc(&d_C, vectorBytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vectorBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // Naive matrix-vector multiplication
    cudaEventRecord(start);
    matrixVectorMultiplyNaiveInt64<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C, d_C, vectorBytes, cudaMemcpyDeviceToHost);

    // Coalesced matrix-vector multiplication
    cudaEventRecord(start);
    matrixVectorMultiplyCoalescedInt64<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C_coalesced, d_C, vectorBytes, cudaMemcpyDeviceToHost);

    // Coalesced and Prefetched matrix-vector multiplication
    cudaEventRecord(start);
    matrixVectorMultiplyCoalescedPrefetchedInt64<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced and Prefetched execution time: " << milliseconds << " ms\n";
    cudaMemcpy(h_C_prefetched, d_C, vectorBytes, cudaMemcpyDeviceToHost);

    // Verify
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 2 * N || h_C_coalesced[i] != 2 * N || h_C_prefetched[i] != 2 * N) {
            std::cerr << "Error: Vector result is incorrect at index " << i << std::endl;
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